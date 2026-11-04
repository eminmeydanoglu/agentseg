"""
ProjectStore - CRUD operations for projects, images, labels, and annotations.
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image

from core.db import init_db, migrate_db, get_connection, _set_kv, _get_kv
from core.models import (
    Project,
    Label,
    ImageRecord,
    Annotation,
    Track,
    bbox_to_json,
    polygon_to_json,
    mask_rle_to_json,
)


# Project database file name. Projects created before the agentseg rename use
# the legacy name and are still opened.
DB_FILENAME = "agentseg.db"
LEGACY_DB_FILENAMES = ("autoseg.db",)


def find_project_db(project_dir: str) -> Optional[str]:
    """Return the project database inside ``project_dir``, or None."""
    for name in (DB_FILENAME, *LEGACY_DB_FILENAMES):
        path = os.path.join(project_dir, name)
        if os.path.exists(path):
            return path
    return None


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Default label colors (will cycle through these)
DEFAULT_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
    "#F8B500",
    "#00CED1",
    "#FF69B4",
    "#32CD32",
    "#FFD700",
]


def natural_sort_key(s: str):
    """
    Key function for natural sorting of strings.
    E.g., sorts "img2.jpg" before "img10.jpg"
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


class ProjectStore:
    """
    Handles all database operations for a project.
    """

    def __init__(self, db_path: str):
        """
        Initialize store with database path.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_connection(self.db_path)
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def commit(self):
        """Commit current transaction."""
        self.conn.commit()

    # ==================== Project Operations ====================

    @classmethod
    def create_project(cls, project_dir: str, image_dir: str, name: str) -> "Project":
        """
        Create a new project from an image directory.

        Args:
            project_dir: Directory where project files will be stored
            image_dir: Directory containing images to import
            name: Name of the project

        Returns:
            Created Project instance
        """
        project_dir = os.path.abspath(project_dir)
        image_dir = os.path.abspath(image_dir)

        # Create project directory if needed
        os.makedirs(project_dir, exist_ok=True)

        # Database path
        db_path = os.path.join(project_dir, DB_FILENAME)

        # Initialize database
        init_db(db_path)

        # Create store instance
        store = cls(db_path)

        try:
            # Create project record
            cursor = store.conn.execute(
                """
                INSERT INTO projects (name, root_dir, db_path, settings_json)
                VALUES (?, ?, ?, ?)
                """,
                (name, project_dir, db_path, json.dumps({"image_dir": image_dir})),
            )
            project_id = cursor.lastrowid

            # Scan and import images
            store._import_images(project_id, image_dir)

            store.commit()

            # Fetch and return the project
            return store._get_project(project_id)
        finally:
            store.close()

    @classmethod
    def load_project(cls, project_dir: str) -> "Project":
        """
        Load an existing project.

        Args:
            project_dir: Directory containing the project

        Returns:
            Loaded Project instance
        """
        project_dir = os.path.abspath(project_dir)
        db_path = find_project_db(project_dir)

        if db_path is None:
            raise FileNotFoundError(f"No project database found in {project_dir}")

        # Run any pending migrations
        migrate_db(db_path)

        # Load project
        store = cls(db_path)
        try:
            cursor = store.conn.execute("SELECT * FROM projects ORDER BY id LIMIT 1")
            row = cursor.fetchone()
            if not row:
                raise ValueError("No project found in database")

            # The project folder may have been moved or renamed since it was
            # created; keep the stored paths pointing at where it is now.
            if row["root_dir"] != project_dir or row["db_path"] != db_path:
                store.conn.execute(
                    "UPDATE projects SET root_dir = ?, db_path = ? WHERE id = ?",
                    (project_dir, db_path, row["id"]),
                )
                store.commit()
                row = store.conn.execute(
                    "SELECT * FROM projects WHERE id = ?", (row["id"],)
                ).fetchone()

            return store._row_to_project(row)
        finally:
            store.close()

    def _get_project(self, project_id: int) -> Project:
        """Get project by ID."""
        cursor = self.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Project {project_id} not found")
        return self._row_to_project(row)

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        """Convert database row to Project object."""
        return Project(
            id=row["id"],
            name=row["name"],
            root_dir=row["root_dir"],
            db_path=row["db_path"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.now(),
            settings_json=row["settings_json"],
        )

    def _import_images(self, project_id: int, image_dir: str) -> None:
        """Import all images from a directory."""
        image_dir = Path(image_dir)

        # Find all image files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))

        # Sort naturally
        image_files = sorted(set(image_files), key=lambda p: natural_sort_key(p.name))

        # Import each image
        for idx, img_path in enumerate(image_files):
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {img_path}: {e}")
                continue

            # Store relative path from image_dir
            rel_path = str(
                img_path.relative_to(image_dir.parent)
                if image_dir.parent != img_path.parent
                else img_path.name
            )

            self.conn.execute(
                """
                INSERT INTO images (project_id, path, width, height, order_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, str(img_path), width, height, idx),
            )

    def resync_images(self, project_id: int, image_dir: str) -> dict[str, int]:
        """
        Resync images from filesystem with database.

        - Adds new images found in directory
        - Removes records for images that no longer exist (along with their annotations)
        - Preserves annotations for existing images

        Args:
            project_id: ID of the project
            image_dir: Directory to scan for images

        Returns:
            Dict with counts: {"added": int, "removed": int, "unchanged": int}
        """
        image_dir_path = Path(image_dir)

        # Get current images from filesystem
        fs_image_files: set[Path] = set()
        for ext in IMAGE_EXTENSIONS:
            fs_image_files.update(image_dir_path.glob(f"*{ext}"))
            fs_image_files.update(image_dir_path.glob(f"*{ext.upper()}"))

        # Normalize to absolute path strings
        fs_paths = {str(p.absolute()) for p in fs_image_files}

        # Get current images from database
        db_images = self.list_images(project_id)
        db_paths = {img.path: img for img in db_images}

        added = 0
        removed = 0
        unchanged = 0

        # Find images to remove (in DB but not in filesystem)
        for db_path, img in db_paths.items():
            if db_path not in fs_paths:
                # Delete annotations first
                annotations = self.list_annotations(img.id)
                for ann in annotations:
                    self.conn.execute("DELETE FROM annotations WHERE id = ?", (ann.id,))
                # Delete image record
                self.conn.execute("DELETE FROM images WHERE id = ?", (img.id,))
                removed += 1
            else:
                unchanged += 1

        # Find images to add (in filesystem but not in DB)
        existing_paths = set(db_paths.keys())
        new_paths = [p for p in fs_paths if p not in existing_paths]
        new_paths_sorted = sorted(
            new_paths, key=lambda p: natural_sort_key(Path(p).name)
        )

        # Get max order_index for new images
        max_index = max((img.order_index for img in db_images), default=-1)

        for path in new_paths_sorted:
            max_index += 1
            img_path = Path(path)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {path}: {e}")
                continue

            self.conn.execute(
                """
                INSERT INTO images (project_id, path, width, height, order_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, path, width, height, max_index),
            )
            added += 1

        self.commit()

        # Reindex order_index to be contiguous after deletions
        if removed > 0:
            self._reindex_images(project_id)

        return {"added": added, "removed": removed, "unchanged": unchanged}

    def _reindex_images(self, project_id: int) -> None:
        """Reindex order_index to be contiguous after deletions."""
        images = self.list_images(project_id)
        images_sorted = sorted(images, key=lambda img: img.order_index)

        for new_idx, img in enumerate(images_sorted):
            if img.order_index != new_idx:
                self.conn.execute(
                    "UPDATE images SET order_index = ? WHERE id = ?",
                    (new_idx, img.id),
                )

        self.commit()

    # ==================== Image Operations ====================

    def list_images(self, project_id: int) -> list[ImageRecord]:
        """
        List all images in a project.

        Args:
            project_id: ID of the project

        Returns:
            List of ImageRecord objects ordered by order_index
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM images 
            WHERE project_id = ? 
            ORDER BY order_index
            """,
            (project_id,),
        )
        return [self._row_to_image(row) for row in cursor.fetchall()]

    def get_image_by_index(
        self, project_id: int, order_index: int
    ) -> Optional[ImageRecord]:
        """
        Get image by its order index.

        Args:
            project_id: ID of the project
            order_index: Order index of the image

        Returns:
            ImageRecord or None if not found
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM images 
            WHERE project_id = ? AND order_index = ?
            """,
            (project_id, order_index),
        )
        row = cursor.fetchone()
        return self._row_to_image(row) if row else None

    def get_image_by_id(self, image_id: int) -> Optional[ImageRecord]:
        """Get image by its ID."""
        cursor = self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        return self._row_to_image(row) if row else None

    def get_image_count(self, project_id: int) -> int:
        """Get total number of images in project."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)
        )
        return cursor.fetchone()[0]

    def _row_to_image(self, row: sqlite3.Row) -> ImageRecord:
        """Convert database row to ImageRecord object."""
        return ImageRecord(
            id=row["id"],
            project_id=row["project_id"],
            path=row["path"],
            width=row["width"],
            height=row["height"],
            order_index=row["order_index"],
        )

    # ==================== Label Operations ====================

    def upsert_label(
        self, project_id: int, name: str, color_hex: Optional[str] = None
    ) -> Label:
        """
        Create or update a label.

        Args:
            project_id: ID of the project
            name: Name of the label
            color_hex: Color in hex format (e.g., "#FF0000")

        Returns:
            Created or existing Label
        """
        # Check if label exists
        cursor = self.conn.execute(
            "SELECT * FROM labels WHERE project_id = ? AND name = ?", (project_id, name)
        )
        existing = cursor.fetchone()

        if existing:
            if color_hex and existing["color_hex"] != color_hex:
                # Update color
                self.conn.execute(
                    "UPDATE labels SET color_hex = ? WHERE id = ?",
                    (color_hex, existing["id"]),
                )
                self.commit()
            return self._row_to_label(existing)

        # Generate color if not provided
        if not color_hex:
            # Get count of existing labels for color cycling
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM labels WHERE project_id = ?", (project_id,)
            )
            count = cursor.fetchone()[0]
            color_hex = DEFAULT_COLORS[count % len(DEFAULT_COLORS)]

        # Create new label
        cursor = self.conn.execute(
            """
            INSERT INTO labels (project_id, name, color_hex)
            VALUES (?, ?, ?)
            """,
            (project_id, name, color_hex),
        )
        self.commit()

        return self.get_label_by_id(cursor.lastrowid)

    def list_labels(self, project_id: int) -> list[Label]:
        """
        List all labels in a project.

        Args:
            project_id: ID of the project

        Returns:
            List of Label objects
        """
        cursor = self.conn.execute(
            "SELECT * FROM labels WHERE project_id = ? ORDER BY name", (project_id,)
        )
        return [self._row_to_label(row) for row in cursor.fetchall()]

    def get_label_by_id(self, label_id: int) -> Optional[Label]:
        """Get label by its ID."""
        cursor = self.conn.execute("SELECT * FROM labels WHERE id = ?", (label_id,))
        row = cursor.fetchone()
        return self._row_to_label(row) if row else None

    def update_label(
        self,
        label_id: int,
        name: Optional[str] = None,
        color_hex: Optional[str] = None,
    ) -> Label:
        """
        Update a label's name and/or color.

        Args:
            label_id: ID of the label to update
            name: New name (optional)
            color_hex: New color in hex format (optional)

        Returns:
            Updated Label object

        Raises:
            ValueError: If label not found
        """
        label = self.get_label_by_id(label_id)
        if label is None:
            raise ValueError(f"Label with ID {label_id} not found")

        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if color_hex is not None:
            updates.append("color_hex = ?")
            params.append(color_hex)

        if updates:
            params.append(label_id)
            self.conn.execute(
                f"UPDATE labels SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            self.commit()

        return self.get_label_by_id(label_id)

    def _row_to_label(self, row: sqlite3.Row) -> Label:
        """Convert database row to Label object."""
        return Label(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            color_hex=row["color_hex"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.now(),
        )

    # ==================== Annotation Operations ====================

    def create_annotation(
        self,
        image_id: int,
        label_id: int,
        bbox_xyxy: list[float],
        source: str = "manual",
        status: str = "approved",
        confidence: Optional[float] = None,
        mask_rle: Optional[dict] = None,
        polygon_norm: Optional[list[float]] = None,
        track_id: Optional[int] = None,
    ) -> Annotation:
        """
        Create a new annotation.

        Args:
            image_id: ID of the image
            label_id: ID of the label
            bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixels
            source: Source of annotation ("manual", "propagated", "sam")
            status: Status ("approved", "needs_review", "rejected")
            confidence: Confidence score (0-1)
            mask_rle: Optional RLE encoded mask
            polygon_norm: Optional normalized polygon

        Returns:
            Created Annotation
        """
        cursor = self.conn.execute(
            """
            INSERT INTO annotations 
            (image_id, label_id, bbox_xyxy_json, source, status, confidence, mask_rle_json, polygon_norm_json, track_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                label_id,
                bbox_to_json(bbox_xyxy),
                source,
                status,
                confidence,
                mask_rle_to_json(mask_rle) if mask_rle else None,
                polygon_to_json(polygon_norm) if polygon_norm else None,
                track_id,
            ),
        )
        self.commit()

        return self.get_annotation_by_id(cursor.lastrowid)

    def update_annotation(self, annotation_id: int, **fields) -> Annotation:
        """
        Update an annotation.

        Args:
            annotation_id: ID of the annotation
            **fields: Fields to update (bbox_xyxy, polygon_norm, mask_rle,
                      label_id, source, status, confidence)

        Returns:
            Updated Annotation
        """
        allowed_fields = {
            "bbox_xyxy",
            "polygon_norm",
            "mask_rle",
            "label_id",
            "source",
            "status",
            "confidence",
            "track_id",
        }

        updates = []
        values = []

        for key, value in fields.items():
            if key not in allowed_fields:
                continue

            if key == "bbox_xyxy":
                updates.append("bbox_xyxy_json = ?")
                values.append(bbox_to_json(value) if value else None)
            elif key == "polygon_norm":
                updates.append("polygon_norm_json = ?")
                values.append(polygon_to_json(value) if value else None)
            elif key == "mask_rle":
                updates.append("mask_rle_json = ?")
                values.append(mask_rle_to_json(value) if value else None)
            else:
                updates.append(f"{key} = ?")
                values.append(value)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            values.append(annotation_id)

            self.conn.execute(
                f"UPDATE annotations SET {', '.join(updates)} WHERE id = ?", values
            )
            self.commit()

        return self.get_annotation_by_id(annotation_id)

    def delete_annotation(self, annotation_id: int) -> None:
        """
        Delete an annotation.

        Args:
            annotation_id: ID of the annotation to delete
        """
        self.conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        self.commit()

    def delete_annotations_after_index(self, project_id: int, after_index: int) -> int:
        """
        Delete all annotations for images after a given order_index.

        Args:
            project_id: ID of the project
            after_index: Delete annotations for images with order_index > this value
                         (0-based, so after_index=799 deletes from image 801 onward)

        Returns:
            Number of annotations deleted
        """
        cursor = self.conn.execute(
            """
            DELETE FROM annotations
            WHERE image_id IN (
                SELECT id FROM images
                WHERE project_id = ? AND order_index > ?
            )
            """,
            (project_id, after_index),
        )
        self.commit()
        return cursor.rowcount

    def list_annotations(self, image_id: int) -> list[Annotation]:
        """
        List all annotations for an image.

        Args:
            image_id: ID of the image

        Returns:
            List of Annotation objects
        """
        cursor = self.conn.execute(
            "SELECT * FROM annotations WHERE image_id = ? ORDER BY id", (image_id,)
        )
        return [self._row_to_annotation(row) for row in cursor.fetchall()]

    def get_annotation_by_id(self, annotation_id: int) -> Optional[Annotation]:
        """Get annotation by its ID."""
        cursor = self.conn.execute(
            "SELECT * FROM annotations WHERE id = ?", (annotation_id,)
        )
        row = cursor.fetchone()
        return self._row_to_annotation(row) if row else None

    def list_annotations_by_status(
        self, project_id: int, status: str
    ) -> list[tuple[Annotation, ImageRecord]]:
        """
        List annotations by status with their images.

        Args:
            project_id: ID of the project
            status: Status to filter by

        Returns:
            List of (Annotation, ImageRecord) tuples
        """
        cursor = self.conn.execute(
            """
            SELECT a.*, i.id AS img_id, i.project_id AS img_project_id,
                   i.path AS img_path, i.width AS img_width, i.height AS img_height,
                   i.order_index AS img_order_index
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ? AND a.status = ?
            ORDER BY i.order_index, a.id
            """,
            (project_id, status),
        )

        results = []
        for row in cursor.fetchall():
            ann = self._row_to_annotation(row)
            img = ImageRecord(
                id=row["img_id"],
                project_id=row["img_project_id"],
                path=row["img_path"],
                width=row["img_width"],
                height=row["img_height"],
                order_index=row["img_order_index"],
            )
            results.append((ann, img))

        return results

    def list_annotations_for_project(
        self, project_id: int
    ) -> dict[int, list[Annotation]]:
        """
        List all annotations for all images in a project.

        More efficient than calling list_annotations per image (N+1 query problem).

        Args:
            project_id: ID of the project

        Returns:
            Dict mapping image_id to list of Annotation objects
        """
        cursor = self.conn.execute(
            """
            SELECT a.* FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE i.project_id = ?
            ORDER BY a.image_id, a.id
            """,
            (project_id,),
        )

        result: dict[int, list[Annotation]] = {}
        for row in cursor.fetchall():
            ann = self._row_to_annotation(row)
            if ann.image_id not in result:
                result[ann.image_id] = []
            result[ann.image_id].append(ann)

        return result

    def _row_to_annotation(self, row: sqlite3.Row) -> Annotation:
        """Convert database row to Annotation object."""
        return Annotation(
            id=row["id"],
            image_id=row["image_id"],
            label_id=row["label_id"],
            bbox_xyxy_json=row["bbox_xyxy_json"],
            polygon_norm_json=row["polygon_norm_json"],
            mask_rle_json=row["mask_rle_json"],
            source=row["source"],
            confidence=row["confidence"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
            updated_at=datetime.fromisoformat(row["updated_at"])
            if row["updated_at"]
            else None,
            track_id=row["track_id"] if "track_id" in row.keys() else None,
        )

    # ==================== Track Operations ====================

    def create_track(
        self,
        project_id: int,
        label_id: int,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "active",
    ) -> Track:
        """Create a new track (object identity)."""
        cursor = self.conn.execute(
            "INSERT INTO tracks (project_id, label_id, name, notes, status) VALUES (?, ?, ?, ?, ?)",
            (project_id, label_id, name, notes, status),
        )
        self.commit()
        return self.get_track(cursor.lastrowid)

    def get_track(self, track_id: int) -> Optional[Track]:
        row = self.conn.execute(
            "SELECT * FROM tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return self._row_to_track(row) if row else None

    def list_tracks(
        self,
        project_id: int,
        label_id: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[Track]:
        query = "SELECT * FROM tracks WHERE project_id = ?"
        params: list = [project_id]
        if label_id is not None:
            query += " AND label_id = ?"
            params.append(label_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id"
        return [self._row_to_track(r) for r in self.conn.execute(query, params)]

    def list_track_annotations(
        self, track_id: int
    ) -> list[tuple[Annotation, ImageRecord]]:
        """All annotations of a track with their images, ordered by frame."""
        cursor = self.conn.execute(
            """
            SELECT a.*, i.id AS img_id, i.project_id AS img_project_id,
                   i.path AS img_path, i.width AS img_width, i.height AS img_height,
                   i.order_index AS img_order_index
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE a.track_id = ?
            ORDER BY i.order_index, a.id
            """,
            (track_id,),
        )
        results = []
        for row in cursor.fetchall():
            img = ImageRecord(
                id=row["img_id"],
                project_id=row["img_project_id"],
                path=row["img_path"],
                width=row["img_width"],
                height=row["img_height"],
                order_index=row["img_order_index"],
            )
            results.append((self._row_to_annotation(row), img))
        return results

    def delete_propagated_annotations(
        self, track_id: int, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> int:
        """Delete derived video annotations for one track and optional inclusive range."""
        query = """
            DELETE FROM annotations WHERE track_id = ? AND source = 'propagated'
              AND image_id IN (SELECT id FROM images WHERE project_id = (SELECT project_id FROM tracks WHERE id = ?)
        """
        params: list = [track_id, track_id]
        if start_frame is not None:
            query += " AND order_index >= ?"
            params.append(start_frame)
        if end_frame is not None:
            query += " AND order_index <= ?"
            params.append(end_frame)
        query += ")"
        cur = self.conn.execute(query, params)
        self.commit()
        return cur.rowcount

    def add_track_absences(self, track_id: int, frames) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO track_absences (track_id, frame) VALUES (?, ?)",
            [(track_id, int(f)) for f in frames],
        )
        self.commit()

    def list_track_absences(self, track_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT frame FROM track_absences WHERE track_id = ? ORDER BY frame", (track_id,)
        ).fetchall()
        return [int(r[0]) for r in rows]

    def create_issue(self, track_id: int, issue: dict, job_id: Optional[str] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO issues
               (track_id, job_id, start_frame, end_frame, worst_frame, kind, n_frames, n_lost,
                reasons_json, signals_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (track_id, job_id, issue["start"], issue["end"], issue["worst_frame"], issue["kind"],
             issue.get("n_frames", 0), issue.get("n_lost", 0), json.dumps(issue.get("reasons", {})),
             json.dumps(issue.get("signals", {}))),
        )
        self.commit()
        return int(cur.lastrowid)

    def list_issues(self, track_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM issues WHERE track_id = ? ORDER BY start_frame, id", (track_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["reasons"] = json.loads(d.pop("reasons_json") or "{}")
            d["signals"] = json.loads(d.pop("signals_json") or "{}")
            out.append(d)
        return out

    def review_issues(self, track_id: int, issue_ids: Optional[list[int]], by: str,
                      note: Optional[str] = None) -> list[dict]:
        """Mark issues as checked ('ok'); None = every open issue of the track. Returns them."""
        issues = [i for i in self.list_issues(track_id)
                  if (i["id"] in issue_ids if issue_ids is not None else not i.get("review"))]
        for i in issues:
            self.conn.execute("UPDATE issues SET review = 'ok', reviewed_by = ?, review_note = ? WHERE id = ?",
                              (by, note, i["id"]))
        self.commit()
        return issues

    def clear_issues(self, track_id: int) -> None:
        self.conn.execute("DELETE FROM issues WHERE track_id = ?", (track_id,))
        self.commit()

    def _row_to_track(self, row: sqlite3.Row) -> Track:
        return Track(
            id=row["id"],
            project_id=row["project_id"],
            label_id=row["label_id"],
            status=row["status"],
            name=row["name"],
            notes=row["notes"],
            lost_at_image_id=row["lost_at_image_id"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
            updated_at=datetime.fromisoformat(row["updated_at"])
            if row["updated_at"]
            else None,
        )

    # ==================== KV Store Operations ====================

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        _set_kv(self.conn, key, value)
        self.commit()

    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        return _get_kv(self.conn, key)
