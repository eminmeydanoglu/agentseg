"""
Database initialization and migrations for AgentSeg.
"""

import sqlite3
from typing import Optional

# Current schema version
# Keep the public schema version at 2 for compatibility with existing project
# databases; the v3 tables are additive and are created idempotently below.
SCHEMA_VERSION = 2


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    # FastAPI executes sync handlers in a worker pool and background jobs in
    # their own threads; ProjectStore connections are protected by the store
    # lock at the API layer and must therefore be usable across those threads.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    """
    Initialize a new database with all required tables.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        # Version tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Projects table (for multi-project support, though typically one per DB)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                root_dir TEXT NOT NULL,
                db_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings_json TEXT
            )
        """)
        
        # Labels/classes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                color_hex TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, name)
            )
        """)
        
        # Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                order_index INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, path)
            )
        """)
        
        # Create index for efficient ordering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_order 
            ON images(project_id, order_index)
        """)
        
        # Annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                bbox_xyxy_json TEXT,
                polygon_norm_json TEXT,
                mask_rle_json TEXT,
                source TEXT DEFAULT 'manual',
                confidence REAL,
                status TEXT DEFAULT 'approved',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                track_id INTEGER REFERENCES tracks(id) ON DELETE SET NULL,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE
            )
        """)
        
        # Create index for efficient lookup by image
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_annotations_image 
            ON annotations(image_id)
        """)

        # Object identity (v2): tracks + annotations.track_id
        _create_tracks_schema(cursor)

        # Video tracking metadata (v3).  Absent frames are explicit engine
        # conditioning state and issues are durable job diagnostics.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_absences (
                track_id INTEGER NOT NULL,
                frame INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (track_id, frame),
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                job_id TEXT,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                worst_frame INTEGER NOT NULL,
                kind TEXT NOT NULL,
                n_frames INTEGER NOT NULL DEFAULT 0,
                n_lost INTEGER NOT NULL DEFAULT 0,
                reasons_json TEXT,
                signals_json TEXT,
                review TEXT,  -- NULL = open, 'ok' = checked, mask is right
                reviewed_by TEXT,
                review_note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_track ON issues(track_id)")
        
        # Key-value store for settings, last open image, etc.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Record schema version
        cursor.execute("""
            INSERT OR IGNORE INTO schema_version (version) VALUES (?)
        """, (SCHEMA_VERSION,))
        
        conn.commit()
    finally:
        conn.close()


def get_schema_version(db_path: str) -> int:
    """Get the current schema version of the database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def migrate_db(db_path: str) -> None:
    """
    Run any pending migrations on the database.
    
    Args:
        db_path: Path to the SQLite database file
    """
    current_version = get_schema_version(db_path)
    
    # Run historical migrations when needed, and always apply the additive
    # video tables (their CREATE statements are idempotent).
    conn = get_connection(db_path)
    try:
        # Migration 0 -> 1: Initial schema (handled by init_db)
        if current_version < 1:
            init_db(db_path)
        if current_version < 2:
            _migrate_v1_to_v2(conn)
        _migrate_v2_to_v3(conn)
        conn.commit()
    finally:
        conn.close()


TRACK_STATUSES = ("active", "lost", "ended")


def _create_tracks_schema(cursor) -> None:
    """Create the tracks table and track indexes (idempotent).

    Assumes annotations.track_id exists (either in CREATE TABLE or added by
    the v1->v2 migration).
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            label_id INTEGER NOT NULL,
            name TEXT,
            notes TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            lost_at_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracks_project ON tracks(project_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_track ON annotations(track_id)"
    )


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 -> v2: add tracks table and nullable annotations.track_id.

    Existing annotations keep track_id = NULL (no identity); new initial
    labels get tracks going forward.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(annotations)")}
    if "track_id" not in cols:
        conn.execute(
            "ALTER TABLE annotations ADD COLUMN track_id INTEGER "
            "REFERENCES tracks(id) ON DELETE SET NULL"
        )
    _create_tracks_schema(conn.cursor())
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (2)")


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """v2 -> v3: durable absent-frame and health-issue records."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS track_absences (
            track_id INTEGER NOT NULL,
            frame INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (track_id, frame),
            FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            job_id TEXT,
            start_frame INTEGER NOT NULL,
            end_frame INTEGER NOT NULL,
            worst_frame INTEGER NOT NULL,
            kind TEXT NOT NULL,
            n_frames INTEGER NOT NULL DEFAULT 0,
            n_lost INTEGER NOT NULL DEFAULT 0,
            reasons_json TEXT,
            signals_json TEXT,
            review TEXT,  -- NULL = open, 'ok' = checked, mask is right
            reviewed_by TEXT,
            review_note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_issues_track ON issues(track_id)")
    have = {r[1] for r in cur.execute("PRAGMA table_info(issues)")}
    for col in ("review", "reviewed_by", "review_note"):
        if col not in have:
            cur.execute(f"ALTER TABLE issues ADD COLUMN {col} TEXT")


def _set_kv(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set a key-value pair in the kv table."""
    conn.execute(
        "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
        (key, value)
    )


def _get_kv(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Get a value from the kv table."""
    cursor = conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None
