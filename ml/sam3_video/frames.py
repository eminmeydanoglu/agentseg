"""Frame lists -> the ``<int>.jpg`` folder layout sam3's video loader expects.

sam3 reads a session's frames from a directory of ``<int>.jpg`` files (numeric order).
``FrameSet`` keeps the caller's ordered list of image paths and builds a folder of
symlinks ``0.jpg .. N-1.jpg`` pointing at them, so frame index i <-> paths[i].
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from typing import Iterable, Optional, Sequence

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(images_dir: str, order: str = "auto") -> list[str]:
    """Image files of a directory in sequence order.

    order: "numeric" (``<int>.ext`` names by value), "name" (lexicographic) or "auto"
    (numeric when every stem is an integer, else name).
    """
    names = [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS)]
    numeric = all(re.fullmatch(r"\d+", os.path.splitext(f)[0]) for f in names)
    if order == "numeric" or (order == "auto" and numeric):
        names.sort(key=lambda f: int(os.path.splitext(f)[0]))
    else:
        names.sort()
    return [os.path.join(images_dir, f) for f in names]


class FrameSet:
    """Ordered frames plus their sam3 symlink folder.

    ``folder`` is created on first access (under ``root`` or a temp dir). The folder name
    is derived from the path list, so the same frames map to the same folder and a changed
    list never reuses a stale one.
    """

    def __init__(self, paths: Sequence[str], root: Optional[str] = None):
        if not paths:
            raise ValueError("empty frame list")
        self.paths = [os.path.abspath(p) for p in paths]
        missing = [p for p in self.paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} frames missing, e.g. {missing[0]}")
        self._root = root
        self._folder: Optional[str] = None
        self._owns_root = False
        self._size: Optional[tuple[int, int]] = None
        self._index = {p: i for i, p in enumerate(self.paths)}

    @classmethod
    def from_dir(cls, images_dir: str, order: str = "auto", root: Optional[str] = None) -> "FrameSet":
        return cls(list_images(images_dir, order), root=root)

    def __len__(self) -> int:
        return len(self.paths)

    def path(self, i: int) -> str:
        return self.paths[i]

    def index_of(self, path: str) -> int:
        return self._index[os.path.abspath(path)]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of frame 0 (all frames of a sam3 session share one size)."""
        if self._size is None:
            from PIL import Image

            with Image.open(self.paths[0]) as im:
                self._size = im.size
        return self._size

    def digest(self) -> str:
        return hashlib.sha1("\n".join(self.paths).encode()).hexdigest()[:12]

    @property
    def folder(self) -> str:
        if self._folder is None:
            if self._root is None:
                self._root = tempfile.mkdtemp(prefix="sam3frames_")
                self._owns_root = True
            self._folder = build_symlink_folder(self.paths, os.path.join(self._root, f"frames_{self.digest()}"))
        return self._folder

    def cleanup(self) -> None:
        """Remove the symlink folder (and the temp root if this FrameSet created it)."""
        if self._folder and os.path.isdir(self._folder):
            shutil.rmtree(self._folder, ignore_errors=True)
        if self._owns_root and self._root and os.path.isdir(self._root):
            shutil.rmtree(self._root, ignore_errors=True)
        self._folder = None


def build_symlink_folder(paths: Iterable[str], folder: str) -> str:
    """(Re)create ``folder`` with ``<i>.jpg -> paths[i]`` symlinks; returns its absolute path.

    The ``.jpg`` name is what sam3 lists; it decodes by content, so PNG targets work too.
    An existing folder is reused when its links already match.
    """
    paths = [os.path.abspath(p) for p in paths]
    folder = os.path.abspath(folder)
    if os.path.isdir(folder):
        entries = os.listdir(folder)
        ok = len(entries) == len(paths) and all(
            os.path.islink(os.path.join(folder, f"{i}.jpg"))
            and os.readlink(os.path.join(folder, f"{i}.jpg")) == p for i, p in enumerate(paths))
        if ok:
            return folder
        shutil.rmtree(folder)
    os.makedirs(folder)
    for i, p in enumerate(paths):
        os.symlink(p, os.path.join(folder, f"{i}.jpg"))
    return folder
