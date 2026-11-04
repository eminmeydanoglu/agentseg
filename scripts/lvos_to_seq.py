"""Turn one LVOS v2 object into an agentseg sequence: `<i>.jpg` frames + YOLO box GT.

Reads straight from the LVOS zip, so only the chosen videos are extracted.
Keeps the frames that have an annotation PNG; GT is the object's box (class 0),
an empty label file where the object is absent.

    python scripts/lvos_to_seq.py <valid.zip> <video> <obj_id> <out_dir> [--start N --end N]
    python scripts/lvos_to_seq.py <valid.zip> --list          # videos, objects, frame counts
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def index(zf: zipfile.ZipFile):
    """{video: {"jpg": {stem: name}, "png": {stem: name}}}"""
    out: dict = defaultdict(lambda: {"jpg": {}, "png": {}})
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) < 3 or not parts[-1]:
            continue
        kind = parts[-3]
        stem, ext = Path(parts[-1]).stem, Path(parts[-1]).suffix.lower()
        if kind == "JPEGImages" and ext == ".jpg":
            out[parts[-2]]["jpg"][stem] = name
        elif kind == "Annotations" and ext == ".png":
            out[parts[-2]]["png"][stem] = name
    return out


def ids_of(zf, name) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(zf.read(name))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("video", nargs="?")
    ap.add_argument("obj", nargs="?", type=int)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="first kept annotated frame (index into annotated frames)")
    ap.add_argument("--end", type=int, default=None, help="one past the last kept annotated frame")
    a = ap.parse_args()

    zf = zipfile.ZipFile(a.zip)
    vids = index(zf)
    if a.list:
        print(json.dumps({v: {"jpg": len(d["jpg"]), "png": len(d["png"])} for v, d in sorted(vids.items())}, indent=1))
        return

    d = vids[a.video]
    stems = sorted((s for s in d["png"] if s in d["jpg"]), key=natural)[a.start:a.end]
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    present = 0
    for i, stem in enumerate(stems):
        (out / "images" / f"{i}.jpg").write_bytes(zf.read(d["jpg"][stem]))
        m = ids_of(zf, d["png"][stem]) == a.obj
        line = ""
        if m.any():
            h, w = m.shape
            ys, xs = np.nonzero(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            line = f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"
            present += 1
        (out / "labels" / f"{i}.txt").write_text(line)
    info = {"video": a.video, "obj": a.obj, "frames": len(stems), "present": present,
            "absent": len(stems) - present, "source_stems": [stems[0], stems[-1]] if stems else []}
    (out / "source.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
