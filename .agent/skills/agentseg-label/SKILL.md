---
name: agentseg-label
description: Label or track an object through a video / image sequence in this repo with the `agentseg` CLI (SAM 3 tracking + health checks). Use for requests like "label the torpedo", "label the board in this folder", "track X through these frames", "fix the labels of track N", or exporting those labels.
---

# Labeling a sequence with `agentseg`

You segment the object once, SAM 3 tracks it through every frame, and the
backend flags frames that look wrong ("issues"). Your job is to give a good
first mask, then inspect issues and fix only what is actually wrong. Most
sequences need 0–3 fixes.

Run the CLI from the repo root as `.venv/bin/agentseg …` (or `uv run agentseg …`).
Every command prints one JSON object; read `ok`, `error` and `next`. Images are
written as PNG files and returned as `"image": "<path>"`. **Open that path with
your file-reading tool to see it.** You cannot judge a mask without looking.

Conventions: pixel coordinates, boxes `x0,y0,x1,y1`, points `x,y`, frames are
0-based indices in the project's image order. Frame ranges are written like
`120-139,150`.

## 0. Services

```
agentseg status            # backend + sam_server + open project
agentseg up                # starts whatever is down (sam_server takes ~1–2 min)
```

If `up` fails, read the log path it prints. Do not kill other GPU processes.
**Never run `agentseg down`** during a run: the backend hosts you, and stopping it
ends your run. The GPU may be shared, so a command can fail with a sam_server
error (CUDA out of memory). Wait 30–60 s and retry it; `agentseg up` only starts
what is down. If it keeps failing, stop and report.

## 1. Open the project and find the object

```
agentseg project open <image_folder>     # or an existing project dir (has agentseg.db)
agentseg sheet                           # ~24 thumbnails across the whole sequence
```

`project open` returns `frames` and `image_size`. Look at the sheet. Pick a
prompt frame where the object is **fully visible, large and sharp**, not
necessarily frame 0. Tracking runs both forward and backward from keyframes.

## 2. Resolve the label before the first mask

Choose the label name before calling `segment` or creating a track. Use the
object name from the user's request; when it is not explicit, infer a short,
specific name from the image and state that choice in the run. Never leave the
label as `unknown`, an empty string, or a placeholder. If there are several
plausible objects and the choice changes the annotation target, ask the user
before running SAM. The resolved name is carried into the first `track create`
call and every later report.

For example, for “label the torpedo” resolve `torpedo`; for “label the board
in this folder” resolve `board` (or `torpedo_board` when that is the clear
object identity). Keep it stable for the entire sequence.

Register it in the project's label list before the first mask (this is the
same list a person picks from in the web UI; reuse an existing label when one
already names the object):

```
agentseg labels                     # existing labels
agentseg labels --add <resolved_label>
```

## 3. First mask

```
agentseg view <f> --grid                          # 50 px labelled grid for reading coordinates
agentseg segment <f> --box x0,y0,x1,y1            # preview only, nothing saved
```

**Boxes: snug, around the whole object.** On the grid, find the object's
leftmost, rightmost, top and bottom pixels (bumper to bumper, wheels, roof),
then add a small margin (about 5–10% of its size). Not more: a box with lots
of room lets SAM take in the object's **shadow** or the ground around it, and
the tracker then copies that to every frame.

- Read `check`: `segment` also tries a 20% larger box. A `WARNING` that the mask
  grows with the larger box means your box cut the object off. If the zoom of
  that bigger mask shows the whole object, use its `padded_preview_id`;
  otherwise redraw the box around all of the object. Never ignore it and
  re-run the same box.
- `segment` returns `image` (whole frame, yellow fill) and `zoom` (close-up,
  outline only, cyan = the box used). **Open the zoom** and walk around the
  outline: every end of the object must be inside it.
- **Too much?** The outline must follow the object's own edge. If it runs into
  the shadow under or beside the object, the road or a neighbour, re-run with
  `--neg x,y` on that part (and a snugger box).
- **Still a partial mask?** Do not re-run the same box: it gives the same
  mask. Put `--pos x,y` on the part that is missing (SAM often stops at a
  color or light boundary: sunlit half, shadowed half, two-tone vehicle).
- **Scope:** include exactly the object the user means: "the board" without its
  stand, "the car" without the trailer.
- Up to 4 tries. A keyframe that covers 2/3 of the object is worse than none:
  the tracker copies it to every frame and health cannot see it. If you cannot
  get the whole object on this frame, pick another frame. Then:

```
agentseg track create --label <resolved_label> --preview <preview_id>
agentseg track propagate <track_id>               # ~1 min per 500 frames; returns issues
```

For keyframes added later, pass `--track T` to `segment`: the zoom then also
shows the track's nearest keyframe, so you can check it is the same object.

## 3b. When tracking loses the object

SAM 3 usually picks the object up again by itself after it was gone for a
while, so short gaps need nothing. `propagate` stops tracking in a direction
only once the mask has been empty for 60 frames in a row, and returns `lost`:

```
"lost": [{"keyframe": 420, "direction": "forward", "lost_from": 579, "stopped_at": 638, "untracked": [639, 999]}]
```

The more common failure is the opposite: after a gap SAM comes back on a
*different*, similar object. Wherever the track reappears after empty frames
(red in `sheet --track`), check identity with `view <first frame after the gap> --track T --zoom`.

Deal with `lost` **before** issues. Look at what really happens from
`lost_from`: `agentseg sheet --frames <lost_from-5>-<lost_from+40>` (without
`--track`, so you see the plain frames).

- The object is gone (tunnel, behind a building, camera looks away, a cut):
  find where it comes back, `track absent T <lost_from>-<last gone frame> --no-propagate`,
  then `segment` + `track refine` on a clear frame where it is back.
- The object is still visible: the tracker dropped it. `segment` +
  `track refine` on a clear frame right after `lost_from`.
- It never comes back before the end: mark the rest `absent`.

Each fix re-propagates; repeat until there is no `lost`.

Use the label name the user gave, or a short descriptive one such as `torpedo_board`.

## 4. Review issues

`propagate` (and `refine`, `absent`, `keyframes --rm`) return the current
issue list:

```
{"id": 9, "start": 210, "end": 228, "worst_frame": 218, "kind": "drift", "n_frames": 19, "reasons": {...}}
```

For each issue, look at its `worst_frame`:

```
agentseg view <worst_frame> --track <T> --grid
```

In the view, the track mask is magenta with an outline. A yellow box marks a
keyframe and a cyan box a propagated frame. The tint hides the object's color;
for anything about identity or coverage use the close-up:

```
agentseg view <f> --track <T> --zoom     # [nearest keyframe] [frame] [frame + outline], no fill
```

**Identity.** After an occlusion, a tunnel, a camera cut or a fast pan, SAM
often jumps to a *similar* object nearby. Compare with the keyframe panel:
same color, same body type (SUV vs sedan, hatchback vs trunk), same details?
"A dark car in the right place" is not enough. Any issue whose `reasons` has
`dino_low` (the mask looks unlike the keyframe) needs a zoom comparison before
you may review it. If it is another object, the real one is either elsewhere
(`segment` + `track refine` on it) or not visible (`track absent`).

| What you see | Action |
|---|---|
| Mask is correct; the object just got smaller, larger, rotated or moved fast | **Mark it checked:** `agentseg track review T <issue_id> --note "<one line why>"`. Its frames turn from review to approved. |
| Mask spills onto background or other objects, or has stray specks far away (huge bbox) | `segment` that frame with a tight box (+ `--neg` on the spill), then `track refine T <f> --preview P` |
| Mask covers only part of the object (check the zoom), or sits on the wrong object (compare with the keyframe panel) | Same: `segment` → `track refine` |
| Object is not visible at all (out of frame, fully occluded), but a mask exists | `track absent T <range>`. Find the range first with `sheet --track T --frames a-b`. |
| Object is visible but there is no mask (red "empty" in the sheet) | `segment` → `track refine` on a frame where it is clearly visible |

Rules:
- **Never refine without looking at the preview first.** A bad keyframe poisons
  the tracker for many frames.
- If a keyframe you added made things worse: `agentseg track keyframes T --rm <f>`.
- **Batch fixes:** add `--no-propagate` to all but the last `refine` or
  `absent`, or follow them with one `agentseg track propagate T`. Each
  propagation costs 30–60 s.
- For `absent`, check the boundaries. Use `sheet --track T --frames a-b`
  around the range and mark only frames where the object is truly not visible
  (a sliver at the image edge still counts as visible).
- Same issue still there after 2 fixes → stop fixing it, leave it open (do not
  `review` it) and mention it in your report. Open issues are what a person
  checks later.
- Only `review` an issue after you looked at its frames. For a long issue,
  check a `sheet --track T --frames a-b` of the range, not just `worst_frame`.
  Several checked issues at once: `track review T 2,3,5 --note ...`, but only
  issues of the same kind that you actually checked; the note must be true
  for every id in it. Never review a `lost` issue (empty frames) as "mask correct".
- Budget: about 10 fixes and about 40 viewed images per sequence. Prefer one
  `sheet` over many `view`s when scanning a range.

## 5. Final check and export

```
agentseg sheet --track T                 # border colours: yellow keyframe, green ok,
                                        # orange review, grey absent, red no mask
agentseg export                          # YOLO-seg (default); --format yolo-detect|coco
```

Look at the final sheet. Orange frames belong to issues still open: check and
`review` them, fix them, or leave them open on purpose for a person. Red frames need attention, unless the object is truly
absent there; then mark them `absent`.

## 6. Report to the user

Short:
- track id and label
- number of frames
- keyframes (frames and why)
- absent ranges
- issues reviewed as fine, with one-line reasons
- issues left unresolved
- export path
- final sheet path

## Reference

```
agentseg status | up [--no-sam] [--no-ui] | down [--all]
agentseg project open <dir> [--name N] [--project-dir D]
agentseg labels [--add NAME]
agentseg view <f> [--track T] [--grid] [--preview P] [--zoom]
agentseg sheet [--track T] [--every N | --frames a-b,c]
agentseg segment <f> --box x0,y0,x1,y1 [--pos x,y]... [--neg x,y]... [--track T] [--no-grid]
agentseg track create --label L --preview P
agentseg track propagate T [--full] [--start a --end b] [--max-lost N] [--no-wait]
agentseg track issues T [--verbose]
agentseg track review T <ids|all> [--note TEXT]
agentseg track refine T <f> (--preview P | --box ... [--pos ..] [--neg ..]) [--no-propagate]
agentseg track keyframes T [--rm F] [--no-propagate]
agentseg track absent T <frames> [--no-propagate]
agentseg track list
agentseg export [--format yolo-seg|yolo-detect|coco] [--out DIR] [--approved-only] [--labels-only]
```

All commands accept `--help`. Errors print `{"ok": false, "error": "..."}` and
exit non-zero; read the message, since it usually names the fix.
