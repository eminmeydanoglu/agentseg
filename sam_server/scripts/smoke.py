"""GPU smoke test against a running sam_server.

uv run python scripts/smoke.py --url http://localhost:8765 --frames /tmp/frames \
    --image /tmp/frames/100.jpg --image-prompt apple --video-prompt "robot hand"
"""

import argparse
import json
import subprocess
import threading
import time

import httpx

from sam_server.codec import decode


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--image-prompt", default="apple")
    ap.add_argument("--video-prompt", default="robot hand")
    ap.add_argument("--model", default="sam3.1")
    ap.add_argument("--evict", action="store_true", help="propagate evict_cached_frame_outputs")
    ap.add_argument("--offload-video", action="store_true", help="start_session offload_video_to_cpu")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.url, timeout=600)

    while not c.get("/health").json()["ready"]:
        time.sleep(2)
    info = c.get("/info").json()
    print("models:", info["models"], "load_times:", info["load_times_sec"])
    print("sam3_commit:", info["sam3_commit"], "torch:", info["torch"])
    print(f"VRAM after load: allocated={info['gpu']['allocated_bytes']/2**30:.2f} GiB "
          f"reserved={info['gpu']['reserved_bytes']/2**30:.2f} GiB nvidia-smi={gpu_used_mib()} MiB")

    peak = {"mib": 0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["mib"] = max(peak["mib"], gpu_used_mib())
            time.sleep(0.2)

    threading.Thread(target=sampler, daemon=True).start()

    # ── image ──
    if "image" in info["models"]:
        t = time.time()
        sid = c.post("/image/set_image", json={"image_path": a.image}).json()["state_id"]
        t1 = time.time()
        r = c.post("/image/call/set_text_prompt",
                   json={"state_id": sid, "kwargs": {"prompt": a.image_prompt}}).json()
        t2 = time.time()
        res = decode(r["result"])
        print(f"[image] set_image {1e3*(t1-t):.0f} ms, set_text_prompt('{a.image_prompt}') "
              f"{1e3*(t2-t1):.0f} ms: {len(res['scores'])} dets, scores={res['scores']}, "
              f"presence={res['presence']:.3f}, masks={res['masks'].shape}, boxes={res['boxes']}")
        if len(res["scores"]):
            x0, y0, x1, y1 = res["boxes"][0]
            pt = [[(x0 + x1) / 2, (y0 + y1) / 2]]
        else:
            pt = [[100, 100]]
        t = time.time()
        r = c.post("/image/call/predict_inst",
                   json={"state_id": sid, "kwargs": {"point_coords": pt, "point_labels": [1],
                                                      "multimask_output": True}})
        r.raise_for_status()
        masks, scores, logits = decode(r.json()["result"])
        print(f"[image] predict_inst {1e3*(time.time()-t):.0f} ms: masks {masks.shape} "
              f"{masks.dtype}, scores {scores}, logits {logits.shape}")
        c.delete(f"/image/states/{sid}")

    # ── video ──
    m = a.model
    t = time.time()
    s = c.post(f"/video/{m}/request", json={"type": "start_session",
                                            "resource_path": a.frames,
                                            "offload_video_to_cpu": a.offload_video}).json()
    if "session_id" not in s:
        raise RuntimeError(s)
    sess = s["session_id"]
    print(f"[video] start_session {time.time()-t:.2f} s -> {sess}")
    t = time.time()
    r = decode(c.post(f"/video/{m}/request", json={"type": "add_prompt", "session_id": sess,
                                                    "frame_index": 0, "text": a.video_prompt}).json())
    print(f"[video] add_prompt('{a.video_prompt}') {time.time()-t:.2f} s: frame "
          f"{r['frame_index']}, keys={sorted(r['outputs'])}, obj_ids={r['outputs'].get('out_obj_ids')}")
    n, nobj, t = 0, 0, time.time()
    first = None
    with c.stream("POST", f"/video/{m}/stream_request",
                  json={"type": "propagate_in_video", "session_id": sess,
                        "propagation_direction": "forward",
                        "evict_cached_frame_outputs": a.evict}) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            item = decode(json.loads(line))
            if "error" in item:
                raise RuntimeError(f"after {n} frames: {item}")
            if first is None:
                first = time.time() - t
            n += 1
            nobj = max(nobj, len(item["outputs"].get("out_obj_ids", [])))
            masks = item["outputs"].get("out_binary_masks")
    dt = time.time() - t
    print(f"[video] propagate forward: {n} frames in {dt:.2f} s = {n/dt:.1f} FPS over HTTP "
          f"(first frame {first:.2f} s), max objs {nobj}, last masks "
          f"{None if masks is None else masks.shape}")

    # cancel during a running stream
    got = []

    def consume():
        with c.stream("POST", f"/video/{m}/stream_request",
                      json={"type": "propagate_in_video", "session_id": sess,
                            "propagation_direction": "forward"}) as resp:
            for line in resp.iter_lines():
                if line:
                    got.append(json.loads(line))

    th = threading.Thread(target=consume)
    th.start()
    while len(got) < 5 and th.is_alive():
        time.sleep(0.01)
    cr = httpx.post(f"{a.url}/video/{m}/request",
                    json={"type": "cancel_propagation", "session_id": sess}).json()
    th.join()
    print(f"[video] cancel_propagation -> {cr}; stream stopped after {len(got)} lines, "
          f"last={list(got[-1])}")

    info = c.get("/info").json()
    r = c.post(f"/video/{m}/request", json={"type": "close_session", "session_id": sess,
                                            "clear_cache_threshold": 0}).json()
    stop.set()
    print(f"VRAM peak: torch max_allocated={info['gpu']['max_allocated_bytes']/2**30:.2f} GiB, "
          f"nvidia-smi peak={peak['mib']} MiB (whole GPU, incl. other processes)")
    print("close_session:", {k: v for k, v in r.items() if k != "gpu_mem"})


if __name__ == "__main__":
    main()
