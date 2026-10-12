import argparse
import asyncio
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import websockets


def key_point(obj):
    return float(obj.get("x", 0.0)), float(obj.get("y", 0.0))


async def run(url, seconds, out_path, progress_every):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "messages": 0,
        "first_frames": None,
        "last_frames": None,
        "people_counts": Counter(),
        "object_counts": Counter(),
        "track_samples": defaultdict(int),
        "track_kind": {},
        "track_first": {},
        "track_last": {},
        "track_min_dist": defaultdict(lambda: float("inf")),
        "track_max_dist": defaultdict(float),
        "track_path": defaultdict(float),
        "track_last_xy": {},
        "track_seen_moving": defaultdict(bool),
        "fg_sum": 0,
        "cluster_sum": 0,
        "rpm_sum": 0.0,
        "pts_sum": 0,
    }
    start = time.time()
    next_progress = start + progress_every

    with out_path.open("w", encoding="utf-8") as f:
        async with websockets.connect(url, max_size=2**20, open_timeout=10) as ws:
            while True:
                now = time.time()
                if now - start >= seconds:
                    break
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(msg)
                data["_t"] = round(now, 3)
                f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")

                stats["messages"] += 1
                frames = data.get("frames")
                if stats["first_frames"] is None:
                    stats["first_frames"] = frames
                stats["last_frames"] = frames
                objects = data.get("objects", [])
                people = int(data.get("n_people", 0) or 0)
                stats["people_counts"][people] += 1
                stats["object_counts"][len(objects)] += 1
                stats["fg_sum"] += int(data.get("fg_count", 0) or 0)
                stats["cluster_sum"] += int(data.get("cluster_count", 0) or 0)
                stats["rpm_sum"] += float(data.get("rpm", 0.0) or 0.0)
                stats["pts_sum"] += len(data.get("pts", []))

                for obj in objects:
                    tid = int(obj.get("id", -1))
                    if tid < 0:
                        continue
                    x, y = key_point(obj)
                    stats["track_samples"][tid] += 1
                    stats["track_kind"][tid] = obj.get("kind", "?")
                    stats["track_first"].setdefault(tid, now)
                    stats["track_last"][tid] = now
                    dist = float(obj.get("dist", math.hypot(x, y)) or 0.0)
                    stats["track_min_dist"][tid] = min(stats["track_min_dist"][tid], dist)
                    stats["track_max_dist"][tid] = max(stats["track_max_dist"][tid], dist)
                    if tid in stats["track_last_xy"]:
                        lx, ly = stats["track_last_xy"][tid]
                        step = math.hypot(x - lx, y - ly)
                        if step < 1.2:
                            stats["track_path"][tid] += step
                    stats["track_last_xy"][tid] = (x, y)
                    if obj.get("moving") or obj.get("ever_moved") or float(obj.get("speed", 0) or 0) > 0.04:
                        stats["track_seen_moving"][tid] = True

                if now >= next_progress:
                    elapsed = now - start
                    avg_fg = stats["fg_sum"] / max(1, stats["messages"])
                    avg_clusters = stats["cluster_sum"] / max(1, stats["messages"])
                    print(
                        f"t={elapsed:5.0f}s msg={stats['messages']:4d} "
                        f"people={people} objects={len(objects)} fg_avg={avg_fg:.1f} "
                        f"cl_avg={avg_clusters:.1f} frames={frames}",
                        flush=True,
                    )
                    next_progress += progress_every

    elapsed = time.time() - start
    frame_delta = 0
    if stats["first_frames"] is not None and stats["last_frames"] is not None:
        frame_delta = int(stats["last_frames"] - stats["first_frames"])

    tracks = []
    for tid, count in stats["track_samples"].items():
        duration = max(0.0, stats["track_last"][tid] - stats["track_first"][tid])
        tracks.append({
            "id": tid,
            "kind": stats["track_kind"].get(tid, "?"),
            "samples": count,
            "duration_s": round(duration, 1),
            "path_m": round(stats["track_path"][tid], 2),
            "min_dist_m": round(stats["track_min_dist"][tid], 2),
            "max_dist_m": round(stats["track_max_dist"][tid], 2),
            "seen_moving": bool(stats["track_seen_moving"][tid]),
        })
    tracks.sort(key=lambda t: (-t["duration_s"], -t["samples"]))

    summary = {
        "duration_s": round(elapsed, 1),
        "messages": stats["messages"],
        "frame_delta": frame_delta,
        "packet_rate_hz": round(frame_delta / elapsed, 1) if elapsed else 0,
        "avg_rpm": round(stats["rpm_sum"] / max(1, stats["messages"]), 1),
        "avg_raw_points": round(stats["pts_sum"] / max(1, stats["messages"]), 1),
        "avg_fg_count": round(stats["fg_sum"] / max(1, stats["messages"]), 1),
        "avg_cluster_count": round(stats["cluster_sum"] / max(1, stats["messages"]), 1),
        "people_counts": dict(sorted(stats["people_counts"].items())),
        "object_counts": dict(sorted(stats["object_counts"].items())),
        "tracks": tracks[:40],
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8081")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-every", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.seconds, Path(args.out), args.progress_every))


if __name__ == "__main__":
    main()
