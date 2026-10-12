"""Analytics engine for a STATIONARY LDROBOT D500 in the middle of a room.

Turns the raw 360deg distance ring into meaning:
  * BackgroundModel  - learns the static walls/furniture (the "empty room").
  * foreground       - points closer than the wall = something dynamic.
  * clustering       - groups dynamic points into objects.
  * MultiTracker     - assigns stable IDs, velocity and trails to moving objects,
                       classifies person vs small object by physical width.
  * room metrics     - width / length / area / perimeter from the background.

All math is numpy; a full pipeline step is ~0.5 ms, so it runs at full lidar rate.

Angle convention here: 0deg -> +X, CCW. If your plan looks mirrored, set FLIP=True.
"""
import math
import numpy as np

# D500/LD19 raw stream is CLOCKWISE, 0deg = front of sensor. Official ROS driver
# flips to CCW (REP-103) by default -> we do the same so the plan isn't mirrored.
FLIP = True
MIN_INTENS = 5   # keep distant body returns; clustering rejects isolated noise later
MIN_RANGE_MM = 250
MAX_RANGE_MM = 12000
N = 360


def polar_to_xy(dist_mm, flip=FLIP):
    """dist_mm: length-360 array (mm). Returns x,y (m) for all bins, NaN where dist==0."""
    ang = np.deg2rad(np.arange(N))
    d = dist_mm / 1000.0
    x = d * np.cos(ang)
    y = d * np.sin(ang) * (-1.0 if flip else 1.0)
    x = np.where(dist_mm > 0, x, np.nan)
    y = np.where(dist_mm > 0, y, np.nan)
    return x, y


class BackgroundModel:
    """Per-angle estimate of the static wall distance, robust to noise & people.

    We keep a short history per azimuth bin and take a high percentile as the
    background (the "far", stable surface = the wall). A person only makes a bin
    read SHORTER; as long as they occupy the bin for less than `pct`% of the
    window, the percentile stays on the wall and they show up as foreground.
    A single noisy far spike can't ratchet the background outward (unlike a
    running-max), and a truly static object (present the whole window) is
    correctly absorbed into the background.
    """
    def __init__(self, window=90, pct=65, min_learn_frames=45):
        self.window = window
        self.pct = pct
        self.hist = np.full((N, window), np.nan, dtype=np.float32)
        self.ptr = 0
        self.filled = 0
        self.bg = np.zeros(N, dtype=np.float32)
        self.frames = 0
        self.min_learn_frames = min_learn_frames

    def update(self, dist, hold_mask=None):
        sample = np.where(dist > 0, dist, np.nan)
        if hold_mask is not None:
            # Once the room is learned, do not train the background on points
            # that are clearly closer than the wall. Otherwise a slowly moving
            # or paused person gets absorbed into the room after a few seconds.
            sample = sample.copy()
            keep_bg = hold_mask & (self.bg > 0)
            sample[keep_bg] = self.bg[keep_bg]
            sample[hold_mask & ~keep_bg] = np.nan
        self.hist[:, self.ptr] = sample
        self.ptr = (self.ptr + 1) % self.window
        self.filled = min(self.filled + 1, self.window)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.bg = np.nan_to_num(np.nanpercentile(self.hist, self.pct, axis=1), nan=0.0)
        self.frames += 1

    @property
    def ready(self):
        return self.filled >= self.min_learn_frames

    def foreground(self, dist, rel=0.035, abs_mm=90):
        """Boolean mask of bins that are dynamic (closer than the wall).
        rel: fraction of bg distance; abs_mm: floor margin in mm."""
        margin = np.maximum(abs_mm, self.bg * rel)
        return (dist > 0) & (self.bg > 0) & (dist < self.bg - margin)


def cluster_foreground(dist, mask, gap_m=0.30, min_pts=3):
    """Group masked (dynamic) bins into clusters around the 360deg ring.

    Returns list of dicts: {x,y,dist,width,npts,ang}.
    Splits when the Euclidean gap between consecutive dynamic points exceeds gap_m.
    Handles wrap-around at 0/360.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    ang = np.deg2rad(np.arange(N))
    d = dist / 1000.0
    px = d * np.cos(ang)
    py = d * np.sin(ang) * (-1.0 if FLIP else 1.0)

    def same_object_gap(a, b):
        gap = math.hypot(px[b] - px[a], py[b] - py[a])
        ang_gap = (b - a) % N
        avg_range = max(0.0, float((d[a] + d[b]) * 0.5))
        # A person at 7-8 m may occupy only 2-4 one-degree bins and can have
        # dropped bins between returns. Keep the near-field strict, but widen
        # the allowed physical gap with range.
        range_gap = min(0.75, max(gap_m, 0.12 + 0.06 * avg_range))
        max_ang_gap = 12 if avg_range >= 3.0 else 8
        return gap <= range_gap and ang_gap <= max_ang_gap

    # order the dynamic bins and cut into runs by spatial gap
    clusters = []
    cur = [idx[0]]
    for k in range(1, len(idx)):
        a, b = idx[k - 1], idx[k]
        if not same_object_gap(a, b):
            clusters.append(cur); cur = [b]
        else:
            cur.append(b)
    clusters.append(cur)

    # merge wrap-around (last touches first)
    if len(clusters) > 1:
        a, b = clusters[-1][-1], clusters[0][0]
        if same_object_gap(a, b):
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()

    out = []
    for c in clusters:
        cx = float(np.mean(px[c])); cy = float(np.mean(py[c]))
        w = float(math.hypot(px[c[-1]] - px[c[0]], py[c[-1]] - py[c[0]]))
        cd = float(math.hypot(cx, cy))
        far_sparse_person = len(c) >= 2 and cd >= 2.0 and 0.07 <= w <= 1.25
        if len(c) < min_pts and not far_sparse_person:
            continue
        out.append({"x": cx, "y": cy, "dist": cd, "width": w, "idxs": list(map(int, c)),
                    "npts": len(c), "ang": float(math.degrees(math.atan2(cy, cx)) % 360)})
    return out


class Track:
    _next = 1

    def __init__(self, c):
        self.id = Track._next; Track._next += 1
        self.x = c["x"]; self.y = c["y"]
        self.vx = 0.0; self.vy = 0.0
        self.width = c["width"]; self.npts = c["npts"]
        self.idxs = c.get("idxs", [])
        self.age = 1; self.missed = 0
        self.trail = [(self.x, self.y)]
        self.total_dist = 0.0
        self.min_x = self.max_x = self.x
        self.min_y = self.max_y = self.y
        self.motion_span = 0.0
        self.ever_moved = False

    def predict(self):
        return self.x + self.vx, self.y + self.vy

    def update(self, c, a=0.5):
        nx, ny = c["x"], c["y"]
        ndx, ndy = nx - self.x, ny - self.y
        step = math.hypot(ndx, ndy)
        self.total_dist += step
        self.vx = (1 - a) * self.vx + a * ndx
        self.vy = (1 - a) * self.vy + a * ndy
        self.min_x = min(self.min_x, nx); self.max_x = max(self.max_x, nx)
        self.min_y = min(self.min_y, ny); self.max_y = max(self.max_y, ny)
        self.motion_span = math.hypot(self.max_x - self.min_x, self.max_y - self.min_y)
        if self.motion_span >= 0.45:
            self.ever_moved = True
        self.x, self.y = nx, ny
        self.width = 0.6 * self.width + 0.4 * c["width"]
        self.npts = c["npts"]
        self.idxs = c.get("idxs", [])
        self.age += 1; self.missed = 0
        self.trail.append((self.x, self.y))
        if len(self.trail) > 60:
            self.trail.pop(0)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def kind(self):
        # person-sized cluster at torso/leg height vs small object
        strong_shape = 0.12 <= self.width <= 0.9 and self.npts >= 3
        sparse_moving_shape = 0.09 <= self.width <= 0.45 and self.npts >= 2 and self.ever_moved
        if self.ever_moved and (strong_shape or sparse_moving_shape):
            return "person"
        if self.width < 0.12:
            return "object"
        return "object"

    def to_dict(self):
        return {"id": self.id, "x": round(self.x, 3), "y": round(self.y, 3),
                "vx": round(self.vx, 3), "vy": round(self.vy, 3),
                "speed": round(self.speed, 3), "width": round(self.width, 3),
                "npts": self.npts, "age": self.age, "kind": self.kind,
                "missed": self.missed, "ever_moved": self.ever_moved,
                "motion_span": round(self.motion_span, 3),
                "dist": round(math.hypot(self.x, self.y), 3),
                "ang": round(math.degrees(math.atan2(self.y, self.x)) % 360, 1),
                "moving": self.age > 3 and self.speed > 0.01,
                "trail": [[round(p[0], 3), round(p[1], 3)] for p in self.trail[-40:]]}


class MultiTracker:
    def __init__(self, gate=0.6, max_missed=10):
        self.tracks = []
        self.gate = gate
        self.max_missed = max_missed

    def _gate_for(self, track):
        dist = math.hypot(track.x, track.y)
        return min(0.75, max(0.32, 0.20 + 0.07 * dist))

    def step(self, clusters):
        # greedy nearest-neighbour association on predicted positions
        unused = list(range(len(clusters)))
        for t in self.tracks:
            px, py = t.predict()
            best, bd = -1, min(self.gate, self._gate_for(t))
            for j in unused:
                c = clusters[j]
                dd = math.hypot(c["x"] - px, c["y"] - py)
                if dd < bd:
                    bd, best = dd, j
            if best >= 0:
                t.update(clusters[best]); unused.remove(best)
            else:
                t.missed += 1
        # spawn new tracks
        for j in unused:
            self.tracks.append(Track(clusters[j]))
        # cull stale
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks

    def confirmed(self):
        return [t for t in self.tracks if t.age >= 3]


def room_metrics(bg, flip=FLIP):
    """From the background wall ring, estimate room dimensions."""
    x, y = polar_to_xy(bg, flip)
    m = ~np.isnan(x)
    if m.sum() < 10:
        return None
    xs, ys = x[m], y[m]
    minx, maxx, miny, maxy = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    # shoelace area on the ordered wall polygon
    order = np.argsort(np.arange(N)[m])
    poly_x, poly_y = xs[order], ys[order]
    area = 0.5 * abs(np.dot(poly_x, np.roll(poly_y, -1)) - np.dot(poly_y, np.roll(poly_x, -1)))
    perim = float(np.sum(np.hypot(np.diff(poly_x, append=poly_x[0]),
                                  np.diff(poly_y, append=poly_y[0]))))
    return {"w": round(maxx - minx, 2), "l": round(maxy - miny, 2),
            "area": round(float(area), 2), "perim": round(perim, 2),
            "minx": round(minx, 2), "maxx": round(maxx, 2),
            "miny": round(miny, 2), "maxy": round(maxy, 2)}


class Pipeline:
    """Convenience wrapper: feed raw 360 dist arrays, get everything out."""
    def __init__(self):
        self.reset()

    def reset(self):
        Track._next = 1
        self.bg = BackgroundModel()
        self.tracker = MultiTracker()

    def _should_output_track(self, track):
        if track.missed > 0:
            return False
        if track.kind == "person":
            return track.age >= 8
        # Keep the dashboard and zone logic quiet: short object fragments are
        # usually edges/noise during a pass, not meaningful tracked objects.
        return (
            track.age >= 10 and
            track.npts >= 4 and
            track.width >= 0.10 and
            (track.motion_span >= 0.25 or track.age >= 30)
        )

    def process(self, dist, intens, speed_dps):
        # gate out weak/noisy returns so walls & tracks stay clean
        valid = (
            (dist >= MIN_RANGE_MM) &
            (dist <= MAX_RANGE_MM) &
            (intens >= MIN_INTENS)
        )
        dist = np.where(valid, dist, 0.0).astype(np.float32)
        mask = self.bg.foreground(dist) if self.bg.ready else np.zeros(N, bool)
        clusters = cluster_foreground(dist, mask)
        tracks = self.tracker.step(clusters)
        confirmed = [t.to_dict() for t in tracks if self._should_output_track(t)]

        hold = np.zeros(N, dtype=bool)
        if self.bg.ready:
            for t in tracks:
                if t.missed == 0 and t.age >= 2 and (t.kind == "person" or t.ever_moved):
                    for idx in t.idxs:
                        hold[(idx - 1) % N] = True
                        hold[idx % N] = True
                        hold[(idx + 1) % N] = True
        self.bg.update(dist, hold_mask=hold if hold.any() else None)

        x, y = polar_to_xy(dist)
        bx, by = polar_to_xy(self.bg.bg)
        pts = [[round(float(x[i]), 3), round(float(y[i]), 3), int(intens[i])]
               for i in range(N) if dist[i] > 0]
        wall = [[round(float(bx[i]), 3), round(float(by[i]), 3)]
                for i in range(N) if self.bg.bg[i] > 0]

        return {
            "pts": pts,
            "wall": wall,
            "fg": [[round(float(x[i]), 3), round(float(y[i]), 3)] for i in np.where(mask)[0]],
            "objects": confirmed,
            "n_people": sum(1 for t in confirmed if t["kind"] == "person"),
            "fg_count": int(mask.sum()),
            "cluster_count": len(clusters),
            "room": room_metrics(self.bg.bg) if self.bg.ready else None,
            "bg_ready": bool(self.bg.ready),
            "rpm": round(speed_dps / 6.0, 1),
        }


if __name__ == "__main__":
    import sys, time
    from ld_lidar import LidarReader
    port = sys.argv[1] if len(sys.argv) > 1 else "COM9"
    r = LidarReader(port); pipe = Pipeline()
    print("Learning background & detecting... (walk in front of the lidar)\n")
    t0 = time.time()
    while time.time() - t0 < 12:
        dist, intens = r.get_arrays()
        out = pipe.process(dist, intens, r.speed)
        objs = out["objects"]
        line = (f"rpm={out['rpm']:4.1f} bg={'Y' if out['bg_ready'] else '.'} "
                f"people={out['n_people']} objs={len(objs)}")
        if objs:
            o = objs[0]
            line += f" | id{o['id']} {o['kind']} @ {o['dist']:.2f}m {o['ang']:.0f}deg v={o['speed']*10:.2f}"
        if out["room"]:
            line += f" | room {out['room']['w']}x{out['room']['l']}m {out['room']['area']}m2"
        print("\r" + line + " " * 8, end="", flush=True)
        time.sleep(0.1)
    print("\ndone."); r.close()
