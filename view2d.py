"""Real-time TOP-DOWN map of the room from the LDROBOT D500 (2D slice).
Lidar sits at the centre; you see the outline of walls/furniture at lidar height.

Run:  python view2d.py [COM9]
Close the window to stop.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ld_lidar import LidarReader

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM9"
MAX_R = 6.0  # metres shown

r = LidarReader(PORT)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title(f"LDROBOT D500 - top-down map ({PORT})")
ax.set_aspect("equal")
ax.set_xlim(-MAX_R, MAX_R)
ax.set_ylim(-MAX_R, MAX_R)
ax.set_title("room map (top view)  -  lidar at centre", color="w")

# range rings
for rr in range(1, int(MAX_R) + 1):
    ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="#264", lw=0.6))
    ax.text(0, rr, f"{rr}m", color="#4a6", fontsize=7, ha="center", va="bottom")
ax.axhline(0, color="#223", lw=0.5)
ax.axvline(0, color="#223", lw=0.5)

outline, = ax.plot([], [], "-", color="#2f8", lw=1.0, alpha=0.6)
scatter = ax.scatter([], [], s=6, c=[], cmap="turbo", vmin=0, vmax=255)
ax.plot(0, 0, "o", color="#f44", ms=8)  # lidar
info = ax.text(-MAX_R + 0.2, MAX_R - 0.4, "", color="w", fontsize=9, family="monospace")


def update(_):
    scan = r.get_scan()
    if len(scan) == 0:
        return outline, scatter, info
    ang = np.deg2rad(scan[:, 0])
    d = scan[:, 1] / 1000.0
    intens = scan[:, 2]
    m = (d > 0.02) & (d < MAX_R)
    x = d[m] * np.cos(ang[m])
    y = d[m] * np.sin(ang[m])
    order = np.argsort(scan[m, 0])
    outline.set_data(np.append(x[order], x[order][:1]), np.append(y[order], y[order][:1]))
    scatter.set_offsets(np.column_stack([x, y]))
    scatter.set_array(intens[m])
    info.set_text(f"points: {m.sum():3d}   frames: {r.frames}")
    return outline, scatter, info


ani = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    r.close()
