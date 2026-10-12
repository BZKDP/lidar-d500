"""Real-time 3D room view from the LDROBOT D500.

The D500 is a 2D lidar: it measures one horizontal ring around itself.
To make it look like a room, we EXTRUDE that ring upward into walls
(floor contour + vertical wall lines + ceiling contour). You can orbit
the view with the mouse while it updates live.

Uses Open3D if installed (smooth orbit), otherwise falls back to matplotlib 3D.

Run:  python view3d.py [COM9]
Mouse: drag = rotate, wheel = zoom.  Close window / Ctrl+C to stop.
"""
import sys, time
import numpy as np
from ld_lidar import LidarReader

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM9"
WALL_H = 2.5      # assumed ceiling height (m)
MAX_R = 8.0       # ignore points beyond this (m)


def scan_to_ring(r):
    """Return floor points Nx3 (z=0), ordered by angle, plus intensity."""
    scan = r.get_scan()
    if len(scan) == 0:
        return None, None
    order = np.argsort(scan[:, 0])
    scan = scan[order]
    ang = np.deg2rad(scan[:, 0])
    d = scan[:, 1] / 1000.0
    m = (d > 0.02) & (d < MAX_R)
    ang, d, intens = ang[m], d[m], scan[m, 2]
    x = d * np.cos(ang)
    y = d * np.sin(ang)
    floor = np.column_stack([x, y, np.zeros_like(x)])
    return floor, intens


def build_geometry(floor):
    """Vertices + lines for floor ring, ceiling ring and vertical walls."""
    n = len(floor)
    ceil = floor.copy()
    ceil[:, 2] = WALL_H
    verts = np.vstack([floor, ceil])          # 0..n-1 floor, n..2n-1 ceiling
    lines = []
    for i in range(n):
        j = (i + 1) % n
        lines.append([i, j])                  # floor contour
        lines.append([n + i, n + j])          # ceiling contour
        lines.append([i, n + i])              # vertical wall edge
    return verts, np.array(lines, dtype=np.int32)


def run_open3d():
    import open3d as o3d
    r = LidarReader(PORT)
    time.sleep(0.4)
    floor, intens = scan_to_ring(r)
    while floor is None or len(floor) < 3:
        time.sleep(0.1); floor, intens = scan_to_ring(r)

    verts, lines = build_geometry(floor)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts)
    ls.lines = o3d.utility.Vector2iVector(lines)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(floor)

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"LDROBOT D500 - live 3D room ({PORT})", 1100, 800)
    vis.add_geometry(ls)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    opt.line_width = 1.5
    opt.point_size = 4.0

    # nice starting camera
    vc = vis.get_view_control()
    vc.set_front([0.4, -0.8, -0.4]); vc.set_up([0, 0, 1]); vc.set_zoom(0.7)

    try:
        while True:
            floor, intens = scan_to_ring(r)
            if floor is not None and len(floor) >= 3:
                verts, lines = build_geometry(floor)
                ls.points = o3d.utility.Vector3dVector(verts)
                ls.lines = o3d.utility.Vector2iVector(lines)
                # colour walls by height (floor green -> ceiling cyan)
                lc = []
                for a, b in lines:
                    z = (verts[a, 2] + verts[b, 2]) / 2 / WALL_H
                    lc.append([0.1 + 0.1 * z, 0.9 - 0.4 * z, 0.4 + 0.6 * z])
                ls.colors = o3d.utility.Vector3dVector(np.array(lc))
                pcd.points = o3d.utility.Vector3dVector(floor)
                pcd.paint_uniform_color([1.0, 0.85, 0.1])
                vis.update_geometry(ls)
                vis.update_geometry(pcd)
            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        r.close(); vis.destroy_window()


def run_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    r = LidarReader(PORT)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"LDROBOT D500 - live 3D room ({PORT})")
    ax = fig.add_subplot(111, projection="3d")

    def update(_):
        ax.clear()
        floor, intens = scan_to_ring(r)
        ax.set_xlim(-MAX_R/2, MAX_R/2); ax.set_ylim(-MAX_R/2, MAX_R/2); ax.set_zlim(0, WALL_H)
        ax.set_title("live 3D room  (drag to orbit)", color="w")
        if floor is None or len(floor) < 3:
            return
        x, y = floor[:, 0], floor[:, 1]
        xr = np.append(x, x[:1]); yr = np.append(y, y[:1])
        for h in (0, WALL_H):
            ax.plot(xr, yr, h, color="#2f8", lw=1)
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [y[i], y[i]], [0, WALL_H], color="#08a", lw=0.4, alpha=0.5)
        ax.scatter(x, y, 0, c=intens, cmap="turbo", s=6)
        ax.scatter(0, 0, 0, c="r", s=40)

    ani = FuncAnimation(fig, update, interval=120, cache_frame_data=False)
    try:
        plt.show()
    finally:
        r.close()


if __name__ == "__main__":
    try:
        import open3d  # noqa
        print("Using Open3D renderer. Drag to orbit, wheel to zoom.")
        run_open3d()
    except ImportError:
        print("Open3D not found -> using matplotlib 3D fallback.")
        run_matplotlib()
