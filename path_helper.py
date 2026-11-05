from inkex.paths import Move, Path
import inkex
import numpy as np
import math


def path_polyline(path, samples_per_seg=24):
    """Split an inkex Path into subpaths (lists of segments), then sample subpaths into polylines (list of (x, y))."""
    current = []
    for seg in path:
        if isinstance(seg, Move) and current:
            yield subpath_polyline(current, samples_per_seg)
            current = [seg]
        else:
            current.append(seg)
    if current:
        yield subpath_polyline(current, samples_per_seg)


def subpath_polyline(subpath, samples_per_seg=24):
    """Sample a subpath into a polyline (list of (x, y))."""
    if not subpath:
        return []

    path = Path(subpath)
    poly = []
    subpath_start = None
    last_point = None

    for proxy in path.proxy_iterator():
        name = proxy.name

        if name == "Move":
            p = proxy.end_point
            poly.append((p.x, p.y))
            subpath_start = (p.x, p.y)
            last_point = (p.x, p.y)
            continue

        if name == "ZoneClose":
            if subpath_start is not None and last_point is not None:
                sx, sy = subpath_start
                lx, ly = last_point
                if math.hypot(sx - lx, sy - ly) > 1e-9:
                    for i in range(1, samples_per_seg + 1):
                        t = i / samples_per_seg
                        poly.append((lx + t * (sx - lx), ly + t * (sy - ly)))
            last_point = subpath_start
            continue

        for i in range(1, samples_per_seg + 1):
            t = i / samples_per_seg
            p = proxy.point(t)
            poly.append((p.x, p.y))
        last_point = (p.x, p.y)

    return poly


def resample_polyline_by_arclength_np(poly, n, closed):
    """Vectorized resampling of a polyline to n points evenly spaced by arc length."""
    if len(poly) < 2:
        return np.zeros((n, 2), dtype=np.float64)

    pts = np.asarray(poly, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]

    if total <= 0.0:
        return np.tile(pts[0], (n, 1))

    if closed:
        t_vals = (np.arange(n, dtype=np.float64) / n) * total
    else:
        if n > 1:
            t_vals = (np.arange(n, dtype=np.float64) / (n - 1)) * total
        else:
            t_vals = np.zeros(1, dtype=np.float64)

    x_out = np.interp(t_vals, cum, pts[:, 0])
    y_out = np.interp(t_vals, cum, pts[:, 1])
    return np.stack([x_out, y_out], axis=1)


def boundary_polylines(path, samples_per_seg=64, target_resample=None):
    """Sample an inkex.Path into a list of polylines (each list of (x, y))."""
    for poly in path_polyline(path, samples_per_seg=samples_per_seg):
        if not poly:
            continue

        is_closed = (
            math.hypot(poly[0][0] - poly[-1][0],
                        poly[0][1] - poly[-1][1]) < 1e-6
        )
        if target_resample is not None and len(poly) >= 2:
            arr = resample_polyline_by_arclength_np(
                poly, target_resample, closed=is_closed
            )
            poly = [(float(x), float(y)) for x, y in arr]

        yield (poly, is_closed)
