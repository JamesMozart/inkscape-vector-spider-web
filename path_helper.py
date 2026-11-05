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
    """Sample a subpath into a polyline (Nx2 float64 array)."""
    if not subpath:
        return np.empty((0, 2), dtype=np.float64)

    path = Path(subpath)

    # Precompute parameter samples once (skip t=0; start point is added
    # explicitly via Move, so segments only need t in (0, 1]).
    ts = np.linspace(0.0, 1.0, samples_per_seg + 1)[1:]

    chunks = []
    subpath_start = None
    last_point = None

    for proxy in path.proxy_iterator():
        name = proxy.name

        if name == "Move":
            p = proxy.end_point
            pt = (p.x, p.y)
            chunks.append(np.asarray([pt], dtype=np.float64))
            subpath_start = pt
            last_point = pt
            continue

        if name == "ZoneClose":
            if subpath_start is not None and last_point is not None:
                sx, sy = subpath_start
                lx, ly = last_point
                if math.hypot(sx - lx, sy - ly) > 1e-9:
                    # Vectorized straight-line sampling
                    seg = np.column_stack((
                        lx + ts * (sx - lx),
                        ly + ts * (sy - ly),
                    ))
                    chunks.append(seg)
            last_point = subpath_start
            continue

        for i in range(1, samples_per_seg + 1):
            t = i / samples_per_seg
            p = proxy.point(t)
            chunks.append(np.column_stack((
                p.x,
                p.y,
            )))
            last_point = p

    if not chunks:
        return np.empty((0, 2), dtype=np.float64)

    return np.concatenate(chunks, axis=0)


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
        if len(poly) == 0:
            continue

        is_closed = (
            math.hypot(poly[0][0] - poly[-1][0],
                        poly[0][1] - poly[-1][1]) < 1e-6
        )
        if target_resample is not None and len(poly) >= 2:
            poly = resample_polyline_by_arclength_np(
                poly, target_resample, closed=is_closed
            )

        yield (poly, is_closed)
