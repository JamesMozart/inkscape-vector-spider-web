#!/usr/bin/env python3
"""Radial spider web generator for Inkscape.

Approach: radial spokes + concentric rings connected by sagging arcs,
with per-vertex jitter driven by Simplex noise in polar coordinates.
Pure Python + NumPy. No scipy, no matplotlib.

NEW: The web outer boundary is defined by a user-selected path.
Spokes are cast from the path center until they intersect the path;
rings are placed proportionally along each spoke direction.

NEW: Each cell (region between two adjacent spokes and two adjacent rings)
is also emitted as a closed polygon in a separate group, reusing the exact
Q-curve parameters of the stroked web (no resampling).
"""

import math
import inkex
from inkex import PathElement
import numpy as np

from simplex_noise import SimplexNoise
from path_helper import boundary_polylines


# ============================================================
# Noise helper
# ============================================================
def polar_noise_batch(noise, xs, ys, cx, cy, scale, noise_type):
    """Vectorised polar noise: same coordinate transform as the scalar
    version, but accepts arrays and returns an array of the same shape."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dx = xs - cx
    dy = ys - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    nx = r * scale
    ny = theta * scale * 2.0

    if noise_type == "fbm":
        return noise.fbm_batch(nx, ny)
    if noise_type == "turbulence":
        return noise.turbulence_batch(nx, ny)
    return noise.noise_batch(nx, ny)


def sag_bezier_full_batch(p1_arr, p2_arr, sag, sag_noise_arr, rng, sag_jitter):
    """Vectorised sagging Q-curve builder.

    p1_arr, p2_arr: (n, 2) float arrays
    sag_noise_arr:  (n,)   float array (per-edge noise sample)
    Returns:
        ds:       list[str]   — one path 'd' string per edge
        segs:     list[tuple] — one (p1, ctrl, p2) per edge
    """
    p1_arr = np.asarray(p1_arr, dtype=float)
    p2_arr = np.asarray(p2_arr, dtype=float)
    sag_noise_arr = np.asarray(sag_noise_arr, dtype=float)
    n = p1_arr.shape[0]

    # per-edge random factor, drawn in one shot
    rand = rng.random(n) * 2.0 - 1.0  # in [-1, 1)

    mx = (p1_arr[:, 0] + p2_arr[:, 0]) / 2.0
    my = (p1_arr[:, 1] + p2_arr[:, 1]) / 2.0
    dx = p2_arr[:, 0] - p1_arr[:, 0]
    dy = p2_arr[:, 1] - p1_arr[:, 1]
    length = np.hypot(dx, dy)

    # default: straight line (sag<=0 or degenerate length)
    straight_mask = (sag <= 0) | (length < 1e-9)

    # control point = midpoint for straight cases
    ctrl_x = mx.copy()
    ctrl_y = my.copy()

    # curved cases only
    curved = ~straight_mask
    if np.any(curved):
        L = length[curved]
        nx = -dy[curved] / L
        ny = dx[curved] / L

        local_sag = sag * (1.0 + sag_jitter * sag_noise_arr[curved])
        local_sag = local_sag * (1.0 + 0.05 * rand[curved])

        # flip normal if midpoint . normal < 0
        dot = mx[curved] * nx + my[curved] * ny
        flip = dot < 0
        nx = np.where(flip, -nx, nx)
        ny = np.where(flip, -ny, ny)

        ctrl_x[curved] = mx[curved] + nx * local_sag * L
        ctrl_y[curved] = my[curved] + ny * local_sag * L

    # build strings
    ds = []
    segs = []
    for i in range(n):
        x1, y1 = p1_arr[i]
        x2, y2 = p2_arr[i]
        if straight_mask[i]:
            ds.append(f"M {x1:.4f},{y1:.4f} L {x2:.4f},{y2:.4f}")
            segs.append(((x1, y1), (mx[i], my[i]), (x2, y2)))
        else:
            cx_i = ctrl_x[i]
            cy_i = ctrl_y[i]
            ds.append(
                f"M {x1:.4f},{y1:.4f} Q {cx_i:.4f},{cy_i:.4f} "
                f"{x2:.4f},{y2:.4f}"
            )
            segs.append(((x1, y1), (cx_i, cy_i), (x2, y2)))
    return ds, segs


def ray_intersect_boundary(cx, cy, angle, polylines):
    """Cast a ray from (cx, cy) at `angle` and return the farthest
    intersection distance with the boundary polylines."""
    dx = math.cos(angle)
    dy = math.sin(angle)
    best_t = None

    for poly in polylines:
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            ex = x2 - x1
            ey = y2 - y1

            denom = dx * ey - dy * ex
            if abs(denom) < 1e-12:
                continue

            t = ((x1 - cx) * ey - (y1 - cy) * ex) / denom
            s = ((x1 - cx) * dy - (y1 - cy) * dx) / denom

            if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
                if best_t is None or t > best_t:
                    best_t = t

    return best_t


def boundary_bbox(polylines):
    xs = [pt[0] for poly in polylines for pt in poly]
    ys = [pt[1] for poly in polylines for pt in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def centroid(polylines):
    xs = [pt[0] for poly in polylines for pt in poly]
    ys = [pt[1] for poly in polylines for pt in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ============================================================
# Web generation
# ============================================================
def generate_web(center, polylines, n_rays, n_rings, min_radius,
                 ring_growth, angle_jitter, radius_jitter,
                 sag, sag_jitter, spoke_curve, seed,
                 noise_type, noise_scale, spoke_phase=0.0):
    """Build web vertices."""
    rng = np.random.default_rng(seed)
    noise = SimplexNoise(seed=seed)
    cx, cy = center

    if n_rings <= 1:
        ring_fracs = np.array([1.0])
    else:
        raw = np.array([ring_growth ** i for i in range(n_rings)])
        raw = (raw - raw[0]) / (raw[-1] - raw[0])
        ring_fracs = raw
    ring_fracs = np.asarray(ring_fracs, dtype=float)

    all_pts = []
    for poly in polylines:
        for pt in poly:
            if not all_pts or (abs(pt[0] - all_pts[-1][0]) > 1e-9
                               or abs(pt[1] - all_pts[-1][1]) > 1e-9):
                all_pts.append((pt[0], pt[1]))
    # If the boundary is closed, drop the duplicated first point at the end
    if len(all_pts) > 1:
        fx, fy = all_pts[0]
        lx, ly = all_pts[-1]
        if abs(fx - lx) < 1e-9 and abs(fy - ly) < 1e-9:
            all_pts.pop()

    n_pts = len(all_pts)
    if n_pts == 0:
        # Extreme fallback: degenerate to center point, all rays use min_radius
        hit_pts = [(cx + min_radius, cy)] * n_rays
    else:
        # Phase offset: shift the starting index by the number of points
        # corresponding to spoke_phase
        phase_frac = (spoke_phase % (2.0 * math.pi)) / (2.0 * math.pi)
        start_idx = int(round(phase_frac * n_pts))
        idxs = np.linspace(0, n_pts, n_rays, endpoint=False).astype(int)
        hit_pts = [all_pts[(i + start_idx) % n_pts] for i in idxs]

    # 3) Derive angle and radius from center to hit point
    base_angles = np.array([
        math.atan2(py - cy, px - cx) for (px, py) in hit_pts
    ])
    ray_radius = np.array([
        max(math.hypot(px - cx, py - cy), min_radius)
        for (px, py) in hit_pts
    ])

    max_jitter_angle = angle_jitter * (2.0 * math.pi / n_rays) * 0.5
    max_jitter_radius_frac = radius_jitter * 0.5

    n_ring_pts = len(ring_fracs)

    # --- Batch: rotation noise for all rings (depends on frac and ray_radius.mean()) ---
    ring_rot_xs = cx + ring_fracs * ray_radius.mean()
    ring_rot_ys = np.full(n_ring_pts, cy, dtype=float)
    ring_rot_ns = polar_noise_batch(
        noise, ring_rot_xs, ring_rot_ys, cx, cy, noise_scale, noise_type
    )

    verts = []
    for ri, frac in enumerate(ring_fracs):
        ring_rot = float(ring_rot_ns[ri]) * max_jitter_angle * 0.5

        is_outer = (ri == len(ring_fracs) - 1)

        if is_outer:
            ring_pts = []
            for ai in range(n_rays):
                a = float(base_angles[ai])
                rr = float(ray_radius[ai])
                x = cx + rr * math.cos(a)
                y = cy + rr * math.sin(a)
                ring_pts.append((x, y))
            verts.append(ring_pts)
            continue

        # ---- Batch-prepare sample coordinates for all vertices of this ring ----
        a_arr = base_angles + ring_rot
        rr_arr = min_radius + (ray_radius - min_radius) * frac
        px_arr = cx + rr_arr * np.cos(a_arr)
        py_arr = cy + rr_arr * np.sin(a_arr)

        tx_arr = -np.sin(a_arr)
        ty_arr = np.cos(a_arr)
        px2_arr = px_arr + tx_arr * rr_arr * 0.5
        py2_arr = py_arr + ty_arr * rr_arr * 0.5

        # Two batches of noise, one batch call each
        n_ang_arr = polar_noise_batch(
            noise, px_arr, py_arr, cx, cy, noise_scale, noise_type
        )
        n_rad_arr = polar_noise_batch(
            noise, px2_arr, py2_arr, cx, cy, noise_scale, noise_type
        )

        # ---- Apply per vertex (pure numeric, no noise calls) ----
        ring_pts = []
        for ai in range(n_rays):
            a = float(a_arr[ai]) + float(n_ang_arr[ai]) * max_jitter_angle
            rr_j = float(rr_arr[ai]) * (
                1.0 + float(n_rad_arr[ai]) * max_jitter_radius_frac
            )
            x = cx + rr_j * math.cos(a)
            y = cy + rr_j * math.sin(a)
            ring_pts.append((x, y))
        verts.append(ring_pts)

    return verts, base_angles, ray_radius, noise


# ============================================================
# Inkscape extension
# ============================================================
class RadialSpiderWeb(inkex.EffectExtension):

    def add_arguments(self, pars):
        pars.add_argument("--n_rays", type=int, default=14)
        pars.add_argument("--n_rings", type=int, default=7)
        pars.add_argument("--min_radius", type=float, default=12.0)
        pars.add_argument("--ring_growth", type=float, default=1.35)
        pars.add_argument("--angle_jitter", type=float, default=0.18)
        pars.add_argument("--radius_jitter", type=float, default=0.10)
        pars.add_argument("--sag", type=float, default=0.12)
        pars.add_argument("--sag_jitter", type=float, default=0.5)
        pars.add_argument("--spoke_curve", type=float, default=0.03)
        pars.add_argument("--spoke_phase", type=float, default=0.0)
        pars.add_argument("--seed", type=int, default=3)
        pars.add_argument(
            "--noise_type", type=str, default="simplex",
            choices=["simplex", "fbm", "turbulence"],
        )
        pars.add_argument("--noise_scale", type=float, default=0.05)
        pars.add_argument("--web_error_prob", type=float, default=0)
        pars.add_argument("--stroke_spoke", type=str, default="#444444")
        pars.add_argument("--stroke_ring", type=str, default="#888888")
        pars.add_argument("--spoke_width", type=float, default=1)
        pars.add_argument("--ring_width", type=float, default=1)
        # Cell polygon options
        pars.add_argument("--cell_fill", type=str, default="#cccccc")
        pars.add_argument("--cell_fill_opacity", type=float, default=1.0)
        pars.add_argument("--cell_stroke", type=str, default="#444444")
        pars.add_argument("--cell_stroke_width", type=float, default=1.0)
        pars.add_argument("--center_offset_x", type=float, default=0.0)
        pars.add_argument("--center_offset_y", type=float, default=0.0)
        pars.add_argument(
            "--generate_mode", type=str, default="both",
            choices=["strokes", "cells", "both"],
        )

    def _get_selected_path(self):
        for el in self.svg.selection.values():
            if isinstance(el, PathElement) or hasattr(el, "path"):
                try:
                    path = el.path
                except Exception:
                    continue
                # Bake element + ancestor transforms into path coordinates
                path = path.transform(el.composed_transform())
                return path
        return None

    def effect(self):
        o = self.options

        path = self._get_selected_path()
        if path is None:
            inkex.errormsg(
                "Please select a path to use as the web boundary, "
                "or convert shape to path first."
            )
            return

        polylines = boundary_polylines(
            path.to_absolute(),
            samples_per_seg=24,
            target_resample=2048,
        )
        if not polylines:
            inkex.errormsg("Invalid path.")
            return

        x0, y0, x1, y1 = boundary_bbox(polylines)
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) / 2 + (o.center_offset_x / 100.0) * w
        cy = (y0 + y1) / 2 + (o.center_offset_y / 100.0) * h
        # cx, cy = centroid(polylines)

        rng = np.random.default_rng(o.seed)

        verts, base_angles, ray_radius, noise = generate_web(
            center=(cx, cy),
            polylines=polylines,
            n_rays=o.n_rays,
            n_rings=o.n_rings,
            min_radius=o.min_radius,
            ring_growth=o.ring_growth,
            angle_jitter=o.angle_jitter,
            radius_jitter=o.radius_jitter,
            sag=o.sag,
            sag_jitter=o.sag_jitter,
            spoke_curve=o.spoke_curve,
            seed=o.seed,
            noise_type=o.noise_type,
            noise_scale=o.noise_scale,
            spoke_phase=math.radians(o.spoke_phase),
        )

        layer = self.svg.get_current_layer()

        mode = o.generate_mode  # "strokes" | "cells" | "both"
        want_strokes = mode in ("strokes", "both")
        want_cells = mode in ("cells", "both")

        # Cell polygon group (created only if requested)
        cell_group = None
        if want_cells:
            cell_group = inkex.Group()
            cell_group.set("id", f"spiderweb_cells_{o.seed}")
            cell_group.set("inkscape:label", "Spider Web Cells")
            layer.append(cell_group)

        # Stroke group (created only if requested)
        group = None
        if want_strokes:
            group = inkex.Group()
            group.set("id", f"spiderweb_radial_{o.seed}")
            group.set("inkscape:label", "Radial Spider Web")
            layer.append(group)

        n_rings = len(verts)

        # spoke_seg[ai][ri] = (p0, ctrl, p1), from ring ri-1 to ring ri
        spoke_seg = [[None] * n_rings for _ in range(o.n_rays)]

        # --- Spokes: batch bend + midpoint, emit only if want_strokes ---
        n_seg_per_spoke = n_rings - 1
        if n_seg_per_spoke > 0:
            # endpoints per (ai, ri): prev = verts[ri-1][ai], p = verts[ri][ai]
            # Order must match the old version: outer ai, inner ri
            prev_list = []
            p_list = []
            ai_list = []
            ri_list = []
            for ai in range(o.n_rays):
                for ri in range(1, n_rings):
                    prev_list.append(verts[ri - 1][ai])
                    p_list.append(verts[ri][ai])
                    ai_list.append(ai)
                    ri_list.append(ri)

            prev_arr = np.array(prev_list, dtype=float)
            p_arr = np.array(p_list, dtype=float)
            n_seg = prev_arr.shape[0]

            bend_arr = o.spoke_curve * (rng.random() * 2.0 - 1.0)

            mid_x = (prev_arr[:, 0] + p_arr[:, 0]) / 2.0 \
                - (p_arr[:, 1] - prev_arr[:, 1]) * bend_arr
            mid_y = (prev_arr[:, 1] + p_arr[:, 1]) / 2.0 \
                + (p_arr[:, 0] - prev_arr[:, 0]) * bend_arr

            # Fill spoke_seg
            for i in range(n_seg):
                ai = ai_list[i]
                ri = ri_list[i]
                spoke_seg[ai][ri] = (
                    tuple(prev_arr[i]), (mid_x[i], mid_y[i]), tuple(p_arr[i])
                )

            if want_strokes:
                # One path per spoke: M first + several Q
                for ai in range(o.n_rays):
                    d_parts = []
                    first = verts[0][ai]
                    d_parts.append(f"M {first[0]:.4f},{first[1]:.4f}")
                    for ri in range(1, n_rings):
                        i = ai * n_seg_per_spoke + (ri - 1)
                        d_parts.append(
                            f"Q {mid_x[i]:.4f},{mid_y[i]:.4f} "
                            f"{p_arr[i, 0]:.4f},{p_arr[i, 1]:.4f}"
                        )
                    path = PathElement()
                    path.set("d", " ".join(d_parts))
                    path.style = {
                        "stroke": o.stroke_spoke,
                        "stroke-width": f"{o.spoke_width}",
                        "stroke-linecap": "round",
                        "fill": "none",
                    }
                    group.append(path)

        # ring_seg[ri][ai] = (p1, ctrl, p2), direction ai -> ai+1
        ring_seg = [[None] * o.n_rays for _ in range(n_rings)]

        # --- Rings: pass 1 — fix geometry & collect midpoints ---
        # web_error_prob rewrites endpoints, so geometry must be fixed
        # before the batch noise.
        ring_edge_info = []  # (ri, ai, p1, p2, mid_x, mid_y)
        for ri in range(n_rings):
            ring_pts = verts[ri]
            for ai in range(o.n_rays):
                p1 = ring_pts[ai]
                p2 = ring_pts[(ai + 1) % o.n_rays]

                if rng.random() < o.web_error_prob and n_rings > 1:
                    candidates = []
                    if ri > 0:
                        candidates.append(ri - 1)
                    if ri < n_rings - 1:
                        candidates.append(ri + 1)
                    if candidates:
                        target_ri = candidates[rng.integers(len(candidates))]
                        if rng.random() < 0.5:
                            p1 = verts[target_ri][ai]
                        else:
                            p2 = verts[target_ri][(ai + 1) % o.n_rays]

                mid_x = (p1[0] + p2[0]) / 2.0
                mid_y = (p1[1] + p2[1]) / 2.0
                ring_edge_info.append(
                    (ri, ai, p1, p2, mid_x, mid_y)
                )

        # --- Rings: pass 2 — batch sag noise ---
        if ring_edge_info:
            mid_arr = np.array(
                [(e[4], e[5]) for e in ring_edge_info], dtype=float
            )
            sag_ns = polar_noise_batch(
                noise, mid_arr[:, 0], mid_arr[:, 1],
                cx, cy, o.noise_scale, o.noise_type,
            )
        else:
            sag_ns = np.array([], dtype=float)

        # --- Rings: pass 3 — batch build Q-curves and emit paths ---
        if ring_edge_info:
            p1_arr = np.array([e[2] for e in ring_edge_info], dtype=float)
            p2_arr = np.array([e[3] for e in ring_edge_info], dtype=float)
            ds, segs = sag_bezier_full_batch(
                p1_arr, p2_arr, o.sag, sag_ns, rng, o.sag_jitter
            )
            for idx, (ri, ai, _p1, _p2, _mx, _my) in enumerate(ring_edge_info):
                ring_seg[ri][ai] = segs[idx]
                if want_strokes:
                    path = PathElement()
                    path.set("d", ds[idx])
                    path.style = {
                        "stroke": o.stroke_ring,
                        "stroke-width": f"{o.ring_width}",
                        "stroke-linecap": "round",
                        "fill": "none",
                    }
                    group.append(path)

        # --- Cell polygons: emit only if want_cells ---
        if want_cells:
            def _fmt(pt):
                return f"{pt[0]:.4f},{pt[1]:.4f}"

            for ri in range(n_rings - 1):
                for ai in range(o.n_rays):
                    ai_next = (ai + 1) % o.n_rays

                    in_p1, in_c, in_p2 = ring_seg[ri][ai]
                    r_p0, r_c, r_p1 = spoke_seg[ai_next][ri + 1]
                    out_p1, out_c, out_p2 = ring_seg[ri + 1][ai]
                    l_p0, l_c, l_p1 = spoke_seg[ai][ri + 1]

                    c_in_left = in_p1
                    c_in_right = in_p2
                    c_out_left = out_p1
                    c_out_right = out_p2

                    parts = []
                    parts.append(f"M {_fmt(c_in_left)}")
                    parts.append(f"Q {_fmt(in_c)} {_fmt(c_in_right)}")
                    parts.append(f"Q {_fmt(r_c)} {_fmt(c_out_right)}")
                    parts.append(f"Q {_fmt(out_c)} {_fmt(c_out_left)}")
                    parts.append(f"Q {_fmt(l_c)} {_fmt(c_in_left)}")
                    parts.append("Z")

                    cell = PathElement()
                    cell.set("d", " ".join(parts))
                    cell.style = {
                        "fill": o.cell_fill,
                        "fill-opacity": f"{o.cell_fill_opacity}",
                        "stroke": o.cell_stroke,
                        "stroke-width": f"{o.cell_stroke_width}",
                    }
                    cell_group.append(cell)

        # Make the generated elements the active selection
        if want_strokes and want_cells:
            self.svg.selection.set(group)
            self.svg.selection.add(cell_group)
        elif want_strokes:
            self.svg.selection.set(group)
        elif want_cells:
            self.svg.selection.set(cell_group)


if __name__ == "__main__":
    RadialSpiderWeb().run()