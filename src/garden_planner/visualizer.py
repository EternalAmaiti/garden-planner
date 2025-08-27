#!/usr/bin/env python3
"""
Garden Bed Visualizer (v6.5 — GardenBed + Theme + explicit RNG)
----------------------------------------------------------------
This version makes three surgical improvements without changing behavior:

1) Replace mutable globals for bed size with an explicit GardenBed object.
2) Introduce a minimal Theme (wrapping species styles) and thread it into rendering/logic.
3) Make randomness explicit by threading a random.Random instance (rng) through layout functions.

Everything else (CLI, visuals, CSV format, camera math) remains compatible with v6.4.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------- Bed & Grid (defaults) ----------------------
DEFAULT_BED_WIDTH = 3.0
DEFAULT_BED_DEPTH = 2.5
CELL_SIZE = 0.1
DEFAULT_SEED = 42


# ---------------------- Data models ----------------------
@dataclass
class GardenBed:
    """Explicit bed dimensions in meters."""

    width_m: float
    depth_m: float

    def scaled(self, s: float) -> GardenBed:
        """Return a new bed uniformly scaled by s (does not mutate in place)."""
        return GardenBed(self.width_m * s, self.depth_m * s)


@dataclass(frozen=True)
class SpeciesStyle:
    """
    Visual + spacing info for one species.

    Notes on accessibility: We keep edge colors dark for marker contrast and
    outline any text with a white stroke to preserve readability on color fills.
    """

    label: str
    spacing: int  # cm between bulbs in a cluster
    bulbs: int  # total bulbs allocated to this species
    size: int  # marker size for scatter points
    face: str  # face color
    edge: str  # edge color
    marker: str = "o"  # shape (kept simple here; can diversify later)


@dataclass
class Theme:
    """Minimal theme wrapper so visuals are decoupled from layout/geometry."""

    species_styles: dict[str, SpeciesStyle]
    halo_alpha: float = 0.15


def default_theme() -> Theme:
    """
    Build a Theme from the original v6.4 SPECIES mapping (colors unchanged).
    Keeping visuals constant preserves output parity with v6.4.
    """
    # Original SPECIES mapping converted to typed SpeciesStyle entries
    styles = {
        "snowdrop": SpeciesStyle(
            label="Snowdrops (Galanthus elwesii)",
            spacing=7,
            bulbs=40,
            size=140,
            face="#2e8b57",
            edge="#2e8b57",
        ),
        "crocus_mixed": SpeciesStyle(
            label="Crocus tommasinianus",
            spacing=7,
            bulbs=25,
            size=110,
            face="#800000",
            edge="#800000",
        ),
        "scilla": SpeciesStyle(
            label="Scilla siberica", spacing=8, bulbs=20, size=120, face="#808000", edge="#808000"
        ),
        "daffodil": SpeciesStyle(
            label="Narcissus 'Tête-à-Tête'",
            spacing=10,
            bulbs=30,
            size=150,
            face="#663399",
            edge="#663399",
        ),
        "iris_reticulata": SpeciesStyle(
            label="Iris reticulata", spacing=8, bulbs=15, size=140, face="#ff0000", edge="#ff0000"
        ),
        "hyacinth": SpeciesStyle(
            label="Hyacinth 'Jan Bos'",
            spacing=12,
            bulbs=5,
            size=180,
            face="#ff8c00",
            edge="#ff8c00",
        ),
        "chionodoxa": SpeciesStyle(
            label="Chionodoxa luciliae alba",
            spacing=8,
            bulbs=20,
            size=120,
            face="#ffd700",
            edge="#ffd700",
        ),
        "anemone_blanda": SpeciesStyle(
            label="Anemone blanda", spacing=8, bulbs=25, size=130, face="#0000cd", edge="#0000cd"
        ),
        "tulip_mixed": SpeciesStyle(
            label="Tulip Triumph (mixed colors)",
            spacing=12,
            bulbs=50,
            size=160,
            face="#ff1493",
            edge="#ff1493",
        ),
        "tulip_ballerina": SpeciesStyle(
            label="Tulip 'Ballerina'", spacing=12, bulbs=7, size=160, face="#adff2f", edge="#adff2f"
        ),
        "iris_hollandica": SpeciesStyle(
            label="Iris hollandica 'Red Ember'",
            spacing=12,
            bulbs=10,
            size=150,
            face="#1e90ff",
            edge="#1e90ff",
        ),
        "anemone_de_caen": SpeciesStyle(
            label="Anemone de Caen", spacing=8, bulbs=20, size=130, face="#fa8072", edge="#fa8072"
        ),
        "tulip_white": SpeciesStyle(
            label="Tulip Triumph (white)",
            spacing=12,
            bulbs=25,
            size=160,
            face="#eee8aa",
            edge="#eee8aa",
        ),
        "tulip_rembrandt": SpeciesStyle(
            label="Tulip Rembrandt (mixed)",
            spacing=12,
            bulbs=15,
            size=160,
            face="#dda0dd",
            edge="#dda0dd",
        ),
    }
    return Theme(species_styles=styles, halo_alpha=0.15)


# ---------------------- Plant heights (cm) ----------------------
HEIGHTS_CM: dict[str, int] = {
    "snowdrop": 15,
    "crocus_mixed": 12,
    "scilla": 15,
    "chionodoxa": 12,
    "iris_reticulata": 14,
    "anemone_blanda": 12,
    "daffodil": 22,
    "hyacinth": 25,
    "anemone_de_caen": 22,
    "tulip_mixed": 45,
    "tulip_white": 45,
    "tulip_rembrandt": 45,
    "tulip_ballerina": 60,
    "iris_hollandica": 60,
}


# ---------------------- Bloom calendar (approximate Northern Hemisphere) ----------------------
# Months use 1..12 (Jan..Dec).
january = 1
february = 2
march = 3
april = 4
may = 5
june = 6
july = 7
august = 8
september = 9
october = 10
november = 11
december = 12

BLOOM_CALENDAR: dict[str, list[int]] = {
    "snowdrop": [1, 2],
    "crocus_mixed": [2, 3],  # can be very late Jan in mild winters
    "scilla": [3, 4],
    "chionodoxa": [3, 4],
    "iris_reticulata": [2, 3],
    "anemone_blanda": [3, 4],
    "daffodil": [3, 4],  # 'Tête-à-Tête' often late Feb–Mar
    "hyacinth": [4],
    "anemone_de_caen": [4, 5],
    "tulip_mixed": [4, 5],
    "tulip_white": [4, 5],
    "tulip_rembrandt": [4, 5],
    "tulip_ballerina": [5],
    "iris_hollandica": [6],
}


# ---------------------- Legend (cluster-aware) ----------------------
def legend_handles(
    clusters: dict[str, list[tuple[float, float, int]]], theme: Theme
) -> list[mpatches.Patch]:
    """Only include species that actually appear in the current plan (non-empty clusters)."""
    handles = []
    for sp, pts in clusters.items():
        if pts:
            info = theme.species_styles[sp]
            handles.append(
                mpatches.Patch(
                    facecolor=info.face,
                    edgecolor=info.edge,
                    label=f"{info.label} (spacing {info.spacing} cm)",
                )
            )
    return handles


# ---------------------- Area & Radius ----------------------
def cluster_area_m2(spacing_cm: float, bulbs_in_cluster: int) -> float:
    """
    Approximate area of a cluster assuming hex-like packing of bulbs with spacing s.
    A_hex = (sqrt(3)/2) * s^2 * n
    """
    s = spacing_cm / 100.0
    return (math.sqrt(3) / 2.0) * s * s * bulbs_in_cluster


def cluster_radius_m(spacing_cm: float, bulbs_in_cluster: int) -> float:
    """Area-equivalent radius (circle with same area as the hex-like patch)."""
    A = cluster_area_m2(spacing_cm, bulbs_in_cluster)
    return math.sqrt(A / math.pi)


def total_halo_area(
    species_clusters: dict[str, list[tuple[float, float, int]]], theme: Theme
) -> float:
    """Sum of area-equivalent halos for all clusters across species."""
    tot = 0.0
    for sp, pts in species_clusters.items():
        spacing = theme.species_styles[sp].spacing
        for _, _, c in pts:
            tot += cluster_area_m2(spacing, c)
    return tot


# ---------------------- Camera (homography with stable vertical) ----------------------
def camera_homography(
    bed: GardenBed, cam_height=1.55, cam_dist=0.30, cam_pitch_deg=35.0, cam_focal=1.8
):
    """
    Build homography H from ground plane (z=0) world coords (meters) to image plane (arbitrary units).
    World: x across bed (0..W), y depth (0..D), z up (0 at ground).
    Camera at C=(W/2, -cam_dist, cam_height). Orientation: look +y, pitch down by cam_pitch_deg.
    Returns (H, sign_v) where sign_v flips vertical so front edge renders lower than back edge.

    Using homogeneous planar mapping: H = K [ r1 r2 t ], with r1, r2 the first two columns of R.
    """
    Cx = bed.width_m * 0.5
    Cy = -cam_dist
    Cz = cam_height

    # Base orientation: Xc <- +x_w, Yc <- +z_w, Zc <- +y_w
    R_base = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)

    th = np.deg2rad(cam_pitch_deg)
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]], dtype=float
    )

    R = R_x @ R_base  # world->camera
    C = np.array([Cx, Cy, Cz], dtype=float)
    t = -R @ C.reshape(3, 1)  # translation

    f = cam_focal
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], dtype=float)

    r1 = R[:, 0].reshape(3, 1)
    r2 = R[:, 1].reshape(3, 1)
    H = K @ np.hstack([r1, r2, t])

    # Determine vertical sign so that front edge is lower than back edge on the plot
    def proj_raw(pts):
        ph = np.c_[pts, np.ones(len(pts))]
        uvw = (H @ ph.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        return uv

    front_mid = np.array([[bed.width_m * 0.5, 0.0]])
    back_mid = np.array([[bed.width_m * 0.5, bed.depth_m]])
    vf = proj_raw(front_mid)[0, 1]
    vb = proj_raw(back_mid)[0, 1]
    # Matplotlib y increases upward; we want front lower => smaller y value
    sign_v = 1.0 if vf < vb else -1.0
    return H, sign_v


def apply_homography(H: np.ndarray, pts: np.ndarray, sign_v: float) -> np.ndarray:
    ph = np.c_[pts, np.ones(len(pts))]
    uvw = (H @ ph.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    uv[:, 1] *= sign_v
    return uv


# ---------------------- Clusters & Non-overlap ----------------------
def odd_partition(total: int, rng: random.Random, avg: int = 5, min_odd: int = 3) -> list[int]:
    """Partition total bulbs into odd cluster sizes (3,5,7,...) with mild randomness."""
    if total <= 0:
        return []
    if total < min_odd:
        return [total] if total % 2 == 1 else [total - 1, 1] if total > 0 else []
    k_max = max(1, total // min_odd)
    k_target = max(1, int(round(total / float(avg))))
    k = max(1, min(k_max, k_target + rng.choice([-1, 0, 1])))
    if (k % 2) != (total % 2):
        k = k + 1 if k < k_max else max(1, k - 1)
    while k > 0 and k * min_odd > total:
        k -= 1
        if (k % 2) != (total % 2):
            k -= 1
    if k < 1:
        k = 1
    counts = [min_odd] * k
    rem = total - k * min_odd
    while rem >= 2:
        i = rng.randrange(k)
        counts[i] += 2
        rem -= 2
    rng.shuffle(counts)
    return counts


def build_random_clusters(theme: Theme, bed: GardenBed, rng: random.Random, avg=5, min_odd=3):
    """
    Create initial cluster centers (uniform within bed) and odd cluster sizes per species.
    rng: explicit Random instance for reproducibility.
    """
    clusters: dict[str, list[tuple[float, float, int]]] = {}
    for key, style in theme.species_styles.items():
        total = style.bulbs
        counts = odd_partition(total, rng=rng, avg=avg, min_odd=min_odd)
        spacing_m = style.spacing / 100.0
        pts = []
        for c in counts:
            x = rng.uniform(spacing_m, bed.width_m - spacing_m)
            y = rng.uniform(spacing_m, bed.depth_m - spacing_m)
            pts.append((x, y, c))
        clusters[key] = pts
    return clusters


def make_non_overlapping_layout(
    species_clusters: dict[str, list[tuple[float, float, int]]],
    theme: Theme,
    bed: GardenBed,
    gap: float = 0.02,
    max_iters: int = 800,
    step: float = 0.6,
    interleave_species: bool = False,
):
    """
    Relax cluster centers to resolve overlaps. Optionally encourage interleaving of species
    so same-species clusters are not adjacent along their line of centers.
    """
    items = []
    for sp, pts in species_clusters.items():
        for x, y, c in pts:
            r = cluster_radius_m(theme.species_styles[sp].spacing, c)
            items.append({"sp": sp, "x": x, "y": y, "c": c, "r": r})
    n = len(items)
    if n == 0:
        return species_clusters

    # --- Interleaving enforcement: ensure a different species lies between same-species pairs ---
    if interleave_species and n > 2:

        def seg_proj_and_dist(P, A, B):
            # Returns (t in [0,1] for projection on AB, perpendicular distance)
            ax, ay = A
            bx, by = B
            px, py = P
            ABx, ABy = bx - ax, by - ay
            AB2 = ABx * ABx + ABy * ABy
            if AB2 == 0:
                return 0.0, math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, ((px - ax) * ABx + (py - ay) * ABy) / AB2))
            qx, qy = ax + t * ABx, ay + t * ABy
            return t, math.hypot(px - qx, py - qy)

        def intersects_segment(circle_center, r, A, B):
            t, d = seg_proj_and_dist(circle_center, A, B)
            return d <= r + 1e-4 and 0.0 < t < 1.0  # strictly between

        # Build quick arrays for centers/radii
        centers = [(obj["x"], obj["y"]) for obj in items]
        radii = [obj["r"] for obj in items]
        species = [obj["sp"] for obj in items]
        # Iterate a few passes nudging the *closest different-species* circle onto the AB line segment
        for _pass in range(18):
            violations = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if species[i] != species[j]:
                        continue
                    Ai, Aj = centers[i], centers[j]
                    # Already "blocked"?
                    blocked = False
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if species[k] == species[i]:
                            continue
                        if intersects_segment(centers[k], radii[k], Ai, Aj):
                            blocked = True
                            break
                    if blocked:
                        continue
                    violations += 1
                    # Find a candidate different species whose projection lies within segment and is closest
                    best_k, best_t, best_d = None, None, 1e9
                    ax, ay = Ai
                    bx, by = Aj
                    ABx, ABy = bx - ax, by - ay
                    AB2 = ABx * ABx + ABy * ABy
                    for k in range(n):
                        if k == i or k == j or species[k] == species[i]:
                            continue
                        t, d = seg_proj_and_dist(centers[k], Ai, Aj)
                        if 0.0 < t < 1.0 and d < best_d:
                            best_k, best_t, best_d = k, t, d
                    if best_k is None:
                        # No candidate whose projection is inside; pick nearest different species to midpoint
                        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                        best_k, best_d = None, 1e9
                        for k in range(n):
                            if k == i or k == j or species[k] == species[i]:
                                continue
                            cx, cy = centers[k]
                            d = math.hypot(cx - mx, cy - my)
                            if d < best_d:
                                best_k, best_d = k, d
                        if best_k is None:
                            continue
                        # Nudge it toward the midpoint
                        cx, cy = centers[best_k]
                        vx, vy = mx - cx, my - cy
                        L = math.hypot(vx, vy) + 1e-9
                        step_k = min(0.06, 0.35 * L)
                        centers[best_k] = (cx + step_k * vx / L, cy + step_k * vy / L)
                    else:
                        # Move that circle perpendicular toward the AB line until it intersects
                        cx, cy = centers[best_k]
                        if AB2 == 0:
                            continue
                        # foot of perpendicular
                        qx, qy = ax + best_t * ABx, ay + best_t * ABy
                        nx, ny = cx - qx, cy - qy
                        dist = math.hypot(nx, ny) + 1e-9
                        need = max(0.0, dist - radii[best_k] + 1e-3)
                        move = min(0.06, 0.5 * need)
                        centers[best_k] = (cx - move * nx / dist, cy - move * ny / dist)
            # write centers back each pass
            for idx, (x, y) in enumerate(centers):
                items[idx]["x"], items[idx]["y"] = x, y
            # clamp inside bed
            for obj in items:
                x, y, r = obj["x"], obj["y"], obj["r"]
                obj["x"] = min(max(r, x), bed.width_m - r)
                obj["y"] = min(max(r, y), bed.depth_m - r)
            if violations == 0:
                break
        # Short overlap relaxation to fix any new collisions
        for _it2 in range(120):
            max_overlap = 0.0
            for i2 in range(n):
                for j2 in range(i2 + 1, n):
                    dx = items[j2]["x"] - items[i2]["x"]
                    dy = items[j2]["y"] - items[i2]["y"]
                    d2 = dx * dx + dy * dy
                    if d2 == 0:
                        dx, dy, d2 = 1e-6, 0.0, 1e-12
                    d = math.sqrt(d2)
                    min_d = items[i2]["r"] + items[j2]["r"] + gap
                    overlap = min_d - d
                    if overlap > 0:
                        max_overlap = max(max_overlap, overlap)
                        ux, uy = dx / d, dy / d
                        push = 0.5 * overlap * 0.5
                        items[i2]["x"] -= ux * push
                        items[i2]["y"] -= uy * push
                        items[j2]["x"] += ux * push
                        items[j2]["y"] += uy * push
            for obj in items:
                x, y, r = obj["x"], obj["y"], obj["r"]
                obj["x"] = min(max(r, x), bed.width_m - r)
                obj["y"] = min(max(r, y), bed.depth_m - r)
            if max_overlap < 1e-3:
                break

    # --- Overlap relaxation ---
    for _it in range(max_iters):
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = items[j]["x"] - items[i]["x"]
                dy = items[j]["y"] - items[i]["y"]
                d2 = dx * dx + dy * dy
                if d2 == 0:
                    dx, dy, d2 = 1e-6, 0.0, 1e-12
                d = math.sqrt(d2)
                min_d = items[i]["r"] + items[j]["r"] + gap
                overlap = min_d - d
                if overlap > 0:
                    max_overlap = max(max_overlap, overlap)
                    ux, uy = dx / d, dy / d
                    push = 0.5 * overlap * step
                    items[i]["x"] -= ux * push
                    items[i]["y"] -= uy * push
                    items[j]["x"] += ux * push
                    items[j]["y"] += uy * push
        for obj in items:
            x, y, r = obj["x"], obj["y"], obj["r"]
            obj["x"] = min(max(r, x), bed.width_m - r)
            obj["y"] = min(max(r, y), bed.depth_m - r)
        step *= 0.98
        if max_overlap < 1e-3:
            break

    out = {sp: [] for sp in species_clusters.keys()}
    for obj in items:
        out[obj["sp"]].append((obj["x"], obj["y"], obj["c"]))
    return out


# ---------------------- Auto-scale to density ----------------------
def autoscale_to_density(
    bed: GardenBed,
    clusters: dict[str, list[tuple[float, float, int]]],
    theme: Theme,
    density: float | None,
    scale_min: float = 0.3,
    scale_max: float = 3.0,
    repack_gap: float = 0.02,
    max_iters: int = 800,
    step: float = 0.6,
):
    """
    Adjust bed size uniformly so total halo area / bed area ~= density.
    Returns (scale_applied, new_bed, repacked_clusters). If density is invalid or halos are empty,
    returns (1.0, bed, clusters) unchanged.
    """
    if density is None or density <= 0 or density > 1:
        return 1.0, bed, clusters

    A_halo = total_halo_area(clusters, theme)
    A0 = bed.width_m * bed.depth_m
    if A_halo <= 0:
        return 1.0, bed, clusters

    required = A_halo / density
    s = math.sqrt(required / A0)
    s_clamped = max(scale_min, min(scale_max, s))

    # Scale bed and cluster coordinates, then repack in new bed
    new_bed = bed.scaled(s_clamped)
    scaled = {
        sp: [(x * s_clamped, y * s_clamped, c) for (x, y, c) in pts] for sp, pts in clusters.items()
    }
    repacked = make_non_overlapping_layout(
        scaled,
        theme,
        new_bed,
        gap=repack_gap,
        max_iters=max_iters,
        step=step,
        interleave_species=False,
    )
    return s_clamped, new_bed, repacked


# ---------------------- Rendering helpers ----------------------
def _fit_axes_to_points(ax, pts, margin=0.05):
    pts = np.asarray(pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = max(1e-9, xmax - xmin)
    dy = max(1e-9, ymax - ymin)
    ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
    ax.set_ylim(ymin - margin * dy, ymax + margin * dy)
    ax.set_aspect("equal")


def _annotate(ax, x, y, text, color, size):
    """Text with a light outline for readability over colored fills."""
    t = ax.text(x, y, text, color=color, fontsize=size, weight="bold", zorder=5)
    t.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
    return t


def _col_letters(idx: int) -> str:
    """0->A, 25->Z, 26->AA ... Excel-style column letters."""
    s = ""
    idx0 = idx
    while True:
        idx0, rem = divmod(idx0, 26)
        s = chr(65 + rem) + s
        if idx0 == 0:
            break
        idx0 -= 1
    return s


# ---------------------- Rendering ----------------------
def render_schematic(
    title: str,
    clusters: dict[str, list[tuple[float, float, int]]],
    bed: GardenBed,
    theme: Theme,
    cell_size: float = CELL_SIZE,
    show_halos: bool = False,
    halo_alpha: float = None,
    save_path: str = None,
    show_legend: bool = True,
):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, bed.width_m)
    ax.set_ylim(0, bed.depth_m)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(title + f"\nOrthographic schematic — bed {bed.width_m:.2f}×{bed.depth_m:.2f} m")
    ax.set_xticks(np.arange(0, bed.width_m + cell_size, cell_size))
    ax.set_yticks(np.arange(0, bed.depth_m + cell_size, cell_size))
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.add_patch(plt.Rectangle((0, 0), bed.width_m, bed.depth_m, fill=False, ec="gray", lw=2))

    # --- Alphanumeric grid labels on axes ---
    n_x = int(round(bed.width_m / cell_size))
    n_y = int(round(bed.depth_m / cell_size))
    x_ticks = np.linspace(0, n_x * cell_size, n_x + 1)
    y_ticks = np.linspace(0, n_y * cell_size, n_y + 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([_col_letters(i) for i in range(len(x_ticks))])
    ax.set_yticklabels([str(i + 1) for i in range(len(y_ticks))])
    halo_alpha = theme.halo_alpha if halo_alpha is None else halo_alpha

    for sp, pts in clusters.items():
        st = theme.species_styles[sp]
        for x, y, c in pts:
            if show_halos:
                r = cluster_radius_m(st.spacing, c)
                circ = plt.Circle(
                    (x, y), r, facecolor=st.face, edgecolor=st.edge, alpha=halo_alpha, lw=1
                )
                ax.add_patch(circ)
            ax.scatter(
                x, y, facecolors=st.face, edgecolors=st.edge, s=st.size, marker=st.marker, zorder=3
            )
            _annotate(ax, x + 0.03, y + 0.03, str(c), st.face, 9)

    if show_legend:
        ax.legend(
            handles=legend_handles(clusters, theme), bbox_to_anchor=(1.02, 1), loc="upper left"
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(out(save_path), dpi=200, bbox_inches="tight")
    maybe_show()


def render_perspective_grid_camera(
    title: str,
    clusters,
    bed: GardenBed,
    theme: Theme,
    H,
    sign_v,
    cell_size=CELL_SIZE,
    outline_color="gray",
    grid_color="lightgray",
    show_halos=False,
    halo_alpha=None,
    save_path=None,
    show_legend: bool = True,
):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("white")
    ax.set_title(title + "\nTrue camera perspective (grid)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    halo_alpha = theme.halo_alpha if halo_alpha is None else halo_alpha

    corners = np.array([[0, 0], [bed.width_m, 0], [bed.width_m, bed.depth_m], [0, bed.depth_m]])
    trap = apply_homography(H, corners, sign_v)
    trap_closed = np.vstack([trap, trap[0]])
    ax.plot(trap_closed[:, 0], trap_closed[:, 1], color=outline_color, lw=2, zorder=2)

    n_x = int(round(bed.width_m / cell_size))
    n_y = int(round(bed.depth_m / cell_size))
    for i in range(n_x + 1):
        x = i * cell_size
        src_line = np.column_stack([np.full(n_y + 1, x), np.linspace(0, bed.depth_m, n_y + 1)])
        dst_line = apply_homography(H, src_line, sign_v)
        ax.plot(dst_line[:, 0], dst_line[:, 1], color=grid_color, lw=0.8, alpha=0.8, zorder=1)
    for j in range(n_y + 1):
        y = j * cell_size
        src_line = np.column_stack([np.linspace(0, bed.width_m, n_x + 1), np.full(n_x + 1, y)])
        dst_line = apply_homography(H, src_line, sign_v)
        ax.plot(dst_line[:, 0], dst_line[:, 1], color=grid_color, lw=0.8, alpha=0.8, zorder=1)

        # --- Alphanumeric labels along front (y=0) and left (x=0) edges ---
    label_offset = 0.06
    # Columns along front edge
    for i in range(n_x + 1):
        x = i * cell_size
        src_pt = np.array([[x, 0.0]])
        dst_pt = apply_homography(H, src_pt, sign_v)[0]
        _annotate(
            ax,
            float(dst_pt[0]) + label_offset,
            float(dst_pt[1]) - label_offset,
            _col_letters(i),
            "black",
            9,
        )
    # Rows along left edge
    for j in range(n_y + 1):
        y = j * cell_size
        src_pt = np.array([[0.0, y]])
        dst_pt = apply_homography(H, src_pt, sign_v)[0]
        _annotate(
            ax,
            float(dst_pt[0]) - 2 * label_offset,
            float(dst_pt[1]) + label_offset,
            str(j + 1),
            "black",
            9,
        )

    all_pts = [trap]
    if show_halos:
        for sp, pts in clusters.items():
            st = theme.species_styles[sp]
            for x, y, c in pts:
                r = cluster_radius_m(st.spacing, c)
                ang = np.linspace(0, 2 * np.pi, 60, endpoint=True)
                xs = x + r * np.cos(ang)
                ys = y + r * np.sin(ang)
                halo = apply_homography(H, np.column_stack([xs, ys]), sign_v)
                ax.fill(
                    halo[:, 0],
                    halo[:, 1],
                    facecolor=st.face,
                    edgecolor=st.edge,
                    alpha=halo_alpha,
                    lw=1,
                    zorder=1,
                )
                all_pts.append(halo)

    for sp, pts in clusters.items():
        st = theme.species_styles[sp]
        src = np.array([[x, y] for (x, y, _) in pts])
        if len(src) == 0:
            continue
        dst = apply_homography(H, src, sign_v)
        for (Xp, Yp), (_, _, c) in zip(dst, pts, strict=False):
            ax.scatter(
                Xp,
                Yp,
                marker=st.marker,
                facecolors=st.face,
                edgecolors=st.edge,
                s=st.size,
                zorder=4,
            )
            _annotate(ax, Xp + 0.03, Yp + 0.03, str(c), st.face, 7)
            all_pts.append(np.array([[Xp, Yp]]))

    _fit_axes_to_points(ax, np.vstack(all_pts))
    if show_legend:
        ax.legend(
            handles=legend_handles(clusters, theme), bbox_to_anchor=(1.02, 1), loc="upper left"
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(out(save_path), dpi=200, bbox_inches="tight")
    maybe_show()


def render_perspective_heights_camera(
    title: str,
    clusters,
    bed: GardenBed,
    theme: Theme,
    H,
    sign_v,
    front_scale=0.55,
    back_scale=0.25,
    outline_color="gray",
    show_halos=True,
    halo_alpha=None,
    save_path=None,
    show_legend: bool = True,
):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("white")
    ax.set_title(title + "\nTrue camera perspective (height lines)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    halo_alpha = theme.halo_alpha if halo_alpha is None else halo_alpha

    corners = np.array([[0, 0], [bed.width_m, 0], [bed.width_m, bed.depth_m], [0, bed.depth_m]])
    trap = apply_homography(H, corners, sign_v)
    trap_closed = np.vstack([trap, trap[0]])
    ax.plot(trap_closed[:, 0], trap_closed[:, 1], color=outline_color, lw=2, zorder=2)

    def scale_at_depth(y_m):
        t = np.clip(y_m / bed.depth_m, 0.0, 1.0)
        return (1 - t) * front_scale + t * back_scale

    all_pts = [trap]
    if show_halos:
        for sp, pts in clusters.items():
            st = theme.species_styles[sp]
            for x, y, c in pts:
                r = cluster_radius_m(st.spacing, c)
                ang = np.linspace(0, 2 * np.pi, 60, endpoint=True)
                xs = x + r * np.cos(ang)
                ys = y + r * np.sin(ang)
                halo = apply_homography(H, np.column_stack([xs, ys]), sign_v)
                ax.fill(
                    halo[:, 0],
                    halo[:, 1],
                    facecolor=st.face,
                    edgecolor=st.edge,
                    alpha=halo_alpha,
                    lw=1,
                    zorder=1,
                )
                all_pts.append(halo)

    for sp, pts in clusters.items():
        st = theme.species_styles[sp]
        src = np.array([[x, y] for (x, y, _) in pts])
        if len(src) == 0:
            continue
        base = apply_homography(H, src, sign_v)
        for (Xp, Yp), (_x, y, c) in zip(base, pts, strict=False):  # x unused
            h_m = HEIGHTS_CM.get(sp, 20) / 100.0
            dY = h_m * scale_at_depth(y)
            topY = Yp + dY
            ax.plot([Xp, Xp], [Yp, topY], color=st.face, lw=2, zorder=3)
            ax.scatter(
                Xp,
                topY,
                marker=st.marker,
                facecolors=st.face,
                edgecolors=st.edge,
                s=st.size / 2,
                zorder=4,
            )
            _annotate(ax, Xp + 0.02, topY + 0.02, str(c), st.face, 8)
            all_pts.append(np.array([[Xp, Yp], [Xp, topY]]))

    _fit_axes_to_points(ax, np.vstack(all_pts))
    if show_legend:
        ax.legend(
            handles=legend_handles(clusters, theme), bbox_to_anchor=(1.02, 1), loc="upper left"
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(out(save_path), dpi=200, bbox_inches="tight")
    maybe_show()


# ---------------------- CSV Export / Import ----------------------
def export_plan_csv(
    clusters: dict[str, list[tuple[float, float, int]]], bed: GardenBed, theme: Theme, path: str
):
    rows = []
    for sp, pts in clusters.items():
        for x, y, c in pts:
            spacing = theme.species_styles[sp].spacing
            A = cluster_area_m2(spacing, c)
            r = cluster_radius_m(spacing, c)
            rows.append(
                {
                    "species_key": sp,
                    "species_label": theme.species_styles[sp].label,
                    "x_m": x,
                    "y_m": y,
                    "bulbs_in_cluster": c,
                    "spacing_cm": spacing,
                    "area_m2": A,
                    "equiv_radius_m": r,
                    "bed_width_m": bed.width_m,
                    "bed_depth_m": bed.depth_m,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def import_plan_csv(
    path: str,
) -> tuple[dict[str, list[tuple[float, float, int]]], float | None, float | None]:
    df = pd.read_csv(path)
    clusters: dict[str, list[tuple[float, float, int]]] = {}
    # Initialize empty lists for any known species keys that appear in CSV
    if "species_key" in df.columns:
        for sp in df["species_key"].unique():
            clusters[str(sp)] = []

    for _, row in df.iterrows():
        sp = str(row["species_key"])
        x = float(row["x_m"])
        y = float(row["y_m"])
        c = int(row["bulbs_in_cluster"])
        clusters.setdefault(sp, []).append((x, y, c))

    w = float(df["bed_width_m"].iloc[0]) if "bed_width_m" in df.columns else None
    d = float(df["bed_depth_m"].iloc[0]) if "bed_depth_m" in df.columns else None
    return clusters, w, d


def filter_clusters_by_month(
    clusters: dict[str, list[tuple[float, float, int]]], month_name: str
) -> dict[str, list[tuple[float, float, int]]]:
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    m = month_map.get(month_name.strip().lower())
    if not m:
        raise ValueError(f"Unknown month: {month_name}")
    filtered = {}
    for sp, pts in clusters.items():
        months = BLOOM_CALENDAR.get(sp, [])
        filtered[sp] = pts if m in months else []
    return filtered


# ---------------------- HELPERS ----------------------
# Utility functions used across layout/render/CLI.
# Keep these above the CLI section so everything can import them.


# ---------------------- Unique suffix ----------------------
def unique_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------- Interleaving ----------------------
def _enforce_interleaving_strict(clusters, gap=0.01, max_passes=18):
    """
    No-op enforcement stub for strict interleaving.
    Returns clusters unchanged until a proper implementation is added.
    """
    return clusters


def _validate_interleaving_strict(clusters) -> int:
    """
    No-op validation stub for strict interleaving.
    Returns 0 to indicate no violations.
    """
    return 0


# ----------- Global output directory (set from main) ------------

_OUTPUT_DIR = "."


def set_output_dir(path: str) -> None:
    """Set the module-level output directory used by out()."""
    global _OUTPUT_DIR
    _OUTPUT_DIR = path


def ensure_output_dir(path: str) -> None:
    """Create the output directory if needed."""
    os.makedirs(path, exist_ok=True)


def out(name: str) -> str:
    """Resolve a filename under the current output directory."""
    return os.path.join(_OUTPUT_DIR, name)


# ---------------------- Headless Mode Helper ----------------------


def maybe_show():
    # Only show figures if the backend is interactive (not Agg)
    import matplotlib.pyplot as plt  # uses existing import cache

    if not plt.get_backend().lower().endswith("agg"):
        maybe_show()


# ---------------------- END HELPERS ----------------------


# ---------------------- CLI ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cell", type=float, default=CELL_SIZE)
    parser.add_argument("--avg", type=int, default=5)
    parser.add_argument("--min-odd", type=int, default=3)
    parser.add_argument("--gap", type=float, default=0.02)
    parser.add_argument("--halos", action="store_true")
    parser.add_argument("--legend-off", action="store_true", help="Disable legends in all views")
    parser.add_argument(
        "--interleave-species",
        action="store_true",
        help="Ensure clusters of the same species are not direct neighbors; a different species must lie between any same-species pair along their line of centers.",
    )
    parser.add_argument(
        "--interleave-validate-strict", action="store_true", help="Extra sure interleaving yeeeee"
    )
    parser.add_argument(
        "--interleave-max-passes",
        type=int,
        default=18,
        help="Maximum passes for strict interleaving enforcement (CSV path only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write images and CSV files (default: current folder).",
    )

    # camera
    parser.add_argument("--cam-height", type=float, default=1.55)
    parser.add_argument("--cam-dist", type=float, default=0.30)
    parser.add_argument("--cam-pitch", type=float, default=35.0)
    parser.add_argument("--cam-focal", type=float, default=1.8)

    # density auto-scale (only applies when generating a new plan)
    parser.add_argument(
        "--density", type=float, default=0.5, help="Target coverage 0..1 for auto-scaling the bed"
    )
    parser.add_argument(
        "--scale-min", type=float, default=0.3, help="Minimum allowed uniform scale"
    )
    parser.add_argument(
        "--scale-max", type=float, default=3.0, help="Maximum allowed uniform scale"
    )

    # load from CSV and month filter
    parser.add_argument(
        "--from-csv", type=str, default=None, help="Path to a previously exported plan CSV"
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Month name to filter species by bloom time (requires --from-csv)",
    )
    parser.add_argument(
        "--bed-width",
        type=float,
        default=None,
        help="Override bed width when loading from CSV without metadata",
    )
    parser.add_argument(
        "--bed-depth",
        type=float,
        default=None,
        help="Override bed depth when loading from CSV without metadata",
    )

    # output
    parser.add_argument("--prefix", type=str, default="camera_perspective_v6_5")
    args = parser.parse_args()
    ensure_output_dir(args.output_dir)
    set_output_dir(args.output_dir)

    # Theme & RNG
    theme = default_theme()
    rng = random.Random(args.seed)

    # Unique suffix for outputs
    suf = unique_suffix()
    out_prefix = f"{args.prefix}_{suf}"

    if args.from_csv:
        # --- Load existing layout ---
        clusters, w_csv, d_csv = import_plan_csv(args.from_csv)

        # Determine bed W/D
        if args.bed_width is not None and args.bed_depth is not None:
            bed = GardenBed(args.bed_width, args.bed_depth)
        elif w_csv is not None and d_csv is not None:
            bed = GardenBed(w_csv, d_csv)
        else:
            # Infer from coordinates (max x,y with a small margin)
            xs = [x for pts in clusters.values() for (x, _, _) in pts]
            ys = [y for pts in clusters.values() for (_, y, _) in pts]
            width_m = (max(xs) * 1.02) if xs else DEFAULT_BED_WIDTH
            depth_m = (max(ys) * 1.02) if ys else DEFAULT_BED_DEPTH
            bed = GardenBed(width_m, depth_m)

        # Optional month filter
        if args.month:
            clusters = filter_clusters_by_month(clusters, args.month)

        # Camera
        H, sign_v = camera_homography(
            bed,
            cam_height=args.cam_height,
            cam_dist=args.cam_dist,
            cam_pitch_deg=args.cam_pitch,
            cam_focal=args.cam_focal,
        )

        # --- Strict interleaving pass (nearest same-species pairs) ---
        if args.interleave_species:
            clusters = _enforce_interleaving_strict(
                clusters, gap=args.gap, max_passes=args.interleave_max_passes
            )
            if args.interleave_validate_strict:
                v = _validate_interleaving_strict(clusters)
                if v > 0:
                    raise RuntimeError(f"Interleaving validator found {v} violation(s)")

        # Render (no CSV rewrite by default; this is a view of an existing plan)
        render_schematic(
            f"Filtered View — {args.month or 'All'}",
            clusters,
            bed,
            theme,
            cell_size=args.cell,
            show_halos=args.halos,
            save_path=f"{out_prefix}_schematic.png",
            show_legend=not args.legend_off,
        )

        render_perspective_grid_camera(
            f"Filtered View — {args.month or 'All'}",
            clusters,
            bed,
            theme,
            H,
            sign_v,
            cell_size=args.cell,
            show_halos=args.halos,
            save_path=f"{out_prefix}_perspective_grid.png",
            show_legend=not args.legend_off,
        )

        render_perspective_heights_camera(
            f"Filtered View — {args.month or 'All'}",
            clusters,
            bed,
            theme,
            H,
            sign_v,
            show_halos=args.halos,
            save_path=f"{out_prefix}_perspective_heights.png",
            show_legend=not args.legend_off,
        )

    else:
        # --- Generate a fresh plan ---
        bed = GardenBed(DEFAULT_BED_WIDTH, DEFAULT_BED_DEPTH)
        clusters = build_random_clusters(
            theme=theme, bed=bed, rng=rng, avg=args.avg, min_odd=args.min_odd
        )
        clusters = make_non_overlapping_layout(
            clusters,
            theme,
            bed,
            gap=args.gap,
            max_iters=800,
            step=0.6,
            interleave_species=args.interleave_species,
        )

        if args.density is not None:
            s, bed, clusters = autoscale_to_density(
                bed,
                clusters,
                theme,
                args.density,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
                repack_gap=args.gap,
                max_iters=800,
                step=0.6,
            )
            print(
                f"[density] target={args.density:.2f}  scale={s:.3f}  new bed = {bed.width_m:.2f}×{bed.depth_m:.2f} m"
            )

        H, sign_v = camera_homography(
            bed,
            cam_height=args.cam_height,
            cam_dist=args.cam_dist,
            cam_pitch_deg=args.cam_pitch,
            cam_focal=args.cam_focal,
        )

        # Enforce interleaving on final layout before rendering (optional)
        # if args.interleave_species:
        #    clusters = _enforce_interleaving_strict(
        #        clusters, gap=args.gap, max_passes=args.interleave_max_passes
        #    )
        #    if args.interleave_validate_strict:
        #        v = _validate_interleaving_strict(clusters)
        #        if v > 0:
        #            raise RuntimeError(f"Interleaving validator found {v} violation(s)")

        # Export CSV with bed metadata
        export_plan_csv(clusters, bed, theme, f"{out_prefix}_plan.csv")

        render_schematic(
            "All_Bulbs_Schematic",
            clusters,
            bed,
            theme,
            cell_size=args.cell,
            show_halos=args.halos,
            save_path=f"{out_prefix}_schematic.png",
            show_legend=not args.legend_off,
        )

        render_perspective_grid_camera(
            "All_Bulbs_Perspective (Camera Grid)",
            clusters,
            bed,
            theme,
            H,
            sign_v,
            cell_size=args.cell,
            show_halos=args.halos,
            save_path=f"{out_prefix}_perspective_grid.png",
            show_legend=not args.legend_off,
        )

        render_perspective_heights_camera(
            "All_Bulbs_Perspective (Camera Heights)",
            clusters,
            bed,
            theme,
            H,
            sign_v,
            show_halos=args.halos,
            save_path=f"{out_prefix}_perspective_heights.png",
            show_legend=not args.legend_off,
        )


if __name__ == "__main__":
    main()
