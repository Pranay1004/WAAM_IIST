"""
toolpath.py
Converts sliced layer polygons into ordered toolpaths for WAAM deposition.

WAAM-specific notes:
- Bead width (not nozzle diameter) drives contour offsetting and infill spacing.
- Typical WAAM bead widths: 4-8mm (vs ~0.4mm FDM nozzles) - this changes wall
  counts and infill density dramatically compared to polymer slicers.
- Contour strategy: outer perimeter bead(s) + zigzag (raster) infill for the interior.

Robustness notes (see LayerToolpath.warnings / summarize_warnings() for
anything the planner had to work around):
- If a region is too thin for even one full-width perimeter, we fall back to
  a single centerline bead instead of silently dropping the material.
- Perimeter count adapts downward per-region if the requested count doesn't
  physically fit, instead of failing closed (0 contours).
- Regions smaller than `min_feature_area` are dropped explicitly, with a
  warning, rather than silently - so you can see what got skipped and decide
  if that's acceptable.
- MultiPolygon results from buffer() operations (common when a shape pinches
  and splits) are always iterated, never assumed single.
"""

import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, box
from shapely.ops import unary_union
from shapely.affinity import rotate as shp_rotate
from dataclasses import dataclass, field
import warnings as _warnings
from contextlib import contextmanager


@contextmanager
def _quiet_buffer():
    """
    Shapely emits RuntimeWarning (divide/invalid in buffer) on degenerate
    thin-region offsets. The callers below already handle empty/degraded
    results explicitly, so the warning is pure log noise - suppress it locally.
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        yield


@dataclass
class Bead:
    """A single continuous weld bead - a sequence of (x, y) points."""
    points: list          # [(x, y), ...]
    kind: str             # "perimeter", "infill", or "centerline" (thin-feature fallback)
    z: float = 0.0


@dataclass
class LayerToolpath:
    z: float
    index: int
    beads: list = field(default_factory=list)     # list[Bead], in deposition order
    warnings: list = field(default_factory=list)   # human-readable diagnostics for this layer


def _iter_polys(geom):
    """Normalize a shapely geometry (Polygon/MultiPolygon/GeometryCollection/empty) to a flat polygon list."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(_iter_polys(g))
        return out
    return []  # lines/points from degenerate buffers - not fabricatable, drop


def offset_contours(polygon: Polygon, bead_width: float, num_perimeters: int,
                     min_feature_area: float = None, warnings: list = None):
    """
    Generate up to `num_perimeters` concentric offset contours inward from the
    polygon boundary, spaced by bead_width.

    Robust to thin features: if the very first inward offset (at -bead_width/2)
    is already empty, the region is thinner than one bead. Instead of dropping
    it, we fall back to a single centerline bead so the feature isn't silently
    lost from the output.

    Returns (contours, infill_region, centerline_beads).
    """
    if warnings is None:
        warnings = []
    if min_feature_area is None:
        min_feature_area = (bead_width * bead_width) * 0.5  # heuristic: half a bead-square

    if polygon.area < min_feature_area:
        warnings.append(
            f"Skipped a region with area {polygon.area:.2f}mm^2 "
            f"(below min_feature_area={min_feature_area:.2f}mm^2) - too small to fabricate reliably."
        )
        return [], Polygon(), []

    contours = []
    centerline_beads = []
    fitted_perimeters = 0

    for i in range(num_perimeters):
        offset_dist = -(bead_width / 2.0 + i * bead_width)
        with _quiet_buffer():
            offset_poly = polygon.buffer(offset_dist, join_style=2)

        if offset_poly.is_empty:
            if i == 0:
                # Thinner than one full bead width anywhere along this loop.
                # Fall back to a centerline trace so the feature isn't lost entirely.
                with _quiet_buffer():
                    approx = polygon.buffer(-min(bead_width * 0.1, polygon.area ** 0.5 * 0.1), join_style=2)
                polys = _iter_polys(approx)
                if not polys:
                    polys = [polygon]
                for p in polys:
                    centerline_beads.append(list(p.exterior.coords))
                warnings.append(
                    f"Region area {polygon.area:.1f}mm^2 is thinner than one bead width "
                    f"({bead_width}mm) - used a centerline fallback bead instead of a normal "
                    f"perimeter. Check this area on the real part; bead may be oversized here."
                )
            break

        for g in _iter_polys(offset_poly):
            if g.exterior is not None and g.exterior.length > 0:
                contours.append(LineString(g.exterior.coords))
            for interior in g.interiors:
                contours.append(LineString(interior.coords))

        fitted_perimeters = i + 1

    if fitted_perimeters < num_perimeters and fitted_perimeters > 0:
        warnings.append(
            f"Requested {num_perimeters} perimeters but only {fitted_perimeters} fit "
            f"in a region of area {polygon.area:.1f}mm^2 - infill skipped there (adapted automatically)."
        )

    infill_region = None
    with _quiet_buffer():
        infill_region = polygon.buffer(-(bead_width / 2.0 + fitted_perimeters * bead_width), join_style=2)

    return contours, infill_region, centerline_beads


INFILL_STYLES = ("zigzag", "lines", "contour", "concentric", "grid", "triangles")
"""
Available infill styles (WAAM-relevant differences, not just looks):
- "zigzag": one connected boustrophedon raster per region (single arc strike,
  travel jumps between disjoint segments are air moves with the arc OFF).
- "lines": every scanline segment is its own bead (own arc start/stop + crater
  fill each). More strikes, but no long air jumps dragging over hot metal -
  useful on regions with holes/concavities.
- "contour": perimeters only, hollow interior. The classic WAAM thin-wall mode.
- "concentric": interior filled with successive inward offset loops
  (contour-parallel). Continuous deposition like zigzag but follows the part
  shape - even heat, no raster turnarounds. Good default for WAAM blocks.
- "grid": two crossing raster sets (angle, angle+90) in the SAME layer, each
  segment its own bead. Stiffer, but double heat input and double material at
  crossings - watch interpass temperature.
- "triangles": three raster sets (angle, angle+60, angle+120) in the same
  layer. Same trade-offs as grid, more so.

Deliberately NOT included: honeycomb/gyroid-style sparse plastic infills.
They optimize polymer stiffness-per-gram with thin extrusion; a 5mm WAAM bead
makes the cells enormous, every cell wall costs an arc strike, and sparse
metal cells trap heat and slump. Grid/triangles cover the cross-hatch need.
"""


def raster_scan_segments(polygon, bead_width: float, angle_deg: float = 0.0):
    """
    Clip parallel scanlines spaced `bead_width` apart against the polygon.
    Returns (lines, centroid, angle_deg): lines[i] holds the disjoint segments
    on scanline i, each a list of (x, y) in the ROTATED scanline frame.

    NOTE: segments stay in the rotated frame on purpose - ordering/stitching
    decisions (which segment follows which) must happen in the frame where
    "back-and-forth" is defined. Callers convert to world coordinates only
    at the end via unrotate_segment(). Shared core behind the "zigzag" and
    "lines" styles, so both always agree on geometry.
    """
    if polygon is None or polygon.is_empty or polygon.area < 1e-6:
        return [], None, angle_deg

    centroid = polygon.centroid
    rotated = shp_rotate(polygon, -angle_deg, origin=centroid, use_radians=False)

    minx, miny, maxx, maxy = rotated.bounds
    if not all(np.isfinite([minx, miny, maxx, maxy])):
        return [], centroid, angle_deg  # degenerate geometry guard

    y = miny + bead_width / 2.0
    lines = []
    max_lines = 100000  # hard safety cap against pathological/near-zero bead_width loops

    n = 0
    while y <= maxy and n < max_lines:
        n += 1
        scan_line = LineString([(minx - 1, y), (maxx + 1, y)])
        try:
            clipped = rotated.intersection(scan_line)
        except Exception:
            y += bead_width
            continue

        if clipped.is_empty:
            y += bead_width
            continue

        segments = []
        if clipped.geom_type == "LineString":
            if clipped.length > 1e-6:
                segments = [list(clipped.coords)]
        elif clipped.geom_type == "MultiLineString":
            segments = [list(g.coords) for g in clipped.geoms if g.length > 1e-6]
        # Point/GeometryCollection intersections (tangent scanlines) are skipped -
        # not fabricatable as a bead.

        if segments:
            lines.append(segments)

        y += bead_width

    return lines, centroid, angle_deg


def unrotate_segment(segment, angle_deg: float, centroid):
    """Map one scanline-frame segment back to world coordinates."""
    return list(shp_rotate(LineString(segment), angle_deg,
                           origin=centroid, use_radians=False).coords)


def concentric_fill(polygon, bead_width: float):
    """
    Fill a region with successive inward offset loops (contour-parallel),
    spaced `bead_width` apart, outer to inner. Returns a list of closed
    coordinate loops in world coordinates.

    Handles holes (interior loops emitted too) and pinched splits (buffer
    results iterated as MultiPolygon, never assumed single). Stops when the
    offset vanishes - regions narrower than one bead simply contribute no
    loop here (the centerline fallback in offset_contours already covers the
    too-thin-for-perimeter case upstream).
    """
    if polygon is None or polygon.is_empty or polygon.area < 1e-6:
        return []
    try:
        minx, miny, maxx, maxy = polygon.bounds
        import math
        diag = math.hypot(maxx - minx, maxy - miny)
        max_loops = int(diag / bead_width) + 3 if bead_width > 0 else 3
    except Exception:
        max_loops = 10000

    loops = []
    k = 0
    while k < max_loops:
        try:
            with _quiet_buffer():
                off = polygon.buffer(-(bead_width / 2.0 + k * bead_width), join_style=2)
        except Exception:
            break
        if off is None or off.is_empty:
            break
        found = False
        for g in _iter_polys(off):
            try:
                if g.exterior is not None and g.exterior.length > 1e-6:
                    loops.append(list(g.exterior.coords))
                    found = True
                for interior in g.interiors:
                    if interior.length > 1e-6:
                        loops.append(list(interior.coords))
                        found = True
            except Exception:
                continue
        if not found:
            break
        k += 1
    return loops


def zigzag_infill(polygon, bead_width: float, angle_deg: float = 0.0):
    """
    Generate a zigzag (raster) infill toolpath for the given polygon region.
    Lines are spaced `bead_width` apart, rotated by angle_deg.
    Returns a single connected polyline (list of (x,y)) approximating the raster,
    stitched together in a boustrophedon (back-and-forth) pattern.
    """
    lines, centroid, angle = raster_scan_segments(polygon, bead_width, angle_deg)
    if not lines:
        return []

    stitched = []
    direction = 1
    for segments in lines:
        ordered = sorted(segments, key=lambda s: s[0][0], reverse=(direction < 0))
        for seg in ordered:
            stitched.extend(seg if direction > 0 else seg[::-1])
        direction *= -1

    line_geom = LineString(stitched)
    unrotated = shp_rotate(line_geom, angle, origin=centroid, use_radians=False)
    return list(unrotated.coords)


def plan_layer_toolpath(layer, bead_width: float, num_perimeters: int = 2,
                          infill_angle_step: float = 45.0, min_feature_area: float = None,
                          infill_style: str = "zigzag"):
    """
    Build a full toolpath for one sliced layer: perimeters first (outer to inner),
    then infill, then any thin-feature centerline fallbacks. Alternates infill
    angle by layer index for better interlayer bonding.

    infill_style: "zigzag" (one connected raster per region), "lines" (each
    scanline segment its own bead/arc-strike), "concentric" (inward offset
    loops), "grid" / "triangles" (2-3 crossing raster sets per layer), or
    "contour" (perimeters only, hollow - the classic WAAM thin-wall mode).

    Collects human-readable warnings on the returned LayerToolpath so callers
    can surface them instead of geometry silently disappearing.
    """
    if infill_style not in INFILL_STYLES:
        raise ValueError(f"infill_style must be one of {INFILL_STYLES}, got `{infill_style}`.")
    tp = LayerToolpath(z=layer.z, index=layer.index)
    infill_angle = (layer.index * infill_angle_step) % 180

    if not layer.polygons:
        tp.warnings.append("Layer has no polygons (empty slice) - nothing to deposit.")
        return tp

    for poly in layer.polygons:
        if not poly.is_valid:
            with _quiet_buffer():
                poly = poly.buffer(0)  # attempt auto-repair of minor self-intersections
            if not poly.is_valid or poly.is_empty:
                tp.warnings.append("Dropped an invalid polygon that could not be auto-repaired.")
                continue

        contours, infill_region, centerline_beads = offset_contours(
            poly, bead_width, num_perimeters, min_feature_area=min_feature_area, warnings=tp.warnings
        )

        for c in contours:
            tp.beads.append(Bead(points=list(c.coords), kind="perimeter", z=layer.z))

        for cb in centerline_beads:
            tp.beads.append(Bead(points=cb, kind="centerline", z=layer.z))

        for region in _iter_polys(infill_region):
            if infill_style == "contour":
                continue
            if infill_style == "concentric":
                for loop in concentric_fill(region, bead_width):
                    tp.beads.append(Bead(points=loop, kind="infill", z=layer.z))
                continue
            if infill_style in ("lines", "grid", "triangles"):
                angles = {"lines": (infill_angle,),
                          "grid": (infill_angle, (infill_angle + 90) % 180),
                          "triangles": (infill_angle, (infill_angle + 60) % 180,
                                        (infill_angle + 120) % 180)}[infill_style]
                for a in angles:
                    seg_lines, _c, _a = raster_scan_segments(region, bead_width, angle_deg=a)
                    for segs in seg_lines:
                        for seg in segs:
                            tp.beads.append(Bead(points=unrotate_segment(seg, _a, _c),
                                                 kind="infill", z=layer.z))
                continue
            path = zigzag_infill(region, bead_width, angle_deg=infill_angle)
            if path:
                tp.beads.append(Bead(points=path, kind="infill", z=layer.z))

    if not tp.beads:
        tp.warnings.append(
            "Layer produced zero beads despite having input geometry - "
            "likely all regions were below min_feature_area or a geometry op failed."
        )

    return tp


def plan_all_layers(layers, bead_width: float, num_perimeters: int = 2,
                      infill_angle_step: float = 45.0, min_feature_area: float = None,
                      infill_style: str = "zigzag"):
    toolpaths = []
    for l in layers:
        tp = plan_layer_toolpath(l, bead_width, num_perimeters, infill_angle_step,
                                 min_feature_area, infill_style)
        toolpaths.append(tp)
    return toolpaths


def summarize_warnings(toolpaths):
    """Collect all per-layer warnings into a single flat report with layer indices attached."""
    report = []
    for tp in toolpaths:
        for w in tp.warnings:
            report.append(f"[layer {tp.index}, z={tp.z:.2f}] {w}")
    return report
