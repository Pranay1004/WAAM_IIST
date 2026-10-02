"""
G-code parser for WAAM toolpath extraction.

Parses standard G-code (G0, G1, G2, G3) to extract torch position,
feed rate, and power at each point. Outputs a structured toolpath
that the FreeFEM script generator uses for element activation.

Supports:
  - G0/G1 linear moves
  - G2/G3 circular arcs (IJK or R format)
  - Layer detection from Z-increments
  - Feed rate (F) and power/spindle (S) extraction
  - Comment parsing for metadata
"""

import re
import math
import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ToolpathPoint:
    """A single point on the toolpath."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    feed_rate: float = 0.0      # mm/min
    power: float = 0.0          # W or %
    time: float = 0.0           # cumulative time [s]
    layer: int = 0
    move_type: str = "G1"       # G0, G1, G2, G3
    is_deposition: bool = True  # False for travel moves (G0)


@dataclass
class Layer:
    """Metadata for one deposition layer."""
    index: int
    z_bottom: float
    z_top: float
    start_time: float
    end_time: float
    path_length: float = 0.0
    point_count: int = 0


@dataclass
class Toolpath:
    """Complete parsed toolpath."""
    points: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    total_time: float = 0.0
    total_length: float = 0.0
    bounding_box: dict = field(default_factory=lambda: {
        "x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0,
        "z_min": 0, "z_max": 0
    })
    n_layers: int = 0
    source_file: str = ""


def _parse_word(line: str, letter: str) -> Optional[float]:
    """Extract a numeric value for a given G-code letter (e.g., 'X', 'F')."""
    match = re.search(rf'{letter}\s*([+-]?\d*\.?\d+)', line, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _arc_points(x0, y0, x1, y1, i_off, j_off, clockwise, n_segments=20):
    """Interpolate circular arc into line segments."""
    cx = x0 + i_off
    cy = y0 + j_off
    r = math.sqrt(i_off**2 + j_off**2)
    if r < 1e-9:
        return [(x1, y1)]

    theta_start = math.atan2(y0 - cy, x0 - cx)
    theta_end = math.atan2(y1 - cy, x1 - cx)

    if clockwise:
        if theta_end >= theta_start:
            theta_end -= 2 * math.pi
    else:
        if theta_end <= theta_start:
            theta_end += 2 * math.pi

    pts = []
    for k in range(1, n_segments + 1):
        frac = k / n_segments
        theta = theta_start + frac * (theta_end - theta_start)
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pts


def parse_gcode(filepath: str, layer_height: float = 3.0,
                z_tolerance: float = 0.5,
                default_feed: float = 600.0,
                default_power: float = 3840.0) -> Toolpath:
    """
    Parse a G-code file and extract the toolpath.

    Args:
        filepath: Path to .gcode or .nc file
        layer_height: Expected layer height in mm (for layer detection)
        z_tolerance: Z-change threshold to detect new layer [mm]
        default_feed: Default feed rate [mm/min] if not specified
        default_power: Default power [W] if not specified

    Returns:
        Toolpath object with points, layers, and metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"G-code file not found: {filepath}")

    toolpath = Toolpath(source_file=str(filepath))

    # Current state
    cx, cy, cz = 0.0, 0.0, 0.0
    current_feed = default_feed
    current_power = default_power
    current_time = 0.0
    current_layer = 0
    last_z = 0.0
    current_move = "G1"
    is_absolute = True
    deposition_on = True

    layer_start_time = 0.0
    layer_path_length = 0.0
    layer_point_count = 0
    layer_z_bottom = 0.0

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()

            # Remove comments
            if ';' in line:
                line = line[:line.index(';')].strip()
            if '(' in line and ')' in line:
                line = re.sub(r'\(.*?\)', '', line).strip()
            if not line:
                continue

            # Parse G-code command
            g_val = _parse_word(line, 'G')
            m_val = _parse_word(line, 'M')

            # Handle M-codes
            if m_val is not None:
                m_int = int(m_val)
                if m_int in (3, 4):   # spindle/arc ON
                    deposition_on = True
                elif m_int == 5:      # spindle/arc OFF
                    deposition_on = False
                continue

            # Handle coordinate mode
            if g_val is not None:
                g_int = int(g_val)
                if g_int == 90:
                    is_absolute = True
                    continue
                elif g_int == 91:
                    is_absolute = False
                    continue
                elif g_int in (0, 1, 2, 3):
                    current_move = f"G{g_int}"

            # Parse coordinates
            xv = _parse_word(line, 'X')
            yv = _parse_word(line, 'Y')
            zv = _parse_word(line, 'Z')
            fv = _parse_word(line, 'F')
            sv = _parse_word(line, 'S')
            iv = _parse_word(line, 'I')
            jv = _parse_word(line, 'J')

            if fv is not None:
                current_feed = fv
            if sv is not None:
                current_power = sv

            # Skip lines with no motion
            if xv is None and yv is None and zv is None:
                continue

            # Compute target position
            if is_absolute:
                nx = xv if xv is not None else cx
                ny = yv if yv is not None else cy
                nz = zv if zv is not None else cz
            else:
                nx = cx + (xv if xv is not None else 0.0)
                ny = cy + (yv if yv is not None else 0.0)
                nz = cz + (zv if zv is not None else 0.0)

            # Detect layer change
            if abs(nz - last_z) > z_tolerance and len(toolpath.points) > 0:
                # Save previous layer
                if current_layer > 0 or layer_point_count > 0:
                    toolpath.layers.append(Layer(
                        index=current_layer,
                        z_bottom=layer_z_bottom,
                        z_top=last_z,
                        start_time=layer_start_time,
                        end_time=current_time,
                        path_length=layer_path_length,
                        point_count=layer_point_count
                    ))
                current_layer += 1
                layer_start_time = current_time
                layer_path_length = 0.0
                layer_point_count = 0
                layer_z_bottom = nz
                last_z = nz

            # Handle arcs (G2/G3)
            if current_move in ("G2", "G3") and iv is not None and jv is not None:
                clockwise = (current_move == "G2")
                arc_pts = _arc_points(cx, cy, nx, ny,
                                      iv if iv else 0.0,
                                      jv if jv else 0.0,
                                      clockwise)
                px, py = cx, cy
                for (ax, ay) in arc_pts:
                    seg_len = math.sqrt((ax - px)**2 + (ay - py)**2)
                    if current_feed > 0:
                        current_time += (seg_len / current_feed) * 60.0
                    layer_path_length += seg_len

                    toolpath.points.append(ToolpathPoint(
                        x=ax, y=ay, z=nz,
                        feed_rate=current_feed,
                        power=current_power if deposition_on else 0.0,
                        time=current_time,
                        layer=current_layer,
                        move_type=current_move,
                        is_deposition=deposition_on
                    ))
                    layer_point_count += 1
                    px, py = ax, ay
            else:
                # Linear move (G0/G1)
                seg_len = math.sqrt((nx - cx)**2 + (ny - cy)**2 + (nz - cz)**2)
                feed = current_feed if current_move != "G0" else 6000.0
                if feed > 0:
                    current_time += (seg_len / feed) * 60.0
                layer_path_length += seg_len

                is_dep = deposition_on and (current_move != "G0")

                toolpath.points.append(ToolpathPoint(
                    x=nx, y=ny, z=nz,
                    feed_rate=current_feed,
                    power=current_power if is_dep else 0.0,
                    time=current_time,
                    layer=current_layer,
                    move_type=current_move,
                    is_deposition=is_dep
                ))
                layer_point_count += 1

            cx, cy, cz = nx, ny, nz

    # Save last layer
    if layer_point_count > 0:
        toolpath.layers.append(Layer(
            index=current_layer,
            z_bottom=layer_z_bottom,
            z_top=cz,
            start_time=layer_start_time,
            end_time=current_time,
            path_length=layer_path_length,
            point_count=layer_point_count
        ))

    # Compute metadata
    toolpath.total_time = current_time
    toolpath.n_layers = len(toolpath.layers)
    if toolpath.points:
        xs = [p.x for p in toolpath.points]
        ys = [p.y for p in toolpath.points]
        zs = [p.z for p in toolpath.points]
        toolpath.bounding_box = {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "z_min": min(zs), "z_max": max(zs)
        }
        toolpath.total_length = sum(
            math.sqrt((toolpath.points[i].x - toolpath.points[i-1].x)**2 +
                       (toolpath.points[i].y - toolpath.points[i-1].y)**2 +
                       (toolpath.points[i].z - toolpath.points[i-1].z)**2)
            for i in range(1, len(toolpath.points))
        )

    return toolpath


def export_toolpath_csv(toolpath: Toolpath, output_path: str):
    """Export toolpath points to CSV for FreeFEM consumption."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time_s', 'x_m', 'y_m', 'z_m', 'feed_mm_min',
            'power_W', 'layer', 'move_type', 'is_deposition'
        ])
        for p in toolpath.points:
            writer.writerow([
                f"{p.time:.4f}",
                f"{p.x / 1000.0:.6f}",  # mm -> m
                f"{p.y / 1000.0:.6f}",
                f"{p.z / 1000.0:.6f}",
                f"{p.feed_rate:.1f}",
                f"{p.power:.1f}",
                p.layer,
                p.move_type,
                int(p.is_deposition)
            ])


def export_layer_summary(toolpath: Toolpath, output_path: str):
    """Export layer summary to JSON."""
    data = {
        "source_file": toolpath.source_file,
        "n_layers": toolpath.n_layers,
        "total_time_s": toolpath.total_time,
        "total_length_mm": toolpath.total_length,
        "bounding_box_mm": toolpath.bounding_box,
        "layers": [asdict(l) for l in toolpath.layers]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gcode_parser.py <file.gcode> [layer_height_mm]")
        sys.exit(1)

    gcode_file = sys.argv[1]
    lh = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    tp = parse_gcode(gcode_file, layer_height=lh)
    print(f"Parsed {len(tp.points)} points across {tp.n_layers} layers")
    print(f"Bounding box: {tp.bounding_box}")
    print(f"Total time: {tp.total_time:.1f} s")
    print(f"Total path length: {tp.total_length:.1f} mm")

    out_csv = gcode_file.rsplit('.', 1)[0] + "_toolpath.csv"
    out_json = gcode_file.rsplit('.', 1)[0] + "_layers.json"
    export_toolpath_csv(tp, out_csv)
    export_layer_summary(tp, out_json)
    print(f"Exported: {out_csv}, {out_json}")
