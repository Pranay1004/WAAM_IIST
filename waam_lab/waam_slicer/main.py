"""
main.py
End-to-end WAAM slicer CLI: STL -> layers -> toolpaths -> G-code.

Fails loudly (non-zero exit, clear message) on:
  - unreadable/empty mesh files
  - meshes too damaged to slice into any layers (unless --allow-damaged-mesh)
  - toolpaths that produce zero fabricable beads
  - toolpaths exceeding machine travel limits (unless --skip-bounds-check)
  - invalid process parameters (negative/zero speeds, etc.)

Usage:
    python main.py part.stl -o output.gcode \
        --layer-height 2.0 --bead-width 5.0 --perimeters 2 \
        --travel-speed 400 --weld-speed 150 \
        --x-max 300 --y-max 300 --z-max 150
"""

import argparse
import os
import sys
import time

from slicer_core import load_mesh, slice_mesh, auto_place_mesh, apply_orientation, MeshValidationError, SlicingError
from toolpath import plan_all_layers, summarize_warnings, INFILL_STYLES
from gcode_gen import WAAMParams, MachineLimits, generate_gcode, write_gcode_file, GCodeGenerationError, compute_job_stats, _format_time


def main():
    ap = argparse.ArgumentParser(description="WAAM G-code slicer")
    ap.add_argument("stl_file", help="Path to input mesh: STL, 3MF, OBJ, or STEP/IGES B-rep (.step/.stp/.iges/.igs, tessellated via gmsh)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output G-code file path. Default: geometry/<inputname>.gcode")
    ap.add_argument("--step-mesh-size", type=float, default=5.0,
                    help="Max triangle size (mm) when tessellating STEP/IGES via gmsh. "
                         "Keep at/below bead width. Converted STL is cached as <name>_waam.stl.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Uniform scale factor applied after loading (e.g. 0.3 = 30%% size). "
                         "Use when the part is physically larger than the machine envelope.")
    ap.add_argument("--auto-place", action="store_true",
                    help="Center the part in XY and rest Z-min on the envelope floor before slicing. "
                         "Still refuses if the (scaled) part is larger than the envelope.")
    ap.add_argument("--rotate-x", type=float, default=0.0, help="Rotate part about X (degrees, about centroid, before auto-place)")
    ap.add_argument("--rotate-y", type=float, default=0.0, help="Rotate part about Y (degrees, about centroid, before auto-place)")
    ap.add_argument("--rotate-z", type=float, default=0.0, help="Rotate part about Z (degrees, about centroid, before auto-place)")

    ap.add_argument("--layer-height", type=float, default=2.0, help="Layer height in mm (WAAM typical: 1.5-4mm)")
    ap.add_argument("--bead-width", type=float, default=5.0, help="Weld bead width in mm (WAAM typical: 4-8mm)")
    ap.add_argument("--perimeters", type=int, default=2, help="Number of perimeter beads per layer (adapts down automatically if it doesn't fit)")
    ap.add_argument("--min-feature-area", type=float, default=None,
                     help="Regions smaller than this (mm^2) are skipped with a warning instead of fabricated. Default: 0.5 * bead_width^2")

    ap.add_argument("--travel-speed", type=float, default=400.0, help="Rapid travel speed, mm/min")
    ap.add_argument("--weld-speed", type=float, default=150.0, help="Deposition (welding) speed, mm/min")
    ap.add_argument("--wire-feed-speed", type=float, default=None, help="Wire feed speed, m/min (optional, comment-only)")
    ap.add_argument("--infill-angle-step", type=float, default=45.0, help="Infill angle increment per layer, degrees")
    ap.add_argument("--infill-style", type=str, default="zigzag", choices=list(INFILL_STYLES),
                    help="Infill hatching: zigzag (one connected raster per region), "
                         "lines (each scanline its own bead/arc-strike), "
                         "contour (perimeters only, hollow thin-wall mode)")
    ap.add_argument("--interlayer-dwell", type=float, default=0.0, help="Pause between layers, seconds (0 = disabled) - for interpass cooling")
    ap.add_argument("--lead-in-out", type=float, default=0.0, help="Lead-in/out distance on closed loops, mm (0 = disabled) - staggers seam position and avoids stacked craters")

    ap.add_argument("--voltage", type=float, default=28.0, help="Arc voltage, V (for heat input calc)")
    ap.add_argument("--current", type=float, default=180.0, help="Welding current, A (for heat input calc)")
    ap.add_argument("--wire-efficiency", type=float, default=0.8, help="Deposition efficiency η (0.7-0.9 typical for GMAW)")
    ap.add_argument("--wire-diameter", type=float, default=1.2, help="Filler wire diameter, mm")
    ap.add_argument("--material", type=str, default="steel", choices=["steel", "aluminium", "titanium"],
                    help="Wire material (auto-sets density for mass calcs)")

    # Defaults match the nameplate on the Vashishtha Research 3-axis WAAM
    # cell (DDCS V2.1 + Champ Pulse 500): X/Y 800mm, Z 400mm strokes.
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=800.0)
    ap.add_argument("--y-min", type=float, default=0.0)
    ap.add_argument("--y-max", type=float, default=800.0)
    ap.add_argument("--z-min", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=400.0)
    ap.add_argument("--max-feed-rate", type=float, default=1000.0,
                     help="Machine's documented max feed rate, mm/min (nameplate: 1000 mm/min)")
    ap.add_argument("--skip-bounds-check", action="store_true",
                     help="Skip machine travel limit validation (NOT recommended - only for simulation-only use)")

    ap.add_argument("--allow-damaged-mesh", action="store_true",
                     help="Proceed even if the mesh remains non-watertight after repair (may produce incomplete geometry)")

    ap.add_argument("--arc-on-mcode", type=str, default="M3",
                     help="M-code that starts the arc/torch (default M3, standard DDCS spindle-output ON). "
                          "CONFIRM against this machine's actual wiring - see README.")
    ap.add_argument("--arc-off-mcode", type=str, default="M5",
                     help="M-code that stops the arc/torch (default M5, standard DDCS spindle-output OFF)")
    ap.add_argument("--wire-feed-on-mcode", type=str, default=None,
                     help="Separate M-code to start wire feed, only if it's switched independently of the arc")
    ap.add_argument("--wire-feed-off-mcode", type=str, default=None,
                     help="Separate M-code to stop wire feed, only if switched independently of the arc")

    args = ap.parse_args()

    # Default output: geometry/<inputname>.gcode (sibling to the mesh file)
    if args.output is None:
        mesh_dir = os.path.dirname(os.path.abspath(args.stl_file))
        mesh_base = os.path.splitext(os.path.basename(args.stl_file))[0]
        args.output = os.path.join(mesh_dir, f"{mesh_base}.gcode")

    t0 = time.time()

    try:
        print(f"[1/4] Loading mesh: {args.stl_file}")
        mesh, report = load_mesh(args.stl_file, strict=not args.allow_damaged_mesh,
                                 step_mesh_size=args.step_mesh_size, scale=args.scale)
        report.print_summary()

        limits = MachineLimits(
            x_min=args.x_min, x_max=args.x_max,
            y_min=args.y_min, y_max=args.y_max,
            z_min=args.z_min, z_max=args.z_max,
            max_feed_rate=args.max_feed_rate,
        )
        if args.rotate_x or args.rotate_y or args.rotate_z:
            apply_orientation(mesh, args.rotate_x, args.rotate_y, args.rotate_z)
            print(f"  Rotated part by ({args.rotate_x}, {args.rotate_y}, {args.rotate_z})deg "
                  f"about centroid; new bounds={mesh.bounds.tolist()}")
        if args.auto_place:
            place_warns: list = []
            shift = auto_place_mesh(mesh, limits, _warn=place_warns)
            for w in place_warns:
                print(f"  [warn] {w}")
            if shift is not None:
                print(f"  Auto-placed part by ({shift[0]:.1f}, {shift[1]:.1f}, {shift[2]:.1f})mm; "
                      f"new bounds={mesh.bounds.tolist()}")

        print(f"\n[2/4] Slicing at {args.layer_height}mm layer height...")
        layers = slice_mesh(mesh, layer_height=args.layer_height)
        print(f"       -> {len(layers)} layers")

        print(f"\n[3/4] Planning toolpaths (bead width={args.bead_width}mm, {args.perimeters} perimeters, infill={args.infill_style})...")
        toolpaths = plan_all_layers(layers, bead_width=args.bead_width,
                                     num_perimeters=args.perimeters,
                                     infill_angle_step=args.infill_angle_step,
                                     min_feature_area=args.min_feature_area,
                                     infill_style=args.infill_style)
        total_beads = sum(len(tp.beads) for tp in toolpaths)
        print(f"       -> {total_beads} total beads across {len(toolpaths)} layers")

        warnings = summarize_warnings(toolpaths)
        if warnings:
            print(f"\n       {len(warnings)} toolpath warning(s) - review before running on hardware:")
            for w in warnings[:15]:
                print(f"         {w}")
            if len(warnings) > 15:
                print(f"         ... and {len(warnings) - 15} more")

        print(f"\n[4/4] Generating G-code...")
        params = WAAMParams(
            travel_speed=args.travel_speed,
            weld_speed=args.weld_speed,
            wire_feed_speed=args.wire_feed_speed,
            interlayer_dwell=args.interlayer_dwell,
            lead_in_out=args.lead_in_out,
            arc_on_mcode=args.arc_on_mcode,
            arc_off_mcode=args.arc_off_mcode,
            wire_feed_on_mcode=args.wire_feed_on_mcode,
            wire_feed_off_mcode=args.wire_feed_off_mcode,
        )
        stats = compute_job_stats(toolpaths, params,
                                  bead_width=args.bead_width,
                                  layer_height=args.layer_height,
                                  voltage=args.voltage,
                                  current=args.current,
                                  wire_efficiency=args.wire_efficiency,
                                  wire_diameter=args.wire_diameter,
                                  wire_feed_speed=args.wire_feed_speed,
                                  material=args.material)
        gcode = generate_gcode(toolpaths, params, bead_width=args.bead_width,
                                layer_height=args.layer_height,
                                header_comment=f"WAAM slice of {args.stl_file}",
                                machine_limits=limits,
                                skip_bounds_check=args.skip_bounds_check,
                                job_stats=stats,
                                voltage=args.voltage,
                                current=args.current)
        write_gcode_file(gcode, args.output)

        elapsed = time.time() - t0
        n_lines = gcode.count("\n")
        print(f"\nDone in {elapsed:.2f}s -> {args.output} ({n_lines} lines)")
        print(f"\n{'='*55}")
        print(f"  JOB SUMMARY")
        print(f"{'='*55}")
        print(f"  Layers:            {stats.total_layers}")
        print(f"  Beads:             {stats.total_beads}")
        print(f"  Weld path:         {stats.weld_distance_mm:.0f} mm ({stats.weld_distance_mm/1000:.2f} m)")
        print(f"  Weld time:         {_format_time(stats.weld_time_s)}")
        print(f"  Travel time:       {_format_time(stats.travel_time_s)}")
        print(f"  Arc dwell time:    {_format_time(stats.arc_dwell_s)}")
        if args.interlayer_dwell > 0:
            print(f"  Interlayer dwell:  {_format_time(stats.interlayer_dwell_s)}")
        print(f"  TOTAL BUILD TIME:  {_format_time(stats.total_build_time_s)}")
        print(f"")
        print(f"  Wire from spool:   {stats.wire_length_m:.1f} m ({stats.wire_mass_g:.0f} g)")
        print(f"  Deposited volume:  {stats.deposited_volume_mm3:.0f} mm³ ({stats.deposited_volume_mm3/1000:.1f} cm³)")
        print(f"  Mass deposition:   {stats.mass_deposition_rate_kg_hr:.3f} kg/hr")
        print(f"")
        print(f"  Arc power:         P = {stats.arc_power_w:.0f} W  (V={args.voltage} × I={args.current})")
        print(f"  Net heat power:    P_net = {stats.net_heat_power_w:.0f} W  (η={args.wire_efficiency})")
        print(f"  Linear heat input: HI = {stats.heat_input_j_per_mm:.1f} J/mm")
        print(f"  Bead width (calc): w = {stats.bead_width_calc:.2f} mm  (parabolic model)")
        print(f"{'='*55}")
        if warnings:
            print(f"\nRemember: {len(warnings)} warning(s) were generated - see above.")

    except MeshValidationError as e:
        print(f"\n[FAILED] Mesh validation error:\n  {e}", file=sys.stderr)
        print("\nRe-export the mesh with hole-filling, or pass --allow-damaged-mesh to "
              "proceed anyway (not recommended for production parts).", file=sys.stderr)
        sys.exit(1)

    except SlicingError as e:
        print(f"\n[FAILED] Slicing error:\n  {e}", file=sys.stderr)
        sys.exit(1)

    except GCodeGenerationError as e:
        print(f"\n[FAILED] G-code generation error:\n  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
