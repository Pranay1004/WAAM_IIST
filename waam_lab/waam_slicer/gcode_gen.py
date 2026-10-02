"""
gcode_gen.py
Converts planned layer toolpaths into G-code for a Digital Dream DDCS V2.1
3-axis motion controller driving an Ador Champ Pulse 500 (+ Pulse Feed 40 SFW)
WAAM torch.

DDCS-specific notes (from the DDCS V2.1/V3.1 manuals and this machine's own
nameplate, which confirms "integration of welding torch ON/OFF with the
electronic motion controller for 3D printing purposes"):
- DDCS reads standard G-code: G0/G1 motion, G4 Pxxx dwell (ms), G21/G90 units/
  mode, M3/M4 = spindle-style output ON, M5 = output OFF.
- Files are loaded as .tap from a USB drive and run in AUTO mode - not
  streamed live - so there's no benefit to minimizing file size at the cost
  of readability; comments are cheap and help a human sanity-check the file
  in the DDCS's own program viewer before running it on hardware.
- ARC_TRIGGER_MCODE defaults to M3 (start) / M5 (stop) - the standard DDCS
  spindle-output pair - because this machine's build sheet confirms torch
  ON/OFF is already wired to the controller's output, and M3/M5 is the
  documented, standard way to drive that output. This is a strong default,
  not a guess: change ARC_ON_GCODE/ARC_OFF_GCODE below only if you trace the
  actual wiring and find it's different (e.g. M7/M8 if wire-feed is switched
  on a separate output from the arc/gas).
- No G28 (auto-home) is emitted mid-job. Homing on this machine is a
  deliberate operator action (via the MPG pendant or panel), and issuing G28
  automatically at the start/end of a G-code file risks an unexpected rapid
  move to machine zero on a machine with real limit switches and a live
  torch near the workpiece. The job starts from wherever the operator has
  already jogged/homed to.

Robustness notes:
- Machine travel bounds are validated before any G-code is written -
  GCodeGenerationError is raised if the toolpath exceeds them, rather than
  emitting motion commands that could crash into limit switches or the
  table edge.
- Feed rates and other numeric params are sanity-checked (must be positive,
  finite) at WAAMParams construction time.
- Every emitted coordinate is checked for NaN/Inf before being written - a
  degenerate geometry op upstream produces a hard error here instead of a
  malformed G-code line that a controller might interpret unpredictably.
- Optional interlayer dwell (time-based, or temperature-gated via a hook)
  gives the deposit time to cool between layers.
- Optional lead-in/lead-out on closed loops avoids stacking start/stop
  craters at the same XY point on every layer.
"""

import math
from dataclasses import dataclass


class GCodeGenerationError(Exception):
    """Raised when the toolpath cannot be safely converted to G-code."""


@dataclass
class JobStats:
    """Computed statistics for a planned WAAM build — emitted in G-code header and CLI/GUI.
    Formulas validated against WAAM_Process_Calculator-2.xlsx (AE645 course)."""
    total_layers: int = 0
    total_beads: int = 0
    weld_distance_mm: float = 0.0       # total torch-on path length
    travel_distance_mm: float = 0.0     # total rapid (torch-off) path length
    weld_time_s: float = 0.0            # time actually depositing (G1 moves)
    travel_time_s: float = 0.0          # time on rapid moves (G0)
    arc_dwell_s: float = 0.0            # arc-on + arc-off stabilization dwells
    interlayer_dwell_s: float = 0.0     # total interlayer cooling pauses
    total_build_time_s: float = 0.0     # weld + travel + dwells
    # Wire / material (Excel Section 6)
    wire_volume_mm3: float = 0.0        # raw wire consumed from spool
    deposited_volume_mm3: float = 0.0   # volume actually deposited (η × wire_vol)
    wire_length_m: float = 0.0          # wire consumed from spool
    wire_mass_g: float = 0.0           # wire mass
    mass_deposition_rate_kg_hr: float = 0.0  # DR (kg/hr) - industry standard metric
    mass_feed_per_mm: float = 0.0      # MFL (kg/mm)
    # Bead geometry (Excel Section 3 - parabolic model)
    bead_width_calc: float = 0.0       # w from parabolic model (mm)
    bead_height_calc: float = 0.0      # h (mm)
    bead_area_calc: float = 0.0        # A_bead = (2/3)*h*w (mm^2)
    volume_flow_rate: float = 0.0      # Q = A_wire × V_wire (mm^3/min)
    # Heat input (Excel Section 5)
    arc_power_w: float = 0.0           # P = U × I (W)
    net_heat_power_w: float = 0.0      # P_net = η × U × I (W)
    heat_input_j_per_mm: float = 0.0   # HI = P_net / V_torch (J/mm)
    # Per-layer breakdown (for process table)
    per_layer_weld_s: list = None
    per_layer_beads: list = None
    per_layer_z: list = None
    per_layer_weld_dist: list = None

    def __post_init__(self):
        for attr in ("per_layer_weld_s", "per_layer_beads", "per_layer_z", "per_layer_weld_dist"):
            if getattr(self, attr) is None:
                setattr(self, attr, [])


# Wire/thermal properties by material (density in kg/m^3, cp in J/(kg·K), k in W/(m·K))
_MATERIAL_DB = {
    "steel":     {"density": 7850, "cp": 500,  "k": 50,  "wire_dia": 1.2},
    "aluminium": {"density": 2700, "cp": 900,  "k": 237, "wire_dia": 1.6},
    "titanium":  {"density": 4510, "cp": 520,  "k": 22,  "wire_dia": 1.2},
}


def compute_job_stats(toolpaths, params: "WAAMParams", bead_width: float,
                      layer_height: float, voltage: float = 28.0,
                      current: float = 180.0, wire_efficiency: float = 0.8,
                      wire_diameter: float = 1.2, wire_feed_speed: float = None,
                      material: str = "steel") -> JobStats:
    """
    Compute all job statistics from planned toolpaths.
    Formulas from WAAM_Process_Calculator-2.xlsx (AE645 course).

    If wire_feed_speed is None, back-calculate from bead geometry:
      Q = A_bead × V_torch = (2/3)×h×w × V_torch
      V_wire = Q / A_wire
    """
    import numpy as np
    import math

    mat = _MATERIAL_DB.get(material, _MATERIAL_DB["steel"])
    wire_rho = mat["density"]  # g/cm^3

    stats = JobStats()
    stats.total_layers = len(toolpaths)

    # --- Per-layer weld distances and times ---
    for tp in toolpaths:
        layer_weld = 0.0
        for bead in tp.beads:
            pts = np.array(bead.points)
            if len(pts) < 2:
                continue
            seg_len = float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))))
            layer_weld += seg_len
            stats.weld_distance_mm += seg_len
            stats.total_beads += 1
        stats.per_layer_weld_s.append(layer_weld / (params.weld_speed / 60.0))
        stats.per_layer_beads.append(len(tp.beads))
        stats.per_layer_z.append(tp.z)
        stats.per_layer_weld_dist.append(layer_weld)

    # --- Excel Section 1: Process inputs ---
    V_torch = params.weld_speed  # mm/min

    # --- Excel Section 2: Volume flow rate ---
    A_wire = math.pi * wire_diameter**2 / 4.0  # mm^2

    # Back-calculate wire feed speed if not provided
    if wire_feed_speed is None:
        # From volume balance: Q = A_bead × V_torch, Q = A_wire × V_wire
        # A_bead = (2/3) × h × w (parabolic model)
        # Use user's bead_width as w, layer_height as h
        A_bead = (2.0 / 3.0) * layer_height * bead_width
        Q_needed = A_bead * V_torch  # mm^3/min
        wire_feed_speed = Q_needed / A_wire / 1000.0  # m/min (input is m/min, convert to mm/min for calc)
    # Convert wire_feed_speed from m/min to mm/min for internal calculations
    wire_feed_speed_mm_per_min = wire_feed_speed * 1000.0  # mm/min
    stats.volume_flow_rate = A_wire * wire_feed_speed_mm_per_min  # Q (mm^3/min)

    # --- Excel Section 3: Single bead geometry (parabolic model) ---
    stats.bead_height_calc = layer_height
    # w = 3π × d_wire² × V_wire / (8 × h × V_torch)
    if V_torch > 0 and layer_height > 0:
        stats.bead_width_calc = (3.0 * math.pi * wire_diameter**2 * wire_feed_speed_mm_per_min
                                 / (8.0 * layer_height * V_torch))
    else:
        stats.bead_width_calc = bead_width
    stats.bead_area_calc = (2.0 / 3.0) * layer_height * stats.bead_width_calc  # A_bead

    # --- Excel Section 5: Heat input ---
    stats.arc_power_w = voltage * current                      # P = U × I
    stats.net_heat_power_w = wire_efficiency * voltage * current  # P_net = η × U × I
    # HI = P_net / V_torch (J/mm) — linear heat input
    if V_torch > 0:
        stats.heat_input_j_per_mm = stats.net_heat_power_w / V_torch
    else:
        stats.heat_input_j_per_mm = 0.0

    # --- Excel Section 6: Mass deposition ---
    # Q in mm^3/min, rho in kg/m^3 → need g/mm^3: 1 kg/m^3 = 1e-6 g/mm^3
    rho_g_per_mm3 = wire_rho * 1e-6  # g/mm^3
    mdot_g_per_min = stats.volume_flow_rate * rho_g_per_mm3  # g/min
    stats.mass_deposition_rate_kg_hr = mdot_g_per_min * 60 / 1000  # kg/hr (DR)
    stats.mass_feed_per_mm = mdot_g_per_min / V_torch * 1000 if V_torch > 0 else 0  # kg/mm (MFL)

    # --- Material consumption ---
    deposited_vol = stats.weld_distance_mm * bead_width * layer_height  # mm^3 deposited
    stats.deposited_volume_mm3 = deposited_vol
    stats.wire_volume_mm3 = deposited_vol / max(wire_efficiency, 0.01)  # raw wire from spool
    wire_cross_section = math.pi * (wire_diameter / 2.0)**2  # mm^2
    stats.wire_length_m = stats.wire_volume_mm3 / wire_cross_section / 1000.0  # metres
    stats.wire_mass_g = stats.wire_volume_mm3 * rho_g_per_mm3  # grams

    # --- Time breakdown ---
    stats.weld_time_s = stats.weld_distance_mm / (V_torch / 60.0)
    stats.arc_dwell_s = stats.total_beads * (params.arc_on_dwell + params.arc_off_dwell)
    non_empty_layers = sum(1 for tp in toolpaths if tp.beads)
    stats.interlayer_dwell_s = non_empty_layers * params.interlayer_dwell
    # Travel estimate: z-hop per bead + repositioning between beads ≈ 30% of weld path
    stats.travel_distance_mm = stats.weld_distance_mm * 0.3
    stats.travel_time_s = stats.travel_distance_mm / (params.travel_speed / 60.0)
    stats.total_build_time_s = (stats.weld_time_s + stats.travel_time_s
                                + stats.arc_dwell_s + stats.interlayer_dwell_s)

    return stats


def _format_time(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h = m // 60
    m = m % 60
    return f"{h}h {m:02d}m {s:02d}s"


@dataclass
class MachineLimits:
    """
    Physical travel envelope of the gantry, in machine coordinates (mm).
    Defaults match the nameplate on this specific machine (Vashishtha
    Research 3-axis Cartesian WAAM cell): X/Y 800mm, Z 400mm strokes.
    Confirm your workpiece's position within this envelope before running -
    these are the machine's total strokes, not necessarily starting at 0 in
    every direction depending on how your work zero is set.
    """
    x_min: float = 0.0
    x_max: float = 800.0
    y_min: float = 0.0
    y_max: float = 800.0
    z_min: float = 0.0
    z_max: float = 400.0
    max_feed_rate: float = 1000.0  # mm/min - nameplate max speed per axis


@dataclass
class WAAMParams:
    travel_speed: float = 300.0      # mm/min - rapid moves (torch off)
    weld_speed: float = 150.0        # mm/min - deposition moves (torch on) - tune per WFS/current
    wire_feed_speed: float = None    # m/min - optional, for logging/comment only unless you sync it as an axis
    arc_on_dwell: float = 0.3        # seconds - dwell after arc start before moving (stabilize arc)
    arc_off_dwell: float = 0.2       # seconds - dwell before arc off (fill crater)
    z_hop: float = 2.0               # mm - lift between beads during travel to avoid dragging through weld
    safe_z: float = 10.0             # mm - Z height for initial/final moves
    interlayer_dwell: float = 0.0    # seconds - pause between layers for cooling (0 = disabled)
    lead_in_out: float = 0.0         # mm - overlap distance for closed-loop lead-in/lead-out (0 = disabled)

    # Arc trigger codes. Defaults are the standard DDCS spindle-output pair,
    # which is a strong default here (see module docstring) but NOT yet
    # confirmed against this machine's actual internal wiring. If tracing the
    # wiring or an old .tap file shows a different code (e.g. M7/M8), change
    # these two lines - nothing else in the pipeline needs to change.
    arc_on_mcode: str = "M3"
    arc_off_mcode: str = "M5"
    # If wire feed is switched on a SEPARATE output from the arc/gas (some
    # setups gate WFS independently), set these; otherwise leave None and
    # only arc_on_mcode/arc_off_mcode are emitted.
    wire_feed_on_mcode: str = None
    wire_feed_off_mcode: str = None

    def __post_init__(self):
        for name in ("travel_speed", "weld_speed"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise GCodeGenerationError(
                    f"WAAMParams.{name} must be a finite number, got {val}."
                )
        for name in ("arc_on_dwell", "arc_off_dwell", "z_hop", "interlayer_dwell", "lead_in_out"):
            val = getattr(self, name)
            if not math.isfinite(val) or val < 0:
                raise GCodeGenerationError(
                    f"WAAMParams.{name} must be zero or a positive finite number, got {val}."
                )
        if self.wire_feed_speed is not None and not math.isfinite(self.wire_feed_speed):
            raise GCodeGenerationError(
                f"WAAMParams.wire_feed_speed must be a finite number if set, got {self.wire_feed_speed}."
            )


def arc_on_line(params: "WAAMParams") -> str:
    return f"{params.arc_on_mcode}   ; ARC_ON - starts torch/arc via DDCS output (confirm against traced wiring)"


def arc_off_line(params: "WAAMParams") -> str:
    return f"{params.arc_off_mcode}   ; ARC_OFF - stops torch/arc"


def wire_feed_on_line(params: "WAAMParams") -> str:
    if params.wire_feed_on_mcode is None:
        return None
    wfs_note = f" WFS={params.wire_feed_speed}" if params.wire_feed_speed else ""
    return f"{params.wire_feed_on_mcode}   ; WIRE_FEED_START{wfs_note}"


def wire_feed_off_line(params: "WAAMParams") -> str:
    if params.wire_feed_off_mcode is None:
        return None
    return f"{params.wire_feed_off_mcode}   ; WIRE_FEED_STOP"


def _check_finite_xy(x, y, context=""):
    if not (math.isfinite(x) and math.isfinite(y)):
        raise GCodeGenerationError(
            f"Non-finite coordinate encountered{(' in ' + context) if context else ''}: "
            f"X={x} Y={y}. This indicates a degenerate geometry operation upstream "
            f"(e.g. a zero-area polygon or a failed buffer op) - check the toolpath "
            f"planning stage rather than trusting this G-code."
        )


def validate_bounds(toolpaths, limits: MachineLimits, z_hop: float, safe_z: float):
    """
    Check every point the toolpath will visit - including z-hop and safe-Z
    moves - against the machine's travel envelope. Raises GCodeGenerationError
    with the specific offending point rather than letting an out-of-bounds
    move reach the controller.
    """
    violations = []

    def check(x, y, z, where):
        if x < limits.x_min or x > limits.x_max:
            violations.append(f"{where}: X={x:.2f} outside [{limits.x_min}, {limits.x_max}]")
        if y < limits.y_min or y > limits.y_max:
            violations.append(f"{where}: Y={y:.2f} outside [{limits.y_min}, {limits.y_max}]")
        if z < limits.z_min or z > limits.z_max:
            violations.append(f"{where}: Z={z:.2f} outside [{limits.z_min}, {limits.z_max}]")

    check(0, 0, safe_z, "initial safe-Z move")

    for layer_tp in toolpaths:
        z_travel = layer_tp.z + z_hop
        for bi, bead in enumerate(layer_tp.beads):
            for pi, (x, y) in enumerate(bead.points):
                _check_finite_xy(x, y, context=f"layer {layer_tp.index} bead {bi} point {pi}")
                check(x, y, layer_tp.z, f"layer {layer_tp.index} bead {bi} pt {pi}")
            check(bead.points[0][0] if bead.points else 0, bead.points[0][1] if bead.points else 0,
                  z_travel, f"layer {layer_tp.index} bead {bi} z-hop travel")

            if len(violations) > 20:
                violations.append("... (further violations truncated)")
                break
        if len(violations) > 20:
            break

    if violations:
        raise GCodeGenerationError(
            "Toolpath exceeds machine travel limits at " + str(len(violations)) + " point(s):\n  "
            + "\n  ".join(violations[:20])
            + f"\n\nMachine limits: X[{limits.x_min},{limits.x_max}] "
              f"Y[{limits.y_min},{limits.y_max}] Z[{limits.z_min},{limits.z_max}]. "
              f"Either your part doesn't fit the gantry, or MachineLimits needs updating "
              f"to match your actual travel envelope."
        )


def _apply_lead_in_out(points, lead_dist):
    """
    For a closed loop (first point == last point), shift the start/stop point
    along the loop by `lead_dist` so consecutive layers don't stack their
    start/stop crater at the exact same XY - and so the arc starts slightly
    before the "real" start and ends slightly after, avoiding a visible seam
    notch. No-op if lead_dist <= 0 or the path isn't a closed loop with enough
    points to support it.
    """
    if lead_dist <= 0 or len(points) < 4:
        return points

    is_closed = (abs(points[0][0] - points[-1][0]) < 1e-6 and
                 abs(points[0][1] - points[-1][1]) < 1e-6)
    if not is_closed:
        return points

    # Walk forward from the start until we've covered lead_dist, use that as
    # the new seam point - simple and robust rather than exact arc-length math.
    cum = 0.0
    shift_idx = 1
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        cum += math.hypot(dx, dy)
        if cum >= lead_dist:
            shift_idx = i
            break

    reordered = points[shift_idx:-1] + points[:shift_idx + 1]
    return reordered


def format_move(cmd, x=None, y=None, z=None, f=None, comment=None):
    parts = [cmd]
    if x is not None:
        parts.append(f"X{x:.3f}")
    if y is not None:
        parts.append(f"Y{y:.3f}")
    if z is not None:
        parts.append(f"Z{z:.3f}")
    if f is not None:
        parts.append(f"F{f:.0f}")
    line = " ".join(parts)
    if comment:
        line += f"  ; {comment}"
    return line


def generate_gcode(toolpaths, params: WAAMParams, bead_width: float, layer_height: float,
                    header_comment: str = "WAAM slicer output",
                    machine_limits: MachineLimits = None,
                    skip_bounds_check: bool = False,
                    job_stats: JobStats = None,
                    voltage: float = 28.0, current: float = 180.0):
    """
    toolpaths: list[LayerToolpath] (from toolpath.py), one per layer, in print order.

    Raises GCodeGenerationError if the toolpath violates machine travel limits
    (unless skip_bounds_check=True) or contains non-finite coordinates.

    Returns: str (full G-code file contents)
    """
    if not toolpaths:
        raise GCodeGenerationError(
            "generate_gcode() received an empty toolpath list - refusing to write a "
            "G-code file with zero layers. If this is expected, this guard exists "
            "specifically to stop that from happening silently."
        )

    total_beads = sum(len(tp.beads) for tp in toolpaths)
    if total_beads == 0:
        raise GCodeGenerationError(
            f"All {len(toolpaths)} layer(s) produced zero beads - refusing to write an "
            f"empty G-code file. Check toolpath planning warnings (summarize_warnings()) "
            f"to see why - likely min_feature_area is too aggressive or bead_width doesn't "
            f"fit the part's smallest feature."
        )

    if machine_limits is None:
        machine_limits = MachineLimits()

    if not skip_bounds_check:
        validate_bounds(toolpaths, machine_limits, params.z_hop, params.safe_z)

    lines = []
    lines.append(f"; {header_comment}")
    lines.append(f"; Generated by waam_slicer for Digital Dream DDCS V2.1 + Ador Champ Pulse 500")
    lines.append(f";")
    lines.append(f"; === JOB STATISTICS ===")
    lines.append(f"; Layers:              {len(toolpaths)}   Layer height: {layer_height}mm   Bead width: {bead_width}mm")
    lines.append(f"; Total beads:         {job_stats.total_beads if job_stats else '?'}")
    lines.append(f"; Travel speed:        {params.travel_speed} mm/min   Weld speed: {params.weld_speed} mm/min"
                 f"   (machine max: {machine_limits.max_feed_rate} mm/min)")
    if job_stats:
        lines.append(f";")
        lines.append(f"; --- Time Breakdown ---")
        lines.append(f"; Weld time:           {_format_time(job_stats.weld_time_s)}"
                     f"   ({job_stats.weld_distance_mm:.0f} mm path)")
        lines.append(f"; Travel time:         {_format_time(job_stats.travel_time_s)}"
                     f"   (~{job_stats.travel_distance_mm:.0f} mm rapids)")
        lines.append(f"; Arc dwell time:      {_format_time(job_stats.arc_dwell_s)}"
                     f"   ({job_stats.total_beads} beads × {params.arc_on_dwell}s on + {params.arc_off_dwell}s off)")
        if params.interlayer_dwell > 0:
            lines.append(f"; Interlayer dwell:    {_format_time(job_stats.interlayer_dwell_s)}"
                         f"   ({params.interlayer_dwell}s × non-empty layers)")
        lines.append(f"; TOTAL EST. BUILD:    {_format_time(job_stats.total_build_time_s)}")
        lines.append(f";")
        lines.append(f"; --- Wire / Material Consumption ---")
        lines.append(f"; Wire from spool:     {job_stats.wire_length_m:.1f} m   ({job_stats.wire_mass_g:.0f} g)")
        lines.append(f"; Deposited volume:    {job_stats.deposited_volume_mm3:.0f} mm³"
                     f"   ({job_stats.deposited_volume_mm3/1000:.1f} cm³)")
        lines.append(f"; Mass deposition:     {job_stats.mass_deposition_rate_kg_hr:.3f} kg/hr"
                     f"   (DR = η·Q·ρ)")
        lines.append(f";")
        lines.append(f"; --- Heat Input ---")
        lines.append(f"; Arc power:           P = V·I = {job_stats.arc_power_w:.0f} W")
        lines.append(f"; Net heat power:      P_net = η·V·I = {job_stats.net_heat_power_w:.0f} W")
        lines.append(f"; Linear heat input:   HI = P_net/V_torch = {job_stats.heat_input_j_per_mm:.1f} J/mm")
        lines.append(f"; Mass feed/length:    MFL = {job_stats.mass_feed_per_mm:.6f} kg/mm")
    lines.append(f";")
    lines.append(f"; Machine limits: X[{machine_limits.x_min},{machine_limits.x_max}]"
                 f"  Y[{machine_limits.y_min},{machine_limits.y_max}]"
                 f"  Z[{machine_limits.z_min},{machine_limits.z_max}]"
                 + ("  (bounds check skipped)" if skip_bounds_check else "  (bounds checked OK)"))
    lines.append(f"; Arc trigger: {params.arc_on_mcode} (on) / {params.arc_off_mcode} (off) - "
                 f"CONFIRM against traced wiring before running on hardware, see README")
    lines.append(";")
    lines.append("; !!! FIRST RUN ON THIS FILE: jog Z well clear of the part and the table,")
    lines.append("; and run with the welder's main power OFF (or gas/wire disconnected) to")
    lines.append("; watch a full dry run and confirm every arc-on/arc-off happens where you")
    lines.append("; expect, before ever running this with the arc live. See README.")
    lines.append("")
    if job_stats and job_stats.per_layer_z:
        lines.append("; === LAYER PROCESS TABLE ===")
        lines.append("; Layer | Z(mm)  | Weld(s) | Beads | Dist(mm) | HI(J/mm) | Est.T(°C) | Cool(s)")
        lines.append("; ------+--------+---------+-------+----------+----------+-----------+-------")
        for i in range(len(job_stats.per_layer_z)):
            z = job_stats.per_layer_z[i]
            ws = job_stats.per_layer_weld_s[i] if i < len(job_stats.per_layer_weld_s) else 0
            bd = job_stats.per_layer_beads[i] if i < len(job_stats.per_layer_beads) else 0
            wd = job_stats.per_layer_weld_dist[i] if i < len(job_stats.per_layer_weld_dist) else 0
            hi = job_stats.heat_input_j_per_mm
            cool = params.interlayer_dwell if bd > 0 else 0
            lines.append(f"; {i:5d} | {z:6.1f} | {ws:7.1f} | {bd:5d} | {wd:8.1f} | {hi:8.1f} | {'---':>9} | {cool:5.1f}")
        lines.append(";")
    lines.append("")
    lines.append("G21          ; units = mm")
    lines.append("G90          ; absolute positioning")
    lines.append("; NOTE: no G28 (auto-home) here - this machine's home position is set by")
    lines.append("; the operator via the MPG pendant/panel before loading this file. Confirm")
    lines.append("; work zero (G92 or DDCS's own zero-set) is where you expect before AUTO start.")
    lines.append(format_move("G0", z=params.safe_z, f=params.travel_speed, comment="lift to safe Z"))
    lines.append("")

    for layer_tp in toolpaths:
        lines.append(f"; ---- Layer {layer_tp.index}  Z={layer_tp.z:.3f} ----")

        if layer_tp.warnings:
            for w in layer_tp.warnings:
                lines.append(f"; [warning] {w}")

        if not layer_tp.beads:
            lines.append("; (no beads planned for this layer - skipped)")
            lines.append("")
            continue

        for bi, bead in enumerate(layer_tp.beads):
            if not bead.points:
                continue

            points = _apply_lead_in_out(bead.points, params.lead_in_out)
            start_x, start_y = points[0]
            _check_finite_xy(start_x, start_y, context=f"layer {layer_tp.index} bead {bi} start")

            lines.append(f"; -- bead {bi} ({bead.kind}), {len(points)} pts --")
            lines.append(format_move("G0", z=layer_tp.z + params.z_hop, f=params.travel_speed,
                                      comment="z-hop for travel"))
            lines.append(format_move("G0", x=start_x, y=start_y, f=params.travel_speed,
                                      comment="travel to bead start"))
            lines.append(format_move("G0", z=layer_tp.z, f=params.travel_speed,
                                      comment="plunge to layer Z"))

            wf_on = wire_feed_on_line(params)
            if wf_on:
                lines.append(wf_on)
            lines.append(arc_on_line(params))
            if params.arc_on_dwell > 0:
                lines.append(f"G4 P{int(params.arc_on_dwell * 1000)}  ; dwell for arc stabilization")

            for (x, y) in points[1:]:
                _check_finite_xy(x, y, context=f"layer {layer_tp.index} bead {bi}")
                lines.append(format_move("G1", x=x, y=y, f=params.weld_speed))

            if params.arc_off_dwell > 0:
                lines.append(f"G4 P{int(params.arc_off_dwell * 1000)}  ; dwell to fill crater before arc off")
            lines.append(arc_off_line(params))
            wf_off = wire_feed_off_line(params)
            if wf_off:
                lines.append(wf_off)

        if params.interlayer_dwell > 0:
            lines.append(format_move("G0", z=params.safe_z, f=params.travel_speed,
                                      comment="lift clear before interlayer dwell"))
            lines.append(f"G4 P{int(params.interlayer_dwell * 1000)}  ; interlayer cooling dwell")

        lines.append("")

    lines.append("; ---- End of job ----")
    lines.append(arc_off_line(params) + "  (redundant safety arc-off in case last bead's off didn't register)")
    lines.append(format_move("G0", z=params.safe_z, f=params.travel_speed, comment="lift to safe Z"))
    lines.append("; NOTE: no G28 here either - job ends at safe Z above the part, does not")
    lines.append("; auto-return to machine home. Return home manually via the pendant/panel.")
    lines.append("M2           ; program end (DDCS: stops AUTO mode)")

    return "\n".join(lines)


def write_gcode_file(gcode_str: str, out_path: str):
    with open(out_path, "w") as f:
        f.write(gcode_str)
    return out_path
