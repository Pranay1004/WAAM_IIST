"""
beamWeldFoam OpenFOAM 2412 Case Generator for WAAM Melt Pool CFD

Generates a complete, ready-to-run beamWeldFoam case from the
WAAM-Framework GUI parameter dictionary. Produces all OpenFOAM
case directories (system/, constant/, initial/) with:
  - Diverging-oscillating-Gaussian heat source via MELTING dict
  - Two-phase VOF (metal + atmosphere) transport properties
  - Marangoni surface tension and vapor recoil pressure
  - Darcy mushy-zone damping
  - Local blockMesh window auto-sized from GUI geometry / CAD / G-code
  - Automated Allrun/Allclean execution scripts
  - Built-in postprocessing (postprocess_cfd.py, ParaView loader, case.foam)

Parameter mapping:
  GUI I_weld * U_weld * eta_arc  ->  MELTING/HS_Q  [W]
  GUI v_scan                     ->  MELTING/HS_velocity  [m/s] (beam travels +Z)
  GUI bG / 2                     ->  MELTING/HS_a  [m] (port convention, see note 4)
  gas_gap (CFD gas layer)        ->  MELTING/focY  [m] (metal surface plane)

PHYSICS-CORRECTNESS NOTES (verified against
beamweldfoam/beamWeldFoam_ported solver source + cases/waam_arc + upstream
tutorials/ArcCase — NOT blindly copied):
  1. Coordinate frame is load-bearing. The solver (DivergingOscillatingGaussian.H)
     travels the beam along +Z (dz = z - v*t - lg), scans rays over (x,z), and
     deposits at the MINIMUM-Y metal cell (first alpha1 > cutoff). Therefore the
     domain MUST be: X transverse, Y vertical with gas below (0..gap) and metal
     above (gap..gap+hz), Z travel. Upstream ArcCase uses the identical layout.
     Do NOT "fix" gas-below-metal: flipping it moves deposition to the wrong face.
  2. Gravity sign: port uses (0 -9.81 0), upstream used (0 +9.81 0). At the
     millisecond timescales of VOF melt-pool runs (fall distance ~1 nm) gravity
     is negligible either way; the port value is kept.
  3. Q_ramp_rate is a power DECAY [W/s] applied AFTER tshift
     (Qnow = max(Q - rate*(t-tshift), 0)), not a ramp-up. "auto" = 10*Q mirrors
     the upstream pattern and is inert while tshift > endTime. Defaults
     (tshift 1.0 s, endTime 5e-4 s) give constant power. A None ramp is treated
     as auto, never formatted raw (previously crashed str.format on NoneType).
  4. HS_a is the 1/e^2 beam radius of the solver's exp(-2r^2/a^2) Gaussian,
     while Goldak bG belongs to an exp(-3r^2/b^2) profile. The port derives
     HS_a = bG/2 from DEFAULT.edp; this generator follows the port convention.
  5. Y_R = 0.15 and focY = gas-gap thickness are the port-tuned values for the
     WAAM arc case (upstream default Y_R 0.2 is kept available via GUI).
  6. Deliberate deviations from the port (documented, not accidental):
     sigma/dsigma_dT are literature-strength (port values 0.8/-1e-05 suppress
     Marangoni ~40x); Tliquidus/LatentHeat match material_database.json so FEM
     and CFD agree on phase-change energy.
  7. Cost model (why a LOCAL window, not the full plate): per step the ray
     tracer costs O(Nx*Nz*Ny) mesh.findCell calls, and VOF needs ~1 mm cells to
     resolve a 7 mm pool. Full-plate 200 mm at 1 mm is unaffordable on an M4 Air;
     the local window (default 30 x (3+10) x 40 mm ~ 15k cells, 0.5 ms) runs in
     minutes. A cell-budget guard auto-coarsens instead of generating heavy cases.
"""

import os
import math
import json
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================
# TWO-PHASE MATERIAL PROPERTIES FOR beamWeldFoam
# ============================================================
# Keyed by material name matching material_database.json alloy keys.
# Values needed by beamWeldFoam transportProperties that are NOT
# in the existing FEM-only database.

VOF_MATERIAL_PROPERTIES = {
    "SS308L": {
        "nu_liquid":        0.9e-06,
        "rho":              7850,
        "cp_liquid":        540,
        "cp_solid":         500,
        "kappa_liquid":     23.0,
        "kappa_solid":      15.0,
        "T_solidus":        1675,       # 1402 C
        "T_liquidus":       1730,       # 1457 C
        "latent_heat":      270000,
        "beta":             1.6e-5,
        "sigma":            1.72,
        "dsigma_dT":        -4.3e-4,
        "T_vap":            3135,
        "Mm":               0.0558,
        "latent_heat_vap":  6.09e6,
    },
    "SS316L": {
        "nu_liquid":        0.9e-06,
        "rho":              7900,
        "cp_liquid":        530,
        "cp_solid":         500,
        "kappa_liquid":     22.0,
        "kappa_solid":      14.0,
        "T_solidus":        1648,       # 1375 C
        "T_liquidus":       1713,       # 1440 C
        "latent_heat":      260000,
        "beta":             1.8e-5,
        "sigma":            1.70,
        "dsigma_dT":        -4.3e-4,
        "T_vap":            3135,
        "Mm":               0.0558,
        "latent_heat_vap":  6.09e6,
    },
    "316L": {
        "nu_liquid":        0.9e-06,
        "rho":              7900,
        "cp_liquid":        530,
        "cp_solid":         500,
        "kappa_liquid":     22.0,
        "kappa_solid":      14.0,
        "T_solidus":        1648,
        "T_liquidus":       1713,
        "latent_heat":      260000,
        "beta":             1.8e-5,
        "sigma":            1.70,
        "dsigma_dT":        -4.3e-4,
        "T_vap":            3135,
        "Mm":               0.0558,
        "latent_heat_vap":  6.09e6,
    },
    "308L": {
        "nu_liquid":        0.9e-06,
        "rho":              7850,
        "cp_liquid":        540,
        "cp_solid":         500,
        "kappa_liquid":     23.0,
        "kappa_solid":      15.0,
        "T_solidus":        1675,
        "T_liquidus":       1730,
        "latent_heat":      270000,
        "beta":             1.6e-5,
        "sigma":            1.72,
        "dsigma_dT":        -4.3e-4,
        "T_vap":            3135,
        "Mm":               0.0558,
        "latent_heat_vap":  6.09e6,
    },
    "Ti-6Al-4V": {
        "nu_liquid":        1.1e-06,
        "rho":              4430,
        "cp_liquid":        700,
        "cp_solid":         526,
        "kappa_liquid":     25.0,
        "kappa_solid":      6.7,
        "T_solidus":        1877,       # 1604 C
        "T_liquidus":       1933,       # 1660 C
        "latent_heat":      370000,
        "beta":             8.6e-6,
        "sigma":            1.65,
        "dsigma_dT":        -2.6e-4,
        "T_vap":            3560,
        "Mm":               0.0479,
        "latent_heat_vap":  8.88e6,
    },
    "ER70S-6": {
        "nu_liquid":        0.8e-06,
        "rho":              7870,
        "cp_liquid":        800,
        "cp_solid":         500,
        "kappa_liquid":     27.0,
        "kappa_solid":      45.0,
        "T_solidus":        1723,       # 1450 C
        "T_liquidus":       1793,       # 1520 C
        "latent_heat":      270000,
        "beta":             1.2e-5,
        "sigma":            1.85,
        "dsigma_dT":        -4.3e-4,
        "T_vap":            3135,       # 2862 C
        "Mm":               0.0558,
        "latent_heat_vap":  6.09e6,
    },
    "IN625": {
        "nu_liquid":        1.0e-06,
        "rho":              8440,
        "cp_liquid":        700,
        "cp_solid":         410,
        "kappa_liquid":     30.0,
        "kappa_solid":      10.0,
        "T_solidus":        1563,       # 1290 C
        "T_liquidus":       1623,       # 1350 C
        "latent_heat":      290000,
        "beta":             1.1e-5,
        "sigma":            1.85,
        "dsigma_dT":        -3.0e-4,
        "T_vap":            3180,
        "Mm":               0.0610,
        "latent_heat_vap":  5.90e6,
    },
    "AlSi10Mg": {
        "nu_liquid":        0.55e-06,
        "rho":              2670,
        "cp_liquid":        1000,
        "cp_solid":         750,
        "kappa_liquid":     90.0,
        "kappa_solid":      130.0,
        "T_solidus":        830,        # 557 C
        "T_liquidus":       869,        # 596 C
        "latent_heat":      390000,
        "beta":             2.1e-5,
        "sigma":            0.87,
        "dsigma_dT":        -3.5e-4,
        "T_vap":            2792,       # 2519 C (Al)
        "Mm":               0.0272,
        "latent_heat_vap":  10.9e6,
    },
    "Pure Al (Al1xxx)": {
        "nu_liquid":        0.55e-06,
        "rho":              2700,
        "cp_liquid":        1180,
        "cp_solid":         904,
        "kappa_liquid":     95.0,
        "kappa_solid":      222.0,
        "T_solidus":        916,        # 643 C
        "T_liquidus":       933,        # 660 C
        "latent_heat":      397000,
        "beta":             2.3e-5,
        "sigma":            0.87,
        "dsigma_dT":        -3.5e-4,
        "T_vap":            2792,       # 2519 C
        "Mm":               0.0270,
        "latent_heat_vap":  10.9e6,
    },
    "Pure Al": {
        "nu_liquid":        0.55e-06,
        "rho":              2700,
        "cp_liquid":        1180,
        "cp_solid":         904,
        "kappa_liquid":     95.0,
        "kappa_solid":      222.0,
        "T_solidus":        916,
        "T_liquidus":       933,
        "latent_heat":      397000,
        "beta":             2.3e-5,
        "sigma":            0.87,
        "dsigma_dT":        -3.5e-4,
        "T_vap":            2792,
        "Mm":               0.0270,
        "latent_heat_vap":  10.9e6,
    },
}

# Case-insensitive / alias resolution so GUI and FEM material names always map
# to a VOF entry (previously _get_metal_props raised KeyError on its own
# default argument because "ER70S-6" was not a table key).
VOF_MATERIAL_ALIASES = {
    "ER70S6": "ER70S-6",
    "ER70S-6": "ER70S-6",
    "ER70S": "ER70S-6",
    "SG2": "ER70S-6",
    "308L": "SS308L",
    "SS308L": "SS308L",
    "ER308L": "SS308L",
    "316L": "SS316L",
    "SS316L": "SS316L",
    "ER316L": "SS316L",
    "TI-6AL-4V": "Ti-6Al-4V",
    "TI64": "Ti-6Al-4V",
    "GRADE 5": "Ti-6Al-4V",
    "IN625": "IN625",
    "INCONEL 625": "IN625",
    "ALLOY 625": "IN625",
    "ALSI10MG": "AlSi10Mg",
    "AL-SI10MG": "AlSi10Mg",
    "PURE AL (AL1XXX)": "Pure Al (Al1xxx)",
    "PURE AL": "Pure Al",
    "AL1100": "Pure Al (Al1xxx)",
    "1100": "Pure Al (Al1xxx)",
}

# Phase 2: Air (atmosphere above melt pool)
# NOTE: rho=0.1 (not real air 1.2) is a deliberate stabilization fudge:
# beamWeldFoam VOF with density ratio 78500:1 is unstable; 7850:0.1 keeps
# the gas inertialess while preserving thermal Marangoni coupling.
AIR_PROPERTIES = {
    "nu_liquid":        1.48e-05,
    "rho":              0.1,
    "cp_liquid":        500,
    "cp_solid":         500,
    "kappa_liquid":     0.016,
    "kappa_solid":      0.016,
    "T_solidus":        1.0,
    "T_liquidus":       10.0,
    "latent_heat":      1,
    "beta":             1.0e-5,
}


# ============================================================
# OPENFOAM HEADER TEMPLATE
# ============================================================
OF_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  2412
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {of_class};
    {location_line}object      {of_object};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _header(of_class: str, of_object: str, location: str = "") -> str:
    """Generate a standard OpenFOAM file header."""
    loc = f'location    "{location}";\n    ' if location else ""
    return OF_HEADER.format(of_class=of_class, of_object=of_object, location_line=loc)


def _get_metal_props(params: Dict[str, Any]) -> Dict[str, Any]:
    """Look up two-phase metal properties from material key.

    Resolution order: exact key -> case-insensitive key -> alias table ->
    material_database.json category-based match -> ER70S-6 fallback.
    Never raises KeyError (the old eager ``dict[key]`` default threw on every
    call because "ER70S-6" was not itself a table key).
    """
    mat_key = str(params.get("material_dep") or params.get("material") or "ER70S-6")
    if mat_key in VOF_MATERIAL_PROPERTIES:
        return VOF_MATERIAL_PROPERTIES[mat_key]

    upper = mat_key.strip().upper()
    if upper in VOF_MATERIAL_ALIASES:
        return VOF_MATERIAL_PROPERTIES[VOF_MATERIAL_ALIASES[upper]]

    for key in VOF_MATERIAL_PROPERTIES:
        if key.upper() == upper:
            return VOF_MATERIAL_PROPERTIES[key]

    # Last resort: match by FEM material database category (SS/Ti/Al/Ni/steel).
    try:
        from core.multimaterial.alloy_metallurgy import get_alloy_category
        cat = get_alloy_category(mat_key)
        cat_default = {
            "Austenitic Stainless Steel": "SS308L",
            "Titanium Alloy": "Ti-6Al-4V",
            "Nickel Superalloy": "IN625",
            "Aluminium Alloy": "AlSi10Mg",
            "Carbon Steel": "ER70S-6",
        }.get(cat, "ER70S-6")
        return VOF_MATERIAL_PROPERTIES[cat_default]
    except Exception:
        return VOF_MATERIAL_PROPERTIES["ER70S-6"]


def _num(cfd: Dict[str, Any], key: str, default: float) -> float:
    """Fetch a numeric CFD option, tolerating None / strings / garbage.

    None (e.g. GUI 'auto' sentinel) falls back to the default instead of
    crashing downstream ':.6g' formatting (unsupported format on NoneType).
    """
    try:
        v = cfd.get(key, default)
        if v is None:
            return float(default)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "auto", "none", "default"):
                return float(default)
            return float(s)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def derive_cfd_window(params: Dict[str, Any], cfd_params: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the LOCAL CFD window (travel x transverse) from inputs.

    Robust pipeline, cheapest-first:
      parametric -> GUI bead dims (4x bead width, 40 mm travel default)
      gcode_file -> G-code toolpath bounding box (pure-python parse, fast)
      cad_file   -> CAD bounding box (trimesh for STL; Gmsh OCC bbox-only for
                    STEP/IGES — import only, no volume meshing, fast)
    Everything is clamped to sane ranges and wrapped in try/except so a bad
    file can never break case generation (falls back to bead-based defaults).
    Units: params are SI (m); returns meters.
    """
    def _clamp(v, lo, hi, fb):
        try:
            v = float(v)
            if not (v > 0):
                return fb
            return min(max(v, lo), hi)
        except (TypeError, ValueError):
            return fb

    ty_dep = float(params.get("ty_dep", 0.008) or 0.008)
    if ty_dep > 1.0:  # tolerate mm input
        ty_dep /= 1000.0

    # Explicit GUI overrides win when positive numbers.
    travel = _num(cfd_params, "travel_length_mm", -1.0)
    width = _num(cfd_params, "domain_width_mm", -1.0)
    travel_m = travel / 1000.0 if travel > 0 else None
    width_m = width / 1000.0 if width > 0 else None

    src = ""
    try:
        mesh_src = str(params.get("mesh_source", "parametric")).lower()
        cand = (params.get("gcode_file") or params.get("mesh_file")
                or params.get("stl_file") or "")
        if travel_m is None or width_m is None:
            if mesh_src == "gcode_file" or str(cand).lower().endswith((".gcode", ".nc")):
                if cand and os.path.exists(str(cand)):
                    from mesh.gcode_parser import parse_gcode
                    tp = parse_gcode(str(cand))
                    bb = tp.bounding_box or {}
                    xs = max(float(bb.get("x_max", 0)) - float(bb.get("x_min", 0)), 0.0)
                    ys = max(float(bb.get("y_max", 0)) - float(bb.get("y_min", 0)), 0.0)
                    if travel_m is None and xs > 0:
                        travel_m = _clamp(xs / 1000.0, 0.020, 0.080, 0.040)
                        src = "gcode-bbox"
                    if width_m is None and ys > 0:
                        width_m = _clamp(ys / 1000.0, 0.015, 0.060, 0.030)
                        src = "gcode-bbox"
            elif mesh_src == "cad_file" and cand and os.path.exists(str(cand)):
                suf = Path(str(cand)).suffix.lower()
                spans = None
                if suf in (".stl", ".ply", ".obj"):
                    import trimesh
                    m = trimesh.load(str(cand), force="mesh")
                    if hasattr(m, "to_mesh"):
                        try:
                            m = m.to_mesh()
                        except Exception:
                            pass
                    import numpy as _np
                    v = _np.asarray(getattr(m, "vertices", []))
                    if v.size:
                        # STL units are mm in this framework's slicer pipeline.
                        spans = (v.max(axis=0) - v.min(axis=0)) / 1000.0
                elif suf in (".step", ".stp", ".iges", ".igs", ".brep"):
                    import gmsh
                    gmsh.initialize(interruptible=False)
                    try:
                        gmsh.option.setNumber("General.Terminal", 0)
                        gmsh.model.add("cfd_bbox_probe")
                        gmsh.model.occ.importShapes(str(cand))
                        gmsh.model.occ.synchronize()
                        ents = gmsh.model.getEntities(3) or gmsh.model.getEntities(2)
                        if ents:
                            x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(
                                ents[0][0], ents[0][1])
                            spans = (x1 - x0, y1 - y0, z1 - z0)  # already SI (m)
                    finally:
                        try:
                            gmsh.finalize()
                        except Exception:
                            pass
                if spans is not None and len(spans) >= 2:
                    if travel_m is None and float(spans[0]) > 0:
                        travel_m = _clamp(float(spans[0]), 0.020, 0.080, 0.040)
                        src = "cad-bbox"
                    if width_m is None and float(spans[1]) > 0:
                        width_m = _clamp(float(spans[1]), 0.015, 0.060, 0.030)
                        src = "cad-bbox"
    except Exception:
        pass

    if travel_m is None:
        travel_m, src = 0.040, (src or "default")
    if width_m is None:
        width_m = _clamp(max(4.0 * ty_dep, 0.020), 0.015, 0.060, 0.030)
    return {"travel_m": float(travel_m), "width_m": float(width_m), "source": src}


# ============================================================
# MAIN CASE GENERATOR
# ============================================================

def setup_beamweldfoam_case(
    case_dir: str,
    params: Dict[str, Any],
    cfd_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a complete beamWeldFoam OpenFOAM 2412 case.

    Args:
        case_dir:   Path to the case directory to create
        params:     Standard WAAM-Framework parameter dictionary (from GUI)
        cfd_params: Optional CFD-specific overrides (endTime, maxCo, mesh resolution, etc.)

    Returns:
        Absolute path to the created case directory
    """
    if cfd_params is None:
        cfd_params = {}

    case = Path(case_dir)
    dir_system = case / "system"
    dir_constant = case / "constant"
    dir_initial = case / "initial"

    for d in [dir_system, dir_constant, dir_initial]:
        d.mkdir(parents=True, exist_ok=True)

    metal = _get_metal_props(params)
    metal_name = str(params.get("material_dep", params.get("material", "ER70S-6")))
    notes: List[str] = []

    def _pm(key: str, default: float) -> float:
        """Positive-float param fetch with mm tolerance (values >1.0 are mm)."""
        try:
            v = float(params.get(key, default))
        except (TypeError, ValueError):
            v = float(default)
        if not (v > 0):
            v = float(default)
        if v > 1.0 and key in ("hz_base", "ty_dep", "h_bead", "bG",
                               "lx", "ly_base"):
            v /= 1000.0
        return v

    # --- Local CFD window (meters). Frame: X transverse, Y vertical with
    # gas 0..gap and metal gap..gap+hz, Z travel (solver travels +Z). ---
    hz_base_m = _pm("hz_base", 0.010)
    win = derive_cfd_window(params, cfd_params)
    travel_m = win["travel_m"]
    width_m = win["width_m"]
    notes.append(
        f"CFD window source={win['source']}: "
        f"travel(Z)={travel_m*1000.0:.0f}mm transverse(X)={width_m*1000.0:.0f}mm"
    )

    gas_gap_m = _num(cfd_params, "atmosphere_height", 0.003)
    if gas_gap_m > 1.0:  # tolerate mm input
        gas_gap_m /= 1000.0
    gas_gap_m = min(max(gas_gap_m, 0.001), 0.020)
    y_top_m = gas_gap_m + hz_base_m  # metal surface (deposition plane) at gas_gap

    # --- Process ---
    I_weld = _pm("I_weld", 200.0)
    U_weld = _pm("U_weld", 24.0)
    try:
        eta_arc = float(params.get("eta_arc", 0.8))
    except (TypeError, ValueError):
        eta_arc = 0.8
    eta_arc = min(max(eta_arc, 0.05), 1.0)
    v_scan = _pm("v_scan", 10.0e-3)
    if v_scan > 1.0:  # tolerate mm/s input
        v_scan /= 1000.0
    Q_arc = I_weld * U_weld * eta_arc         # [W]
    try:
        T_amb_K = float(params.get("T_amb", 25.0)) + 273.15
    except (TypeError, ValueError):
        T_amb_K = 298.15

    # --- MELTING dict parameters ---
    # Beam radius follows the port convention HS_a = bG/2 (see note 4).
    bG_m = _pm("bG", 7.0e-3)
    HS_a = bG_m / 2.0
    HS_velocity = v_scan
    HS_Q = Q_arc
    HS_bg = 0.0
    HS_lg = 0.0
    HS_deposition_cutoff = _num(cfd_params, "deposition_cutoff", 0.5)
    HS_deposition_cutoff = min(max(HS_deposition_cutoff, 0.01), 0.99)

    # Focus on the metal surface plane (min-Y metal = deposition plane).
    focY = gas_gap_m
    foc_shift_vel = _num(cfd_params, "foc_shift_vel", 0.0)
    Y_R = _num(cfd_params, "Y_R", 0.15)
    if Y_R <= 0:
        Y_R = 0.15

    # Oscillation
    osc_amp = _num(cfd_params, "oscillation_amplitude", 0.0)
    osc_freq = _num(cfd_params, "oscillation_frequency", 0.0)

    # Power ramp (DECAY after tshift; None/auto -> 10x arc power, upstream pattern)
    Q_ramp_raw = cfd_params.get("Q_ramp_rate", None)
    if Q_ramp_raw is None or (isinstance(Q_ramp_raw, str)
                              and Q_ramp_raw.strip().lower() in ("", "auto", "none", "default")):
        Q_ramp_rate = Q_arc * 10.0
        notes.append(f"Q_ramp auto: {Q_ramp_rate:.0f} W/s (=10x Q_arc, upstream pattern)")
    else:
        Q_ramp_rate = _num(cfd_params, "Q_ramp_rate", Q_arc * 10.0)
    tshift = _num(cfd_params, "tshift", 1.0)

    # Temperature correctors
    minTCorr = int(_num(cfd_params, "minTempCorrector", 1))
    maxTCorr = int(_num(cfd_params, "maxTempCorrector", 150))
    epsTol = _num(cfd_params, "epsilonTolerance", 1e-6)
    epsRel = _num(cfd_params, "epsilonRelaxation", 0.95)

    # Time control (millisecond VOF regime — port values, NOT seconds-scale)
    end_time = _num(cfd_params, "endTime", 5.0e-4)
    end_time = min(max(end_time, 1.0e-6), 0.05)
    delta_t = _num(cfd_params, "deltaT", 1.0e-7)
    write_interval = cfd_params.get("writeInterval", None)
    try:
        write_interval = float(write_interval) if write_interval is not None else end_time / 10.0
    except (TypeError, ValueError):
        write_interval = end_time / 10.0
    if not (write_interval > 0):
        write_interval = end_time / 10.0
    max_co = _num(cfd_params, "maxCo", 0.2)
    max_alpha_co = _num(cfd_params, "maxAlphaCo", 0.2)
    max_delta_t = _num(cfd_params, "maxDeltaT", 1.0e-5)

    if tshift <= end_time and Q_ramp_rate > 0:
        t_zero = tshift + HS_Q / max(Q_ramp_rate, 1e-30)
        notes.append(
            f"WARNING: tshift ({tshift:.6g}s) <= endTime ({end_time:.6g}s): "
            f"power decays to zero at ~{t_zero:.6g}s. Set tshift > endTime for "
            f"constant power."
        )
    est_steps = end_time / max(delta_t, 1e-12)
    notes.append(f"Estimated steps ~{est_steps:.0f} (endTime/deltaT)")

    # Mesh resolution with compute-budget guard (M4 Air / i9 friendly)
    cell_size_mm = _num(cfd_params, "cell_size_mm", 1.0)
    cell_size_mm = min(max(cell_size_mm, 0.25), 2.5)
    max_cells = int(_num(cfd_params, "max_cells", 60000))
    max_cells = min(max(int(max_cells), 4000), 500000)
    cell_size_m = cell_size_mm * 1e-3
    nx = max(int(round(width_m / cell_size_m)), 4)
    ny = max(int(round(y_top_m / cell_size_m)), 4)
    nz = max(int(round(travel_m / cell_size_m)), 4)
    while nx * ny * nz > max_cells and cell_size_m < 2.5e-3:
        cell_size_m *= 1.5
        cell_size_mm = cell_size_m * 1e3
        nx = max(int(round(width_m / cell_size_m)), 4)
        ny = max(int(round(y_top_m / cell_size_m)), 4)
        nz = max(int(round(travel_m / cell_size_m)), 4)
        notes.append(f"Cell budget: coarsened to {cell_size_mm:.2f}mm to stay <={max_cells} cells")
    notes.append(f"Mesh {nx}x{ny}x{nz}={nx*ny*nz} cells at {cell_size_mm:.2f}mm")
    est_findcell = (nx * nz) * ny * est_steps
    if est_findcell > 2.0e8:
        notes.append(
            f"WARNING: ray-tracer cost ~{est_findcell:.1e} findCell ops — heavy for "
            f"M4 Air; shorten endTime or coarsen cells."
        )

    damper_switch = "true" if cfd_params.get("damperSwitch", False) else "false"

    # ================================================================
    # WRITE ALL CASE FILES (port-faithful layout + values)
    # ================================================================

    _write_control_dict(dir_system, end_time, delta_t, write_interval,
                        max_co, max_alpha_co, max_delta_t)

    _write_block_mesh_dict(dir_system, width_m, gas_gap_m, y_top_m,
                           travel_m, nx, ny, nz)

    _write_fv_schemes(dir_system)

    _write_fv_solution(dir_system,
                       HS_a, HS_bg, HS_velocity, HS_lg, HS_Q,
                       HS_deposition_cutoff, osc_amp, osc_freq,
                       Y_R, focY, foc_shift_vel, Q_ramp_rate, tshift,
                       minTCorr, maxTCorr, epsTol, epsRel, damper_switch)

    _write_set_fields_dict(dir_system, width_m, gas_gap_m, y_top_m, travel_m,
                           metal_name)

    _write_transport_properties(dir_constant, metal, metal_name)

    _write_dynamic_mesh_dict(dir_constant)

    _write_gravity(dir_constant)

    _write_turbulence_properties(dir_constant)

    _write_initial_temperature(dir_initial, T_amb_K)
    _write_initial_velocity(dir_initial)
    _write_initial_alpha(dir_initial, metal_name)
    _write_initial_p_rgh(dir_initial)
    # NOTE: TRHS/sourceTerm are created at runtime by the solver (the port's
    # initial/ has no such files); do not invent them here.

    _write_allrun(case)
    _write_allclean(case)
    _write_case_foam(case)
    _write_readme(case, width_m, gas_gap_m, y_top_m, travel_m, HS_a,
                  HS_Q, HS_velocity, focY, Q_ramp_rate, tshift,
                  end_time, nx, ny, nz, cell_size_mm, notes)
    _write_cfd_post_scripts(case)

    # Write a parameter summary JSON for traceability
    summary = {
        "generator": "WAAM-Framework beamWeldFoam Case Generator",
        "solver": "beamWeldFoam",
        "material": params.get("material_dep", params.get("material", "ER70S-6")),
        "Q_arc_W": round(float(Q_arc), 6),
        "v_scan_m_s": round(float(v_scan), 9),
        "HS_a_m": round(float(HS_a), 9),
        "frame": "X transverse, Y vertical (gas 0..gap, metal gap..top), Z travel (+Z)",
        "domain_m": {"x_transverse": round(float(width_m), 6),
                     "y_total": round(float(y_top_m), 6),
                     "z_travel": round(float(travel_m), 6)},
        "gas_gap_m": round(float(gas_gap_m), 6),
        "substrate_m": round(float(hz_base_m), 6),
        "focY_m": round(float(focY), 6),
        "Q_ramp_rate_W_per_s": round(float(Q_ramp_rate), 6),
        "tshift_s": float(tshift),
        "mesh_cells": {"nx": nx, "ny": ny, "nz": nz, "total": nx * ny * nz},
        "cell_mm": round(float(cell_size_mm), 4),
        "endTime_s": float(end_time),
        "deltaT_s": float(delta_t),
        "cfd_params_echo": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in cfd_params.items()},
        "notes": notes,
    }
    with open(case / "case_parameters.json", "w") as f:
        json.dump(summary, f, indent=2)

    return str(case.resolve())


# ============================================================
# FILE WRITERS
# ============================================================

def _write_control_dict(d: Path, end_time, delta_t, write_interval,
                        max_co, max_alpha_co, max_delta_t):
    # Port-faithful: startFrom startTime, precision 8, compression off.
    # Millisecond regime: endTime ~5e-4, NOT seconds-scale.
    # Key meanings: adjustableRunTime + adjustTimeStep keep Courant numbers
    # (maxCo for flow, maxAlphaCo for the VOF interface) inside the bounds
    # below by shrinking deltaT (never above maxDeltaT); writeInterval sets
    # the saved time directories analysed by postprocess_cfd.py.
    with open(d / "controlDict", "w") as f:
        f.write(_header("dictionary", "controlDict", "system"))
        f.write(f"""
application     beamWeldFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         {end_time:.6g};

deltaT          {delta_t:.6g};

writeControl    adjustableRunTime;

writeInterval   {write_interval:.6g};

purgeWrite      0;

writeFormat     ascii;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   8;

runTimeModifiable yes;

adjustTimeStep  yes;

maxCo           {max_co};

maxAlphaCo      {max_alpha_co};

maxDeltaT       {max_delta_t:.6g};


// ************************************************************************* //
""")


def _write_block_mesh_dict(d: Path, width_m, gas_gap_m, y_top_m,
                           travel_m, nx, ny, nz):
    """Generate blockMeshDict (millimetres via scale 0.001; convertToMeters
    is deprecated in 2412 and warns on every mesh build).

    Frame (solver convention, port-faithful): X transverse, Y vertical with
    gas 0..gap and metal gap..top, Z travel. Patch names/faces intentionally
    match cases/waam_arc: 'lowerWall' is the TOP wall (y=max), 'atmosphere'
    the BOTTOM patch (y=0) — historic upstream naming, kept for solver compat.
    """
    xh = width_m * 1000.0 / 2.0
    yt = y_top_m * 1000.0
    zh = travel_m * 1000.0 / 2.0

    with open(d / "blockMeshDict", "w") as f:
        f.write(_header("dictionary", "blockMeshDict", "system"))
        f.write(f"""
// WAAM beamWeldFoam Case -- Auto-Generated Local Window
// X: transverse ({2*xh:.1f} mm), Y: vertical gas+metal ({yt:.1f} mm),
// Z: travel direction ({2*zh:.1f} mm). Beam travels +Z from domain centre.
// (Full-plate domains are unaffordable for VOF ray-tracing; this local
// window around the melt pool is the validated approach.)
//
// Patch legend (names are historic upstream labels, kept for solver compat):
//   lowerWall    = TOP wall at y=max (solid clamp lid, U=noSlip, T=zeroGradient)
//   atmosphere   = BOTTOM patch at y=0 (open gas boundary, do NOT wall it)
//   leftWall/rightWall = transverse far-field sides (X=min/max, U=slip)
//   frontAndBack = travel inlet/outlet faces (Z=min/max, U=slip)

scale 0.001;

vertices
(
    ({-xh:.4f} 0 {-zh:.4f})    // 0
    ({xh:.4f}  0 {-zh:.4f})    // 1
    ({xh:.4f}  {yt:.4f} {-zh:.4f})    // 2
    ({-xh:.4f} {yt:.4f} {-zh:.4f})    // 3
    ({-xh:.4f} 0 {zh:.4f})    // 4
    ({xh:.4f}  0 {zh:.4f})    // 5
    ({xh:.4f}  {yt:.4f} {zh:.4f})    // 6
    ({-xh:.4f} {yt:.4f} {zh:.4f})    // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    lowerWall
    {{
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }}

    atmosphere
    {{
        type patch;
        faces
        (
            (1 5 4 0)
        );
    }}

    frontAndBack
    {{
        type patch;
        faces
        (
            (0 3 2 1)
            (5 6 7 4)
        );
    }}

    leftWall
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}

    rightWall
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
);

mergePatchPairs
(
);


// ************************************************************************* //
""")


def _write_fv_schemes(d: Path):
    """Write validated fvSchemes matching the waam_arc reference case."""
    with open(d / "fvSchemes", "w") as f:
        f.write(_header("dictionary", "fvSchemes", "system"))
        f.write("""
ddtSchemes
{
    default         Euler;
}

d2dt2Schemes
{
    d2dt2(D) steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(rhoPhi,U)  Gauss linearUpwind grad(U);

    div((interpolate(epsilon1mask)*rhoPhi),U)    Gauss linearUpwind grad(U);

    div(phi,alpha)  Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;

    div(phi*cp,Temperature)   Gauss linearUpwind grad(Temperature);

    div(rhoPhi,epsilon1)   Gauss linear;

    div((interpolate(cp)*rhoPhi),Temperature)   Gauss upwind;

    div(rhoPhi,Temperature)    Gauss upwind;
    div(phi,Temperature)   Gauss upwind;

    div(phi,epsilon1) Gauss linear;

    div(rhoPhi,Le1) Gauss linear;

    div((interpolate(LatentHeat)*rhoPhi),epsilon1)    Gauss vanLeer;

}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}


// ************************************************************************* //
""")


def _write_fv_solution(d: Path,
                       HS_a, HS_bg, HS_velocity, HS_lg, HS_Q,
                       HS_deposition_cutoff, osc_amp, osc_freq,
                       Y_R, focY, foc_shift_vel, Q_ramp_rate, tshift,
                       minTCorr, maxTCorr, epsTol, epsRel, damper_switch):
    """Write fvSolution with full PIMPLE + MELTING sub-dictionary."""
    with open(d / "fvSolution", "w") as f:
        f.write(_header("dictionary", "fvSolution", "system"))
        f.write(f"""
solvers
{{
    // Volume-fraction (VOF) advection for the METAL phase:
    //   alpha.phase1 = 1 -> fully metal (phase1, see transportProperties),
    //   alpha.phase1 = 0 -> fully atmosphere gas (phase2 = air).
    "alpha.phase1.*"
    {{
        nAlphaCorr      3;
        nAlphaSubCycles 3;
        cAlpha          1.0;
        scAlpha         0.5;

        MULESCorr       yes;
        nLimiterIter    5;

        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0;
    }}

    p_rgh
    {{
        solver          GAMG;
        tolerance       1e-9;
        relTol          0.0;
        smoother        GaussSeidel;
        nPreSweeps      2;
        nPostSweeps     3;
        nFinestSweeps   2;
        cacheAgglomeration on;
        nCellsInCoarsestLevel 10;
        agglomerator    faceAreaPair;
        mergeLevels     1;
    }}

    p_rghFinal
    {{
        solver          PCG;
        preconditioner
        {{
            preconditioner  GAMG;
            tolerance       1e-9;
            relTol          0;
            nVcycles        5;
            smoother        GaussSeidel;
            nPreSweeps      2;
            nPostSweeps     3;
            nFinestSweeps   2;
            cacheAgglomeration on;
            nCellsInCoarsestLevel 10;
            agglomerator    faceAreaPair;
            mergeLevels     1;
        }}
        tolerance       1e-9;
        relTol          0;
        maxIter         25;
    }}

    "pcorr.*"
    {{
        solver          PCG;
        preconditioner
        {{
            preconditioner  GAMG;
            tolerance       1e-9;
            relTol          0;
            smoother        GaussSeidel;
        }}
        tolerance       1e-9;
        relTol          0;
        maxIter         100;
    }}

    U
    {{
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-9;
        relTol          0.0;
        nSweeps         5;
    }}

    UFinal
    {{
        $U;
        tolerance       1e-9;
        relTol          0;
    }}

    Temperature
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0.0;
    }}

    TemperatureFinal
    {{
        $Temperature;
    }}
}}

PIMPLE
{{
    nOuterCorrectors 1;
    nCorrectors     3;
    nNonOrthogonalCorrectors 0;
    pRefCell        0.0;
    pRefValue       0.0;
}}

MELTING
{{
    // Key meanings: the solver reads every entry below each step.
    // minTempCorrector/maxTempCorrector = temperature sub-iteration bounds;
    // epsilonTolerance/epsilonRelaxation = phase-fraction convergence control.
    minTempCorrector {minTCorr};
    maxTempCorrector {maxTCorr};
    epsilonTolerance {epsTol:.6g};
    epsilonRelaxation {epsRel};

    // Diverging-Gaussian beam (auto-generated from WAAM-Framework GUI):
    // HS_a        = beam 1/e^2 radius at focus [m] (= Goldak bG/2, port convention)
    // HS_bg       = lateral beam offset in X [m] (0 = centred on joint line)
    // HS_velocity = beam travel speed along +Z [m/s] (NOT X: Z is travel here)
    // HS_lg       = beam lead offset along Z [m] (0 = starts at domain centre)
    // HS_Q        = beam power [W] (= I*U*eta from the Process tab)
    HS_a {HS_a:.6g};
    HS_bg {HS_bg:.6g};
    HS_velocity {HS_velocity:.6g};
    HS_lg {HS_lg:.6g};
    HS_Q {HS_Q:.6g};

    // HS_deposition_cutoff = metal-fraction threshold for ray strike:
    //   0.99 = conduction mode (beam stops at first dense cell),
    //   0.01 = keyhole mode (beam penetrates deep into dilute cells).
    HS_deposition_cutoff {HS_deposition_cutoff};

    // Beam weave across X (0/0 = straight stringer bead):
    // Oscillation_Amplitude [m], Oscillation_Frequency [Hz].
    Oscillation_Amplitude   {osc_amp:.6g};
    Oscillation_Frequency   {osc_freq:.6g};

    // Beam divergence/focus (port-tuned):
    // Y_R           = divergence length [m] (small = pencil beam, large = cone),
    // focY          = Y plane of the beam waist [m] (= metal surface plane),
    // foc_shift_vel = waist drift rate [m/s] (+ = upward with time),
    // Q_ramp_rate   = power DECAY [W/s] applied only AFTER tshift:
    //                 Q(t) = max(Q - rate*(t-tshift), 0),
    // tshift        = decay start time [s] (keep > endTime for constant power),
    // damperSwitch  = extra interface-velocity damping (false = off).
    Y_R {Y_R};
    focY    {focY:.6g};
    foc_shift_vel {foc_shift_vel:.6g};
    Q_ramp_rate {Q_ramp_rate:.6g};
    tshift {tshift:.6g};

    damperSwitch {damper_switch};
}}

relaxationFactors
{{
    "U.*"           0.82;
}}

// ************************************************************************* //
""")


def _write_set_fields_dict(d: Path, width_m, y_lo_m, y_hi_m, travel_m,
                           metal_name: str = "ER70S-6"):
    """Write setFieldsDict: metal slab y=[gap, gap+hz], full local XZ window.

    Phase legend (fixed by the solver, stated here so the file reads clearly):
      alpha.phase1 = volume fraction of METAL ({name});
                     1 inside the box below (substrate slab), 0 elsewhere.
      phase2       = atmosphere gas (air); alpha.phase2 = 1 - alpha.phase1.
    """.format(name=metal_name)
    xh = width_m / 2.0
    zh = travel_m / 2.0
    with open(d / "setFieldsDict", "w") as f:
        f.write(_header("dictionary", "setFieldsDict", "system"))
        f.write(f"""
// Phase legend: alpha.phase1 = {metal_name} (METAL) volume fraction
//   1 = fully metal inside the substrate box, 0 = atmosphere gas (phase2 = air).
// Box meaning: (xmin ymin zmin)(xmax ymax zmax) in METERS; the metal slab spans
// full local X/Z with Y from the gas-gap plane up through the substrate.
defaultFieldValues
(
    volScalarFieldValue alpha.phase1 0
);

regions
(
    boxToCell
    {{
        box ({-xh:.6g} {y_lo_m:.6g} {-zh:.6g}) ({xh:.6g} {y_hi_m:.6g} {zh:.6g});
        fieldValues
        (
            volScalarFieldValue alpha.phase1 1
        );
    }}
);


// ************************************************************************* //
""")


def _write_transport_properties(d: Path, metal: Dict[str, Any],
                                metal_name: str = "ER70S-6"):
    """Write constant/transportProperties with two-phase metal+air properties.

    Phase legend: phase1 = {name} (METAL), phase2 = air (atmosphere gas).
    The names phase1/phase2 are fixed by the solver; their meaning is fixed here.
    """
    with open(d / "transportProperties", "w") as f:
        f.write(_header("dictionary", "transportProperties", "constant"))
        f.write(f"""
// Phase legend: phase1 = {metal_name} (METAL, VOF field alpha.phase1),
//               phase2 = air (atmosphere gas, alpha.phase2 = 1 - alpha.phase1).
phases (phase1 phase2);

// Phase 1: {metal_name} (WAAM metal)
phase1
{{
    transportModel  Newtonian;
    nu              {metal['nu_liquid']:.6g};        // Kinematic viscosity [m2/s]
    rho             {metal['rho']};           // Density [kg/m3]
    cp              {metal['cp_liquid']};            // Liquid specific heat [J/(kg.K)]
    cpsolid         {metal['cp_solid']};            // Solid specific heat [J/(kg.K)]
    kappa           {metal['kappa_liquid']};           // Liquid thermal conductivity [W/(m.K)]
    kappasolid      {metal['kappa_solid']};           // Solid thermal conductivity [W/(m.K)]
    Tsolidus        {metal['T_solidus']};           // Solidus temperature [K]
    Tliquidus       {metal['T_liquidus']};           // Liquidus temperature [K]
    LatentHeat      {metal['latent_heat']};         // Latent heat of fusion [J/kg]
    beta            {metal['beta']:.6g};         // Thermal expansion coefficient [1/K]
}}

// Phase 2: Air (surrounding atmosphere gas; VOF field alpha.phase2 = 1 - alpha.phase1)
phase2
{{
    transportModel  Newtonian;
    nu              {AIR_PROPERTIES['nu_liquid']:.6g};       // Kinematic viscosity [m2/s]
    rho             {AIR_PROPERTIES['rho']};            // Density [kg/m3]
    cp              {AIR_PROPERTIES['cp_liquid']};            // Specific heat [J/(kg.K)]
    cpsolid         {AIR_PROPERTIES['cp_solid']};            // Solid specific heat [J/(kg.K)]
    kappa           {AIR_PROPERTIES['kappa_liquid']};          // Thermal conductivity [W/(m.K)]
    kappasolid      {AIR_PROPERTIES['kappa_solid']};          // Solid thermal conductivity [W/(m.K)]
    Tsolidus        {AIR_PROPERTIES['T_solidus']};            // Solidus temperature [K]
    Tliquidus       {AIR_PROPERTIES['T_liquidus']};           // Liquidus temperature [K]
    LatentHeat      {AIR_PROPERTIES['latent_heat']};              // Latent heat [J/kg]
    beta            {AIR_PROPERTIES['beta']:.6g};         // Thermal expansion coefficient [1/K]
}}

// Surface tension properties (drive Marangoni flow along the pool surface):
// sigma    = surface tension at liquidus [N/m] (capillary hold on the pool),
// dsigmadT = its temperature slope [N/(m.K)]; negative = surface flow runs
//            from the hot beam centre toward the cold pool edge.
// Evaporation (vapor recoil pressure on the pool when T approaches Tvap):
// p0 = ambient pressure [Pa]; Tvap = boiling point [K];
// Mm = vapour molar mass [kg/mol]; LatentHeatVap = vaporisation energy [J/kg].
sigma            {metal['sigma']};              // Surface tension [N/m]
dsigmadT         {metal['dsigma_dT']:.6g};          // Temperature coefficient of surface tension [N/(m.K)]

// Evaporation properties
p0               100000.0;         // Atmospheric pressure [Pa]
Tvap             {metal['T_vap']};           // Vaporization temperature [K]
Mm               {metal['Mm']};           // Molar mass [kg/mol]
LatentHeatVap    {metal['latent_heat_vap']:.6g};           // Latent heat of vaporization [J/kg]

// ************************************************************************* //
""")


def _write_dynamic_mesh_dict(d: Path):
    # staticFvMesh = fixed mesh (no mesh-motion solver); the beam moves through
    # the heat-source term, not by deforming cells.
    with open(d / "dynamicMeshDict", "w") as f:
        f.write(_header("dictionary", "dynamicMeshDict", "constant"))
        f.write("""
dynamicFvMesh   staticFvMesh;

// ************************************************************************* //
""")


def _write_gravity(d: Path):
    # Gravity acceleration vector [m/s2], -Y (down toward the gas layer).
    # At millisecond VOF timescales gravity displacement is ~nanometres, so the
    # pool shape is set by surface tension + recoil, not by this vector.
    with open(d / "g", "w") as f:
        f.write(_header("uniformDimensionedVectorField", "g", "constant"))
        f.write("""
dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);

// ************************************************************************* //
""")


def _write_turbulence_properties(d: Path):
    # Melt pools at this scale are laminar (pool Reynolds number ~ hundreds),
    # so no turbulence model is solved; viscous stress is exact Navier-Stokes.
    with open(d / "turbulenceProperties", "w") as f:
        f.write(_header("dictionary", "turbulenceProperties", "constant"))
        f.write("""
simulationType  laminar;

// ************************************************************************* //
""")


# --- Initial condition files ---

def _write_initial_temperature(d: Path, T_amb_K: float):
    # Temperature T [K] (solver field name 'Temperature', NOT 'T').
    # zeroGradient = insulated wall (no heat flux); inletOutlet = open gas
    # boundary that holds Tamb on inflow and lets hot gas leave freely.
    with open(d / "Temperature", "w") as f:
        f.write(_header("volScalarField", "Temperature"))
        f.write(f"""
dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_amb_K:.1f};

boundaryField
{{
    lowerWall
    {{
        type            zeroGradient;
    }}

    atmosphere
    {{
        type            inletOutlet;
        inletValue      uniform {T_amb_K:.1f};
        value           uniform {T_amb_K:.1f};
    }}

    rightWall
    {{
        type            zeroGradient;
    }}

    leftWall
    {{
        type            zeroGradient;
    }}

    frontAndBack
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
""")


def _write_initial_velocity(d: Path):
    # Velocity U [m/s], uniform (0 0 0) = quiescent start, no pre-flow.
    # noSlip = wall adhesion (fluid sticks to the top clamp lid);
    # pressureInletOutletVelocity = open gas boundary (gas may enter/leave);
    # slip = frictionless far-field sides (no artificial viscous boundary layer).
    # Port-faithful: far-field sides use slip (no-slip there over-damps the pool).
    with open(d / "U", "w") as f:
        f.write(_header("volVectorField", "U"))
        f.write("""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    lowerWall
    {
        type            noSlip;
    }

    atmosphere
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }

    rightWall
    {
        type            slip;
    }

    leftWall
    {
        type            slip;
    }

    frontAndBack
    {
        type            slip;
    }
}

// ************************************************************************* //
""")


def _write_initial_alpha(d: Path, metal_name: str = "ER70S-6"):
    """Write initial alpha.phase1 (METAL volume fraction, uniform 0 = all gas).

    Phase legend: alpha.phase1 = {name} (METAL); 0 = atmosphere gas everywhere
    at t=0, setFields then stamps 1 inside the substrate box (phase2 = air).
    """.format(name=metal_name)
    with open(d / "alpha.phase1", "w") as f:
        f.write(_header("volScalarField", "alpha.phase1"))
        f.write("// alpha.phase1 = " + metal_name + " (METAL) volume fraction:"
                " 1 = metal, 0 = gas.\n")
        f.write("""// Initialised to 0 (all atmosphere); setFields stamps 1 in the substrate box.
dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    lowerWall
    {
        type            zeroGradient;
    }

    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }

    rightWall
    {
        type            zeroGradient;
    }

    leftWall
    {
        type            zeroGradient;
    }

    frontAndBack
    {
        type            zeroGradient;
    }
}


// ************************************************************************* //
""")


def _write_initial_p_rgh(d: Path):
    # p_rgh = buoyancy pressure, i.e. (p - rho*(g.h)) / rho, units [m2/s2]
    # (NOT Pascals: OpenFOAM factors density out of the pressure equation).
    # fixedFluxPressure = wall-consistent pressure flux (walls + far-field);
    # totalPressure with p0 = 0 = open gas boundary reference (gauge zero).
    with open(d / "p_rgh", "w") as f:
        f.write(_header("volScalarField", "p_rgh"))
        f.write("""
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    lowerWall
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }

    atmosphere
    {
        type            totalPressure;
        p0              uniform 0;
        U               U;
        phi             phi;
    }

    rightWall
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }

    leftWall
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }

    frontAndBack
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
}

// ************************************************************************* //
""")


# --- Case helpers: ParaView marker, README, postprocessing ---

def _write_case_foam(case: Path):
    """Write the empty case.foam marker so ParaView opens the case directly."""
    try:
        (case / "case.foam").touch(exist_ok=True)
    except Exception:
        pass


def _write_readme(case: Path, width_m, gas_gap_m, y_top_m, travel_m, HS_a,
                  HS_Q, HS_velocity, focY, Q_ramp_rate, tshift,
                  end_time, nx, ny, nz, cell_size_mm, notes) -> None:
    """Write a human-readable README explaining this case's physics choices."""
    lines = [
        "WAAM beamWeldFoam case -- auto-generated by WAAM-Framework",
        "============================================================",
        "",
        "COORDINATE FRAME (solver convention, do not rearrange):",
        "  X transverse  : +/-%.1f mm" % (width_m * 1000.0 / 2.0),
        "  Y vertical    : gas 0..%.1f mm, metal %.1f..%.1f mm"
        % (gas_gap_m * 1000.0, gas_gap_m * 1000.0, y_top_m * 1000.0),
        "  Z travel      : +/-%.1f mm (beam starts at Z=0, moves +Z)" % (travel_m * 1000.0 / 2.0),
        "  'lowerWall' = TOP wall (historic upstream name), 'atmosphere' = BOTTOM patch.",
        "  Gas-below/metal-above is load-bearing: the ray tracer deposits at the",
        "  minimum-Y metal cell, i.e. the metal-gas interface at Y=%.1f mm." % (gas_gap_m * 1000.0),
        "  The matching FreeFEM wall runs along X; the CFD window is a LOCAL",
        "  sub-volume around the pool with travel remapped to +Z (same physics).",
        "  Phase legend: alpha.phase1 = metal volume fraction (1 = metal,",
        "  0 = atmosphere gas; phase2 = air, alpha.phase2 = 1 - alpha.phase1).",
        "",
        "HEAT SOURCE (MELTING dict):",
        "  HS_a=%.6g m (= Goldak bG/2, port convention)" % HS_a,
        "  HS_Q=%.1f W, HS_velocity=%.6g m/s (+Z)" % (HS_Q, HS_velocity),
        "  focY=%.6g m (metal surface plane)" % focY,
        "  Q_ramp_rate=%.6g W/s DECAYS power after tshift=%.6g s:" % (Q_ramp_rate, tshift),
        "    Q(t) = max(Q - rate*(t-tshift), 0). For constant power keep tshift > endTime.",
        "",
        "RUN:",
        "  endTime=%.6g s, mesh %dx%dx%d (%d cells at %.2f mm)" % (
            end_time, nx, ny, nz, nx * ny * nz, cell_size_mm),
        "  ./Allrun    (needs OpenFOAM 2412 + beamWeldFoam in PATH)",
        "  ./Allclean  (removes 0*, postProcessing, polyMesh, logs)",
        "",
        "POSTPROCESS:",
        "  python3 postprocess_cfd.py   (log + field history -> cfd_history.csv/png)",
        "  paraview --script=open_in_paraview_cfd.py  (or open case.foam in ParaView)",
        "",
        "GENERATION NOTES:",
    ]
    lines += ["  - %s" % n for n in notes] or ["  - (none)"]
    lines += ["",
              "Reference: beamweldfoam/beamWeldFoam_ported (solver + cases/waam_arc).",
              "Deviations from the port (deliberate): literature-strength Marangoni,",
              "Tliquidus/LatentHeat matched to the FEM material database."]
    try:
        with open(case / "README_WAAM.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def _write_cfd_post_scripts(case: Path):
    """Generate case-local CFD postprocessing: parser + ParaView loader."""
    post_path = case / "postprocess_cfd.py"
    post_code = '''#!/usr/bin/env python3
"""WAAM CFD postprocessing: solver log + time-dir fields -> CSV/PNG/summary.

Phase legend: alpha.phase1 = METAL volume fraction (1 = metal, 0 = gas).

Usage:  python3 postprocess_cfd.py [case_dir]
Reads log.beamWeldFoam (Time / Courant lines) and per-step Temperature /
alpha.phase1 internalFields (nonuniform List<scalar>), writes
cfd_history.csv + cfd_history.png, prints a summary. Never fails the
pipeline: parse problems become warnings, exit code stays 0.
"""
import os
import re
import sys
import csv

CASE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(
    os.path.dirname(__file__))


def _is_time_dir(name):
    if name in ("0", "constant", "system", "initial", "postProcessing",
                "processor0"):
        return False
    try:
        float(name)
        return os.path.isdir(os.path.join(CASE, name))
    except ValueError:
        return False


def _read_solidus_K(default=1500.0):
    """Solidus guard for 'molten' classification, read from the case itself."""
    try:
        with open(os.path.join(CASE, "constant", "transportProperties"),
                  errors="ignore") as f:
            txt = f.read()
        m = re.search(r"Tsolidus\\s+([-+\\d\\.eE]+)", txt)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return float(default)


def parse_log():
    times, co_mean, co_max, dts = [], [], [], []
    for cand in ("log.beamWeldFoam", "log", "sim_openfoam.log"):
        p = os.path.join(CASE, cand)
        if os.path.exists(p):
            log_path = p
            break
    else:
        return {"times": [], "co_mean": [], "co_max": [], "dts": []}, None
    t = None
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = re.search(r"^Time\\s*=\\s*([\\d\\.eE+\\-]+)", line)
            if m:
                t = float(m.group(1))
                times.append(t)
                continue
            m = re.search(
                r"Courant Number mean:\\s*([\\d\\.eE+\\-]+)\\s*max:\\s*([\\d\\.eE+\\-]+)",
                line)
            if m and t is not None:
                co_mean.append(float(m.group(1)))
                co_max.append(float(m.group(2)))
            m = re.search(r"deltaT\\s*=\\s*([\\d\\.eE+\\-]+)", line)
            if m:
                dts.append(float(m.group(1)))
    return {"times": times, "co_mean": co_mean, "co_max": co_max,
            "dts": dts}, log_path


def read_internal_list(time_dir, field):
    """Read 'internalField nonuniform List<scalar>' values or None."""
    p = os.path.join(CASE, time_dir, field)
    if not os.path.exists(p):
        return None
    try:
        with open(p, errors="ignore") as f:
            txt = f.read()
        m = re.search(r"internalField\\s+nonuniform\\s+List<scalar>\\s*\\n(\\d+)\\s*\\n\\(", txt)
        if not m:
            return None  # uniform field (no spatial info)
        n = int(m.group(1))
        body = txt[m.end():m.end() + n * 24 + 8]
        vals = re.findall(r"[-+\\d\\.eE]+", body)[:n]
        if len(vals) < n:
            return None
        return [float(v) for v in vals]
    except Exception:
        return None


def detect_phases(case_dir=None):
    """Detect what VOF phases exist in an OpenFOAM case.

    Returns a dict like:
      {"phase1": "ER70S-6", "phase2": "air"}
    or for multi-phase cases:
      {"phase1": "steel", "phase2": "copper", "phase3": "air"}

    Detection methods (in priority order):
      1. constant/transportProperties phaseNames list
      2. 0/alpha.phase* files
      3. Fallback: {"phase1": "metal", "phase2": "air"}
    """
    case = Path(case_dir) if case_dir else Path(CASE)
    phases = {}

    # Method 1: transportProperties phaseNames
    tp = case / "constant" / "transportProperties"
    if tp.exists():
        txt = tp.read_text(errors="ignore")
        m = re.search("phaseNames\\s*\\(([^)]+)\\)", txt)
        if m:
            names = m.group(1).split()
            for i, name in enumerate(names, 1):
                phases[f"phase{i}"] = name.strip('"').strip("'")

    # Method 2: 0/alpha.phase* files
    if not phases:
        zero_dir = case / "0"
        if zero_dir.is_dir():
            for f in sorted(zero_dir.glob("alpha.phase*")):
                m = re.search("alpha\\.(phase\\d+)", f.name)
                if m:
                    phases[m.group(1)] = "unknown"

    # Method 3: fallback for beamWeldFoam 2-phase
    if not phases:
        phases = {"phase1": "metal", "phase2": "air"}

    return phases


def main():
    log, log_path = parse_log()
    t_solidus = _read_solidus_K()
    tdirs = sorted([d for d in os.listdir(CASE)
                    if _is_time_dir(d)], key=float)
    rows = []
    for td in tdirs:
        t = float(td)
        T = read_internal_list(td, "Temperature")
        A = read_internal_list(td, "alpha.phase1")
        row = {"time_s": t, "ncells": len(T) if T else (len(A) if A else 0)}
        if T:
            row["Tmax_K"] = max(T)
            row["Tmean_K"] = sum(T) / len(T)
        if A:
            metal = sum(1 for a in A if a > 0.5)
            row["metal_cells"] = metal
            row["metal_frac"] = metal / len(A)
            if T and len(T) == len(A):
                # molten = metal cells above the case solidus (from
                # constant/transportProperties, not a hardcoded guess)
                row["molten_cells"] = sum(
                    1 for a, x in zip(A, T) if a > 0.5 and x > t_solidus)
        rows.append(row)

    csv_path = os.path.join(CASE, "cfd_history.csv")
    if rows:
        keys = ["time_s", "ncells", "Tmax_K", "Tmean_K", "metal_cells",
                "metal_frac", "molten_cells"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    png_path = os.path.join(CASE, "cfd_history.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150)
        if log["times"]:
            axes[0].plot(log["times"], label="Time steps")
            axes[0].set_ylabel("Time [s]")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.4)
        if rows and any("Tmax_K" in r for r in rows):
            ax = axes[1]
            ax.plot([r["time_s"] for r in rows],
                    [r.get("Tmax_K", float("nan")) for r in rows],
                    label="Tmax [K]", color="#b91c1c")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Temperature [K]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)
        fig.suptitle("beamWeldFoam run history")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
    except Exception as e:
        print("[POST] plot skipped: %s" % e)

    print("=" * 60)
    print("CFD POSTPROCESS SUMMARY: %s" % CASE)
    print("  log file      : %s" % (log_path or "(none found)"))
    print("  solver steps  : %d" % len(log["times"]))
    if log["co_max"]:
        print("  max Courant   : %.4g" % max(log["co_max"]))
    print("  time dirs     : %d%s" % (
        len(tdirs), (" (last %s)" % tdirs[-1]) if tdirs else ""))
    if rows and "Tmax_K" in rows[-1]:
        print("  last Tmax     : %.1f K at t=%.6g s" % (
            rows[-1]["Tmax_K"], rows[-1]["time_s"]))
    if rows and "molten_cells" in rows[-1]:
        print("  molten cells  : %d" % rows[-1]["molten_cells"])
    print("  wrote         : cfd_history.csv%s" % (
        ", cfd_history.png" if os.path.exists(png_path) else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
'''
    pv_code = '''# ParaView loader for the WAAM beamWeldFoam case.
# Terminal: paraview --script=open_in_paraview_cfd.py
# Or: Tools -> Python Shell -> Run Script (defines case.foam path first).
import os
try:
    from paraview.simple import *
except ImportError:
    raise SystemExit("Run this script inside ParaView (Tools -> Python Shell).")
vis_dir = os.path.dirname(os.path.abspath(__file__))
foam = os.path.join(vis_dir, "case.foam")
if not os.path.exists(foam):
    raise SystemExit("case.foam not found next to this script: %s" % foam)
view = GetActiveViewOrCreate("RenderView")
reader = OpenFOAMReader(FileName=foam)
Show(reader, view)
ColorBy(GetDisplayProperties(reader, view), ("POINTS", "Temperature"))
GetDisplayProperties(reader, view).SetScalarBarVisibility(view, True)
ResetCamera()
Render()
print("[OK] Loaded beamWeldFoam case with Temperature coloring: %s" % foam)
'''
    try:
        with open(post_path, "w", encoding="utf-8") as f:
            f.write(post_code)
        os.chmod(post_path, 0o755)
        with open(case / "open_in_paraview_cfd.py", "w", encoding="utf-8") as f:
            f.write(pv_code)
    except Exception:
        pass


# --- Execution scripts ---

def _write_allrun(case: Path):
    """Write Allrun script for beamWeldFoam execution (portable, verified Mac+binary layouts).

    Tool discovery order (no hardcoded user paths baked in as requirements):
      OpenFOAM utilities (blockMesh/setFields): PATH, else OPENFOAM_LAUNCHER env,
        else the Homebrew openfoam2412 app launcher (verified working).
      Solver: BEAMWELDFOAM_BIN env, else PATH, else native darwin64 build.
      CRITICAL: On macOS, the native binary needs DYLD_LIBRARY_PATH set by the
      openfoam2412 launcher. We source the launcher's environment first, then
      run the solver directly (avoids double-wrapping).
    """
    with open(case / "Allrun", "w") as f:
        f.write("""#!/bin/bash
# ============================================================
# WAAM beamWeldFoam -- Allrun Script
# Auto-generated by WAAM-Framework
# Tool routing: PATH -> OPENFOAM_LAUNCHER -> Homebrew openfoam2412 launcher;
# solver: BEAMWELDFOAM_BIN -> PATH -> native darwin64 build -> launcher.
# ============================================================
set -e

cd "${0%/*}" || exit 1

# Source OpenFOAM environment to set DYLD_LIBRARY_PATH (critical on macOS).
# Without this, the native beamWeldFoam binary cannot find libfiniteVolume.dylib.
if [ -x /opt/homebrew/bin/openfoam2412 ]; then
    . /opt/homebrew/bin/openfoam2412
elif [ -n "${OPENFOAM_LAUNCHER:-}" ] && [ -x "$OPENFOAM_LAUNCHER" ]; then
    . "$OPENFOAM_LAUNCHER"
fi

# Route an OpenFOAM utility through the macOS app launcher when needed.
of_run() {
    if command -v "$1" >/dev/null 2>&1; then
        "$@"
    elif [ -n "${OPENFOAM_LAUNCHER:-}" ] && [ -x "$OPENFOAM_LAUNCHER" ]; then
        "$OPENFOAM_LAUNCHER" "$@"
    elif [ -x /opt/homebrew/bin/openfoam2412 ]; then
        /opt/homebrew/bin/openfoam2412 "$@"
    else
        echo "ERROR: OpenFOAM utility '$1' not found (tried PATH and launchers)." >&2
        echo "Set OPENFOAM_LAUNCHER=/path/to/openfoam2412 and retry." >&2
        exit 1
    fi
}

# Locate the beamWeldFoam solver binary.
# Note: DYLD_LIBRARY_PATH is already set by sourcing openfoam2412 above,
# so native binaries work without the launcher wrapper.
find_solver() {
    if [ -n "${BEAMWELDFOAM_BIN:-}" ] && [ -x "$BEAMWELDFOAM_BIN" ]; then
        echo "$BEAMWELDFOAM_BIN"; return 0
    fi
    if command -v beamWeldFoam >/dev/null 2>&1; then
        command -v beamWeldFoam; return 0
    fi
    native="$(ls -d "$HOME"/OpenFOAM/*/platforms/darwin64*/bin/beamWeldFoam 2>/dev/null | head -n 1)"
    if [ -n "$native" ] && [ -x "$native" ]; then
        echo "$native"; return 0
    fi
    return 1
}

echo "=== Setting up beamWeldFoam WAAM Melt Pool Case ==="

# Copy initial conditions to 0/
rm -rf 0
cp -r initial 0

# Generate mesh
echo "Generating mesh..."
of_run blockMesh -dict system/blockMeshDict

# Set initial fields (alpha.phase1 = METAL volume fraction in substrate region)
echo "Setting fields..."
of_run setFields

# Run beamWeldFoam in foreground so errors propagate (use ./Allrun & for background)
echo "Starting beamWeldFoam solver..."
if SOLVER_BIN="$(find_solver)"; then
    exec "$SOLVER_BIN"
else
    echo "Native solver not found; trying container launcher..." >&2
    of_run beamWeldFoam
fi
""")
    os.chmod(case / "Allrun", 0o755)


def _write_allclean(case: Path):
    """Write Allclean script."""
    with open(case / "Allclean", "w") as f:
        f.write("""#!/bin/bash
# ============================================================
# WAAM beamWeldFoam -- Allclean Script
# ============================================================
cd "${0%/*}" || exit 1

rm -rf 0 0.* postProcessing
rm -rf processor*
rm -rf constant/polyMesh
rm -f log.*

echo "Case cleaned."
""")
    os.chmod(case / "Allclean", 0o755)


# ============================================================
# LEGACY COMPATIBILITY: keep the old function name working
# ============================================================

def setup_openfoam_case(case_dir: str, params: Dict[str, Any]) -> str:
    """Legacy wrapper -- redirects to setup_beamweldfoam_case."""
    return setup_beamweldfoam_case(case_dir, params)


if __name__ == "__main__":
    test_dir = "./test_beamweldfoam_case"
    result = setup_beamweldfoam_case(test_dir, {
        "lx": 0.200, "ly_base": 0.100, "hz_base": 0.010,
        "ty_dep": 0.008, "h_bead": 0.003, "n_layers": 26,
        "I_weld": 200.0, "U_weld": 24.0, "eta_arc": 0.8,
        "v_scan": 10.0e-3, "bG": 7.0e-3,
        "material_dep": "ER70S-6",
    })
    print(f"[OK] beamWeldFoam case generated at: {result}")
