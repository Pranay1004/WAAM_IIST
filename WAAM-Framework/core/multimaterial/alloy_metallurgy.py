"""
Centralized Alloy Metallurgy & Phase Transformation Framework for WAAM.

Defines material family classifications, authentic phase constituents,
physical kinetic formulations, and discrete hardness relationships.
Prevents cross-alloy contamination (e.g. steel bainite/ferrite in Titanium or Inconel).
"""

from typing import Dict, Any, List, Tuple, Optional


def _calc_ti_eq(comp: Dict[str, Any]) -> str:
    al = float(comp.get("Al", 6.0))
    o = float(comp.get("O", 0.18))
    v = float(comp.get("V", 4.0))
    fe = float(comp.get("Fe", 0.25))
    al_eq = al + 10.0 * o
    mo_eq = 0.67 * v + 2.5 * fe
    return f"Al_eq = {al_eq:.2f}% | Mo_eq = {mo_eq:.2f}%"


def _calc_ss_eq(comp: Dict[str, Any]) -> str:
    cr = float(comp.get("Cr", 17.5))
    mo = float(comp.get("Mo", 2.5))
    si = float(comp.get("Si", 0.5))
    ni = float(comp.get("Ni", 12.0))
    c = float(comp.get("C", 0.02))
    mn = float(comp.get("Mn", 1.6))
    cr_eq = cr + mo + 1.5 * si
    ni_eq = ni + 30.0 * c + 0.5 * mn
    ratio = cr_eq / max(ni_eq, 0.1)
    return f"Cr_eq = {cr_eq:.2f}% | Ni_eq = {ni_eq:.2f}% (Cr_eq/Ni_eq = {ratio:.2f})"


def _calc_ni_eq(comp: Dict[str, Any]) -> str:
    nb = float(comp.get("Nb", 3.6))
    mo = float(comp.get("Mo", 9.0))
    refr = nb + mo
    return f"Nb + Mo = {refr:.2f} wt% | Ni Bal."


def _calc_al_eq(comp: Dict[str, Any]) -> str:
    si = float(comp.get("Si", 10.0))
    pct_eut = (si / 12.6) * 100.0
    return f"Si = {si:.2f}% (Hypoeutectic: {pct_eut:.1f}% of Eutectic)"


def _calc_steel_eq(comp: Dict[str, Any]) -> str:
    c = float(comp.get("C", 0.08))
    mn = float(comp.get("Mn", 1.5))
    cr = float(comp.get("Cr", 0.05))
    mo = float(comp.get("Mo", 0.02))
    v = float(comp.get("V", 0.01))
    ni = float(comp.get("Ni", 0.05))
    cu = float(comp.get("Cu", 0.0))
    ce = c + (mn / 6.0) + ((cr + mo + v) / 5.0) + ((ni + cu) / 15.0)
    return f"CE = {ce:.3f} (IIW formula)"


ALLOY_SCHEMES: Dict[str, Dict[str, Any]] = {
    "Titanium Alloy": {
        "category_id": "titanium",
        "family": "Titanium & Ti-alloys (α+β / near-α)",
        "typical_alloys": ["Ti-6Al-4V", "Ti-6-4", "Grade 5", "Ti-6Al-2Sn-4Zr-2Mo"],
        "phases": [
            {
                "id": "prior_beta",
                "name": "Prior-β Matrix (BCC)",
                "short": "Prior-β",
                "desc": "High-temperature parent BCC phase retained at elevated temperature",
                "color": "#6366f1"
            },
            {
                "id": "transf_alpha",
                "name": "Transformed-α (Basketweave/Colony)",
                "short": "Transf. α",
                "desc": "Diffusional HCP Widmanstätten platelet laths and colony α",
                "color": "#10b981"
            },
            {
                "id": "martensite_alpha_p",
                "name": "Martensitic α' (Acicular)",
                "short": "Martensite α'",
                "desc": "Diffusionless displacive HCP needle martensite formed under fast cooling",
                "color": "#ef4444"
            },
            {
                "id": "retained_beta",
                "name": "Retained-β (Interlamellar)",
                "short": "Retained β",
                "desc": "Metastable thin BCC films stabilized between alpha plates by Vanadium",
                "color": "#f59e0b"
            },
        ],
        "dominant_phase_default": "Transformed-α",
        "base_default_phases": [0.0, 92.0, 0.0, 8.0],
        "base_hardness_HV": 320.0,
        "kinetic_title": "Burgers Orientation & Continuous Cooling Kinetics (BOR)",
        "kinetic_principles": [
            ("β → α Continuous Cooling Transformation",
             "Decomposition governed by Burgers orientation relationship (0001)_α ∥ {110}_β and [11-20]_α ∥ [1-11]_β. Cooling rate across β-transus (995 °C) dictates morphology: basketweave/colony α platelets vs. acicular α' needles."),
            ("Displacive Martensite Transformation (α')",
             "Rapid quenching (cooling rate > 410 °C/s or undercooling below Ms ≈ 575–650 °C) triggers shear transformation of BCC β into HCP acicular α' martensite with high dislocation density and twin boundaries."),
            ("Solid-State Retained β & Microhardness",
             "Vanadium partitioning stabilizes thin inter-lamellar β films (~5–10%). Hardness rule of mixtures: HV = f_α'·410 + f_α·340 + f_β·300 - ΔHV_aging.")
        ],
        "composition_elements": ["Ti", "Al", "V", "Fe", "O"],
        "eq_metric_name": "Al / Mo Equivalents",
        "eq_calc": _calc_ti_eq,
        "kpi_hardness_sub": "Burgers BOR CCT Model",
        "kpi_cooling_sub": "Martensite Critical Rate (410 °C/s)"
    },
    "Austenitic Stainless Steel": {
        "category_id": "stainless",
        "family": "Austenitic Stainless Steels",
        "typical_alloys": ["316L", "308L", "304L", "309L"],
        "phases": [
            {
                "id": "austenite",
                "name": "Austenite γ (FCC Matrix)",
                "short": "Austenite γ",
                "desc": "Primary face-centered cubic matrix with high ductility",
                "color": "#3b82f6"
            },
            {
                "id": "delta_ferrite",
                "name": "Delta-Ferrite δ (Skeletal/Vermicular)",
                "short": "Delta-Ferrite δ",
                "desc": "Residual primary ferrite preventing hot solidification cracking (3–8 FN)",
                "color": "#f59e0b"
            },
            {
                "id": "sigma_phase",
                "name": "Sigma Phase σ (Intermetallic)",
                "short": "Sigma σ",
                "desc": "Brittle tetragonal Fe-Cr intermetallic phase (suppressed in rapid WAAM)",
                "color": "#ef4444"
            },
            {
                "id": "cellular_sub",
                "name": "Cellular Substructure",
                "short": "Cellular Sub.",
                "desc": "Sub-micron solidification cells decorated with segregated Mo and Cr",
                "color": "#8b5cf6"
            },
        ],
        "dominant_phase_default": "Austenite γ",
        "base_default_phases": [95.0, 5.0, 0.0, 0.0],
        "base_hardness_HV": 170.0,
        "kinetic_title": "WRC-1992 Primary Solidification & Hall-Petch Cellular Hardening",
        "kinetic_principles": [
            ("Primary Ferrite-Austenite (FA) Solidification",
             "Solidification initiates as primary δ-ferrite followed by peritectic-eutectic transformation to austenite (L → L + δ → L + δ + γ → γ + δ_skeletal). Retained δ-ferrite predicted by WRC-1992 Ferrite Number (FN ≈ 4–8)."),
            ("Hall-Petch Solidification Cell Strengthening",
             "Sub-grain cellular/dendritic network with solute segregation (Cr, Mo) creates high dislocation density boundaries. Microhardness governed by cell spacing: HV = 165 + 38 / sqrt(SDAS)."),
            ("Intermetallic Phase Stability",
             "Suppression of embrittling sigma phase (σ) and chromium carbides (M23C6) due to ultra-low carbon (C ≤ 0.03 wt%) and rapid WAAM interpass thermal cycles.")
        ],
        "composition_elements": ["Fe", "Cr", "Ni", "Mo", "Mn", "Si", "C"],
        "eq_metric_name": "WRC-1992 Cr_eq / Ni_eq",
        "eq_calc": _calc_ss_eq,
        "kpi_hardness_sub": "Hall-Petch Cell Model",
        "kpi_cooling_sub": "WRC-1992 FA Solidification"
    },
    "Nickel Superalloy": {
        "category_id": "nickel",
        "family": "Nickel-Base Superalloys",
        "typical_alloys": ["IN625", "IN718", "Inconel 625", "Inconel 718"],
        "phases": [
            {
                "id": "gamma_matrix",
                "name": "Gamma Matrix γ (FCC Solid Solution)",
                "short": "γ Matrix",
                "desc": "FCC austenitic nickel-chromium solid solution matrix",
                "color": "#3b82f6"
            },
            {
                "id": "precipitates",
                "name": "Precipitates (γ' / γ'')",
                "short": "γ' / γ''",
                "desc": "Metastable BCT Ni3Nb (γ'') and FCC Ni3(Al,Ti) (γ') strengthening coherency phases",
                "color": "#10b981"
            },
            {
                "id": "laves_phase",
                "name": "Interdendritic Laves Phase",
                "short": "Laves Phase",
                "desc": "Hexagonal (Ni,Fe,Cr)2(Nb,Mo,Ti) intermetallic formed by Nb microsegregation",
                "color": "#ef4444"
            },
            {
                "id": "delta_carbides",
                "name": "Carbides & Delta (δ) Needles",
                "short": "Carbides / δ",
                "desc": "Primary MC/M6C carbides and orthorhombic Ni3Nb needle-like precipitates",
                "color": "#f59e0b"
            },
        ],
        "dominant_phase_default": "γ Matrix",
        "base_default_phases": [94.0, 0.0, 5.0, 1.0],
        "base_hardness_HV": 240.0,
        "kinetic_title": "Scheil-Gulliver Interdendritic Segregation & Precipitation Kinetics",
        "kinetic_principles": [
            ("Non-Equilibrium Solute Segregation (Nb/Mo)",
             "Strong partition coefficient (k_Nb ≈ 0.48) drives extensive Nb microsegregation to the terminal interdendritic liquid, triggering eutectic L → γ + Laves reaction at 1150–1180 °C."),
            ("Solid Solution & Grain Boundary Strengthening",
             "Molybdenum and chromium provide potent matrix solution strengthening. Hardness scales with solidification cell refinement: HV = 230 + 50 / sqrt(SDAS) + ΔHV_ppt."),
            ("Secondary Reheating & Aging Response",
             "Cumulative thermal passes in WAAM trigger localized precipitation of metastable BCT γ'' (Ni3Nb) platelets and needle-like equilibrium δ phase at grain boundaries.")
        ],
        "composition_elements": ["Ni", "Cr", "Fe", "Mo", "Nb", "Ti", "Al", "C"],
        "eq_metric_name": "Refractory Content (Nb+Mo)",
        "eq_calc": _calc_ni_eq,
        "kpi_hardness_sub": "Scheil Segregation Model",
        "kpi_cooling_sub": "Interdendritic Cell Spacing"
    },
    "Aluminium Alloy": {
        "category_id": "aluminum",
        "family": "Aluminium Alloys (Al-Si / Al-Mg)",
        "typical_alloys": ["AlSi10Mg", "ER4043", "ER5356", "AlSi7Mg"],
        "phases": [
            {
                "id": "alpha_al",
                "name": "Primary α-Al Matrix (FCC Cells)",
                "short": "α-Al Matrix",
                "desc": "Primary face-centered cubic aluminum solid solution cellular dendrites",
                "color": "#3b82f6"
            },
            {
                "id": "eutectic_si",
                "name": "Eutectic Silicon Network",
                "short": "Eutectic Si",
                "desc": "Intercellular fibrous Al-Si eutectic structure decorating cell boundaries",
                "color": "#f59e0b"
            },
            {
                "id": "mg2si_ppt",
                "name": "Mg₂Si Precipitates (β'')",
                "short": "Mg₂Si (β'')",
                "desc": "Coherent nano-needles responsible for precipitation hardening",
                "color": "#10b981"
            },
            {
                "id": "solid_solution",
                "name": "Solid Solution (SSSS)",
                "short": "Solid Sol.",
                "desc": "Supersaturated solid solution retained from rapid weld solidification",
                "color": "#8b5cf6"
            },
        ],
        "dominant_phase_default": "α-Al Matrix",
        "base_default_phases": [88.0, 12.0, 0.0, 0.0],
        "base_hardness_HV": 85.0,
        "kinetic_title": "Cellular Eutectic Solidification & Multi-Pass Thermal Recovery",
        "kinetic_principles": [
            ("Fine Cellular-Dendritic Solidification",
             "Rapid heat extraction produces ultra-fine α-Al cells (SDAS ≈ 2–8 µm) surrounded by an interconnected eutectic Si fibrous network, giving high as-deposited yield strength."),
            ("Cell Boundary Strengthening",
             "Hardness scales inversely with cell spacing via modified Hall-Petch relationship: HV = 70 + 65 / sqrt(SDAS)."),
            ("Intrinsic Thermal Recovery & Overaging",
             "Subsequent thermal cycles from depositing upper layers cause spheroidization/coarsening of eutectic Si and loss of precipitation hardening (auto-annealing factor: 0.78–0.95).")
        ],
        "composition_elements": ["Al", "Si", "Mg", "Fe", "Mn", "Cu", "Zn"],
        "eq_metric_name": "Si Eutectic Ratio",
        "eq_calc": _calc_al_eq,
        "kpi_hardness_sub": "Cellular Eutectic Model",
        "kpi_cooling_sub": "Thermal Auto-Annealing"
    },
    "Carbon Steel": {
        "category_id": "steel",
        "family": "Carbon & Low-Alloy Structural Steels",
        "typical_alloys": ["ER70S-6", "S355JR", "AISI 4140", "E71T-1"],
        "phases": [
            {
                "id": "ferrite",
                "name": "Ferrite α (Polygonal/Acicular)",
                "short": "Ferrite α",
                "desc": "Diffusional allotriomorphic, polygonal, and acicular ferrite",
                "color": "#3b82f6"
            },
            {
                "id": "bainite",
                "name": "Bainite (Upper/Lower)",
                "short": "Bainite",
                "desc": "Non-lamellar displacive shear product with fine carbide dispersions",
                "color": "#f59e0b"
            },
            {
                "id": "martensite",
                "name": "Martensite α' (Lath)",
                "short": "Martensite α'",
                "desc": "Diffusionless BCT lath martensite formed via rapid quenching below Ms",
                "color": "#ef4444"
            },
            {
                "id": "pearlite",
                "name": "Pearlite (Lamellar α+Fe₃C)",
                "short": "Pearlite",
                "desc": "Eutectoid lamellar aggregate of ferrite and cementite",
                "color": "#10b981"
            },
        ],
        "dominant_phase_default": "Ferrite α",
        "base_default_phases": [85.0, 0.0, 0.0, 15.0],
        "base_hardness_HV": 145.0,
        "kinetic_title": "Kirkaldy-Venugopalan & Koistinen-Marburger Kinetics",
        "kinetic_principles": [
            ("Diffusional Austenite Decomposition",
             "Kirkaldy-Venugopalan rate equations partition austenite into polygonal/acicular ferrite, pearlite, and bainite based on alloy composition (%C, %Mn, %Si, %Ni, %Cr, %Mo) and cooling parameter Δt₈/₅."),
            ("Displacive Martensite Transformation",
             "Koistinen-Marburger kinetic equation below Ms (calc. ~475 °C): f_M = (1 - f_diff) · (1 - exp(-0.011 · (Ms - T)))."),
            ("Maynier Hardness Constitution & Auto-Tempering",
             "Composition-weighted rule of mixtures: HV = f_M·HV_M + f_B·HV_B + f_FP·HV_FP minus multi-pass reheat tempering loss ΔHV_temper.")
        ],
        "composition_elements": ["Fe", "C", "Mn", "Si", "Cr", "Ni", "Mo", "V"],
        "eq_metric_name": "Carbon Equivalent (IIW CE)",
        "eq_calc": _calc_steel_eq,
        "kpi_hardness_sub": "Kirkaldy-Venugopalan Model",
        "kpi_cooling_sub": "Kirkaldy Rate Parameter"
    }
}


def get_alloy_category(material_name: Optional[str]) -> str:
    """Classify an alloy name into its primary metallurgical family."""
    if not material_name:
        return "Carbon Steel"

    m = str(material_name).strip().upper()

    # Titanium detection (avoid false substring match on common English words)
    if any(k in m for k in ["TI-", "TI64", "TI-6", "TITANIUM", "GRADE 5", "GRADE 23", "TA6V", "TI6AL4V"]):
        return "Titanium Alloy"

    # Austenitic Stainless detection
    if any(k in m for k in ["316", "308", "304", "309", "347", "STAINLESS", "SS316", "SS304", "1.4404"]):
        return "Austenitic Stainless Steel"

    # Nickel Superalloy detection
    if any(k in m for k in ["IN625", "IN718", "625", "718", "INCONEL", "HASTELLOY", "NICKEL", "MONEL", "WASPALOY"]):
        return "Nickel Superalloy"

    # Aluminium detection
    if any(k in m for k in ["ALSI", "4043", "5356", "6061", "2319", "ALUMIN", "AL-"]):
        return "Aluminium Alloy"

    # Carbon / Low-Alloy Steel
    return "Carbon Steel"


def get_alloy_scheme(material_name: Optional[str]) -> Dict[str, Any]:
    """Retrieve the authoritative metallurgy scheme for a given material name."""
    cat = get_alloy_category(material_name)
    return ALLOY_SCHEMES.get(cat, ALLOY_SCHEMES["Carbon Steel"])


def resolve_indent_phases(row: Dict[str, Any], alloy_name: Optional[str]) -> Dict[str, Any]:
    """
    Parse a hardness indentation row and resolve physical phase percentages
    and the authentic dominant phase for the alloy.
    """
    scheme = get_alloy_scheme(alloy_name)
    phases = scheme["phases"]

    # Support both Phase1_pct..Phase4_pct and legacy Ferrite_pct..Martensite_pct
    p1 = float(row.get("Phase1_pct", row.get("Ferrite_pct", 0.0)))
    p2 = float(row.get("Phase2_pct", row.get("Pearlite_pct", 0.0)))
    p3 = float(row.get("Phase3_pct", row.get("Bainite_pct", 0.0)))
    p4 = float(row.get("Phase4_pct", row.get("Martensite_pct", 0.0)))

    # If all 0.0 (e.g. uninitialized row), fallback to base default
    if p1 == 0.0 and p2 == 0.0 and p3 == 0.0 and p4 == 0.0:
        defs = scheme.get("base_default_phases", [100.0, 0.0, 0.0, 0.0])
        p1, p2, p3, p4 = defs[0], defs[1], defs[2], defs[3]

    # Normalize if sum > 0
    total = p1 + p2 + p3 + p4
    if total > 0 and abs(total - 100.0) > 0.5:
        p1 = (p1 / total) * 100.0
        p2 = (p2 / total) * 100.0
        p3 = (p3 / total) * 100.0
        p4 = (p4 / total) * 100.0

    # Determine dominant phase
    vals = [p1, p2, p3, p4]
    max_idx = vals.index(max(vals))
    dom_name = phases[max_idx]["short"]

    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p1_short": phases[0]["short"],
        "p2_short": phases[1]["short"],
        "p3_short": phases[2]["short"],
        "p4_short": phases[3]["short"],
        "p1_name": phases[0]["name"],
        "p2_name": phases[1]["name"],
        "p3_name": phases[2]["name"],
        "p4_name": phases[3]["name"],
        "dominant_phase": dom_name,
        "category": scheme["family"]
    }
