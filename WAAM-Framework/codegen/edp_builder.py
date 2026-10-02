"""
FreeFEM .edp script generator for WAAM simulations.

Assembles a complete, validated .edp script from user parameters
and selected physics modules. Uses Jinja2-like string templating
(built-in, no external dependency required for simple cases).
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.multimaterial.alloy_metallurgy import get_alloy_category, get_alloy_scheme


class EDPBuilder:
    """Builds a complete FreeFEM .edp script from parameters."""

    def __init__(self, params: dict):
        """
        Args:
            params: Dictionary of all simulation parameters.
                    See `default_params()` for the full schema.
        """
        self.p = {**self.default_params(), **(params or {})}
        self._validate()

    @staticmethod
    def solver_setup(solver: str) -> tuple:
        """
        Map a requested solver to (freefem_load_statement, set() keyword).

        MUMPS is the only backend here that factors a 249k-DoF 3D elasticity
        system within 16 GB (6.7 GB peak; UMFPACK OOMs). In FreeFem 4.15 the
        plugin registers itself *as* `sparsesolver`, so `solver=MUMPS` inside
        set() is a compile error — we must `load "MUMPS"` and then ask for
        `sparsesolver`.
        """
        name = str(solver or "UMFPACK").strip()
        key = name.upper()
        if key == "MUMPS":
            return 'load "MUMPS"', "sparsesolver"
        if key == "SPARSESOLVER":
            return "", "sparsesolver"
        if key in ("UMFPACK", "CG", "GMRES"):
            return "", key
        return "", name

    @property
    def solver_kw(self) -> str:
        return self.solver_setup(self.p.get("solver", "UMFPACK"))[1]

    @property
    def solver_load(self) -> str:
        return self.solver_setup(self.p.get("solver", "UMFPACK"))[0]

    @staticmethod
    def default_params() -> dict:
        """Return the full default parameter dictionary."""
        return {
            # --- Geometry (Substrate 150x150x15mm, Build Height 150mm) ---
            "lx": 0.150,           # deposit/substrate length X [m]
            "ly_base": 0.150,      # substrate width Y [m]
            "hz_base": 0.015,      # substrate thickness Z [m]
            "ty_dep": 0.008,       # deposit width Y [m]
            "h_bead": 0.004,       # first layer height Z [m]
            "h_bead_first": 0.004, # first layer height Z [m]
            "remelt_ratio": 0.80,  # subsequent layer fusion / remelt ratio (e.g. 0.80 = 80%)
            "h_bead_subseq": 0.0032, # subsequent layer height Z [m] (0.80 * 4mm = 3.2mm)
            "n_layers": 26,        # number of layers

            # --- Process ---
            "I_weld": 200.0,       # welding current [A]
            "U_weld": 24.0,        # voltage [V]
            "eta_arc": 0.8,        # arc efficiency
            "v_scan": 10.0e-3,     # scan speed [m/s]
            "t_wait": 60.0,        # inter-pass cooling time [s]

            # --- Heat Source ---
            "heat_source": "goldak",   # goldak, gaussian_vol, gaussian_surf,
                                       # conical, uniform, prescribed
            "af": 8.0e-3,          # Goldak front length [m]
            "ar": 20.0e-3,         # Goldak rear length [m]
            "bG": 7.0e-3,          # Goldak width [m]
            "cG": 4.0e-3,          # Goldak depth [m]
            "ff": 0.6,             # front fraction (ff+fr=2)
            "fr": 1.4,             # rear fraction
            "R_beam": 5.0e-3,      # Gaussian beam radius [m]
            "R_spot": 3.0e-3,      # Surface spot radius [m]
            "rt": 3.0e-3,          # Conical top radius [m]
            "rb": 1.0e-3,          # Conical bottom radius [m]
            "d_pen": 5.0e-3,       # Conical penetration depth [m]
            "a_band": 10.0e-3,     # Uniform half-length [m]
            "b_band": 5.0e-3,      # Uniform half-width [m]
            "c_band": 3.0e-3,      # Uniform half-depth [m]
            "R_activate": 10.0e-3, # Prescribed T activation radius [m]

            # --- Material ---
            "material": "SS308L",
            "material_base": "SS316L",
            "material_dep": "SS308L",
            "rho": 7850.0,
            "T_sol": 1450.0,
            "T_liq": 1520.0,
            "T_amb": 25.0,
            "latent_heat": 270000.0,
            "emissivity": 0.7,
            "h_conv": 15.0,
            "h_bot": 50.0,
            "nu": 0.3,
            "T_mech": 800.0,      # mechanical melting T for inherent strain

            # --- Default Mesh (Deposit Z: >= 1 element per layer for psi/Goldak) ---
            "nx_base": 25, "ny_base": 8, "nz_base": 2,
            "nx_dep": 25,  "ny_dep": 3,   "nz_dep": 26,
            "mesh_source": "parametric",  # parametric, cad_file, gmsh_file
            "mesh_file": "",

            # --- Solver ---
            "theta": 1.0,         # time integration (1.0=Backward Euler)
            "dt_weld": 0.333,     # welding timestep [s]
            "dt_wait": 10.0,      # inter-pass timestep [s]
            "dt_cool": 120.0,     # final cooling timestep [s]
            "n_cool_steps": 20,   # number of final cooling steps
            "solver": "MUMPS",  # MUMPS (249k DoF @ 6.7GB, works), UMFPACK (OOMs >100k DoF), CG

            # --- Physics Modules ---
            "enable_radiation": True,
            "k_liq_factor": 1.0,     # multiply k above liquidus (1.0=off; 2-5 = effective convective pool conductivity)
            "enable_mechanical": True,
            "enable_microstructure": True,
            "mechanical_model": "anisotropic_inherent_strain",  # anisotropic_inherent_strain, isotropic_inherent_strain
            "chi_x": 1.2,          # Longitudinal inherent strain ratio (w.r.t yield strain sigY/E)
            "chi_y": 0.4,          # Transverse inherent strain ratio
            "chi_z": 0.0,          # Through-thickness inherent strain ratio
            "mechanical_bc": "isostatic",  # isostatic, fixture_clamped, cantilever
            "k_spring_z": 1.0e4,   # Isostatic vertical soft spring [N/m^3]
            "k_spring_xy": 1.0e3,  # Isostatic lateral stabilization spring [N/m^3]
            "enable_boiling_cap": True,  # Cap T at Tboil (evaporation sink; prevents 5000+ C overheat)
            "enable_yield_truncation": False, # Obsolete post-hoc clipping, replaced by true Cauchy equilibrium

            # --- Output ---
            "output_dir": str((Path(__file__).resolve().parent.parent / "waam_output").resolve()) + "/",
            "save_every_n_weld": 8,   # ~10 frames/layer; 3 wrote 22 MB VTUs every 1 s of sim time (16+ GB/run)
            "save_every_n_cool": 5,

            # --- Multi-material ---
            "multi_material": False,
            "material_schedule": [],  # [{"layers": [1,10], "material": "ER70S-6"}, ...]
        }

    @staticmethod
    def calculate_target_mesh(params: dict, target_nodes: int = 2000, strategy: str = "physics_adaptive") -> dict:
        """Calculate optimal structured divisions (Nx, Ny, Nz) using intelligent rule-based engine."""
        try:
            from mesh.intelligent_mesher import calculate_intelligent_structured_mesh
            return calculate_intelligent_structured_mesh(params, target_nodes=target_nodes, strategy=strategy)
        except Exception:
            # Fallback calculation
            def get_m(key, default):
                v = float(params.get(key, default))
                return v / 1000.0 if v > 1.0 else v

            lx = get_m("lx", 0.2)
            ly_base = get_m("ly_base", 0.1)
            hz_base = get_m("hz_base", 0.01)
            ty_dep = get_m("ty_dep", 0.008)
            h_bead = get_m("h_bead", 0.003)
            n_lay = max(int(float(params.get("n_layers", 10))), 1)

            nz_dep = max(n_lay, 1)
            nz_base = 2 if hz_base < 0.02 else 3
            ny_dep = 3 if ty_dep >= 0.008 else 2

            aspect_y = max(ly_base / max(ty_dep, 1e-4), 1.0)
            ny_base = min(max(int(round(ny_dep * math.sqrt(aspect_y))), 4), 12)

            k = (ny_dep + 1) * (nz_dep + 1) + (ny_base + 1) * (nz_base + 1)
            nx = max(int(round(target_nodes / max(k, 1))) - 1, 4)

            return {
                "nx_dep": nx, "ny_dep": ny_dep, "nz_dep": nz_dep,
                "nx_base": nx, "ny_base": ny_base, "nz_base": nz_base,
                "target_nodes": target_nodes,
                "total_nodes": (nx + 1)*(ny_dep + 1)*(nz_dep + 1) + (nx + 1)*(ny_base + 1)*(nz_base + 1)
            }


    def _validate(self):
        p = self.p
        errors = []
        self.warnings = []

        # Auto-detect if geometric dimensions were entered in millimeters instead of meters
        for dim_key, threshold in [("lx", 5.0), ("ly_base", 5.0), ("hz_base", 1.0), ("ty_dep", 1.0), ("h_bead", 0.5)]:
            val = p.get(dim_key, 0.0)
            if val > threshold:
                p[dim_key] = val / 1000.0
                self.warnings.append(
                    f"Auto-scaled {dim_key}: {val:.1f} -> {p[dim_key]:.4f} m "
                    f"(assumed mm input; VERIFY UNITS - a legitimate meter value this large would be wrong)"
                )

        # Auto-detect if Goldak dimensions were entered in millimeters
        for src_key in ["af", "ar", "bG", "cG", "R_beam", "R_spot", "rt", "rb", "d_pen"]:
            val = p.get(src_key, 0.0)
            if val > 1.0:
                p[src_key] = val / 1000.0
                self.warnings.append(
                    f"Auto-scaled {src_key}: {val:.1f} -> {p[src_key]:.4f} m "
                    f"(assumed mm input; VERIFY UNITS)"
                )

        # Physics checks
        Pweld = p["I_weld"] * p["U_weld"] * p["eta_arc"]
        if Pweld <= 0:
            errors.append(f"Arc power must be positive (got {Pweld} W)")

        if abs(p["ff"] + p["fr"] - 2.0) > 0.01:
            errors.append(f"ff + fr must equal 2.0 (got {p['ff']+p['fr']:.3f})")

        # --- Deposit mesh must resolve each layer in Z ---
        # psiActive activation and the Goldak depth axis both assume ~1 element
        # per layer. Under-resolved NzD makes layer gating span multiple layers
        # and leaves the vertical source profile meaningless.
        n_lay = int(p.get("n_layers", 1))
        nz_dep = int(p.get("nz_dep", 1))
        if nz_dep < n_lay:
            p["nz_dep"] = n_lay
            self.warnings.append(
                f"Raised nz_dep {nz_dep} -> {n_lay} (= n_layers): deposit needs "
                f">= 1 element per layer for correct psi activation and Goldak depth resolution."
            )

        # Mesh vs source size (Non-blocking advisory warning on ALL Goldak axes)
        if p["heat_source"] == "goldak":
            hx = p["lx"] / max(p["nx_dep"], 1)
            hy = p["ty_dep"] / max(p["ny_dep"], 1)
            hz_dep = (p.get("h_bead_first", p.get("h_bead", 0.004))
                      + max(n_lay - 1, 0) * float(p.get("h_bead_subseq",
                         float(p.get("h_bead_first", p.get("h_bead", 0.004))) * float(p.get("remelt_ratio", 0.8))))
                      ) / max(int(p["nz_dep"]), 1)
            for axis, h, source, rec in [
                ("X", hx, p["af"], p["nx_dep"]),
                ("Y", hy, p["bG"], p["ny_dep"]),
                ("Z", hz_dep, p["cG"], p["nz_dep"]),
            ]:
                if h > source:
                    self.warnings.append(
                        f"Notice: mesh dz{axis} ({h*1000:.1f} mm) > source semi-axis "
                        f"({source*1000:.1f} mm). Peak temperature will be mesh-dependent; "
                        f"consider more elements along {axis} (current: {rec})."
                    )

        # CFL-like check (Non-blocking advisory warning)
        dx_min = p["lx"] / max(p["nx_dep"], 1)
        courant = p["v_scan"] * p["dt_weld"] / max(dx_min, 1e-9)
        if courant > 2.0:
            self.warnings.append(
                f"Notice: Courant ratio is {courant:.1f}. "
                f"Consider smaller dt_weld for finer thermal gradients."
            )

        if p["T_sol"] >= p["T_liq"]:
            errors.append("Solidus must be below liquidus")

        if errors:
            raise ValueError("Parameter validation failed:\n" +
                             "\n".join(f"  • {e}" for e in errors))

    def _resolve_material_entry(self, mat_name: str, db: dict) -> dict:
        """Resolve alloy name against material database with alias matching.

        Returns {} when the alloy is genuinely unknown so callers fall back to
        the analytic default functions -- NEVER silently substitutes another
        alloy (the old fallback returned SS308L for ER70S-6/IN625/etc.).
        """
        if not mat_name:
            return {}
        if mat_name in db:
            return db[mat_name]

        m = str(mat_name).strip().upper()
        # Explicit alias table (upper-case keys)
        alias = {
            "ER70S-6": "ER70S-6", "ER70S6": "ER70S-6", "ER70S": "ER70S-6",
            "SG2": "ER70S-6", "E71T-1": "ER70S-6", "S355JR": "ER70S-6",
            "308L": "SS308L", "SS308L": "SS308L", "ER308L": "SS308L",
            "316L": "SS316L", "SS316L": "SS316L", "ER316L": "SS316L",
            "304L": "SS316L", "SS304": "SS316L",
            "TI64": "Ti-6Al-4V", "TI-6AL-4V": "Ti-6Al-4V", "GRADE 5": "Ti-6Al-4V",
            "GRADE 23": "Ti-6Al-4V", "TA6V": "Ti-6Al-4V",
            "IN625": "IN625", "INCONEL 625": "IN625", "ALLOY 625": "IN625",
            "ALSI10MG": "AlSi10Mg", "AL-SI10MG": "AlSi10Mg",
            "AL1100": "Pure Al (Al1xxx)", "1100": "Pure Al (Al1xxx)",
            "PURE AL": "Pure Al (Al1xxx)",
        }
        target = alias.get(m)
        if target and target in db:
            return db[target]

        if "308" in m:
            return db.get("SS308L", {})
        if "316" in m or "304" in m:
            return db.get("SS316L", {})
        if any(k in m for k in ("TI-", "TI64", "TITANIUM", "GRADE 5", "GRADE 23", "6AL")):
            return db.get("Ti-6Al-4V", {})
        if any(k in m for k in ("IN625", "INCONEL", "HASTELLOY", "MONEL")):
            return db.get("IN625", {})
        if "ALSI10" in m:
            return db.get("AlSi10Mg", {})
        if any(k in m for k in ("AL1", "1100", "PURE AL", "ALUMIN")):
            return db.get("Pure Al (Al1xxx)", {})
        # Unknown alloy: fall back to analytic defaults (they carry the
        # correct fusion-latent Cp band), do not mislabel as 308L.
        return {}

    def _generate_interp_functions(self, mat_dict: dict, suffix: str = "") -> str:
        """Generate FreeFEM++ piecewise linear interpolation functions for an alloy."""
        if not mat_dict:
            return self._generate_default_functions(suffix=suffix)

        name = mat_dict.get("name", suffix)
        tsol = float(mat_dict.get("solidus", 1450.0))
        tliq = float(mat_dict.get("liquidus", 1520.0))
        latent = float(mat_dict.get("latent_heat", 270000.0))

        def make_pw_func(func_name: str, t_list: list, val_list: list, min_val: float = None) -> str:
            if not t_list or not val_list or len(t_list) != len(val_list):
                return ""
            out = [f"func real {func_name}(real T) {{"]
            n = len(t_list)
            for i in range(n - 1):
                t0, t1 = float(t_list[i]), float(t_list[i+1])
                v0, v1 = float(val_list[i]), float(val_list[i+1])
                if i == 0:
                    out.append(f"    if (T <= {t0:.1f}) return {v0:.6e};")
                dt = t1 - t0
                if abs(dt) < 1e-9:
                    continue
                out.append(f"    if (T <= {t1:.1f}) return {v0:.6e} + ({v1:.6e} - {v0:.6e}) * (T - {t0:.1f}) / {dt:.1f};")
            v_last = float(val_list[-1])
            if min_val is not None:
                out.append(f"    return max({v_last:.6e}, {min_val:.6e});")
            else:
                out.append(f"    return {v_last:.6e};")
            out.append("}")
            return "\n".join(out)

        # Conductivity
        k_data = mat_dict.get("conductivity", {})
        k_func = make_pw_func(f"kMat{suffix}", k_data.get("T", []), k_data.get("k", []))
        if not k_func:
            k_func = f"""func real kMat{suffix}(real T) {{
    if (T <= 800.) return 30.0;
    return 27.0;
}}"""

        # Specific Heat
        cp_data = mat_dict.get("specific_heat", {})
        cp_func = make_pw_func(f"CpMat{suffix}", cp_data.get("T", []), cp_data.get("Cp", []))
        if not cp_func:
            cp_func = f"""func real CpMat{suffix}(real T) {{
    if (T <= {tsol:.1f}) return 500.0 + (650.0 - 500.0)*T/{tsol:.1f};
    if (T <= {tliq:.1f}) return 650.0 + {latent:.1f}/({tliq:.1f} - {tsol:.1f});
    return 800.0;
}}"""

        # Thermal Expansion
        alpha_data = mat_dict.get("thermal_expansion", {})
        alpha_func = make_pw_func(f"alphaMat{suffix}", alpha_data.get("T", []), alpha_data.get("alpha", []))
        if not alpha_func:
            alpha_func = f"""func real alphaMat{suffix}(real T) {{
    if (T <= 800.) return 1.2e-5 + 0.4e-5*T/800.;
    return 1.5e-5;
}}"""

        # Young's Modulus
        e_data = mat_dict.get("youngs_modulus", {})
        e_func = make_pw_func(f"EMat{suffix}", e_data.get("T", []), e_data.get("E", []), min_val=1.0e8)
        if not e_func:
            e_func = f"""func real EMat{suffix}(real T) {{
    real ET = 200.e9;
    if (T > 600.) ET = 200.e9 - (200. - 30.)*1.e9*(T - 600.)/({tsol:.1f} - 600.);
    return max(ET, 1.e8);
}}"""

        return f"// Property functions for {name} ({suffix})\n" + "\n\n".join([k_func, cp_func, alpha_func, e_func])

    def _material_functions(self) -> str:
        """Generate temperature-dependent material property functions for substrate and arbitrary N deposition zones."""
        p = self.p
        mat_base_name = p.get("material_base", p.get("material", "ER70S-6"))
        
        # Load database
        db_path = Path(__file__).parent.parent / "core" / "multimaterial" / "material_database.json"
        db = {}
        if db_path.exists():
            try:
                with open(db_path, "r", encoding="utf-8") as ff:
                    db = json.load(ff).get("alloys", {})
            except Exception:
                db = {}

        # Resolve zones list
        raw_zones = p.get("zones")
        if not raw_zones or not isinstance(raw_zones, list):
            # Fallback to legacy zonea / zoneb or single material
            za_layers = int(p.get("zonea_layers", p.get("n_layers", 10)))
            mat_za = p.get("zonea_material", p.get("material_dep", "SS308L"))
            mat_zb = p.get("zoneb_material", p.get("material_dep", "SS316L"))
            total_l = int(p.get("n_layers", 10))
            zb_layers = max(total_l - za_layers, 0)
            raw_zones = [{"layers": za_layers, "material": mat_za}]
            if zb_layers > 0:
                raw_zones.append({"layers": zb_layers, "material": mat_zb})

        # Sanitize zones
        zones = []
        for i, z in enumerate(raw_zones):
            try:
                l_count = int(z.get("layers", 1))
            except Exception:
                l_count = 1
            m_name = str(z.get("material", "ER70S-6")).strip()
            if l_count > 0:
                zones.append({"zone_idx": len(zones) + 1, "layers": l_count, "material": m_name})
        
        if not zones:
            zones = [{"zone_idx": 1, "layers": int(p.get("n_layers", 10)), "material": mat_base_name}]

        h_bead_first = float(p.get("h_bead_first", p.get("h_bead", 0.004)))
        remelt_ratio = float(p.get("remelt_ratio", 0.80))
        h_bead_subseq = float(p.get("h_bead_subseq", h_bead_first * remelt_ratio))

        lines = []
        lines.append("// ============================================================")
        lines.append("// ULTRASONIC & MULTI-MATERIAL THERMOPHYSICAL MODEL")
        zone_desc = " | ".join([f"Zone {z['zone_idx']}: {z['material']} ({z['layers']}L)" for z in zones])
        lines.append(f"// Substrate: {mat_base_name} | {zone_desc}")
        lines.append(f"// Layer 1: {h_bead_first*1000.0:.2f} mm | Subseq: {h_bead_subseq*1000.0:.2f} mm ({remelt_ratio*100.0:.0f}% fusion)")
        lines.append("// ============================================================\n")

        # 1. Generate Substrate material functions
        mat_base_data = self._resolve_material_entry(mat_base_name, db)
        if mat_base_name == "ER70S-6" and not mat_base_data:
            lines.append(self._generate_default_functions(suffix="Base"))
        else:
            lines.append(self._generate_interp_functions(mat_base_data, suffix="Base"))
        lines.append("")

        # 2. Generate Material functions for each zone
        for z in zones:
            z_idx = z["zone_idx"]
            m_name = z["material"]
            m_data = self._resolve_material_entry(m_name, db)
            lines.append(self._generate_interp_functions(m_data, suffix=f"Z{z_idx}"))
            lines.append("")

        # 3. Dynamic Dispatchers for kMat, CpMat, EMat, alphaMat, TsolMat, TliqMat
        base_tsol = float(mat_base_data.get("solidus", p.get("T_sol", 1450.0)))
        base_tliq = float(mat_base_data.get("liquidus", p.get("T_liq", 1520.0)))
        cum_layers = 0
        dispatch_k = ["func real kMat(real T, real zv) {", "    if (zv <= HzBase + 1.e-6) return kMatBase(T);"]
        dispatch_cp = ["func real CpMat(real T, real zv) {", "    if (zv <= HzBase + 1.e-6) return CpMatBase(T);"]
        dispatch_e = ["func real EMat(real T, real zv) {", "    if (zv <= HzBase + 1.e-6) return EMatBase(T);"]
        dispatch_alpha = ["func real alphaMat(real T, real zv) {", "    if (zv <= HzBase + 1.e-6) return alphaMatBase(T);"]
        dispatch_tsol = ["func real TsolMat(real zv) {", f"    if (zv <= HzBase + 1.e-6) return {base_tsol:.1f};"]
        dispatch_tliq = ["func real TliqMat(real zv) {", f"    if (zv <= HzBase + 1.e-6) return {base_tliq:.1f};"]

        for i, z in enumerate(zones):
            z_idx = z["zone_idx"]
            m_data = self._resolve_material_entry(z["material"], db)
            z_tsol = float(m_data.get("solidus", p.get("T_sol", 1400.0)))
            z_tliq = float(m_data.get("liquidus", p.get("T_liq", 1460.0)))
            cum_layers += z["layers"]
            if cum_layers <= 1:
                z_limit_expr = "HzBase + hBead1 + 1.e-6"
            else:
                z_limit_expr = f"HzBase + hBead1 + {cum_layers - 1}*hBeadSub + 1.e-6"
            
            if i < len(zones) - 1:
                dispatch_k.append(f"    if (zv <= {z_limit_expr}) return kMatZ{z_idx}(T);")
                dispatch_cp.append(f"    if (zv <= {z_limit_expr}) return CpMatZ{z_idx}(T);")
                dispatch_e.append(f"    if (zv <= {z_limit_expr}) return EMatZ{z_idx}(T);")
                dispatch_alpha.append(f"    if (zv <= {z_limit_expr}) return alphaMatZ{z_idx}(T);")
                dispatch_tsol.append(f"    if (zv <= {z_limit_expr}) return {z_tsol:.1f};")
                dispatch_tliq.append(f"    if (zv <= {z_limit_expr}) return {z_tliq:.1f};")
            else:
                # Last zone covers all remaining upper domain
                dispatch_k.append(f"    return kMatZ{z_idx}(T);")
                dispatch_cp.append(f"    return CpMatZ{z_idx}(T);")
                dispatch_e.append(f"    return EMatZ{z_idx}(T);")
                dispatch_alpha.append(f"    return alphaMatZ{z_idx}(T);")
                dispatch_tsol.append(f"    return {z_tsol:.1f};")
                dispatch_tliq.append(f"    return {z_tliq:.1f};")

        dispatch_k.append("}")
        dispatch_cp.append("}")
        dispatch_e.append("}")
        dispatch_alpha.append("}")
        dispatch_tsol.append("}")
        dispatch_tliq.append("}")

        lines.extend([
            "\n".join(dispatch_k), "",
            "\n".join(dispatch_cp), "",
            "\n".join(dispatch_e), "",
            "\n".join(dispatch_alpha), "",
            "\n".join(dispatch_tsol), "",
            "\n".join(dispatch_tliq), ""
        ])
        return "\n".join(lines)

    def _generate_default_functions(self, suffix: str = "") -> str:
        """Generate authentic steel material functions matching DEFAULT.edp.

        Cp carries BOTH energy sinks: the Curie/eutectoid spike (600-800 C) and
        an exact fusion-latent band [Tsol,Tliq] held at 600 + L/dT so the
        trapezoidal integral recovers base*deltaT + 270 kJ/kg exactly. The old
        version returned flat 800 above Tsol and silently dropped the entire
        latent heat of fusion (~0.27 MJ/kg).
        """
        return f"""// Default steel material functions (Suffix: {suffix})
func real kMat{suffix}(real T) {{
    if (T<=800.)  return 60.-(60.-38.)*T/800.;
    if (T<=Tsol)  return 38.-(38.-27.)*(T-800.)/(Tsol-800.);
    return 27.;
}}
func real CpMat{suffix}(real T) {{
    if (T<=600.)  return 440.+(600.-440.)*T/600.;
    if (T<=720.)  return 600.+(5000.-600.)*(T-600.)/120.;
    if (T<=800.)  return 5000.-(5000.-650.)*(T-720.)/80.;
    if (T<=Tsol)  return 650.-(650.-600.)*(T-800.)/(Tsol-800.);
    if (T<=Tliq)  return 600.+270000./(Tliq-Tsol); // fusion latent as Cp spike
    return 800.;
}}
func real alphaMat{suffix}(real T) {{
    if (T<=800.)  return 1.2e-5 + (1.6e-5-1.2e-5)*T/800.;
    return 1.5e-5;
}}
func real EMat{suffix}(real T) {{
    real ET = 210.e9;
    if(T>600.) ET = 210.e9-(210.-80.)*1.e9*(T-600.)/(Tsol-600.);
    return max(ET, 1.e8);
}}"""

    def _heat_source_block(self) -> str:
        """Generate the heat source computation block."""
        p = self.p
        Pweld = p["I_weld"] * p["U_weld"] * p["eta_arc"]
        pi = math.pi

        if p["heat_source"] == "goldak":
            # Energy check (tanh split): the front term integrated over x>xT is
            # Q0f*abc*(pi/3)^1.5/2, so with Q0 = 3*sqrt(3)*f*P/(pi*sqrt(pi)*abc)
            # each side carries f*P/2 and total = (ff+fr)*P/2 = P for ff+fr=2.
            # (The textbook 6*sqrt(3) constant double-counts: it gives f*P per
            #  side, i.e. 2*P total when ff+fr=2.)
            Q0f = 3*math.sqrt(3)*p["ff"]*Pweld / (pi*math.sqrt(pi)*p["af"]*p["bG"]*p["cG"])
            Q0r = 3*math.sqrt(3)*p["fr"]*Pweld / (pi*math.sqrt(pi)*p["ar"]*p["bG"]*p["cG"])
            # Front/rear split smoothing scaled with the SOURCE (not an absolute
            # 1 mm, which swells to >100% of af if the source is laser-sized),
            # floored at ~1/4 element so the tanh spans the mesh.
            hx = p["lx"] / max(p["nx_dep"], 1)
            eps_split = max(0.1 * min(p["af"], p["ar"]), 0.25 * hx)
            return f"""        // Goldak double ellipsoid (normalized: int Q dV = Pweld; ff+fr=2 with Q0=3*sqrt(3)*f*P/(pi*sqrt(pi)*abc))
        {{
            real xT = xTorch; real yT = yTorch; real zT = zTorch;
            real epsSplit = {eps_split:.6e};
            real frontFrac = 0.5*(1.+tanh((x-xT)/epsSplit));
            real rearFrac  = 1.-frontFrac;
            Tsource = ({Q0f:.6e}*exp(-3.*(x-xT)^2/{p['af']}^2)*frontFrac
                     + {Q0r:.6e}*exp(-3.*(x-xT)^2/{p['ar']}^2)*rearFrac)
                    * exp(-3.*(y-yT)^2/{p['bG']}^2)
                    * exp(-3.*(z-zT)^2/{p['cG']}^2)
                    * psiActive;
            if((xTorch<0. || xTorch>{p['lx']}) && nPts == 0) Tsource = 0.;
            if(nPts > 0 && pState < 0.5) Tsource = 0.; // Arc off during rapid travel
        }}"""

        elif p["heat_source"] == "gaussian_vol":
            R = p["R_beam"]
            # int exp(-3 r^2/R^2) dV = (pi R^2/3)^1.5 = R^3*pi*sqrt(pi)/(3*sqrt(3))
            # => Q0 = 3*sqrt(3)*P/(pi*sqrt(pi)*R^3). (The old 6*sqrt(3) constant
            # integrated to 2*P.)
            Q0 = 3*math.sqrt(3)*Pweld / (pi*math.sqrt(pi)*R**3)
            return f"""        // 3D Gaussian volumetric (normalized: int Q dV = Pweld)
        {{
            real xT = xTorch; real yT = yTorch; real zT = zTorch;
            real r2 = (x-xT)^2 + (y-yT)^2 + (z-zT)^2;
            Tsource = {Q0:.6e} * exp(-3.*r2/{R}^2) * psiActive;
            if((xTorch<0. || xTorch>{p['lx']}) && nPts == 0) Tsource = 0.;
            if(nPts > 0 && pState < 0.5) Tsource = 0.; // Arc off during rapid travel
        }}"""

        elif p["heat_source"] == "gaussian_surf":
            # Surface flux q_s = Q0s*exp(-3 r^2/R^2) with Q0s = 3P/(pi R^2)
            # (plane integral of exp(-3r^2/R^2) = pi R^2/3), spread through the
            # current layer thickness zLayH so the volumetric integral = P.
            R = p["R_spot"]
            Q0s = 3.0 * Pweld / (pi * R * R)
            return f"""        // Gaussian surface source (normalized: int Q dV = Pweld over the layer)
        {{
            real xT = xTorch; real yT = yTorch;
            real Rspot = {R};
            real dzLayer = max(zLayH, 1.e-6);
            real r2xy = (x-xT)^2 + (y-yT)^2;
            real inBand = (z >= zLayBot-1.e-9 && z <= zLayTop+1.e-9) ? 1. : 0.;
            Tsource = {Q0s:.6e}/dzLayer * exp(-3.*r2xy/(Rspot^2)) * inBand * psiActive;
            if((xTorch<0. || xTorch>{p['lx']}) && nPts == 0) Tsource = 0.;
            if(nPts > 0 && pState < 0.5) Tsource = 0.; // Arc off during rapid travel
        }}"""

        elif p["heat_source"] == "conical":
            rt, rb, d = p["rt"], p["rb"], p["d_pen"]
            # Normalization: layer power at depth zeta is Q0*pi*Rloc(zeta)^2/3.
            # Q0 = 9P/(pi*d*(rt^2+rt*rb+rb^2)) gives
            #   int_0^d Q0*pi*Rloc^2/3 dz = P  exactly.
            # The old Q0 = P/V form integrated to only P/3 (3x under-power).
            S = rt * rt + rt * rb + rb * rb
            Q0 = 9.0 * Pweld / (pi * d * max(S, 1e-18))
            return f"""        // Conical frustum (normalized: int Q dV = Pweld)
        {{
            real xT = xTorch; real yT = yTorch; real zT = zTorch;
            real zeta = max(zT - z, 0.);
            real Rloc = {rt} + ({rb}-{rt})*min(zeta/{d}, 1.);
            real r2xy = (x-xT)^2 + (y-yT)^2;
            Tsource = (zeta <= {d}+1.e-6) ?
                {Q0:.6e} * exp(-3.*r2xy/(Rloc^2)) * psiActive : 0.;
            if((xTorch<0. || xTorch>{p['lx']}) && nPts == 0) Tsource = 0.;
            if(nPts > 0 && pState < 0.5) Tsource = 0.; // Arc off during rapid travel
        }}"""

        elif p["heat_source"] == "uniform":
            a, b, c = p["a_band"], p["b_band"], p["c_band"]
            Q0 = Pweld / (8*a*b*c)
            return f"""        // Uniform band source
        {{
            real xT = xTorch; real yT = yTorch; real zT = zTorch;
            real epsBox = 0.5e-3;
            real fxP = 0.5*(1.+tanh((x-(xT-{a}))/epsBox));
            real fxN = 0.5*(1.+tanh(((xT+{a})-x)/epsBox));
            real fyP = 0.5*(1.+tanh((y-(yT-{b}))/epsBox));
            real fyN = 0.5*(1.+tanh(((yT+{b})-y)/epsBox));
            real fzP = 0.5*(1.+tanh((z-(zT-{c}))/epsBox));
            real fzN = 0.5*(1.+tanh(((zT+{c})-z)/epsBox));
            Tsource = {Q0:.6e} * fxP*fxN*fyP*fyN*fzP*fzN * psiActive;
            if((xTorch<0. || xTorch>{p['lx']}) && nPts == 0) Tsource = 0.;
            if(nPts > 0 && pState < 0.5) Tsource = 0.; // Arc off during rapid travel
        }}"""

        elif p["heat_source"] == "prescribed":
            return f"""        // Prescribed temperature (no volumetric source)
        Tsource = 0.;"""

        else:
            raise ValueError(f"Unknown heat source: {p['heat_source']}")

    def build(self) -> str:
        """Generate the complete .edp script."""
        p = self.p
        Pweld = p["I_weld"] * p["U_weld"] * p["eta_arc"]
        h_bead_first = float(p.get("h_bead_first", p.get("h_bead", 0.004)))
        remelt_ratio = float(p.get("remelt_ratio", 0.80))
        h_bead_subseq = float(p.get("h_bead_subseq", h_bead_first * remelt_ratio))
        n_layers = int(p.get("n_layers", 10))
        hz_dep = h_bead_first if n_layers <= 1 else (h_bead_first + (n_layers - 1) * h_bead_subseq)
        hz_tot = p["hz_base"] + hz_dep
        y_centre = p["ly_base"] / 2.0
        y_left = y_centre - p["ty_dep"] / 2.0
        y_right = y_centre + p["ty_dep"] / 2.0
        t_pass = p["lx"] / p["v_scan"]

        mat_funcs = self._material_functions()
        source_block = self._heat_source_block()

        is_cad = bool(
            p.get("mesh_source") == "cad_file" or
            p.get("stl_file") or
            (p.get("mesh_file") and str(p.get("mesh_file")).lower().endswith((".mesh", ".mesh3", ".msh")))
        )
        # Activation radius: use a tighter radius to avoid pre-activating nodes
        # far ahead of the torch. The old r_act = max(af, bG) + 0.004 was too large.
        r_act = max(p.get("af", 0.008), p.get("bG", 0.007)) * 0.5
        if is_cad:
            act_cond = f"(zv >= zLayBot-1.e-6 && zv <= zLayTop+1.e-6 && ((xv-xTorch)^2 + (yv-yTorch)^2) <= {r_act**2:.6e})"
        else:
            act_cond = f"(zv >= zLayBot-1.e-6 && zv <= zLayTop+1.e-6 && yv >= yDepLeft-1.e-6 && yv <= yDepRight+1.e-6 && xv <= xTorch + {p['af']})"

        # Build radiation block
        rad_update = ""
        if p["enable_radiation"]:
            rad_update = f"""
            real TvK = Tv + 273.15;
            real TaK = {p['T_amb']} + 273.15;
            real hRad = {p['emissivity']}*5.67e-8*(TvK*TvK + TaK*TaK)*(TvK + TaK);
            hSurf[][iv] = {p['h_conv']} + hRad;"""
        else:
            rad_update = f"""
            hSurf[][iv] = {p['h_conv']};"""

        # Multi-material properties for base plate vs deposited beads
        mat_base = p.get("material_base", p.get("material", "ER70S-6"))
        mat_dep = p.get("material_dep", p.get("material", "ER70S-6"))
        db_path = Path(__file__).parent.parent / "core" / "multimaterial" / "material_database.json"
        db = {}
        if db_path.exists():
            with open(db_path) as f:
                db = json.load(f).get("alloys", {})
        rho_base = db.get(mat_base, {}).get("density", p.get("rho", 7850.0))
        rho_dep = db.get(mat_dep, {}).get("density", p.get("rho", 7900.0))

        # Build mechanical stage
        mech_stage = ""
        if p["enable_mechanical"]:
            mech_stage = self._mechanical_stage_setup()

        # Build microstructure stage (hardness/phase CSVs + z profile products)
        micro_stage = self._microstructure_stage() if p.get("enable_microstructure", True) else ""


        # Heat Input Calculations
        I_weld = float(p.get("I_weld", 180.0))
        U_weld = float(p.get("U_weld", 20.0))
        eta_arc = float(p.get("eta_arc", 0.8))
        v_scan = float(p.get("v_scan", 0.008)) # m/s
        v_scan_mm_s = v_scan * 1000.0          # mm/s
        p_gross = U_weld * I_weld              # W
        p_net = eta_arc * p_gross              # W
        h_net_j_mm = p_net / v_scan_mm_s if v_scan_mm_s > 0 else 0.0
        h_gross_j_mm = p_gross / v_scan_mm_s if v_scan_mm_s > 0 else 0.0
        w_bead_mm = float(p.get("ty_dep", 0.008)) * 1000.0
        h_bead_mm = float(p.get("h_bead", 0.003)) * 1000.0
        bead_area = w_bead_mm * h_bead_mm
        e_vol = h_net_j_mm / bead_area if bead_area > 0 else 0.0

        # Physical Boiling / Vaporization Cutoff (Thermodynamic temperature governor)
        def get_tboil(mat_name):
            mat_low = mat_name.lower()
            if "al" in mat_low or "1xx" in mat_low: return 2519.0 # Pure Al / AlSi10Mg (Al bp 2519 C)
            if "ti" in mat_low or "6al" in mat_low: return 3287.0 # Ti-6Al-4V
            if "625" in mat_low or "in6" in mat_low or mat_low.startswith("ni"):
                return 2913.0  # IN625 / Ni-base (Ni bp 2913 C)
            return 2862.0 # Carbon / austenitic steels (Fe bp 2862 C)

        def get_yield_strength(mat_name, default_val=380.0e6):
            mat_obj = db.get(mat_name, {})
            ys_list = mat_obj.get("yield_strength", {}).get("sigma_y", [])
            return float(ys_list[0]) if ys_list else default_val

        t_boil = get_tboil(mat_dep)
        sig_yield_base = get_yield_strength(mat_base, 290.0e6)
        sig_yield_dep = get_yield_strength(mat_dep, 380.0e6)

        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outdir_norm = p['output_dir'].rstrip('/') + '/'
        visdir_norm = outdir_norm + 'visual_outputs/'

        cat_dep = get_alloy_category(mat_dep)
        dep_entry = self._resolve_material_entry(mat_dep, db)
        dep_hard = dep_entry.get("hardness_coefficients", {})
        dep_comp = dep_entry.get("composition", {})
        if cat_dep == "Austenitic Stainless Steel":
            # Display formula mirrors the kinetics FN blend (WRC-1992 Cr_eq/Ni_eq)
            cr_eq = dep_comp.get("Cr", 0.0) + dep_comp.get("Mo", 0.0) + 1.5 * dep_comp.get("Si", 0.0)
            ni_eq = dep_comp.get("Ni", 0.0) + 30.0 * dep_comp.get("C", 0.0) + 0.5 * dep_comp.get("Mn", 0.0)
            fn_disp = min(12.0, max(3.0, 4.0 + 10.0 * (cr_eq / max(ni_eq, 1e-9) - 1.4)))
            hv_g = float(dep_hard.get("HV_austenite", 185))
            hv_d = float(dep_hard.get("HV_delta_ferrite", 210))
            hv_blend = (1.0 - fn_disp / 100.0) * hv_g + (fn_disp / 100.0) * hv_d
            hv_formula_code = f"{hv_blend:.2f}"
        elif cat_dep == "Carbon Steel":
            hv_formula_code = (f"(fM*{float(dep_hard.get('HV_martensite', 400)):.1f}"
                               f" + fB*{float(dep_hard.get('HV_bainite', 280)):.1f}"
                               f" + fP*{float(dep_hard.get('HV_pearlite', 175)):.1f}"
                               f" + fF*{float(dep_hard.get('HV_ferrite', 140)):.1f})")
        elif cat_dep == "Nickel Superalloy":
            hv_formula_code = f"{float(dep_hard.get('HV_gamma_matrix', 230)):.1f}"
        elif cat_dep == "Titanium Alloy":
            hv_formula_code = (f"(Tpeak(xMidSamp, yMidSamp, zMidSamp) > Tsol) ?"
                               f" (fM*{float(dep_hard.get('HV_martensite_alpha_prime', 420)):.1f}"
                               f" + (fB+fF)*{float(dep_hard.get('HV_alpha_beta', 340)):.1f}"
                               f" + fP*{float(dep_hard.get('HV_beta', 300)):.1f})"
                               f" : {float(dep_hard.get('HV_alpha_beta', 340)):.1f}")
        elif "Aluminium" in cat_dep:
            hv_disp = float(dep_hard.get("HV_as_built", dep_hard.get("HV_annealed", 28)))
            hv_formula_code = f"{hv_disp:.1f}"
        else:  # unknown -> austenitic stainless blend
            hv_formula_code = "185.0"

        hs_type = p.get('heat_source', 'goldak')
        if hs_type == 'goldak':
            hs_details = f"//    - Model Type:                   Goldak Double Ellipsoid\n//    - Front Semi-Axis (af):         {p['af']*1000.0:.2f} mm  ({p['af']} m)\n//    - Rear Semi-Axis (ar):          {p['ar']*1000.0:.2f} mm  ({p['ar']} m)\n//    - Semi-Width (bG):              {p['bG']*1000.0:.2f} mm  ({p['bG']} m)\n//    - Depth Penetration (cG):       {p['cG']*1000.0:.2f} mm  ({p['cG']} m)\n//    - Energy Fractions:             ff={p['ff']:.2f}, fr={p['fr']:.2f}"
        elif hs_type == 'gaussian_vol':
            hs_details = f"//    - Model Type:                   3D Gaussian Volumetric\n//    - Effective Radius (R_beam):    {p.get('R_beam', 0.005)*1000.0:.2f} mm"
        elif hs_type == 'conical':
            hs_details = f"//    - Model Type:                   Conical Frustum\n//    - Top Radius (rt):              {p.get('rt', 0.003)*1000.0:.2f} mm\n//    - Bottom Radius (rb):           {p.get('rb', 0.001)*1000.0:.2f} mm\n//    - Penetration Depth (d_pen):    {p.get('d_pen', 0.005)*1000.0:.2f} mm"
        else:
            hs_details = f"//    - Model Type:                   {hs_type.upper()}"

        boil_segment = (
            "EcapJ += int3d(Th)(rhoEff*CpEff*max(Temp-Tboil,0.)*psiActive);\n        Temp = min(Temp, Tboil);"
            if p.get("enable_boiling_cap", False)
            else "// Raw unclamped temperature (enable_boiling_cap=False)"
        )

        script = f"""// ================================================================================
// WAAM MULTI-PHYSICS SIMULATION — AUTO-GENERATED FREEFEM++ SCRIPT
// Framework: WAAM-Framework (Wire Arc Additive Manufacturing Multi-Physics)
// Generated: {now_ts}
// Output Run Directory: {outdir_norm}
// Visual Outputs (VTU): {visdir_norm}
// Native Terminal Log:  {outdir_norm}sim.log
// ================================================================================
//
// === COMPLETE SIMULATION PARAMETER SPECIFICATION ===
//
// 1. GEOMETRY & BUILD ENVELOPE:
//    - Substrate Length (X):          {p['lx']*1000.0:.2f} mm  ({p['lx']:.4f} m)
//    - Substrate Width (Y):           {p['ly_base']*1000.0:.2f} mm  ({p['ly_base']:.4f} m)
//    - Substrate Thickness (Z):       {p['hz_base']*1000.0:.2f} mm  ({p['hz_base']:.4f} m)
//    - Deposited Bead Width:          {p['ty_dep']*1000.0:.2f} mm  ({p['ty_dep']:.4f} m)
//    - First Layer Height (h1):       {h_bead_first*1000.0:.2f} mm  ({h_bead_first:.4f} m)
//    - Remelt / Fusion Ratio:         {remelt_ratio*100.0:.1f} %
//    - Subsequent Layer Height:       {h_bead_subseq*1000.0:.2f} mm  ({h_bead_subseq:.4f} m)
//    - Total Deposition Layers:       {p['n_layers']}
//    - Deposited Wall Height:         {hz_dep*1000.0:.2f} mm  ({hz_dep:.4f} m)
//    - Total Model Height (Z):        {hz_tot*1000.0:.2f} mm  ({hz_tot:.4f} m)
//    - Wall Centerline Y:             {y_centre*1000.0:.2f} mm
//    - Wall Y Span:                   [{y_left*1000.0:.2f}, {y_right*1000.0:.2f}] mm
//
// 2. PROCESS & ARC HEAT INPUT:
//    - Welding Arc Current:           {I_weld:.1f} A
//    - Welding Arc Voltage:           {U_weld:.1f} V
//    - Arc Thermal Efficiency (eta):  {eta_arc:.2f} ({eta_arc*100.0:.0f}%)
//    - Wire Feed Speed (WFS):         {p.get('wire_feed_speed', 5.0):.2f} m/min
//    - Torch Travel Speed (TS):       {v_scan*1000.0*60.0:.1f} mm/min  ({v_scan*1000.0:.2f} mm/s, {v_scan:.5f} m/s)
//    - Wire Diameter:                 {p.get('wire_diameter', 1.2e-3)*1000.0 if p.get('wire_diameter', 1.2e-3)<0.1 else p.get('wire_diameter', 1.2):.2f} mm
//    - Gross Electrical Power:        {p_gross:.1f} W  ({p_gross/1000.0:.2f} kW)
//    - Net Thermal Arc Power:         {p_net:.1f} W  ({p_net/1000.0:.2f} kW)
//    - Net Linear Heat Input (H_net): {h_net_j_mm:.2f} J/mm  ({h_net_j_mm/1000.0:.3f} kJ/mm, {h_net_j_mm:.0f} kJ/m)
//    - Gross Linear Heat Input:       {h_gross_j_mm:.2f} J/mm  ({h_gross_j_mm/1000.0:.3f} kJ/mm)
//    - Volumetric Energy Density:     {e_vol:.2f} J/mm^3  ({e_vol*1000.0:.1f} MJ/m^3)
//    - Single Pass Deposition Time:   {t_pass:.2f} s
//    - Inter-Pass Cooling Dwell:      {p['t_wait']:.2f} s
//
// 3. HEAT SOURCE MODEL:
{hs_details}
//
// 4. ALLOYS & THERMOPHYSICAL PROPERTIES:
//    - Substrate Base Alloy:          {mat_base}
//      * Base Density (rhoBase):      {rho_base:.1f} kg/m^3
//      * Base Yield Strength (RT):    {sig_yield_base/1e6:.1f} MPa
//    - Deposited Bead Alloy:          {mat_dep}
//      * Deposit Density (rhoDep):    {rho_dep:.1f} kg/m^3
//      * Deposit Yield Strength (RT): {sig_yield_dep/1e6:.1f} MPa
//    - Melting Solidus Temp (Tsol):   {p['T_sol']:.1f} deg C
//    - Melting Liquidus Temp (Tliq):  {p['T_liq']:.1f} deg C
//    - Boiling / Vaporization (Tboil): {t_boil:.1f} deg C
//    - Stress Annealing Temp (Tmech): {p['T_mech']:.1f} deg C
//    - Latent Heat of Fusion:         {p['latent_heat']:.1f} J/kg
//    - Ambient Room Temp (Tamb):      {p['T_amb']:.1f} deg C
//    - Surface Convection (hConv):    {p['h_conv']:.2f} W/(m^2 K)
//    - Baseplate Clamp (hBot):        {p['h_bot']:.2f} W/(m^2 K)
//    - Surface Radiation Emissivity:  {p.get('emissivity', 0.7):.2f}
//    - Poisson's Ratio (nu):          {p['nu']:.2f}
//
// 5. DISCRETIZATION & SOLVER:
//    - Mesh Mode:                     {p.get('mesh_source', 'parametric')}
//    - Substrate Elements (Nx,Ny,Nz): ({p['nx_base']}, {p['ny_base']}, {p['nz_base']})
//    - Deposit Elements (Nx,Ny,Nz):   ({p['nx_dep']}, {p['ny_dep']}, {p['nz_dep']})
//    - Linear Solver Backend:         {p.get('solver', 'UMFPACK')}
//    - Time Integration Scheme:       theta = {p.get('theta', 1.0)} (Backward Euler Implicit)
//    - Welding Time-Step (dtWeld):    {p['dt_weld']:.4f} s
//    - Inter-Pass Time-Step (dtWait): {p['dt_wait']:.4f} s
//    - Final Cooling Step (dtCool):   {p.get('dt_cool', 120.0):.2f} s ({p.get('n_cool_steps', 20)} steps)
//
// 6. MULTI-PHYSICS MODULES & GOVERNING TOGGLES:
//    - Surface Radiation Heat Loss:   {'ENABLED' if p.get('enable_radiation', True) else 'DISABLED'}
//    - Mechanical Model:              {p.get('mechanical_model', 'anisotropic_inherent_strain').upper()}
//    - Anisotropic Strain Ratios:     (chi_x={p.get('chi_x', 1.2):.2f}, chi_y={p.get('chi_y', 0.4):.2f}, chi_z={p.get('chi_z', 0.0):.2f})
//    - Support Boundary Condition:    {p.get('mechanical_bc', 'isostatic').upper()}
//    - Boiling Temp Clamping:         {'ENABLED' if p.get('enable_boiling_cap', False) else 'DISABLED (Raw Unclamped Physics)'}
//    - Cauchy Equilibrium Stresses:  DIRECT CAUCHY EQUILIBRIUM SOLVE (div sigma = 0 enforced weakly)
//
// 7. ARTIFACT DESTINATIONS:
//    - Visual VTU / PVD Results:      {visdir_norm}
//    - Tabular Data & Metrics:        {outdir_norm}
//    - Native Console Log:            {outdir_norm}sim.log
// ================================================================================

{self.solver_load}
load "iovtk"

string outdir = "{outdir_norm}";
string visdir = outdir + "visual_outputs/";
system("mkdir -p " + outdir);
system("mkdir -p " + visdir);

// === GEOMETRY & LAYER REMELTING FUSION ===
real LxW        = {p['lx']};
real LyBase     = {p['ly_base']};
real HzBase     = {p['hz_base']};
real TyDep      = {p['ty_dep']};
real hBead1     = {h_bead_first};
real remeltRatio= {remelt_ratio};
real hBeadSub   = {h_bead_subseq};
int  Nlayers    = {p['n_layers']};
real HzDep      = (Nlayers <= 1) ? (Nlayers * hBead1) : (hBead1 + (Nlayers - 1) * hBeadSub);
real HzTot      = HzBase + HzDep;
real yDepCentre = LyBase/2.;
real yDepLeft   = yDepCentre - TyDep/2.;
real yDepRight  = yDepCentre + TyDep/2.;

// === PROCESS & HEAT INPUT ===
real Pweld  = {Pweld};
real vscan  = {p['v_scan']};
real tPass  = LxW/vscan;
real tWait  = {p['t_wait']};
// Net Linear Heat Input:   {h_net_j_mm:.2f} J/mm ({h_net_j_mm/1000.0:.3f} kJ/mm)
// Gross Linear Heat Input: {h_gross_j_mm:.2f} J/mm ({h_gross_j_mm/1000.0:.3f} kJ/mm)
// Net Thermal Arc Power:   {p_net:.1f} W ({p_net/1000.0:.2f} kW)
// Volumetric Energy Density: {e_vol:.2f} J/mm^3

// === MATERIAL CONSTANTS & PHYSICAL LIMITS ===
real Tsol   = {p['T_sol']};
real Tliq   = {p['T_liq']};
real Tboil  = {t_boil};
real Tamb   = {p['T_amb']};
real rhoBase= {rho_base};
real rhoDep = {rho_dep};
real hConv  = {p['h_conv']};
real hBot   = {p['h_bot']};
real nuS    = {p['nu']};
real Tmech  = {p['T_mech']};
real sigYieldBase = {sig_yield_base};
real sigYieldDep  = {sig_yield_dep};

// === TIMESTEP ===
real dtWeld = {p['dt_weld']};
real dtWait = {p['dt_wait']};

// === G-CODE TOOLPATH READER ===
int nPts = 0;
real[int] tpT(1), tpX(1), tpY(1), tpZ(1), tpP(1);
{{
    ifstream fgcode(outdir + "toolpath.txt");
    if(fgcode.good()){{
        fgcode >> nPts;
        if(nPts > 0) {{
            tpT.resize(nPts); tpX.resize(nPts); tpY.resize(nPts); tpZ.resize(nPts); tpP.resize(nPts);
            for(int i=0; i<nPts; ++i){{
                fgcode >> tpT[i] >> tpX[i] >> tpY[i] >> tpZ[i] >> tpP[i];
            }}
            cout << " [G-CODE] Successfully loaded toolpath array with " << nPts << " coordinates." << endl;
        }}
    }} else {{
        cout << " [G-CODE] No toolpath.txt found. Falling back to linear scan assumption." << endl;
    }}
}}

// === PER-LAYER PASS TIME READER ===
// Reads per-layer pass times from the slicer output. These times account for
// varying layer perimeters (e.g., equatorial vs pole passes of a sphere).
int nLayerTimes = 0;
real[int] layerPassTimes(1);
{{
    ifstream flt(outdir + "layer_pass_times.txt");
    if(flt.good()){{
        flt >> nLayerTimes;
        if(nLayerTimes > 0) {{
            layerPassTimes.resize(nLayerTimes);
            for(int i=0; i<nLayerTimes; ++i){{
                int layIdx;
                flt >> layIdx >> layerPassTimes[i];
            }}
            cout << " [G-CODE] Loaded " << nLayerTimes << " per-layer pass times." << endl;
        }}
    }}
}}


cout << "==========================================" << endl;
cout << " T-JOINT AS WAAM PROCESS (LAYER REMELTING FUSION)" << endl;
cout << " Substrate: " << LxW*1000. << "x" << LyBase*1000. << "x" << HzBase*1000. << "mm" << endl;
cout << " Deposit: " << Nlayers << " layers | Layer 1: " << hBead1*1000. << "mm, Subseq: " << hBeadSub*1000. << "mm (" << remeltRatio*100. << "% fusion)" << endl;
cout << " Total Build Height: " << HzDep*1000. << "mm" << endl;
cout << " Width: " << TyDep*1000. << "mm  centred at y=" << yDepCentre*1000. << "mm" << endl;
cout << " P=" << Pweld << "W  v=" << vscan*1000. << "mm/s" << endl;
cout << " tPass=" << tPass << "s  tWait=" << tWait << "s  Total=" << Nlayers*(tPass+tWait) << "s" << endl;
cout << "==========================================" << endl;

// === MATERIAL FUNCTIONS ===
{mat_funcs}

// === MESH ===
{"// Using external mesh file" if p['mesh_source'] != 'parametric' else ""}
{self._mesh_block()}

// === FE SPACES ===
fespace Sh(Th, P1);
fespace Vh(Th, [P1,P1,P1]);

Sh Temp, Temp0, dTemp;
Sh kEff, CpEff, rhoEff, Tsource;
Sh psiActive, Tpeak, hSurf, TempVis;
Sh dTdt;        // Instantaneous cooling rate field [C/s]
Sh TdotCool;    // Maximum transient cooling rate at solidification front [C/s]
Sh dt85Track;   // Elapsed continuous cooling time between 800C and 500C [s]
Sh tEnter800;   // Entry timestamp for 800C boundary [s]
Sh nReheat;     // Multi-pass thermal reheat cycle counter

Vh [ux,uy,uz];
Vh [vx,vy,vz];
Sh dispM, sigM, sigXX;

{mech_stage}

// === INITIAL CONDITIONS ===
Temp  = Tamb;
Temp0 = Tamb;
Tpeak = Tamb;
TdotCool  = 0.;
dt85Track = 0.;
tEnter800 = -1.;
nReheat   = 0.;
ux[]  = 0.;

for(int iv=0; iv<Th.nv; iv++){{
    real zv = Th(iv).z;
    if(zv <= HzBase+1.e-6){{
        psiActive[][iv] = 1.;
        rhoEff[][iv] = rhoBase;
        kEff[][iv]  = kMat(Tamb, zv);
        CpEff[][iv] = CpMat(Tamb, zv);
    }} else {{
        psiActive[][iv] = 0.;
        rhoEff[][iv] = rhoDep;
        // Quiet elements: insulate (k x 1e-4) but KEEP full thermal mass.
        // The old version scaled Cp too, which left diffusivity alpha=k/rhoCp
        // unchanged (still conducting) while making the mass 1e-4x, so any
        // psi-interpolation leak spiked the temperature and only the Tboil
        // cap kept the run alive.
        kEff[][iv]  = kMat(Tamb, zv)*1.e-4;
        CpEff[][iv] = CpMat(Tamb, zv);
    }}
    hSurf[][iv] = hConv;
}}

// Boiling-cap energy accounting [J]: excess enthalpy deleted by the Tboil clip
real EcapJ = 0.;
// Energy-balance audit accumulators [J] (printed in SIMULATION SUMMARY)
real EarcJ = 0.;   // arc energy ACTUALLY injected: sum(int Tsource dV * dt)
real EintJ = 0.;   // intended arc energy: sum(Pweld * pState * dt)
real ElossJ = 0.;  // surface losses: clamp (label 1) + convection/radiation (labels 2,3)
real EquietJ = 0.; // energy discarded by the quiet-node Tamb clamp (smeared-source artifact)

// === VARIATIONAL FORMS ===
real thetaT = {p['theta']};
real kLiqF = {float(p.get('k_liq_factor', 1.0))}; // liquid-state k enhancement (1.0 = off; 2-5 mimics convective pool transport)
real dt     = dtWeld;

varf aThermal(Temp, dTemp) =
    int3d(Th)(
        rhoEff*CpEff/dt * Temp*dTemp
      + thetaT*kEff*(dx(Temp)*dx(dTemp)+dy(Temp)*dy(dTemp)+dz(Temp)*dz(dTemp))
    )
    + int2d(Th,1)( hBot  * psiActive * Temp*dTemp )
    + int2d(Th,2)( hSurf * psiActive * Temp*dTemp )
    + int2d(Th,3)( hSurf * psiActive * Temp*dTemp )
    + int2d(Th,4)( hSurf * psiActive * Temp*dTemp )
    + int2d(Th,5)( hSurf * psiActive * Temp*dTemp )
    + int2d(Th,6)( hSurf * psiActive * Temp*dTemp )
    ;

varf lThermal(Temp, dTemp) =
    int3d(Th)(
        rhoEff*CpEff/dt * Temp0*dTemp
      - (1.-thetaT)*kEff*(
            dx(Temp0)*dx(dTemp)+dy(Temp0)*dy(dTemp)+dz(Temp0)*dz(dTemp))
      + Tsource*dTemp
    )
    + int2d(Th,1)( hBot  * psiActive * Tamb*dTemp )
    + int2d(Th,2)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,3)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,4)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,5)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,6)( hSurf * psiActive * Tamb*dTemp )
    ;

// === OUTPUT SETUP ===
{{
    ofstream f(visdir+"waam.pvd");
    f<<"<?xml version='1.0'?>"<<endl;
    f<<"<VTKFile type='Collection' version='0.1'><Collection>"<<endl;
}}
{{
    ofstream f(outdir+"waam_data.csv");
    f<<"time_s,layer,phase,xTorch_mm,zLayer_mm,Tmax_C,Tpeak_max_C,HV_layer"<<endl;
}}

int frame = 0;

// === DEPOSITION LOOP ===
cout << " === WAAM DEPOSITION ===" << endl;
real t = 0.;

for(int iL=0; iL<Nlayers; iL++){{
    real zLayBot = (iL == 0) ? HzBase : (HzBase + hBead1 + (iL-1)*hBeadSub);
    real zLayH   = (iL == 0) ? hBead1 : hBeadSub;
    real zLayTop = zLayBot + zLayH;
    real zLayMid = zLayBot + zLayH/2.;

    cout << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "--- Layer " << iL+1 << "/" << Nlayers
         << "  z=[" << zLayBot*1000. << "," << zLayTop*1000. << "]mm (dz=" << zLayH*1000. << "mm) ---" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;

    // WELDING PASS
    // Use per-layer pass time from toolpath data when available (accounts for
    // varying layer perimeters). Fall back to constant tPass for linear scan.
    real tPassLayer = (nPts > 0 && nLayerTimes > 0 && iL < nLayerTimes) ? layerPassTimes[iL] : tPass;
    int NweldSteps = int(tPassLayer/dtWeld + 0.5);
    if(NweldSteps < 1) NweldSteps = 1;
    dt = tPassLayer / real(NweldSteps);

    for(int ws=1; ws<=NweldSteps; ws++){{
        real tLocal = ws*dt;
        real xTorch = vscan*tLocal;
        real yTorch = yDepCentre;
        real zTorch = zLayMid;
        real pState = 1.0;
        
        // G-Code Override
        if(nPts > 0){{
            real tGcode = min(t + tLocal, tpT[nPts-1]);
            // Find bounding indices
            int idx1 = 0; int idx2 = 0;
            for(int i=0; i<nPts-1; ++i){{
                if(tGcode >= tpT[i] && tGcode <= tpT[i+1]){{
                    idx1 = i; idx2 = i+1; break;
                }}
            }}
            if(idx1 == idx2 && idx1 == 0) idx2 = 1;
            
            real dtG = tpT[idx2] - tpT[idx1];
            real wG = (dtG > 1e-9) ? (tGcode - tpT[idx1]) / dtG : 0.0;
            xTorch = tpX[idx1] * (1.-wG) + tpX[idx2] * wG;
            yTorch = tpY[idx1] * (1.-wG) + tpY[idx2] * wG;
            zTorch = tpZ[idx1] * (1.-wG) + tpZ[idx2] * wG;
            pState = (tpP[idx1] > 0 || tpP[idx2] > 0) ? 1.0 : 0.0;
        }}

        // Progressive activation
        for(int iv=0; iv<Th.nv; iv++){{
            real xv=Th(iv).x; real zv=Th(iv).z; real yv=Th(iv).y;
            if({act_cond})
                psiActive[][iv] = 1.;
        }}

        // Update properties
        for(int iv=0; iv<Th.nv; iv++){{
            real Tv = Temp0[][iv];
            real zv = Th(iv).z;
            real psiEff = max(psiActive[][iv], 1.e-4);
            kEff[][iv]  = kMat(Tv, zv)*psiEff*(Tv > TliqMat(zv) ? kLiqF : 1.);
            CpEff[][iv] = CpMat(Tv, zv);{rad_update}
        }}

        EintJ += Pweld * pState * dt;
        // Heat source
{source_block}
        // Actual energy entering the PDE this step: int Tsource dV * dt.
        // (EintJ is the intended Pweld*dt; their ratio audits source quadrature.)
        EarcJ += dt * int3d(Th)(Tsource);

        // Solve
        matrix MT = aThermal(Sh,Sh);
        real[int] fT = lThermal(0,Sh);
        set(MT, solver={self.solver_kw});
        Temp[] = MT^-1*fT;
        Temp = max(Temp, Tamb);
        // Quiet (not-yet-deposited) nodes held at ambient: P1 source
        // interpolation smears arc energy across shared interface elements,
        // which otherwise superheats quiet nodes to 3000+ C.
        EquietJ += int3d(Th)(rhoEff*CpEff*max(Temp-Tamb, 0.)*(1.-psiActive));
        for(int iv=0; iv<Th.nv; iv++) if(psiActive[][iv] < 0.5) Temp[][iv] = Tamb;
        ElossJ += dt * (int2d(Th,1)(hBot*psiActive*(Temp - Tamb))
                      + int2d(Th,2)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,3)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,4)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,5)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,6)(hSurf*psiActive*(Temp - Tamb)));
        {boil_segment}

        // Cooling rate and thermal history tracking: dT/dt [C/s], dt85, and reheat cycles
        for(int iv=0; iv<Th.nv; iv++){{
            real Tnew = Temp[][iv];
            real Told = Temp0[][iv];
            real zv   = Th(iv).z;
            real crInst = (Told - Tnew) / max(dt, 1.e-12);
            dTdt[][iv] = -crInst;

            // 1. Solidification cooling rate [C/s] at solidification front
            if (Told >= TsolMat(zv) - 120. && Tnew <= TsolMat(zv) + 60. && crInst > 0.) {{
                if (crInst > TdotCool[][iv]) TdotCool[][iv] = crInst;
            }}
            // 2. Continuous cooling time dt85 tracking (800C -> 500C)
            if (Told >= 800. && Tnew < 800. && tEnter800[][iv] < 0.) {{
                tEnter800[][iv] = t + tLocal;
            }}
            if (tEnter800[][iv] > 0. && Told >= 500. && Tnew < 500.) {{
                real elapsed85 = (t + tLocal) - tEnter800[][iv];
                if (elapsed85 > 0.01 && (dt85Track[][iv] < 1.e-4 || elapsed85 < dt85Track[][iv])) {{
                    dt85Track[][iv] = elapsed85;
                }}
                tEnter800[][iv] = -1.;
            }}
            // 3. Multi-pass reheat cycles above transformation boundary
            if (Told < 720. && Tnew >= 720. && psiActive[][iv] > 0.5) {{
                nReheat[][iv] += 1.0;
            }}
        }}

{self._prescribed_T_override() if p['heat_source'] == 'prescribed' else ''}
        // Track Tpeak
        for(int iv=0; iv<Th.nv; iv++)
            Tpeak[][iv] = max(Tpeak[][iv], Temp[][iv]);
        Temp0 = Temp;

        // Save
        if(ws%{p['save_every_n_weld']}==0){{
            string num;
            if      (frame<   10) num="000"+frame;
            else if (frame<  100) num="00" +frame;
            else if (frame< 1000) num="0"  +frame;
            else                  num=""   +frame;
            for(int iv=0;iv<Th.nv;iv++)
                TempVis[][iv] = (psiActive[][iv]>0.5) ? Temp[][iv] : -1.;
            int[int] ord=[1,1,1,1];
            savevtk(visdir+"T_"+num+".vtu", Th,
                TempVis, psiActive, Tpeak, dTdt,
                dataname="TempC Activation Tpeak CoolingRate_C_per_s", order=ord);
            real xMidSamp = LxW / 2.0;
            real yMidSamp = yDepCentre;
            real zMidSamp = zLayMid;
            real crCur = max(abs(dTdt(xMidSamp, yMidSamp, zMidSamp)), 0.1);
            real dtC = (800.0 - 500.0) / max(crCur, 1.0);
            real fM = 0., fB = 0., fF = 1., fP = 0.;
            if(dtC < 5.0) {{ fM = 1.0; fF = 0.; }}
            else if(dtC < 20.0) {{ fB = (dtC-5.0)/15.0; fM = 1.0 - fB; fF = 0.; }}
            else {{ fP = min(0.4, 0.1 + 0.005*(dtC-20.0)); fF = 1.0 - fP; }}
            real hvCur = {hv_formula_code};

            {{
                ofstream pv(visdir+"waam.pvd",append);
                pv<<"  <DataSet timestep='"<<t+tLocal
                  <<"' file='T_"<<num<<".vtu'/>"<<endl;
            }}
            {{
                ofstream cv(outdir+"waam_data.csv",append);
                cv<<t+tLocal<<","<<iL+1<<",WELDING,"
                  <<xTorch*1000.<<","<<zLayMid*1000.<<","
                  <<Temp[].max<<","<<Tpeak[].max<<","<<hvCur<<endl;
            }}
            cout<<"  L"<<iL+1<<" ws="<<ws
                <<"  x="<<xTorch*1000.<<"mm"
                <<"  Tmax="<<Temp[].max<<"C"<<endl;
            frame++;
        }}
    }}
    // Consolidate full layer: all nodes in the deposited layer are activated
    for(int iv=0; iv<Th.nv; iv++){{
        if(Th(iv).z <= zLayTop + 1.e-5)
            psiActive[][iv] = 1.;
    }}
    t += tPassLayer;

    // INTER-PASS COOLING (RAMPED dt)
    // The first seconds after arc extinction carry the entire post-peak
    // transient (Fo = alpha*dt/dx^2 ~ 1.3 for a flat 10 s step). A geometric
    // ramp 0.5/1/2/5 s resolves it, then coarse dtWait finishes the dwell.
    Tsource = 0.;
    real[int] dtRampW = [0.5, 1., 2., 5.];
    real tWaitRem = tWait;
    int iw = 0;
    while(tWaitRem > 1.e-9){{
        dt = (iw < dtRampW.n) ? min(dtRampW[iw], tWaitRem) : min(dtWait, tWaitRem);
        for(int iv=0; iv<Th.nv; iv++){{
            real Tv = Temp0[][iv];
            real zv = Th(iv).z;
            real psiEff = max(psiActive[][iv], 1.e-4);
            kEff[][iv]  = kMat(Tv, zv)*psiEff*(Tv > TliqMat(zv) ? kLiqF : 1.);
            CpEff[][iv] = CpMat(Tv, zv);{rad_update}
        }}
        matrix MT = aThermal(Sh,Sh);
        real[int] fT = lThermal(0,Sh);
        set(MT, solver={self.solver_kw});
        Temp[] = MT^-1*fT;
        Temp = max(Temp, Tamb);
        // Quiet (not-yet-deposited) nodes held at ambient: P1 source
        // interpolation smears arc energy across shared interface elements,
        // which otherwise superheats quiet nodes to 3000+ C.
        EquietJ += int3d(Th)(rhoEff*CpEff*max(Temp-Tamb, 0.)*(1.-psiActive));
        for(int iv=0; iv<Th.nv; iv++) if(psiActive[][iv] < 0.5) Temp[][iv] = Tamb;
        ElossJ += dt * (int2d(Th,1)(hBot*psiActive*(Temp - Tamb))
                      + int2d(Th,2)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,3)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,4)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,5)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,6)(hSurf*psiActive*(Temp - Tamb)));
        for(int iv=0; iv<Th.nv; iv++){{
            real Tnew = Temp[][iv];
            real Told = Temp0[][iv];
            real crInst = (Told - Tnew) / max(dt, 1.e-12);
            dTdt[][iv] = -crInst;
            if (Told >= 800. && Tnew < 800. && tEnter800[][iv] < 0.) {{
                tEnter800[][iv] = t;
            }}
            if (tEnter800[][iv] > 0. && Told >= 500. && Tnew < 500.) {{
                real elapsed85 = t - tEnter800[][iv];
                if (elapsed85 > 0.01 && (dt85Track[][iv] < 1.e-4 || elapsed85 < dt85Track[][iv])) {{
                    dt85Track[][iv] = elapsed85;
                }}
                tEnter800[][iv] = -1.;
            }}
            Tpeak[][iv] = max(Tpeak[][iv], Temp[][iv]);
        }}
        Temp0 = Temp;
        t += dt;
        tWaitRem -= dt;
        iw++;
    }}

    // Inter-pass snapshot
    {{
        string num;
        if      (frame<   10) num="000"+frame;
        else if (frame<  100) num="00" +frame;
        else if (frame< 1000) num="0"  +frame;
        else                  num=""   +frame;
        for(int iv=0;iv<Th.nv;iv++)
            TempVis[][iv] = (psiActive[][iv]>0.5) ? Temp[][iv] : -1.;
        int[int] ord=[1,1,1];
        savevtk(visdir+"T_"+num+".vtu", Th,
            TempVis, psiActive, Tpeak,
            dataname="TempC Activation Tpeak", order=ord);

        real xMidSamp = LxW / 2.0;
        real yMidSamp = yDepCentre;
        real zMidSamp = zLayMid;
        real crCur = max(abs(dTdt(xMidSamp, yMidSamp, zMidSamp)), 0.1);
        real dtC = (800.0 - 500.0) / max(crCur, 1.0);
        real fM = 0., fB = 0., fF = 1., fP = 0.;
        if(dtC < 5.0) {{ fM = 1.0; fF = 0.; }}
        else if(dtC < 20.0) {{ fB = (dtC-5.0)/15.0; fM = 1.0 - fB; fF = 0.; }}
        else {{ fP = min(0.4, 0.1 + 0.005*(dtC-20.0)); fF = 1.0 - fP; }}
        real hvCur = {hv_formula_code};

        {{
            ofstream pv(visdir+"waam.pvd",append);
            pv<<"  <DataSet timestep='"<<t<<"' file='T_"<<num<<".vtu'/>"<<endl;
        }}
        {self._mechanical_stage_solve() if p.get('enable_mechanical', True) else ''}
        cout << "  -> Inter-pass cool:            Tmax=" << Temp[].max << "C  t=" << t << "s" << endl;
        cout << endl;
        frame++;
    }}
}}

// === FINAL COOLING ===
cout << endl;
cout << "================================================================================" << endl;
cout << " === FINAL COOLING STAGE ===" << endl;
cout << "================================================================================" << endl;
{{
    real dtCool = {p['dt_cool']};
    int nCoolSave = {p['save_every_n_cool']};
    // Ramped final cooling: the first post-build seconds decide the last
    // layer's dt85; a flat 120 s step steps straight over them.
    real[int] dtRampC = [1., 2., 5., 10., 30., 60.];
    real tCoolRem = dtCool * {p['n_cool_steps']};
    Tsource = 0.;
    int ic = 0;
    while(tCoolRem > 1.e-9){{
        dt = (ic < dtRampC.n) ? min(dtRampC[ic], tCoolRem) : min(dtCool, tCoolRem);
        for(int iv=0; iv<Th.nv; iv++){{
            real Tv = Temp0[][iv];
            real zv = Th(iv).z;
            real psiEff = max(psiActive[][iv], 1.e-4);
            kEff[][iv]  = kMat(Tv, zv)*psiEff*(Tv > TliqMat(zv) ? kLiqF : 1.);
            CpEff[][iv] = CpMat(Tv, zv);{rad_update}
        }}
        matrix MT = aThermal(Sh,Sh);
        real[int] fT = lThermal(0,Sh);
        set(MT, solver={self.solver_kw});
        Temp[] = MT^-1*fT;
        Temp = max(Temp, Tamb);
        // Quiet (not-yet-deposited) nodes held at ambient: P1 source
        // interpolation smears arc energy across shared interface elements,
        // which otherwise superheats quiet nodes to 3000+ C.
        EquietJ += int3d(Th)(rhoEff*CpEff*max(Temp-Tamb, 0.)*(1.-psiActive));
        for(int iv=0; iv<Th.nv; iv++) if(psiActive[][iv] < 0.5) Temp[][iv] = Tamb;
        ElossJ += dt * (int2d(Th,1)(hBot*psiActive*(Temp - Tamb))
                      + int2d(Th,2)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,3)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,4)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,5)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,6)(hSurf*psiActive*(Temp - Tamb)));
        for(int iv=0; iv<Th.nv; iv++){{
            real Tnew = Temp[][iv];
            real Told = Temp0[][iv];
            real crInst = (Told - Tnew) / max(dt, 1.e-12);
            dTdt[][iv] = -crInst;
            if (Told >= 800. && Tnew < 800. && tEnter800[][iv] < 0.) {{
                tEnter800[][iv] = t;
            }}
            if (tEnter800[][iv] > 0. && Told >= 500. && Tnew < 500.) {{
                real elapsed85 = t - tEnter800[][iv];
                if (elapsed85 > 0.01 && (dt85Track[][iv] < 1.e-4 || elapsed85 < dt85Track[][iv])) {{
                    dt85Track[][iv] = elapsed85;
                }}
                tEnter800[][iv] = -1.;
            }}
            Tpeak[][iv] = max(Tpeak[][iv], Temp[][iv]);
        }}
        Temp0 = Temp;
        t += dt;
        tCoolRem -= dt;
        if(ic%nCoolSave==0){{
            string num;
            if      (frame<   10) num="000"+frame;
            else if (frame<  100) num="00" +frame;
            else if (frame< 1000) num="0"  +frame;
            else                  num=""   +frame;
            int[int] ord=[1,1,1];
            savevtk(visdir+"T_"+num+".vtu", Th,
                Temp, psiActive, Tpeak,
                dataname="TempC Activation Tpeak", order=ord);
            {{
                ofstream pv(visdir+"waam.pvd",append);
                pv<<"  <DataSet timestep='"<<t<<"' file='T_"<<num<<".vtu'/>";
                pv<<endl;
            }}
            cout<<"COOL t="<<t<<"s  Tmax="<<Temp[].max<<"C"<<endl;
            frame++;
        }}
        ic++;
    }}
}}
{{
    ofstream pv(visdir+"waam.pvd",append);
    pv<<"</Collection></VTKFile>"<<endl;
}}

// --- Final Mechanical Solve (Post Final Cooling at Room Temperature) ---
cout << endl;
cout << "================================================================================" << endl;
cout << " === FINAL MECHANICAL STAGE (Room Temperature Residual Stress & Distortion) ===" << endl;
cout << "================================================================================" << endl;
{self._mechanical_stage_solve() if p.get('enable_mechanical', True) else ''}
cout << endl;



cout << endl;
cout << "================================================================================" << endl;
{micro_stage}
cout << " SIMULATION SUMMARY" << endl;
cout << "================================================================================" << endl;
cout << " Thermal frames saved:          " << frame << endl;
cout << " Simulated time:                " << t << " s" << endl;
cout << " Energy removed by boiling cap: " << EcapJ/1000. << " kJ"
     << " (excess enthalpy clamped at Tboil)" << endl;
cout << endl;
cout << " === ENERGY BALANCE AUDIT ===" << endl;
func real Hcp(real T, real zv){{
    if(T <= Tamb) return 0.;
    int nq = int(max(12., min(200., (T - Tamb)/10.)));
    real dT = (T - Tamb)/real(nq);
    real H = 0.;
    for(int i=0; i<nq; i++) H += CpMat(Tamb + (real(i)+0.5)*dT, zv)*dT;
    return H;
}}
Sh hEnth;
hEnth = 0.;
for(int iv=0; iv<Th.nv; iv++){{
    real zvN = Th(iv).z;
    real rhoN = (zvN <= HzBase+1.e-6) ? rhoBase : rhoDep;
    hEnth[][iv] = rhoN * Hcp(Temp[][iv], zvN);
}}
real EstoredJ = int3d(Th)(hEnth);
real EresJ = EarcJ - EcapJ - ElossJ - EquietJ - EstoredJ;
cout << " Intended arc input (Pweld*dt): " << EintJ/1000. << " kJ" << endl;
cout << " Injected arc heat (int Q dV):  " << EarcJ/1000. << " kJ (100 %)"
     << "  [fidelity: " << 100.*EarcJ/max(EintJ, 1.) << " % of intended]" << endl;
cout << " Stored enthalpy (final T):    " << EstoredJ/1000. << " kJ  ("
     << 100.*EstoredJ/max(EarcJ, 1.) << " % of injected)" << endl;
cout << " Boiling-cap removed:          " << EcapJ/1000. << " kJ  ("
     << 100.*EcapJ/max(EarcJ, 1.) << " %)" << endl;
cout << " Surface losses (conv+rad):    " << ElossJ/1000. << " kJ  ("
     << 100.*ElossJ/max(EarcJ, 1.) << " %)" << endl;
cout << " Quiet-node discarded:         " << EquietJ/1000. << " kJ  ("
     << 100.*EquietJ/max(EarcJ, 1.) << " %)" << endl;
cout << " Residual (numerical):         " << EresJ/1000. << " kJ  ("
     << 100.*EresJ/max(EarcJ, 1.) << " %)" << endl;
cout << " Final cold-state residual warpage (Uz): " << uz[].max*1000. << " mm" << endl;
cout << " Final cold-state peak von Mises stress: " << sigM[].max/1.e6   << " MPa" << endl;
cout << " ParaView PVD file:             " << visdir << "waam.pvd" << endl;
cout << " Mechanical VTU file:           " << visdir << "waam_mech.vtu" << endl;
cout << "================================================================================" << endl;
cout << endl;
"""
        return script

    def _mesh_block(self) -> str:
        """Generate mesh creation or import block."""
        p = self.p
        mesh_src = p.get("mesh_source", "parametric")
        mesh_file = p.get("mesh_file", "").strip()
        stl_file = p.get("stl_file", "").strip()

        # If stl_file is provided and mesh_file not yet generated, resolve or auto-mesh it
        if (not mesh_file or not os.path.exists(mesh_file)) and stl_file and os.path.exists(stl_file):
            outdir = p.get("output_dir", "./waam_output")
            os.makedirs(outdir, exist_ok=True)
            candidate_mesh = os.path.join(outdir, Path(stl_file).stem + ".mesh")
            if not os.path.exists(candidate_mesh):
                try:
                    from mesh.stl_pipeline import stl_to_freefem_mesh
                    stl_to_freefem_mesh(stl_file, candidate_mesh, target_nodes=int(p.get("target_nodes", 2000)))
                except Exception:
                    pass
            if os.path.exists(candidate_mesh):
                mesh_file = candidate_mesh

        # Check if valid FreeFEM .mesh / .mesh3 file or Gmsh MSH
        if mesh_file and os.path.exists(mesh_file) and mesh_file.lower().endswith((".mesh", ".mesh3", ".msh")):
            load_cmd = 'load "gmsh"\nmesh3 ThCadRaw = gmshload3("' + mesh_file + '");' if mesh_file.lower().endswith(".msh") else 'load "msh3"\nmesh3 ThCadRaw = readmesh3("' + mesh_file + '");'
            return f"""{load_cmd}
// Find minimum Z of imported CAD mesh to ensure 100% flush contact with substrate (ZERO GAP)
real zCadMin = 1.e9;
for(int iv=0; iv<ThCadRaw.nv; iv++){{
    if(ThCadRaw(iv).z < zCadMin) zCadMin = ThCadRaw(iv).z;
}}
// Place CAD flush on top of substrate base (z = HzBase)
mesh3 ThCad = movemesh3(ThCadRaw, transfo=[x/1000.0, y/1000.0, HzBase + (z - zCadMin)/1000.0]);

// Standard 150x150x15 mm Base Plate Substrate
int NxB={p['nx_base']}; int NyB={p['ny_base']}; int NzB={p['nz_base']};
mesh3 ThBaseBox = cube(NxB,NyB,NzB);
mesh3 ThBase    = movemesh3(ThBaseBox, transfo=[LxW*x, LyBase*y, HzBase*z]);

mesh3 Th = ThBase + ThCad;
cout << " Substrate Base: " << ThBase.nv << " nodes" << endl;
cout << " CAD Deposit: " << ThCad.nv << " nodes  " << ThCad.nt << " tets" << endl;
cout << " Combined WAAM Domain: " << Th.nv << " nodes  " << Th.nt << " tets" << endl;"""
        else:
            return f"""int NxB={p['nx_base']}; int NyB={p['ny_base']}; int NzB={p['nz_base']};
int NxD={p['nx_dep']}; int NyD={p['ny_dep']}; int NzD={p['nz_dep']};
mesh3 ThBaseBox = cube(NxB,NyB,NzB);
mesh3 ThBase    = movemesh3(ThBaseBox, transfo=[LxW*x, LyBase*y, HzBase*z]);
mesh3 ThDepBox = cube(NxD,NyD,NzD);
mesh3 ThDep    = movemesh3(ThDepBox, transfo=[
    LxW*x, yDepLeft + TyDep*y, HzBase + HzDep*z]);
mesh3 Th = ThBase + ThDep;
cout << " Mesh: " << Th.nv << " nodes  " << Th.nt << " tets" << endl;"""

    def _prescribed_T_override(self) -> str:
        """Override temperature for prescribed-T heat source."""
        p = self.p
        return f"""        // Prescribed temperature override
        for(int iv=0; iv<Th.nv; iv++){{
            real r2 = (Th(iv).x-xTorch)^2 + (Th(iv).y-yTorch)^2 + (Th(iv).z-zTorch)^2;
            if(r2 < {p['R_activate']}^2 && psiActive[][iv] > 0.5 && Temp0[][iv] < Tamb + 10.)
                Temp[][iv] = Tliq;
        }}
"""

    def _get_alloy_kinetics_freefem_code(self, mat_name: str) -> str:
        """Generate FreeFEM code for alloy-specific phase transformations and Vickers hardness.

        Constants are read from material_database.json so the kinetics map and
        the per-frame HV formula stay in lock-step. Austenitic delta-ferrite is
        predicted from WRC-1992 Cr_eq/Ni_eq (FN) instead of a hardcoded 7.5-8%,
        so 308L (FN ~8.0) and 316L (FN ~5.5) separate correctly.
        """
        cat = get_alloy_category(mat_name)
        db_path = Path(__file__).parent.parent / "core" / "multimaterial" / "material_database.json"
        db = {}
        if db_path.exists():
            with open(db_path) as f:
                db = json.load(f).get("alloys", {})
        entry = self._resolve_material_entry(mat_name, db)
        hard = entry.get("hardness_coefficients", {})
        comp = entry.get("composition", {})

        if cat == "Austenitic Stainless Steel":
            # WRC-1992 ferrite number (FN) from Schaeffler-equivalents
            cr_eq = comp.get("Cr", 0.0) + comp.get("Mo", 0.0) + 1.5 * comp.get("Si", 0.0)
            ni_eq = comp.get("Ni", 0.0) + 30.0 * comp.get("C", 0.0) + 0.5 * comp.get("Mn", 0.0)
            ratio = cr_eq / max(ni_eq, 1e-9)
            fn = min(12.0, max(3.0, 4.0 + 10.0 * (ratio - 1.4)))
            f_delta = fn / 100.0
            hv_g = float(hard.get("HV_austenite", 185))
            hv_d = float(hard.get("HV_delta_ferrite", 210))
            hv_blend = (1.0 - f_delta) * hv_g + f_delta * hv_d
            return f"""
        // {mat_name} kinetics: austenite matrix + retained delta-ferrite (WRC-1992 FN = {fn:.1f})
        fFerrite[][iv]    = {1.0 - f_delta:.4f}; // slot 1: gamma austenite
        fBainite[][iv]    = {f_delta:.4f};       // slot 2: skeletal delta-ferrite
        fMartensite[][iv] = 0.000;               // slot 3: sigma phase (suppressed)
        fPearlite[][iv]   = 0.000;               // slot 4: -
        // Hall-Petch sub-grain cell boundary strengthening on top of the two-phase blend
        HVmap[][iv] = {hv_blend:.2f} + 35.0 / sqrt(max(SDAS[][iv], 1.0));
"""

        elif cat == "Carbon Steel":
            # Low-carbon (ER70S-6 etc.): diffusional partition by dt85 with
            # athermal martensite on the fast side. Slot meanings:
            # 1=ferrite 2=pearlite(shifted to slot 2 marker) ... use slot map:
            #   slot1=ferrite, slot2=bainite, slot3=martensite, slot4=pearlite
            hv_f = float(hard.get("HV_ferrite", 140))
            hv_p = float(hard.get("HV_pearlite", 175))
            hv_b = float(hard.get("HV_bainite", 280))
            hv_m = float(hard.get("HV_martensite", 400))
            return f"""
        // {mat_name} (Carbon Steel) kinetics: dt85-partition ferrite/bainite/martensite
        real fMcs = 0.; real fBcs = 0.; real fFcs = 0.; real fPcs = 0.;
        if(dtC < 1.0)      {{ fMcs = 0.80; fBcs = 0.15; fFcs = 0.03; fPcs = 0.02; }}
        else if(dtC < 5.0) {{ fMcs = 0.25; fBcs = 0.55; fFcs = 0.15; fPcs = 0.05; }}
        else if(dtC < 20.) {{ fMcs = 0.05; fBcs = 0.30; fFcs = 0.45; fPcs = 0.20; }}
        else               {{ fMcs = 0.00; fBcs = 0.05; fFcs = 0.65; fPcs = 0.30; }}
        fFerrite[][iv]    = fFcs; // slot 1: proeutectoid/ACF ferrite
        fBainite[][iv]    = fBcs; // slot 2: bainite
        fMartensite[][iv] = fMcs; // slot 3: martensite (Ms = 475 C, KM athermal)
        fPearlite[][iv]   = fPcs; // slot 4: pearlite
        HVmap[][iv] = fFcs*{hv_f:.1f} + fPcs*{hv_p:.1f} + fBcs*{hv_b:.1f} + fMcs*{hv_m:.1f};
"""

        elif cat == "Nickel Superalloy":
            # IN625: single-phase FCC gamma, no allotropic transformation.
            # Slot map: slot1=gamma matrix, slot4=terminal Laves/NbC eutectic.
            hv_g = float(hard.get("HV_gamma_matrix", 230))
            return f"""
        // {mat_name} (Nickel Superalloy) kinetics: FCC gamma matrix + Laves eutectic
        fFerrite[][iv]    = 0.960; // slot 1: gamma (Ni-Cr-Mo-Nb) matrix
        fBainite[][iv]    = 0.000; // slot 2: -
        fMartensite[][iv] = 0.000; // slot 3: none (no martensite in FCC Ni alloy)
        fPearlite[][iv]   = 0.040; // slot 4: terminal Laves/NbC eutectic (Scheil Nb segregation)
        HVmap[][iv] = {hv_g:.1f} + 40.0 / sqrt(max(SDAS[][iv], 1.0));
"""

        elif cat == "Titanium Alloy":
            # Ti-6Al-4V: beta-transus CCT (BOR). Slot map:
            # slot1=prior-beta/colony-alpha, slot2=basketweave alpha+beta,
            # slot3=acicular alpha-prime, slot4=retained beta.
            hv_ap = float(hard.get("HV_martensite_alpha_prime", 420))
            hv_ab = float(hard.get("HV_alpha_beta", 340))
            hv_b = float(hard.get("HV_beta", 300))
            return f"""
        // {mat_name} kinetics: beta-transus continuous cooling transformation (BOR CCT)
        if(crEff > 410.0){{
            fFerrite[][iv]    = 0.00; // slot 1: prior-beta / colony alpha
            fBainite[][iv]    = 0.00; // slot 2: transformed-alpha basketweave
            fMartensite[][iv] = 0.95; // slot 3: acicular alpha-prime martensite
            fPearlite[][iv]   = 0.05; // slot 4: retained beta
        }} else if(crEff > 20.0){{
            real fWid = (crEff - 20.0) / 390.0;
            fFerrite[][iv]    = 0.00;
            fBainite[][iv]    = 0.90 * (1.0 - fWid);
            fMartensite[][iv] = 0.90 * fWid;
            fPearlite[][iv]   = 0.10;
        }} else {{
            fFerrite[][iv]    = 0.85;
            fBainite[][iv]    = 0.05;
            fMartensite[][iv] = 0.00;
            fPearlite[][iv]   = 0.10;
        }}
        HVmap[][iv] = fMartensite[][iv]*{hv_ap:.1f}
                    + (fFerrite[][iv] + fBainite[][iv])*{hv_ab:.1f}
                    + fPearlite[][iv]*{hv_b:.1f};
"""

        elif "Aluminium" in cat:
            if "AlSi" in mat_name or "10Mg" in mat_name:
                # AlSi10Mg: near-eutectic Al-Si; slot1=alpha-Al, slot2=eutectic Si.
                # SDAS-linked as-built hardness, softened by reheat loops.
                hv_ab = float(hard.get("HV_as_built", 105))
                return f"""
        // {mat_name} kinetics: alpha-Al + eutectic Si network (non-heat-treatable as-built)
        fFerrite[][iv]    = 0.880; // slot 1: primary alpha-Al
        fBainite[][iv]    = 0.120; // slot 2: eutectic Si / Mg2Si
        fMartensite[][iv] = 0.000; // slot 3: -
        fPearlite[][iv]   = 0.000; // slot 4: -
        real annealFactorAS = max(0.70, 1.0 - 0.05 * min(nReheat[][iv], 6.0));
        HVmap[][iv] = ({hv_ab:.1f} + 25.0 / sqrt(max(SDAS[][iv], 1.0))) * annealFactorAS;
"""
            # Commercially pure aluminium (Al1xxx / Al1100): single-phase FCC
            return """
        // Pure Aluminium (Al1xxx / Al1100) Kinetics: Single-phase FCC alpha-Al matrix
        fFerrite[][iv]    = 0.995; // slot 1: primary alpha-Al matrix
        fBainite[][iv]    = 0.005; // slot 2: dispersed FeAl3/AlFeSi intermetallics
        fMartensite[][iv] = 0.000; // slot 3: -
        fPearlite[][iv]   = 0.000; // slot 4: -
        real hvAsBuilt = 26.0 + 15.0 / sqrt(max(SDAS[][iv], 1.0));
        real annealFactor = max(0.85, 1.0 - 0.04 * min(nReheat[][iv], 4.0));
        HVmap[][iv] = max(24.0, min(50.0, hvAsBuilt * annealFactor));
"""

        # Unknown category: austenitic fallback (SS308L/SS316L family)
        return """
        // Default kinetics (austenitic stainless fallback)
        fFerrite[][iv]    = 0.930;
        fBainite[][iv]    = 0.070;
        fMartensite[][iv] = 0.000;
        fPearlite[][iv]   = 0.000;
        HVmap[][iv] = 185.0 + 35.0 / sqrt(max(SDAS[][iv], 1.0));
"""

    
    def _mechanical_stage_setup(self) -> str:
        p = self.p
        bc_mode = p.get("mechanical_bc", "isostatic").lower()
        k_spring_z = float(p.get("k_spring_z", 1.0e4))
        k_spring_xy = float(p.get("k_spring_xy", 1.0e3))
        if bc_mode == "fixture_clamped":
            bc_code = "+ on(1, ux=0., uy=0., uz=0.) + on(2, ux=0., uy=0., uz=0.)"
        elif bc_mode == "cantilever":
            bc_code = "+ on(1, ux=0., uy=0., uz=0.)"
        else:
            bc_code = f"+ int2d(Th, 5)({k_spring_z}*uz*vz + {k_spring_xy}*(ux*vx + uy*vy))"

        return f"""
// --- Live Mechanical Setup & Variational Forms ---
Sh Emod, lambdaL, muL;
Sh epsInhXX, epsInhYY, epsInhZZ;
Sh sigInhXX, sigInhYY, sigInhZZ;

varf aMech([ux,uy,uz],[vx,vy,vz]) =
    int3d(Th)(
        lambdaL*(dx(ux)+dy(uy)+dz(uz))*(dx(vx)+dy(vy)+dz(vz))
      + 2.*muL*(
            dx(ux)*dx(vx) + dy(uy)*dy(vy) + dz(uz)*dz(vz)
          + 0.5*(dy(ux)+dx(uy))*(dy(vx)+dx(vy))
          + 0.5*(dz(ux)+dx(uz))*(dz(vx)+dx(vz))
          + 0.5*(dz(uy)+dy(uz))*(dz(vy)+dy(vz))
        )
    )
    {bc_code};

varf lMech([ux,uy,uz],[vx,vy,vz]) =
    int3d(Th)(
        sigInhXX*dx(vx) + sigInhYY*dy(vy) + sigInhZZ*dz(vz)
    );
"""

    def _mechanical_stage_solve(self) -> str:
        p = self.p
        chi_x = float(p.get("chi_x", 1.2))
        chi_y = float(p.get("chi_y", 0.4))
        chi_z = float(p.get("chi_z", 0.0))
        freq = float(p.get("ultrasonic_frequency", 0.0))
        softening_factor = 1.0 - min(0.3, freq / 100000.0) if freq > 0 else 1.0

        return f"""
        // --- Live Layer-by-Layer Cauchy Mechanical Solve ---
        real chiX = {chi_x};
        real chiY = {chi_y};
        real chiZ = {chi_z};
        real softenF = {softening_factor};

        for(int iv=0; iv<Th.nv; iv++){{
            real psiEff = max(psiActive[][iv], 1.e-4);
            real Tv = Temp[][iv];
            real zv = Th(iv).z;
            Emod[][iv] = EMat(Tv, zv)*psiEff * softenF; // Ultrasonic Softening

            if(psiActive[][iv] > 0.5 && Tpeak[][iv] > Tmech){{
                real sigY = (zv <= HzBase + 1.e-6) ? sigYieldBase : sigYieldDep;
                real ERef = max(EMat(Tamb, zv), 1.e6);
                real epsYield = (sigY / ERef) * softenF;

                epsInhXX[][iv] = -chiX * epsYield * psiEff;
                epsInhYY[][iv] = -chiY * epsYield * psiEff;
                epsInhZZ[][iv] = -chiZ * epsYield * psiEff;
            }} else {{
                epsInhXX[][iv] = 0.;
                epsInhYY[][iv] = 0.;
                epsInhZZ[][iv] = 0.;
            }}
        }}

        lambdaL = Emod*nuS/((1.+nuS)*(1.-2.*nuS));
        muL     = Emod/(2.*(1.+nuS));

        for(int iv=0; iv<Th.nv; iv++){{
            real trInh = epsInhXX[][iv] + epsInhYY[][iv] + epsInhZZ[][iv];
            sigInhXX[][iv] = lambdaL[][iv]*trInh + 2.*muL[][iv]*epsInhXX[][iv];
            sigInhYY[][iv] = lambdaL[][iv]*trInh + 2.*muL[][iv]*epsInhYY[][iv];
            sigInhZZ[][iv] = lambdaL[][iv]*trInh + 2.*muL[][iv]*epsInhZZ[][iv];
        }}

        matrix KM = aMech(Vh,Vh);
        real[int] rhsMech = lMech(0,Vh);
        set(KM, solver={self.solver_kw});
        ux[] = KM^-1*rhsMech;

        // Assign the GLOBAL dispM/sigM fields (a local `Sh dispM = ...` used to
        // shadow them, so the final report always printed 0 mm / 0 MPa).
        dispM = sqrt(ux^2+uy^2+uz^2);
        cout << "  -> Transient Peak Displacement: " << dispM[].max*1000. << " mm | Vertical Warpage (Uz): " << uz[].max*1000. << " mm" << endl;

        // Nodal Cauchy stress: sigma = C:(eps(u) - eps_inh); von Mises
        for(int iv=0; iv<Th.nv; iv++){{
            real px = Th(iv).x, py = Th(iv).y, pz = Th(iv).z;
            real exx = dx(ux)(px,py,pz) - epsInhXX[][iv];
            real eyy = dy(uy)(px,py,pz) - epsInhYY[][iv];
            real ezz = dz(uz)(px,py,pz) - epsInhZZ[][iv];
            real exy = 0.5*(dy(ux)(px,py,pz) + dx(uy)(px,py,pz));
            real exz = 0.5*(dz(ux)(px,py,pz) + dx(uz)(px,py,pz));
            real eyz = 0.5*(dz(uy)(px,py,pz) + dy(uz)(px,py,pz));
            real lam = lambdaL[][iv];
            real muu = muL[][iv];
            real trE = exx + eyy + ezz;
            real sxx = lam*trE + 2.*muu*exx;
            real syy = lam*trE + 2.*muu*eyy;
            real szz = lam*trE + 2.*muu*ezz;
            real sxy = 2.*muu*exy;
            real sxz = 2.*muu*exz;
            real syz = 2.*muu*eyz;
            sigM[][iv] = sqrt(0.5*((sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2)
                            + 3.*(sxy^2 + sxz^2 + syz^2));
            sigXX[][iv] = sxx;
        }}
        cout << "  -> Peak von Mises Stress: " << sigM[].max/1.e6 << " MPa" << endl;
        
        string numM;
        if      (frame<   10) numM="000"+frame;
        else if (frame<  100) numM="00" +frame;
        else if (frame< 1000) numM="0"  +frame;
        else                  numM=""   +frame;
        
        {{
            int[int] ord=[1,1,1,1,1,1,1,1];
            savevtk(visdir+"waam_mech_"+numM+".vtu", Th,
                ux,uy,uz, dispM, Tpeak, epsInhXX, epsInhYY, sigXX,
                dataname="Ux Uy Uz DispMag Tpeak InherentStrainX InherentStrainY Sxx_Pa",
                order=ord);
        }}
        // ---------------------------------------------------
"""
    def _microstructure_stage(self) -> str:
        """Generate alloy-specific microstructure phase kinetics and discrete hardness indentation test."""
        p = self.p
        mat_base = p.get("material_base", p.get("substrate_material", "ER70S-6"))
        mat_dep = p.get("material_dep", p.get("deposit_material", p.get("material", "ER70S-6")))
        base_scheme = get_alloy_scheme(mat_base)
        dep_scheme = get_alloy_scheme(mat_dep)

        base_defs = base_scheme.get("base_default_phases", [85.0, 0.0, 0.0, 15.0])
        base_p1 = base_defs[0] / 100.0
        base_p2 = base_defs[1] / 100.0
        base_p3 = base_defs[2] / 100.0
        base_p4 = base_defs[3] / 100.0
        base_hv = base_scheme.get("base_hardness_HV", 145.0)

        dep_defs = dep_scheme.get("base_default_phases", [85.0, 0.0, 0.0, 15.0])
        dep_p1 = dep_defs[0] / 100.0
        dep_p2 = dep_defs[1] / 100.0
        dep_p3 = dep_defs[2] / 100.0
        dep_p4 = dep_defs[3] / 100.0
        dep_base_hv = dep_scheme.get("base_hardness_HV", 160.0)

        alloy_kinetics = self._get_alloy_kinetics_freefem_code(mat_dep)
        cat_dep = get_alloy_category(mat_dep)
        if cat_dep in ["Austenitic Stainless Steel", "Nickel Superalloy"]:
            sdas_code = "real sdasVal = 45.0 * (crEff)^(-0.33); SDAS[][iv] = sdasVal;"
        elif cat_dep == "Aluminium Alloy":
            sdas_code = "real sdasVal = 18.0 * (crEff)^(-0.30); SDAS[][iv] = sdasVal;"
        elif cat_dep == "Titanium Alloy":
            sdas_code = "real sdasVal = 12.0 * (crEff)^(-0.25); SDAS[][iv] = sdasVal;"
        else:
            sdas_code = "real sdasVal = 50.0 * (crEff)^(-0.33); SDAS[][iv] = sdasVal;"

        return f"""
// === MICROSTRUCTURE & HARDNESS STAGE ===
cout << " === MICROSTRUCTURE & HARDNESS STAGE ===" << endl;
cout << " Alloy Kinetic Model: {mat_dep} ({dep_scheme['family']})" << endl;

Sh SDAS, dt85;
Sh fFerrite, fBainite, fMartensite, fPearlite, HVmap;

// Microstructure phase kinetics dispatch
for(int iv=0; iv<Th.nv; iv++){{
    real Tp = Tpeak[][iv];
    real zv = Th(iv).z;
    real crEff = max(TdotCool[][iv], 0.5);
    real dtC = (dt85Track[][iv] > 0.05) ? dt85Track[][iv] : ((800.0 - 500.0) / crEff);
    dt85[][iv] = dtC;

    if(Tp > Tsol && psiActive[][iv] > 0.5){{
        {sdas_code}
        {alloy_kinetics}
    }} else {{
        SDAS[][iv] = 0.;
        if (zv <= HzBase + 1.e-6) {{
            fFerrite[][iv]    = {base_p1};
            fBainite[][iv]    = {base_p2};
            fMartensite[][iv] = {base_p3};
            fPearlite[][iv]   = {base_p4};
            HVmap[][iv]       = {base_hv};
        }} else {{
            fFerrite[][iv]    = {dep_p1};
            fBainite[][iv]    = {dep_p2};
            fMartensite[][iv] = {dep_p3};
            fPearlite[][iv]   = {dep_p4};
            HVmap[][iv]       = {dep_base_hv};
        }}
    }}
}}

cout << "  Predicted Vickers Hardness: " << HVmap[].min << " to " << HVmap[].max << " HV" << endl;



// === DISCRETE HARDNESS TEST INDENTATION LOCATIONS (CSV Output) ===
cout << " === EXTRACTING DISCRETE HARDNESS TEST LOCATIONS ACROSS LAYERS ===" << endl;
{{
    ofstream fH(outdir+"hardness_test_locations.csv");
    fH << "Indent_ID,Layer_Name,Region,X_mm,Y_mm,Z_mm,Vickers_HV,SDAS_um,Phase1_pct,Phase2_pct,Phase3_pct,Phase4_pct,Tpeak_C,CoolingRate_C_s,ReheatCycles" << endl;

    real xMid = LxW / 2.0;
    real yMid = LyBase / 2.0;
    int indId = 0;

    // 1. Substrate Base Indents
    for(int is=0; is<3; is++){{
        real zTest = 0.002 + is * (HzBase - 0.004) / 2.0;
        indId++;
        real hvVal = HVmap(xMid, yMid, zTest);
        real sdasVal = SDAS(xMid, yMid, zTest);
        real p1 = fFerrite(xMid, yMid, zTest)*100.0;
        real p2 = fBainite(xMid, yMid, zTest)*100.0;
        real p3 = fMartensite(xMid, yMid, zTest)*100.0;
        real p4 = fPearlite(xMid, yMid, zTest)*100.0;
        real TpVal = Tpeak(xMid, yMid, zTest);
        real CrVal = TdotCool(xMid, yMid, zTest);
        real nrVal = nReheat(xMid, yMid, zTest);

        fH << indId << ",Substrate,BaseMetal,"
           << xMid*1000. << "," << yMid*1000. << "," << zTest*1000. << ","
           << hvVal << "," << sdasVal << "," << p1 << "," << p2 << "," << p3 << "," << p4 << ","
           << TpVal << "," << CrVal << "," << nrVal << endl;
    }}

    // 2. Fusion Boundary / HAZ Indent
    {{
        indId++;
        real zTest = HzBase;
        real hvVal = HVmap(xMid, yMid, zTest);
        real sdasVal = SDAS(xMid, yMid, zTest);
        real p1 = fFerrite(xMid, yMid, zTest)*100.0;
        real p2 = fBainite(xMid, yMid, zTest)*100.0;
        real p3 = fMartensite(xMid, yMid, zTest)*100.0;
        real p4 = fPearlite(xMid, yMid, zTest)*100.0;
        real TpVal = Tpeak(xMid, yMid, zTest);
        real CrVal = TdotCool(xMid, yMid, zTest);
        real nrVal = nReheat(xMid, yMid, zTest);

        fH << indId << ",Fusion_Line,HAZ,"
           << xMid*1000. << "," << yMid*1000. << "," << zTest*1000. << ","
           << hvVal << "," << sdasVal << "," << p1 << "," << p2 << "," << p3 << "," << p4 << ","
           << TpVal << "," << CrVal << "," << nrVal << endl;
    }}

    // 3. Layer-by-Layer Deposited Wall Indents (Layer 1 to Nlayers)
    for(int iL=0; iL<Nlayers; iL++){{
        indId++;
        real zTest = (iL == 0) ? (HzBase + hBead1/2.) : (HzBase + hBead1 + (iL-1)*hBeadSub + hBeadSub/2.);
        real hvVal = HVmap(xMid, yMid, zTest);
        real sdasVal = SDAS(xMid, yMid, zTest);
        real p1 = fFerrite(xMid, yMid, zTest)*100.0;
        real p2 = fBainite(xMid, yMid, zTest)*100.0;
        real p3 = fMartensite(xMid, yMid, zTest)*100.0;
        real p4 = fPearlite(xMid, yMid, zTest)*100.0;
        real TpVal = Tpeak(xMid, yMid, zTest);
        real CrVal = TdotCool(xMid, yMid, zTest);
        real nrVal = nReheat(xMid, yMid, zTest);

        fH << indId << ",Layer_" << (iL+1) << ",Deposit,"
           << xMid*1000. << "," << yMid*1000. << "," << zTest*1000. << ","
           << hvVal << "," << sdasVal << "," << p1 << "," << p2 << "," << p3 << "," << p4 << ","
           << TpVal << "," << CrVal << "," << nrVal << endl;
    }}
}}

// 4. Detailed 1D Vertical Hardness Line Profile across Z (for ParaView Plot Over Line)
{{
    ofstream fZ(outdir+"hardness_profile_z.csv");
    fZ << "Z_mm,X_mm,Y_mm,Vickers_HV,SDAS_um,Tpeak_C,CoolingRate_C_s" << endl;
    real xMid = LxW / 2.0;
    real yMid = LyBase / 2.0;
    int nZSamples = 100;
    for(int k=0; k<=nZSamples; k++){{
        real zP = (k * HzTot) / nZSamples;
        real hvP = HVmap(xMid, yMid, zP);
        real sdasP = SDAS(xMid, yMid, zP);
        real tpP = Tpeak(xMid, yMid, zP);
        real crP = TdotCool(xMid, yMid, zP);
        fZ << zP*1000. << "," << xMid*1000. << "," << yMid*1000. << ","
           << hvP << "," << sdasP << "," << tpP << "," << crP << endl;
    }}
}}

// 5. Automatic Native 2D Matplotlib Postprocessing Engine
system("python3 postprocess/plot_generator.py " + outdir + " 2>/dev/null || python3 WAAM-Framework/postprocess/plot_generator.py " + outdir + " 2>/dev/null || true");
"""
        return script
    def save(self, filepath: str):
        """Generate and save the .edp script to disk."""
        script = self.build()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(script)
        return filepath


if __name__ == "__main__":
    params = EDPBuilder.default_params()
    builder = EDPBuilder(params)
    out = builder.save("./waam_output/waam_generated.edp")
    print(f"Generated: {out}")
