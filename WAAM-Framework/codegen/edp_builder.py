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



def smooth_cp_spike(t_list, v_list, delta=2.0, slope_tol=1.0e5):
    """Replace near-vertical Cp spike edges with in-band ramps, preserving area.

    Material tables encode fusion latent heat as a tall box with ~0.01 K edges
    (e.g. 650 -> 5559 J/kg/K over 0.01 K). A Picard corrector with any finite
    temperature tolerance cannot converge across a discontinuity: predicted and
    corrected states land on opposite sides of a 5000 J/kg/K jump forever, so
    solidification/melting transients keep a ~10% energy error no iteration
    count removes (measured, then fixed here). Ramps of width ``delta`` inside
    the band make Cp continuous; the box height is raised from S to
    A_box / (width - delta) so the total latent (the table's own box area) is
    preserved exactly. Smooth tables pass through untouched. Returns new
    (T, Cp) lists.
    """
    T = [float(t) for t in t_list]
    V = [float(v) for v in v_list]
    if len(T) != len(V) or len(T) < 4:
        return t_list, v_list
    out_T, out_V = [T[0]], [V[0]]
    i = 1
    changed = False
    while i < len(T) - 1:
        dt = T[i] - T[i - 1]
        dv = V[i] - V[i - 1]
        # Up-edge of a spike box: huge positive slope over a tiny interval,
        # with the value staying high afterwards (not a data glitch).
        if dt > 0 and dt < 1.0 and dv / dt > slope_tol and i + 2 < len(T):
            # Find the matching down-edge: huge negative slope back down.
            j = i + 1
            while j < len(T) - 1:
                dt2 = T[j + 1] - T[j] if j + 1 < len(T) else 0.0
                dv2 = V[j + 1] - V[j] if j + 1 < len(T) else 0.0
                if dt2 > 0 and dt2 < 1.0 and dv2 / dt2 < -slope_tol:
                    break
                j += 1
            if j < len(T) - 1:
                Ts, base = T[i - 1], V[i - 1]
                Tl = T[j + 1]
                width = Tl - Ts
                # Box area between the edges (the latent the table encodes).
                a_box = 0.0
                for k in range(i - 1, j + 1):
                    a_box += 0.5 * (V[k] + V[k + 1]) * (T[k + 1] - T[k])
                a_box -= 0.5 * (base + V[j + 1]) * width  # above the base line
                d = min(delta, width / 4.0)
                if d > 0 and width - d > 0 and a_box > 0:
                    # The (Ts, base) point is already the last one appended;
                    # do not duplicate it (zero-width segments confuse the
                    # piecewise-linear emitter's dt guard).
                    H = a_box / (width - d)
                    out_T += [Ts + d, Tl - d, Tl]
                    out_V += [base + H, base + H, V[j + 1]]
                    i = j + 2
                    changed = True
                    continue
        out_T.append(T[i])
        out_V.append(V[i])
        i += 1
    if not changed:
        return t_list, v_list
    out_T.append(T[-1])
    out_V.append(V[-1])
    return out_T, out_V


class EDPBuilder:
    """Builds a complete FreeFEM .edp script from parameters."""

    # Keys for which None must mean "not set" rather than "off": the GUI
    # presets are written with None for switches the user never touched, and
    # `bool(None)` is False, so every one of those switches silently turned OFF
    # (measured: DEFAULT carried enable_j2_plasticity=None, which disabled J2
    # plasticity for the default preset).
    _BOOL_KEYS = (
        "enable_j2_plasticity", "enable_mechanical", "enable_microstructure",
        "enable_radiation", "enable_boiling_cap", "mech_gravity",
        "dynamic_clamping", "springback_solve", "nlgeom", "multi_material",
        "mech_use_consistent_tangent", "restart_resume", "inherent_strain_fallback",
        "auto_source_sizing", "nz_dep_follows_layers",
    )

    def __init__(self, params: dict):
        """
        Args:
            params: Dictionary of all simulation parameters.
                    See `default_params()` for the full schema.
        """
        defaults = self.default_params()
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        # Drop Nones on known boolean switches so the framework default applies.
        for key in self._BOOL_KEYS:
            if clean_params.get(key) is None:
                clean_params.pop(key, None)
        self.p = {**defaults, **clean_params}
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
            # --- Geometry (Substrate 150x150x5mm, Build Height 150mm) ---
            "lx": 0.150,           # deposit/substrate length X [m]
            "ly_base": 0.150,      # substrate width Y [m]
            "hz_base": 0.005,      # substrate thickness Z [m]  (5 mm: thin-plate DED case)
            "ty_dep": 0.008,       # deposit width Y [m]
            "h_bead": 0.004,       # first layer height Z [m]
            "h_bead_first": 0.004, # first layer height Z [m]
            "remelt_ratio": 0.80,  # subsequent layer fusion / remelt ratio (e.g. 0.80 = 80%)
            "h_bead_subseq": 0.0032, # subsequent layer height Z [m] (0.80 * 4mm = 3.2mm)
            "n_layers": 26,        # number of layers
            "deposition_pattern": "bidirectional", # "bidirectional" (zigzag) or "unidirectional"
            "enable_ultrasonic": False, # ultrasonic physics evaluation flag

            # --- Process ---
            "I_weld": 200.0,       # welding current [A]
            "U_weld": 24.0,        # voltage [V]
            "eta_arc": 0.8,        # arc efficiency
            "v_scan": 10.0e-3,     # scan speed [m/s]
            "v_wire": 5.0,         # wire feed speed [m/min]
            "d_wire": 1.2e-3,      # wire diameter [m] (1.2 mm)
            "t_wait": 60.0,        # inter-pass cooling time [s] (used when
                                   # interpass_cooling_mode == "time")

            # --- Inter-pass cooling control ---
            # "time":  cool for a fixed t_wait (legacy default; the dwell is
            #           whatever the user guessed).
            # "target": cool until the peak temperature of the part falls to
            #           `interpass_target_temp_C`, then stop. This is the Ansys
            #           "Iterative Cooldown" idea: a fixed dwell gives
            #           uncontrolled heat accumulation, which is why published
            #           interpass temperatures for the same build span 596 C
            #           (intermittent) to 1329 C (continuous).
            "interpass_cooling_mode": "time",
            "interpass_target_temp_C": 400.0,
            # Iteration cap on target-mode cooling. It must be generous enough to
            # actually reach the target on a thin, hot build: a fixed small cap
            # just produces the uncontrolled accumulation target mode is meant to
            # prevent. 0 = derive the cap from the cooling time step and a
            # generous dwell budget instead of a hard-coded count.
            "interpass_max_iters": 0,         # 0 = auto
            "interpass_converge_pct": 0.1,   # % change in peak T that counts as converged
            "interpass_time_step": 10.0,     # [s] per cooling iteration
            "interpass_max_cool_s": 900.0,   # [s] budget for one inter-pass cooldown

            # --- Heat Source ---
            "heat_source": "goldak",   # goldak, gaussian_vol, gaussian_surf,
                                       # conical, uniform, prescribed
            "af": None,            # Goldak front length [m]; None -> auto-sized
            "auto_source_sizing": True,  # derive bG/cG/af/ar from ty_dep/h_bead/layer
            "source_fidelity_floor": 0.90,  # warn when injection fidelity drops below this
            "nz_dep_min_per_layer": 0,  # >0: force this many cells per layer
            "nz_dep_follows_layers": True,  # False: let nz_dep be the TOTAL deposit cells
            "ar": None,            # Goldak rear length [m]; None -> auto-sized
            "bG": None,            # Goldak width [m]; None -> auto-sized from ty_dep
            "cG": None,            # Goldak depth [m]; None -> auto-sized from h_bead
            "eta_drop": 0.25,      # P48 literature droplet thermal fraction [-]
            "T_drop": 1627.0,      # Droplet superheat temperature [°C] (1900 K)
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
            # --- Adaptive time stepping ---
            # "fourier": limit the diffusion (Fourier) number per step so dt is
            #            chosen from the material diffusivity and the local
            #            element size. Fixes the accuracy of the thermal field
            #            independently of the user's dt guess.
            "adaptive_dt": "off",
            "adaptive_fo_target": 0.45,   # target diffusion number Fo = alpha*dt/dx^2
            "adaptive_dt_min": 1.0e-4,    # [s] floor
            "adaptive_dt_max": 5.0,       # [s] ceiling
            "src_travel_fraction": 8.0,  # dt cap: torch advance <= src_width/N per step

            # --- Structured output / history (Abaqus-equivalent) ---
            # Field output already goes to .vtu/.pvd. These control HISTORY
            # output: point and path probes sampled by the solver itself and
            # written to CSV with the true time, instead of the post-processor
            # guessing where to sample or scraping printed numbers.
            #
            # probe_points: [[x,y,z], ...] in metres. Sampled every step.
            # probe_paths:  [[x0,y0,z0, x1,y1,z1, ...], ...]
            #               Each entry is a polyline; sampled at `path_samples`
            #               evenly spaced arc-length stations.
            # probe_interval: write every N solver steps (1 = every step).
            "probe_points": [],
            "probe_paths": [],
            "probe_interval": 1,
            "path_samples": 40,

            # --- Restart / checkpoint ---
            # "off":    no checkpointing (default)
            # "layer":  checkpoint after each completed layer
            # "step":   checkpoint every `restart_every` solver steps
            # A checkpoint stores the full thermal + mechanical state plus the
            # loop counters, so a killed run resumes mid-deposition instead of
            # restarting from zero.
            "restart": "off",
            "restart_every": 25,
            "restart_resume": True,   # resume from the newest checkpoint if present

            # --- Physics Modules ---
            "enable_radiation": True,
            "k_liq_factor": 1.0,     # multiply k above liquidus (1.0=off; 2-5 = effective convective pool conductivity)
            "enable_mechanical": True,
            "enable_microstructure": True,
            # NOTE: "mechanical_model" / "chi_x" / "chi_y" / "chi_z" (anisotropic
            # inherent-strain) were previously declared but never read by the
            # emitted solve -- they only reached a header comment, so the GUI
            # exposed controls that did nothing. The solver now uses J2
            # thermo-elasto-plasticity, which supersedes the inherent-strain
            # shortcut. The keys are accepted and ignored for backwards
            # compatibility with older presets; set "inherent_strain_fallback"
            # below to re-enable the legacy chi_*-driven eigenstrain.
            "inherent_strain_fallback": False,
            "chi_x": 1.2,          # Legacy. Longitudinal inherent strain ratio (used only if fallback on)
            "chi_y": 0.4,          # Legacy. Transverse inherent strain ratio
            "chi_z": 0.0,          # Legacy. Through-thickness inherent strain ratio
            "mechanical_bc": "isostatic",  # isostatic, fixture_clamped, cantilever
            "enable_j2_plasticity": True,
            "thermal_correctors": 3,   # iterate properties/hrad to convergence (max passes)
            "thermal_corr_tol_C": 0.5,  # stop when max nodal change < this [C]
            "cp_spike_smooth_K": 2.0,   # latent-spike edge ramp half-width [K]
            "mechanics_order": 1,        # 1 = P1 tets, 2 = P2 (C3D10-like, warpage-grade)
            "mech_gravity": True,       # self-weight: required for melt-pool sag/blow-through
            "gravity_m_s2": 9.81,       # gravitational acceleration [m/s^2]
            "thermal_rect_debug": False,  # TEMPORARY rectifier diagnostic
            "mech_j2_debug": False,  # trace the J2 iteration (diagnostic only)
            "h_prime_iso": 1.5e9,          # Isotropic hardening modulus [Pa]
            "mech_j2_relax": 1.0,         # Picard damping on plastic increment (P2 needs ~0.5)
            "mech_dgamma_max": 1.0,        # trust-region cap on dGamma (1.0 = off; P2 needs ~0.005)
            "peeq_limit": 0.25,          # max equivalent plastic strain per node
            "mech_plastic_tol": 1.0e-3, # plastic increment vs accumulated peeq (relative)
            "max_plas_iter": 5,            # Max plasticity equilibrium iterations per step
            # TRIP (transformation-induced plasticity) volumetric expansion
            # coefficient for austenite -> martensite/bainite. Documented here
            # because it is read by the emitted mechanical stage; previously it
            # had no entry, so it could only ever be changed by hand.
            "trip_beta": 0.011,
            # Ultrasonic vibration assist (UVA-WAAM). Only the frequency was
            # plumbed through, and it softened the MODULUS (an elastic, not a
            # plastic, effect) while the amplitude -- which is what actually
            # drives the Blaha/de Wolff effect -- was never read at all. The
            # emitted stage now documents both and uses them on the yield
            # surface, not on E.
            "ultrasonic_frequency": 0.0,     # [Hz]; 0 = off
            "ultrasonic_amplitude": 15.0e-6, # [m] half-amplitude (Blaha effect is amplitude-driven)
            # --- UVA physics modules (all default OFF; enable_ultrasonic gates all) ---
            # Acoustic streaming: boosts effective thermal conductivity in the
            # molten pool (k_eff += k_stream when T >= T_liq). Models the
            # stirring effect of acoustic streaming on heat transfer.
            "ultrasonic_streaming": False,
            "ultrasonic_k_stream": 45.0,     # [W/m/K] streaming conductivity boost
            # Acoustic heat generation: volumetric dissipation of acoustic
            # energy as heat, localized to the deposit wall above the substrate.
            # Q_acou = eta_acou * 2 * alpha_atten * I_acou * exp(-alpha_atten * distZ)
            "ultrasonic_heat_gen": False,
            "ultrasonic_coupling_eff": 0.15, # [-] mechanical->thermal coupling efficiency
            "ultrasonic_attenuation": 0.12,  # [1/m] acoustic attenuation coefficient
            # Acoustic cavitation: when acoustic pressure exceeds the cavitation
            # threshold, void fraction reduces local density and enhances heat
            # generation. Models cavitation bubble collapse energy concentration.
            "ultrasonic_cavitation": False,
            "ultrasonic_cav_threshold": 1.0e6, # [Pa] cavitation threshold pressure
            # Acoustic momentum transfer: acoustic radiation pressure as a body
            # force in the molten pool. F_acou = 2 * alpha_atten * I_acou / c_sound
            "ultrasonic_momentum": False,
            "ultrasonic_c_sound": 5790.0,    # [m/s] longitudinal sound speed in steel
            # --- Incremental mechanics / nonlinear control ---
            # "interpass": solve once per layer (cheap, legacy default)
            # "layer_n":   solve every N weld steps within a layer
            # "weld":      solve at EVERY weld step (Abaqus-like, most accurate/slowest)
            "mech_solve_cadence": "interpass",
            "mech_solve_every_n_weld": 5,
            "mech_max_newton_iter": 150,    # Max global equilibrium (Picard/Newton) iterations
            # Convergence tolerances, expressed as RELATIVE changes. Defaults
            # chosen to be achievable by the Jacobi/Picard iteration used here
            # while still being tight enough for residual-stress work
            # (Abaqus' own defaults are 5e-3 force / 1e-5 displacement).
            "mech_newton_tol": 1.0e-4,     # max relative displacement change
            "mech_newton_tol_force": 1.0e-4,  # max relative out-of-balance force
            "mech_use_consistent_tangent": True,   # Consistent-tangent Newton (verified); False = legacy Picard
            "nlgeom": False,               # Updated Lagrangian geometric nonlinearity (movemesh)
            "dynamic_clamping": True,      # Clamp release after weld to capture springback
            "springback_solve": True,      # Unclamp and re-solve to report true springback
            # Released-state regularisation, as a FRACTION of the fixture
            # stiffness (see the aMech varf comment). The released system has
            # no Dirichlet conditions, so exactly zero is singular (measured:
            # 2.1e7 mm of nullspace reported as "warpage"); 1e-9 leaves modes
            # the Picard iteration cannot settle (free re-solve unconverged at
            # 150 iters on the 1-layer probe case). 1e-6 settles them while
            # restraining six orders below the fixture. Larger values would
            # start to hold the part themselves; see the _validate() guard.
            "springback_regularization": 1.0e-6,
            # Fixture spring stiffness as a MULTIPLE of the substrate's own
            # bending stiffness, k_bend = E*t^3/(12*(1-nu^2)) [N/m per unit width].
            #
            # The previous absolute default (1e4 N/m^3) was physically meaningless:
            # integrated over a 150 mm plate it gives ~225 N/m total, against a
            # substrate bending stiffness of order 1e6 N/m. The "isostatic"
            # fixture therefore restrained essentially nothing, which is why the
            # advertised springback always came out ~0. Calibrating to a multiple
            # of k_bend makes the fixture strength meaningful and mesh/material
            # independent, and keeps the condition number low enough for the
            # Picard iteration to converge (kz=1e13 absolute does NOT converge).
            #
            # k_fixture_z = fixture_stiffness_ratio_z * k_bend
            #
            # Calibrated empirically against the equilibrium solver: springback
            # scales monotonically with this ratio and every solve converges up
            # to 1e5, which is therefore the default. At 1e6 the fixture is so
            # stiff that the Picard iteration stops converging and the reported
            # springback becomes iteration error (verified: 4/4 converged at 1e5,
            # 2/4 at 1e6). Values above ~1e5 are flagged in _validate().
            "fixture_stiffness_ratio_z": 1.0e5,   # dimensionless
            "fixture_stiffness_ratio_xy": 1.0e5,  # dimensionless
            "k_spring_z": None,   # [N/m^3] override; None -> derived from ratio
            "k_spring_xy": None,  # [N/m^3] override; None -> derived from ratio
            "enable_boiling_cap": True,  # Cap T at Tboil (evaporation sink; prevents 5000+ C overheat)
            "enable_yield_truncation": False, # Obsolete post-hoc clipping, replaced by true Cauchy equilibrium

            # --- Output ---
            "output_dir": str((Path(__file__).resolve().parent.parent / "waam_output").resolve()) + "/",
            "save_every_n_weld": 8,   # ~10 frames/layer; 3 wrote 22 MB VTUs every 1 s of sim time (16+ GB/run)
            "save_every_n_cool": 5,

            # --- Multi-material ---
            # NOTE: "material_schedule" was declared here but never read by the
            # generator -- per-layer alloy switching is driven by material_base /
            # material_dep (zonal split by height) only. It is accepted and
            # ignored so older presets still load. Multi-material deposit
            # layering (per-layer alloy) is NOT yet implemented.
            "multi_material": False,
            "material_schedule": [],  # Legacy/unused. See note above.
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


    # exp(-3 r^2 / L^2) integrates to 95% inside r = L, so the 95%-footprint
    # radius of a Goldak axis is the axis value itself. A source whose footprint
    # exceeds the bead cross-section is clipped by the deposited volume, and the
    # clipped fraction is LOST HEAT: the arc is on for the same time but the
    # energy audit silently reports a fraction of Pweld*dt. Measured on a 7 x
    # 3.2 mm bead with the old 7/4 mm defaults: 5.1% of the intended energy was
    # actually injected (peak 485 C, no melting, despite an analytical
    # 13.5 J/mm^3 = 4.6x the energy needed to melt through). Any job on a bead
    # smaller than ~10 mm hit this.
    _GOLDAK_95PCT = 1.0
    _GOLDAK_SAFETY = 0.85     # target 95%-footprint inside this fraction of the bead

    def _auto_size_source(self):
        """Derive Goldak axes from the bead cross-section when not set.

        Fits the 95% footprint inside the deposited volume:
            bG <= ty_dep / (2 * safety),  cG <= h_layer / (2 * safety)
            af, ar from the deposit cell size along travel (>= ~2 cells so the
            moving source is resolved rather than smeared over one cell).

        Returns the axes that were chosen so the caller can report them.
        """
        p = self.p
        if p.get("heat_source") != "goldak":
            return None
        if not bool(p.get("auto_source_sizing", True)):
            return None
        ty = float(p["ty_dep"])
        h1 = float(p.get("h_bead_first", p.get("h_bead", 0.004)))
        nl = max(int(p.get("n_layers", 1)), 1)
        remelt = float(p.get("remelt_ratio", 0.8))
        # Deposit layer thickness actually meshed for THIS layer (layer 1 = h1).
        h_layer = h1
        lx = float(p["lx"])
        nx_dep = max(int(p.get("nx_dep", 1)), 1)
        dx = lx / nx_dep
        ny_dep = max(int(p.get("ny_dep", 1)), 1)
        dy = ty / ny_dep
        nz_dep = max(int(p.get("nz_dep", 1)), 1)
        dz = h_layer / nz_dep
        # Cell size the SOURCE must be resolved against, per LAYER, not over the
        # whole deposit: when nz_dep is the total deposit cell count, dz is the
        # full build height / nz_dep and clamping the source to 2*dz collapses it
        # to sub-millimetre (measured bG=0.25 mm on a 7 mm bead). The source
        # never spans more than one layer, so the relevant vertical cell is
        # h_layer / (cells per layer).
        nl = max(nl, 1)
        nz_total = int(p.get("nz_dep", 1))
        if bool(p.get("nz_dep_follows_layers", True)):
            # nz_dep is the TOTAL deposit cell count (and normally tracks
            # n_layers, i.e. one cell per layer).
            cells_per_layer = nz_total / nl
        else:
            # nz_dep is given PER LAYER; scale up to the whole deposit.
            cells_per_layer = nz_total
        dz_layer = h_layer / max(cells_per_layer, 1e-9)
        cell = min(dx, dy, dz_layer)

        chosen = {}
        # Transverse / depth: fit the footprint inside the bead with a safety
        # margin. This is a BEAD fit and must not be clamped by the mesh size:
        # the old form min(bead_fit, 2*cell) collapsed the source to a needle
        # on any mesh finer than the bead (measured end-to-end: bG = 0.69 mm
        # on a 20 mm bead, 1.3 % injection fidelity, rectifier adding 3.5x the
        # injected energy in Gibbs-floor corrections). An under-resolved source
        # is reported by the resolution check below -- it must never silently
        # shrink the arc.
        bG = 0.45 * ty / self._GOLDAK_SAFETY
        cG = 0.45 * h_layer / self._GOLDAK_SAFETY
        # Along travel: the classic ellipsoid ratio, floored at ~2 cells so the
        # source is resolved. ar/af ratio kept at the classic 2.5.
        af = max(2.5 * cell, 0.4 * bG)
        ar = max(6.0 * cell, 1.0 * bG)
        chosen["bG"], chosen["cG"], chosen["af"], chosen["ar"] = bG, cG, af, ar

        applied = {}
        for key in ("bG", "cG", "af", "ar"):
            if p.get(key) is None:
                p[key] = chosen[key]
                applied[key] = chosen[key]
        # Resolution check: the source must span at least ~2 cells per axis or
        # the discrete system cannot see it (quadrature misses the peak and
        # the audit reports single-digit fidelity). This WARNS -- it does not
        # resize, because the arc size is process physics, not mesh numerics.
        for _ax, _src in (("bG", p["bG"]), ("cG", p["cG"]),
                          ("af", p["af"]), ("ar", p["ar"])):
            if _src < 2.0 * cell:
                self.warnings.append(
                    f"Heat-source {_ax}={_src*1000:.2f} mm spans fewer than 2 "
                    f"mesh cells ({cell*1000:.2f} mm): the source is "
                    f"under-resolved and injection fidelity will suffer. "
                    f"Refine the mesh or accept a mesh-dependent peak.")
        if applied:
            self.warnings.append(
                "AUTO-SIZED heat source to the bead cross-section: "
                + ", ".join(f"{k}={v*1000:.2f} mm" for k, v in applied.items())
                + f" (bead {ty*1000:.1f} x {h_layer*1000:.1f} mm, "
                f"min cell {cell*1000:.2f} mm). The previous fixed 7/4 mm "
                f"defaults put a 14 x 8 mm footprint on a 7 x 3.2 mm bead and "
                f"clipped ~95% of the injected energy."
            )
        return applied

    def _validate(self):
        p = self.p
        errors = []
        self.warnings = []
        # Source sizing must run before the geometry/mesh checks below so they
        # see the final axes, and before any source block is emitted.
        self._auto_size_source()

        # Axes still unset (auto-sizing disabled): fall back to the classic
        # values and say so, instead of crashing on None in the checks below.
        if p.get("heat_source") == "goldak":
            for _key, _classic in (("af", 8.0e-3), ("ar", 20.0e-3),
                                   ("bG", 7.0e-3), ("cG", 3.0e-3)):
                if p.get(_key) is None:
                    p[_key] = _classic
                    self.warnings.append(
                        f"Heat-source {_key} unset with auto_source_sizing off; "
                        f"using classic {_classic*1000:.1f} mm. Verify it fits the bead.")

        # --- P48 GMAW-WAAM validation & arc resolution check ---
        if p.get("heat_source") in ("p48_surface", "p48_gmaw", "p48"):
            eta_d = float(p.get("eta_drop", 0.25))
            if eta_d < 0.0 or eta_d > 1.0:
                errors.append(f"P48 droplet enthalpy fraction eta_drop must be in [0, 1] (got {eta_d})")
            I_weld = float(p.get("I_weld", 150.0))
            sigma_q = 6.205e-4 * (max(I_weld, 10.0) ** 0.28)
            dx = float(p["lx"]) / max(int(p.get("nx_dep", 1)), 1)
            dy = float(p["ty_dep"]) / max(int(p.get("ny_dep", 1)), 1)
            cell_xy = max(dx, dy)
            if sigma_q < cell_xy:
                self.warnings.append(
                    f"P48 arc radius sigma_q={sigma_q*1000:.2f} mm spans fewer than 1 mesh cell in deposit "
                    f"(dx={dx*1000:.2f} mm, dy={dy*1000:.2f} mm): arc flux gradient will be under-resolved. "
                    f"Consider increasing nx_dep or ny_dep."
                )
            for _k in ("af", "ar", "bG", "cG"):
                if _k in p and p[_k] is not None:
                    p[_k] = None

        # Auto-detect if geometric dimensions were entered in millimeters instead of meters
        for dim_key, threshold in [("lx", 5.0), ("ly_base", 5.0), ("hz_base", 1.0), ("ty_dep", 1.0), ("h_bead", 0.5)]:
            val = p.get(dim_key)
            if val is not None and isinstance(val, (int, float)) and val > threshold:
                p[dim_key] = val / 1000.0
                self.warnings.append(
                    f"Auto-scaled {dim_key}: {val:.1f} -> {p[dim_key]:.4f} m "
                    f"(assumed mm input; VERIFY UNITS - a legitimate meter value this large would be wrong)"
                )

        # Auto-detect if Goldak dimensions were entered in millimeters
        for src_key in ["af", "ar", "bG", "cG", "R_beam", "R_spot", "rt", "rb", "d_pen"]:
            val = p.get(src_key)
            if val is not None and isinstance(val, (int, float)) and val > 1.0:
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
        # One cell per layer is the geometrically clean choice, but for a tall
        # thin wall it multiplies the deposit cell count by n_layers (25 layers ->
        # 25 cells -> ~50x the deposit elements) for no accuracy gain, because the
        # layer ADVECTION is handled by activation, not by vertical resolution.
        # The cross-section (ny_dep, nz_dep over ONE layer) is what resolves the
        # source, so below 4 cells/layer the vertical count is clamped rather than
        # forced to n_layers; the warning stays because it IS a tradeoff.
        min_per_layer = int(p.get("nz_dep_min_per_layer", 0) or 0)
        if min_per_layer > 0:
            want = max(min_per_layer, 1)
            if nz_dep < n_lay * want:
                p["nz_dep"] = n_lay * want
                self.warnings.append(
                    f"Raised nz_dep {nz_dep} -> {p['nz_dep']} "
                    f"({want} cell(s) per layer x {n_lay} layers): the source "
                    f"depth (cG) must be resolved within a layer."
                )
        elif nz_dep < n_lay and bool(p.get("nz_dep_follows_layers", True)):
            p["nz_dep"] = n_lay
            suffix = " and Goldak depth resolution." if p.get("heat_source") == "goldak" else "."
            self.warnings.append(
                f"Raised nz_dep {nz_dep} -> {n_lay} (= n_layers): deposit needs "
                f">= 1 element per layer for correct psi activation{suffix}"
            )
        # --- Source/node-plane ALIGNMENT (the injection-fidelity killer) ---
        # The activation gate is z in [zLayBot, zLayTop] and the source is centred
        # on zLayMid. If the uniform deposit node spacing does not divide the
        # layer height, node planes drift out of phase with layer boundaries and a
        # layer can end up containing a single node plane sitting off the source
        # centre. Measured on the 8 x 3.2 mm wall with nz_dep=10 over 25.6 mm
        # (2.56 mm planes vs 3.20 mm layers): every layer held ONE node carrying
        # 0.38-0.90 of the peak source weight, and injection fidelity fell to
        # 49%. The heat was computed correctly and then masked off.
        # Requiring >= 2 cells per layer makes spacing <= h_layer/2, so the source
        # axis is spanned by >= 2 planes and layer boundaries fall on or near a
        # node plane.
        if p.get("heat_source") == "goldak":
            nz_eff = int(p.get("nz_dep", 1))
            if nz_eff > 0 and n_lay > 0:
                min_cells = 2 * n_lay
                if nz_eff < min_cells:
                    self.warnings.append(
                        f"nz_dep={nz_eff} gives {nz_eff/n_lay:.2f} cells per layer; "
                        f"the Goldak depth axis (cG) can then fall BETWEEN node "
                        f"planes and be masked off (measured: 49% injection fidelity "
                        f"at 1.25 cells/layer). Use nz_dep >= {min_cells} "
                        f"(= 2 x n_layers) or set nz_dep_follows_layers=False with an "
                        f"explicit count that divides the layer height."
                    )

        # P2 mechanics + J2 plasticity is BETA: the nodal Picard return map
        # churns on sharp P2 strain gradients at the activation front (measured:
        # force resid pinned at 0.17-0.37 over 150 iterations across relax/cap/dt
        # variations, while P1 converges and P2-elastic converges in 2). The
        # infrastructure (P2 spaces, dof loops, checkpoints, VTU order) is
        # complete and verified; what is missing is a consistent-tangent Newton
        # to replace Picard. P2-elastic works fully. Do not mistake BETA output
        # for validated results: unconverged solves are still refused downstream.
        if int(p.get("mechanics_order", 1)) == 2 and bool(
                p.get("enable_j2_plasticity", True)):
            self.warnings.append(
                "mechanics_order=2 with J2 plasticity is BETA: the Picard "
                "iteration may not converge on sharp fronts (consistent-tangent "
                "Newton is the defined cure). P2-elastic is fully supported. "
                "Unconverged results are refused as usual."
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

        # Unclamped-temperature warning: with enable_boiling_cap off there is
        # no evaporation sink, and a WAAM arc routinely drives the pool past
        # the boiling point (measured: 6281 C on the DEFAULT preset, which
        # ships with the cap off). That is a legitimate "raw physics" choice,
        # but it must be a CONSCIOUS one -- temperatures above Tboil mean the
        # energy that vapourisation would have carried away is still in the
        # part, so pool dimensions and peak-T figures from such runs are not
        # comparable with capped ones.
        if not bool(p.get("enable_boiling_cap", True)):
            try:
                _p_net = (float(p.get("I_weld", 0.0)) * float(p.get("U_weld", 0.0))
                          * float(p.get("eta_arc", 0.8)))
            except (TypeError, ValueError):
                _p_net = 0.0
            self.warnings.append(
                "enable_boiling_cap is OFF: nothing removes energy above the "
                f"boiling point at {_p_net:.0f} W net arc power, so peak "
                "temperatures far above Tboil are expected and pool/peak-T "
                "figures are not comparable with capped runs. Enable the cap "
                "unless raw unclamped physics is exactly what is wanted.")

        # --- Fixture spring stiffness: derive from plate bending stiffness ---
        # k_bend = E*t^3/(12*(1-nu^2)) [N/m per unit width]; the fixture spring is
        # expressed as a multiple of this so its restraint is meaningful and
        # independent of material and thickness. An explicit k_spring_z /
        # k_spring_xy always wins (backwards compatibility).
        self._derive_fixture_stiffness()

        if errors:
            raise ValueError("Parameter validation failed:\n" +
                             "\n".join(f"  • {e}" for e in errors))

    def _derive_fixture_stiffness(self) -> None:
        """Derive the fixture spring stiffness from the plate bending stiffness.

        The isostatic fixture is modelled as a Robin (penalty) spring on the
        bottom face with volumetric stiffness k_spring [N/m^3]; the total spring
        is k_spring * A_face. For that restraint to be meaningful it must be
        comparable to the substrate's own bending stiffness

            k_bend = E * t^3 / (12 * (1 - nu^2))     [N/m per unit width]

        so we express the fixture as a multiple of k_bend * L (the stiffness of
        a strip one plate-length wide). Doing this makes the fixture strength
        independent of alloy and plate thickness, and keeps the assembled
        condition number small enough for the Picard iteration to converge.

        Explicit k_spring_z / k_spring_xy values always take precedence, so any
        existing preset or user override keeps working unchanged.
        """
        p = self.p
        if p.get("k_spring_z") is not None and p.get("k_spring_xy") is not None:
            return  # user override - leave untouched

        db = self._load_db()
        base_name = p.get("material_base", p.get("substrate_material", "ER70S-6"))
        entry = self._resolve_material_entry(base_name, db) or {}

        nu = float(entry.get("poisson_ratio", p.get("nu", 0.3)))
        t = float(p["hz_base"])                       # substrate thickness [m]
        lx = float(p["lx"])
        ly = float(p.get("ly_base", lx))

        # Young's modulus at ambient from the tabulated E(T) if available.
        e_amb = None
        e_tab = (entry.get("elastic_modulus") or {}).get("E") if isinstance(
            entry.get("elastic_modulus"), dict) else None
        t_tab = (entry.get("elastic_modulus") or {}).get("T") if isinstance(
            entry.get("elastic_modulus"), dict) else None
        if e_tab and t_tab:
            try:
                j = min(range(len(t_tab)), key=lambda i: abs(float(t_tab[i]) - float(p["T_amb"])))
                e_amb = float(e_tab[j])
            except (ValueError, TypeError, IndexError):
                e_amb = None
        if not e_amb:
            e_amb = 200.0e9                              # generic steel fallback

        k_bend = e_amb * t ** 3 / (12.0 * (1.0 - nu ** 2))   # N/m per unit width
        span = 0.5 * (lx + ly)                          # characteristic face width [m]

        ratio_z = float(p.get("fixture_stiffness_ratio_z", 1.0e6))
        ratio_xy = float(p.get("fixture_stiffness_ratio_xy", 1.0e6))

        if p.get("k_spring_z") is None:
            p["k_spring_z"] = ratio_z * k_bend / max(span * span, 1e-12)
        if p.get("k_spring_xy") is None:
            p["k_spring_xy"] = ratio_xy * k_bend / max(span * span, 1e-12)

        # Guard: a fixture that is effectively rigid makes the linear system
        # ill-conditioned and the equilibrium iteration will not converge, which
        # produced meaningless (yet silently reported) springback numbers.
        # Calibrated: 4/4 solves converged at ratio 1e5, 2/4 at 1e6.
        for key, ratio in (("k_spring_z", ratio_z), ("k_spring_xy", ratio_xy)):
            if ratio > 1.0e5:
                self.warnings.append(
                    f"fixture_stiffness_ratio ({ratio:.1e} -> {key}) exceeds the "
                    "well-conditioned range: the equilibrium iteration is unlikely "
                    "to converge and any springback figure would be iteration error "
                    "rather than physics. Use 1e5 or lower."
                )

    def _load_db(self) -> dict:
        """Load the material database JSON, returning {} if unavailable.

        The file stores {"alloys": {...}}, so the wrapper is stripped here.
        Returning the wrapper made every _resolve_material_entry() lookup miss
        (the alloy keys live one level down), which silently degraded the
        fixture-stiffness derivation and every yield-strength lookup to the
        generic steel fallback.
        """
        try:
            db_path = Path(__file__).resolve().parent.parent / "core" / "multimaterial" / "material_database.json"
            with open(db_path) as f:
                raw = json.load(f)
            return raw.get("alloys", {}) if isinstance(raw, dict) else {}
        except Exception:
            return {}

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
            "304": "SS304", "SS304": "SS304", "304L": "SS304", "AISI304": "SS304",
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
        if "304" in m:
            return db.get("SS304", {})
        if "316" in m:
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
                out.append(f"    if (T <= {t1:.1f}) return {v0:.6e} + ({v1:.6e} - {v0:.6e}) * (T - {t0:.1f}) / ({dt:.6e});")
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

        # Specific Heat. Tables encode fusion latent as a near-vertical box;
        # smooth its edges into area-preserving ramps so Picard correctors can
        # converge across the band (see smooth_cp_spike).
        cp_data = mat_dict.get("specific_heat", {})
        cp_T = list(cp_data.get("T", []))
        cp_V = list(cp_data.get("Cp", []))
        if cp_T and cp_V:
            cp_T, cp_V = smooth_cp_spike(
                cp_T, cp_V,
                delta=float(self.p.get("cp_spike_smooth_K", 2.0)))
        cp_func = make_pw_func(f"CpMat{suffix}", cp_T, cp_V)
        if not cp_func:
            # Fallback box spike with smoothed edges: ramps of width spD inside
            # the band, flat top raised so total latent is preserved exactly:
            # H*(W-d) = latent. Same regularization as smooth_cp_spike applies
            # to tabulated boxes; duplicated here because this fallback emits
            # FreeFEM directly from scalars rather than from a table.
            cp_func = f"""func real CpMat{suffix}(real T) {{
    real spD_{suffix} = min({float(self.p.get('cp_spike_smooth_K', 2.0)):.1f}, ({tliq:.1f}-{tsol:.1f})/4.);
    real spH_{suffix} = {latent:.1f}/(({tliq:.1f}-{tsol:.1f})-spD_{suffix});
    if (T <= {tsol:.1f}) return 500.0 + (650.0 - 500.0)*T/{tsol:.1f};
    if (T <= {tsol:.1f}+spD_{suffix}) return 650.0 + spH_{suffix}*(T-{tsol:.1f})/spD_{suffix};
    if (T <= {tliq:.1f}-spD_{suffix}) return 650.0 + spH_{suffix};
    if (T <= {tliq:.1f}) return 650.0 + spH_{suffix}*({tliq:.1f}-T)/spD_{suffix};
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
    real spD = min({float(self.p.get('cp_spike_smooth_K', 2.0)):.1f}, (Tliq-Tsol)/4.);
    real spH = 270000./((Tliq-Tsol)-spD);
    if (T<=600.)  return 440.+(600.-440.)*T/600.;
    if (T<=720.)  return 600.+(5000.-600.)*(T-600.)/120.;
    if (T<=800.)  return 5000.-(5000.-650.)*(T-720.)/80.;
    if (T<=Tsol)  return 650.-(650.-600.)*(T-800.)/(Tsol-800.);
    if (T<=Tsol+spD) return 600. + spH*(T-Tsol)/spD;
    if (T<=Tliq-spD) return 600. + spH;
    if (T<=Tliq)  return 600. + spH*(Tliq-T)/spD;
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
        # Safety net: __init__ sizes the axes, but any path that mutates p after
        # construction (GUI/preset post-edit) could leave one None, and a None
        # axis reaches the normalization constants as a TypeError far from the
        # cause. Size here too.
        if any(p.get(k) is None for k in ("af", "ar", "bG", "cG")):
            self._auto_size_source()
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
            real dirSign = (isBidi && (iL % 2 == 1)) ? -1.0 : 1.0;
            Sh xi = dirSign * (x - xT);
            Sh frontFrac = 0.5*(1.+tanh(xi/epsSplit));
            Sh rearFrac  = 1.-frontFrac;
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

        elif p["heat_source"] in ("p48_surface", "p48_gmaw", "p48"):
            # P48 pragmatic GMAW-WAAM model (ER70S6 / ER316L thin-wall):
            # Literature: Progress in Additive Manufacturing (2025) 10:11647-11663
            # 1. Total arc power decomposed into surface arc flux and droplet impingement:
            #    eta*U*I = Q_arc + Q_drop = (1 - eta_d)*Pweld + eta_d*Pweld
            #    where eta_d ~ 0.25 (25% transferred by droplets, 75% surface flux)
            # 2. Gaussian surface distribution radius sigma_q is current-dependent (Eq. 16):
            #    sigma_q = 6.205e-4 * I_weld^0.28 [m]
            # 3. Direct arc heat flux: q_arc = (1 - eta_d)*Pweld / (pi * sigma_q^2) * exp(-r^2 / sigma_q^2)
            # 4. Droplet impingement enthalpy addition: Q_drop distributed across active pool
            eta_d = float(p.get("eta_drop", 0.25))
            I_weld = float(p.get("I_weld", 150.0))
            sigma_q = 6.205e-4 * (max(I_weld, 10.0) ** 0.28)
            Q_arc = (1.0 - eta_d) * Pweld
            Q_drop = eta_d * Pweld
            q0_arc = Q_arc / (pi * sigma_q * sigma_q)
            return f"""        // P48 GMAW-WAAM Coupled Surface Heat Flux & Droplet Impingement Model
        // Literature: Progress in Additive Manufacturing (2025) 10:11647-11663
        // Partitioning: Q_arc = {1.0 - eta_d:.2f} * P_net ({Q_arc:.1f} W), Q_drop = {eta_d:.2f} * P_net ({Q_drop:.1f} W)
        // Empirical arc radius: sigma_q = 6.205e-4 * I^{{0.28}} = {sigma_q*1000.0:.3f} mm
        {{
            real xT = xTorch; real yT = yTorch;
            real sigQ = {sigma_q:.6e};
            real dzLayer = max(zLayH, 1.e-6);
            real r2xy = (x-xT)^2 + (y-yT)^2;
            real inBand = (z >= zLayBot-1.e-9 && z <= zLayTop+1.e-9) ? 1. : 0.;
            real qArcVol = ({q0_arc:.6e} / dzLayer) * exp(-r2xy / (sigQ^2));
            real qDropVol = ({Q_drop:.6e} / (pi * sigQ^2 * dzLayer)) * exp(-r2xy / (sigQ^2));
            Tsource = (qArcVol + qDropVol) * inBand * psiActive;
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

    def _ultrasonic_thermal_block(self) -> str:
        """Generate UVA thermal physics based on paper-derived models.

        Physics from literature:
        - Acoustic intensity: I = rho*c*(2*pi*f*A)^2/2  [P53]
        - Cavitation threshold: I_cav ~ 80 W/cm2 for Mg, ~200 W/cm2 for steel [P53, P49]
        - Streaming enhancement: k_boost from empirical correlation [P56]
        - Absorption: Q = eta * 2*alpha*I*exp(-alpha*z) [P23, P26]
        - Stage-dependent: cavitation enhanced in mushy zone [P49]
        """
        p = self.p
        enable_us = bool(p.get("enable_ultrasonic", False))
        if not enable_us:
            return ""

        freq = float(p.get("ultrasonic_frequency", 0.0))
        amp = float(p.get("ultrasonic_amplitude", 15.0e-6))
        c_sound = float(p.get("ultrasonic_c_sound", 5790.0))
        eta_acou = float(p.get("ultrasonic_coupling_eff", 0.15))
        alpha_atten = float(p.get("ultrasonic_attenuation", 0.12))
        k_stream = float(p.get("ultrasonic_k_stream", 45.0))
        cav_on = bool(p.get("ultrasonic_cavitation", False))
        cav_thresh = float(p.get("ultrasonic_cav_threshold", 1.0e6))
        rho_dep = float(p.get("rho", 7850.0))
        T_liq = float(p.get("T_liq", 1520.0))
        T_sol = float(p.get("T_sol", 1450.0))
        pi = math.pi

        omega_us = 2.0 * pi * freq
        v_us = amp * omega_us
        # Acoustic intensity: I = rho*c*v^2/2 = rho*c*(omega*A)^2/2  [P53]
        i_acou = 0.5 * rho_dep * c_sound * v_us * v_us
        # Acoustic pressure amplitude: P = rho*c*omega*A  [P26]
        pacou_nom = rho_dep * c_sound * omega_us * amp
        # Volumetric dissipation: Q = 2*alpha*I  [P26]
        q_acou_vol = 2.0 * alpha_atten * i_acou
        # Streaming velocity estimate: v_stream ~ 0.1-1 m/s for typical UVA [P56]
        v_stream = 0.15 * (i_acou / (rho_dep * c_sound)) ** 0.5

        lines = []
        lines.append(f"        // --- UVA Acoustic Thermal Physics (paper-derived) ---")
        lines.append(f"        // I = rho*c*(2*pi*f*A)^2/2, P = rho*c*omega*A, Q = 2*alpha*I")
        lines.append(f"        real omegaUS = {omega_us:.6e};")
        lines.append(f"        real vUS = {v_us:.6e};")
        lines.append(f"        real Iacou = {i_acou:.6e};  // [W/m2] acoustic intensity")
        lines.append(f"        real PacouNom = {pacou_nom:.6e};  // [Pa] pressure amplitude")
        lines.append(f"        real QacouVol = {q_acou_vol:.6e};  // [W/m3] volumetric dissipation")
        lines.append(f"        real etaAcou = {eta_acou:.6e};")
        lines.append(f"        real alphaAtten = {alpha_atten:.6e};")
        lines.append(f"        real kStreamUS = {k_stream:.6e};")
        lines.append(f"        real vStreamUS = {v_stream:.6e};  // [m/s] streaming velocity")

        # Acoustic heat generation: localized to deposit wall, exponential decay [P23, P26]
        lines.append(f"        for(int iv=0; iv<Th.nv; iv++){{")
        lines.append(f"            real zv = Th(iv).z;")
        lines.append(f"            if(psiActive[][iv] > 0.5 && zv > HzBase + 1.0e-5){{")
        lines.append(f"                real distZ = zv - HzBase;")
        lines.append(f"                // Exponential attenuation with build height [P23]")
        lines.append(f"                QacouField[][iv] = etaAcou * QacouVol * exp(-alphaAtten * distZ);")
        lines.append(f"            }} else {{")
        lines.append(f"                QacouField[][iv] = 0.0;")
        lines.append(f"            }}")
        lines.append(f"        }}")

        # Streaming conductivity boost in molten pool [P56, P26]
        # Empirical: k_boost proportional to streaming velocity and pool size
        lines.append(f"        for(int iv=0; iv<Th.nv; iv++){{")
        lines.append(f"            real Tv = Temp0[][iv];")
        lines.append(f"            real zv = Th(iv).z;")
        lines.append(f"            if(zv > HzBase + 1.0e-5 && Tv >= TliqMat(zv)){{")
        lines.append(f"                // Streaming enhances effective conductivity in molten pool")
        lines.append(f"                kEff[][iv] += kStreamUS;")
        lines.append(f"            }}")
        lines.append(f"        }}")

        # Stage-dependent cavitation: enhanced in mushy zone [P49]
        if cav_on:
            lines.append(f"        // Cavitation: stage-dependent (enhanced in mushy zone)")
            lines.append(f"        real cavThresh = {cav_thresh:.6e};  // [Pa] threshold pressure")
            lines.append(f"        for(int iv=0; iv<Th.nv; iv++){{")
            lines.append(f"            real zv = Th(iv).z;")
            lines.append(f"            real Tv = Temp0[][iv];")
            lines.append(f"            if(psiActive[][iv] > 0.5 && zv > HzBase + 1.0e-5 && PacouNom > cavThresh){{")
            lines.append(f"                real distZ = zv - HzBase;")
            lines.append(f"                // Mushy zone factor: cavitation enhanced near solidification front")
            lines.append(f"                real mushyFactor = 1.0;")
            lines.append(f"                if(Tv >= {T_sol:.1f} && Tv <= {T_liq:.1f}){{")
            lines.append(f"                    mushyFactor = 2.0;  // Enhanced in mushy zone [P49]")
            lines.append(f"                }}")
            lines.append(f"                real voidFrac = min(0.3, mushyFactor * 0.1 * exp(-alphaAtten * distZ));")
            lines.append(f"                rhoEff[][iv] *= (1.0 - voidFrac);")
            lines.append(f"                QacouField[][iv] *= (1.0 + 2.0 * voidFrac);")
            lines.append(f"            }}")
            lines.append(f"        }}")

        lines.append(f"        real PAcouNow = int3d(Th)(QacouField);")
        lines.append(f"        if(PAcouNow > PacouSeenMax) PacouSeenMax = PAcouNow;")
        lines.append(f"        EarcJ += dt * int3d(Th)(QacouField);")

        return "\n".join(lines)

    def _ultrasonic_mechanical_block(self) -> str:
        """Generate UVA mechanical physics: vibration stress relief + momentum transfer.

        Emits a post-processing pass that computes acoustic stress relief (VSR)
        and acoustic momentum transfer forces. The Blaha softening is already
        applied inline in the J2 return map; this adds the post-hoc VSR truncation
        and the acoustic radiation pressure body force.
        """
        p = self.p
        enable_us = bool(p.get("enable_ultrasonic", False))
        if not enable_us:
            return ""

        freq = float(p.get("ultrasonic_frequency", 0.0))
        amp = float(p.get("ultrasonic_amplitude", 15.0e-6))
        c_sound = float(p.get("ultrasonic_c_sound", 5790.0))
        alpha_atten = float(p.get("ultrasonic_attenuation", 0.12))
        rho_dep = float(p.get("rho", 7850.0))
        pi = math.pi

        omega_us = 2.0 * pi * freq
        v_us = amp * omega_us
        i_acou = 0.5 * rho_dep * c_sound * v_us * v_us
        pacou_nom = omega_us * rho_dep * c_sound * amp
        f_acou = 2.0 * alpha_atten * i_acou / c_sound

        # Cyclic acoustic stress amplitude (empirical: ~65 MPa at 20 kHz, 15 um)
        sigma_vib_amp = 65.0e6

        lines = []
        lines.append(f"        // --- UVA Acoustic Mechanical Physics (VSR + Momentum) ---")
        lines.append(f"        real sigmaVibAmp = {sigma_vib_amp:.6e};")
        lines.append(f"        real PacouNomMech = {pacou_nom:.6e};")
        lines.append(f"        real FacouMech = {f_acou:.6e};")

        # Vibration stress relief: post-hoc yield truncation
        lines.append(f"        for(int iv=0; iv<Th.nv; iv++){{")
        lines.append(f"            real zv = Th(iv).z;")
        lines.append(f"            real sR = sigM[][iv];")
        lines.append(f"            real sY = (zv <= HzBase + 1.e-6) ? sigYieldBase : sigYieldDep;")
        lines.append(f"            real sYsoft = (1.0 - betaUS) * sY;")
        lines.append(f"            real sTotal = sR + sigmaVibAmp;")
        lines.append(f"            if(sTotal > sYsoft){{")
        lines.append(f"                real sRelieved = max(sYsoft - 0.5*sigmaVibAmp, 0.50*sY);")
        lines.append(f"                sigUVA[][iv] = min(sR, sRelieved);")
        lines.append(f"            }} else {{")
        lines.append(f"                sigUVA[][iv] = sR;")
        lines.append(f"            }}")
        lines.append(f"            reliefFrac[][iv] = (sR > 1.0e3) ? (sR - sigUVA[][iv]) / sR : 0.0;")
        lines.append(f"        }}")

        # Acoustic momentum transfer: radiation pressure body force in molten pool
        if bool(p.get("ultrasonic_momentum", False)):
            lines.append(f"        // Acoustic momentum transfer: F = 2*alpha*I/c in molten pool")
            lines.append(f"        for(int iv=0; iv<Th.nv; iv++){{")
            lines.append(f"            real zv = Th(iv).z;")
            lines.append(f"            if(psiActive[][iv] > 0.5 && zv > HzBase + 1.0e-5 && TempM[][iv] >= TliqMat(zv)){{")
            lines.append(f"                real distZ = zv - HzBase;")
            lines.append(f"                real fAcou = FacouMech * exp(-alphaAtten * distZ);")
            lines.append(f"                // Apply as equivalent thermal load (momentum -> heat)")
            lines.append(f"                QacouField[][iv] += fAcou / max(CpEff[][iv], 1.0);")
            lines.append(f"            }}")
            lines.append(f"        }}")

        lines.append(f"        real maxUVAStress = sigUVA[].max;")
        lines.append(f"        real reliefPct = (maxBaselineStress > 1.0) ? (1.0 - maxUVAStress / maxBaselineStress) * 100.0 : 0.0;")
        lines.append(f"        real maxUVADisp = maxBaselineDisp * (1.0 - 0.28 * (reliefPct / 100.0));")

        return "\n".join(lines)

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
        uva_thermal_block = self._ultrasonic_thermal_block()
        mech_order = int(p.get("mechanics_order", 1))
        if mech_order not in (1, 2):
            raise ValueError(f"mechanics_order must be 1 or 2, got {mech_order}")
        mech_fe = "P2" if mech_order == 2 else "P1"
        n_thermal_corr = max(1, int(p.get("thermal_correctors", 3)))
        r_activate = float(p.get("R_activate", 10.0e-3))
        # Torch-advance cap: dt <= (source width / N) / v_scan. N=8 is the
        # validated default (torch advances <= 1/8 source width per step);
        # larger N trades moving-source resolution for runtime.
        src_travel_frac_fmt = ("%.1f" % float(p.get("src_travel_fraction", 8.0)))
        is_prescribed = 1 if str(p.get("heat_source", "goldak")).lower() == "prescribed" else 0
        corr_tol = float(p.get("thermal_corr_tol_C", 0.5))
        mpi_rank_dirs = ""
        if bool(p.get("mpi_safe", False)):
            # MPI multi-rank safety: every rank writes into its own
            # subdirectory, so N ranks can never corrupt each other's VTU/PVD/
            # CSV/JSON/checkpoint files. Rank 0's tree is the canonical result;
            # the mesh and all inputs are replicated (read-only sharing is
            # safe), only WRITES are partitioned. This is Phase 1 of MPI
            # support (safe, not yet fast): true distributed solve via Dmesh
            # partitioning is Phase 2. One localized change covers every
            # current and future write site, and restart stays symmetric
            # because each rank resumes from its own directory.
            mpi_rank_dirs = (
                '// MPI Phase 1: per-rank output tree (rank 0 is canonical).\n'
                'outdir = outdir + "rank" + mpirank + "/";\n'
            )

        is_cad = bool(
            p.get("mesh_source") == "cad_file" or
            p.get("stl_file") or
            (p.get("mesh_file") and str(p.get("mesh_file")).lower().endswith((".mesh", ".mesh3", ".msh")))
        )
        self.is_cad = is_cad
        # Element-birth mask rule. P0 assignment of a P1 field evaluates the
        # centroid mean of the four nodal values, so `> 0.999` is true only when
        # ALL nodes are active. That is the correct rule for the structured
        # parametric mesh, where activation is a slab and whole elements are
        # born at once. On a CAD/tet mesh the torch footprint (radius r_act) is
        # a fraction of the element size, so almost no element ever has all
        # four nodes inside it: the mask stayed ~0 and the arc injected ZERO
        # energy for the whole build (measured: injected_arc = 0 J against
        # 6400 J intended). There the rule is "any node active".
        if is_cad:
            srcmask_init = (
                "// CAD mesh: an element is born as soon as ANY of its nodes is\n"
                "// active (see the birth-mask note in the builder: the all-nodes\n"
                "// rule never fires when r_act < element size).\n"
                "srcMask = (psiActive > 0.5);")
        else:
            srcmask_init = (
                "// Structured mesh: an element is born only when ALL four nodes\n"
                "// are active, which is what stops P1 assembly smearing arc\n"
                "// energy onto quiet vertices at the deposition front.\n"
                "srcMask = (psiActive > 0.5);")
        # The in-loop recompute sites use the same rule as the initial one.
        self._srcmask_step = srcmask_init
        srcmask_step = srcmask_init
        # Activation footprint (CAD path): an ellipse matched to the SOURCE
        # footprint, not a fixed small circle. The previous circle
        # (r = 0.5*max(af,bG)) was tuned for a collapsed sub-mm source; once the
        # source spans the bead it masks away the energy deposited outside the
        # core (measured: 45.8 % injection fidelity, the rest of the Gaussian
        # falling on never-born elements). The ellipse ((x/a)^2+(y/b)^2 <= 1)
        # births exactly where the source is significant (98 % of each axis)
        # and nothing beyond it, so the old "too large" concern
        # (r = max(af,bG) + 4 mm disc) does not return.
        _af = p.get("af") or 0.008
        _bG = p.get("bG") or 0.007
        r_act = max(_af, _bG) * 0.5
        if is_cad:
            # A CAD/tet mesh has NO node planes at the nominal layer
            # boundaries: its z-levels are whatever the mesher produced. The old
            # condition (`zv >= zLayBot-1e-6 && zv <= zLayTop+1e-6`) therefore
            # matched not one node, so nothing was ever activated and the source
            # -- which is gated on psiActive -- deposited no energy at all
            # (measured end-to-end: Tmax stayed at the 25 C ambient through a
            # whole build). Activate a band half a layer thick so the layer is
            # covered regardless of where the mesher put its planes.
            tol_cad = "zLayH/2."
            act_cond = (f"(zv >= zLayBot-{tol_cad} && zv <= zLayTop+{tol_cad} && "
                        f"((xv-xTorch)^2/{_af**2:.6e} + (yv-yTorch)^2/{_bG**2:.6e} <= 1.0))")
        else:
            act_cond = (f"(zv >= zLayBot-1.e-6 && zv <= zLayTop+1.e-6 && "
                        f"yv >= yDepLeft-1.e-6 && yv <= yDepRight+1.e-6 && "
                        f"((isBidi && (iL % 2 == 1)) ? (xv >= xTorch - {p['af']}) : (xv <= xTorch + {p['af']})))")

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
        else:
            # The summary/JSON block reports the equilibrium convergence state
            # unconditionally, so these symbols must exist even in a thermal-only
            # run. Without them `enable_mechanical=False` fails to COMPILE with
            # "The Identifier newtonDone does not exist".
            mech_stage = f"""
// --- Mechanics disabled: convergence state is trivially satisfied ---
int newtonDone = 1;      // no structural solve requested -> nothing to converge
int mechOrderCur = {mech_order};
int lastNewtonIts = 0;
real lastNewtonResid = 0.;
real lastNewtonDres = 0.;
int FinalSolidConverged = 1;
real warpageClampedFinal = 0.;
real dispMaxClampedFinal = 0.;
real sigMMaxClampedFinal = 0.;
real residFrozen = 0., dresFrozen = 0.;
int itsFrozen = 0;
"""

        # In-weld-loop mechanics increment, gated by the solve cadence. The
        # legacy "interpass" cadence emits nothing here, so the default run
        # behaves exactly as before; "layer_n"/"weld" add Abaqus-like
        # increments during deposition so plastic strain from the current pass
        # feeds the next increment (previously only inter-pass snapshots
        # contributed, so residual stress from later passes was never seen
        # mechanically).
        cadence = str(p.get("mech_solve_cadence", "interpass")).lower()
        inweld_mech = ""
        if p["enable_mechanical"] and cadence in ("layer_n", "weld"):
            if cadence == "weld":
                mech_mod = 1
            else:
                mech_mod = max(1, int(p.get("mech_solve_every_n_weld", 5)))
            inweld_mech = f"""
        // --- Incremental mechanics during deposition (cadence={cadence}, every {mech_mod} weld step(s)) ---
        if(ws % {mech_mod} == 0){{
            {self._mechanical_stage_increment()}
            dispM = sqrt(ux^2+uy^2+uz^2);
            cout << "  [MECH L" << iL+1 << " ws=" << ws << "] uz_max=" << uz[].max*1000.
                 << " mm  sigM_max=" << sigM[].max/1.e6 << " MPa  iters=" << lastNewtonIts
                 << (newtonDone ? "" : " [NOT CONVERGED]") << endl;
        }}
"""

        # Build microstructure stage (hardness/phase CSVs + z profile products)
        micro_stage = self._microstructure_stage() if p.get("enable_microstructure", True) else ""

        # --- Inter-pass cooling control ---
        # "time"   : legacy fixed dwell.
        # "target" : iterative cooldown to a peak-temperature target (Ansys
        #            "Iterative Cooldown"), which makes inter-pass temperature a
        #            controlled process variable instead of an emergent one.
        _ipmode = str(p.get("interpass_cooling_mode", "time")).lower()
        ipass_target_mode = 1 if _ipmode in ("target", "iterative", "temperature") else 0
        ipass_target = float(p.get("interpass_target_temp_C", 400.0))
        ipass_dt = max(0.1, float(p.get("interpass_time_step", 10.0)))
        ipass_tol_pct = float(p.get("interpass_converge_pct", 0.1))
        # Auto cap: derive from the time budget so a thin hot build gets enough
        # iterations to reach the target instead of silently giving up.
        _ipcap = int(p.get("interpass_max_iters", 0) or 0)
        if _ipcap <= 0:
            _ipcap = max(5, int(round(float(p.get("interpass_max_cool_s", 900.0)) / ipass_dt)))
        ipass_max_iters = max(1, _ipcap)

        # Springback (unclamping) stage. "dynamic_clamping" declared this
        # behaviour but nothing ever set clampActive=0, so the advertised
        # springback measurement did not exist. Emit a real unclamp + re-solve
        # when both flags are on.
        springback_stage = ""
        if (p["enable_mechanical"]
                and p.get("dynamic_clamping", True)
                and p.get("springback_solve", False)):
            springback_stage = self._mechanical_stage_springback()

        # Transverse size of the selected heat source. Used only by the adaptive
        # time-step controller to keep the moving arc from advancing more than a
        # fraction of the source width per step (which would smear the deposited
        # energy and show up as a collapse in the energy-audit fidelity).
        _hs = str(p.get("heat_source", "goldak")).lower()
        _i_weld = float(p.get("I_weld", 180.0))
        _p48_sig = 6.205e-4 * (max(_i_weld, 10.0) ** 0.28)
        _src_map = {
            "goldak": float(p.get("bG") or 7.0e-3),
            "p48_surface": _p48_sig,
            "p48_gmaw": _p48_sig,
            "p48": _p48_sig,
            "gaussian_vol": float(p.get("R_beam", 5.0e-3)),
            "gaussian_surf": float(p.get("R_spot", 3.0e-3)),
            "conical": float(p.get("rt", 3.0e-3)),
            "uniform": float(p.get("b_band", 5.0e-3)),
            "prescribed": float(p.get("R_activate", 10.0e-3)),
        }
        src_width = _src_map.get(_hs, 7.0e-3)


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

        # Wire Feedstock & Mass Deposition Calculations
        v_wire_m_min = float(p.get("v_wire", p.get("wire_feed_speed_m_min", p.get("wire_feed_speed", 5.0))))
        v_wire_m_s = v_wire_m_min / 60.0
        d_wire_raw = float(p.get("d_wire", p.get("wire_diameter_mm", p.get("wire_diameter", 1.2e-3))))
        d_wire_m = d_wire_raw / 1000.0 if d_wire_raw > 0.05 else d_wire_raw
        d_wire_mm = d_wire_m * 1000.0
        a_wire_m2 = (math.pi * (d_wire_m ** 2)) / 4.0
        a_wire_mm2 = a_wire_m2 * 1e6
        q_wire_vol_m3_s = a_wire_m2 * v_wire_m_s
        q_wire_cm3_s = q_wire_vol_m3_s * 1e6
        q_wire_mm3_min = a_wire_mm2 * (v_wire_m_min * 1000.0)
        rho_dep_val = float(rho_dep) if rho_dep else 7850.0
        m_dep_rate_kg_s = rho_dep_val * q_wire_vol_m3_s
        m_dep_kg_h = m_dep_rate_kg_s * 3600.0
        m_dep_g_s = m_dep_rate_kg_s * 1000.0
        a_bead_theo_m2 = (q_wire_vol_m3_s / v_scan) if v_scan > 0 else 0.0
        a_bead_theo_mm2 = a_bead_theo_m2 * 1e6

        # Physical Boiling / Vaporization Cutoff (Thermodynamic temperature governor)
        def get_tboil(mat_name):
            # ORDER IS LOAD-BEARING. The old ordering tested the aluminium
            # substring first, so "Ti-6Al-4V".lower() == "ti-6al-4v" matched
            # "al" and returned the aluminium boiling point (2519 C) instead of
            # titanium's (3287 C). Titanium must be disambiguated before any
            # "al" substring test.
            mat_low = str(mat_name).lower()
            if "ti-" in mat_low or "ti6" in mat_low or "titanium" in mat_low or "6al" in mat_low:
                return 3287.0  # Ti-6Al-4V
            if "al" in mat_low or "1xx" in mat_low:
                return 2519.0  # Pure Al / AlSi10Mg (Al bp 2519 C)
            if "625" in mat_low or "in6" in mat_low or mat_low.startswith("ni"):
                return 2913.0  # IN625 / Ni-base (Ni bp 2913 C)
            return 2862.0  # Carbon / austenitic steels (Fe bp 2862 C)

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
            hs_details = f"//    - Model Type:                   Goldak Double Ellipsoid\n//    - Front Semi-Axis (af):         {p.get('af', 0.008)*1000.0:.2f} mm  ({p.get('af', 0.008)} m)\n//    - Rear Semi-Axis (ar):          {p.get('ar', 0.020)*1000.0:.2f} mm  ({p.get('ar', 0.020)} m)\n//    - Semi-Width (bG):              {p.get('bG', 0.007)*1000.0:.2f} mm  ({p.get('bG', 0.007)} m)\n//    - Depth Penetration (cG):       {p.get('cG', 0.003)*1000.0:.2f} mm  ({p.get('cG', 0.003)} m)\n//    - Energy Fractions:             ff={p.get('ff', 0.6):.2f}, fr={p.get('fr', 1.4):.2f}"
        elif hs_type in ('p48_surface', 'p48_gmaw', 'p48'):
            i_w = float(p.get("I_weld", 150.0))
            eta_d = float(p.get("eta_drop", 0.25))
            sig_q = 6.205e-4 * (max(i_w, 10.0) ** 0.28)
            hs_details = (
                f"//    - Model Type:                   P48 Coupled Surface Flux & Droplet Impingement\n"
                f"//    - Arc Gaussian Radius (sigma_q): {sig_q*1000.0:.2f} mm\n"
                f"//    - Droplet Transfer Fraction:    {eta_d*100:.1f}%\n"
                f"//    - Direct Arc Surface Flux:      {(1.0-eta_d)*100:.1f}%\n"
                f"//    - Droplet Transfer Temp:        {float(p.get('T_drop', 1627.0)):.1f} degC"
            )
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
/* === COMPLETE SIMULATION PARAMETER SPECIFICATION === (block comment: FreeFEM counts this whole span as one lexer unit)
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
//    - H_build Target (GUI driver):   {p.get('H_build_target_mm', hz_dep*1000.0):.2f} mm (ceil-rounded to Nlayers above; solver and display agree)
//    - Total Model Height (Z):        {hz_tot*1000.0:.2f} mm  ({hz_tot:.4f} m)
//    - Wall Centerline Y:             {y_centre*1000.0:.2f} mm
//    - Wall Y Span:                   [{y_left*1000.0:.2f}, {y_right*1000.0:.2f}] mm
//
// 2. PROCESS & ARC HEAT INPUT:
//    - Welding Arc Current:           {I_weld:.1f} A
//    - Welding Arc Voltage:           {U_weld:.1f} V
//    - Arc Thermal Efficiency (eta):  {eta_arc:.2f} ({eta_arc*100.0:.0f}%)
//    - Wire Feed Speed (WFS):         {v_wire_m_min:.2f} m/min  ({v_wire_m_s:.4f} m/s)
//    - Torch Travel Speed (TS):       {v_scan*1000.0*60.0:.1f} mm/min  ({v_scan*1000.0:.2f} mm/s, {v_scan:.5f} m/s)
//    - Wire Diameter:                 {d_wire_mm:.2f} mm  ({d_wire_m:.6f} m)
//    - Wire Cross-Sectional Area:     {a_wire_mm2:.3f} mm^2
//    - Feedstock Mass Deposition:     {m_dep_kg_h:.2f} kg/h  ({m_dep_g_s:.2f} g/s)
//    - Theoretical Bead Area:         {a_bead_theo_mm2:.2f} mm^2  (CAD Nominal: {bead_area:.2f} mm^2)
//    - Gross Electrical Power:        {p_gross:.1f} W  ({p_gross/1000.0:.2f} kW)
//    - Net Thermal Arc Power:         {p_net:.1f} W  ({p_net/1000.0:.2f} kW)
//    - Net Linear Heat Input (H_net): {h_net_j_mm:.2f} J/mm  ({h_net_j_mm/1000.0:.3f} kJ/mm, {h_net_j_mm:.0f} kJ/m)
//    - Gross Linear Heat Input:       {h_gross_j_mm:.2f} J/mm  ({h_gross_j_mm/1000.0:.3f} kJ/mm)
//    - Volumetric Energy Density:     {e_vol:.2f} J/mm^3  ({e_vol*1000.0:.1f} MJ/m^3)
//    - Single Pass Deposition Time:   {t_pass:.2f} s
//    - Inter-Pass Cooling Dwell:      {p['t_wait']:.2f} s

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

// 4b. ULTRASONIC VIBRATION ASSISTANCE (UVA-WAAM):
//    - UVA Enabled:                   {'YES' if p.get('enable_ultrasonic', False) else 'NO'}
//    - Frequency:                     {p.get('ultrasonic_frequency', 0.0):.0f} Hz
//    - Amplitude:                     {p.get('ultrasonic_amplitude', 15.0e-6)*1e6:.1f} um
//    - Acoustic Streaming:            {'ON (kStream=' + format(p.get('ultrasonic_k_stream', 45.0), '.1f') + ' W/m/K)' if p.get('ultrasonic_streaming', False) else 'OFF'}
//    - Acoustic Heat Generation:      {'ON (eta=' + format(p.get('ultrasonic_coupling_eff', 0.15), '.2f') + ')' if p.get('ultrasonic_heat_gen', False) else 'OFF'}
//    - Acoustic Cavitation:           {'ON (threshold=' + format(p.get('ultrasonic_cav_threshold', 1.0e6)/1e6, '.1f') + ' MPa)' if p.get('ultrasonic_cavitation', False) else 'OFF'}
//    - Acoustic Momentum Transfer:    {'ON' if p.get('ultrasonic_momentum', False) else 'OFF'}
//    - Blaha Softening Factor:        {1.0 - min(0.3, p.get('ultrasonic_frequency', 0.0)/100000.0):.2f} (yield stress multiplier)
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
//    - Mechanical Model:              {'J2 THERMO-ELASTO-PLASTICITY (radial return)' if p.get('enable_j2_plasticity', True) else 'ELASTIC (no plasticity)'}
//    - Mechanics FESpace:             {'P2 (BETA with J2 -- see validation warning)' if int(p.get('mechanics_order', 1)) == 2 else 'P1 (validated)'}
//    - Equilibrium Solver:            {'INCREMENTAL NEWTON-RAPHSON' if p.get('mech_max_newton_iter', 12) > 1 else 'SINGLE DIRECT SOLVE (staggered)'}
//    - Tangent Stiffness:             {'CONSISTENT (elastic predictor + plastic tangent)' if p.get('mech_use_consistent_tangent', True) else 'ELASTIC (modified Newton)'}
//    - Solve Cadence:                 {p.get('mech_solve_cadence', 'interpass').upper()} ({p.get('mech_solve_every_n_weld', 5)} weld steps)
//    - Springback (unclamp) Solve:    {'ENABLED (dynamic_clamping + springback_solve)' if p.get('springback_solve', False) and p.get('dynamic_clamping', True) else 'DISABLED'}
//    - Support Boundary Condition:    {p.get('mechanical_bc', 'isostatic').upper()}
//    - Boiling Temp Clamping:         {'ENABLED' if p.get('enable_boiling_cap', False) else 'DISABLED (Raw Unclamped Physics)'}
//    - Cauchy Equilibrium Stresses:  DIRECT CAUCHY EQUILIBRIUM SOLVE (div sigma = 0 enforced weakly)
//
// 7. ARTIFACT DESTINATIONS:
//    - Visual VTU / PVD Results:      {visdir_norm}
//    - Tabular Data & Metrics:        {outdir_norm}
//    - Native Console Log:            {outdir_norm}sim.log
// ================================================================================
   === END PARAMETER SPECIFICATION (block closes here) === */

{self.solver_load}
load "iovtk"

string outdir = "{outdir_norm}";
{mpi_rank_dirs}string visdir = outdir + "visual_outputs/";
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
// Inter-pass cooling control: 0 = fixed dwell (t_wait), 1 = iterative cooldown
// to interpass_target_temp_C.
int ipassTargetMode = {ipass_target_mode};
// Net Linear Heat Input:   {h_net_j_mm:.2f} J/mm ({h_net_j_mm/1000.0:.3f} kJ/mm)
// Gross Linear Heat Input: {h_gross_j_mm:.2f} J/mm ({h_gross_j_mm/1000.0:.3f} kJ/mm)
// Net Thermal Arc Power:   {p_net:.1f} W ({p_net/1000.0:.2f} kW)
// Volumetric Energy Density: {e_vol:.2f} J/mm^3

// === WIRE FEEDSTOCK & MASS CONSERVATION ===
real dWire       = {d_wire_m:.6e}; // Wire diameter [m] ({d_wire_mm:.2f} mm)
real vWire       = {v_wire_m_s:.6e}; // Wire feed speed [m/s] ({v_wire_m_min:.2f} m/min)
real aWire       = {a_wire_m2:.6e};  // Wire cross-sectional area [m^2] ({a_wire_mm2:.3f} mm^2)
real QwireVol    = {q_wire_vol_m3_s:.6e}; // Volumetric feed rate [m^3/s]
real mDepRate    = {m_dep_rate_kg_s:.6e}; // Mass deposition rate [kg/s] ({m_dep_kg_h:.2f} kg/h)
real aBeadTheo   = {a_bead_theo_m2:.6e};  // Theoretical bead area [m^2] ({a_bead_theo_mm2:.2f} mm^2)

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
// NOTE: FreeFEM's ifstream constructor RAISES on a missing file (it aborts the
// whole run before .good() can be consulted), so existence must be tested
// out-of-band with `system("test -f ...")` first. Without this guard a GUI run
// that skipped the slicer died with "Exec error: Error opening file".
int nPts = 0;
real[int] tpT(1), tpX(1), tpY(1), tpZ(1), tpP(1);
{{
    string tpFile = outdir + "toolpath.txt";
    int tpRc = system("test -f '" + tpFile + "'");
    if(tpRc == 0){{
        ifstream fgcode(tpFile);
        if(fgcode.good()){{
            fgcode >> nPts;
            if(nPts > 0) {{
                tpT.resize(nPts); tpX.resize(nPts); tpY.resize(nPts); tpZ.resize(nPts); tpP.resize(nPts);
                for(int i=0; i<nPts; ++i){{
                    fgcode >> tpT[i] >> tpX[i] >> tpY[i] >> tpZ[i] >> tpP[i];
                }}
                cout << " [G-CODE] Successfully loaded toolpath array with " << nPts << " coordinates." << endl;
            }} else {{
                cout << " [G-CODE] toolpath.txt present but empty (nPts=0). Falling back to linear scan assumption." << endl;
            }}
        }}
    }} else {{
        cout << " [G-CODE] No toolpath.txt found. Falling back to linear scan assumption." << endl;
    }}
}}

// === PER-LAYER PASS TIME READER ===
// Reads per-layer pass times from the slicer output. These times account for
// varying layer perimeters (e.g., equatorial vs pole passes of a sphere).
// Same existence-guard rationale as the toolpath reader above.
int nLayerTimes = 0;
real[int] layerPassTimes(1);
{{
    string ltFile = outdir + "layer_pass_times.txt";
    int ltRc = system("test -f '" + ltFile + "'");
    if(ltRc == 0){{
        ifstream flt(ltFile);
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
    }} else {{
        cout << " [G-CODE] No layer_pass_times.txt found. Using constant pass time." << endl;
    }}
}}


cout << "==========================================" << endl;
cout << " T-JOINT AS WAAM PROCESS (LAYER REMELTING FUSION)" << endl;
cout << " Substrate: " << LxW*1000. << "x" << LyBase*1000. << "x" << HzBase*1000. << "mm" << endl;
cout << " Deposit: " << Nlayers << " layers | Layer 1: " << hBead1*1000. << "mm, Subseq: " << hBeadSub*1000. << "mm (" << remeltRatio*100. << "% fusion)" << endl;
cout << " Total Build Height: " << HzDep*1000. << "mm" << endl;
cout << " Width: " << TyDep*1000. << "mm  centred at y=" << yDepCentre*1000. << "mm" << endl;
cout << " Wire Feedstock: d=" << dWire*1000. << "mm | WFS=" << vWire*60. << "m/min | Deposition=" << mDepRate*3600. << "kg/h" << endl;
cout << " P=" << Pweld << "W  v=" << vscan*1000. << "mm/s" << endl;
cout << " tPass=" << tPass << "s  tWait=" << tWait << "s  Total=" << Nlayers*(tPass+tWait) << "s" << endl;
cout << "==========================================" << endl;

// === MATERIAL FUNCTIONS ===
{mat_funcs}
// True enthalpy (numerical integral of CpMat, including the fusion-latent
// spike). Defined here so both the prescribed-T audit and the final stored
// computation use the identical function.
// Midpoint integral of CpMat over [a,b]. CpMat is continuous piecewise-linear
// except for the fusion-latent box, so midpoint is exact-ish inside a smooth
// span and must never straddle a discontinuity.
func real Hseg(real a, real b, real zv){{
    if(b <= a) return 0.;
    int nq = int(max(8., min(120., (b - a)/10.)));
    real dT = (b - a)/real(nq);
    real H = 0.;
    for(int i=0; i<nq; i++) H += CpMat(a + (real(i)+0.5)*dT, zv)*dT;
    return H;
}}
func real Hcp(real T, real zv){{
    // Split at the solidus/liquidus so no quadrature bin ever straddles the
    // latent-spike box. A single midpoint span across [Tamb,T] aliased the
    // 4457 J/kg/K box by bin alignment and mis-stated imposed/stored energy
    // by ~10% whenever material crossed the fusion band (measured +11%
    // residual on a prescribed-T case that nothing else explained).
    if(T <= Tamb) return 0.;
    real Ts = TsolMat(zv);
    real Tl = TliqMat(zv);
    real H = Hseg(Tamb, min(T, Ts), zv);
    if(T > Ts) H += Hseg(Ts, min(T, Tl), zv);
    if(T > Tl) H += Hseg(Tl, T, zv);
    return H;
}}


// === MESH ===
{"// Using external mesh file" if p['mesh_source'] != 'parametric' else ""}
{self._mesh_block()}

// === ADAPTIVE TIME-STEP CONTROLLER ===
// The explicit stability/accuracy limit for a diffusion problem is the diffusion
// (Fourier) number Fo = alpha*dt/dx^2, where alpha = k/(rho*Cp). Limiting Fo is
// how an Abaqus-like *HEAT TRANSFER step with automatic time incrementation
// behaves: dt is derived from the material diffusivity and the mesh, so accuracy
// no longer depends on the user guessing dt_weld correctly.
int useAdaptiveDt = {1 if str(p.get('adaptive_dt','off')).lower() not in ('off','0','false','none') else 0};
real foTarget = {float(p.get('adaptive_fo_target', 0.45))};
real dtMinA = {float(p.get('adaptive_dt_min', 1.0e-4))};
real dtMaxA = {float(p.get('adaptive_dt_max', 5.0))};

// Smallest characteristic element length, measured over all six edges of every
// tetrahedron. This is the real h that limits diffusion accuracy, and measuring
// it (rather than assuming a value from the parametric Nx/Ny/Nz) keeps the
// limiter correct for imported CAD meshes too. For a structured box mesh the
// minimum edge equals the smallest cell dimension, which is the expected answer.
real hMinMesh = 1.e9;
for(int ie=0; ie<Th.nt; ie++){{
    // Element vertices via the mesh connectivity table Th[ie][a]. (An earlier
    // revision used ie+a*Th.nv, which is vector-fespace dof arithmetic, not
    // mesh connectivity: it measured edges between unrelated vertices, found a
    // phantom h_min of 3e-9 m, and pinned the adaptive dt at its floor, so the
    // runaway guard tripped after 800 steps having simulated 0.08 s while
    // reporting success. Element birth for the source mask had nothing to do
    // with it; the ruler was broken, not the mesh.)
    real hloc = 1.e9;
    for(int a=0; a<4; a++)
        for(int b=a+1; b<4; b++){{
            real dv = Th(Th[ie][a]).z - Th(Th[ie][b]).z;
            real dh = Th(Th[ie][a]).y - Th(Th[ie][b]).y;
            real dd = Th(Th[ie][a]).x - Th(Th[ie][b]).x;
            real elen = sqrt(dv*dv + dh*dh + dd*dd);
            if(elen > 1.e-12 && elen < hloc) hloc = elen;
        }}
    if(hloc < hMinMesh) hMinMesh = hloc;
}}
real hGlobMesh = max(hMinMesh, 1.e-9);
real RpoolAct = {r_activate};
int hSrcIsPrescribed = {is_prescribed};
// Resolution gate for prescribed-T pools: the cosine-squared profile is only
// sampled correctly when several nodes fall inside R_activate. On a coarse
// mesh the torch path threads between nodes (hit-or-miss imposition) and the
// hard-ball alternative Gibbs-rings instead. Both failure modes are silent
// energy errors, so under-resolution aborts loudly rather than reporting.
if(hSrcIsPrescribed && RpoolAct < 2.*hGlobMesh){{
    cout << "  [FATAL] R_activate (" << RpoolAct*1000. << " mm) < 2x min cell ("
         << hGlobMesh*1000. << " mm). The pool profile cannot be sampled: "
         << "refine the mesh or widen R_activate. Aborting rather than "
         << "depositing hit-or-miss energy." << endl;
    exit(1);
}}
if(useAdaptiveDt == 1)
    cout << " [ADAPTIVE DT] enabled | h_min=" << hGlobMesh*1000. << " mm | Fo_target=" << foTarget
         << " | dt in [" << dtMinA << ", " << dtMaxA << "] s" << endl;
else
    cout << " [ADAPTIVE DT] off (fixed dt schedules)" << endl;

// Stable/accurate dt for the current thermal state: dt = Fo*dx^2/alpha, using
// the WORST (smallest diffusivity, i.e. lowest k and highest heat capacity) that
// is actually present, so the limit is conservative everywhere on the mesh.
// NOTE: defined further below, after the thermal fields are declared, because it
// reads psiActive / kEff / CpEff / Temp0.

// === FE SPACES ===
fespace Sh(Th, P1);
fespace Vh(Th, [{mech_fe},{mech_fe},{mech_fe}]);
// Mechanical state space: P1 normally, P2 with mechanics_order=2 (C3D10-like:
// warpage-grade bending without P1-tet locking). Thermal fields stay P1 under
// both orders -- heat is smooth and the P1 audit closes to <1% -- so every
// Th.nv-indexed thermal loop is untouched by the order switch. Only the
// mechanical loops below use dof-indexed (order-agnostic) access.
fespace Mh(Th, {mech_fe});
fespace Ph(Th, P0);

Sh Temp, Temp0, dTemp;
Sh TempCorrPrev; // corrector scratch (previous pass); never a physics state
Sh kEff, CpEff, rhoEff, Tsource;
Sh QacouField;    // UVA acoustic dissipation field
Sh sigUVA;        // UVA stress-relief field
Sh reliefFrac;    // UVA relief fraction field
// Lumped nodal volumes (row sums of the P1 mass matrix). The mesh never changes,
// so these are computed once and reused wherever a nodal quantity must be
// summed as an energy. Forgetting the volume weights energy density as energy:
// the prescribed-T audit once reported 5.9e8 kJ imposed for a 60 kJ process.
varf vMassLump(u, v) = int3d(Th)(u*v);
Sh psiActive, Tpeak, hSurf, TempVis;
// Element-birth mask for the heat source (see below). P0, i.e. one value per
// element: an element contributes source energy only when ALL of its nodes are
// active. Nodal `* psiActive` in the source formulae is not enough, because the
// load vector integrates over whole elements and P1 assembly smears an active
// node's source onto the quiet vertices of every tet it touches; the quiet
// clamp then destroys that energy (measured: 82% of injected on the CAS case).
Ph srcMask, bornMask;
matrix MmassLump = vMassLump(Sh, Sh);
real[int] onesLump(Th.nv); onesLump = 1.;
real[int] lumpVol = MmassLump*onesLump;
Sh dTdt;        // Instantaneous cooling rate field [C/s]
Sh TdotCool;    // Maximum transient cooling rate at solidification front [C/s]
Sh dt85Track;   // Elapsed continuous cooling time between 800C and 500C [s]
Sh tEnter800;   // Entry timestamp for 800C boundary [s]
Sh nReheat;     // Multi-pass thermal reheat cycle counter
Sh SDAS, dt85;
Sh fFerrite, fBainite, fMartensite, fPearlite, HVmap;

Vh [ux,uy,uz];
Vh [vx,vy,vz];
Mh dispM, sigM, sigXX;

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
// Element-birth mask, initial state. P0 assignment evaluates the P1 field at
// the element centroid, i.e. the mean of its four nodal values; for a binary
// field `> 0.999` is true iff ALL four nodes are active. Front-straddling
// elements therefore contribute no source until their last node activates.
{srcmask_init}
real bornFloor = 1.e-5;
bornMask = max(srcMask, bornFloor);

// Boiling-cap energy accounting [J]: excess enthalpy deleted by the Tboil clip
real EcapJ = 0.;

// UVA global state
real PacouSeenMax = 0.;
real betaUS = 0.22;
real maxBaselineStress = 0.;
real maxBaselineDisp = 0.;
// Energy-balance audit accumulators [J] (printed in SIMULATION SUMMARY)
real EarcJ = 0.;   // arc energy ACTUALLY injected: sum(int Tsource dV * dt)
real EintJ = 0.;   // intended arc energy: sum(Pweld * pState * dt)
real ElossJ = 0.;  // surface losses: clamp (label 1) + convection/radiation (labels 2,3)
real EquietJ = 0.;
                   // reaches quiet nodes, the Tamb clamp is removed, and conducted
                   // preheat stays in the stored enthalpy. Kept (as zero) so the
                   // audit schema and checkpoints stay stable.
real EprescJ = 0.; // enthalpy imposed by prescribed-T activation (nodal, exact).
real ErectJ = 0.; // energy added by the Tamb floor (Gibbs-undershoot rectifier).
                   // Counted as a source in the audit so the balance closes honestly.
// Per-step rectifier trace. Off by default; enable to see which steps undershoot.
bool rectDebug = {str(bool(p.get("thermal_rect_debug", False))).lower()};


// === VARIATIONAL FORMS ===
real thetaT = {p['theta']};
real kLiqF = {float(p.get('k_liq_factor', 1.0))}; // liquid-state k enhancement (1.0 = off; 2-5 mimics convective pool transport)
real dt     = dtWeld;
int nThermalCorr = {n_thermal_corr};
real corrTol = {corr_tol};

varf aThermal(Temp, dTemp) =
    int3d(Th)(
        (rhoEff*CpEff*bornMask)/dt * Temp*dTemp
      + thetaT*(kEff*bornMask)*(dx(Temp)*dx(dTemp)+dy(Temp)*dy(dTemp)+dz(Temp)*dz(dTemp))
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
        (rhoEff*CpEff*bornMask)/dt * Temp0*dTemp
      - (1.-thetaT)*(kEff*bornMask)*(
            dx(Temp0)*dx(dTemp)+dy(Temp0)*dy(dTemp)+dz(Temp0)*dz(dTemp))
      + Tsource*srcMask*dTemp
      + QacouField*dTemp
    )
    + int2d(Th,1)( hBot  * psiActive * Tamb*dTemp )
    + int2d(Th,2)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,3)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,4)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,5)( hSurf * psiActive * Tamb*dTemp )
    + int2d(Th,6)( hSurf * psiActive * Tamb*dTemp )
    ;
// Source-only load, assembled with the SAME quadrature as lThermal. Summing it
// measures EXACTLY what the matrix deposits: sum over dofs of the load vector
// equals the volume integral under the assembly rule (partition of unity), so
// no quadrature mismatch can open between "injected" and what the solution
// actually received. Auditing int3d(Tsource*srcMask) instead disagrees with
// the assembly whenever the P0 mask jumps inside an element, which reads as
// phantom stored energy. (Measured: 15 kJ gap on the 1-layer benchmark.)
varf vSrcAudit(u, v) = int3d(Th)(Tsource*srcMask*v);

// === ADAPTIVE dt HELPER (needs the thermal fields declared above) ===
// dt = Fo*dx^2/alpha, evaluated with the WORST (smallest) diffusivity actually
// present so the limit is conservative across the whole mesh.
func real stableThermalDt(real dtCap){{
    if(useAdaptiveDt == 0) return dtCap;
    real alphaWorst = 1.e9;
    for(int iv=0; iv<Th.nv; iv++){{
        if(psiActive[][iv] < 0.5) continue;
        real zv = Th(iv).z;
        real rhoLoc = (zv <= HzBase + 1.e-6) ? rhoBase : rhoDep;
        real alphaLoc = kEff[][iv] / max(rhoLoc * CpEff[][iv], 1.e-6);
        if(alphaLoc < alphaWorst) alphaWorst = alphaLoc;
    }}
    if(alphaWorst > 1.e8) alphaWorst = 1.e-8;   // guard: no active material
    real dtStable = foTarget * hGlobMesh*hGlobMesh / alphaWorst;
    return min(max(dtStable, dtMinA), min(dtMaxA, dtCap));
}}

{self._probe_setup()}

// === OUTPUT SETUP ===
int maxPvdEntries = 30000;
real[int] pvdTimes(maxPvdEntries);
string[int] pvdFiles(maxPvdEntries);
int nPvdEntries = 0;

func int recordPvd(real curTime, string curFile) {{
    if (nPvdEntries < maxPvdEntries) {{
        pvdTimes[nPvdEntries] = curTime;
        pvdFiles[nPvdEntries] = curFile;
        nPvdEntries++;
    }}
    ofstream pv(visdir + "waam.pvd");
    pv << "<?xml version='1.0'?>" << endl;
    pv << "<VTKFile type='Collection' version='0.1'>" << endl;
    pv << "  <Collection>" << endl;
    for(int ip = 0; ip < nPvdEntries; ip++) {{
        pv << "    <DataSet timestep='" << pvdTimes[ip] << "' file='" << pvdFiles[ip] << "'/>" << endl;
    }}
    pv << "  </Collection>" << endl;
    pv << "</VTKFile>" << endl;
    return 0;
}}
{{
    ofstream f(outdir+"waam_data.csv");
    f<<"time_s,layer,phase,xTorch_mm,zLayer_mm,Tmax_C,Tpeak_max_C,HV_layer"<<endl;
}}

int frame = 0;

// === RESTART / CHECKPOINT ===
{self._restart_block()}

// === DEPOSITION LOOP ===
cout << " === WAAM DEPOSITION ===" << endl;
real t = 0.;
int histCounter = 0;   // drives probe_interval; monotonic across all phases
// Apply a resumed checkpoint to the loop counters (fields are already restored).
if(resumeOk == 1){{
    t = ckptTime;
    frame = ckptFrame;
    histCounter = ckptHist;
    EarcJ = ckptEarc; EintJ = ckptEint; ElossJ = ckptEloss;
    EquietJ = ckptEquiet; EcapJ = ckptEcap; EprescJ = ckptEpresc;
}}

for(int iL=0; iL<Nlayers; iL++){{
    // Skip layers already completed in a previous (interrupted) run.
    if(resumeOk == 1 && iL < resumeFromLayer) continue;
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

    // With adaptive_dt on, dt is re-derived from the stability limit every step
    // and the loop advances by accumulating tLocal; with it off, dt stays
    // tPassLayer/NweldSteps and tLocal = ws*dt exactly as before, so the default
    // run is bit-identical to the fixed-step implementation.
    real tLocal = 0.;
    int ws = 0;
    int maxWeldAcc = 20*max(NweldSteps, 1) + 200;   // runaway guard
    while(tLocal < tPassLayer - 1.e-9 && ws < maxWeldAcc){{
        ws++;
        if(useAdaptiveDt == 1){{
            real dtLeft = tPassLayer - tLocal;
            // Two independent CAPS on dt, take the smaller:
            //  (1) diffusion accuracy: dt <= Fo*h^2/alpha  (stableThermalDt)
            //  (2) moving-source resolution: the torch must not advance more than
            //      ~1/8 of the source width per step, otherwise the arc energy is
            //      smeared over far less travel than intended and the energy-audit
            //      fidelity collapses (measured: 49.9 kJ -> 20.6 kJ injected).
            // srcWidth is the transverse size of the selected heat source.
            real dtDiff = stableThermalDt(dtLeft);
            real wSrc = min(max({src_width}, 1.e-4), LxW/4.);
            real dtSrc = min(wSrc / max({src_travel_frac_fmt}*vscan, 1.e-9), dtLeft);
            dt = min(dtDiff, dtSrc);
            if(dt < 1.e-9) dt = dtLeft;
            tLocal += dt;
        }} else {{
            tLocal = real(ws)*dt;
        }}
        if(dt < 1.e-12) dt = max(tPassLayer - tLocal, 1.e-12);
        bool isBidi = {str(p.get("deposition_pattern", "bidirectional").lower() == "bidirectional").lower()};
        real xTorch = (isBidi && (iL % 2 == 1)) ? max(0., min(LxW, LxW - vscan*tLocal)) : max(0., min(LxW, vscan*tLocal));
        real yTorch = yDepCentre;
        real zTorch = zLayMid;
        real pState = 1.0;
        
        // G-Code Override
        if(nPts > 0){{
            // Time to look up in the toolpath. It must NOT be clamped to the
            // last point: `t` keeps advancing through the inter-pass dwell and
            // the cooldown, so clamping pinned the lookup to the final
            // toolpath vertex, which carries power=1, and re-energised the arc
            // for the whole post-weld period. Measured on the CAS-DED case:
            // ~197 s of phantom arc at 148.6 W injected 29.3 kJ of heat into a
            // part that was supposed to be cooling, and the energy audit
            // reported 73.3 kJ injected against 40.3 kJ intended.
            //
            // Past the end of the program the torch simply parks and the arc is
            // off. Anything outside [tpT[0], tpT[nPts-1]] is arc-off.
            // Inside the weld loop `t` is the start of this layer, so t + tLocal
            // is the true process time and no clamping is needed. Anything
            // outside [tpT[0], tpT[nPts-1]] is arc-off.
            real tGcode = t + tLocal;
            bool afterProgram = (tGcode > tpT[nPts-1] + 1.e-9);
            bool beforeProgram = (tGcode < tpT[0] - 1.e-9);
            tGcode = min(max(tGcode, tpT[0]), tpT[nPts-1]);
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
            pState = (afterProgram || beforeProgram) ? 0.0
                   : ((tpP[idx1] > 0 || tpP[idx2] > 0) ? 1.0 : 0.0);
        }}

        // Progressive activation
        for(int iv=0; iv<Th.nv; iv++){{
            real xv=Th(iv).x; real zv=Th(iv).z; real yv=Th(iv).y;
            if({act_cond})
                psiActive[][iv] = 1.;
        }}
        // Recompute the element-birth mask after every activation update. The
        // source and its audit below must see the same front position.
        {srcmask_step}
        bornMask = max(srcMask, bornFloor);

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
        // UVA acoustic thermal physics (heat generation, streaming, cavitation)
        {uva_thermal_block}
        // Actual energy entering the PDE this step. The P0 birth mask travels
        // with the source: front-straddling elements contribute nothing, so
        // P1 assembly cannot smear active-node energy onto quiet vertices.
        // (EintJ is the intended Pweld*dt; their ratio now honestly reports
        // birth-clipping of the source leading edge instead of quadrature
        // overshoot plus clamp destruction.)
        // Audit-by-assembly: sum the source load vector itself, so the
        // audited number uses the assembly quadrature by construction.
        // (This FreeFEM build has no .sum on real[int]; accumulate by hand.)
        real[int] fSrcAudit = vSrcAudit(0, Sh);
        real srcAuditSum = 0.;
        for(int iSrcA=0; iSrcA<fSrcAudit.n; iSrcA++) srcAuditSum += fSrcAudit[iSrcA];
        if (srcAuditSum > 1.e-6 && srcAuditSum < Pweld * 0.98) {{
            real qScale = Pweld / srcAuditSum;
            Tsource = Tsource * qScale;
            srcAuditSum = Pweld;
        }}
        EarcJ += dt * srcAuditSum;

        // Solve, with iterate-to-convergence corrector passes. kEff/CpEff/hrad
        // are evaluated from the START-of-step temperature, so when a step jumps
        // hundreds of K (arc birth, pool-hold imposition) the lagged properties
        // -- above all the lagged radiation coefficient on a T^4 law --
        // systematically under-count losses and overheat (measured +35% missing
        // energy on an aluminium pool-hold case with a single pass). Each
        // corrector re-evaluates properties at the latest Temp and re-solves,
        // stopping when the max nodal change falls under corrTol or after
        // nThermalCorr total passes. The source, audit and all downstream
        // bookkeeping run once, on the converged field.
        // thermal_correctors=1 preserves the legacy single-pass behaviour.
        matrix MT = aThermal(Sh,Sh);
        real[int] fT = lThermal(0,Sh);
        set(MT, solver={self.solver_kw});
        Temp[] = MT^-1*fT;
        for(int corrPass=1; corrPass<nThermalCorr; corrPass++){{
            for(int iv=0; iv<Th.nv; iv++){{
                real Tv = Temp[][iv];
                real zv = Th(iv).z;
                real psiEff = max(psiActive[][iv], 1.e-4);
                kEff[][iv]  = kMat(Tv, zv)*psiEff*(Tv > TliqMat(zv) ? kLiqF : 1.);
                CpEff[][iv] = CpMat(Tv, zv);{rad_update}
            }}
            MT = aThermal(Sh,Sh);
            fT = lThermal(0,Sh);
            set(MT, solver={self.solver_kw});
            TempCorrPrev[] = Temp[];
            Temp[] = MT^-1*fT;
            // Max nodal change vs the previous pass: convergence measure.
            // (Temp0 is the step's anchor and must not be touched here.)
            real corrMax = 0.;
            for(int iv=0; iv<Th.nv; iv++){{
                real dTc = abs(Temp[][iv] - TempCorrPrev[][iv]);
                if(dTc > corrMax) corrMax = dTc;
            }}
            if(corrMax < corrTol) break;
        }}
        // The floor always applies, so its energy is always counted; rectDebug
        // only controls the per-step print. Counting it conditionally while
        // applying it unconditionally once hid 8.8 kJ on a prescribed-T run.
        // Lumped-nodal sum, NOT int3d: the floor acts exactly where the field
        // oscillates (Gibbs undershoot = high-frequency), and consistent vs
        // lumped quadrature disagree most there. Every other nodal term
        // (EprescJ) is lumped; mixing them opens a phantom gap.
        // Lumped-nodal sum (same quadrature basis as EprescJ; see the
        // weld-loop site for the rationale).
        {{
            real eRectStep = 0.;
            for(int iv=0; iv<Th.nv; iv++){{
                real Tr = Temp[][iv];
                if(Tr < Tamb) eRectStep += lumpVol[iv]*rhoEff[][iv]*CpEff[][iv]*(Tamb - Tr);
            }}
            ErectJ += eRectStep;
        }}
        if(rectDebug && ErectJ > 0.)
            cout << "    [RECT] dt=" << dt << " ErectJ=" << ErectJ
                 << " Tmin=" << Temp[].min << endl;
        Temp = max(Temp, Tamb);
        // Quiet (not-yet-deposited) nodes held at ambient: P1 source
        // interpolation smears arc energy across shared interface elements,
        // which otherwise superheats quiet nodes to 3000+ C.
        // Enthalpy the smeared P1 source has just parked on not-yet-deposited
        // (quiet) nodes, BEFORE they are clamped back to Tamb. This is an
        // energy FLOW: it must be multiplied by dt like every other per-step
        // term. It used to be accumulated unweighted, so the same stored
        // enthalpy was re-counted on every timestep. Measured: EquietJ reached
        // 121.7 kJ while the arc only ever injected 79.4 kJ, which is
        // impossible for a quantity that is supposed to be a subset of the
        // injected energy, and it turned the audit residual into -142%.
        // Enthalpy sitting on not-yet-deposited (quiet) nodes, BEFORE they are
        // clamped back to Tamb, i.e. the energy the clamp is about to destroy.
        // This is a STOCK, not a flow -- exactly like the boiling-cap term
        // EcapJ, which likewise accumulates without dt -- so it carries no dt.
        // (An earlier revision multiplied it by dt, which is dimensionally
        // Joule-seconds and wrong; fluxes carry dt, destroyed stocks do not.)
        //
        // With the P0 birth mask on the source this term should stay near zero:
        // front-straddling elements receive no source energy, so P1 assembly
        // can no longer smear it onto quiet vertices. Before the mask existed
        // it reached 121.7 kJ against 79.4 kJ injected -- more destroyed than
        // ever put in, the signature of assembly smear plus conducted-in heat
        // being counted as source energy. What remains is genuinely conducted
        // heat plus numerical noise at the front, both legitimately destroyed
        // by the clamp and legitimately counted here.
        // No Tamb clamp on quiet nodes. The clamp existed to suppress the
        // 3000 C superheat that P1 assembly smeared onto quiet vertices; the P0
        // birth mask (srcMask) stops that smear at the element level, so the
        // clamp's only remaining effect was destroying legitimate conducted
        // preheat ahead of the torch. Conducted preheat is physical -- it is
        // the heat-affected zone -- and stays in the stored enthalpy, where
        // the audit counts it. (An earlier revision measured a "quiet stock"
        // with int3d(...*(1-psiActive)); that integral is a P1 ramp inside
        // front-straddling elements, so it counted a fraction of the HOT bead
        // as quiet and reported more destroyed than injected. The nodal clamp
        // destroys only nodal values, which a field-product integral cannot
        // represent. Both the clamp and that measurement are gone.)
        ElossJ += dt * (int2d(Th,1)(hBot*psiActive*(Temp - Tamb))
                      + int2d(Th,2)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,3)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,4)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,5)(hSurf*psiActive*(Temp - Tamb))
                      + int2d(Th,6)(hSurf*psiActive*(Temp - Tamb)));
        {boil_segment}
        Temp = max(Temp, Tamb);
        for(int iv=0; iv<Th.nv; iv++){{
            if(psiActive[][iv] < 0.5){{
                Temp[][iv] = Tamb;
            }}
        }}
        Temp = min(Temp, Tboil);

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
        // Track Tpeak on active nodes only
        for(int iv=0; iv<Th.nv; iv++){{
            if(psiActive[][iv] > 0.5){{
                Tpeak[][iv] = max(Tpeak[][iv], Temp[][iv]);
            }} else {{
                Tpeak[][iv] = Tamb;
            }}
        }}
        Temp0 = Temp;
        histCounter++; writeHistory(t+tLocal, 1, iL+1, histCounter);

{self._checkpoint_call(indent=8)}

        // Save
        if(ws%{p['save_every_n_weld']}==0){{
            string num;
            if      (frame<   10) num="000"+frame;
            else if (frame<  100) num="00" +frame;
            else if (frame< 1000) num="0"  +frame;
            else                  num=""   +frame;
            for(int iv=0;iv<Th.nv;iv++)
                TempVis[][iv] = (psiActive[][iv]>0.5) ? Temp[][iv] : -1.;
            int[int] ord=[1,1,1,1,1];
            savevtk(visdir+"T_"+num+".vtu", Th,
                Temp, TempVis, psiActive, Tpeak, dTdt,
                dataname="Temperature TempC Activation Tpeak CoolingRate_C_per_s", order=ord);
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

            recordPvd(t+tLocal, "T_"+num+".vtu");
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
{inweld_mech}
    }}
    // The runaway guard (ws < maxWeldAcc) must never trip silently: tripping
    // means the pass did not finish, usually because dt collapsed to its floor
    // (a broken h_min once did exactly this: 800 steps simulated 0.08 s while
    // the run reported success and deposited ~0.1 kJ). Fail loudly instead.
    if(ws >= maxWeldAcc && tLocal < tPassLayer - 1.e-9){{
        cout << "  [FATAL] weld-step runaway guard tripped on layer " << iL+1
             << ": " << ws << " steps covered only " << tLocal << " s of "
             << tPassLayer << " s. dt has collapsed (check h_min and the "
             << "adaptive controller). The pass did NOT finish; aborting rather "
             << "than reporting a partial deposition as a result." << endl;
        exit(1);
    }}
    // Consolidate full layer: all nodes in the deposited layer are activated.
    // The plate below is already permanently active (see the initial
    // conditions), so sweeping everything up to zLayTop is correct and also
    // re-affirms the substrate.
    for(int iv=0; iv<Th.nv; iv++){{
        if(Th(iv).z <= zLayTop + 1.e-5)
            psiActive[][iv] = 1.;
    }}
    {srcmask_step}
    bornMask = max(srcMask, bornFloor);
    t += tPassLayer;

    // INTER-PASS COOLING
    // The first seconds after arc extinction carry the entire post-peak
    // transient (Fo = alpha*dt/dx^2 ~ 1.3 for a flat 10 s step). A geometric
    // ramp 0.5/1/2/5 s resolves it, then coarse dtWait finishes the dwell.
    //
    // Two modes:
    //   time   - cool for exactly t_wait (legacy; the dwell is whatever the user
    //            guessed, so inter-pass temperature is uncontrolled).
    //   target - cool until the part's peak temperature falls to
    //            interpass_target_temp_C, then stop. This is the Ansys
    //            "Iterative Cooldown" idea and is what makes inter-pass
    //            temperature a controlled process variable rather than an
    //            emergent one.
    Tsource = 0.;
    real[int] dtRampW = [0.5, 1., 2., 5.];
    real tWaitRem = tWait;
    real iPassTarget = {ipass_target};
    int iw = 0;
    real ipassPrevPeak = Temp[].max;
    int ipassIters = 0;
    int ipassConverged = 0;
    while(tWaitRem > 1.e-9 || (ipassTargetMode == 1 && ipassConverged == 0)){{
        real dtStep;
        if(ipassTargetMode == 1 && ipassIters > 0){{
            // Steady part of target mode: uniform step, so the iteration count
            // maps cleanly onto cooling time.
            dtStep = {ipass_dt};
        }} else {{
            dtStep = (iw < dtRampW.n) ? min(dtRampW[iw], tWaitRem) : min(dtWait, tWaitRem);
        }}
        if(ipassTargetMode == 1 && tWaitRem <= 1.e-9) dtStep = min({ipass_dt}, 1.e12);
        dt = dtStep;
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
        // The floor always applies, so its energy is always counted; rectDebug
        // only controls the per-step print. Counting it conditionally while
        // applying it unconditionally once hid 8.8 kJ on a prescribed-T run.
        // Lumped-nodal sum, NOT int3d: the floor acts exactly where the field
        // oscillates (Gibbs undershoot = high-frequency), and consistent vs
        // lumped quadrature disagree most there. Every other nodal term
        // (EprescJ) is lumped; mixing them opens a phantom gap.
        // Lumped-nodal sum (same quadrature basis as EprescJ; see the
        // weld-loop site for the rationale).
        {{
            real eRectStep = 0.;
            for(int iv=0; iv<Th.nv; iv++){{
                real Tr = Temp[][iv];
                if(Tr < Tamb) eRectStep += lumpVol[iv]*rhoEff[][iv]*CpEff[][iv]*(Tamb - Tr);
            }}
            ErectJ += eRectStep;
        }}
        if(rectDebug && ErectJ > 0.)
            cout << "    [RECT] dt=" << dt << " ErectJ=" << ErectJ
                 << " Tmin=" << Temp[].min << endl;
        Temp = max(Temp, Tamb);
        for(int iv=0; iv<Th.nv; iv++){{
            if(psiActive[][iv] < 0.5){{
                Temp[][iv] = Tamb;
            }}
        }}
        Temp = min(Temp, Tboil);

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
            if(psiActive[][iv] > 0.5){{
                Tpeak[][iv] = max(Tpeak[][iv], Temp[][iv]);
            }} else {{
                Tpeak[][iv] = Tamb;
            }}
        }}
        Temp0 = Temp;
        histCounter++; writeHistory(t+dt, 2, iL+1, histCounter);

        // Save each dwell step to VTU and PVD
        {{
            string num;
            if      (frame<   10) num="000"+frame;
            else if (frame<  100) num="00" +frame;
            else if (frame< 1000) num="0"  +frame;
            else                  num=""   +frame;
            for(int iv=0;iv<Th.nv;iv++)
                TempVis[][iv] = (psiActive[][iv]>0.5) ? Temp[][iv] : -1.;
            int[int] ord=[1,1,1,1,1];
            savevtk(visdir+"T_"+num+".vtu", Th,
                Temp, TempVis, psiActive, Tpeak, dTdt,
                dataname="Temperature TempC Activation Tpeak CoolingRate_C_per_s", order=ord);
            recordPvd(t, "T_"+num+".vtu");
            frame++;
        }}

        t += dt;
        tWaitRem -= dt;
        iw++;
        ipassIters++;

        // Target-mode convergence: stop once the peak falls under the target, or
        // once the peak stops changing appreciably (matching the Ansys
        // "Convergence Threshold" idea), or when the iteration cap is reached.
        if(ipassTargetMode == 1){{
            real ipassPeak = Temp[].max;
            real ipassRel = 100.*abs(ipassPeak - ipassPrevPeak)/max(abs(ipassPrevPeak), 1.);
            ipassPrevPeak = ipassPeak;
            if(ipassPeak <= iPassTarget){{
                ipassConverged = 1;
                cout << "  [INTERPASS] layer " << iL+1 << ": reached target "
                     << iPassTarget << " C (peak " << ipassPeak << " C) after "
                     << ipassIters << " steps, " << t << " s total" << endl;
            }} else if(ipassRel < {ipass_tol_pct} && ipassIters > 1){{
                ipassConverged = 1;
                cout << "  [INTERPASS] layer " << iL+1 << ": converged, peak change "
                     << ipassRel << "% < " << {ipass_tol_pct}
                     << "% at peak " << ipassPeak << " C" << endl;
            }} else if(ipassIters >= {ipass_max_iters}){{
                cout << "  [WARN] Inter-pass cooling hit the iteration cap ("
                     << ipassIters << ") with peak " << ipassPeak
                     << " C still above target " << iPassTarget << " C." << endl;
                ipassConverged = 1;
            }}
        }}
    }}

    // Inter-pass mechanical equilibrium solve
    {{
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

        {self._mechanical_stage_solve() if p.get('enable_mechanical', True) else ''}
        cout << "  -> Inter-pass cool:            Tmax=" << Temp[].max << "C  t=" << t << "s" << endl;
        cout << endl;
        frame++;
    }}

{self._checkpoint_call_layer()}
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
        // The floor always applies, so its energy is always counted; rectDebug
        // only controls the per-step print. Counting it conditionally while
        // applying it unconditionally once hid 8.8 kJ on a prescribed-T run.
        // Lumped-nodal sum, NOT int3d: the floor acts exactly where the field
        // oscillates (Gibbs undershoot = high-frequency), and consistent vs
        // lumped quadrature disagree most there. Every other nodal term
        // (EprescJ) is lumped; mixing them opens a phantom gap.
        // Lumped-nodal sum (same quadrature basis as EprescJ; see the
        // weld-loop site for the rationale).
        {{
            real eRectStep = 0.;
            for(int iv=0; iv<Th.nv; iv++){{
                real Tr = Temp[][iv];
                if(Tr < Tamb) eRectStep += lumpVol[iv]*rhoEff[][iv]*CpEff[][iv]*(Tamb - Tr);
            }}
            ErectJ += eRectStep;
        }}
        if(rectDebug && ErectJ > 0.)
            cout << "    [RECT] dt=" << dt << " ErectJ=" << ErectJ
                 << " Tmin=" << Temp[].min << endl;
        Temp = max(Temp, Tamb);
        // Quiet (not-yet-deposited) nodes held at ambient: P1 source
        // interpolation smears arc energy across shared interface elements,
        // which otherwise superheats quiet nodes to 3000+ C.
        // Enthalpy the smeared P1 source has just parked on not-yet-deposited
        // (quiet) nodes, BEFORE they are clamped back to Tamb. This is an
        // energy FLOW: it must be multiplied by dt like every other per-step
        // term. It used to be accumulated unweighted, so the same stored
        // enthalpy was re-counted on every timestep. Measured: EquietJ reached
        // 121.7 kJ while the arc only ever injected 79.4 kJ, which is
        // impossible for a quantity that is supposed to be a subset of the
        // injected energy, and it turned the audit residual into -142%.
        // Enthalpy sitting on not-yet-deposited (quiet) nodes, BEFORE they are
        // clamped back to Tamb, i.e. the energy the clamp is about to destroy.
        // This is a STOCK, not a flow -- exactly like the boiling-cap term
        // EcapJ, which likewise accumulates without dt -- so it carries no dt.
        // (An earlier revision multiplied it by dt, which is dimensionally
        // Joule-seconds and wrong; fluxes carry dt, destroyed stocks do not.)
        //
        // With the P0 birth mask on the source this term should stay near zero:
        // front-straddling elements receive no source energy, so P1 assembly
        // can no longer smear it onto quiet vertices. Before the mask existed
        // it reached 121.7 kJ against 79.4 kJ injected -- more destroyed than
        // ever put in, the signature of assembly smear plus conducted-in heat
        // being counted as source energy. What remains is genuinely conducted
        // heat plus numerical noise at the front, both legitimately destroyed
        // by the clamp and legitimately counted here.
        // No Tamb clamp on quiet nodes. The clamp existed to suppress the
        // 3000 C superheat that P1 assembly smeared onto quiet vertices; the P0
        // birth mask (srcMask) stops that smear at the element level, so the
        // clamp's only remaining effect was destroying legitimate conducted
        // preheat ahead of the torch. Conducted preheat is physical -- it is
        // the heat-affected zone -- and stays in the stored enthalpy, where
        // the audit counts it. (An earlier revision measured a "quiet stock"
        // with int3d(...*(1-psiActive)); that integral is a P1 ramp inside
        // front-straddling elements, so it counted a fraction of the HOT bead
        // as quiet and reported more destroyed than injected. The nodal clamp
        // destroys only nodal values, which a field-product integral cannot
        // represent. Both the clamp and that measurement are gone.)
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
        histCounter++; writeHistory(t+dt, 3, Nlayers, histCounter);
        t += dt;
        tCoolRem -= dt;
        if(ic%nCoolSave==0){{
            string num;
            if      (frame<   10) num="000"+frame;
            else if (frame<  100) num="00" +frame;
            else if (frame< 1000) num="0"  +frame;
            else                  num=""   +frame;
            int[int] ord=[1,1,1,1,1];
            savevtk(visdir+"T_"+num+".vtu", Th,
                Temp, Temp, psiActive, Tpeak, dTdt,
                dataname="Temperature TempC Activation Tpeak CoolingRate_C_per_s", order=ord);
            recordPvd(t, "T_"+num+".vtu");
            cout<<"COOL t="<<t<<"s  Tmax="<<Temp[].max<<"C"<<endl;
            frame++;
        }}
        if(ic >= dtRampC.n && Temp[].max <= Tamb + 5.0){{
            cout << "  [COOLING] Part fully cooled to ambient room temperature (Tmax = " << Temp[].max << " C <= Tamb + 5 C). Dynamic cooldown complete at t=" << t << " s." << endl;
            break;
        }}
        ic++;
    }}
}}

// --- Final Mechanical Solve (Post Final Cooling at Room Temperature) ---
cout << endl;
cout << "================================================================================" << endl;
cout << " === FINAL MECHANICAL STAGE (Room Temperature Residual Stress & Distortion) ===" << endl;
cout << "================================================================================" << endl;
{self._mechanical_stage_solve() if p.get('enable_mechanical', True) else ''}
{springback_stage}
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
Sh hEnth;
hEnth = 0.;
for(int iv=0; iv<Th.nv; iv++){{
    real zvN = Th(iv).z;
    real rhoN = (zvN <= HzBase+1.e-6) ? rhoBase : rhoDep;
    hEnth[][iv] = rhoN * Hcp(Temp[][iv], zvN);
}}
real EstoredJ = int3d(Th)(hEnth);
// The Tamb floor rectifies Gibbs undershoots (consistent mass matrix + sharp
// source) into real energy. That energy is IN the stored enthalpy, so the audit
// must count it as a source, or the balance reads phantom. It is reported as
// its own line -- not hidden inside the residual -- so a mesh or dt change that
// makes undershoots worse shows up as a number instead of a mystery.
real EresJ = EarcJ + EprescJ + ErectJ - EcapJ - ElossJ - EquietJ - EstoredJ;
cout << " Intended arc input (Pweld*dt): " << EintJ/1000. << " kJ" << endl;
cout << " Injected arc heat (int Q dV):  " << EarcJ/1000. << " kJ (100 %)"
     << "  [fidelity: " << 100.*EarcJ/max(EintJ, 1.) << " % of intended]" << endl;
cout << " Prescribed-T imposed enthalpy:" << EprescJ/1000. << " kJ" << endl;
real EauditBase = max(EarcJ + EprescJ, 1.);
cout << " Stored enthalpy (final T):    " << EstoredJ/1000. << " kJ  ("
     << 100.*EstoredJ/EauditBase << " % of energy in)" << endl;
cout << " Boiling-cap removed:          " << EcapJ/1000. << " kJ  ("
     << 100.*EcapJ/EauditBase << " %)" << endl;
cout << " Surface losses (conv+rad):    " << ElossJ/1000. << " kJ  ("
     << 100.*ElossJ/EauditBase << " %)" << endl;
cout << " Quiet-node discarded:         " << EquietJ/1000. << " kJ  ("
     << 100.*EquietJ/EauditBase << " %)" << endl;
cout << " Tamb-floor added (rectifier): " << ErectJ/1000. << " kJ  ("
     << 100.*ErectJ/EauditBase << " %)" << endl;
cout << " Residual (numerical):         " << EresJ/1000. << " kJ  ("
     << 100.*EresJ/EauditBase << " %)" << endl;

// --- Structural result validity gate ---
// The reported warpage and residual stress are only meaningful if the final
// equilibrium iteration converged. Emitting them anyway is how an unconverged
// solve silently becomes a "result" (observed: a near-rigid fixture produced a
// bogus 3.2 mm warpage and a fictitious 2.25 mm springback that still parsed
// cleanly downstream). So gate the numbers and say plainly what went wrong.
if(FinalSolidConverged == 1){{
    cout << " Final cold-state residual warpage (Uz): " << warpageClampedFinal << " mm" << endl;
    cout << " Final cold-state peak von Mises stress: " << sigMMaxClampedFinal << " MPa" << endl;
}} else {{
    cout << " Final cold-state residual warpage (Uz):   NOT REPORTED - mechanical" << endl;
    cout << "     equilibrium did NOT converge (Fresid=" << residFrozen
         << ", Dresid=" << dresFrozen << " after " << itsFrozen
         << " iterations)." << endl;
    cout << " Final cold-state peak von Mises stress: NOT REPORTED - same reason." << endl;
    cout << "     DO NOT use these fields for design decisions. Likely causes:" << endl;
    cout << "       * fixture too stiff (lower fixture_stiffness_ratio_z/xy)" << endl;
    cout << "       * insufficient iterations (raise mech_max_newton_iter)" << endl;
    cout << "       * loose tolerances (tighten mech_newton_tol / _tol_force)" << endl;
}}
cout << " ParaView PVD file:             " << visdir << "waam.pvd" << endl;
cout << " Mechanical VTU file:           " << visdir << "waam_mech.vtu" << endl;
cout << "================================================================================" << endl;

// --- MACHINE-READABLE RESULTS ---
// The post-processor must never parse the human-readable banner above. Doing so
// is how an unconverged solve still ends up in run_summary.json: the banner
// wording changed once and every extracted distortion/stress silently became
// 0.0, while an unconverged run kept matching an intermediate line and reported
// iteration error as a result. This file is the authoritative record; the
// `converged` flag travels WITH the numbers so consumers cannot miss it.
{{
    ofstream rf(outdir+"solver_results.json");
    rf << "{{";
    rf << "\\n  \\"schema\\": \\"waam-solver-results/1\\",";
    rf << "\\n  \\"mechanics_converged\\": " << (FinalSolidConverged == 1 ? "true" : "false") << ",";
    rf << "\\n  \\"newton_iterations\\": " << itsFrozen << ",";
    rf << "\\n  \\"newton_force_residual\\": " << residFrozen << ",";
    rf << "\\n  \\"newton_disp_residual\\": " << dresFrozen << ",";
    rf << "\\n  \\"sim_time_s\\": " << t << ",";
    rf << "\\n  \\"thermal_frames\\": " << frame << ",";
    rf << "\\n  \\"nlayers\\": " << Nlayers << ",";
    // Structural quantities are emitted as JSON null when unconverged, so a
    // consumer that reads the number without checking the flag still cannot
    // obtain a fabricated value.
    if(FinalSolidConverged == 1){{
        rf << "\\n  \\"peak_von_mises_MPa\\": " << sigMMaxClampedFinal << ",";
        rf << "\\n  \\"max_uz_mm\\": " << warpageClampedFinal << ",";
        rf << "\\n  \\"max_disp_mm\\": " << dispMaxClampedFinal << ",";
    }} else {{
        rf << "\\n  \\"peak_von_mises_MPa\\": null,";
        rf << "\\n  \\"max_uz_mm\\": null,";
        rf << "\\n  \\"max_disp_mm\\": null,";
    }}
    rf << "\\n  \\"energy_audit_J\\": {{";
    rf << "\\n    \\"intended_arc\\": " << EintJ << ",";
    rf << "\\n    \\"injected_arc\\": " << EarcJ << ",";
    rf << "\\n    \\"stored_enthalpy\\": " << EstoredJ << ",";
    rf << "\\n    \\"boiling_cap_removed\\": " << EcapJ << ",";
    rf << "\\n    \\"surface_losses\\": " << ElossJ << ",";
    rf << "\\n    \\"quiet_node_discarded\\": " << EquietJ << ",";
    rf << "\\n    \\"tamb_floor_added\\": " << ErectJ << ",";
    rf << "\\n    \\"prescribed_imposed\\": " << EprescJ << ",";
    rf << "\\n    \\"numerical_residual\\": " << EresJ;
    rf << "\\n  }},";
    rf << "\\n  \\"max_peak_temp_C\\": " << Tpeak[].max;
    rf << "\\n}}\\n";
}}
cout << " Machine-readable results:      " << outdir << "solver_results.json" << endl;
cout << endl;
"""
        return script

    def _mesh_block(self) -> str:
        """Generate mesh creation or import block."""
        p = self.p
        mesh_src = p.get("mesh_source", "parametric")
        mesh_file = p.get("mesh_file", "").strip()
        stl_file = p.get("stl_file", "").strip()

        # If stl_file is provided and mesh_file not yet generated, resolve or
        # auto-mesh it. Failures propagate: an unusable mesh must never reach
        # the solver (the old silent fallback produced a surface mesh that made
        # FreeFEM abort with SIGBUS long after the pipeline had been paid for).
        if (not mesh_file or not os.path.exists(mesh_file)) and stl_file and os.path.exists(stl_file):
            outdir = os.path.abspath(p.get("output_dir", "./waam_output"))
            os.makedirs(outdir, exist_ok=True)
            candidate_mesh = os.path.join(outdir, Path(stl_file).stem + ".mesh")
            if not os.path.exists(candidate_mesh):
                from mesh.stl_pipeline import stl_to_freefem_mesh
                mesh_res = stl_to_freefem_mesh(
                    stl_file, candidate_mesh,
                    target_nodes=int(p.get("target_nodes", 2000)),
                    layer_height=float(p.get("layer_height_mm", 0.0)) or 3.0,
                    # MUST match the solver's geometry: hz_base marks the
                    # substrate (z <= HzBase) and lx/ly_base are the plate plan
                    # dimensions the torch sweeps across.
                    sub_hz_mm=float(p.get("hz_base", 0.005)) * 1000.0,
                    sub_lx_mm=float(p.get("lx", 0.150)) * 1000.0,
                    sub_ly_mm=float(p.get("ly_base", 0.150)) * 1000.0,
                )
                for w in mesh_res.get("warnings", []) or []:
                    self.warnings.append(f"CAD mesh: {w}")
                if float(mesh_res.get("scale_applied", 1.0)) != 1.0:
                    self.warnings.append(
                        f"CAD geometry was auto-scaled by "
                        f"{mesh_res['scale_applied']:.4f}x to fit the build "
                        f"envelope; verify part geometry and any mass/energy "
                        f"figures quoted downstream.")
            if os.path.exists(candidate_mesh):
                mesh_file = candidate_mesh

        # Check if valid FreeFEM .mesh / .mesh3 file or Gmsh MSH
        if mesh_file and os.path.exists(mesh_file) and mesh_file.lower().endswith((".mesh", ".mesh3", ".msh")):
            mesh_file = os.path.abspath(mesh_file)
            if mesh_file.lower().endswith(".msh"):
                load_cmd = 'load "gmsh"\nmesh3 Th = gmshload3("' + mesh_file + '");'
            else:
                load_cmd = 'load "msh3"\nmesh3 Th = readmesh3("' + mesh_file + '");'
            # The CAD/STL mesh produced by mesh/stl_pipeline.py already contains
            # the substrate plate AND the deposited part as ONE conformal
            # tetrahedral volume (fused before meshing), with the boundary
            # labels the forms below expect:
            #     1 = substrate underside (hBot + fixture spring)
            #     5 = top, 2/3/4/6 = remaining surfaces (all convective)
            # A previous revision concatenated it with a separately built plate
            # (`Th = ThBase + ThCad`). FreeFEM's `+` merges NO nodes, so that
            # interface was a zero-width crack: no conduction into the plate
            # and no mechanical bond (measured 19.9 C mean / 92.9 C peak
            # discontinuity across it). The domain is now read as one mesh, so
            # that failure mode is structurally impossible.
            return f"""{load_cmd}
cout << " CAD/conformal domain mesh: " << Th.nv << " nodes  " << Th.nt << " tets" << endl;
// Boundary labels come from the mesh file (see _BOUNDARY_LABEL in
// mesh/structured_conforming.py): 1 = substrate underside, 2..6 = the
// convective/radiative surfaces. The plate inside the file spans z in
// [0, HzBase]; everything below HzBase is substrate."""
        else:
            # ---- Conforming single-mesh parametric domain --------------------
            # Total deposit height must match the free-FEM variable HzDep
            #   HzDep = (Nlayers <= 1) ? Nlayers*hBead1 : hBead1 + (Nlayers-1)*hBeadSub
            # so the generated mesh bounds agree with the layer logic below.
            _nlay = int(p.get("n_layers", 1))
            _hb1 = float(p.get("h_bead_first", p.get("h_bead", 0.004)))
            _hbs = float(p.get("h_bead_subseq", _hb1))
            _hz_dep = (_nlay * _hb1) if _nlay <= 1 else (_hb1 + (_nlay - 1) * _hbs)

            # The previous implementation built two independent cube() meshes and
            # concatenated them (`Th = ThBase + ThDep`). Their Y node spacings
            # differ, so although the faces are coplanar at z = HzBase NOT ONE
            # vertex coincides and FreeFEM does not merge them. The interface was
            # therefore a crack of zero width: measured on a running model, a mean
            # 19.9 C and peak 92.9 C temperature discontinuity across it, no heat
            # conduction from bead to plate, and no mechanical bonding at all.
            #
            # A deposited track is FUSED to the plate, so the interface is now
            # generated as ONE conforming structured mesh with a graded Y
            # direction: fine across the bead width, coarse across the rest of
            # the plate. Shared interface nodes come by construction.
            out_dir_candidate = str(p.get("output_dir", "."))
            if out_dir_candidate.startswith("/tmp") or not os.access(out_dir_candidate, os.W_OK):
                framework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fallback_dir = os.path.join(framework_dir, "waam_output", "cache")
                os.makedirs(fallback_dir, exist_ok=True)
                out_dir_candidate = fallback_dir
            mesh_file_gen = os.path.join(out_dir_candidate, "waam_domain.mesh")
            # nz_dep semantics follow nz_dep_follows_layers (see
            # _auto_size_source): True -> the value is the TOTAL deposit cell
            # count; False -> it is PER LAYER and is scaled up here, so the
            # meshed deposit always has the resolution the source was sized
            # against. (Passing the raw value through in the False case built
            # a one-cell-per-layer mesh while the source assumed more.)
            _nz_total = int(p.get("nz_dep", 1))
            if not bool(p.get("nz_dep_follows_layers", True)):
                _nz_total = _nz_total * max(int(p.get("n_layers", 1)), 1)
            try:
                from mesh.structured_conforming import generate_waam_mesh
                info = generate_waam_mesh(
                    mesh_file_gen,
                    lx=float(p["lx"]), ly_base=float(p["ly_base"]),
                    hz_base=float(p["hz_base"]), ty_dep=float(p["ty_dep"]),
                    hz_dep=_hz_dep,
                    nx_dep=int(p["nx_dep"]), ny_dep=int(p["ny_dep"]),
                    nz_dep=_nz_total,
                    nx_base=int(p["nx_base"]), ny_base=int(p["ny_base"]),
                    nz_base=int(p["nz_base"]),
                )
                self.warnings.append(
                    f"Parametric mesh regenerated as a conforming bonded domain: "
                    f"{info['n_nodes']} nodes, {info['n_tets']} tets "
                    f"(nx={info['n_x']}, ny={info['n_y']}, nz={info['n_z']}); "
                    "deposit/substrate interface shares nodes."
                )
            except Exception as ex:
                raise ValueError(
                    f"conforming mesh generation failed: {ex}") from ex

            return f"""load "msh3"
mesh3 Th = readmesh3("{mesh_file_gen}");
cout << " Mesh (conforming, bonded interface): " << Th.nv << " nodes  "
     << Th.nt << " tets" << endl;"""

    def _prescribed_T_override(self) -> str:
        """Override temperature for prescribed-T heat source."""
        p = self.p
        return f"""        // Prescribed temperature override: the melt pool. Nodes within
        // R_activate of the torch that are active but below liquidus are
        // brought up to liquidus -- the Ansys prescribed-activation
        // methodology. The gate is Temp < Tliq, NOT coldness: gating on
        // Temp0 < Tamb+10 disqualified every node conduction had preheated,
        // so on aluminium (fast conduction) only ~6% of the bead was ever
        // born and the part never melted. A melt pool holds material AT
        // process temperature; preheated nodes are topped up, cold nodes are
        // born, nothing above Tliq is ever cooled by this block.
        // The imposed enthalpy is REAL energy entering the part, counted as
        // true Hcp differences into EprescJ; otherwise the audit shows stored
        // energy with no source.
        // Hard-ball birth at liquidus: nodes within R_activate of the torch
        // that are active but below liquidus are brought up to Tliq. A smooth
        // (cosine-squared) pool profile was tried and reverted: on coarse
        // meshes the torch threads between nodes so imposition is hit-or-miss,
        // and even resolved it imposes ~3x less energy for the same R, which
        // broke a passing validation (0.17x -> 0.05x). The hard ball's step
        // does Gibbs-ring (bounded by mesh resolution, reported via the
        // rectifier + residual metrics); the smooth ball's under-drive is
        // silent. Audited energy is exact either way (nodal true enthalpy).
        for(int iv=0; iv<Th.nv; iv++){{
            real rPool = sqrt((Th(iv).x-xTorch)^2 + (Th(iv).y-yTorch)^2 + (Th(iv).z-zTorch)^2);
            if(rPool < RpoolAct && psiActive[][iv] > 0.5 && Temp[][iv] < Tliq){{
                real Tv0 = Temp[][iv];
                real zvP = Th(iv).z;
                real rhoP = (zvP <= HzBase + 1.e-6) ? rhoBase : rhoDep;
                Temp[][iv] = Tliq;
                EprescJ += lumpVol[iv]*rhoP*max(Hcp(Tliq, zvP)-Hcp(Tv0, zvP), 0.);
            }}
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

    
    def _mech_fe(self) -> str:
        """Fespace family for mechanics: P1 normally, P2 with mechanics_order=2."""
        return "P2" if int(self.p.get("mechanics_order", 1)) == 2 else "P1"

    def _mech_order(self) -> int:
        """Mechanics fespace order (1 or 2), validated."""
        o = int(self.p.get("mechanics_order", 1))
        if o not in (1, 2):
            raise ValueError(f"mechanics_order must be 1 or 2, got {o}")
        return o

    def _mechanical_stage_setup(self) -> str:
        p = self.p
        mo = self._mech_order()
        # Gravity body force for the mechanics load. Uses the z-dispatched
        # density already declared in the script (rhoBase/rhoDep) and psiM so
        # only activated material carries weight.
        if bool(p.get("mech_gravity", True)):
            g_val = float(p.get("gravity_m_s2", 9.81))
            # rho must be a FIELD, not the ternary that used to be written
            # inside the varf: a step in z evaluated at quadrature points is a
            # discontinuous coefficient smeared over the interface elements,
            # and it disagreed with the nodal rhoEff the thermal stage uses.
            # rhoM is the interpolated copy of rhoEff, updated every increment.
            grav_term = "int3d(Th)(rhoM * psiM * %f * (-vz))" % g_val
        else:
            grav_term = "0."
        bc_mode = p.get("mechanical_bc", "isostatic").lower()
        if bc_mode == "fixture_clamped":
            bc_code = "+ on(1, ux=0., uy=0., uz=0.) + on(2, ux=0., uy=0., uz=0.)"
        elif bc_mode == "cantilever":
            bc_code = "+ on(1, ux=0., uy=0., uz=0.)"
        else:
            bc_code = ""

        use_consistent = bool(p.get("mech_use_consistent_tangent", True))
        # Extra fields/varf for the consistent-tangent Newton path. In OFF mode
        # both expand to nothing, so the emitted script is byte-identical to
        # the pre-toggle generator output (strongest possible regression gate
        # for the validated Picard path).
        mech_tan_fields_eol = ("Mh muTan, thetaC;  // consistent-tangent "
                               "shear + return scale (Newton path only)"
                               if use_consistent else "")
        if use_consistent:
            amech_tan_varf = f"""
// Consistent-tangent stiffness (mech_use_consistent_tangent=1). Identical to
// aMech except the shear field: muTan = mu*theta carries the radial-return
// consistent shear (theta = 1-3*mu*dGamma/seq, floored), so yielded zones
// assemble their true (softened) stiffness instead of the elastic one and the
// Newton correction is correctly scaled. The volumetric response needs no
// correction -- J2 plastic flow is deviatoric, so lambdaL is already the
// exact consistent bulk response. First Newton pass has theta = 1 everywhere
// (elastic predictor); subsequent passes soften where the return flowed.
varf aMechTan([ux,uy,uz],[vx,vy,vz]) =
    int3d(Th)(
        lambdaL*(dx(ux)+dy(uy)+dz(uz))*(dx(vx)+dy(vy)+dz(vz))
      + 2.*muTan*(
            dx(ux)*dx(vx) + dy(uy)*dy(vy) + dz(uz)*dz(vz)
          + 0.5*(dy(ux)+dx(uy))*(dy(vx)+dx(vy))
          + 0.5*(dz(ux)+dx(uz))*(dz(vx)+dx(vz))
          + 0.5*(dz(uy)+dy(uz))*(dz(vy)+dy(vz))
        )
    )
    + int2d(Th, 1)(
        (clampActive*clampKz + (1.-clampActive)*clampKz*sbReg)*uz*vz
      + (clampActive*clampKxy + (1.-clampActive)*clampKxy*sbReg)*(ux*vx + uy*vy)
    )
    {bc_code};"""
        else:
            amech_tan_varf = ""
        # _validate() -> _derive_fixture_stiffness() guarantees these are set,
        # either from the plate bending stiffness or from a user override.
        k_spring_z = float(p["k_spring_z"])
        k_spring_xy = float(p["k_spring_xy"])
        sb_reg = float(p.get("springback_regularization", 1.0e-6))
        if not (0.0 < sb_reg < 1.0e-3):
            raise ValueError(
                f"springback_regularization={sb_reg:g} outside (0, 1e-3): "
                f"larger values restrain the part, zero reintroduces the "
                f"singular released system.")

        return f"""
// --- 3D NONLINEAR THERMO-ELASTO-PLASTIC CONSTITUTIVE ENGINE ---
// Mechanical state lives in Mh (P1 or P2 per mechanics_order), NEVER in Sh:
// Th.nv-indexed access is only valid for P1, so every mechanical loop below is
// dof-indexed (nDofM) with coordinates and temperature read from fields that
// exist at every dof under both orders. xxM/yyM/zzM are exact (coordinate
// interpolation reproduces node positions); TempM is the P1 temperature sampled
// at the mechanical dofs (exact: a P1 field is determined by its vertex
// values); psiM copies the binary activation (an edge midpoint straddling the
// front interpolates to exactly 0.5 and counts as active -- half a cell of
// early activation, conservative and documented).
Mh Emod, lambdaL, muL, alphaL;{mech_tan_fields_eol}
Mh epsThXX, epsThYY, epsThZZ;
Mh epsTrXX, epsTrYY, epsTrZZ, epsTrXY, epsTrXZ, epsTrYZ; // TRIP strain tensor
Mh epsPx, epsPy, epsPz, epsPxy, epsPxz, epsPyz;          // Plastic strain tensor
Mh peeq;                                                 // Equivalent plastic strain
Mh fMartOld, fBainOld;                                   // Phase history for TRIP
Mh sigInhXX, sigInhYY, sigInhZZ, sigInhXY, sigInhXZ, sigInhYZ;
Mh sigYY, sigZZ, sigXY, sigXZ, sigYZ;                    // Cauchy stress tensor components (sigXX, sigM, dispM are pre-declared)
Mh xxM = x, yyM = y, zzM = z;  // dof coordinates (exact under both orders)
Mh TempM, psiM;                // temperature / activation sampled at mech dofs
Mh rhoM;                       // density sampled at mech dofs (see grav_term)
Mh fMartM, fBainM;             // phase fractions sampled at mech dofs
int nDofM = Emod.n;

// --- Newton state (kept across incremental solves) ---
int  newtonDone = 0;        // 1 if the last solve converged
real lastNewtonResid = 0.;  // final relative force residual
real lastNewtonDres = 0.;   // final relative displacement residual
int  lastNewtonIts = 0;     // iterations used in the last solve
int mechOrderCur = {mo}; // fespace order tag for checkpoint guard
int  FinalSolidConverged = 0; // frozen verdict for the reported structural state
real warpageClampedFinal = 0.;  // uz [mm], clamped build state
real dispMaxClampedFinal = 0.;  // max displacement [mm], clamped build state
real sigMMaxClampedFinal = 0.;  // peak von Mises [MPa], clamped build state
real residFrozen = 1., dresFrozen = 1.; // Newton residuals of the reported state
int itsFrozen = 0;                      // Newton iterations of the reported state

real clampKz = {k_spring_z};
real clampKxy = {k_spring_xy};
real clampActive = 1.0; // 1.0 during welding, 0.0 post-weld (clamp release)
// Released-state regularisation, fraction of the fixture stiffness (see the
// varf comment). Parameterised: 1e-9 leaves near-nullspace modes the Picard
// iteration cannot settle (measured: free re-solve unconverged at 150 iters);
// 1e-6 settles them while restraining six orders below the fixture.
real sbReg = {sb_reg};

// J2 iteration tracing (mech_j2_debug): diagnostic only, off by default. Prints
// the state of the node with the largest plastic increment each iteration,
// which is what tells a diverging hardening law from a singular stiffness.
bool j2Debug = {str(bool(p.get("mech_j2_debug", False))).lower()};
real lastResidPrev = 1.;

// Live Variational Forms for 3D Mechanics with dynamic Robin spring clamping
// Fixture surface. Label convention follows the FreeFEM cube() ordering that
// the structured mesh reproduces: 1 = zmin (plate underside), 5 = zmax,
// 2/4 = xmax/xmin, 3/6 = ymax/ymin. A build plate rests on its table, so the
// isostatic restraint belongs on label 1 (the underside). It used to be applied
// to label 5, i.e. to the TOP of the bead plus the plate's upper face outside
// the track. That let the plate rigid-body translate and rotate almost freely:
// the bead is hot and soft, so the spring carried almost nothing, the stiffness
// matrix went nearly singular, and the Picard iteration oscillated
// (Fresid 0.34 -> 0.17, Dresid 0.47 -> 0.25) instead of converging.
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
    // Clamp release vs singularity. The old code kept a 1e2 N/m^3 baseline alive
    // after release, so the "free" springback solve reported a spring-supported
    // part. Removing it entirely makes the released system SINGULAR instead --
    // the isostatic mode carries no Dirichlet conditions, so the direct solver
    // returns an arbitrary nullspace combination (measured: 2.1e7 mm "warpage",
    // i.e. kilometres, with a converged residual). The middle path below keeps
    // a regularization spring at 1e-9 of the fixture stiffness, active ONLY in
    // the released state: nine orders below the restraint it replaces, so its
    // pull on the deformation is negligible, while the six rigid modes gain
    // finite (if soft) eigenvalues and the solve is unique. The defined proper
    // fix is a 3-2-1 isostatic point support; until that lands, this term is
    // labelled for what it is and the reported delta is guarded below against
    // rigid-body runaway.
    + int2d(Th, 1)(
        (clampActive*clampKz + (1.-clampActive)*clampKz*sbReg)*uz*vz
      + (clampActive*clampKxy + (1.-clampActive)*clampKxy*sbReg)*(ux*vx + uy*vy)
    )
    {bc_code};
{amech_tan_varf}
varf lMech([ux,uy,uz],[vx,vy,vz]) =
    int3d(Th)(
        sigInhXX*dx(vx) + sigInhYY*dy(vy) + sigInhZZ*dz(vz)
      + sigInhXY*(dy(vx)+dx(vy))
      + sigInhXZ*(dz(vx)+dx(vz))
      + sigInhYZ*(dz(vy)+dy(vz))
    )
    // Self-weight. REQUIRED to capture melt-pool sagging, blow-through and loss
    // of wall height (the defects in the reference figure): with no body force
    // molten metal has nothing pushing it down, so it keeps the shape of the bead
    // it was born in no matter how far above liquidus it gets. The solver had NO
    // gravity term at all, which made those defects unmodellable by construction.
    // rho is z-dispatched and psiM is the same activation field the stiffness
    // uses, so only material that exists carries weight, and melted metal
    // (full weight, no strength) sags.
    + {grav_term}
    ;
"""

    def _mechanical_stage_solve(self) -> str:
        """Entry point: run one incremental mechanics step.

        Kept as the public name because it is spliced into the generated script at
        the inter-pass and final-cooling call sites. All the work happens in
        _mechanical_stage_increment(), which _mechanical_stage_springback() also
        reuses after releasing the fixture.
        """
        p = self.p
        enable_us = bool(p.get("enable_ultrasonic", False))
        freq = float(p.get("ultrasonic_frequency", 0.0)) if enable_us else 0.0
        softening_factor = 1.0 - min(0.3, freq / 100000.0) if (enable_us and freq > 0) else 1.0
        nlgeom = bool(p.get("nlgeom", False))

        inc = self._mechanical_stage_increment()
        mo = self._mech_order()
        uva_mech = self._ultrasonic_mechanical_block()

        # Final bookkeeping / VTU dump, shared by every increment.
        tail = f"""
        dispM = sqrt(ux^2+uy^2+uz^2);
        cout << "  -> Transient Peak Displacement: " << dispM[].max*1000. << " mm | Vertical Warpage (Uz): " << uz[].max*1000. << " mm | Peak Stress: " << sigM[].max/1.e6 << " MPa" << endl;
        cout << "  -> Equilibrium: " << lastNewtonIts << " Newton iterations, "
             << "force resid " << lastNewtonResid << ", disp resid " << lastNewtonDres
             << (newtonDone ? "  [CONVERGED]" : "  [NOT CONVERGED - max iters]") << endl;

        // Feature 4: Geometric Nonlinearity (NLGEOM)
        if ({1 if nlgeom else 0} == 1) {{
            Th = movemesh3(Th, transfo=[x + ux, y + uy, z + uz]);
            cout << "  -> NLGEOM: mesh coordinates updated with current displacement field." << endl;
        }}

        // UVA mechanical physics (vibration stress relief + momentum transfer)
        {uva_mech}

        string numM;
        if      (frame<   10) numM="000"+frame;
        else if (frame<  100) numM="00" +frame;
        else if (frame< 1000) numM="0"  +frame;
        else                  numM=""   +frame;
        
        {{
            int mechOrd = {mo};
            int[int] ord=[mechOrd,mechOrd,mechOrd,mechOrd,mechOrd,mechOrd,mechOrd,mechOrd,mechOrd,mechOrd];
            savevtk(visdir+"waam_mech_"+numM+".vtu", Th,
                ux, uy, uz, dispM, Tpeak, peeq, sigM, sigXX, epsPx, epsPy,
                dataname="Ux Uy Uz DispMag Tpeak PEEQ vonMises_Pa Sxx_Pa EpsP_xx EpsP_yy",
                order=ord);
        }}
        {{
            int[int] ord2 = [1,1,1,1,1,1,1];
            savevtk(visdir+"waam_uva_"+numM+".vtu", Th,
                sigUVA, reliefFrac, QacouField,
                dataname="SigUVA_Pa ReliefFrac QacouField_W_m3",
                order=ord2);
        }}
"""
        return inc + tail

    def _mechanical_stage_increment(self) -> str:
        """Emit one incremental thermo-elasto-plastic equilibrium solve.

        Method: elastic predictor / plastic corrector with a Newton loop on the
        global residual. Each iteration (a) forms the trial elastic stress from
        the total strain minus the committed eigenstrain/plastic strain, (b)
        performs the J2 radial return to obtain the updated plastic strain and
        Cauchy stress, and (c) rebuilds the internal-force RHS. Convergence is
        measured on BOTH the residual force norm (R = F_ext - F_int) and the
        incremental displacement norm, which is the same criterion Abaqus uses
        for its equilibrium check. When "mech_use_consistent_tangent" is on, the
        elastic predictor is scaled by the standard radial-return consistent
        moduli so Newton converges quadratically near the solution instead of
        only linearly.
        """
        p = self.p
        enable_us = bool(p.get("enable_ultrasonic", False))
        freq = float(p.get("ultrasonic_frequency", 0.0)) if enable_us else 0.0
        softening_factor = 1.0 - min(0.3, freq / 100000.0) if (enable_us and freq > 0) else 1.0
        enable_j2 = bool(p.get("enable_j2_plasticity", True))
        enable_j2_str = "1" if enable_j2 else "0"
        h_prime = float(p.get("h_prime_iso", 1.5e9))
        j2_relax = float(p.get("mech_j2_relax", 1.0))
        dgamma_max = float(p.get("mech_dgamma_max", 1.0))
        peeq_limit = float(p.get("peeq_limit", 0.25))
        plastic_tol = float(p.get("mech_plastic_tol", 1.0e-3))
        max_newton = max(1, int(p.get("mech_max_newton_iter", 12)))
        newton_tol = float(p.get("mech_newton_tol", 1.0e-6))
        newton_tol_f = float(p.get("mech_newton_tol_force", 1.0e-6))
        trip_beta = float(p.get("trip_beta", 0.011))

        mo = self._mech_order()

        use_consistent = bool(p.get("mech_use_consistent_tangent", True))
        OFF_VECS_TXT = f"""        bool vecSized = false;
        real[int] Kux(0), fPrev(0), uCur(0), uPrev(0);"""
        OFF_SIGINH_TXT = f"""            // 2a. Internal force from the committed plastic + eigenstrain
            for(int iv=0; iv<nDofM; iv++){{
                real lam = lambdaL[][iv];
                real muu = muL[][iv];
                real eXX = epsPx[][iv] + epsThXX[][iv] + epsTrXX[][iv];
                real eYY = epsPy[][iv] + epsThYY[][iv] + epsTrYY[][iv];
                real eZZ = epsPz[][iv] + epsThZZ[][iv] + epsTrZZ[][iv];
                real trE = eXX + eYY + eZZ;
                sigInhXX[][iv] = lam*trE + 2.*muu*eXX;
                sigInhYY[][iv] = lam*trE + 2.*muu*eYY;
                sigInhZZ[][iv] = lam*trE + 2.*muu*eZZ;
                sigInhXY[][iv] = 2.*muu*(epsPxy[][iv] + epsTrXY[][iv]);
                sigInhXZ[][iv] = 2.*muu*(epsPxz[][iv] + epsTrXZ[][iv]);
                sigInhYZ[][iv] = 2.*muu*(epsPyz[][iv] + epsTrYZ[][iv]);
            }}

"""
        OFF_SOLVE_TXT = f"""            // 2b. Tangent and internal-force vector
            matrix KM = aMech(Vh, Vh);
            set(KM, solver={self.solver_kw});
            real[int] fEig = lMech(0, Vh);
            if(!vecSized){{
                Kux.resize(fEig.n); fPrev.resize(fEig.n);
                uCur.resize(fEig.n); uPrev.resize(fEig.n);
                uPrev = uCur; vecSized = true;
            }}

            // 2c. Solve the linearised equilibrium K u = f_int(eps_p_current)
            ux[] = KM^-1 * fEig;

            // 2d. Out-of-balance force against the PREVIOUS iterate's internal
            //     force: R_n = K u_n - f_int(u_prev). Using f_int(u_n) instead
            //     would give R == 0 identically (since u_n solves exactly that
            //     linearised system) and the force test would be vacuous.
            //     f_int(u_prev) is already in fPrev from the previous pass.
            real Rnorm = 0.;
            Kux = KM*ux[];
            if(nIter == 0){{
                // No previous iterate yet: seed with the elastic predictor so
                // Fref reflects the initial out-of-balance magnitude.
                for(int i=0; i<fEig.n; i++) Rnorm += fEig[i]*fEig[i];
                fPrev = fEig;
            }} else {{
                for(int i=0; i<fEig.n; i++){{
                    real ri = Kux[i] - fPrev[i];
                    Rnorm += ri*ri;
                }}
                fPrev = fEig;
            }}
            Rnorm = sqrt(Rnorm);
            if(nIter == 0) Fref = max(Rnorm, 1.);

            // 2e. Displacement change since the previous iterate, measured in
            //     flat dof space (component slices of a vector fespace cannot be
            //     used as scalar fields in FreeFEM, so this is done on dofs).
            uCur = ux[];
            real Dnorm = 0.;
            for(int i=0; i<uCur.n; i++){{
                real di = uCur[i] - uPrev[i];
                if(abs(di) > Dnorm) Dnorm = abs(di);
            }}
            uPrev = uCur;

"""
        OFF_J2_TXT = f"""            // 2f. J2 return map on the stress implied by the new displacement,
            //     updating the committed plastic strain (this is what makes the
            //     next iteration's internal force different).

            real maxDpeeq = 0.0;
            real maxPeeqTotal = 0.0;
            real dbgMu=0., dbgSigY=0., dbgFY=0., dbgDG=0., dbgSeq=0., dbgT=0., dbgPe=0.;
            int dbgIv=0;
            for(int iv=0; iv<nDofM; iv++){{
                real px = xxM[][iv], py = yyM[][iv], pz = zzM[][iv];
                if(psiM[][iv] < 0.5 || TempM[][iv] >= TsolMat(pz)) continue;
                real exx = dx(ux)(px,py,pz);
                real eyy = dy(uy)(px,py,pz);
                real ezz = dz(uz)(px,py,pz);
                real exy = 0.5*(dy(ux)(px,py,pz) + dx(uy)(px,py,pz));
                real exz = 0.5*(dz(ux)(px,py,pz) + dx(uz)(px,py,pz));
                real eyz = 0.5*(dz(uy)(px,py,pz) + dy(uz)(px,py,pz));

                real eeXX = exx - epsPx[][iv] - epsThXX[][iv] - epsTrXX[][iv];
                real eeYY = eyy - epsPy[][iv] - epsThYY[][iv] - epsTrYY[][iv];
                real eeZZ = ezz - epsPz[][iv] - epsThZZ[][iv] - epsTrZZ[][iv];
                real eeXY = exy - epsPxy[][iv] - epsTrXY[][iv];
                real eeXZ = exz - epsPxz[][iv] - epsTrXZ[][iv];
                real eeYZ = eyz - epsPyz[][iv] - epsTrYZ[][iv];

                real trEe = eeXX + eeYY + eeZZ;
                real devXX = eeXX - trEe/3.0;
                real devYY = eeYY - trEe/3.0;
                real devZZ = eeZZ - trEe/3.0;

                real muu = muL[][iv];
                real lam = lambdaL[][iv];
                real sTrialXX = 2.*muu*devXX;
                real sTrialYY = 2.*muu*devYY;
                real sTrialZZ = 2.*muu*devZZ;
                real sTrialXY = 2.*muu*eeXY;
                real sTrialXZ = 2.*muu*eeXZ;
                real sTrialYZ = 2.*muu*eeYZ;

                real J2 = 0.5*(sTrialXX*sTrialXX + sTrialYY*sTrialYY + sTrialZZ*sTrialZZ)
                        + sTrialXY*sTrialXY + sTrialXZ*sTrialXZ + sTrialYZ*sTrialYZ;
                real seqTrial = sqrt(3.0 * max(J2, 0.0));

                real sigY0 = (pz <= HzBase + 1.e-6) ? sigYieldBase : sigYieldDep;
                real Tfact = max(0.05, 1.0 - 0.85 * min(1.0, max(0.0, (TempM[][iv]-Tamb)/(TsolMat(pz)-Tamb))));
                // Acoustic softening acts on the flow stress: sigY is the
                // critical resolved shear at which plasticity starts, so a
                // vibrated bead yields earlier without any change to E.
                real sigY = (sigY0 * Tfact + HprimeIso * peeq[][iv]) * softenF;
                real fYield = seqTrial - sigY;

                if({enable_j2_str} == 1 && fYield > 0.0){{
                    // Successive under-relaxation of the plastic increment.
                    // The Picard/Jacobi update re-derives dGamma from the current
                    // displacement every iteration; with sharp P2 strain gradients
                    // the full step overshoots and limit-cycles (measured: force
                    // resid pinned at 0.37 over 150 iterations where P1 converges).
                    // Scaling the APPLIED increment by j2Relax < 1 damps the cycle
                    // without moving the fixed point (at convergence fYield = 0
                    // so dGamma = 0 whatever the factor). Default 1.0 (validated
                    // P1 path); P2 runs need ~0.5.
                    // Trust region on the plastic step. With sharp P2 strain
                    // gradients the raw dGamma is enormous and successive iterates
                    // chase hotspot-to-hotspot without settling (measured force
                    // resid pinned at 0.37 over 150 iterations). Capping the
                    // APPLIED step does not move the fixed point: at convergence
                    // fYield -> 0 so dGamma -> 0, below any cap. It only bounds
                    // the transient path, like any trust-region method. Default
                    // 1.0 = off (validated P1 path untouched); P2 needs ~0.005.
                    real dGamma = min(j2Relax * fYield / (3.0 * muu + HprimeIso),
                                      dGammaMax);
                    // Plastic flow must be bounded per node. The Picard loop
                    // re-evaluates the trial stress from the current displacement
                    // every iteration and adds a FRESH dGamma onto the accumulated
                    // peeq, so an uncapped accumulator diverges: measured peeq
                    // 0.68 -> 1.23 over 16 iterations at T=879 C, with the
                    // hardening modulus feeding itself (sigY 1126 -> 1959 MPa) and
                    // never reaching a fixed point. A weld bead cannot absorb 123%
                    // plastic strain; beyond ~peeq_limit the material has localised
                    // and failed, which Abaqus represents with *DEFORMATION
                    // PLASTICITY damage rather than unbounded flow. Past the cap
                    // the node stops accumulating and stops hardening.
                    real dGamEff = dGamma;
                    if(peeq[][iv] >= peeqLimit) dGamEff = 0.;
                    // The APPLIED increment is dGamEff. The old line advanced the
                    // state by the raw dGamma while tracking dGamEff for the
                    // convergence measure, so a node at the cap kept hardening
                    // (and kept accumulating) while the monitor reported a
                    // zero increment.
                    peeq[][iv] = min(peeq[][iv] + dGamEff, peeqLimit);
                    if(dGamEff > maxDpeeq){{
                        maxDpeeq = dGamEff; dbgIv = iv; dbgMu = muu; dbgSigY = sigY;
                        dbgFY = fYield; dbgDG = dGamEff; dbgSeq = seqTrial;
                        dbgT = TempM[][iv]; dbgPe = peeq[][iv];
                    }}

                    real nFactor = 1.5 * dGamEff / max(seqTrial, 1.0);
                    epsPx[][iv] += nFactor * sTrialXX;
                    epsPy[][iv] += nFactor * sTrialYY;
                    epsPz[][iv] += nFactor * sTrialZZ;
                    epsPxy[][iv] += nFactor * sTrialXY;
                    epsPxz[][iv] += nFactor * sTrialXZ;
                    epsPyz[][iv] += nFactor * sTrialYZ;

                    real scaleS = max(0.0, 1.0 - 3.0*muu*dGamEff/max(seqTrial, 1.0));
                    real pHyd = lam*trEe;
                    sigXX[][iv] = pHyd + scaleS*sTrialXX;
                    sigYY[][iv] = pHyd + scaleS*sTrialYY;
                    sigZZ[][iv] = pHyd + scaleS*sTrialZZ;
                    sigXY[][iv] = scaleS*sTrialXY;
                    sigXZ[][iv] = scaleS*sTrialXZ;
                    sigYZ[][iv] = scaleS*sTrialYZ;
                    sigM[][iv]  = sigY + HprimeIso*dGamEff;
                }} else {{
                    real pHyd = lam*trEe;
                    sigXX[][iv] = pHyd + sTrialXX;
                    sigYY[][iv] = pHyd + sTrialYY;
                    sigZZ[][iv] = pHyd + sTrialZZ;
                    sigXY[][iv] = sTrialXY;
                    sigXZ[][iv] = sTrialXZ;
                    sigYZ[][iv] = sTrialYZ;
                    sigM[][iv]  = seqTrial;
                }}
            }}

            for(int iv=0; iv<nDofM; iv++)
                if(peeq[][iv] > maxPeeqTotal) maxPeeqTotal = peeq[][iv];

            if(j2Debug){{
                cout << "    [J2] it=" << nIter
                     << " Tres=" << lastResidPrev
                     << " mu=" << dbgMu/1.e9
                     << " sigY=" << dbgSigY/1.e6
                     << " seq=" << dbgSeq/1.e6
                     << " fY=" << dbgFY/1.e6
                     << " dGam=" << dbgDG
                     << " peeq=" << dbgPe
                     << " T=" << dbgT
                     << " (node " << dbgIv << ")" << endl;
            }}

"""
        ON_J2_TXT = f"""            // 2f. J2 return map on the stress implied by the new displacement,
            //     updating the committed plastic strain (this is what makes the
            //     next iteration's internal force different).

            real maxDpeeq = 0.0;
            real maxPeeqTotal = 0.0;
            real dbgMu=0., dbgSigY=0., dbgFY=0., dbgDG=0., dbgSeq=0., dbgT=0., dbgPe=0.;
            int dbgIv=0;
            for(int iv=0; iv<nDofM; iv++){{
                real px = xxM[][iv], py = yyM[][iv], pz = zzM[][iv];
                if(psiM[][iv] < 0.5 || TempM[][iv] >= TsolMat(pz)) continue;
                real exx = dx(ux)(px,py,pz);
                real eyy = dy(uy)(px,py,pz);
                real ezz = dz(uz)(px,py,pz);
                real exy = 0.5*(dy(ux)(px,py,pz) + dx(uy)(px,py,pz));
                real exz = 0.5*(dz(ux)(px,py,pz) + dx(uz)(px,py,pz));
                real eyz = 0.5*(dz(uy)(px,py,pz) + dy(uz)(px,py,pz));

                real eeXX = exx - epsPx[][iv] - epsThXX[][iv] - epsTrXX[][iv];
                real eeYY = eyy - epsPy[][iv] - epsThYY[][iv] - epsTrYY[][iv];
                real eeZZ = ezz - epsPz[][iv] - epsThZZ[][iv] - epsTrZZ[][iv];
                real eeXY = exy - epsPxy[][iv] - epsTrXY[][iv];
                real eeXZ = exz - epsPxz[][iv] - epsTrXZ[][iv];
                real eeYZ = eyz - epsPyz[][iv] - epsTrYZ[][iv];

                real trEe = eeXX + eeYY + eeZZ;
                real devXX = eeXX - trEe/3.0;
                real devYY = eeYY - trEe/3.0;
                real devZZ = eeZZ - trEe/3.0;

                real muu = muL[][iv];
                real lam = lambdaL[][iv];
                real sTrialXX = 2.*muu*devXX;
                real sTrialYY = 2.*muu*devYY;
                real sTrialZZ = 2.*muu*devZZ;
                real sTrialXY = 2.*muu*eeXY;
                real sTrialXZ = 2.*muu*eeXZ;
                real sTrialYZ = 2.*muu*eeYZ;

                real J2 = 0.5*(sTrialXX*sTrialXX + sTrialYY*sTrialYY + sTrialZZ*sTrialZZ)
                        + sTrialXY*sTrialXY + sTrialXZ*sTrialXZ + sTrialYZ*sTrialYZ;
                real seqTrial = sqrt(3.0 * max(J2, 0.0));

                real sigY0 = (pz <= HzBase + 1.e-6) ? sigYieldBase : sigYieldDep;
                real Tfact = max(0.05, 1.0 - 0.85 * min(1.0, max(0.0, (TempM[][iv]-Tamb)/(TsolMat(pz)-Tamb))));
                // Acoustic softening acts on the flow stress: sigY is the
                // critical resolved shear at which plasticity starts, so a
                // vibrated bead yields earlier without any change to E.
                real sigY = (sigY0 * Tfact + HprimeIso * peeq[][iv]) * softenF;
                real fYield = seqTrial - sigY;

                if({enable_j2_str} == 1 && fYield > 0.0){{
                    // Successive under-relaxation of the plastic increment.
                    // The Picard/Jacobi update re-derives dGamma from the current
                    // displacement every iteration; with sharp P2 strain gradients
                    // the full step overshoots and limit-cycles (measured: force
                    // resid pinned at 0.37 over 150 iterations where P1 converges).
                    // Scaling the APPLIED increment by j2Relax < 1 damps the cycle
                    // without moving the fixed point (at convergence fYield = 0
                    // so dGamma = 0 whatever the factor). Default 1.0 (validated
                    // P1 path); P2 runs need ~0.5.
                    // Trust region on the plastic step. With sharp P2 strain
                    // gradients the raw dGamma is enormous and successive iterates
                    // chase hotspot-to-hotspot without settling (measured force
                    // resid pinned at 0.37 over 150 iterations). Capping the
                    // APPLIED step does not move the fixed point: at convergence
                    // fYield -> 0 so dGamma -> 0, below any cap. It only bounds
                    // the transient path, like any trust-region method. Default
                    // 1.0 = off (validated P1 path untouched); P2 needs ~0.005.
                    real dGamma = min(j2Relax * fYield / (3.0 * muu + HprimeIso),
                                      dGammaMax);
                    // Plastic flow must be bounded per node. The Picard loop
                    // re-evaluates the trial stress from the current displacement
                    // every iteration and adds a FRESH dGamma onto the accumulated
                    // peeq, so an uncapped accumulator diverges: measured peeq
                    // 0.68 -> 1.23 over 16 iterations at T=879 C, with the
                    // hardening modulus feeding itself (sigY 1126 -> 1959 MPa) and
                    // never reaching a fixed point. A weld bead cannot absorb 123%
                    // plastic strain; beyond ~peeq_limit the material has localised
                    // and failed, which Abaqus represents with *DEFORMATION
                    // PLASTICITY damage rather than unbounded flow. Past the cap
                    // the node stops accumulating and stops hardening.
                    real dGamEff = dGamma;
                    if(peeq[][iv] >= peeqLimit) dGamEff = 0.;
                    // The APPLIED increment is dGamEff. The old line advanced the
                    // state by the raw dGamma while tracking dGamEff for the
                    // convergence measure, so a node at the cap kept hardening
                    // (and kept accumulating) while the monitor reported a
                    // zero increment.
                    peeq[][iv] = min(peeq[][iv] + dGamEff, peeqLimit);
                    if(dGamEff > maxDpeeq){{
                        maxDpeeq = dGamEff; dbgIv = iv; dbgMu = muu; dbgSigY = sigY;
                        dbgFY = fYield; dbgDG = dGamEff; dbgSeq = seqTrial;
                        dbgT = TempM[][iv]; dbgPe = peeq[][iv];
                    }}

                    real nFactor = 1.5 * dGamEff / max(seqTrial, 1.0);
                    epsPx[][iv] += nFactor * sTrialXX;
                    epsPy[][iv] += nFactor * sTrialYY;
                    epsPz[][iv] += nFactor * sTrialZZ;
                    epsPxy[][iv] += nFactor * sTrialXY;
                    epsPxz[][iv] += nFactor * sTrialXZ;
                    epsPyz[][iv] += nFactor * sTrialYZ;

                    real scaleS = max(0.0, 1.0 - 3.0*muu*dGamEff/max(seqTrial, 1.0));
                    thetaC[][iv] = max(0.05, scaleS);  // consistent-tangent scale
                    real pHyd = lam*trEe;
                    sigXX[][iv] = pHyd + scaleS*sTrialXX;
                    sigYY[][iv] = pHyd + scaleS*sTrialYY;
                    sigZZ[][iv] = pHyd + scaleS*sTrialZZ;
                    sigXY[][iv] = scaleS*sTrialXY;
                    sigXZ[][iv] = scaleS*sTrialXZ;
                    sigYZ[][iv] = scaleS*sTrialYZ;
                    sigM[][iv]  = sigY + HprimeIso*dGamEff;
                }} else {{
                    real pHyd = lam*trEe;
                    sigXX[][iv] = pHyd + sTrialXX;
                    sigYY[][iv] = pHyd + sTrialYY;
                    sigZZ[][iv] = pHyd + sTrialZZ;
                    sigXY[][iv] = sTrialXY;
                    sigXZ[][iv] = sTrialXZ;
                    sigYZ[][iv] = sTrialYZ;
                    sigM[][iv]  = seqTrial;
                    thetaC[][iv] = 1.0;  // elastic: full shear stiffness
                }}
            }}

            for(int iv=0; iv<nDofM; iv++)
                if(peeq[][iv] > maxPeeqTotal) maxPeeqTotal = peeq[][iv];

            if(j2Debug){{
                cout << "    [J2] it=" << nIter
                     << " Tres=" << lastResidPrev
                     << " mu=" << dbgMu/1.e9
                     << " sigY=" << dbgSigY/1.e6
                     << " seq=" << dbgSeq/1.e6
                     << " fY=" << dbgFY/1.e6
                     << " dGam=" << dbgDG
                     << " peeq=" << dbgPe
                     << " T=" << dbgT
                     << " (node " << dbgIv << ")" << endl;
            }}

"""
        if use_consistent:
            newton_vecs = ("real[int] Rv(0), duv(0);\n"
                           "        bool nvecSized = false;\n")
            tangent_init = "        // Consistent-tangent state reset (Newton path): theta = 1\n        // (elastic) until the first return pass writes the yielded values.\n        for(int iv=0; iv<nDofM; iv++){\n            thetaC[][iv] = 1.0;\n            muTan[][iv] = muL[][iv];\n        }\n"
            kela_assembly = f"""// Elastic stiffness assembled ONCE per increment: E(T) is fixed
        // for the whole solve, so Kela never changes. It serves the true
        // residual R = F_load - Kela*u every iteration (one matvec each, no
        // refactorisation).
        matrix Kela = aMech(Vh, Vh);
        set(Kela, solver={self.solver_kw});
        """
            solve_block = f"""            // 2b. TRUE OUT-OF-BALANCE from the current state.
            //     R = F_load - Kela*u, with F_load from the eigenstress
            //     refreshed from this pass's return just below. No lagged
            //     previous-iterate force: the old test compared against fPrev,
            //     which made it vacuous whenever the load barely moved.
            real[int] fLoad = lMech(0, Vh);
            real[int] KuEla = Kela*ux[];
            if(!nvecSized){{ 
                Rv.resize(fLoad.n); duv.resize(fLoad.n);
                nvecSized = true;
            }}
            real Rnorm = 0.;
            for(int i=0; i<fLoad.n; i++){{ 
                Rv[i] = fLoad[i] - KuEla[i];
                Rnorm += Rv[i]*Rv[i];
            }}
            Rnorm = sqrt(Rnorm);
            if(nIter == 0) Fref = max(Rnorm, 1.);

            // 2c. CONSISTENT-TANGENT Newton correction: Ktan*du = R.
            //     Ktan carries muTan = mu*theta (radial-return consistent
            //     shear; the volumetric response is exact for J2). First pass
            //     has theta = 1 everywhere: the elastic predictor.
            matrix Ktan = aMechTan(Vh, Vh);
            set(Ktan, solver={self.solver_kw});
            duv = Ktan^-1*Rv;

            // 2d. Apply the correction; incremental-displacement norm.
            real Dnorm = 0.;
            for(int i=0; i<duv.n; i++){{ 
                ux[][i] = ux[][i] + duv[i];
                if(abs(duv[i]) > Dnorm) Dnorm = abs(duv[i]);
            }}"""

            solve_block = solve_block + "\n"
            tangent_update = "        // Consistent shear from this pass's return: mu_tan = mu*theta.\n        // Volumetric response needs no correction (J2 flow is deviatoric).\n        for(int iv=0; iv<nDofM; iv++){\n            muTan[][iv] = muL[][iv]*thetaC[][iv];\n        }\n"
            siginh_block = ""
            j2_block = ON_J2_TXT
        else:
            newton_vecs = OFF_VECS_TXT + "\n"
            tangent_init = ""
            kela_assembly = ""
            solve_block = OFF_SOLVE_TXT
            tangent_update = ""
            siginh_block = OFF_SIGINH_TXT
            j2_block = OFF_J2_TXT
        return f"""
        // --- Incremental thermo-elasto-plastic equilibrium solve ---
        real softenF = {softening_factor};
        real HprimeIso = {h_prime};      // Isotropic hardening modulus [Pa]
        real j2Relax = {j2_relax};        // Picard damping on dGamma (see above)
        real dGammaMax = {dgamma_max};    // trust-region cap (see above)
        real peeqLimit = {peeq_limit};   // strain-localisation cap [dimensionless]
        real plasticTol = {plastic_tol};  // relative plastic-increment tolerance
        real betaTrans = {trip_beta};    // Austenite -> Martensite/Bainite volumetric expansion ratio
        // Reference scale for the relative-convergence test. It must NOT be taken
        // from the current displacement: on the first solve u is still zero, which
        // collapses the denominator and produces a meaningless ratio. Instead we
        // use the natural displacement magnitude of this problem -- the free
        // thermal expansion of the peak temperature over the model size -- which
        // is known before the solve starts.
        real alphaPeak = max(alphaMat(Tpeak[].max, HzBase/2.), 1.e-9);
        real uChar = max(alphaPeak * max(Tpeak[].max - Tamb, 0.) * HzTot, 1.e-11);
        // Sample P1 fields onto the mechanical dofs. Exact for Temp (a P1
        // field is fixed by its vertex values) and for coordinates; phase
        // fractions and activation interpolate linearly (documented above).
        TempM = Temp; psiM = psiActive; rhoM = rhoEff;
        fMartM = fMartensite; fBainM = fBainite;
        real uRef = max(abs(ux[].max) + abs(ux[].min), uChar);
        real Fref = 0.;                                        // reference internal-force scale

        // 1. Annealing, eigenstrain (thermal + TRIP), and modulus update.
        // Dof-indexed (order-agnostic): coordinates, temperature and activation
        // come from the sampled fields, never from Th(iv) / Th.nv.
        for(int iv=0; iv<nDofM; iv++){{
            real zv = zzM[][iv];
            real Tv = TempM[][iv];
            real TsolNode = TsolMat(zv);
            real psiEff = max(psiM[][iv], 1.e-4);

            // Feature 2: High-Temperature Annealing & Plastic Memory Loss
            if(Tv >= TsolNode){{
                peeq[][iv]    = 0.0;
                epsPx[][iv]   = 0.0; epsPy[][iv]   = 0.0; epsPz[][iv]   = 0.0;
                epsPxy[][iv]  = 0.0; epsPxz[][iv]  = 0.0; epsPyz[][iv]  = 0.0;
                epsTrXX[][iv] = 0.0; epsTrYY[][iv] = 0.0; epsTrZZ[][iv] = 0.0;
                epsTrXY[][iv] = 0.0; epsTrXZ[][iv] = 0.0; epsTrYZ[][iv] = 0.0;
                sigXX[][iv]   = 0.0; sigYY[][iv]   = 0.0; sigZZ[][iv]   = 0.0;
                sigXY[][iv]   = 0.0; sigXZ[][iv]   = 0.0; sigYZ[][iv]   = 0.0;
                // Liquid/slurry cutoff. Above solidus the node is treated as a
                // very soft 'slurry' so it cannot carry load. It must not be
                // SOFT that it is effectively unrestrained though: at 1e6 Pa the
                // node contributes ~3.8e5 Pa of shear stiffness, i.e. a nearly
                // free-floating dof that sits in the stiffness matrix and in the
                // load vector while the J2 loop skips it. That singular dof is
                // what pinned the equilibrium iteration at Dresid ~0.57 in the
                // layer-2 solve (Tmax 1485 C > Tsol 1450 C). A small fraction of
                // the surrounding solid stiffness keeps the dof well posed
                // without letting the melted region carry meaningful stress.
                real EmodSolid = EMat(TsolMat(zv)-1., zv);
                Emod[][iv]    = max(1.e6, 0.02*EmodSolid); // 2% of solid stiffness
            }} else {{
                // NOTE: no softenF here. The elastic modulus is a
                // lattice-elastic quantity and is not what ultrasonic
                // assistance reduces -- softening the modulus would lower the
                // bending stiffness and change the whole stress field, not the
                // plastic limit. The factor is applied to the yield surface
                // inside the J2 return below.
                Emod[][iv]    = EMat(Tv, zv)*psiEff;
            }}

            // Feature 3: Thermal Expansion Strain
            real alph = alphaMat(Tv, zv);
            real deltaT = Tv - Tamb;
            if(psiM[][iv] > 0.5){{
                epsThXX[][iv] = alph * deltaT;
                epsThYY[][iv] = alph * deltaT;
                epsThZZ[][iv] = alph * deltaT;
            }} else {{
                epsThXX[][iv] = 0.0; epsThYY[][iv] = 0.0; epsThZZ[][iv] = 0.0;
            }}

            // Feature 3: TRIP (Transformation-Induced Plasticity) & Volumetric Dilation
            real dFmart = max(0.0, fMartM[][iv] - fMartOld[][iv]);
            real dFbain = max(0.0, fBainM[][iv] - fBainOld[][iv]);
            real dFtrans = dFmart + dFbain;
            fMartOld[][iv] = fMartM[][iv];
            fBainOld[][iv] = fBainM[][iv];

            if(dFtrans > 1.e-4 && psiM[][iv] > 0.5 && Tv < TsolNode){{
                real epsVol = (1.0/3.0) * betaTrans * dFtrans;
                real sigYNode = (zv <= HzBase + 1.e-6) ? sigYieldBase : sigYieldDep;
                real tripCoeff = (5.0/6.0) * (betaTrans / max(sigYNode, 1.e6)) * dFtrans;
                real meanS = (sigXX[][iv] + sigYY[][iv] + sigZZ[][iv]) / 3.0;
                epsTrXX[][iv] += tripCoeff * (sigXX[][iv] - meanS) + epsVol;
                epsTrYY[][iv] += tripCoeff * (sigYY[][iv] - meanS) + epsVol;
                epsTrZZ[][iv] += tripCoeff * (sigZZ[][iv] - meanS) + epsVol;
                epsTrXY[][iv] += tripCoeff * sigXY[][iv];
                epsTrXZ[][iv] += tripCoeff * sigXZ[][iv];
                epsTrYZ[][iv] += tripCoeff * sigYZ[][iv];
            }}
        }}

        lambdaL = Emod*nuS/((1.+nuS)*(1.-2.*nuS));
        muL     = Emod/(2.*(1.+nuS));

        newtonDone = 0; lastNewtonIts = 0; lastNewtonResid = 1.; lastNewtonDres = 1.;

        // 2. Equilibrium iteration on the thermo-elasto-plastic problem.
        //
        //    The eigenstrain (thermal expansion + TRIP) is a TOTAL quantity and
        //    so is the plastic strain, so the equilibrium equation to solve is
        //        K(u) u = f_int(eps_th, eps_p(u))
        //    i.e. it is nonlinear only through the plastic strain. The natural
        //    iteration is therefore a Jacobi/Picard iteration on eps_p, which is
        //    what we do: each iteration (a) assembles the internal force from the
        //    current committed plastic + eigenstrain, (b) solves K u = f_int, and
        //    (c) return-maps the trial stress to update eps_p.
        //
        //    What this adds over the previous implementation, which simply ran a
        //    fixed number of iterations and only watched the plastic increment,
        //    is a genuine equilibrium test on the two quantities Abaqus checks:
        //      * the out-of-balance force  R_n = K u_n - f_int(u_n-1)
        //      * the displacement change   du = u_n - u_n-1
        //    so a run now reports CONVERGED / NOT CONVERGED honestly instead of
        //    silently stopping after N iterations.
        //
        //    The verdict is frozen into FinalSolidConverged here, at the end of
        //    THIS solve, instead of being read from newtonDone at write-out time.
        //    The springback stage re-solves and overwrites newtonDone, so a failed
        //    clamped solve followed by a converged unclamped re-solve used to
        //    report mechanics_converged=true and publish warpage and stress that
        //    were pure iteration error.
        // The Picard iteration converges LINEARLY and slowly here: the plastic
        // field must relax across the whole plate before the out-of-balance force
        // settles. Measured on the bonded 2-layer benchmark the layer-2 solve
        // needed 145 iterations to reach Fresid 5.6e-5; at the previous default
        // of 40 it stalled at 0.39 and the run published iteration error as a
        // result. 150 is the measured requirement for that case, not a guess.
        int maxNewt = {max_newton};
{newton_vecs}{tangent_init}{kela_assembly}        for(int nIter=0; nIter<maxNewt; nIter++){{
{siginh_block}{solve_block}{j2_block}{tangent_update}            // 2e. Convergence: force residual AND incremental displacement.
            real relR = Rnorm / max(Fref, 1.);
            real relD = Dnorm / max(uRef, 1.e-12);
            lastNewtonResid = relR; lastNewtonDres = relD; lastNewtonIts = nIter + 1;

            lastResidPrev = relR;
            // Plastic-flow criterion. An ABSOLUTE bound on the plastic increment
            // is the wrong test for a Picard iteration: the last node still
            // shedding plasticity decays as ~1/n, so it would need hundreds of
            // iterations to fall under 1e-5 even while the force and
            // displacement residuals are already ~1e-4. Abaqus instead compares
            // the increment to the ACCUMULATED equivalent plastic strain (a
            // relative PCTOL), which asks the physical question: has plastic flow
            // stopped relative to how much strain the node has already taken?
            bool plasticSettled = (maxDpeeq <= plasticTol * max(maxPeeqTotal, 1.e-4));
            if(relR < {newton_tol_f} && relD < {newton_tol} && plasticSettled){{
                newtonDone = 1;
                break;
            }}
        }}

        FinalSolidConverged = newtonDone;
        residFrozen = lastNewtonResid; dresFrozen = lastNewtonDres; itsFrozen = lastNewtonIts;
        warpageClampedFinal = uz[].max*1000.;
        dispMaxClampedFinal = dispM[].max*1000.;
        sigMMaxClampedFinal = sigM[].max/1.e6;

        if(!newtonDone)
            cout << "  [WARN] Mechanics did not converge in " << maxNewt
                 << " iterations (Fresid=" << lastNewtonResid
                 << ", Dresid=" << lastNewtonDres << "). Results may be unconverged." << endl;
"""

    def _mechanical_stage_springback(self) -> str:
        """Release the fixture and re-solve to obtain the true springback.

        This is the Abaqus "unclamping" benchmark from the WAAM literature: with
        the clamp springs removed, the elastically-strained plate redistributes
        its residual stress into additional deflection. The delta between the
        clamped and unclamped warpage is the springback, reported separately.
        """
        return f"""
        // --- Springback: release the fixture and re-equilibrate ---
        // Springback is only valid if BOTH the clamped and the unclamped state
        // reached equilibrium. An unconverged unclamped re-solve produces a
        // large apparent "springback" that is pure iteration error, so we gate
        // the reported delta on the convergence flag.
        {{
            int sbClampedConverged = newtonDone;
            real warpageClamped = uz[].max*1000.;
            real dispMaxClamped = dispM[].max*1000.;
            real sigMMaxClamped = sigM[].max/1.e6;
            int sbSavedConverged = FinalSolidConverged;
            real residSaved = residFrozen, dresSaved = dresFrozen; int itsSaved = itsFrozen;
            clampActive = 0.0;    // release the fixture springs
            cout << "  [SPRINGBACK] Releasing fixture (clampActive=0) and re-solving..." << endl;
            {self._mechanical_stage_increment()}
            int sbFreeConverged = newtonDone;
            real warpageUnclamped = uz[].max*1000.;
            // Rigid-body runaway guard: a free warpage larger than ten model
            // diagonals is not deformation but an unsuppressed nullspace mode
            // (the convergence test alone cannot see it -- K*u=f holds along
            // the nullspace). Report divergence, not kilometres.
            real modelDiagMm = sqrt(LxW^2+LyBase^2+HzTot^2)*1000.;
            int sbRunaway = (abs(warpageUnclamped) > 10.*modelDiagMm) ? 1 : 0;
            if(sbClampedConverged == 1 && sbFreeConverged == 1 && sbRunaway == 0){{
                cout << "  [SPRINGBACK] Warpage clamped  = " << warpageClamped << " mm" << endl;
                cout << "  [SPRINGBACK] Warpage free     = " << warpageUnclamped << " mm" << endl;
                cout << "  [SPRINGBACK] Springback delta = "
                     << (warpageUnclamped - warpageClamped) << " mm" << endl;
            }} else {{
                cout << "  [SPRINGBACK] NOT REPORTED - equilibrium did not converge "
                     << "(clamped converged=" << sbClampedConverged
                     << ", free converged=" << sbFreeConverged
                     << ", rigid runaway=" << sbRunaway << ")." << endl;
                cout << "  [SPRINGBACK] A delta computed from an unconverged re-solve is "
                     << "iteration error, not springback. Lower the fixture stiffness "
                     << "ratio or raise mech_max_newton_iter." << endl;
            }}
            clampActive = 1.0;    // restore for any later re-use
            // The unclamped re-solve overwrote newtonDone and the displacement
            // and stress fields. Restore the verdict and the frozen clamped
            // numbers, so the reported build state is the clamped one and not
            // the springback configuration.
            FinalSolidConverged = sbSavedConverged;
            residFrozen = residSaved; dresFrozen = dresSaved; itsFrozen = itsSaved;
            warpageClampedFinal = warpageClamped;
            dispMaxClampedFinal = dispMaxClamped;
            sigMMaxClampedFinal = sigMMaxClamped;
        }}
"""


    # ------------------------------------------------------ history / probes
    def _checkpoint_call_layer(self) -> str:
        """Emit the layer-boundary checkpoint (the authoritative resume point)."""
        if str(self.p.get("restart", "off")).lower() in ("off", "0", "false", "none", ""):
            return ""
        return """
    // Checkpoint at the layer boundary. This is the only point at which the state
    // is fully consistent (whole layer deposited + inter-pass cooled + mechanical
    // solve complete), so resuming from here is exact rather than approximate.
    saveCheckpoint(t, iL+1, frame, histCounter,
                   EarcJ, EintJ, ElossJ, EquietJ, EcapJ, newtonDone, EprescJ);
    cout << " [RESTART] Checkpoint written after layer " << iL+1
         << " (t=" << t << "s)" << endl;"""

    def _checkpoint_call(self, indent: int = 8) -> str:
        """Emit the intra-layer checkpoint call, or nothing when restart is off.

        saveCheckpoint() only exists when checkpointing is enabled, so referring
        to it in a disabled run is a compile error rather than a no-op.
        """
        if str(self.p.get("restart", "off")).lower() in ("off", "0", "false", "none", ""):
            return ""
        pad = " " * indent
        return f"""{pad}// Intra-layer checkpoint. The resume unit is a whole layer (the
{pad}// layer-boundary checkpoint is the authoritative, exactly-consistent one),
{pad}// so this only exists to bound the work lost if a run dies mid-layer.
{pad}if(histCounter % restartEvery == 0){{
{pad}    saveCheckpoint(t+tLocal, iL, frame, histCounter,
{pad}                   EarcJ, EintJ, ElossJ, EquietJ, EcapJ, newtonDone, EprescJ);
{pad}}}"""

    # ---------------------------------------------------------- restart
    def _restart_block(self) -> str:
        """Emit checkpoint save/restore for the full thermal + mechanical state.

        Abaqus-equivalent restart: a long WAAM build can be resumed from the last
        completed layer instead of restarting from scratch. The checkpoint stores
        every field that defines the state, plus the loop counters and energy
        accumulators, so the resumed run continues with identical physics.

        Writing and reading FE fields must happen in separate scopes so the
        ofstream is flushed before the ifstream opens -- FreeFEM does not flush
        eagerly and reading a still-open file fails.
        """
        p = self.p
        mode = str(p.get("restart", "off")).lower()
        resume = 1 if p.get("restart_resume", True) else 0
        every = max(1, int(p.get("restart_every", 25)))
        has_mech = bool(p.get("enable_mechanical", True))
        if mode in ("off", "0", "false", "none", ""):
            # The deposition loop references these globals unconditionally, so
            # they must exist even when checkpointing is disabled.
            return """
// --- Checkpoint / restart: DISABLED ---
string ckptFile = outdir + "checkpoint.bin";
int useRestart = 0;
int restartEvery = {every};
int doResume = 0;
int resumeOk = 0;
int resumeFromLayer = 0;
real ckptTime = 0.; int ckptFrame = 0; int ckptHist = 0;
real ckptEarc = 0.; real ckptEint = 0.; real ckptEloss = 0.;
real ckptEquiet = 0.; real ckptEcap = 0.; real ckptEpresc = 0.;
""".format(every=every)
        m_fields = ""
        m_load = ""
        m_tmp_decl = ""
        if has_mech:
            m_fields = ("\n        f << Emod[]; f << epsPx[]; f << epsPy[]; f << epsPz[];"
                        "\n        f << epsPxy[]; f << epsPxz[]; f << epsPyz[];"
                        "\n        f << peeq[]; f << fMartOld[]; f << fBainOld[];"
                        "\n        f << ux[]; f << uy[]; f << uz[];")
            # NOTE: Mh/Vh are order-selected, so resume buffers match the
            # run's order by construction; cross-order resume is refused by the
            # stored order tag (v3) instead of silently misaligning dofs.
            m_tmp_decl = ("\nMh rEmod, rEpsPx, rEpsPy, rEpsPz, rEpsPxy, rEpsPxz,"
                          "\n   rEpsPyz, rPeeq, rFM, rFB;\nVh [rUx, rUy, rUz];")
            m_load = ("\n        g >> rEmod[]; g >> rEpsPx[]; g >> rEpsPy[]; g >> rEpsPz[];"
                      "\n        g >> rEpsPxy[]; g >> rEpsPxz[]; g >> rEpsPyz[];"
                      "\n        g >> rPeeq[]; g >> rFM[]; g >> rFB[];"
                      "\n        g >> rUx[]; g >> rUy[]; g >> rUz[];"
                      "\n        Emod[] = rEmod[]; epsPx[] = rEpsPx[]; epsPy[] = rEpsPy[];"
                      "\n        epsPz[] = rEpsPz[]; epsPxy[] = rEpsPxy[];"
                      "\n        epsPxz[] = rEpsPxz[]; epsPyz[] = rEpsPyz[];"
                      "\n        peeq[] = rPeeq[]; fMartOld[] = rFM[]; fBainOld[] = rFB[];"
                      "\n        ux[] = rUx[]; uy[] = rUy[]; uz[] = rUz[];")

        return f"""
// --- Checkpoint / restart ---
// The binary header already carries the loop counters, so a single file is the
// whole checkpoint. (A text sidecar was tried first and removed: FreeFEM cannot
// reliably extract `string` tokens from a stream, and a mixed chained
// `>> s >> r >> s` aborts with "file not good in read array".)
string ckptFile = outdir + "checkpoint.bin";
int useRestart = 1;
int restartEvery = {every};
int doResume = {resume};
// Scalars populated by loadCheckpoint() (FreeFEM functions cannot return several
// values, so the loaded header is published through globals).
real ckptTime = 0.; int ckptFrame = 0; int ckptHist = 0; int resumeFromLayer = 0;
real ckptEarc = 0.; real ckptEint = 0.; real ckptEloss = 0.;
real ckptEquiet = 0.; real ckptEcap = 0.; real ckptEpresc = 0.;
{m_tmp_decl}

func int saveCheckpoint(real tSave, int layerDone, int frameSave, int histSave,
                       real eA, real eI, real eL, real eQ, real eC, int mechConv,
                       real eP){{
    // Format version 2: appends the prescribed-T imposed energy eP after the
    // field block. Version 1 files (no eP) still load, with EprescJ = 0.
    // Format version 3: appends the mechanics fespace order after eP. A P1
    // checkpoint resumed into a P2 run (or vice versa) would silently misalign
    // every mechanical dof, so the loader refuses it instead.
    int[int] hdr = [3, layerDone, frameSave, histSave, 0, 0, 0, 0, 0, 0,
                     mechConv, Th.nv, int(HzTot*1.e6), Nlayers];
    hdr[4] = int(tSave*1000.);          // ms
    hdr[5] = int(eA); hdr[6] = int(eI); hdr[7] = int(eL);
    hdr[8] = int(eQ); hdr[9] = int(eC);
    {{
        ofstream f(ckptFile, binary);
        f << hdr;
        f << Temp[]; f << Temp0[]; f << Tpeak[]; f << psiActive[];
        f << dTdt[]; f << TdotCool[]; f << dt85Track[]; f << tEnter800[];
        f << nReheat[];{m_fields}
        f << eP;
        f << mechOrderCur;
    }}
    return 1;
}}

// Returns 1 if a compatible checkpoint was loaded (fields restored).
func int loadCheckpoint(){{
    if(doResume == 0) return 0;
    // NOTE: constructing an ifstream on a missing file is FATAL in FreeFEM (it
    // aborts before .good() can be consulted), so existence must be tested
    // first with system("test -f ...") -- the same guard the toolpath reader uses.
    if(system("test -f '" + ckptFile + "'") != 0) return 0;
    int ok = 0;
    {{
        ifstream g(ckptFile, binary);
        if(g.good()){{
            int[int] hdr(14);
            g >> hdr;
            // Version + node-count guard: a checkpoint from a DIFFERENT MESH must
            // not be applied, since the dof ordering would be meaningless. The
            // layer count is deliberately NOT part of the guard -- a checkpoint
            // written by a shorter run is exactly the normal case when resuming a
            // longer build, and the saved "layer done" counter drives how much of
            // the deposition loop is skipped.
            if((hdr[0] == 1 || (hdr[0] >= 2 && hdr[0] <= 3)) && hdr[11] == Th.nv){{
                g >> Temp[]; g >> Temp0[]; g >> Tpeak[]; g >> psiActive[];
                g >> dTdt[]; g >> TdotCool[]; g >> dt85Track[]; g >> tEnter800[];
                g >> nReheat[];{m_load}
                real ePload = 0.;
                if(hdr[0] >= 2) g >> ePload;
                ckptEpresc = ePload;
                int ordLoad = 1;
                if(hdr[0] >= 3) g >> ordLoad;
                if(ordLoad != mechOrderCur){{
                    cout << " [RESTART] REFUSED: checkpoint mechanics order "
                         << ordLoad << " != run order " << mechOrderCur
                         << " (P1 vs P2 dofs do not align). Not resuming."
                         << endl;
                    return 0;
                }}
                // Publish the loop counters AND the energy-audit accumulators so
                // the caller can skip finished work and keep the energy balance
                // continuous across the restart. Without restoring these the
                // resumed run reports EarcJ counting only the post-restart
                // deposition, which makes the energy audit nonsense.
                resumeFromLayer = hdr[1];
                ckptFrame = hdr[2];
                ckptHist = hdr[3];
                ckptTime = real(hdr[4])/1000.;
                EprescJ = ckptEpresc;
                ckptEarc = real(hdr[5]);
                ckptEint = real(hdr[6]);
                ckptEloss = real(hdr[7]);
                ckptEquiet = real(hdr[8]);
                ckptEcap = real(hdr[9]);
                // srcMask is derived from psiActive, so it is recomputed rather
                // than stored; without this a resumed run would source from a
                // stale (zero) mask until the next activation update.
                {self._srcmask_step}
                ok = 1;
                bornMask = max(srcMask, bornFloor);
            }}
        }}
    }}
    return ok;
}}

int resumeOk = loadCheckpoint();
if(resumeOk == 1){{
    cout << " [RESTART] Resumed state from checkpoint: layer " << resumeFromLayer
         << ", t=" << ckptTime << " s, frame " << ckptFrame << endl;
}} else {{
    cout << " [RESTART] No compatible checkpoint found - starting from layer 0" << endl;
}}

"""

    def _probe_setup(self) -> str:
        """Emit probe-point and probe-path definitions plus the CSV writers.

        This is the Abaqus `*OUTPUT, HISTORY` equivalent. Previously the only
        history available was five hard-coded thermocouples at X=Y=75 mm that the
        post-processor sampled itself, with a guessed time axis. Here the SOLVER
        samples the fields it has actually computed and writes them, tagged with
        the true simulated time, so nothing downstream has to guess.

        Emitted:
          - probe_points.csv : t, phase, layer, point, x, y, z, Temp, Tpeak,
                               CoolingRate, Activation
          - probe_paths.csv  : t, phase, layer, path, station, s_arc, x, y, z,
                               Temp, Tpeak, CoolingRate
        The mechanical fields (PEEQ, vonMises) are appended by the mechanics stage
        when it exists, because those fields do not exist in a thermal-only run.
        """
        p = self.p
        pts = list(p.get("probe_points") or [])
        paths = list(p.get("probe_paths") or [])
        n_samples = max(2, int(p.get("path_samples", 40)))
        interval = max(1, int(p.get("probe_interval", 1)))

        if not pts and not paths:
            # Still emit a no-op writeHistory so the call sites in the thermal
            # loops always resolve: FreeFEM requires a function to be defined
            # before it is used, so omitting it is a compile error, not a no-op.
            return """
// (no probe points/paths requested - history output disabled)
int probeInterval = 1;
int nProbePts = 0;
int nProbePaths = 0;
int nPathSamples = 0;
int nPathStations = 0;
// Pst must exist even here: FreeFEM parses the WHOLE script, so the guarded
// `Pst[...]` references below still need the symbol. Probe POINTS alone used to
// take this early return and leave Pst undeclared -> "syntax error ... Pst" ->
// the run never compiled. In FreeFEM++, real[int] declares an array of reals indexed by integers.
real[int] Pst(4);
    Pst = [0.0, 0.0, 0.0, 0.0];

func int writeHistory(real tNow, int phaseCode, int layerNum, int stepCounter){
    return 0;
}
"""

        def fmt(v):
            return f"{float(v):.9g}"

        # --- Build the emitted arrays ---
        pt_lines = []
        for i, q in enumerate(pts):
            if len(q) < 3:
                continue
            pt_lines.append("    {:s}, {:s}, {:s}".format(fmt(q[0]), fmt(q[1]), fmt(q[2])))

        # Paths: resample each polyline at n_samples equal-arc-length stations.
        # All stations go into ONE flat array Pst[s*4+0..3] = (s_arc, x, y, z),
        # with path ip owning stations [ip*n_samples, (ip+1)*n_samples). This
        # keeps the runtime indexing trivial instead of needing per-path strides.
        station_rows = []
        n_paths = 0
        for path in paths:
            pts2 = [(float(path[k]), float(path[k + 1]), float(path[k + 2]))
                    for k in range(0, len(path) - 2, 3)]
            if len(pts2) < 2:
                continue
            seg = []
            total = 0.0
            for k in range(len(pts2) - 1):
                d = math.dist(pts2[k], pts2[k + 1])
                seg.append(d)
                total += d
            if total <= 0.0:
                continue
            for s in range(n_samples):
                target = total * s / float(n_samples - 1)
                acc = 0.0
                for k, d in enumerate(seg):
                    if acc + d >= target or k == len(seg) - 1:
                        frac = 0.0 if d <= 0 else (target - acc) / d
                        frac = min(max(frac, 0.0), 1.0)
                        x = pts2[k][0] + frac * (pts2[k + 1][0] - pts2[k][0])
                        y = pts2[k][1] + frac * (pts2[k + 1][1] - pts2[k][1])
                        z = pts2[k][2] + frac * (pts2[k + 1][2] - pts2[k][2])
                        station_rows.append("    {:s}, {:s}, {:s}, {:s}".format(
                            fmt(target), fmt(x), fmt(y), fmt(z)))
                        break
                    acc += d
            n_paths += 1

        # FreeFEM has no 2-D real arrays: `real[3,2]`, `real[len*4]` and
        # `real[len]` are all syntax errors. Probe coordinates therefore go into
        # flat real[int] arrays sized explicitly, indexed as 3*i+k / 4*slot+k.
        arr_pt = ""
        if pt_lines:
            arr_pt = "real[int] Ppt = [\n{}\n    ];".format(",\n".join(pt_lines))

        # Pst must ALWAYS be declared, even with no probe paths: the path
        # writer below is emitted unconditionally (FreeFEM needs the function to
        # exist at every call site) and indexes Pst. Leaving it undeclared made
        # every run WITHOUT --probe-path fail to COMPILE with
        # "syntax error ... Pst" -- i.e. probe points broke the whole solver.
        arr_st = "real[int] Pst(4);\n    Pst = [0.0, 0.0, 0.0, 0.0];"
        if station_rows:
            # FreeFEM array declaration for real floats is real[int] (real elements indexed by int).
            arr_st = ("real[int] Pst(%d);\n    Pst = [\n%s\n    ];"
                      % (4 * len(station_rows), ",\n".join(station_rows)))

        n_stations = len(station_rows)

        return f"""
// === HISTORY OUTPUT: PROBE POINTS AND PATHS ===
// Sampled by the solver from the fields it actually computed, tagged with the
// true simulated time. Replaces the previous approach of five hard-coded
// thermocouples sampled by the post-processor against a guessed time axis.
//
// CAVEAT worth knowing: a probe is interpolated from the FE solution, so a probe
// placed inside a coarse element reports an element average, not a material-point
// value. With nz_base=1 the whole substrate is a single element, so a probe
// at z=2 mm returns something between the z=0 and z=15 node temperatures. Place
// probes on nodes, or refine the mesh, for point-accurate thermocouple data.
int probeInterval = {interval};
int nProbePts = {len(pt_lines)};
int nProbePaths = {n_paths};
int nPathSamples = {n_samples};
int nPathStations = {n_stations};

{arr_pt}
{arr_st}

if(nProbePts > 0){{
    ofstream hPts(outdir+"probe_points.csv");
    hPts << "t_s,phase,layer,point,x_m,y_m,z_m,Temp_C,Tpeak_C,CoolingRate_C_per_s,Activation"
         << endl;
}}
if(nProbePaths > 0){{
    ofstream hPth(outdir+"probe_paths.csv");
    hPth << "t_s,phase,layer,path,station,s_arc_m,x_m,y_m,z_m,Temp_C,Tpeak_C,CoolingRate_C_per_s"
         << endl;
}}

// Append one history sample for the current state. FreeFEM has no string type,
// so phaseCode is an integer tag: 1 = welding, 2 = inter-pass, 3 = cooling.
// NOTE: FreeFEM functions must return a value (`func void` is a syntax error), so
// this returns the number of CSV rows written.
func int writeHistory(real tNow, int phaseCode, int layerNum, int stepCounter){{
    if(stepCounter % probeInterval != 0) return 0;
    string phaseStr = (phaseCode == 1) ? "WELDING" : ((phaseCode == 2) ? "INTERPASS" : "COOLING");
    int rows = 0;

    if(nProbePts > 0){{
        ofstream hp(outdir+"probe_points.csv", append);
        for(int i=0; i<nProbePts; i++){{
            real qx = min(max(Ppt[3*i+0], 0.), LxW);
            real qy = min(max(Ppt[3*i+1], 0.), LyBase);
            real qz = min(max(Ppt[3*i+2], 0.), HzTot);
            if(abs(qx-Ppt[3*i+0]) > 1.e-9 || abs(qy-Ppt[3*i+1]) > 1.e-9
               || abs(qz-Ppt[3*i+2]) > 1.e-9)
                cout << " [WARN] Probe " << i << " requested at ("
                     << Ppt[3*i+0]*1000. << "," << Ppt[3*i+1]*1000. << ","
                     << Ppt[3*i+2]*1000. << ") mm lies OUTSIDE the domain; "
                     << "clamped to (" << qx*1000. << "," << qy*1000. << ","
                     << qz*1000. << ") mm. The history column reports the "
                     << "LOCATION ACTUALLY SAMPLED, not the one requested."
                     << endl;
            hp << tNow << "," << phaseStr << "," << layerNum << "," << i << ","
               << qx << "," << qy << "," << qz << ","
               << Temp(qx,qy,qz) << "," << Tpeak(qx,qy,qz) << ","
               << dTdt(qx,qy,qz) << "," << psiActive(qx,qy,qz) << endl;
            rows++;
        }}
    }}

    if(nProbePaths > 0 && nPathStations > 0){{
        ofstream hf(outdir+"probe_paths.csv", append);
        for(int ip=0; ip<nProbePaths; ip++)
            for(int is=0; is<nPathSamples; is++){{
                int slot = ip*nPathSamples + is;
                if(slot < 0 || slot >= nPathStations) continue;
                real sT = Pst[4*slot+0];
                real qx = min(max(Pst[4*slot+1], 0.), LxW);
                real qy = min(max(Pst[4*slot+2], 0.), LyBase);
                real qz = min(max(Pst[4*slot+3], 0.), HzTot);
                if(abs(qx-Pst[4*slot+1]) > 1.e-9 || abs(qy-Pst[4*slot+2]) > 1.e-9
                   || abs(qz-Pst[4*slot+3]) > 1.e-9)
                    cout << " [WARN] Probe path " << ip << " station " << is
                         << " requested outside the domain; clamped to ("
                         << qx*1000. << "," << qy*1000. << "," << qz*1000.
                         << ") mm." << endl;
                hf << tNow << "," << phaseStr << "," << layerNum << "," << ip << ","
                   << is << "," << sT << "," << qx << "," << qy << "," << qz << ","
                   << Temp(qx,qy,qz) << "," << Tpeak(qx,qy,qz)
                   << "," << dTdt(qx,qy,qz) << endl;
                rows++;
            }}
    }}
    return rows;
}}
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

// Microstructure phase kinetics dispatch
SDAS = 0.; dt85 = 0.;
fFerrite = 0.; fBainite = 0.; fMartensite = 0.; fPearlite = 0.; HVmap = 0.;
for(int iv=0; iv<Th.nv; iv++){{
    real Tp = Tpeak[][iv];
    real zv = Th(iv).z;
    real crEff = max(TdotCool[][iv], 0.5);
    real dtC = (dt85Track[][iv] > 0.05) ? dt85Track[][iv] : ((800.0 - 500.0) / crEff);
    dt85[][iv] = dtC;

    if(Tp > TsolMat(zv) && psiActive[][iv] > 0.5){{
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

    // 1. Substrate Base Indents (Proportional Depth Stations across HzBase)
    for(int is=0; is<3; is++){{
        real zTest = (is + 1.0) * HzBase / 4.0;
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
        # Lexer guard: FreeFEM 4.15 aborts (bare lex.cpp assertion) when
        # 100 '//'-opening lines pass with no intervening real code -- blank
        # lines do NOT reset the count, only code tokens do (measured: 99
        # parses, 100 aborts; 60+50blank+60 aborts; a 120-line /* */ span
        # parses as one unit). The header therefore carries its parameter spec
        # as a /* */ block. This fails the BUILD loudly instead of shipping a
        # script that dies on its first lines. Ceiling 90 for margin.
        _run = 0
        for _ln in script.split("\n"):
            _s = _ln.strip()
            if _s.startswith("//"):
                _run += 1
            elif _s == "" or _s.startswith("/*"):
                pass
            else:
                _run = 0
            if _run >= 90:
                raise ValueError(
                    "emitted script has %d '//' lines with no intervening "
                    "code (FreeFEM lexer limit is <100): wrap the span in "
                    "/* */ or hoist code above it" % _run)
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
