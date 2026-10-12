"""WAAM Multi-Physics Output Manager & Enhanced Post-Processing.

Provides automated post-simulation outputs:
1. Executive Run Summary JSON (run_summary.json)
2. Automated 2D Publication-Quality Plots (plots/hardness_profile.png, plots/thermal_cycle_summary.png)
3. One-Click ParaView Python State Script (visual_outputs/open_in_paraview.py)
4. Comprehensive Self-Contained HTML Engineering Report (report.html)
"""

import os
import re
import csv
import json
import base64
import datetime
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.multimaterial.alloy_metallurgy import get_alloy_scheme, resolve_indent_phases


def canonical_run_dir(run_dir: str) -> str:
    """Resolve the canonical result tree of a run.

    MPI Phase 1 writes per-rank output trees (rank0/, rank1/, ...) so ranks can
    never corrupt each other's files; rank 0's tree is canonical. A serial run
    has no rank subdirectories. Resolving here -- once -- keeps every
    downstream path correct for both layouts without touching each reader.
    """
    rank0 = os.path.join(run_dir, "rank0")
    if os.path.isdir(rank0):
        return rank0
    return run_dir


def _load_solver_results(run_dir: str) -> dict:
    """Load solver_results.json written by the generated .edp, or {} if absent.

    This is the authoritative structured record of a run. It exists because the
    console banner is human prose: its wording changed once, after which every
    scraped distortion/stress silently became 0.0 while an unconverged run kept
    matching an intermediate line and reported iteration error as a result.
    """
    run_dir = canonical_run_dir(run_dir)
    for cand in (os.path.join(run_dir, "solver_results.json"),
                 os.path.join(run_dir, "FreeFEM", "solver_results.json")):
        try:
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (OSError, ValueError):
            continue
    return {}


def generate_run_summary(run_dir: str, params: Dict[str, Any], return_code: int = 0) -> Dict[str, Any]:
    """Extract key metrics and compile an executive run summary JSON."""
    run_dir = canonical_run_dir(os.path.abspath(run_dir))
    sim_log_path = os.path.join(run_dir, "sim.log")
    
    # Auto-load parameters.json if available in run_dir
    params = dict(params or {})
    params_file = os.path.join(run_dir, "parameters.json")
    if os.path.exists(params_file):
        try:
            with open(params_file, "r", encoding="utf-8") as pf:
                loaded_p = json.load(pf)
                for k, v in loaded_p.items():
                    if k not in params or params[k] in [None, "Unknown", 0.0]:
                        params[k] = v
        except Exception:
            pass

    # Defaults
    # Defaults are None, not 0.0. A structural value that was never determined
    # must not read as "zero distortion / zero stress", which is a different and
    # much more dangerous claim than "not measured".
    max_tpeak = 0.0
    max_disp_mm = None
    max_stress_mpa = None
    # NOTE: "enable_yield_truncation" is an obsolete parameter kept only for
    # backwards compatibility. The generator replaced post-hoc clipping with a
    # true Cauchy stress return mapping, so the flag no longer describes any
    # behaviour. Report it as False rather than echoing a setting that is inert.
    yield_capped = False
    thermal_frames = 0
    execution_time_s = 0.0
    mechanics_converged = None

    # --- AUTHORITATIVE SOURCE: solver_results.json ---
    # The generated .edp writes its own results file. Reading it is the only
    # trustworthy way to get the structural numbers, because the human-readable
    # banner is prose whose wording can change, and because the convergence flag
    # travels with the values: when the equilibrium solve did not converge the
    # file carries JSON null and the flag is false, so an unconverged iteration
    # cannot be reported as a result.
    #
    # The regex fallback below is retained ONLY for runs produced before this
    # file existed (or by a hand-written .edp), and it explicitly refuses to
    # harvest distortion/stress when the log says the solve did not converge.
    solver_results = _load_solver_results(run_dir)
    if solver_results:
        mechanics_converged = solver_results.get("mechanics_converged")
        max_tpeak = float(solver_results.get("max_peak_temp_C") or 0.0) or max_tpeak
        max_disp_mm = solver_results.get("max_disp_mm")
        max_stress_mpa = solver_results.get("peak_von_mises_MPa")
        thermal_frames = int(solver_results.get("thermal_frames") or 0)

    _NUM = r"([\d\.eE+\-]+)"
    if not solver_results and os.path.exists(sim_log_path):
        try:
            with open(sim_log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                m_tp = re.search(r"Max Tpeak:\s*" + _NUM + r"\s*C", content)
                if m_tp:
                    max_tpeak = float(m_tp.group(1))
                else:
                    # Fallback: peak over per-step Tmax=...C lines the scripts print.
                    step_peaks = [float(v) for v in
                                  re.findall(r"Tmax=" + _NUM + r"C", content)]
                    if step_peaks:
                        max_tpeak = max(step_peaks)

                # Did the final equilibrium solve converge? If the banner says it
                # did NOT, do not scrape distortion/stress at all -- those lines
                # then belong to intermediate, unconverged iterates.
                unconverged = bool(re.search(
                    r"Final cold-state residual warpage \(Uz\):\s*NOT REPORTED", content))
                if unconverged:
                    mechanics_converged = False
                else:
                    # Prefer the explicit final-state lines; fall back to the
                    # legacy per-solve banner spelling.
                    m_disp = re.search(
                        r"Final cold-state residual warpage \(Uz\):\s*" + _NUM + r"\s*mm", content)
                    if not m_disp:
                        m_disp = re.search(
                            r"(?:Max distortion|Transient Peak Displacement):\s*"
                            + _NUM + r"\s*mm", content)
                    if m_disp:
                        max_disp_mm = float(m_disp.group(1))

                    m_str = re.search(
                        r"Final cold-state peak von Mises stress:\s*" + _NUM + r"\s*MPa", content)
                    if not m_str:
                        m_str = re.search(
                            r"(?:Max von Mises|Peak von Mises)(?: Stress)?:\s*"
                            + _NUM + r"\s*MPa", content)
                    if m_str:
                        max_stress_mpa = float(m_str.group(1))

                # Direct VTU fallback if the log yielded nothing. Skipped when the
                # solve is known to be unconverged: the VTU holds the same
                # unconverged field, so harvesting it would republish iteration
                # error as a measurement.
                if not max_disp_mm and mechanics_converged is not False:
                    try:
                        import glob
                        import pyvista as pv
                        mech_vtus = glob.glob(os.path.join(run_dir, "visual_outputs", "waam_mech*.vtu"))
                        if mech_vtus:
                            latest_vtu = max(mech_vtus, key=os.path.getmtime)
                            m = pv.read(latest_vtu)
                            if "DispMag" in m.point_data:
                                max_disp_mm = float(m.point_data["DispMag"].max()) * 1000.0
                            elif "Uz" in m.point_data:
                                max_disp_mm = float(abs(m.point_data["Uz"]).max()) * 1000.0
                    except Exception:
                        pass

                m_fr = re.search(r"DONE\.\s*(\d+)\s*thermal frames", content)
                if m_fr:
                    thermal_frames = int(m_fr.group(1))

                m_time = re.search(r"execution\s*" + _NUM + r"\s*s", content)
                if m_time:
                    execution_time_s = float(m_time.group(1))
        except Exception:
            pass

    # Fallback: peak temperature from waam_data.csv (Tpeak_max_C column).
    if max_tpeak <= 0.0:
        try:
            with open(os.path.join(run_dir, "waam_data.csv"), "r",
                      encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                peaks = [float(r.get("Tpeak_max_C", 0) or 0) for r in reader]
                peaks = [v for v in peaks if v > 0]
                if peaks:
                    max_tpeak = max(peaks)
        except Exception:
            pass

    # Hardness statistics and phase fractions from CSV
    hv_min = 0.0
    hv_max = 0.0
    hv_avg = 0.0
    indent_count = 0
    dom_phases = {}
    h_csv_path = os.path.join(run_dir, "hardness_test_locations.csv")
    if os.path.exists(h_csv_path):
        try:
            hvs = []
            mat_dep = params.get("material_dep", params.get("deposit_material", params.get("material", "ER70S-6")))
            mat_base = params.get("material_base", params.get("substrate_material", "ER70S-6"))
            with open(h_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val = float(row.get("Vickers_HV", 0))
                    if val > 0:
                        hvs.append(val)
                    region = row.get("Region", "")
                    layer_name = row.get("Layer_Name", "")
                    is_base = (region == "BaseMetal" or "Substrate" in layer_name)
                    curr_mat = mat_base if is_base else mat_dep
                    p_info = resolve_indent_phases(row, curr_mat)
                    ph = p_info["dominant_phase"]
                    dom_phases[ph] = dom_phases.get(ph, 0) + 1
            if hvs:
                hv_min = round(min(hvs), 1)
                hv_max = round(max(hvs), 1)
                hv_avg = round(sum(hvs) / len(hvs), 1)
                indent_count = len(hvs)
        except Exception:
            pass

    def _clean_mat_name(val, fallback="ER70S-6"):
        if not val or str(val).strip().lower() in ["unknown", "none", "null", ""]:
            return fallback
        return str(val).strip()

    mat_base_final = _clean_mat_name(params.get("material_base", params.get("substrate_material")), "ER70S-6")
    mat_dep_final = _clean_mat_name(params.get("material_dep", params.get("deposit_material", params.get("material"))), "ER70S-6")

    # Automatic physics-sanity flags so unphysical regimes never pass silently.
    # Thresholds are overridable via params (T_boil_C, sig_yield_MPa); the stress
    # flag explicitly notes the framework's raw-uncapped convention.
    # The DEFAULTS come from the material database, not from carbon steel: an
    # aluminium run (Tboil 2519 C, sigma_y ~150 MPa) used to be checked against
    # 2862 C / 450 MPa, which silently missed both regime violations.
    mat_dep_final = _clean_mat_name(params.get("material_dep", params.get("deposit_material", params.get("material"))), "ER70S-6")
    t_boil = _param_float(params, "T_boil_C", None)
    if t_boil is None:
        t_boil = _material_boiling_C(mat_dep_final)
    sig_yield = _param_float(params, "sig_yield_MPa", None)
    if sig_yield is None:
        sig_yield = _material_yield_MPa(mat_dep_final)
    physics_flags = []
    if mechanics_converged is False:
        physics_flags.append(
            "MECHANICS_NOT_CONVERGED: the structural equilibrium solve did not reach "
            "its tolerance, so distortion and residual stress are reported as null. "
            "These values are iteration error, not physics. Lower "
            "fixture_stiffness_ratio_z/xy, raise mech_max_newton_iter, or tighten "
            "mech_newton_tol / mech_newton_tol_force.")
    if max_tpeak > t_boil:
        physics_flags.append(
            f"TMAX_EXCEEDS_BOIL: peak {max_tpeak:.0f} C > boil {t_boil:.0f} C "
            f"(no evaporation sink / coarse mesh-vs-source / short dwell suspected)")
    if max_stress_mpa is not None and max_stress_mpa > sig_yield:
        physics_flags.append(
            f"STRESS_EXCEEDS_YIELD: {max_stress_mpa:.0f} MPa > {sig_yield:.0f} MPa "
            f"(raw uncapped elastic output, not a capped strength prediction)")

    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "COMPLETED" if return_code == 0 else f"ERROR_EXIT_{return_code}",
        "return_code": return_code,
        "execution_time_s": round(execution_time_s, 2),
        "run_directory": run_dir,
        "visual_outputs_dir": os.path.join(run_dir, "visual_outputs"),
        "process_parameters": {
            "substrate_material": mat_base_final,
            "deposit_material": mat_dep_final,
            "arc_current_A": params.get("I_weld", 0.0),
            "arc_voltage_V": params.get("U_weld", 0.0),
            "arc_efficiency": params.get("eta_arc", 0.8),
            "travel_speed_mm_s": round(params.get("v_scan", 0.01) * 1000.0, 2),
            "gross_power_kW": round(params.get("power_gross_W", 0.0) / 1000.0, 3),
            "net_power_kW": round(params.get("power_net_W", 0.0) / 1000.0, 3),
            "net_heat_input_J_per_mm": params.get("heat_input_net_J_per_mm", 0.0),
            "gross_heat_input_J_per_mm": params.get("heat_input_gross_J_per_mm", 0.0),
            "layers_count": params.get("n_layers", 1),
            "layer_height_mm": round(params.get("h_bead", 0.003) * 1000.0, 2),
            "bead_width_mm": round(params.get("ty_dep", 0.008) * 1000.0, 2),
            "wire_diameter_mm": round(float(params.get("wire_diameter_mm", (params.get("d_wire", 1.2e-3) * 1000.0 if params.get("d_wire", 1.2e-3) < 0.05 else params.get("d_wire", 1.2)))), 2),
            "wire_feed_speed_m_min": round(float(params.get("v_wire", params.get("wire_feed_speed", 5.0))), 2),
            "wire_feed_speed_m_s": round(float(params.get("v_wire", params.get("wire_feed_speed", 5.0))) / 60.0, 4),
            "wire_mass_deposition_rate_kg_h": round(float(params.get("wire_mass_rate_kg_h", 0.0)), 2),
            "theoretical_bead_area_mm2": round(float(params.get("a_bead_theo_mm2", 0.0)), 2),
            "interpass_wait_s": params.get("t_wait", 60.0),
        },
        "simulation_results": {
            "max_peak_temp_C": round(max_tpeak, 2),
            # None, never 0.0, when the equilibrium solve did not converge: a zero
            # would read as "no distortion", which is the opposite of the truth.
            "max_distortion_mm": (None if max_disp_mm is None else round(max_disp_mm, 4)),
            "max_von_mises_stress_MPa": (None if max_stress_mpa is None
                                         else round(max_stress_mpa, 2)),
            "mechanics_converged": mechanics_converged,
            "yield_stress_capped": yield_capped,
            "physics_flags": physics_flags,
            "vickers_hardness_min_HV": hv_min,
            "vickers_hardness_max_HV": hv_max,
            "vickers_hardness_mean_HV": hv_avg,
            "discrete_test_points": indent_count,
            "dominant_phases_detected": dom_phases,
            "thermal_frames_saved": thermal_frames
        },
        "artifacts": {
            "edp_script": os.path.join(run_dir, "waam_simulation.edp"),
            "terminal_log": os.path.join(run_dir, "sim.log"),
            "summary_json": os.path.join(run_dir, "run_summary.json"),
            "report_html": os.path.join(run_dir, "report.html"),
            "waam_data_csv": os.path.join(run_dir, "waam_data.csv"),
            "hardness_test_csv": os.path.join(run_dir, "hardness_test_locations.csv"),
            "hardness_profile_csv": os.path.join(run_dir, "hardness_profile_z.csv"),
            "paraview_pvd": os.path.join(run_dir, "visual_outputs", "waam.pvd"),
            "paraview_micro_vtu": os.path.join(run_dir, "visual_outputs", "waam_micro.vtu"),
            "paraview_mech_vtu": os.path.join(run_dir, "visual_outputs", "waam_mech.vtu"),
        }
    }

    summary_path = os.path.join(run_dir, "run_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    return summary


def generate_post_plots(run_dir: str) -> List[str]:
    """Generate 2D publication-quality analysis plots in <run_dir>/plots/."""
    try:
        try:
            from postprocess.plot_generator import generate_all_plots
        except ImportError:
            import sys
            fw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if fw_dir not in sys.path:
                sys.path.insert(0, fw_dir)
            from postprocess.plot_generator import generate_all_plots
        return generate_all_plots(run_dir)
    except Exception as ex:
        print(f"[OutputManager] Plot generation delegator failed: {ex}")
        return []



def generate_paraview_loader_script(run_dir: str) -> str:
    """Generate a 1-click Python script for loading the results in ParaView."""
    run_dir = os.path.abspath(run_dir)
    vis_dir = os.path.join(run_dir, "visual_outputs")
    os.makedirs(vis_dir, exist_ok=True)
    
    script_path = os.path.join(vis_dir, "open_in_paraview.py")
    script_content = f"""# ==============================================================================
# ParaView Automation Script for WAAM Simulation Results
# Run from terminal: paraview --script=open_in_paraview.py
# Or run from inside ParaView: Tools -> Python Shell -> Run Script
# ==============================================================================
import os
from paraview.simple import *

vis_dir = os.path.dirname(os.path.abspath(__file__))
micro_path = os.path.join(vis_dir, "waam_micro.vtu")
mech_path  = os.path.join(vis_dir, "waam_mech.vtu")
pvd_path   = os.path.join(vis_dir, "waam.pvd")

view = GetActiveViewOrCreate('RenderView')

# 1. Load Microstructure & Hardness Field
if os.path.exists(micro_path):
    micro_reader = XMLUnstructuredGridReader(FileName=[micro_path])
    micro_display = Show(micro_reader, view)
    ColorBy(micro_display, ('POINTS', 'VickersHV'))
    micro_display.SetScalarBarVisibility(view, True)
    print("[OK] Loaded waam_micro.vtu with VickersHV color mapping")

# 2. Load Mechanical Distortion & Stress
if os.path.exists(mech_path):
    mech_reader = XMLUnstructuredGridReader(FileName=[mech_path])
    mech_display = Show(mech_reader, view)
    mech_display.Visibility = 0  # Hidden by default, toggle in pipeline
    print("[OK] Loaded waam_mech.vtu (Residual stress & distortion)")

# 3. Load Transient Thermal History
if os.path.exists(pvd_path):
    pvd_reader = PVDReader(FileName=pvd_path)
    pvd_display = Show(pvd_reader, view)
    pvd_display.Visibility = 0  # Hidden by default, toggle in pipeline
    print("[OK] Loaded waam.pvd (Transient thermal cycle collection)")

ResetCamera()
Render()
"""
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
    except Exception:
        pass
    return script_path


def _param_float(params: dict, key: str, default=None):
    """Read a numeric parameter, ignoring nulls and non-numeric strings."""
    try:
        v = params.get(key, default)
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _material_db_entry(mat_name: str) -> dict:
    """Look up an alloy in core/multimaterial/material_database.json."""
    try:
        dbp = Path(__file__).resolve().parent.parent / "multimaterial" / "material_database.json"
        with open(dbp) as fh:
            db = json.load(fh)
        alloys = db.get("alloys", db)
    except Exception:
        return {}
    if not isinstance(alloys, dict):
        return {}
    if mat_name in alloys:
        return alloys[mat_name] or {}
    up = str(mat_name).upper()
    for k, v in alloys.items():
        if str(k).upper() == up:
            return v or {}
    return {}


def _material_boiling_C(mat_name: str, default: float = 2862.0) -> float:
    """Boiling point [C] of the deposit alloy, or the steel default."""
    val = _param_float(_material_db_entry(mat_name), "boiling_point", None)
    if val and val > 200.0:
        return val
    try:
        from core.multimaterial.alloy_metallurgy import get_alloy_category
        cat = get_alloy_category(mat_name)
    except Exception:
        cat = ""
    return {"Titanium Alloy": 3287.0,
            "Nickel Superalloy": 2913.0,
            "Aluminium Alloy": 2519.0}.get(cat, default)


def _material_yield_MPa(mat_name: str, default: float = 450.0) -> float:
    """Room-temperature yield strength [MPa] of the deposit alloy."""
    entry = _material_db_entry(mat_name)
    ys = entry.get("yield_strength") or {}
    sig = ys.get("sigma_y") if isinstance(ys, dict) else None
    if sig:
        try:
            val = float(sig[0])
            # material_database stores strengths in Pa; the callers of this
            # helper want MPa.
            return val / 1.0e6 if val > 1.0e5 else val
        except (TypeError, ValueError, IndexError):
            pass
    try:
        from core.multimaterial.alloy_metallurgy import get_alloy_category
        cat = get_alloy_category(mat_name)
    except Exception:
        cat = ""
    return {"Aluminium Alloy": 150.0, "Titanium Alloy": 880.0}.get(cat, default)


def _build_layer_history_rows(run_dir: str) -> tuple:
    """Build the per-layer thermal-history table from the solver's waam_data.csv.

    Returns (html_rows, note). The CSV is written by the generated .edp and holds
    time_s,layer,phase,Tmax_C,Tpeak_max_C per saved frame, so every number below is
    measured. If the CSV is missing or unparsable we emit an explicit placeholder
    rather than fabricating a plausible-looking table.
    """
    import os as _os

    candidates = [
        _os.path.join(run_dir, "waam_data.csv"),
        _os.path.join(run_dir, "FreeFEM", "waam_data.csv"),
    ]
    csv_path = next((c for c in candidates if _os.path.exists(c)), None)
    if csv_path is None:
        return ("<tr><td colspan='5'>No thermal history available "
                "(waam_data.csv not found).</td></tr>",
                "Per-layer thermal history requires waam_data.csv from the solver run.")

    # Column positions come from the HEADER ROW, not from fixed indexes: a
    # reordered or extended waam_data.csv used to shift every column (the old
    # code hard-indexed 5 -> Tmax and 7 -> HV, so any header change silently
    # mislabelled the data).
    agg = {}
    header_note = ""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = None
            col = {}
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if not any(parts):
                    continue
                if header is None:
                    header = [h.strip() for h in parts]
                    # Real solver header (see EDPBuilder.build):
                    # time_s,layer,phase,xTorch_mm,zLayer_mm,Tmax_C,
                    # Tpeak_max_C,HV_layer
                    want = {"time_s": "time_s", "layer": "layer",
                            "Tmax_C": "Tmax_C", "HV": "HV_layer"}
                    for logical, name in want.items():
                        if name in header:
                            col[logical] = header.index(name)
                    if "time_s" not in col or "layer" not in col:
                        header_note = ("waam_data.csv header lacks time_s/layer; "
                                       "history table omitted.")
                        break
                    continue
                try:
                    t_s = float(parts[col["time_s"]])
                    layer = int(float(parts[col["layer"]]))
                    tmax = (float(parts[col["Tmax_C"]])
                            if "Tmax_C" in col else float("nan"))
                except (ValueError, IndexError):
                    continue
                import math as _math
                if _math.isnan(tmax):
                    continue
                rec = agg.setdefault(layer, {"t0": t_s, "tmax": tmax, "tend": t_s,
                                             "tmin": tmax, "hv": None})
                rec["tmax"] = max(rec["tmax"], tmax)
                rec["tmin"] = min(rec["tmin"], tmax)
                rec["tend"] = t_s
                if "HV" in col:
                    try:
                        rec["hv"] = float(parts[col["HV"]])
                    except (ValueError, IndexError):
                        pass
    except OSError:
        return ("<tr><td colspan='5'>Could not read waam_data.csv.</td></tr>",
                "Per-layer thermal history requires waam_data.csv from the solver run.")
    if header_note and not agg:
        return ("<tr><td colspan='5'>waam_data.csv header lacks time_s/layer.</td></tr>",
                header_note)

    if not agg:
        return ("<tr><td colspan='5'>waam_data.csv contained no usable rows.</td></tr>",
                "Per-layer thermal history requires waam_data.csv from the solver run.")

    rows = []
    for layer in sorted(agg):
        r = agg[layer]
        # "Interpass" here is the coolest Tmax sampled while THIS layer was the
        # active one -- the closest the CSV gets to the state the next pass
        # starts from. It is NOT a dedicated interpass measurement (the solver
        # writes those rows with phase=INTERPASS into the same file).
        inter = f"{r['tmin']:.1f}"
        rows.append(
            "<tr>"
            f"<td>Layer {layer}</td>"
            f"<td>{r['tmax']:.1f}</td>"
            f"<td>{r['tmin']:.1f}</td>"
            f"<td>{r['tend']:.1f}</td>"
            f"<td>{inter}</td>"
            "</tr>"
        )
    note = ("T_max / T_min / T_end are read directly from the solver's "
            f"waam_data.csv ({_os.path.basename(csv_path)}).")
    return ("\n".join(rows), note)


def generate_enhanced_html_report(run_dir: str, params: dict, summary: dict, plot_paths: list = None) -> str:
    import base64
    import os
    import json
    import glob
    import datetime

    run_dir = os.path.abspath(run_dir)
    plots = plot_paths if plot_paths else glob.glob(os.path.join(run_dir, "plots", "*.png"))
    if not plots:
        plots = glob.glob(os.path.join(run_dir, "*.png"))
    
    # Base64 encode plots into structured cards
    plot_cards = []
    for p in sorted(plots):
        try:
            bname = os.path.basename(p)
            clean_title = bname.replace(".png", "").replace("_", " ").title()
            with open(p, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode('utf-8')
            plot_cards.append(f'''
            <div class="plot-card">
                <div class="plot-card-header">{clean_title}</div>
                <div class="plot-img-wrap">
                    <img src="data:image/png;base64,{b64}" alt="{clean_title}" loading="lazy"/>
                </div>
                <div class="plot-caption">{bname}</div>
            </div>''')
        except Exception:
            pass
    encoded_plots = "\n".join(plot_cards) if plot_cards else '<p style="color:#64748b; font-style:italic;">No 2D post-processing plots generated yet.</p>'

    # Read sim data
    res = summary.get("simulation_results", {})
    converged = res.get("mechanics_converged", None)
    t_peak = res.get("max_peak_temp_C", None)
    dist = res.get("max_distortion_mm", None)
    stress = res.get("max_von_mises_stress_MPa", None)

    def _fmt(v, unit=""):
        if v is None:
            return ("not reported - mechanical equilibrium did not converge"
                    if converged is False else "N/A")
        return f"{v}{unit}"

    t_peak_s = _fmt(t_peak)
    dist_s = _fmt(dist)
    stress_s = _fmt(stress)

    conv_note = ""
    if converged is False:
        conv_note = (
            '<div class="alert-box alert-warning">'
            '<strong>Warning:</strong> the structural equilibrium solve did not converge, '
            'so distortion and residual stress are withheld rather than reported. '
            'See the physics flags in run_summary.json.</div>')

    # Real per-layer thermal history from waam_data.csv (written by the solver).
    per_layer_html, per_layer_note = _build_layer_history_rows(run_dir)

    # Metadata extraction
    case_name = os.path.basename(os.path.dirname(run_dir)) if os.path.basename(run_dir) in ("FreeFEM", "plots") else os.path.basename(run_dir)
    mat_dep = params.get("material_dep", params.get("material", "SS308L"))
    mat_base = params.get("material_base", "SS316L")
    n_layers_val = params.get("n_layers", "N/A")
    timestamp_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Construct robust, non-breaking print-safe HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WAAM Multi-Physics Engineering Report — {case_name}</title>
    <style>
        @page {{
            size: A4 portrait;
            margin: 15mm 15mm 15mm 15mm;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.5;
            color: #0f172a;
            background: #f1f5f9;
            padding: 24px;
            font-size: 13px;
        }}
        .report-container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #ffffff;
            padding: 36px 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border-radius: 8px;
        }}
        .report-header {{
            border-bottom: 2px solid #0f172a;
            padding-bottom: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .report-header-text h1 {{
            font-size: 22px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #0f172a;
            margin-bottom: 4px;
        }}
        .report-subtitle {{
            font-size: 12px;
            color: #475569;
            margin-top: 4px;
        }}
        .report-meta-badges {{
            display: flex;
            gap: 8px;
            margin-top: 8px;
            flex-wrap: wrap;
        }}
        .meta-badge {{
            background: #e2e8f0;
            color: #334155;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .print-btn {{
            background: #2563eb;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(37,99,235,0.2);
            transition: background 0.2s;
        }}
        .print-btn:hover {{
            background: #1d4ed8;
        }}
        h2 {{
            font-size: 15px;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
            margin-top: 24px;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 6px;
            padding: 14px 16px;
        }}
        .metric-card.alert {{
            border-left-color: #ef4444;
        }}
        .metric-card .label {{
            font-size: 11px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .metric-card .val {{
            font-size: 20px;
            font-weight: 800;
            color: #0f172a;
        }}
        .alert-box {{
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 12.5px;
        }}
        .alert-warning {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
            margin-bottom: 12px;
            font-size: 12px;
        }}
        thead {{
            display: table-header-group;
        }}
        th {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 700;
            padding: 8px 12px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}
        td {{
            padding: 6px 12px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        tbody tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .table-note {{
            font-size: 11px;
            color: #64748b;
            font-style: italic;
            margin-top: 4px;
            margin-bottom: 20px;
        }}
        .command-card {{
            background: #0f172a;
            color: #38bdf8;
            padding: 12px 16px;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            word-break: break-all;
        }}
        .copy-pill {{
            background: #38bdf8;
            color: #0f172a;
            border: none;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            margin-left: 12px;
            white-space: nowrap;
        }}
        .copy-pill:hover {{
            background: #7dd3fc;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-top: 14px;
        }}
        .plot-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            display: flex;
            flex-direction: column;
        }}
        .plot-card-header {{
            background: #f8fafc;
            padding: 8px 12px;
            font-size: 11.5px;
            font-weight: 700;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }}
        .plot-img-wrap {{
            padding: 8px;
            background: #ffffff;
            text-align: center;
        }}
        .plot-img-wrap img {{
            width: 100%;
            height: auto;
            max-height: 280px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }}
        .plot-caption {{
            font-size: 10px;
            color: #64748b;
            padding: 4px 12px 8px;
            font-style: italic;
            border-top: 1px solid #f1f5f9;
            background: #ffffff;
        }}
        /* --- Strict print & pagination rules --- */
        @media print {{
            body {{
                background: #ffffff !important;
                padding: 0 !important;
                font-size: 11pt !important;
                color: #000000 !important;
            }}
            .report-container {{
                max-width: 100% !important;
                padding: 0 !important;
                box-shadow: none !important;
                border-radius: 0 !important;
            }}
            .no-print {{
                display: none !important;
            }}
            .print-break-before {{
                break-before: page !important;
                page-break-before: always !important;
            }}
            .metric-card, .metric-box, table, tr, .plot-card, .command-card {{
                break-inside: avoid !important;
                page-break-inside: avoid !important;
            }}
            h1, h2, h3 {{
                break-after: avoid !important;
                page-break-after: avoid !important;
            }}
            .plots-grid {{
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 12px !important;
            }}
            .plot-card {{
                border: 1px solid #94a3b8 !important;
                box-shadow: none !important;
                margin-bottom: 10px !important;
            }}
            .plot-img-wrap img {{
                max-height: 240px !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <div class="report-header-text">
                <h1>WAAM Simulation Engineering Report</h1>
                <div class="report-subtitle">Case: <strong>{case_name}</strong> | Generated on {timestamp_str}</div>
                <div class="report-meta-badges">
                    <span class="meta-badge">Deposit: {mat_dep}</span>
                    <span class="meta-badge">Substrate: {mat_base}</span>
                    <span class="meta-badge">Layers: {n_layers_val}</span>
                    <span class="meta-badge">Solver: FreeFEM++</span>
                </div>
            </div>
            <button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save PDF</button>
        </div>

        <h2>1. Global Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Peak Temperature</div>
                <div class="val"><strong>Peak Temperature:</strong> {t_peak_s} &deg;C</div>
            </div>
            <div class="metric-card">
                <div class="label">Max Distortion (Warpage)</div>
                <div class="val"><strong>Max Distortion (Warpage):</strong> {dist_s} mm</div>
            </div>
            <div class="metric-card">
                <div class="label">Peak Von Mises Stress</div>
                <div class="val"><strong>Max Von Mises Stress:</strong> {stress_s} MPa</div>
            </div>
        </div>
        {conv_note}

        <h2>2. Thermal History (Layer Profile)</h2>
        <table id="thermal-table">
            <thead>
                <tr>
                    <th>Time / Layer</th>
                    <th>T_max (&deg;C)</th>
                    <th>T_min (&deg;C)</th>
                    <th>T_end (&deg;C)</th>
                    <th>Interpass Temp (&deg;C)</th>
                </tr>
            </thead>
            <tbody id="thermal-body">
                {per_layer_html}
            </tbody>
        </table>
        <p class="table-note">{per_layer_note}</p>

        <h2>3. 3D ParaView Post-Processing</h2>
        <p>Run the command below in your terminal to inspect the volumetric multi-physics fields in ParaView:</p>
        <div class="command-card">
            <span id="paraview-cmd">paraview --data="{os.path.join(run_dir, 'visual_outputs', 'waam_mech.vtu')}"</span>
            <button class="copy-pill no-print" onclick="copyCommand()">Copy Command</button>
        </div>

        <h2 class="print-break-before">4. Python Post-Processing Plots</h2>
        <div class="plots-grid">
            {encoded_plots}
        </div>
    </div>

    <script>
        function copyCommand() {{
            const cmd = document.getElementById('paraview-cmd').innerText;
            navigator.clipboard.writeText(cmd).then(() => {{
                alert("ParaView command copied to clipboard!");
            }});
        }}
    </script>
</body>
</html>
'''
    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html

def generate_cfd_html_report(foam_dir: str) -> str:
    """Generate a self-contained CFD engineering report (cfd_report.html).

    Reads case_parameters.json + cfd_history.csv + cfd_history.png from the
    OpenFOAM case folder. Works with partial results (e.g. no time dirs yet):
    missing sections render as 'not available yet' instead of failing.
    """
    foam_dir = os.path.abspath(foam_dir)
    report_path = os.path.join(foam_dir, "cfd_report.html")

    cp_path = os.path.join(foam_dir, "case_parameters.json")
    try:
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
    except Exception:
        cp = {}

    hist_path = os.path.join(foam_dir, "cfd_history.csv")
    rows: List[Dict[str, Any]] = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []

    def _col(name, cast=float, default=None):
        vals = []
        for r in rows:
            try:
                vals.append(cast(r[name]))
            except (KeyError, TypeError, ValueError):
                pass
        return vals

    times = _col("time_s")
    tmaxs = _col("Tmax_K")
    melts = _col("molten_cells")
    last_t = times[-1] if times else None
    last_T = tmaxs[-1] if tmaxs else None
    last_melt = melts[-1] if melts else None

    png_html = ""
    png_path = os.path.join(foam_dir, "cfd_history.png")
    if os.path.exists(png_path):
        try:
            with open(png_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("ascii")
            png_html = ("""
                    <div class="card plot-card">
                      <h3>Run History (Temperature / Melt Evolution)</h3>
                      <img src="data:image/png;base64,%s" alt="CFD history" class="report-img"/>
                    </div>""" % b64)
        except Exception:
            pass

    hist_rows = ""
    for r in rows[-50:]:
        hist_rows += ("<tr><td>%.6g</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            float(r.get("time_s", 0) or 0), r.get("Tmax_K", "-"),
            r.get("Tmean_K", "-"), r.get("metal_frac", "-"),
            r.get("molten_cells", "-")))
    if not hist_rows:
        hist_rows = ('<tr><td colspan="5" style="color:#64748b;">No time-step history yet — '
                     'run the solver, then re-run postprocess_cfd.py.</td></tr>')

    notes_html = ""
    for n in cp.get("notes", []) or []:
        notes_html += "<li>%s</li>" % str(n)
    if not notes_html:
        notes_html = "<li>(none)</li>"

    dom = cp.get("domain_m", {}) or {}
    mesh = cp.get("mesh_cells", {}) or {}
    q = float(cp.get("Q_arc_W", 0.0) or 0.0)
    v = float(cp.get("v_scan_m_s", 0.0) or 0.0)
    h_net = (q / (v * 1000.0)) if v > 0 else 0.0
    material = cp.get("material", "ER70S-6")

    def _fmt(x, default="-"):
        try:
            return "%.6g" % float(x)
        except (TypeError, ValueError):
            return default

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WAAM CFD Melt-Pool Report — beamWeldFoam</title>
<style>
  :root {{
    --bg: #f8fafc; --card-bg: #ffffff; --card-border: #e2e8f0;
    --text-main: #0f172a; --text-muted: #64748b; --accent: #0284c7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg); color: var(--text-main); line-height: 1.55; padding: 24px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  .header {{
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 50%, #e0f2fe 100%);
    border: 1px solid #bae6fd; border-radius: 14px; padding: 26px 30px; margin-bottom: 24px;
  }}
  .header h1 {{ font-size: 24px; font-weight: 800; color: #0c4a6e; }}
  .header p {{ color: #0369a1; font-size: 13px; margin-top: 4px; }}
  .header p code {{ background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
  .grid-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .stat-card {{ background: #fff; border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; }}
  .stat-card .label {{ font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }}
  .stat-card .value {{ font-size: 22px; font-weight: 800; color: var(--accent); }}
  .stat-card .sub {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; }}
  .card {{ background: #fff; border: 1px solid var(--card-border); border-radius: 12px; padding: 22px; margin-bottom: 24px; }}
  .card h2 {{ font-size: 16px; font-weight: 700; color: #0369a1; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0; }}
  .card h3 {{ font-size: 14px; margin-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid #f1f5f9; }}
  th {{ background: #f8fafc; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
  td code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, Menlo, monospace; color: #0369a1; font-size: 12px; }}
  .plot-card {{ text-align: center; }}
  .report-img {{ max-width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 8px; }}
  ul.notes {{ margin-left: 18px; font-size: 13px; color: #334155; }}
  ul.notes li {{ margin-bottom: 4px; }}
  footer {{ text-align: center; font-size: 12px; color: #64748b; margin-top: 32px; padding-top: 20px; border-top: 1px solid #e2e8f0; }}
  @page {{
    size: A4 portrait;
    margin: 15mm;
  }}
  @media print {{
    body {{
      background: #ffffff !important;
      padding: 0 !important;
      color: #000000 !important;
      font-size: 11pt !important;
    }}
    .container {{
      max-width: 100% !important;
      width: 100% !important;
      margin: 0 !important;
    }}
    .header, .card, .stat-card, .plot-card, tr {{
      break-inside: avoid !important;
      page-break-inside: avoid !important;
    }}
    .card h2, .card h3 {{
      break-after: avoid !important;
      page-break-after: avoid !important;
    }}
    table {{
      page-break-inside: auto !important;
    }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>WAAM CFD Melt-Pool Report (beamWeldFoam / OpenFOAM 2412)</h1>
    <p>Case: <code>{foam}</code> &bull; Material: <strong>{mat}</strong> &bull; Generated: {ts}</p>
  </div>

  <div class="grid-stats">
    <div class="stat-card"><div class="label">Beam Power</div><div class="value">{qkw:.3f} kW</div><div class="sub">{qw:.0f} W net (I&middot;U&middot;eta)</div></div>
    <div class="stat-card"><div class="label">Travel Speed</div><div class="value">{vms:.1f} mm/s</div><div class="sub">along +Z, H_net = {hnet:.1f} J/mm</div></div>
    <div class="stat-card"><div class="label">Mesh Cells</div><div class="value">{cells}</div><div class="sub">{nx} x {ny} x {nz} at {cellmm} mm</div></div>
    <div class="stat-card"><div class="label">Simulated Time</div><div class="value">{et:.5g} s</div><div class="sub">endTime, deltaT = {dt} s</div></div>
    <div class="stat-card"><div class="label">Peak Pool Temp</div><div class="value">{pt}</div><div class="sub">{pts}</div></div>
    <div class="stat-card"><div class="label">Molten Cells</div><div class="value">{mc}</div><div class="sub">alpha &gt; 0.5 and hot, last step</div></div>
  </div>

  <div class="card">
    <h2>Phase Legend (applies to every field below)</h2>
    <table><tbody>
      <tr><td><code>alpha.phase1</code></td><td>{mat} (METAL) volume fraction: 1 = metal, 0 = atmosphere gas</td></tr>
      <tr><td><code>phase2</code></td><td>Air (alpha.phase2 = 1 &minus; alpha.phase1)</td></tr>
      <tr><td><code>Temperature</code></td><td>Mixture temperature [K]</td></tr>
      <tr><td><code>U</code></td><td>Mixture velocity [m/s]</td></tr>
      <tr><td><code>p_rgh</code></td><td>Buoyancy pressure (p &minus; rho&middot;g&middot;h)/rho [m&sup2;/s&sup2;], not Pascals</td></tr>
    </tbody></table>
  </div>

  <div class="card">
    <h2>Case Setup (from case_parameters.json)</h2>
    <table><tbody>
      <tr><td>Local window (X transverse / Y vertical / Z travel)</td><td>{dx} x {dy} x {dz} mm</td></tr>
      <tr><td>Beam radius HS_a (= Goldak bG/2)</td><td>{hsa} m</td></tr>
      <tr><td>Focus plane focY (metal surface)</td><td>{focy} m</td></tr>
      <tr><td>Power ramp (decay after tshift)</td><td>{ramp} W/s, tshift = {tshift} s</td></tr>
      <tr><td>Deposition cutoff</td><td>{cut}</td></tr>
    </tbody></table>
  </div>

  {plots}

  <div class="card">
    <h2>Time-Step History (cfd_history.csv, last up to 50 rows)</h2>
    <table><thead><tr><th>time [s]</th><th>Tmax [K]</th><th>Tmean [K]</th><th>metal frac</th><th>molten cells</th></tr></thead>
    <tbody>{hrows}</tbody></table>
  </div>

  <div class="card">
    <h2>Generation Notes</h2>
    <ul class="notes">{notes}</ul>
  </div>

  <div class="card">
    <h2>How To Inspect In 3D</h2>
    <table><tbody>
      <tr><td>ParaView (GUI)</td><td>Open <code>case.foam</code> in this folder</td></tr>
      <tr><td>ParaView (script)</td><td><code>paraview --script=open_in_paraview_cfd.py</code></td></tr>
      <tr><td>Re-run analysis</td><td><code>python3 postprocess_cfd.py</code> (writes cfd_history.csv/png)</td></tr>
    </tbody></table>
  </div>
</div>
<footer>WAAM Multi-Physics Framework — Auto-Generated CFD Dossier &bull; beamWeldFoam / OpenFOAM 2412</footer>
</body>
</html>""".format(
        foam=foam_dir, mat=material,
        ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        qkw=(q / 1000.0), qw=q, vms=(v * 1000.0), hnet=h_net,
        cells=("{:,}".format(int(mesh.get("total", 0))) if mesh.get("total") else "-"),
        nx=mesh.get("nx", "-"), ny=mesh.get("ny", "-"), nz=mesh.get("nz", "-"),
        cellmm=_fmt(cp.get("cell_mm"), "-"),
        et=float(cp.get("endTime_s", 0.0) or 0.0),
        dt=_fmt(cp.get("deltaT_s"), "-"),
        pt=("-" if last_T is None else "%.0f K" % last_T),
        pts=(("at t = %.6g s" % last_t) if last_t is not None else "no time dirs yet"),
        mc=("-" if last_melt is None else "%d" % int(last_melt)),
        dx=_fmt((dom.get("x_transverse", 0) or 0) * 1000.0 if dom.get("x_transverse") else 0) if dom.get("x_transverse") else "-",
        dy=_fmt((dom.get("y_total", 0) or 0) * 1000.0 if dom.get("y_total") else 0) if dom.get("y_total") else "-",
        dz=_fmt((dom.get("z_travel", 0) or 0) * 1000.0 if dom.get("z_travel") else 0) if dom.get("z_travel") else "-",
        hsa=_fmt(cp.get("HS_a_m"), "-"), focy=_fmt(cp.get("focY_m"), "-"),
        ramp=_fmt(cp.get("Q_ramp_rate_W_per_s"), "-"),
        tshift=_fmt(cp.get("tshift_s"), "-"),
        cut=_fmt((cp.get("cfd_params_echo", {}) or {}).get("deposition_cutoff"), "-"),
        plots=png_html, hrows=hist_rows, notes=notes_html,
    )
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass
    return report_path

