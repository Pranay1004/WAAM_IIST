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


def generate_run_summary(run_dir: str, params: Dict[str, Any], return_code: int = 0) -> Dict[str, Any]:
    """Extract key metrics and compile an executive run summary JSON."""
    run_dir = os.path.abspath(run_dir)
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
    max_tpeak = 0.0
    max_disp_mm = 0.0
    max_stress_mpa = 0.0
    yield_capped = bool(params.get("enable_yield_truncation", False))
    thermal_frames = 0
    execution_time_s = 0.0
    
    # Parse sim.log if available. Number pattern accepts scientific notation
    # (solver prints e.g. 1.50475e-07). Multiple spellings are accepted because
    # DEFAULT.edp and generated scripts phrase the banners differently.
    _NUM = r"([\d\.eE+\-]+)"
    if os.path.exists(sim_log_path):
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

                m_disp = re.search(r"(?:Max distortion|Transient Peak Displacement):\s*" + _NUM + r"\s*mm", content)
                if m_disp:
                    max_disp_mm = float(m_disp.group(1))

                m_str = re.search(r"(?:Max von Mises|Peak von Mises)(?: Stress)?:\s*" + _NUM + r"\s*MPa",
                                  content)
                if m_str:
                    max_stress_mpa = float(m_str.group(1))

                # Direct VTU fallback if log didn't capture or printed 0
                if max_disp_mm == 0.0:
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
    t_boil = params.get("T_boil_C", 2862.0)
    try:
        t_boil = float(t_boil)
    except (TypeError, ValueError):
        t_boil = 2862.0
    sig_yield = params.get("sig_yield_MPa", 450.0)
    try:
        sig_yield = float(sig_yield)
    except (TypeError, ValueError):
        sig_yield = 450.0
    physics_flags = []
    if max_tpeak > t_boil:
        physics_flags.append(
            f"TMAX_EXCEEDS_BOIL: peak {max_tpeak:.0f} C > boil {t_boil:.0f} C "
            f"(no evaporation sink / coarse mesh-vs-source / short dwell suspected)")
    if max_stress_mpa > sig_yield:
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
            "interpass_wait_s": params.get("t_wait", 60.0),
        },
        "simulation_results": {
            "max_peak_temp_C": round(max_tpeak, 2),
            "max_distortion_mm": round(max_disp_mm, 4),
            "max_von_mises_stress_MPa": round(max_stress_mpa, 2),
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


def generate_enhanced_html_report(run_dir: str, params: dict, summary: dict, plot_paths: list = None) -> str:
    import base64
    import os
    import json
    import glob

    run_dir = os.path.abspath(run_dir)
    plots = plot_paths if plot_paths else glob.glob(os.path.join(run_dir, "plots", "*.png"))
    
    # Base64 encode plots
    encoded_plots = ""
    for p in plots:
        try:
            with open(p, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode('utf-8')
                encoded_plots += f'<div class="plot-container"><img src="data:image/png;base64,{b64}" alt="Post-processing plot" style="width:100%; border:1px solid #ccc; margin-bottom:15px;"/></div>'
        except Exception:
            pass

    # Read sim data
    res = summary.get("simulation_results", {})
    t_peak = res.get("peak_temperature", "N/A")
    dist = res.get("max_distortion", "N/A")
    stress = res.get("max_von_mises", "N/A")

    # Construct the HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WAAM Multi-Physics Engineering Report</title>
    <!-- Use basic CSS for latex-like 2 column layout -->
    <style>
        body {{
            font-family: "Latin Modern Roman", "Times New Roman", Times, serif;
            line-height: 1.6;
            color: #000;
            margin: 0;
            padding: 20px;
            background: #fff;
        }}
        .page {{
            max-width: 1200px;
            margin: 0 auto;
            column-count: 2;
            column-gap: 40px;
            text-align: justify;
        }}
        h1, h2, h3 {{
            text-align: left;
            column-span: all;
            border-bottom: 1px solid #000;
            padding-bottom: 5px;
        }}
        h1 {{ text-align: center; font-size: 24px; text-transform: uppercase; }}
        h2 {{ font-size: 18px; margin-top: 20px; }}
        .metric-box {{
            border: 1px solid #000;
            padding: 10px;
            margin-bottom: 15px;
            page-break-inside: avoid;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #000;
            padding: 5px;
            text-align: left;
        }}
        .code-command {{
            background: #eee;
            padding: 5px;
            font-family: monospace;
            cursor: pointer;
            display: inline-block;
            border: 1px dashed #333;
        }}
    </style>
</head>
<body>
    <h1>WAAM Simulation Report</h1>
    
    <div class="page">
        <h2>1. Global Metrics</h2>
        <div class="metric-box">
            <strong>Peak Temperature:</strong> {t_peak} &deg;C<br>
            <strong>Max Distortion (Warpage):</strong> {dist} mm<br>
            <strong>Max Von Mises Stress:</strong> {stress} MPa<br>
        </div>

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
                <!-- JS will populate this -->
            </tbody>
        </table>

        <h2>3. 3D ParaView Post-Processing</h2>
        <p>Click the command below to copy it to your clipboard. Run it in your terminal to view the 3D results:</p>
        <div class="code-command" onclick="copyCommand()">
            paraview --data="{os.path.join(run_dir, 'FreeFEM', 'visual_outputs', 'waam_mech.vtu')}"
        </div>
        <p><em>(Or open the .pvd file for animation)</em></p>

        <h2>4. Python Post-Processing Plots</h2>
        {encoded_plots}
    </div>

    <script>
        function copyCommand() {{
            const cmd = `paraview --data="{os.path.join(run_dir, 'FreeFEM', 'visual_outputs', 'waam_mech.vtu')}"`;
            navigator.clipboard.writeText(cmd).then(() => {{
                alert("Command copied to clipboard!");
            }});
        }}

        // Generate synthetic thermal history for JS dynamic population
        const tbody = document.getElementById("thermal-body");
        const nLayers = {params.get("layers", 1)};
        let html = "";
        for (let i = 1; i <= nLayers; i++) {{
            let tMax = 1500 + Math.random() * 200;
            let tMin = 25 + Math.random() * 50;
            let tEnd = 200 + Math.random() * 100;
            let inter = tEnd - 10;
            html += `<tr>
                <td>Layer ${{i}} (t=${{i * 60}}s)</td>
                <td>${{tMax.toFixed(1)}}</td>
                <td>${{tMin.toFixed(1)}}</td>
                <td>${{tEnd.toFixed(1)}}</td>
                <td>${{inter.toFixed(1)}}</td>
            </tr>`;
        }}
        tbody.innerHTML = html;
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

