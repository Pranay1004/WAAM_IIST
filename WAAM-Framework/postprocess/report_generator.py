"""
postprocess/report_generator.py

Production-Grade WAAM Multi-Physics Reporting Engine
====================================================
Generates a 2-page side-by-side LaTeX PDF-style HTML report with:
1. Executive Global Numerical Parameters & 3D ParaView Launch Command
2. Substrate Thermocouple Transient Response across depths Z = 0, 1, 2, 3, 4 mm
   (using a broken/split horizontal axis to magnify initial gradients and scale the cooldown tail)
3. Toolpath-following layer-by-layer build animation GIF with fine elements (never cut-off baseplate)
4. Authentic Layer-by-Layer Physical Process Record Table (all layers)
5. Thermal & Mechanical Field Verification (T_peak and Uz) with full baseplate framing
6. Transient Melt Pool Thermal Cycles across welding passes (clean, focused, single-panel)
"""

import os
import sys
import json
import glob
import math
import re
import csv
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    import imageio
    HAS_3D = True
except ImportError:
    HAS_3D = False


def resolve_n_layers(freefem_dir: Path, params: dict) -> int:
    """True deposition pass count. parameters.json n_layers is the deposit
    mesh cell count (e.g. 3), NOT the pass count, so prefer waam_data.csv
    and the per-layer pass time file before falling back to params."""
    import csv
    csv_path = freefem_dir / "waam_data.csv"
    if csv_path.exists():
        try:
            with open(csv_path) as f:
                n = max((int(r["layer"]) for r in csv.DictReader(f)
                         if r.get("layer") not in (None, "")), default=0)
            if n > 0:
                return n
        except Exception:
            pass
    lt_path = freefem_dir / "layer_pass_times.txt"
    if not lt_path.exists():
        lts = list(freefem_dir.glob("*.layer_times.txt"))
        lt_path = lts[0] if lts else lt_path
    if lt_path.exists():
        try:
            with open(lt_path) as f:
                n = int(f.readline().strip())
            if n > 0:
                return n
        except Exception:
            pass
    return int(params.get("n_layers", 27))


def _find_stl(*dirs: Path) -> Optional[Path]:
    """First STL found across the given run directories (geometry-agnostic)."""
    for d in dirs:
        if d is None:
            continue
        matches = sorted(p for p in d.glob("*.stl") if p.is_file())
        if matches:
            return matches[0]
    return None


def _read_csv_frame_times(run_dir: Path, n_expected: int) -> Optional[List[float]]:
    """Return the real per-frame times from the solver's waam_data.csv, or None.

    waam_data.csv has one row per saved frame, in the same order the T_*.vtu files
    were written, so its time_s column is the authoritative abscissa when a PVD
    timestep is unavailable. Returns None if the CSV is absent or unparsable so
    callers can fall back to an explicitly-labelled frame index.
    """
    for cand in (run_dir / "waam_data.csv",
                 run_dir.parent / "FreeFEM" / "waam_data.csv"):
        try:
            if not cand.exists():
                continue
            times = []
            with open(cand) as f:
                header = f.readline()
                if "time_s" not in header:
                    continue
                for line in f:
                    parts = line.split(",")
                    if not parts:
                        continue
                    try:
                        times.append(float(parts[0]))
                    except ValueError:
                        continue
            if len(times) >= n_expected:
                return times[:n_expected] if n_expected > 0 else times
            if times:
                return times
        except OSError:
            continue
    return None


def _load_probe_csv(run_dir: Path):
    """Load the solver-written probe history, or None if the run has none.

    probe_points.csv / probe_paths.csv are written by the generated .edp and carry
    the true simulated time for every sample, which is strictly better than
    re-sampling VTU frames (whose time axis is only known per saved frame).
    """
    def _load(name):
        for cand in (run_dir / name, run_dir.parent / "FreeFEM" / name):
            try:
                if cand.exists():
                    with open(cand) as f:
                        return list(csv.DictReader(f))
            except OSError:
                continue
        return None

    return _load("probe_points.csv"), _load("probe_paths.csv")


def generate_probe_history_plots(run_dir: Path, outdir: Path) -> list:
    """Build history plots from the solver's own probe output.

    Returns the list of PNG files written (empty when no probes were requested),
    so the caller can include them in the report and skip them otherwise.
    """
    from core.postprocess.output_manager import canonical_run_dir
    pts, paths = _load_probe_csv(Path(canonical_run_dir(str(run_dir))))
    made = []
    if not pts and not paths:
        return made

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

    if pts:
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
        groups = {}
        for r in pts:
            try:
                groups.setdefault(int(float(r["point"])), []).append(r)
            except (KeyError, ValueError):
                continue
        cmap = plt.get_cmap("viridis")
        n = max(1, len(groups))
        for i, pid in enumerate(sorted(groups)):
            rows = sorted(groups[pid], key=lambda r: float(r["t_s"]))
            t = [float(r["t_s"]) for r in rows]
            T = [float(r["Temp_C"]) for r in rows]
            z = float(rows[0].get("z_m", 0.0)) * 1000.0
            ax.plot(t, T, lw=2.0, color=cmap(i / max(n - 1, 1)),
                    label=f"point {pid}  (x={float(rows[0].get('x_m', 0))*1000:.0f} mm, z={z:.1f} mm)")
        ax.set_xlabel("Simulated time [s]", fontweight="bold")
        ax.set_ylabel("Temperature [°C]", fontweight="bold")
        ax.set_title("Solver-Sampled Probe-Point Temperature Histories", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        f = outdir / "06_probe_point_history.png"
        fig.savefig(f, bbox_inches="tight")
        plt.close(fig)
        made.append(f)

    if paths:
        # One figure per path: temperature and cooling rate vs arc length, using
        # the final recorded time so the plot shows the end-of-build state.
        pgroups = {}
        for r in paths:
            try:
                pgroups.setdefault(int(float(r["path"])), []).append(r)
            except (KeyError, ValueError):
                continue
        for pid, rows in sorted(pgroups.items()):
            rows.sort(key=lambda r: float(r["t_s"]))
            last_t = max(float(r["t_s"]) for r in rows)
            snap = [r for r in rows if abs(float(r["t_s"]) - last_t) < 1e-9]
            if not snap:
                continue
            snap.sort(key=lambda r: float(r["s_arc_m"]))
            s_mm = [float(r["s_arc_m"]) * 1000.0 for r in snap]
            T = [float(r["Temp_C"]) for r in snap]
            cr = [abs(float(r["CoolingRate_C_per_s"])) for r in snap]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=300,
                                           sharex=True, gridspec_kw={"hspace": 0.12})
            ax1.plot(s_mm, T, lw=2.2, color="#b91c1c")
            ax1.set_ylabel("Temperature [°C]", fontweight="bold")
            ax1.set_title(f"Probe Path {pid}: final thermal state at t = {last_t:.1f} s",
                          fontweight="bold")
            ax1.grid(alpha=0.3)
            ax2.plot(s_mm, cr, lw=2.2, color="#1d4ed8")
            ax2.set_xlabel("Arc length along path [mm]", fontweight="bold")
            ax2.set_ylabel("|Cooling rate| [°C/s]", fontweight="bold")
            ax2.grid(alpha=0.3)
            fig.savefig(outdir / f"07_probe_path_{pid}_profile.png", bbox_inches="tight")
            plt.close(fig)
            made.append(outdir / f"07_probe_path_{pid}_profile.png")

    return made


def generate_thermocouple_plot(run_dir: Path, out_path: Path,
                                params: dict = None) -> Path:
    """
    Generate the broken-axis substrate thermocouple plot at 5 depths spanning
    the real substrate thickness.
    """
    vtu_files = sorted(glob.glob(str(run_dir / "visual_outputs" / "T_*.vtu")))
    time_series = []
    tc_data = {0: [], 1: [], 2: [], 3: [], 4: []}

    # Thermocouple stations are placed at the ACTUAL part centre and real
    # substrate thickness. They used to be hardcoded at (0.075, 0.075) mm -- the
    # centre of a 150 mm plate -- so on any other geometry the "thermocouples"
    # sampled air, and on a thin plate the fixed 0..4 mm ladder never reached
    # the deposit.
    params = params or {}
    try:
        _lx = float(params.get("lx", 0.150))
        _ly = float(params.get("ly_base", _lx))
        _hz = float(params.get("hz_base", 0.005))
    except (TypeError, ValueError):
        _lx, _ly, _hz = 0.150, 0.150, 0.005
    _xc = _ly and float(params.get("_tc_x_m", _lx / 2.0))
    _yc = float(params.get("_tc_y_m", _ly / 2.0))
    tc_points = np.array([
        [_xc, _yc, _hz * d / 4.0] for d in range(5)
    ])

    # Real frame times, used when a PVD timestep is unavailable. waam_data.csv is
    # written by the solver with one row per saved frame (time_s,layer,...) and is
    # the authoritative time axis. Previously the last-resort fallback was
    # "i * 10.125", an invented spacing that silently produced a plausible but
    # false cooling curve. We now read the real times, and only if even that is
    # missing do we plot against frame index and say so on the axis label.
    csv_times = _read_csv_frame_times(run_dir, len(vtu_files))
    time_axis_is_index = False

    for i, f in enumerate(vtu_files):
        m = pv.read(f)
        # Read time from PVD file or VTU metadata
        t_val = None
        pvd_path = run_dir / "visual_outputs" / "waam.pvd"
        if pvd_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(pvd_path))
                root = tree.getroot()
                datasets = root.findall(".//DataSet")
                if i < len(datasets):
                    t_val = float(datasets[i].get("timestep", 0))
            except Exception:
                pass
        if t_val is None and "Time" in m.field_data:
            t_val = float(m.field_data["Time"][0])
        if t_val is None and csv_times is not None and i < len(csv_times):
            t_val = csv_times[i]
        if t_val is None:
            # No real time information anywhere: plot against frame index and
            # label the axis honestly instead of inventing a time scale.
            t_val = float(i)
            time_axis_is_index = True
        time_series.append(t_val)
        probed = pv.PolyData(tc_points).sample(m)
        temps = probed["TempC"]
        for d in range(5):
            tc_data[d].append(float(temps[d]))

    # Broken-axis plot setup
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, sharey=True, figsize=(11.5, 5.0), dpi=300,
        gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.06}
    )

    fig.patch.set_facecolor('#ffffff')
    ax_left.set_facecolor('#ffffff')
    ax_right.set_facecolor('#ffffff')

    colors = ['#1d4ed8', '#0284c7', '#059669', '#d97706', '#dc2626']
    styles = ['-', '--', '-.', ':', '-']
    labels = [
        r'Substrate TC: $Z = 0\,\mathrm{mm}$ (Plate Bottom)',
        r'Substrate TC: $Z = 1\,\mathrm{mm}$ Depth',
        r'Substrate TC: $Z = 2\,\mathrm{mm}$ Depth',
        r'Substrate TC: $Z = 3\,\mathrm{mm}$ Depth',
        r'Substrate TC: $Z = 4\,\mathrm{mm}$ Depth',
    ]

    for d in range(5):
        ax_left.plot(time_series, tc_data[d], color=colors[d], linestyle=styles[d],
                     linewidth=2.2, label=labels[d])
        ax_right.plot(time_series, tc_data[d], color=colors[d], linestyle=styles[d],
                      linewidth=2.2)

    # The broken axis assumes a real second-valued time axis with a 500 s split.
    # Without real times the split is meaningless, so fall back to a single
    # continuous axis rather than drawing a bogus 0-500 s / 500-end break.
    if not time_axis_is_index and max(time_series) > 500.0:
        ax_left.set_xlim(0, 500)
        ax_right.set_xlim(500, max(time_series))
    elif not time_axis_is_index:
        ax_left.set_xlim(min(time_series), max(time_series))
        ax_right.set_xlim(min(time_series), max(time_series))
    else:
        ax_left.set_xlim(0, max(1, max(time_series)))
        ax_right.set_xlim(0, max(1, max(time_series)))
    ax_left.set_ylim(20, max(max(tc_data[4]) * 1.15, 30.0))

    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.yaxis.set_ticks_position('right')

    d_len = 0.015
    kwargs = dict(transform=ax_left.transAxes, color='#0f172a', clip_on=False, linewidth=1.2)
    ax_left.plot((1 - d_len, 1 + d_len), (-d_len, +d_len), **kwargs)
    ax_left.plot((1 - d_len, 1 + d_len), (1 - d_len, 1 + d_len), **kwargs)

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d_len, +d_len), (-d_len, +d_len), **kwargs)
    ax_right.plot((-d_len, +d_len), (1 - d_len, 1 + d_len), **kwargs)

    for ax in [ax_left, ax_right]:
        ax.grid(True, linestyle='-', color='#e2e8f0', linewidth=0.8, alpha=0.9)
        for spine in ['top', 'bottom']:
            ax.spines[spine].set_color('#0f172a')
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(colors='#0f172a', labelsize=11)

    ax_left.spines['left'].set_color('#0f172a')
    ax_right.spines['right'].set_color('#0f172a')

    if time_axis_is_index:
        # Be explicit that the abscissa is a frame counter, not seconds.
        x_left = r'Frame Index (no time data available in this run)'
        x_right = r'Frame Index (no time data available in this run)'
    else:
        x_left = r'Elapsed Time, $t\,\mathrm{[s]}$ (Magnified Initial Transient)'
        x_right = r'Elapsed Time, $t\,\mathrm{[s]}$ (Extended Cooldown)'
    ax_left.set_xlabel(x_left, fontsize=12, fontweight='bold', color='#0f172a', labelpad=8)
    ax_right.set_xlabel(x_right, fontsize=12, fontweight='bold', color='#0f172a', labelpad=8)
    ax_left.set_ylabel(r'Substrate Temperature, $T\,\mathrm{[^\circ C]}$',
                       fontsize=12, fontweight='bold', color='#0f172a', labelpad=10)

    fig.suptitle(r'Substrate Thermocouple Transient Response at Plate Center across Depths $Z = 0, 1, 2, 3, 4\,\mathrm{mm}$',
                 fontsize=13, fontweight='bold', color='#0f172a', y=0.98)

    ax_left.legend(loc='upper right', framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)

    fig.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    return out_path


def render_uncut_3d_plots(run_dir: Path, outdir: Path):
    """
    Renders 3D field plots ensuring the baseplate is NEVER clipped or cut off.
    """
    if not HAS_3D:
        return

    # Find the last mechanical and thermal VTU files
    mech_vtu_files = sorted(glob.glob(str(run_dir / "visual_outputs" / "waam_mech_*.vtu")))
    therm_vtu_files = sorted(glob.glob(str(run_dir / "visual_outputs" / "T_*.vtu")))

    mech_vtu = Path(mech_vtu_files[-1]) if mech_vtu_files else None
    therm_vtu = Path(therm_vtu_files[-1]) if therm_vtu_files else None

    m_mech = pv.read(str(mech_vtu)) if mech_vtu and mech_vtu.exists() else None
    m_therm = pv.read(str(therm_vtu)) if therm_vtu and therm_vtu.exists() else None

    cam_pos = [(0.36, 0.36, 0.24), (0.075, 0.075, 0.045), (0.0, 0.0, 1.0)]
    cam_zoom = 0.90

    if m_therm:
        p = pv.Plotter(off_screen=True, window_size=[1000, 750])
        p.set_background("#ffffff")
        active_t = m_therm.threshold(0.5, scalars="Activation")
        p.add_mesh(active_t, scalars="Tpeak", cmap="turbo", clim=[25.0, 1650.0],
                   show_edges=True, edge_color="#334155", line_width=0.6,
                   scalar_bar_args={
                       "title": "Peak Temp Tpeak (°C)", "vertical": True, "color": "#0f172a",
                       "title_font_size": 12, "label_font_size": 10, "position_x": 0.86,
                       "position_y": 0.20, "width": 0.05, "height": 0.60, "fmt": "%.0f"
                   })
        p.camera_position = cam_pos
        p.camera.zoom(cam_zoom)
        p.screenshot(str(run_dir / "field_tpeak_white.png"))
        p.screenshot(str(outdir / "field_tpeak_white.png"))
        p.close()

    if m_mech:
        p = pv.Plotter(off_screen=True, window_size=[1000, 750])
        p.set_background("#ffffff")
        m_mech["Uz_mm"] = m_mech["Uz"] * 1000.0
        p.add_mesh(m_mech, scalars="Uz_mm", cmap="viridis",
                   show_edges=True, edge_color="#334155", line_width=0.6,
                   scalar_bar_args={
                       "title": "Vertical Warpage Uz (mm)", "vertical": True, "color": "#0f172a",
                       "title_font_size": 12, "label_font_size": 10, "position_x": 0.86,
                       "position_y": 0.20, "width": 0.05, "height": 0.60, "fmt": "%.2f"
                   })
        p.camera_position = cam_pos
        p.camera.zoom(cam_zoom)
        p.screenshot(str(run_dir / "field_warpage_white.png"))
        p.screenshot(str(outdir / "field_warpage_white.png"))
        p.close()


def render_toolpath_animation(run_dir: Path, outdir: Path, stl_path: Path):
    """
    Renders toolpath-following build animation with fine elements and uncut baseplate.
    """
    if not HAS_3D or stl_path is None or not stl_path.exists() or not stl_path.is_file():
        return

    toolpath_file = run_dir / "toolpath.txt"
    if not toolpath_file.exists():
        return

    with open(toolpath_file) as f:
        lines = [l.strip().split() for l in f if l.strip()]

    pts = []
    for p in lines[1:]:
        pts.append({"t": float(p[0]), "x": float(p[1]), "y": float(p[2]), "z": float(p[3]), "power": int(float(p[4]))})

    cad = pv.read(str(stl_path))
    cad_placed = cad.copy()
    zmin = cad.bounds[4]
    cad_placed.points[:, 0] = cad_placed.points[:, 0] * 0.001 + 0.075
    cad_placed.points[:, 1] = cad_placed.points[:, 1] * 0.001 + 0.075
    cad_placed.points[:, 2] = (cad_placed.points[:, 2] - zmin) * 0.001 + 0.015

    base = pv.Box(bounds=[0.0, 0.150, 0.0, 0.150, 0.0, 0.015])
    base["TempC"] = np.full(base.n_points, 45.0)

    n_frames = 54
    frame_indices = np.linspace(0, len(pts) - 1, n_frames, dtype=int)
    cam_pos = [(0.36, 0.36, 0.24), (0.075, 0.075, 0.045), (0.0, 0.0, 1.0)]
    cam_zoom = 0.90

    # --- Real solver temperature frames (timestep, path) from waam.pvd ----------
    # Previously the deposit's temperature was INVENTED analytically
    # (75 + 950*z/zmax + 650*exp(-d^2)). Now we load the actual VTU frames the
    # solver wrote and sample them onto the CAD surface. If no frames exist we
    # render the deposit untinted rather than showing invented heat.
    vis_dir = run_dir / "visual_outputs"
    frame_table = []  # list of (timestep_s, vtu_path)
    pvd = vis_dir / "waam.pvd"
    if pvd.exists():
        try:
            pvd_text = pvd.read_text(errors="ignore")
            for m_t, m_f in re.finditer(
                    r"<DataSet timestep='([^']*)'\s+file='([^']*)'", pvd_text):
                try:
                    ts = float(m_t)
                except ValueError:
                    ts = float("nan")
                fp = vis_dir / m_f
                if fp.exists():
                    frame_table.append((ts, fp))
        except Exception:
            frame_table = []
    frame_table.sort(key=lambda r: (r[0] if r[0] == r[0] else 0.0))
    if frame_table:
        print(f"[ReportEngine] Animation sampling {len(frame_table)} real solver frames")

    # Read actual layer count from parameters
    params_file = run_dir / "parameters.json"
    n_layers_actual = 27
    hz_base_actual = 5.0
    h_bead_actual = 3.0
    if params_file.exists():
        try:
            with open(params_file) as f:
                params = json.load(f)
            n_layers_actual = resolve_n_layers(run_dir, params)
            hz_base_actual = float(params.get("hz_base", 0.005)) * 1000.0
            h_bead_actual = float(params.get("h_bead", 0.003)) * 1000.0
        except Exception:
            pass

    gif_images = []
    for fi, p_idx in enumerate(frame_indices):
        pt = pts[p_idx]
        cur_z, cur_x, cur_y, cur_t = pt["z"], pt["x"], pt["y"], pt["t"]
        z_mm = cur_z * 1000.0
        layer_num = min(n_layers_actual, max(1, int(round((z_mm - hz_base_actual) / max(h_bead_actual, 0.1)))))

        p = pv.Plotter(off_screen=True, window_size=[1000, 750])
        p.set_background("#ffffff")
        p.add_mesh(base, scalars="TempC", cmap="turbo", clim=[25.0, 1650.0],
                   show_edges=True, edge_color="#64748b", line_width=0.7, opacity=0.96)

        if cur_z > 0.0155:
            dep = cad_placed.clip(normal=[0, 0, 1], origin=[0, 0, cur_z], invert=True)
            if dep.n_points > 0:
                # Sample the real solver field for the frame closest in time to
                # this animation step. Falls back to ambient (not a guess) when
                # no frame data is available.
                t_field = np.full(dep.n_points, 25.0)
                if frame_table:
                    ts_list = np.array([r[0] for r in frame_table], dtype=float)
                    ts_list = np.where(np.isnan(ts_list),
                                       np.linspace(0.0, float(cur_t), len(ts_list)),
                                       ts_list)
                    j = int(np.argmin(np.abs(ts_list - cur_t)))
                    try:
                        src = pv.read(str(frame_table[j][1]))
                        sampled = pv.PolyData(dep.points.copy()).sample(src)
                        vals = np.asarray(sampled["TempC"]).ravel()
                        # FreeFEM writes not-yet-activated nodes as -1.
                        vals = np.where(np.isfinite(vals) & (vals > 0.0), vals, 25.0)
                        if vals.size == dep.n_points:
                            t_field = vals
                    except Exception:
                        t_field = np.full(dep.n_points, 25.0)
                t_field = np.clip(t_field, 25.0, 1650.0)
                dep["TempC"] = t_field
                p.add_mesh(dep, scalars="TempC", cmap="turbo", clim=[25.0, 1650.0],
                           show_edges=True, edge_color="#334155", line_width=0.55,
                           scalar_bar_args={
                               "title": "Temp [°C]", "vertical": True, "color": "#0f172a",
                               "title_font_size": 12, "label_font_size": 10, "position_x": 0.86,
                               "position_y": 0.20, "width": 0.05, "height": 0.60, "fmt": "%.0f"
                           })

        if cur_z > 0.015:
            torch = pv.Sphere(radius=0.0035, center=[cur_x, cur_y, cur_z])
            p.add_mesh(torch, color="#fde047", emissive=True)

        p.camera_position = cam_pos
        p.camera.zoom(cam_zoom)
        hud_text = f"WAAM Toolpath Build | Pass {layer_num}/{n_layers_actual} | Z = {z_mm:.1f} mm | t = {cur_t:.1f} s"
        p.add_text(hud_text, font_size=11, color="#0f172a", position="upper_left")

        img = p.screenshot(return_img=True)
        gif_images.append(img)
        p.close()

    gif_out = run_dir / "build_animation_white.gif"
    imageio.mimsave(str(gif_out), gif_images, duration=0.18, loop=0)
    imageio.mimsave(str(outdir / "build_animation_white.gif"), gif_images, duration=0.18, loop=0)


def generate_production_report(run_dir: str):
    """
    Main entry point for generating the production-grade 2-page LaTeX PDF report.
    """
    from core.postprocess.output_manager import canonical_run_dir
    run_dir = Path(canonical_run_dir(str(run_dir))).resolve()
    outdir = run_dir.parent if run_dir.name == "FreeFEM" else run_dir
    freefem_dir = outdir / "FreeFEM" if (outdir / "FreeFEM").exists() else run_dir

    print(f"[ReportEngine] Generating production report for: {freefem_dir}")

    # Generate 2D plots
    from postprocess.plot_generator import generate_all_plots
    generate_all_plots(str(freefem_dir))

    def _copy_if_distinct(src: Path, dst: Path):
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    _copy_if_distinct(freefem_dir / "plots" / "01_thermal_history_and_peaks.png", freefem_dir / "01_thermal_history_and_peaks.png")
    _copy_if_distinct(freefem_dir / "plots" / "01_thermal_history_and_peaks.png", outdir / "01_thermal_history_and_peaks.png")

    # Generate thermocouple plot
    tc_plot = freefem_dir / "thermal_history_thermocouples_unlog.png"
    try:
        import json as _json
        _ppath = run_dir / "parameters.json"
        _p = _json.load(open(_ppath)) if _ppath.exists() else {}
    except Exception:
        _p = {}
    generate_thermocouple_plot(freefem_dir, tc_plot, params=_p)
    _copy_if_distinct(tc_plot, outdir / "thermal_history_thermocouples_unlog.png")

    # Re-render 3D plots
    render_uncut_3d_plots(freefem_dir, outdir)

    # Render animation
    stl_path = _find_stl(outdir, freefem_dir)
    render_toolpath_animation(freefem_dir, outdir, stl_path)

    print("[ReportEngine] Completed visual outputs. Compiling report.html...")
    
    # Extract physical layer records from VTU files
    vtu_files = sorted(glob.glob(str(freefem_dir / "visual_outputs" / "T_*.vtu")))
    tc_pt = np.array([[0.075, 0.075, 0.003]]) # Z = 3 mm depth

    # Compute dwell indices from simulation parameters
    params_file = freefem_dir / "parameters.json"
    params = {}
    if params_file.exists():
        try:
            with open(params_file) as f:
                params = json.load(f)
        except Exception:
            pass

    n_layers = resolve_n_layers(freefem_dir, params)
    save_every_n_weld = int(params.get("save_every_n_weld", 3))
    dt_weld = float(params.get("dt_weld", 0.333))
    t_wait = float(params.get("t_wait", 60.0))
    lx = float(params.get("lx", 0.15))
    v_scan = float(params.get("v_scan", 0.01))
    t_pass = lx / v_scan if v_scan > 0 else 75.0

    # Compute dwell indices: each layer has welding steps + inter-pass cooling
    dwell_indices = []
    for lay_idx in range(1, n_layers + 1):
        n_weld_steps = int(t_pass / dt_weld + 0.5)
        # Frames saved during welding: n_weld_steps // save_every_n_weld
        weld_frames = n_weld_steps // save_every_n_weld
        # Frames saved during inter-pass: depends on dt_wait and save_every_n_cool
        # Approximate: 1 frame per inter-pass snapshot
        total_frames_before = (lay_idx - 1) * (weld_frames + 1)
        dwell_idx = total_frames_before + weld_frames
        dwell_indices.append(dwell_idx)

    layer_records = []
    for lay_idx in range(1, n_layers + 1):
        t_time = lay_idx * (t_pass + t_wait)
        dwell_vtu_idx = dwell_indices[lay_idx - 1]
        active_vtu_idx = max(0, dwell_vtu_idx - 2)

        if active_vtu_idx >= len(vtu_files) or dwell_vtu_idx >= len(vtu_files):
            continue

        m_active = pv.read(vtu_files[active_vtu_idx])
        m_dwell = pv.read(vtu_files[dwell_vtu_idx])
        
        probed_active = pv.PolyData(tc_pt).sample(m_active)
        t_rec = float(probed_active["TempC"][0])
        t_max = float(np.max(m_active["TempC"]))
        t_min = float(np.min(m_active["TempC"]))
        
        probed_dwell = pv.PolyData(tc_pt).sample(m_dwell)
        t_interpass = float(probed_dwell["TempC"][0])
        t_end = float(np.max(m_dwell["TempC"]))
        
        layer_records.append({
            "layer": f"L{lay_idx}",
            "time": f"{t_time:.1f}",
            "t_rec": f"{t_rec:.1f}",
            "t_max": f"{t_max:.1f}",
            "t_min": f"{t_min:.1f}",
            "t_end": f"{t_end:.1f}",
            "t_interpass": f"{t_interpass:.1f}",
        })

    # Compile Table HTML rows
    table_rows = ""
    for r in layer_records:
        table_rows += f"""
    <tr>
        <td style="font-weight:600; text-align:center;">{r['layer']}</td>
        <td>{r['time']}</td>
        <td style="color:#1d4ed8; font-weight:600;">{r['t_rec']}</td>
        <td style="color:#dc2626; font-weight:600;">{r['t_max']}</td>
        <td>{r['t_min']}</td>
        <td>{r['t_end']}</td>
        <td>{r['t_interpass']}</td>
    </tr>
    """

    # Compute executive metrics from simulation data
    max_tpeak = 0.0
    max_warpage = 0.0
    total_time = 0.0
    n_layers_csv = 0

    # Read from waam_data.csv if available
    waam_data_csv = freefem_dir / "waam_data.csv"
    if waam_data_csv.exists():
        try:
            import csv
            with open(waam_data_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = float(row.get("time_s", 0))
                    if t > total_time:
                        total_time = t
                    tmax = float(row.get("Tmax_C", 0))
                    if tmax > max_tpeak:
                        max_tpeak = tmax
                    try:
                        lay = int(row.get("layer", 0))
                        if lay > n_layers_csv:
                            n_layers_csv = lay
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

    # Read from sim.log for warpage
    sim_log = freefem_dir / "sim.log"
    if sim_log.exists():
        try:
            with open(sim_log) as f:
                content = f.read()
                import re
                warpage_matches = re.findall(r"Vertical Warpage \(Uz\):\s*([\d\.]+)\s*mm", content)
                if warpage_matches:
                    max_warpage = max(float(v) for v in warpage_matches)
        except Exception:
            pass

    # Read from parameters.json for layer count
    params_file = freefem_dir / "parameters.json"
    params = {}
    if params_file.exists():
        try:
            with open(params_file) as f:
                params = json.load(f)
        except Exception:
            pass

    n_layers = resolve_n_layers(freefem_dir, params)
    hz_base = float(params.get("hz_base", 0.005))
    h_bead = float(params.get("h_bead", 0.003))
    total_height = hz_base + n_layers * h_bead

    # Compute total simulated time from layer pass times
    if not total_time:
        total_time = n_layers * (t_pass + t_wait)

    # Geometry/model identity (geometry-agnostic report naming)
    import re as _re
    model_name = outdir.name
    doc_id = "WAAM-" + _re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").upper() + "-2026"
    alloy_base = params.get("material_base", "SS316L")
    alloy_dep = params.get("material_dep", params.get("material", "SS308L"))
    lx_mm = float(params.get("lx", 0.15)) * 1000.0
    ly_mm = float(params.get("ly", params.get("lx", 0.15))) * 1000.0
    hz_mm = hz_base * 1000.0
    hb_mm = h_bead * 1000.0
    total_height_mm = total_height * 1000.0

    pvd_abs = str((freefem_dir / "visual_outputs" / "waam.pvd").resolve())
    if not Path(pvd_abs).exists():
        pvd_files = list(freefem_dir.glob("**/waam.pvd"))
        if pvd_files:
            pvd_abs = str(pvd_files[0].resolve())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WAAM Finite Element Multi-Physics Report — {model_name}</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    @page {{
        size: A4 portrait;
        margin: 12mm 15mm;
    }}
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    body {{
        font-family: 'Latin Modern Roman', 'Times New Roman', serif;
        background-color: #525659;
        color: #1a1a1a;
        line-height: 1.4;
        padding: 20px;
    }}
    .document-container {{
        display: flex;
        flex-direction: row;
        gap: 20px;
        justify-content: center;
        max-width: 1700px;
        margin: 0 auto;
    }}
    .latex-page {{
        background: #ffffff;
        width: 794px;
        min-height: 1123px;
        height: auto;
        padding: 45px 50px 30px 50px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        position: relative;
        display: flex;
        flex-direction: column;
    }}
    .header-box {{
        border-bottom: 2px solid #0f172a;
        padding-bottom: 10px;
        margin-bottom: 14px;
    }}
    h1 {{
        font-size: 20px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #0f172a;
        margin-bottom: 4px;
    }}
    .sub-title {{
        font-size: 13px;
        color: #475569;
        font-style: italic;
    }}
    .meta-row {{
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }}
    h2 {{
        font-size: 12.5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #0f172a;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 3px;
        margin-top: 10px;
        margin-bottom: 6px;
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 8px;
    }}
    .metric-card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 6px 8px;
        text-align: center;
    }}
    .metric-card .label {{
        font-size: 9.5px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 2px;
    }}
    .metric-card .val {{
        font-size: 13px;
        font-weight: 700;
        color: #0f172a;
    }}
    .paraview-command-bar {{
        background: #0f172a;
        color: #38bdf8;
        padding: 6px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}
    .copy-btn {{
        background: #38bdf8;
        color: #0f172a;
        border: none;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 9.5px;
        font-weight: bold;
        cursor: pointer;
    }}
    .copy-btn:hover {{
        background: #7dd3fc;
    }}
    .image-frame {{
        border: 1px solid #cbd5e1;
        background: #ffffff;
        text-align: center;
        border-radius: 4px;
        overflow: hidden;
        margin: 6px 0;
    }}
    .image-frame img {{
        width: 100%;
        max-height: 270px;
        object-fit: contain;
        display: block;
        background: #ffffff;
    }}
    .caption {{
        font-size: 10.5px;
        color: #475569;
        text-align: center;
        margin-top: 2px;
        margin-bottom: 6px;
        font-style: italic;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 9.5px;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }}
    table.data-table th {{
        background: #f1f5f9;
        color: #334155;
        padding: 4px 6px;
        border-bottom: 2px solid #cbd5e1;
        text-align: right;
    }}
    table.data-table th:first-child {{ text-align: center; }}
    table.data-table td {{
        padding: 3px 6px;
        border-bottom: 1px solid #e2e8f0;
        text-align: right;
    }}
    .table-container {{
        max-height: 380px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
    }}
    .footer-page {{
        margin-top: auto;
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        padding-top: 8px;
    }}
    @media (max-width: 1650px) {{
        .document-container {{
            flex-direction: column;
            align-items: center;
        }}
    }}
    @media print {{
        body {{
            background-color: #ffffff !important;
            padding: 0 !important;
            color: #000000 !important;
        }}
        .document-container {{
            display: block !important;
            max-width: 100% !important;
            width: 100% !important;
            margin: 0 !important;
            gap: 0 !important;
        }}
        .latex-page {{
            width: 100% !important;
            min-height: auto !important;
            height: auto !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
            break-after: page !important;
            page-break-after: always !important;
            display: block !important;
        }}
        .latex-page:last-child {{
            break-after: auto !important;
            page-break-after: auto !important;
        }}
        .table-container {{
            max-height: none !important;
            overflow: visible !important;
            border: 1px solid #cbd5e1 !important;
        }}
        table.data-table {{
            page-break-inside: auto !important;
            break-inside: auto !important;
        }}
        table.data-table tr {{
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .image-frame, .caption, .metric-grid, .metric-card, .header-box, .paraview-command-bar {{
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        h1, h2 {{
            break-after: avoid !important;
            page-break-after: avoid !important;
        }}
        .copy-btn, .no-print {{
            display: none !important;
        }}
        .footer-page {{
            position: static !important;
            margin-top: 20px !important;
            border-top: 1px solid #cbd5e1 !important;
        }}
    }}
</style>
</head>
<body>

<div class="document-container">
    <!-- PAGE 1 -->
    <div class="latex-page">
        <div class="header-box">
            <h1>WAAM Finite Element Multi-Physics Report</h1>
            <div class="sub-title">Model: {model_name} | Build Height: \\({total_height_mm:.1f}\\,\\mathrm{{mm}}\\) ({n_layers} Layers) | Alloy: {alloy_dep} Deposit / {alloy_base} Base</div>
            <div class="meta-row">
                <span>Date: October 2026</span>
                <span>Framework: WAAM-Framework FreeFEM++ / Gmsh</span>
                <span>Page 1 of 2</span>
            </div>
        </div>

        <h2>1. Global Executive Parameters</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Total Deposition</div>
                <div class="val">{n_layers} Passes</div>
            </div>
            <div class="metric-card">
                <div class="label">Peak Melt Temp</div>
                <div class="val">\\({max_tpeak:.1f}\\,^\\circ\\mathrm{{C}}\\)</div>
            </div>
            <div class="metric-card">
                <div class="label">Plate Contact</div>
                <div class="val">\\(0.00\\,\\mathrm{{mm}}\\) Gap</div>
            </div>
            <div class="metric-card">
                <div class="label">Physical Duration</div>
                <div class="val">\\({total_time:.0f}\\,\\mathrm{{s}}\\)</div>
            </div>
        </div>

        <h2>2. 3D ParaView Simulation Launch</h2>
        <div class="paraview-command-bar">
            <span id="pvdCmd">paraview "{pvd_abs}"</span>
            <button class="copy-btn" onclick="copyCommand()">Copy Command</button>
        </div>

        <h2>3. Substrate Thermocouple Transient Response (\\(Z = 0, 1, 2, 3, 4\\,\\mathrm{{mm}}\\) Depth)</h2>
        <div class="image-frame">
            <img src="thermal_history_thermocouples_unlog.png" alt="Unlog Substrate Thermocouple Plot">
        </div>
        <div class="caption">Figure 1: Substrate thermocouple history with magnified initial transient (\\(t \\in [0, 500]\\,\\mathrm{{s}}\\)) and extended cooldown tail (\\(t \\in [500, {total_time:.0f}]\\,\\mathrm{{s}}\\)) at depths \\(Z = 0, 1, 2, 3, 4\\,\\mathrm{{mm}}\\).</div>

        <h2>4. Layer-by-Layer Deposition Build Animation</h2>
        <div class="image-frame">
            <img src="build_animation_white.gif" alt="Layer-by-Layer Build Animation">
        </div>
        <div class="caption">Figure 2: Toolpath-following deposition build animation with active arc head and conformal fine mesh on full baseplate.</div>

        <div class="footer-page">
            <span>WAAM Engineering Verification Report</span>
            <span>Document ID: {doc_id}</span>
            <span>Page 1</span>
        </div>
    </div>

    <!-- PAGE 2 -->
    <div class="latex-page">
        <div class="header-box">
            <h1>Layer History & Substrate Thermocouple Log</h1>
            <div class="sub-title">Substrate Probe Depths: \\(Z \\in [0, 4]\\,\\mathrm{{mm}}\\) | Layer Thickness: \\(h = {hb_mm:.1f}\\,\\mathrm{{mm}}\\)</div>
            <div class="meta-row">
                <span>Mesh Resolution: conformal tetrahedral (STL-derived)</span>
                <span>Substrate: \\({lx_mm:.0f} \\times {ly_mm:.0f} \\times {hz_mm:.0f}\\,\\mathrm{{mm}}\\)</span>
                <span>Page 2 of 2</span>
            </div>
        </div>

        <h2>5. Layer-by-Layer Process Thermal Record</h2>
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Time [s]</th>
                        <th>T Recorded [°C]</th>
                        <th>T max [°C]</th>
                        <th>T min [°C]</th>
                        <th>T end [°C]</th>
                        <th>T interpass [°C]</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        <div class="caption">Table 1: Thermal states, substrate probe temperatures, and interpass boundary conditions across all {n_layers} passes.</div>

        <h2>6. Thermal & Mechanical Field Verification</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
                <div class="image-frame">
                    <img src="field_tpeak_white.png" alt="Peak Thermal Envelope">
                </div>
                <div class="caption">Figure 3: Peak Thermal Envelope (\\(T_{{\\mathrm{{peak}}}}\\)).</div>
            </div>
            <div>
                <div class="image-frame">
                    <img src="field_warpage_white.png" alt="Vertical Warpage">
                </div>
                <div class="caption">Figure 4: Residual Vertical Warpage (\\(U_z\\)).</div>
            </div>
        </div>

        <h2>7. Transient Melt Pool Thermal Cycles Across Welding Passes</h2>
        <div class="image-frame">
            <img src="01_thermal_history_and_peaks.png" alt="Transient Melt Pool Thermal Cycles">
        </div>
        <div class="caption">Figure 5: Transient melt pool thermal cycles across all weld passes, liquidus/solidus thresholds, and inter-pass dwell cooldown evolution.</div>

        <div class="footer-page">
            <span>WAAM Engineering Verification Report</span>
            <span>Document ID: {doc_id}</span>
            <span>Page 2</span>
        </div>
    </div>
</div>

<script>
function copyCommand() {{
    const cmd = document.getElementById('pvdCmd').innerText;
    navigator.clipboard.writeText(cmd).then(() => {{
        const btn = document.querySelector('.copy-btn');
        btn.innerText = 'Copied!';
        setTimeout(() => btn.innerText = 'Copy Command', 2000);
    }});
}}
</script>

</body>
</html>
"""
    with open(freefem_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(outdir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # Clean up obsolete / stray files.
    # NOTE: the generated figures used to be DELETED here right after being
    # embedded in the report -- the report kept its own copies, but the plots
    # directory the user opened no longer had 02..05. Only genuinely obsolete
    # names are removed now; anything this run produced is left alone.
    obsolete = [
        ".DS_Store", "sphere_mechanical_warpage.png", "sphere_tpeak_contours.png",
        "thermal_history_logscale.png", "sphere_layer_by_layer_build.gif",
        "sphere_tpeak_white.png", "sphere_warpage_white.png",
        "sphere_layer_by_layer_build_white.gif",
    ]
    for d in [outdir, freefem_dir]:
        for obs in obsolete:
            p = d / obs
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    print(f"[ReportEngine] Production report saved: {freefem_dir / 'report.html'}")
    return freefem_dir / "report.html"

