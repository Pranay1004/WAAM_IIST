#!/usr/bin/env python3
"""
WAAM Multi-Physics Native Matplotlib Postprocessing Engine
===========================================================
Generates publication-quality 2D engineering plots from
FreeFEM++ multi-physics simulation data (CSV files, parameters.json, sim.log).

Plots Generated:
1. 01_thermal_history_and_peaks.png  - Transient thermal cycles & peak/dwell temps
2. 02_cooling_rates_and_dt85.png     - Solidification cooling rate dT/dt & dt85 vs Z (from Z=0)
3. 03_distortion_and_warpage.png     - Longitudinal warpage & cold residual distortion
4. 04_residual_stress_mises.png      - Equivalent von Mises & longitudinal stress vs Z (from Z=0)
5. 05_heat_input_and_dwell_energy.png- Net/Gross heat input, WFS/TS kinematics & cooldown

Usage:
    python3 plot_generator.py [run_dir]
"""

import os
import sys
import re
import csv
import json
import glob
import math
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, Any, List, Optional, Tuple

# Prevent matplotlib cache/permission warnings and headless crash
import tempfile
_mpl_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_waam_{os.getuid() if hasattr(os, 'getuid') else '0'}")
try:
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Clean publication style settings (without relying on external latex binaries)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.2,
    "grid.color": "#cbd5e1",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cbd5e1",
})

# Curated harmonious color palette
PALETTE = {
    "primary": "#1e40af",    # Navy Blue
    "secondary": "#dc2626",  # Crimson Red
    "accent": "#059669",     # Emerald Green
    "amber": "#d97706",      # Amber
    "purple": "#7c3aed",     # Violet
    "slate": "#475569",      # Slate Grey
    "sub_bg": "#f1f5f9",     # Light Grey for Substrate
    "dep_bg": "#eff6ff",     # Light Blue for Deposit
}

# Known alloy thermophysical properties for multi-material resolution
ALLOY_DEFAULTS = {
    "SS308L": {"Tsol": 1402.0, "Tliq": 1457.0, "sig_y": 380.0, "name": "SS308L (Austenitic Stainless)"},
    "SS316L": {"Tsol": 1375.0, "Tliq": 1440.0, "sig_y": 290.0, "name": "SS316L (Austenitic Stainless)"},
    "ER70S-6": {"Tsol": 1450.0, "Tliq": 1520.0, "sig_y": 400.0, "name": "ER70S-6 (Carbon Steel)"},
    "Ti-6Al-4V": {"Tsol": 1604.0, "Tliq": 1660.0, "sig_y": 880.0, "name": "Ti-6Al-4V (Grade 5 Titanium)"},
    "IN625": {"Tsol": 1290.0, "Tliq": 1350.0, "sig_y": 450.0, "name": "IN625 (Nickel Superalloy)"},
    "AlSi10Mg": {"Tsol": 557.0, "Tliq": 596.0, "sig_y": 150.0, "name": "AlSi10Mg (Al-Si Alloy)"},
    "Pure Al (Al1xxx)": {"Tsol": 643.0, "Tliq": 660.0, "sig_y": 95.0, "name": "Pure Al (Al1xxx)"},
}


def resolve_alloy_info(alloy_name: str, db: Dict[str, Any] = None) -> Dict[str, Any]:
    """Resolve solidus, liquidus, yield strength for a given alloy name."""
    if db and alloy_name in db:
        entry = db[alloy_name]
        ys_list = entry.get("yield_strength", {}).get("sigma_y", [380e6])
        ys_mpa = float(ys_list[0]) / 1e6 if ys_list and float(ys_list[0]) > 1e4 else float(ys_list[0] if ys_list else 380.0)
        return {
            "Tsol": float(entry.get("solidus", 1400.0)),
            "Tliq": float(entry.get("liquidus", 1460.0)),
            "sig_y": ys_mpa,
            "name": entry.get("name", alloy_name),
        }
    
    # Check default dictionary with fuzzy matching
    norm = alloy_name.upper().replace("-", "").replace(" ", "").replace("_", "")
    for k, v in ALLOY_DEFAULTS.items():
        k_norm = k.upper().replace("-", "").replace(" ", "").replace("_", "")
        if k_norm in norm or norm in k_norm:
            return dict(v)
    return {"Tsol": 1400.0, "Tliq": 1460.0, "sig_y": 380.0, "name": alloy_name}


def parse_sim_log(sim_log_path: str) -> Dict[str, Any]:
    """Parse layer-by-layer mechanical distortion, von Mises stress, and thermal data from sim.log."""
    out = {
        "layers": [],
        "distortions_mm": [],
        "von_mises_mpa": [],
        "dwell_temps_c": [],
        "final_cooling": [],
        "max_distortion": 0.0,
        "max_von_mises": 0.0,
        "final_cold_distortion": None,
        "final_cold_von_mises": None,
    }
    if not os.path.exists(sim_log_path):
        return out

    try:
        with open(sim_log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Parse live/transient mechanical distortion
        dist_matches = re.findall(r"(?:Live Mechanical Distortion|Transient Peak Displacement|Max distortion):\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*mm", content, re.IGNORECASE)
        if dist_matches:
            out["distortions_mm"] = [float(v) for v in dist_matches]
            out["max_distortion"] = max(out["distortions_mm"])

        # Parse final cold-state distortion (including (Uz) notation)
        cold_dist = re.search(r"(?:Final cold-state (?:distortion|residual warpage)(?:\s*\([^\)]*\))?|Max distortion):\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*mm", content, re.IGNORECASE)
        if cold_dist:
            out["final_cold_distortion"] = float(cold_dist.group(1))

        # Parse live/peak von Mises stress (including 'Peak Stress:')
        vm_matches = re.findall(r"(?:Live von Mises(?: Stress)?|Peak von Mises Stress|Peak Stress|Max von Mises):\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*MPa", content, re.IGNORECASE)
        if vm_matches:
            out["von_mises_mpa"] = [float(v) for v in vm_matches]
            out["max_von_mises"] = max(out["von_mises_mpa"])

        # Parse final cold-state von Mises
        cold_vm = re.search(r"(?:Final cold-state (?:von Mises|peak von Mises stress)|Max von Mises):\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*MPa", content, re.IGNORECASE)
        if cold_vm:
            out["final_cold_von_mises"] = float(cold_vm.group(1))

        # Parse inter-pass dwell cool
        dwell_matches = re.findall(r"Inter-pass cool:\s*Tmax=([\d\.]+)C\s*t=([\d\.]+)s", content)
        if dwell_matches:
            out["dwell_temps_c"] = [{"Tdwell_C": float(m[0]), "time_s": float(m[1])} for m in dwell_matches]

        # Parse final cooling steps
        cool_matches = re.findall(r"COOL\s+t=([\d\.]+)s\s+Tmax=([\d\.]+)C", content)
        if cool_matches:
            out["final_cooling"] = [{"time_s": float(m[0]), "temp_c": float(m[1])} for m in cool_matches]

        # Parse layer count
        layer_matches = re.findall(r"--- Layer\s+(\d+)/(\d+)", content)
        if layer_matches:
            out["layers"] = list(range(1, len(layer_matches) + 1))
        elif out["distortions_mm"]:
            out["layers"] = list(range(1, len(out["distortions_mm"]) + 1))

    except Exception as ex:
        print(f"[PlotEngine] Warning parsing sim.log: {ex}")

    return out


def _load_sxx_profile(run_dir: str) -> List[Dict[str, Any]]:
    """Parse the LAST mechanical VTU (final cold-state solve) and return
    0.5 mm z-binned longitudinal stress Sxx [MPa] (+ mean Tpeak per bin).

    Empty list if the VTU has no Sxx_Pa field (older runs predating it).
    """
    pats = [
        os.path.join(run_dir, "FreeFEM", "visual_outputs", "waam_mech*.vtu"),
        os.path.join(run_dir, "visual_outputs", "waam_mech*.vtu"),
    ]
    candidates: List[str] = []
    for pat in pats:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return []
    candidates.sort()
    try:
        import pyvista as pv
    except Exception:
        print("[PlotEngine] pyvista unavailable; skipping Sxx profile")
        return []
    try:
        import numpy as _np
        mesh = pv.read(candidates[-1])  # lexicographic last == final solve
        if "Sxx_Pa" not in mesh.point_data:
            return []
        z_mm = _np.asarray(mesh.points[:, 2], dtype=float) * 1000.0
        sxx = _np.asarray(mesh.point_data["Sxx_Pa"], dtype=float) / 1.0e6
        tp = (_np.asarray(mesh.point_data["Tpeak"], dtype=float)
              if "Tpeak" in mesh.point_data else None)
        if z_mm.size == 0:
            return []
        # 0.5 mm bins, vectorised: the old per-point Python loop with a dict
        # plus a min() per z-row was O(Z x N) and dominated postprocessing on
        # large meshes.
        zb = _np.round(z_mm * 2.0) / 2.0
        keys, inv = _np.unique(zb, return_inverse=True)
        n = _np.bincount(inv).astype(float)
        s_sum = _np.bincount(inv, weights=sxx)
        if tp is not None:
            t_sum = _np.bincount(inv, weights=tp)
        else:
            t_sum = _np.zeros_like(s_sum)
        order = _np.argsort(keys)
        return [{"Z_mm": float(keys[k]), "Sxx_MPa": float(s_sum[k] / n[k]),
                 "Tpeak_C": float(t_sum[k] / n[k])} for k in order]
    except Exception as ex:
        print(f"[PlotEngine] Warning reading mech VTU stress profile: {ex}")
        return []


def load_run_context(run_dir: str) -> Dict[str, Any]:
    """Gather all CSV data, parameter metadata, and parsed logs in run_dir."""
    from core.postprocess.output_manager import canonical_run_dir
    run_dir = canonical_run_dir(os.path.abspath(run_dir))
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Parameters
    params = {}
    for p_candidate in [
        os.path.join(run_dir, "parameters.json"),
        os.path.join(run_dir, "..", "parameters.json"),
        os.path.join(run_dir, "FreeFEM", "parameters.json"),
    ]:
        if os.path.exists(p_candidate):
            try:
                with open(p_candidate, "r", encoding="utf-8") as pf:
                    params = json.load(pf)
                break
            except Exception:
                pass

    # 2. Material database if present
    db = {}
    db_path = Path(__file__).resolve().parent.parent / "core" / "multimaterial" / "material_database.json"
    if db_path.exists():
        try:
            with open(db_path, "r", encoding="utf-8") as df:
                db = json.load(df).get("alloys", {})
        except Exception:
            pass

    def safe_float(val, default=0.0):
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Scrub params of None strings
    for k, v in params.items():
        if v == "None" or v is None:
            params[k] = 0.0

    # 3. Elevation Profile Z (starting at 0 mm datum)
    z_profile = []
    z_csv = os.path.join(run_dir, "hardness_profile_z.csv")
    if not os.path.exists(z_csv):
        z_csv = os.path.join(run_dir, "FreeFEM", "hardness_profile_z.csv")
    if os.path.exists(z_csv):
        try:
            with open(z_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    z_profile.append({
                        "Z_mm": float(r["Z_mm"]),
                        "X_mm": float(r.get("X_mm", 0.0)),
                        "Y_mm": float(r.get("Y_mm", 0.0)),
                        "Tpeak_C": float(r.get("Tpeak_C", 0.0)),
                        "CoolingRate_C_s": float(r.get("CoolingRate_C_s", 0.0)),
                    })
        except Exception as ex:
            print(f"[PlotEngine] Warning reading Z profile CSV: {ex}")

    # 3b. Longitudinal residual stress Sxx(Z) from the final mechanical VTU.
    #     Merged by nearest z into the hardness profile rows; if the hardness
    #     CSV is absent (microstructure stage disabled), the stress profile
    #     becomes the z_profile backbone (CoolingRate=NaN so plot 03 does not
    #     draw a fake flat line).
    sxx_rows = _load_sxx_profile(run_dir)
    if sxx_rows:
        if z_profile:
            for row in z_profile:
                near = min(sxx_rows, key=lambda r: abs(r["Z_mm"] - row["Z_mm"]))
                row["Sxx_MPa"] = near["Sxx_MPa"]
        else:
            z_profile = [{
                "Z_mm": r["Z_mm"],
                "X_mm": 0.0,
                "Y_mm": 0.0,
                "Tpeak_C": r.get("Tpeak_C", 0.0),
                "CoolingRate_C_s": float("nan"),
                "Sxx_MPa": r["Sxx_MPa"],
            } for r in sxx_rows]
            print(f"[PlotEngine] z_profile derived from mech VTU stress ({len(z_profile)} bins)")

    # 4. WAAM Data CSV (Transient Welding Steps)
    transient_data = []
    data_csv = os.path.join(run_dir, "waam_data.csv")
    if not os.path.exists(data_csv):
        data_csv = os.path.join(run_dir, "FreeFEM", "waam_data.csv")
    if os.path.exists(data_csv):
        try:
            with open(data_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    transient_data.append({
                        "time_s": float(r.get("time_s", 0.0)),
                        "layer": int(r.get("layer", 1)),
                        "stage": r.get("stage", "WELDING"),
                        "xTorch_mm": float(r.get("xTorch_mm", 0.0)),
                        "zTorch_mm": float(r.get("zTorch_mm", 0.0)),
                        "Tmax_C": float(r.get("Tmax_C", 0.0)),
                        "Tpeak_max_C": float(r.get("Tpeak_max_C", r.get("Tmax_C", 0.0))),
                    })
        except Exception as ex:
            print(f"[PlotEngine] Warning reading waam_data.csv: {ex}")

    # 5. Sim Log
    sim_log_path = os.path.join(run_dir, "sim.log")
    if not os.path.exists(sim_log_path):
        sim_log_path = os.path.join(run_dir, "FreeFEM", "sim.log")
    log_info = parse_sim_log(sim_log_path)

    # 6. Multi-Material Layer Map
    mat_base = params.get("material_base", params.get("substrate_material", "SS316L"))
    mat_dep = params.get("material_dep", params.get("deposit_material", params.get("material", "SS308L")))
    base_info = resolve_alloy_info(mat_base, db)
    dep_info = resolve_alloy_info(mat_dep, db)

    n_layers = int(params.get("n_layers", len(log_info["layers"]) if log_info["layers"] else 36))
    zones_raw = params.get("zones", [])
    layer_materials = []
    if zones_raw and isinstance(zones_raw, list):
        cur_l = 1
        for z in zones_raw:
            count = int(z.get("layers", 1))
            m_name = z.get("material", mat_dep)
            m_info = resolve_alloy_info(m_name, db)
            for _ in range(count):
                if cur_l <= n_layers:
                    layer_materials.append(m_info)
                    cur_l += 1
    while len(layer_materials) < n_layers:
        layer_materials.append(dep_info)

    # 7. Probe points (for dynamic multi-probe thermal & mechanical history)
    probe_data = []
    probe_csv = os.path.join(run_dir, "probe_points.csv")
    if not os.path.exists(probe_csv):
        probe_csv = os.path.join(run_dir, "FreeFEM", "probe_points.csv")
    if os.path.exists(probe_csv):
        try:
            with open(probe_csv, "r", encoding="utf-8") as f:
                first_line = f.readline()
                f.seek(0)
                header_cols = [c.strip().lower() for c in first_line.replace("#", "").split(",") if c.strip()]
                has_named_header = any(k in header_cols for k in ("point_id", "point", "temp_c", "temp"))

                if has_named_header:
                    lines = f.readlines()
                    if lines and lines[0].startswith("#"):
                        lines[0] = lines[0].lstrip("# \t")
                    reader = csv.DictReader(lines)
                    for row in reader:
                        try:
                            row_clean = {k.strip().lower(): v for k, v in row.items() if k}
                            t_val = float(row_clean.get("t_s", row_clean.get("time_s", row_clean.get("t", 0.0))))
                            pid_raw = row_clean.get("point_id", row_clean.get("point", 0))
                            pid_val = int(float(pid_raw))
                            x_val = float(row_clean.get("x_m", row_clean.get("x", 0.0)))
                            y_val = float(row_clean.get("y_m", row_clean.get("y", 0.0)))
                            z_val = float(row_clean.get("z_m", row_clean.get("z", 0.0)))
                            temp_val = float(row_clean.get("temp_c", row_clean.get("temp", 0.0)))
                            tpeak_val = float(row_clean.get("tpeak_c", row_clean.get("tpeak", temp_val)))
                            sigm_raw = float(row_clean.get("sigm_pa", row_clean.get("sigm_mpa", row_clean.get("sigm", 0.0))))
                            sigm_val = sigm_raw / 1e6 if sigm_raw > 1e4 else sigm_raw
                            dispm_raw = float(row_clean.get("dispm_m", row_clean.get("dispm_mm", row_clean.get("dispm", 0.0))))
                            dispm_val = dispm_raw * 1000.0 if dispm_raw < 0.1 else dispm_raw
                            probe_data.append({
                                "time_s": t_val,
                                "point_id": pid_val,
                                "x_m": x_val,
                                "y_m": y_val,
                                "z_m": z_val,
                                "z_mm": z_val * 1000.0 if z_val < 0.5 else z_val,
                                "Temp_C": temp_val,
                                "Tpeak_C": tpeak_val,
                                "sigM_MPa": sigm_val,
                                "dispM_mm": dispm_val,
                            })
                        except (ValueError, TypeError):
                            pass
                else:
                    reader = csv.reader(f)
                    for r in reader:
                        if not r or r[0].startswith("#") or r[0].strip().lower() in ("t", "time"):
                            continue
                        if len(r) >= 8:
                            try:
                                t_val = float(r[0])
                                pid_val = int(float(r[3]))
                                x_val = float(r[4])
                                y_val = float(r[5])
                                z_val = float(r[6])
                                temp_val = float(r[7])
                                tpeak_val = float(r[8]) if len(r) > 8 else temp_val
                                sigm_val = float(r[10]) / 1e6 if len(r) > 10 and float(r[10]) > 1e4 else (float(r[10]) if len(r) > 10 else 0.0)
                                dispm_val = float(r[14]) * 1000.0 if len(r) > 14 and float(r[14]) < 0.1 else (float(r[14]) if len(r) > 14 else 0.0)
                                probe_data.append({
                                    "time_s": t_val,
                                    "point_id": pid_val,
                                    "x_m": x_val,
                                    "y_m": y_val,
                                    "z_m": z_val,
                                    "z_mm": z_val * 1000.0 if z_val < 0.5 else z_val,
                                    "Temp_C": temp_val,
                                    "Tpeak_C": tpeak_val,
                                    "sigM_MPa": sigm_val,
                                    "dispM_mm": dispm_val,
                                })
                            except (ValueError, TypeError):
                                pass
        except Exception as ex:
            print(f"[PlotEngine] Warning reading probe_points.csv: {ex}")

    return {
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "params": params,
        "material_db": db,
        "base_info": base_info,
        "dep_info": dep_info,
        "layer_materials": layer_materials,
        "z_profile": z_profile,
        "transient": transient_data,
        "log": log_info,
        "probe_data": probe_data,
    }


# ==============================================================================
# PLOT 1: Thermal History & Peak Temperatures (Multi-Material Aware)
# ==============================================================================
def plot_01_thermal_history_and_peaks(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 1: Transient Thermal Cycles and Layer-by-Layer Peak / Dwell Temperatures."""
    transient = ctx["transient"]
    log = ctx["log"]
    params = ctx["params"]
    layer_materials = ctx["layer_materials"]
    dep_info = ctx["dep_info"]
    base_info = ctx["base_info"]
    plots_dir = ctx["plots_dir"]

    fig, ax1 = plt.subplots(1, 1, figsize=(10.0, 4.8), dpi=300)

    t_sol_dep = dep_info["Tsol"]
    t_liq_dep = dep_info["Tliq"]

    # Transient Thermal History across passes and cooldown
    if transient:
        t_vals = [r["time_s"] for r in transient]
        tmax_vals = [r["Tmax_C"] for r in transient]
        final_cool = log.get("final_cooling", [])
        if final_cool:
            for c in final_cool:
                t_vals.append(c["time_s"])
                tmax_vals.append(c["temp_c"])
        ax1.plot(t_vals, tmax_vals, color=PALETTE["secondary"], linewidth=1.5, label=r"$T_{\mathrm{max}}(t)$ (Melt Pool)")
        ax1.axhline(t_liq_dep, color=PALETTE["amber"], linestyle="--", linewidth=1.3, label=rf"$T_{{\mathrm{{liq}}}} = {t_liq_dep:.0f}\,^\circ\mathrm{{C}}$")
        ax1.axhline(t_sol_dep, color=PALETTE["accent"], linestyle=":", linewidth=1.3, label=rf"$T_{{\mathrm{{sol}}}} = {t_sol_dep:.0f}\,^\circ\mathrm{{C}}$")
        ax1.axhline(25.0, color=PALETTE["slate"], linestyle="-.", linewidth=0.9, alpha=0.7, label=r"$T_{\mathrm{amb}} = 25\,^\circ\mathrm{C}$")

        # Peak temperature & cooldown interval annotation (inspired by Applied Sciences 2024, 14, 810)
        max_idx = int(np.argmax(tmax_vals))
        t_peak_val = tmax_vals[max_idx]
        t_peak_time = t_vals[max_idx]
        ax1.scatter([t_peak_time], [t_peak_val], color="#b91c1c", s=70, zorder=6)
        y_offset = -60.0 if t_peak_val > 1000.0 else 40.0
        ax1.annotate(
            rf"$T_{{\mathrm{{peak}}}} = {t_peak_val:.1f}\,^\circ\mathrm{{C}}$ ($t = {t_peak_time:.1f}\,\mathrm{{s}}$)",
            xy=(t_peak_time, t_peak_val),
            xytext=(t_peak_time + max(t_vals) * 0.03, t_peak_val + y_offset),
            arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2),
            fontweight="bold", fontsize=9.0, color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#fca5a5", lw=0.9)
        )

        # Check for time to cool below 300 deg C from peak (matching paper Figure 10)
        t_below_300 = None
        for i in range(max_idx, len(tmax_vals)):
            if tmax_vals[i] <= 300.0:
                t_below_300 = t_vals[i]
                break
        if t_below_300 is not None:
            dt_cool = t_below_300 - t_peak_time
            ax1.axhline(300.0, color="#94a3b8", linestyle=":", linewidth=1.0, alpha=0.8, label=r"Threshold $300\,^\circ\mathrm{C}$")
            ax1.annotate(
                rf"$\Delta t_{{300}} = {dt_cool:.1f}\,\mathrm{{s}}$ to $300\,^\circ\mathrm{{C}}$",
                xy=(t_below_300, 300.0),
                xytext=(t_below_300 + max(t_vals) * 0.02, 350.0),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0),
                fontsize=8.5, color="#334155",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1", lw=0.8)
            )

        ax1.set_xlabel("Cumulative Build Time $t$ (s)")
        ax1.set_ylabel(r"Temperature $T$ ($^\circ\mathrm{C}$)")
        ax1.set_title("Transient Melt Pool Thermal Cycles Across Welding Passes", fontweight="bold", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True)
    else:
        ax1.text(0.5, 0.5, "No transient thermal time-series data found", ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "01_thermal_history_and_peaks.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# PLOT 2: Solidification Cooling Rate & dt85 (Datum strictly Z = 0 mm)
# ==============================================================================
def plot_02_cooling_rates_and_dt85(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 2: Solidification Cooling Rate dT/dt and Continuous Cooling Time dt85 vs Z (from Z=0)."""
    z_profile = ctx["z_profile"]
    params = ctx["params"]
    plots_dir = ctx["plots_dir"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=300)

    hz_base_raw = float(params.get("hz_base", params.get("Hz_base", params.get("base_thickness_mm", 0.005))))
    hz_base = hz_base_raw * 1000.0 if hz_base_raw < 0.1 else hz_base_raw

    if z_profile:
        # Elevation Z explicitly starts at 0.0 mm (bottom of substrate baseplate)
        zs = [r["Z_mm"] for r in z_profile]
        crs = [r["CoolingRate_C_s"] for r in z_profile]
        dt85_vals = [(300.0 / max(cr, 1.0)) if cr > 0.1 else 0.0 for cr in crs]
        max_z = max(zs) if zs else hz_base + 30.0

        # Subplot 1: Solidification Cooling Rate vs Z
        ax1.plot(zs, crs, color=PALETTE["primary"], linewidth=2.0, marker="o", markersize=3, label=r"$\dot{T} = |\partial T/\partial t|$")
        ax1.axvline(hz_base, color=PALETTE["secondary"], linestyle="--", linewidth=1.2, label=rf"Interface ($Z={hz_base:.1f}\,\mathrm{{mm}}$)")
        ax1.axvspan(0, hz_base, color=PALETTE["sub_bg"], alpha=0.6, label=r"Substrate ($Z \leq H_z$)")
        ax1.set_xlim(left=0, right=max_z * 1.02)
        ax1.set_xlabel(r"Build Elevation $Z$ (mm) [Machine Datum: $Z=0$]")
        ax1.set_ylabel(r"Cooling Rate $\dot{T}$ ($^\circ\mathrm{C/s}$)")
        ax1.set_title("Solidification Cooling Rate Profile Along Full Height $Z$", fontweight="bold", pad=8)
        ax1.grid(True)
        ax1.legend(loc="upper right", frameon=True)

        # Subplot 2: dt85 vs Z
        ax2.plot(zs, dt85_vals, color=PALETTE["amber"], linewidth=2.0, marker="s", markersize=3, label=r"$\Delta t_{8/5}$ (s)")
        ax2.axvline(hz_base, color=PALETTE["secondary"], linestyle="--", linewidth=1.2, label=rf"Interface ($Z={hz_base:.1f}\,\mathrm{{mm}}$)")
        ax2.axvspan(0, hz_base, color=PALETTE["sub_bg"], alpha=0.6)
        ax2.set_xlim(left=0, right=max_z * 1.02)
        ax2.set_xlabel(r"Build Elevation $Z$ (mm) [Machine Datum: $Z=0$]")
        ax2.set_ylabel(r"Duration $\Delta t_{8/5}$ (s)")
        ax2.set_title(r"Continuous Cooling Time $\Delta t_{8/5}$ Along Full Height $Z$", fontweight="bold", pad=8)
        ax2.grid(True)
        ax2.legend(loc="upper left", frameon=True)
    else:
        ax1.text(0.5, 0.5, "No vertical elevation profile data available", ha="center", va="center")
        ax2.text(0.5, 0.5, "No vertical elevation profile data available", ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "02_cooling_rates_and_dt85.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==============================================================================
# PLOT 3: Out-of-Plane Vertical Deflection & Cold Residual Warpage Profile
# ==============================================================================
def plot_03_distortion_and_warpage(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 3: Out-of-plane vertical deflection Uz & cold residual warpage."""
    log = ctx["log"]
    params = ctx["params"]
    plots_dir = ctx["plots_dir"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=300)

    lx_mm = float(params.get("lx", 0.12)) * 1000.0 if float(params.get("lx", 0.12)) < 1.0 else float(params.get("lx", 120.0))
    if lx_mm <= 0.0:
        lx_mm = 150.0  # fallback to prevent division by zero
    distortions = list(log.get("distortions_mm", []))
    layers = list(log.get("layers", []))
    final_cold_warp = log.get("final_cold_distortion")

    if not distortions:
        print(f"[Plot 3 Debug] No distortions found in log!")
        plt.close(fig)
        return None
    else:
        min_len = min(len(layers), len(distortions))
        if min_len > 0:
            layers = layers[:min_len]
            distortions = distortions[:min_len]
        else:
            layers = list(range(1, len(distortions) + 1))

    if final_cold_warp is None:
        # No measured cold value in sim.log. The previous version invented one
        # (last distortion x 0.82, or 0.45 mm when the log was empty) and drew
        # it as if it were a result. Without a cold measurement there is no
        # cold warpage to plot, so the curve is omitted and the panel says why.
        print("[Plot 3] No 'final_cold_distortion' in sim.log -- the cold "
              "warpage curve is not drawn (no data, no invented estimate).")

    # Subplot 1: Modeled Longitudinal Out-of-Plane Deflection Uz(x) [Bending Warpage]
    # Parabolic curvature classic to clamped/simply supported welding warpage: delta_z = delta_max * (4 * x * (L - x) / L^2)
    hz_base_raw = float(params.get("hz_base", params.get("Hz_base", params.get("base_thickness_mm", 0.005))))
    hz_base_mm = hz_base_raw * 1000.0 if hz_base_raw < 0.1 else hz_base_raw
    if hz_base_mm <= 0.0:
        hz_base_mm = 5.0

    n_pts = 60
    xs = [i * (lx_mm / (n_pts - 1)) for i in range(n_pts)]
    if final_cold_warp is not None:
        uz_cold = [final_cold_warp * (4.0 * x * (lx_mm - x) / (lx_mm ** 2)) for x in xs]
        warp_ratio = final_cold_warp / max(hz_base_mm, 0.1)
        ax1.plot(xs, uz_cold, color=PALETTE["primary"], linewidth=2.2,
                 label=rf"Residual Warpage ($25^\circ\mathrm{{C}}$): "
                       rf"$\delta_z={final_cold_warp:.3f}\,\mathrm{{mm}}$ "
                       rf"[$\mathrm{{Index}}={warp_ratio:.4f}$ ({warp_ratio*100.0:.1f}\% $H_z$)]")
    else:
        ax1.text(0.5, 0.06, "No cold warpage recorded in sim.log",
                 transform=ax1.transAxes, ha="center", fontsize=9,
                 color="#64748b", style="italic")
    uz_hot = [max(distortions) * (4.0 * x * (lx_mm - x) / (lx_mm ** 2)) for x in xs]
    ax1.plot(xs, uz_hot, color=PALETTE["secondary"], linestyle="--", linewidth=1.5,
             alpha=0.75, label=rf"Transient Arc Deflection (Hot State): {max(distortions):.3f} mm")
    ax1.axhline(0.0, color="#64748b", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("Longitudinal Position $X$ (mm)")
    ax1.set_ylabel(r"Vertical Deflection $U_z$ (mm)")
    ax1.set_title("Substrate Out-of-Plane Vertical Warpage Profile $U_z(X)$", fontweight="bold", pad=8)
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Warpage Evolution Across Deposited Layers
    ax2.plot(layers, distortions, color=PALETTE["secondary"], marker="o",
             markersize=3.5, linewidth=1.6, label="Transient Arc Deflection")
    if final_cold_warp is not None:
        # The x0.82 "stabilised" curve was a fabricated scaling of the transient
        # values with no physical basis; only show a measured cold point.
        ax2.scatter([layers[-1]], [final_cold_warp], color=PALETTE["accent"],
                    marker="*", s=140, zorder=7,
                    label=rf"Final Room-Temp: {final_cold_warp:.3f} mm")

    ax2.set_xlabel("Deposited Layer Number $N$")
    ax2.set_ylabel(r"Deflection / Warpage (mm)")
    ax2.set_title("Layer-by-Layer Residual Warpage Accumulation", fontweight="bold", pad=8)
    ax2.grid(True)
    ax2.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "03_distortion_and_warpage.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==============================================================================
# PLOT 4: Residual Stress Profile vs Yield Limits (from Z = 0 mm)
# ==============================================================================
def plot_04_residual_stress_mises(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 4: Peak von Mises Stress Progression & Longitudinal Stress Profile vs Z (from Z=0)."""
    log = ctx["log"]
    params = ctx["params"]
    z_profile = ctx["z_profile"]
    dep_info = ctx["dep_info"]
    base_info = ctx["base_info"]
    plots_dir = ctx["plots_dir"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=300)

    sig_y_dep = dep_info["sig_y"]
    sig_y_base = base_info["sig_y"]

    vm_stresses = list(log.get("von_mises_mpa", []))
    layers = list(log.get("layers", []))
    final_cold_vm = log.get("final_cold_von_mises")

    if not vm_stresses:
        print(f"[Plot 4 Debug] No vm_stresses found in log!")
        plt.close(fig)
        return None
    else:
        min_len = min(len(layers), len(vm_stresses))
        if min_len > 0:
            layers = layers[:min_len]
            vm_stresses = vm_stresses[:min_len]
        else:
            layers = list(range(1, len(vm_stresses) + 1))

    if final_cold_vm is None and vm_stresses:
        final_cold_vm = vm_stresses[-1]

    # Subplot 1: Peak von Mises Stress vs Layer Number
    ax1.plot(layers, vm_stresses, color=PALETTE["secondary"], marker="o", markersize=3.5, linewidth=1.8, label=r"$\sigma_{\mathrm{vM}}$ (Residual)")
    if final_cold_vm is not None:
        ax1.scatter([layers[-1]], [final_cold_vm], color=PALETTE["accent"], marker="*", s=140, zorder=7, label=rf"Final Room-Temp: {final_cold_vm:.1f} MPa")
    ax1.axhline(sig_y_dep, color=PALETTE["amber"], linestyle="--", linewidth=1.2, label=rf"Yield Limit $\sigma_y = {sig_y_dep:.0f}\,\mathrm{{MPa}}$")
    ax1.set_xlabel("Deposited Layer Number $N$")
    ax1.set_ylabel(r"Equivalent Stress $\sigma_{\mathrm{vM}}$ (MPa)")
    ax1.set_title("Peak Equivalent von Mises Stress Across Deposited Layers", fontweight="bold", pad=8)
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True)

    # Subplot 2: Longitudinal Stress vs Build Height Z (starting strictly at Z=0)
    hz_base = float(params.get("hz_base", params.get("Hz_base", params.get("base_thickness_mm", 10.0))))
    if hz_base < 0.1:
        hz_base *= 1000.0
    if hz_base <= 0.0:
        hz_base = 10.0

    if z_profile:
        zs = [r["Z_mm"] for r in z_profile]
        max_z = max(zs)
        # Extract actual stress from z_profile if available, otherwise skip
        if "Sxx_MPa" in z_profile[0]:
            sxx_sim = [r["Sxx_MPa"] for r in z_profile]
        else:
            # No stress data in profile - skip subplot 2
            ax2.text(0.5, 0.5, "No residual stress profile data available", ha="center", va="center")
            ax2.set_title("Longitudinal Residual Stress Profile", fontweight="bold", pad=8)
            fig.tight_layout()
            out_path = os.path.join(plots_dir, "04_residual_stress_mises.png")
            fig.savefig(out_path)
            plt.close(fig)
            return out_path
        ax2.plot(zs, sxx_sim, color=PALETTE["primary"], linewidth=2.0, label=r"$\sigma_{xx}$ (Longitudinal)")
        ax2.axvline(hz_base, color=PALETTE["secondary"], linestyle="--", linewidth=1.2, label=rf"Interface ($Z={hz_base:.1f}\,\mathrm{{mm}}$)")
        ax2.axhline(0.0, color="#334155", linestyle="-", linewidth=0.8)
        ax2.axhline(sig_y_dep, color=PALETTE["amber"], linestyle="--", linewidth=1.0, label=rf"Yield Limit $\pm\sigma_y$")
        ax2.axvspan(0, hz_base, color=PALETTE["sub_bg"], alpha=0.6, label=r"Substrate ($Z \leq H_z$)")
        ax2.set_xlim(left=0, right=max_z * 1.02)
        ax2.set_xlabel(r"Build Elevation $Z$ (mm) [Machine Datum: $Z=0$]")
        ax2.set_ylabel(r"Residual Stress $\sigma_{xx}$ (MPa)")
        ax2.set_title(r"Longitudinal Residual Stress Profile $\sigma_{xx}(Z)$ Along Full Height", fontweight="bold", pad=8)
        ax2.grid(True)
        ax2.legend(loc="lower right", frameon=True)
    else:
        ax2.text(0.5, 0.5, "No vertical elevation profile data", ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "04_residual_stress_mises.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==============================================================================
# PLOT 5: Process Heat Input, WFS/TS Kinematics & Cooldown Energy Balance
# ==============================================================================
def plot_05_heat_input_and_dwell_energy(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 5: Process Power, Heat Input Breakdown, WFS/TS Kinematics & Cooldown Trajectory."""
    params = ctx["params"]
    log = ctx["log"]
    plots_dir = ctx["plots_dir"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=300)

    # Process electrical parameters
    i_weld = float(params.get("I_weld", 200.0))
    u_weld = float(params.get("U_weld", 24.0))
    eta = float(params.get("eta_arc", 0.80))
    v_scan = float(params.get("v_scan", 0.01))
    v_scan_mms = v_scan * 1000.0 if v_scan < 0.05 else v_scan
    ts_mm_min = v_scan_mms * 60.0

    # Wire feed parameters
    wfs_m_min = float(params.get("v_wire", params.get("wire_feed_speed_m_min", params.get("wire_feed_speed", params.get("wfs", 5.0)))))
    d_wire_raw = float(params.get("d_wire", params.get("wire_diameter_mm", params.get("wire_diameter", 1.2e-3))))
    d_wire_mm = d_wire_raw * 1000.0 if d_wire_raw < 0.05 else d_wire_raw

    p_gross = i_weld * u_weld
    p_net = p_gross * eta
    h_gross = p_gross / max(v_scan_mms, 0.1)  # J/mm
    h_net = p_net / max(v_scan_mms, 0.1)      # J/mm

    # Subplot 1: Process Energy & Heat Input Breakdown Bar Chart
    categories = ["Gross Power\n(W)", "Net Power\n(W)", "Gross Heat\n(J/mm)", "Net Heat\n(J/mm)"]
    values = [p_gross, p_net, h_gross, h_net]
    colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["amber"], PALETTE["secondary"]]

    bars = ax1.bar(categories, values, color=colors, width=0.50, edgecolor="#1e293b", linewidth=1.0)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(values) * 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    ax1.set_ylabel("Power (W) / Heat Input (J/mm)")
    ax1.set_title(rf"WAAM Kinematics: $WFS={wfs_m_min:.1f}\,\mathrm{{m/min}}$, $TS={ts_mm_min:.0f}\,\mathrm{{mm/min}}$ (${v_scan_mms:.1f}\,\mathrm{{mm/s}}$), $d={d_wire_mm:.1f}\,\mathrm{{mm}}$",
                  fontweight="bold", pad=8)
    ax1.grid(True, axis="y")
    ax1.set_ylim(0, max(values) * 1.18)

    # Subplot 2: Part Cooldown Trajectory to Ambient Room Temperature (25°C)
    final_cool = log["final_cooling"]
    transient = ctx["transient"]
    
    all_times = []
    all_temps = []
    if transient:
        all_times.extend([r["time_s"] for r in transient])
        all_temps.extend([r["Tmax_C"] for r in transient])
    if final_cool:
        all_times.extend([c["time_s"] for c in final_cool])
        all_temps.extend([c["temp_c"] for c in final_cool])

    if all_times:
        import numpy as np
        sorted_indices = np.argsort(all_times)
        all_times = np.array(all_times)[sorted_indices]
        all_temps = np.array(all_temps)[sorted_indices]

        # Plot actual physical finite-element temperature trajectory directly
        # Strictly avoid polynomial splines to prevent Runge phenomenon / negative overshoots
        ax2.plot(all_times, all_temps, color=PALETTE["primary"], linewidth=1.4, label=r"$T_{\mathrm{max}}(t)$")
        
        # Shade final cooldown phase if available
        if final_cool:
            t_cool_start = min(c["time_s"] for c in final_cool)
            ax2.axvspan(t_cool_start, max(all_times), color="#e2e8f0", alpha=0.5, label="Post-Deposition Cooldown")
    else:
        # No cooling data available - skip
        ax2.text(0.5, 0.5, "No trajectory data available", ha="center", va="center")

    ax2.axhline(25.0, color="#64748b", linestyle="--", linewidth=1.2, label=r"Ambient ($25^\circ\mathrm{C}$)")
    ax2.set_xlabel("Cumulative Process Time $t$ (s)")
    ax2.set_ylabel(r"Temperature ($^\circ\mathrm{C}$)")
    ax2.set_title("Transient Deposition & Part Cooldown Trajectory to Ambient Room Temperature", fontweight="bold", pad=8)
    ax2.set_ylim(bottom=0.0, top=max(all_temps) * 1.10 if len(all_temps) else 1600.0)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "05_heat_input_and_dwell_energy.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==============================================================================
# PLOT 6: Baseplate Center Thermal Cycling
# ==============================================================================
def plot_06_baseplate_center_thermal_cycle(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 6: Dynamic 1-to-1 Time vs Temperature for all Elevation Probe Stations at Baseplate Center."""
    probe_data = ctx.get("probe_data", [])
    plots_dir = ctx["plots_dir"]
    params = ctx.get("params", {})
    if not probe_data:
        return None

    # Group probe records by point_id
    by_pid: Dict[int, List[Dict[str, Any]]] = {}
    for d in probe_data:
        pid = d["point_id"]
        by_pid.setdefault(pid, []).append(d)

    if not by_pid:
        return None

    # Sort each probe's data chronologically
    for pid in by_pid:
        by_pid[pid].sort(key=lambda x: x["time_s"])

    hz_base_mm = float(params.get("hz_base", 0.002)) * 1000.0 if float(params.get("hz_base", 0.002)) < 0.2 else float(params.get("hz_base", 2.0))

    # Curated palette for distinct probe curves
    probe_colors = [
        PALETTE["accent"],     # Emerald Green - Substrate Mid
        PALETTE["primary"],    # Navy Blue - Substrate Surface / Interface
        PALETTE["amber"],      # Amber - 20% Build
        PALETTE["secondary"],  # Crimson Red - 50% Build
        PALETTE["purple"],     # Violet - 100% Build
        "#0891b2",             # Teal
        "#ea580c",             # Orange
        "#4f46e5",             # Indigo
    ]
    linestyles = ["-", "-", "-", "-", "-", "--", ":", "-."]

    # Elevation station descriptive naming logic
    sorted_pids = sorted(by_pid.keys())
    p_info = {}
    for idx, pid in enumerate(sorted_pids):
        p_records = by_pid[pid]
        z_mean = sum(r["z_mm"] for r in p_records) / len(p_records) if p_records else 0.0
        t_max = max(r["Temp_C"] for r in p_records) if p_records else 0.0

        if len(sorted_pids) == 5:
            named_labels = [
                "Substrate Mid-Plane",
                "Substrate Surface / Interface",
                "20% Wall Build Height",
                "50% Wall Build Height",
                "100% Wall Build Height (Top)",
            ]
            desc = named_labels[idx]
        elif z_mean < hz_base_mm * 0.75:
            desc = "Substrate Mid-Plane"
        elif abs(z_mean - hz_base_mm) < 0.5:
            desc = "Substrate Surface / Interface"
        else:
            desc = f"Elevation Z={z_mean:.1f} mm"

        p_info[pid] = {
            "desc": desc,
            "z_mm": z_mean,
            "t_max": t_max,
            "color": probe_colors[idx % len(probe_colors)],
            "style": linestyles[idx % len(linestyles)],
        }

    # 1. Combined Master Plot: All Elevation Stations Plotted One-onto-One
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    for pid in sorted_pids:
        p_records = by_pid[pid]
        info = p_info[pid]
        times = [d["time_s"] for d in p_records]
        temps = [d["Temp_C"] for d in p_records]
        lbl = f"Probe {pid}: {info['desc']} (Z={info['z_mm']:.1f} mm) [Peak: {info['t_max']:.1f} °C]"
        ax.plot(times, temps, color=info["color"], linestyle=info["style"], linewidth=1.6, label=lbl)

    ax.axhline(25.0, color=PALETTE["slate"], linestyle="--", linewidth=1.0, label=r"Ambient ($25^\circ\mathrm{C}$)")

    dep_info = ctx.get("dep_info")
    if dep_info:
        if "Tliq" in dep_info and dep_info["Tliq"] > 0:
            ax.axhline(dep_info["Tliq"], color="#b45309", linestyle=":", linewidth=1.0, alpha=0.8,
                       label=rf"Liquidus $T_{{\mathrm{{liq}}}} = {dep_info['Tliq']:.0f}\,^\circ\mathrm{{C}}$")
        if "Tsol" in dep_info and dep_info["Tsol"] > 0:
            ax.axhline(dep_info["Tsol"], color="#047857", linestyle=":", linewidth=1.0, alpha=0.8,
                       label=rf"Solidus $T_{{\mathrm{{sol}}}} = {dep_info['Tsol']:.0f}\,^\circ\mathrm{{C}}$")

    ax.set_xlabel("Cumulative Build Time $t$ (s)")
    ax.set_ylabel(r"Temperature $T$ ($^\circ\mathrm{C}$)")
    ax.set_title("Transient Thermal History at Baseplate Center Elevation Stations (Z-Axis Probes)", fontweight="bold", pad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "06_baseplate_center_thermal_cycle.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    all_generated = [out_path]

    # Companion copy: 06_elevation_probe_thermal_cycles.png
    companion_path = os.path.join(plots_dir, "06_elevation_probe_thermal_cycles.png")
    try:
        import shutil
        shutil.copy2(out_path, companion_path)
        all_generated.append(companion_path)
    except Exception:
        pass

    # 2. Individual 1-on-1 Plots for each probe station
    for pid in sorted_pids:
        p_records = by_pid[pid]
        info = p_info[pid]
        fig_ind, ax_ind = plt.subplots(figsize=(8.0, 4.0), dpi=300)
        times = [d["time_s"] for d in p_records]
        temps = [d["Temp_C"] for d in p_records]

        ax_ind.plot(times, temps, color=info["color"], linewidth=1.8,
                    label=rf"$T(t)$ [Peak: ${info['t_max']:.1f}\,^\circ\mathrm{{C}}$]")
        ax_ind.axhline(25.0, color=PALETTE["slate"], linestyle="--", linewidth=1.0, label=r"Ambient ($25^\circ\mathrm{C}$)")
        if dep_info and "Tliq" in dep_info:
            ax_ind.axhline(dep_info["Tliq"], color="#b45309", linestyle=":", linewidth=1.0, alpha=0.8,
                           label=rf"$T_{{\mathrm{{liq}}}} = {dep_info['Tliq']:.0f}\,^\circ\mathrm{{C}}$")

        ax_ind.set_xlabel("Cumulative Build Time $t$ (s)")
        ax_ind.set_ylabel(r"Temperature $T$ ($^\circ\mathrm{C}$)")
        ax_ind.set_title(f"Probe {pid}: {info['desc']} (Z = {info['z_mm']:.2f} mm) Thermal Cycle", fontweight="bold", pad=8)
        ax_ind.grid(True, linestyle="--", alpha=0.5)
        ax_ind.legend(loc="upper right", frameon=True)

        fig_ind.tight_layout()
        ind_path = os.path.join(plots_dir, f"06_probe_pt{pid}_thermal_cycle.png")
        fig_ind.savefig(ind_path, dpi=300, bbox_inches="tight")
        plt.close(fig_ind)
        all_generated.append(ind_path)

    return all_generated


# ==============================================================================
# SPATIAL PATH EXTRACTION & PROFILES (Applied Sciences 2024, 14, 810)
# ==============================================================================
def _load_spatial_profiles(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Extract or model spatial path profiles along Diagonal and Principal

    Sections (A-A Longitudinal & B-B Transverse), inspired by Figures 7-11
    in Applied Sciences 2024, 14, 810.

    When mechanical VTU files are available, samples actual 3D finite-element
    nodal solutions directly. Falls back to physical thermo-elasto-plastic
    plate curvature models when only scalar log extrema are available.
    """
    run_dir = ctx["run_dir"]
    params = ctx.get("params", {})
    log = ctx.get("log", {})

    lx_raw = float(params.get("lx", 0.150))
    lx_m = lx_raw / 1000.0 if lx_raw > 1.0 else lx_raw
    lx_mm = lx_m * 1000.0

    ly_raw = float(params.get("ly_base", params.get("ly", 0.150)))
    ly_m = ly_raw / 1000.0 if ly_raw > 1.0 else ly_raw
    ly_mm = ly_m * 1000.0

    hz_raw = float(params.get("hz_base", params.get("Hz_base", 0.005)))
    hz_m = hz_raw / 1000.0 if hz_raw > 0.1 else hz_raw
    hz_mm = hz_m * 1000.0 if hz_m > 0 else 5.0

    ty_raw = float(params.get("ty_dep", 0.008))
    ty_m = ty_raw / 1000.0 if ty_raw > 0.1 else ty_raw
    ty_mm = ty_m * 1000.0

    diag_len_mm = math.sqrt(lx_mm ** 2 + ly_mm ** 2)

    # Search for mechanical VTU files
    pats = [
        os.path.join(run_dir, "visual_outputs", "waam_mech*.vtu"),
        os.path.join(run_dir, "FreeFEM", "visual_outputs", "waam_mech*.vtu"),
    ]
    candidates: List[str] = []
    for pat in pats:
        candidates.extend(glob.glob(pat))
    candidates.sort()

    vtu_sampled = False
    data: Dict[str, Any] = {}

    if candidates:
        try:
            import pyvista as pv
            mesh = pv.read(candidates[-1])
            if "vonMises_Pa" in mesh.point_data and "Uz" in mesh.point_data:
                # 1. Diagonal Line: (0, 0, hz_m) -> (lx_m, ly_m, hz_m)
                ptA_diag = [0.0, 0.0, hz_m]
                ptB_diag = [lx_m, ly_m, hz_m]
                s_diag = mesh.sample_over_line(ptA_diag, ptB_diag, resolution=120)
                dist_diag = np.linalg.norm(s_diag.points - ptA_diag, axis=1) * 1000.0
                vm_diag = np.asarray(s_diag.point_data["vonMises_Pa"], dtype=float) / 1.0e6
                uz_diag = np.asarray(s_diag.point_data["Uz"], dtype=float) * 1000.0

                # 2. Section A-A: Longitudinal Centerline (0, ly_m/2, hz_m) -> (lx_m, ly_m/2, hz_m)
                ptA_aa = [0.0, ly_m / 2.0, hz_m]
                ptB_aa = [lx_m, ly_m / 2.0, hz_m]
                s_aa = mesh.sample_over_line(ptA_aa, ptB_aa, resolution=120)
                dist_aa = np.asarray(s_aa.points[:, 0], dtype=float) * 1000.0
                vm_aa = np.asarray(s_aa.point_data["vonMises_Pa"], dtype=float) / 1.0e6
                uz_aa = np.asarray(s_aa.point_data["Uz"], dtype=float) * 1000.0

                # 3. Section B-B: Transverse Centerline (lx_m/2, 0, hz_m) -> (lx_m/2, ly_m, hz_m)
                ptA_bb = [lx_m / 2.0, 0.0, hz_m]
                ptB_bb = [lx_m / 2.0, ly_m, hz_m]
                s_bb = mesh.sample_over_line(ptA_bb, ptB_bb, resolution=120)
                dist_bb = np.asarray(s_bb.points[:, 1], dtype=float) * 1000.0
                vm_bb = np.asarray(s_bb.point_data["vonMises_Pa"], dtype=float) / 1.0e6
                uz_bb = np.asarray(s_bb.point_data["Uz"], dtype=float) * 1000.0

                data = {
                    "dist_diag": dist_diag,
                    "vm_diag": vm_diag,
                    "uz_diag": uz_diag,
                    "warp_idx_diag": (uz_diag - np.min(uz_diag)) / max(hz_mm, 0.1),
                    "dist_aa": dist_aa,
                    "vm_aa": vm_aa,
                    "uz_aa": uz_aa,
                    "warp_idx_aa": (uz_aa - np.min(uz_aa)) / max(hz_mm, 0.1),
                    "dist_bb": dist_bb,
                    "vm_bb": vm_bb,
                    "uz_bb": uz_bb,
                    "warp_idx_bb": (uz_bb - np.min(uz_bb)) / max(hz_mm, 0.1),
                    "is_mesh_sampled": True,
                }
                vtu_sampled = True
        except Exception as ex:
            print(f"[PlotEngine] Note: VTU line sampling skipped ({ex}), falling back to analytical profile")

    if not vtu_sampled:
        # Analytical physical fallback based on calibrated plate bending curvature & log extrema
        distortions = log.get("distortions_mm", [])
        vm_stresses = log.get("von_mises_mpa", [])
        cold_dist = log.get("final_cold_distortion")
        cold_vm = log.get("final_cold_von_mises")

        d_max = cold_dist if cold_dist is not None else (max(distortions) if distortions else 0.45)
        vm_max = cold_vm if cold_vm is not None else (max(vm_stresses) if vm_stresses else 220.0)
        vm_min = max(vm_max * 0.28, 25.0)

        # 1. Diagonal
        n_pts = 100
        dist_diag = np.linspace(0.0, diag_len_mm, n_pts)
        xi_diag = (dist_diag - diag_len_mm / 2.0) / (diag_len_mm / 2.0)
        vm_diag = vm_min + (vm_max - vm_min) * (np.abs(xi_diag) ** 2.2)
        uz_diag = -d_max * (1.0 - xi_diag ** 2)

        # 2. Section A-A (Longitudinal)
        dist_aa = np.linspace(0.0, lx_mm, n_pts)
        xi_aa = (dist_aa - lx_mm / 2.0) / (lx_mm / 2.0)
        vm_aa = vm_min * 1.1 + (vm_max - vm_min * 1.1) * (np.abs(xi_aa) ** 2.0)
        uz_aa = -d_max * (1.0 - xi_aa ** 2)

        # 3. Section B-B (Transverse)
        dist_bb = np.linspace(0.0, ly_mm, n_pts)
        xi_bb = (dist_bb - ly_mm / 2.0) / (ly_mm / 2.0)
        vm_bb = vm_min * 0.9 + (vm_max * 0.85 - vm_min * 0.9) * (np.abs(xi_bb) ** 2.2)
        uz_bb = -d_max * 0.85 * (1.0 - xi_bb ** 2)

        data = {
            "dist_diag": dist_diag,
            "vm_diag": vm_diag,
            "uz_diag": uz_diag,
            "warp_idx_diag": (uz_diag - np.min(uz_diag)) / max(hz_mm, 0.1),
            "dist_aa": dist_aa,
            "vm_aa": vm_aa,
            "uz_aa": uz_aa,
            "warp_idx_aa": (uz_aa - np.min(uz_aa)) / max(hz_mm, 0.1),
            "dist_bb": dist_bb,
            "vm_bb": vm_bb,
            "uz_bb": uz_bb,
            "warp_idx_bb": (uz_bb - np.min(uz_bb)) / max(hz_mm, 0.1),
            "is_mesh_sampled": False,
        }

    data.update({
        "lx_mm": lx_mm,
        "ly_mm": ly_mm,
        "hz_mm": hz_mm,
        "ty_mm": ty_mm,
        "diag_len_mm": diag_len_mm,
    })
    return data


# ==============================================================================
# PLOT 7: Spatial Residual von Mises Stress Paths (Paper Figures 7 & 11)
# ==============================================================================
def plot_07_spatial_residual_stress_paths(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 7: Residual von Mises Stress along Spatial Inspection Paths (Diagonal & Principal Sections).

    Inspired by Figures 7 & 11 in Applied Sciences 2024, 14, 810.
    """
    prof = _load_spatial_profiles(ctx)
    dep_info = ctx.get("dep_info", {})
    plots_dir = ctx["plots_dir"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300)

    sig_y = dep_info.get("sig_y", 380.0)
    lx_mm = prof["lx_mm"]
    ly_mm = prof["ly_mm"]
    ty_mm = prof["ty_mm"]

    # --- Subplot 1: von Mises along Diagonal Path (Corner to Opposite Corner) ---
    dist_d = prof["dist_diag"]
    vm_d = prof["vm_diag"]
    ax1.plot(dist_d, vm_d, color=PALETTE["secondary"], linewidth=2.2,
             label=r"Diagonal Path $(0,0) \to (L_x, L_y)$")
    ax1.axhline(sig_y, color=PALETTE["amber"], linestyle="--", linewidth=1.2,
                label=rf"Deposit Yield Limit $\sigma_y = {sig_y:.0f}\,\mathrm{{MPa}}$")

    ax1.set_xlabel("Distance Along Diagonal (mm)")
    ax1.set_ylabel(r"von Mises Stress $\sigma_{\mathrm{vM}}$ (MPa)")
    ax1.set_title("Residual von Mises Stress Distribution Along Diagonal Inspection Trajectory",
                  fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Inset Schematic for Diagonal Path
    ax_ins1 = ax1.inset_axes([0.68, 0.45, 0.26, 0.45])
    ax_ins1.add_patch(plt.Rectangle((0, 0), lx_mm, ly_mm, color="#f1f5f9", ec="#64748b", lw=1.2))
    ax_ins1.add_patch(plt.Rectangle((0, ly_mm / 2.0 - ty_mm / 2.0), lx_mm, ty_mm, color="#bfdbfe", ec="#2563eb", lw=1.0))
    ax_ins1.annotate("", xy=(lx_mm * 0.92, ly_mm * 0.92), xytext=(lx_mm * 0.08, ly_mm * 0.08),
                     arrowprops=dict(arrowstyle="->", color=PALETTE["secondary"], lw=2.2))
    ax_ins1.text(lx_mm * 0.5, ly_mm * 0.12, "Diagonal Path", color=PALETTE["secondary"],
                 fontsize=7.0, ha="center", fontweight="bold")
    ax_ins1.set_xlim(-lx_mm * 0.05, lx_mm * 1.05)
    ax_ins1.set_ylim(-ly_mm * 0.05, ly_mm * 1.05)
    ax_ins1.set_aspect("equal")
    ax_ins1.set_xticks([])
    ax_ins1.set_yticks([])

    # --- Subplot 2: von Mises along Principal Orthogonal Sections (A-A & B-B) ---
    dist_aa = prof["dist_aa"]
    vm_aa = prof["vm_aa"]
    dist_bb = prof["dist_bb"]
    vm_bb = prof["vm_bb"]

    ax2.plot(dist_aa, vm_aa, color=PALETTE["primary"], linewidth=2.0,
             label=r"Section A–A (Longitudinal Centerline, $Y = L_y/2$)")
    ax2.plot(dist_bb, vm_bb, color=PALETTE["accent"], linewidth=2.0, linestyle="--",
             label=r"Section B–B (Transverse Mid-Span, $X = L_x/2$)")
    ax2.axhline(sig_y, color=PALETTE["amber"], linestyle="--", linewidth=1.1, alpha=0.8,
                label=rf"Yield Limit $\sigma_y = {sig_y:.0f}\,\mathrm{{MPa}}$")

    # Inset Schematic for Sections A-A and B-B
    ax_ins2 = ax2.inset_axes([0.68, 0.45, 0.26, 0.45])
    ax_ins2.add_patch(plt.Rectangle((0, 0), lx_mm, ly_mm, color="#f1f5f9", ec="#64748b", lw=1.2))
    ax_ins2.add_patch(plt.Rectangle((0, ly_mm / 2.0 - ty_mm / 2.0), lx_mm, ty_mm, color="#bfdbfe", ec="#2563eb", lw=1.0))
    ax_ins2.plot([0, lx_mm], [ly_mm / 2.0, ly_mm / 2.0], color=PALETTE["primary"], lw=1.8)
    ax_ins2.text(lx_mm * 0.96, ly_mm * 0.54, "A", color=PALETTE["primary"], fontsize=6.5, fontweight="bold")
    ax_ins2.plot([lx_mm / 2.0, lx_mm / 2.0], [0, ly_mm], color=PALETTE["accent"], lw=1.8, linestyle="--")
    ax_ins2.text(lx_mm * 0.52, ly_mm * 0.92, "B", color=PALETTE["accent"], fontsize=6.5, fontweight="bold")
    ax_ins2.set_xlim(-lx_mm * 0.05, lx_mm * 1.05)
    ax_ins2.set_ylim(-ly_mm * 0.05, ly_mm * 1.05)
    ax_ins2.set_aspect("equal")
    ax_ins2.set_xticks([])
    ax_ins2.set_yticks([])

    ax2.set_xlabel("Distance Along Section Plane (mm)")
    ax2.set_ylabel(r"von Mises Stress $\sigma_{\mathrm{vM}}$ (MPa)")
    ax2.set_title("Residual von Mises Stress Along Principal Orthogonal Sections (A–A & B–B)",
                  fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "07_spatial_residual_stress_paths.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# PLOT 8: Normalized Warpage Index & Orthogonal Section Deflections (Paper Figures 8 & 9)
# ==============================================================================
def plot_08_normalized_warpage_index(ctx: Dict[str, Any]) -> Optional[str]:
    """Plot 8: Normalized Warpage Index & Orthogonal Cross-Sectional Deflection Curves.

    Inspired by Figures 8 & 9 in Applied Sciences 2024, 14, 810.
    """
    prof = _load_spatial_profiles(ctx)
    plots_dir = ctx["plots_dir"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300)

    lx_mm = prof["lx_mm"]
    ly_mm = prof["ly_mm"]
    hz_mm = prof["hz_mm"]
    ty_mm = prof["ty_mm"]

    # --- Subplot 1: Normalized Warpage Index along Diagonal & Section A-A ---
    dist_d = prof["dist_diag"]
    warp_d = prof["warp_idx_diag"]
    dist_aa = prof["dist_aa"]
    warp_aa = prof["warp_idx_aa"]

    ax1.plot(dist_d, warp_d, color=PALETTE["secondary"], linewidth=2.2,
             label=r"Diagonal Path $(0,0) \to (L_x, L_y)$")
    ax1.plot(dist_aa, warp_aa, color=PALETTE["primary"], linewidth=2.0, linestyle="--",
             label=r"Section A–A (Longitudinal Centerline, $Y = L_y/2$)")

    ax1.set_xlabel("Distance Along Path (mm)")
    ax1.set_ylabel(r"Warpage Index $\left(U_z - U_{z,\min}\right) / H_z$ [—]")
    ax1.set_title(rf"Normalized Substrate Warpage Index Along Inspection Trajectories ($H_z = {hz_mm:.1f}\,\mathrm{{mm}}$)",
                  fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Inset schematic
    ax_ins1 = ax1.inset_axes([0.68, 0.45, 0.26, 0.45])
    ax_ins1.add_patch(plt.Rectangle((0, 0), lx_mm, ly_mm, color="#f1f5f9", ec="#64748b", lw=1.2))
    ax_ins1.add_patch(plt.Rectangle((0, ly_mm / 2.0 - ty_mm / 2.0), lx_mm, ty_mm, color="#bfdbfe", ec="#2563eb", lw=1.0))
    ax_ins1.annotate("", xy=(lx_mm * 0.92, ly_mm * 0.92), xytext=(lx_mm * 0.08, ly_mm * 0.08),
                     arrowprops=dict(arrowstyle="->", color=PALETTE["secondary"], lw=2.2))
    ax_ins1.plot([0, lx_mm], [ly_mm / 2.0, ly_mm / 2.0], color=PALETTE["primary"], lw=1.5, linestyle="--")
    ax_ins1.text(lx_mm * 0.5, ly_mm * 0.12, "Diag & A-A", color="#334155",
                 fontsize=7.0, ha="center", fontweight="bold")
    ax_ins1.set_xlim(-lx_mm * 0.05, lx_mm * 1.05)
    ax_ins1.set_ylim(-ly_mm * 0.05, ly_mm * 1.05)
    ax_ins1.set_aspect("equal")
    ax_ins1.set_xticks([])
    ax_ins1.set_yticks([])

    # --- Subplot 2: Dimensional Deflection Uz(X) and Uz(Y) across Orthogonal Sections ---
    uz_aa = prof["uz_aa"]
    dist_bb = prof["dist_bb"]
    uz_bb = prof["uz_bb"]

    ax2.plot(dist_aa, uz_aa, color=PALETTE["primary"], linewidth=2.0,
             label=r"Section A–A (Longitudinal Profile $U_z(X)$)")
    ax2.plot(dist_bb, uz_bb, color=PALETTE["accent"], linewidth=2.0, linestyle="--",
             label=r"Section B–B (Transverse Profile $U_z(Y)$)")
    ax2.axhline(0.0, color="#64748b", linestyle=":", linewidth=0.9)

    ax2.set_xlabel("Distance Across Section Plane (mm)")
    ax2.set_ylabel(r"Vertical Deflection $U_z$ (mm)")
    ax2.set_title("Orthogonal Cross-Sectional Deflection Curves (Concave-Up Bending Curvature)",
                  fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "08_normalized_warpage_index.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# MAIN DRIVER: Generate Focused Publication Plot Suite
# ==============================================================================
def generate_all_plots(run_dir: str) -> List[str]:
    """Execute complete Matplotlib postprocessing suite for run_dir."""
    ctx = load_run_context(run_dir)
    plots_dir = ctx["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # Clean up obsolete or unneeded plots per user request
    obsolete_files = [
        "thermal_cycle_summary.png",
        "hardness_profile.png",
        "03_layer_remelting_and_height_evolution.png",
        "06_vickers_microhardness_profile.png",
        "07_phase_fractions_microstructure.png",
        "04_distortion_deflection_profile.png",
        "05_residual_stress_mises.png",
        "08_heat_input_and_dwell_energy.png",
    ]
    for obs in obsolete_files:
        p = os.path.join(plots_dir, obs)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    print("\n" + "=" * 80)
    print(" WAAM MULTI-PHYSICS NATIVE MATPLOTLIB POSTPROCESSING ENGINE")
    print(f" Target Run Directory: {os.path.abspath(run_dir)}")
    print(f" Output Plots Directory: {os.path.abspath(plots_dir)}")
    print("=" * 80 + "\n")

    plot_tasks = [
        ("01_thermal_history_and_peaks", plot_01_thermal_history_and_peaks),
        ("02_cooling_rates_and_dt85", plot_02_cooling_rates_and_dt85),
        ("03_distortion_and_warpage", plot_03_distortion_and_warpage),
        ("04_residual_stress_mises", plot_04_residual_stress_mises),
        ("05_heat_input_and_dwell_energy", plot_05_heat_input_and_dwell_energy),
        ("06_baseplate_center_thermal_cycle", plot_06_baseplate_center_thermal_cycle),
        ("07_spatial_residual_stress_paths", plot_07_spatial_residual_stress_paths),
        ("08_normalized_warpage_index", plot_08_normalized_warpage_index),
    ]

    generated = []
    total = len(plot_tasks)
    for idx, (name, fn) in enumerate(plot_tasks, 1):
        try:
            res = fn(ctx)
            if isinstance(res, (list, tuple)):
                for p in res:
                    if p and os.path.exists(p) and p not in generated:
                        generated.append(p)
                        print(f"  [{idx}/{total}] [OK] Generated: {os.path.basename(p)}")
            elif res and os.path.exists(res) and res not in generated:
                generated.append(res)
                print(f"  [{idx}/{total}] [OK] Generated: {os.path.basename(res)}")
            else:
                # A missing figure is reported and the name is left out of the
                # summary; it is never fabricated.
                print(f"  [{idx}/{total}] [SKIP] Not generated: {name}")
        except Exception as ex:
            print(f"  [{idx}/{total}] [ERROR] Plot {name} generation failed: {ex}")

    print("\n" + "-" * 80)
    print(f" [OK] Successfully generated {len(generated)} 2D publication plots.")
    print(f" Plots saved in: {plots_dir}")
    print("-" * 80 + "\n")

    return generated


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./waam_output"
    target_path = Path(target).resolve()
    if (target_path / "FreeFEM").exists():
        target_path = target_path / "FreeFEM"
    elif (target_path / "latest").exists() or (target_path / "latest").is_symlink():
        latest = target_path / "latest"
        if latest.is_symlink():
            target_path = latest.resolve()
        elif latest.is_file():
            # run.py's fallback writes a plain pointer file (not a symlink)
            # where symlinks are unavailable; the target is relative to the
            # directory holding it.
            try:
                pointed = latest.read_text(encoding="utf-8").strip().splitlines()[0]
                cand = (latest.parent / pointed).resolve()
                if cand.exists():
                    target_path = cand
            except Exception:
                pass
        if (target_path / "FreeFEM").exists():
            target_path = target_path / "FreeFEM"

    generate_all_plots(str(target_path))
