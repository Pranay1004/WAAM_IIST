#!/usr/bin/env python3
"""
scratch/run_cad_waam_benchmark.py

Executes complete WAAM multi-physics simulation and post-processing
for both sphere_R40mm.stl and cube_80mm.stl.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Ensure WAAM-Framework is on PYTHONPATH
framework_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(framework_dir))

from mesh.stl_pipeline import stl_to_freefem_mesh
from codegen.edp_builder import EDPBuilder
from core.postprocess.output_manager import (
    generate_run_summary,
    generate_paraview_loader_script,
    generate_enhanced_html_report,
)
from postprocess.plot_generator import generate_all_plots


def run_benchmark(stl_filename: str, output_dirname: str):
    print(f"\n{'='*70}")
    print(f"STARTING COMPLETE WAAM PIPELINE FOR: {stl_filename}")
    print(f"{'='*70}\n")

    stl_path = str(framework_dir / stl_filename)
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL not found: {stl_path}")

    out_dir = str(framework_dir / "waam_output" / output_dirname)
    fem_dir = os.path.join(out_dir, "FreeFEM")
    vis_dir = os.path.join(fem_dir, "visual_outputs")
    plots_dir = os.path.join(fem_dir, "plots")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Mesh generation via isolated subprocess
    mesh_path = os.path.join(fem_dir, Path(stl_filename).stem + ".mesh")
    print(f"[STEP 1] Generating 3D tetrahedral volume mesh: {mesh_path}")
    mesh_info = stl_to_freefem_mesh(stl_path, mesh_path, target_nodes=2000)
    print(f"         Mesh generated: {mesh_info['n_nodes']} nodes, {mesh_info['n_tets']} tetrahedra")
    print(f"         BBox: {mesh_info['bbox_mm']}")
    for w in mesh_info.get("warnings", []):
        print(f"         {w}")

    # 2. Build FreeFEM script with EDPBuilder
    print(f"\n[STEP 2] Configuring multi-physics parameters & building FreeFEM script...")
    params = {
        "preset_name": "DEFAULT",
        "mesh_source": "cad_file",
        "mesh_file": mesh_path,
        "stl_file": stl_path,
        "output_dir": fem_dir.rstrip("/") + "/",
        # Base plate in SI meters
        "lx": 0.150,
        "ly_base": 0.150,
        "hz_base": 0.005,
        "nx_base": 25,
        "ny_base": 8,
        "nz_base": 2,
        # Deposition settings in SI units
        "material": "ER70S-6",
        "material_dep": "ER70S-6",
        "material_base": "Structural_Steel",
        "n_layers": 3,
        "nz_dep": 3,
        "ty_dep": 0.008,
        "h_bead": 0.004,
        "h_bead_first": 0.004,
        "remelt_ratio": 0.80,
        "h_bead_subseq": 0.0032,
        "I_weld": 200.0,
        "U_weld": 24.0,
        "eta_arc": 0.80,
        "v_scan": 0.010,
        "t_wait": 5.0,
        "dt_weld": 0.5,
        "dt_cool": 1.5,
        "T_amb": 25.0,
        "T_preheat": 25.0,
        "enable_radiation": True,
        "emissivity": 0.70,
        "h_conv": 15.0,
        "enable_boiling_cap": True,
        "enable_phase_kinetics": True,
        "enable_mechanical": True,
        "mechanical_bc": "isostatic",
        "chi_x": 1.2,
        "chi_y": 0.4,
        "k_spring_z": 1.0e4,
        "k_spring_xy": 1.0e3,
        "solver": "sparsesolver",
        "pvd_thermal_interval": 1,
        "save_vtk_interval": 1,
    }

    edp_builder = EDPBuilder(params)
    edp_file = os.path.join(fem_dir, "waam_simulation.edp")
    edp_builder.save(edp_file)
    print(f"         FreeFEM script ready: {edp_file}")

    with open(os.path.join(fem_dir, "parameters.json"), "w") as pf:
        json.dump(params, pf, indent=2)

    # 3. Execute FreeFEM++ solver
    print(f"\n[STEP 3] Executing FreeFem++ multi-physics solver...")
    ff_bin = shutil.which("FreeFem++") or "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
    if not os.path.exists(ff_bin) and not shutil.which("FreeFem++"):
        raise FileNotFoundError("FreeFem++ binary not found on system!")

    env = os.environ.copy()
    n_cores = str(min(os.cpu_count() or 4, 12))
    env["OMP_NUM_THREADS"] = n_cores
    env["OPENBLAS_NUM_THREADS"] = n_cores
    env["MKL_NUM_THREADS"] = n_cores
    env["VECLIB_MAXIMUM_THREADS"] = n_cores

    cmd = [ff_bin, os.path.abspath(edp_file), os.path.abspath(fem_dir) + "/"]
    sim_log_path = os.path.join(fem_dir, "sim.log")

    with open(sim_log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=fem_dir,
            env=env,
        )
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            if "--- Layer" in line or "ws=" in line or "Max Tpeak" in line or "Max distortion" in line or "DONE" in line:
                print("  [FF++] " + line.strip())
        proc.wait()

    rc = proc.returncode
    print(f"         FreeFem++ completed with return code: {rc}")
    if rc != 0:
        raise RuntimeError(f"FreeFem++ simulation failed with code {rc}. See {sim_log_path}")

    # Copy log to top case directory
    shutil.copy2(sim_log_path, os.path.join(out_dir, "sim.log"))

    # 4. Post-processing
    print(f"\n[STEP 4] Running comprehensive post-processing...")
    plots = generate_all_plots(fem_dir)
    print(f"         Generated {len(plots)} publication-quality plots: {[os.path.basename(p) for p in plots]}")

    pv_script = generate_paraview_loader_script(fem_dir)
    print(f"         ParaView loader script: {pv_script}")

    summary = generate_run_summary(fem_dir, params, return_code=rc)
    summary_path = os.path.join(fem_dir, "run_summary.json")
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    shutil.copy2(summary_path, os.path.join(out_dir, "run_summary.json"))
    print(f"         Run summary: Peak T = {summary.get('max_tpeak_C', 0):.1f} °C, "
          f"Distortion = {summary.get('max_disp_mm', 0):.4f} mm, "
          f"Max Stress = {summary.get('max_stress_mpa', 0):.2f} MPa")

    report_html = generate_enhanced_html_report(fem_dir, params, summary, plots)
    top_report = os.path.join(out_dir, "report.html")
    shutil.copy2(report_html, top_report)
    print(f"         Comprehensive HTML Report generated: {top_report}")

    print(f"\n{'='*70}")
    print(f"SUCCESSFULLY COMPLETED WAAM PIPELINE FOR: {stl_filename}")
    print(f"{'='*70}\n")
    return summary


if __name__ == "__main__":
    sphere_summary = run_benchmark("sphere_R40mm.stl", "sphere_R40mm_complete")
    cube_summary = run_benchmark("cube_80mm.stl", "cube_80mm_complete")

    print("\n" + "#"*70)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print(f"Sphere: Tmax={sphere_summary.get('max_tpeak_C')} C, Uz={sphere_summary.get('max_disp_mm')} mm, Stress={sphere_summary.get('max_stress_mpa')} MPa")
    print(f"Cube:   Tmax={cube_summary.get('max_tpeak_C')} C, Uz={cube_summary.get('max_disp_mm')} mm, Stress={cube_summary.get('max_stress_mpa')} MPa")
    print("#"*70 + "\n")
