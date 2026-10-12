#!/usr/bin/env python3
"""
WAAM Multi-Physics Parametric Sweep Runner
==========================================
Executes FEM thermomechanical analysis for the 6 experimental parameter sets.

Specifications:
  - Substrate Length: 150 mm, Width: 150 mm, Thickness: 5 mm
  - Build Length: 150 mm
  - Target Build Height: ~80 mm
  - Inter-pass Dwell Time: 20 s
  - Deposition Pattern: Bi-directional
  - Mesh Optimization: Conforming structured mesh with minimum cells
    for high numerical accuracy and fast FreeFEM++ solve time.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add WAAM-Framework root to sys.path
FRAMEWORK_ROOT = Path(__file__).resolve().parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from codegen.edp_builder import EDPBuilder
from run import find_freefem_serial, run_command_with_tee

# Define the 6 experimental parameter sets
PARAMETRIC_CASES = [
    {
        "case_id": 1,
        "name": "Case_1_120A_19V_v300",
        "current_A": 120.0,
        "voltage_V": 19.0,
        "wire_feed_m_min": 9.5,
        "torch_speed_mm_min": 300.0,  # 5.0 mm/s
        "bead_height_mm": 3.78,
        "bead_width_mm": 5.50,
        "n_layers": 21,  # 21 * 3.78 mm = 79.38 mm (~80 mm)
    },
    {
        "case_id": 2,
        "name": "Case_2_120A_19V_v400",
        "current_A": 120.0,
        "voltage_V": 19.0,
        "wire_feed_m_min": 9.5,
        "torch_speed_mm_min": 400.0,  # 6.67 mm/s
        "bead_height_mm": 3.00,
        "bead_width_mm": 4.72,
        "n_layers": 27,  # 27 * 3.00 mm = 81.00 mm (~80 mm)
    },
    {
        "case_id": 3,
        "name": "Case_3_140A_19.7V_v400",
        "current_A": 140.0,
        "voltage_V": 19.7,
        "wire_feed_m_min": 11.9,
        "torch_speed_mm_min": 400.0,  # 6.67 mm/s
        "bead_height_mm": 3.25,
        "bead_width_mm": 5.64,
        "n_layers": 25,  # 25 * 3.25 mm = 81.25 mm (~80 mm)
    },
    {
        "case_id": 4,
        "name": "Case_4_140A_19.7V_v300",
        "current_A": 140.0,
        "voltage_V": 19.7,
        "wire_feed_m_min": 11.9,
        "torch_speed_mm_min": 300.0,  # 5.0 mm/s
        "bead_height_mm": 3.78,
        "bead_width_mm": 6.42,
        "n_layers": 21,  # 21 * 3.78 mm = 79.38 mm (~80 mm)
    },
    {
        "case_id": 5,
        "name": "Case_5_200A_24.2V_v300",
        "current_A": 200.0,
        "voltage_V": 24.2,
        "wire_feed_m_min": 19.9,
        "torch_speed_mm_min": 300.0,  # 5.0 mm/s
        "bead_height_mm": 4.11,
        "bead_width_mm": 9.40,
        "n_layers": 20,  # 20 * 4.11 mm = 82.20 mm (~80 mm)
    },
    {
        "case_id": 6,
        "name": "Case_6_70A_25.2V_v300",
        "current_A": 70.0,
        "voltage_V": 25.2,
        "wire_feed_m_min": 4.8,       # Lab session baseline: 4.8 m/min
        "torch_speed_mm_min": 300.0,  # Lab session baseline: 300 mm/min
        "bead_height_mm": 2.50,
        "bead_width_mm": 4.08,
        "n_layers": 32,  # 32 * 2.50 mm = 80.00 mm (exact 80 mm)
    },
]


def setup_case(case_dict: dict, base_outdir: Path) -> dict:
    """Build FreeFEM parameters and generate .edp file for a case."""
    outdir = base_outdir / case_dict["name"]
    outdir.mkdir(parents=True, exist_ok=True)

    v_scan_m_s = (case_dict["torch_speed_mm_min"] / 1000.0) / 60.0
    h_bead_m = case_dict["bead_height_mm"] / 1000.0
    w_bead_m = case_dict["bead_width_mm"] / 1000.0
    n_layers = case_dict["n_layers"]

    p = EDPBuilder.default_params()
    p["output_dir"] = str(outdir.resolve()) + "/"

    # Geometry
    p["lx"] = 0.150       # 150 mm
    p["ly_base"] = 0.150  # 150 mm
    p["hz_base"] = 0.005  # 5 mm
    p["ty_dep"] = w_bead_m
    p["h_bead"] = h_bead_m
    p["h_bead_first"] = h_bead_m
    p["h_bead_subseq"] = h_bead_m
    p["remelt_ratio"] = 1.0
    p["n_layers"] = n_layers

    # Process
    p["I_weld"] = float(case_dict["current_A"])
    p["U_weld"] = float(case_dict["voltage_V"])
    p["v_scan"] = v_scan_m_s
    p["v_wire"] = float(case_dict["wire_feed_m_min"])
    p["t_wait"] = 20.0    # 20s dwell
    p["deposition_pattern"] = "bidirectional"

    # High-Fidelity Optimal Meshing (User specification: 0.5x bead height, 0.5x bead width):
    # dz = 0.50 * bead_height (2 cells per layer)
    # dy = 0.33 to 0.50 * bead_width (3 cells across bead)
    # dx = ~5 mm (30 cells along 150 mm length)
    p["nx_dep"] = 30
    p["ny_dep"] = 3
    p["nz_dep"] = n_layers * 2
    p["nx_base"] = 30
    p["ny_base"] = 8
    p["nz_base"] = 2
    p["nz_dep_follows_layers"] = True
    p["mesh_source"] = "parametric"
    p["solver"] = "sparsesolver"

    # High-Fidelity Case-Specific Goldak Volumetric Heat Source:
    # b_G = bead half-width (0.50 * w_bead)
    # c_G = penetration depth into layer + substrate/prev layer (1.25 * h_bead)
    # a_f = front semi-axis (0.50 * w_bead)
    # a_r = rear semi-axis (2.50 * a_f = 1.25 * w_bead)
    p["bG"] = 0.50 * w_bead_m
    p["cG"] = 1.25 * h_bead_m
    p["af"] = 0.50 * w_bead_m
    p["ar"] = 1.25 * w_bead_m
    p["ff"] = 0.60
    p["fr"] = 1.40
    p["eta_arc"] = 0.85
    p["auto_source_sizing"] = False

    # Time steps & Paraview frame saving (save all timesteps from 0s to last second)
    p["save_every_n_weld"] = 1   # Save every single welding step
    p["save_every_n_cool"] = 1   # Save every single cooling step
    p["dt_weld"] = 1.0           # 1.0 s welding step
    p["dt_wait"] = 2.0           # 2.0 s dwell step: saves 10 frames during each 20s dwell hold
    p["k_liq_factor"] = 3.5      # Realistic molten pool convective stirring (Marangoni)
    p["enable_boiling_cap"] = True

    # 5 Key Thermocouple Elevation Probes at Baseplate Center (x=75mm, y=75mm):
    # Matches Plot 06 stations:
    # Pt 0: Substrate Mid-Plane (Z = 2.5 mm)
    # Pt 1: Substrate Surface / Fusion Interface (Z = 5.0 mm)
    # Pt 2: 20% Wall Build Height (Z = 5.0 + 0.20 * H_build)
    # Pt 3: 50% Wall Build Height (Z = 5.0 + 0.50 * H_build)
    # Pt 4: 100% Wall Build Height (Top) (Z = 5.0 + 1.00 * H_build)
    h_build_m = n_layers * h_bead_m
    xc_m = 0.075
    yc_m = 0.075
    hz_base_m = 0.005
    p["probe_points"] = [
        [xc_m, yc_m, 0.5 * hz_base_m],
        [xc_m, yc_m, hz_base_m],
        [xc_m, yc_m, hz_base_m + 0.20 * h_build_m],
        [xc_m, yc_m, hz_base_m + 0.50 * h_build_m],
        [xc_m, yc_m, hz_base_m + 1.00 * h_build_m],
    ]
    p["probe_interval"] = 1

    builder = EDPBuilder(p)
    edp_path = outdir / "waam_simulation.edp"
    builder.save(str(edp_path))

    with open(outdir / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, default=str)

    return {
        "case_dict": case_dict,
        "outdir": outdir,
        "edp_path": edp_path,
        "params": p,
    }


def run_case(case_info: dict) -> int:
    """Run FreeFEM++ for a single parametric case."""
    c = case_info["case_dict"]
    outdir = case_info["outdir"]
    edp_path = case_info["edp_path"]
    sim_log = outdir / "sim.log"

    ff_bin = find_freefem_serial()
    if not ff_bin:
        print(f"[ERROR] FreeFem++ binary not found for {c['name']}")
        return 1

    cmd = [ff_bin, str(edp_path.resolve()), str(outdir.resolve()) + "/"]
    print(f"\n{'='*70}")
    print(f" RUNNING FEM: [{c['case_id']}/6] {c['name']}")
    print(f" Height: {c['n_layers']} layers ({c['n_layers']*c['bead_height_mm']:.1f} mm) | "
          f"I={c['current_A']} A, U={c['voltage_V']} V, v={c['torch_speed_mm_min']} mm/min")
    print(f" Output dir: {outdir}")
    print(f"{'='*70}")

    t0 = time.time()
    rc = run_command_with_tee(cmd, cwd=str(outdir), log_path=str(sim_log))
    elapsed = time.time() - t0
    status = "SUCCESS" if rc == 0 else f"FAILED (rc={rc})"
    print(f"Finished {c['name']}: {status} in {elapsed:.1f} s")

    # Post-process plots if successful
    if rc == 0:
        try:
            from postprocess.plot_generator import generate_all_plots
            generate_all_plots(str(outdir.resolve()))
        except Exception as ex:
            print(f"Plot generation note: {ex}")

    # Read solver_results.json if available
    res_path = outdir / "solver_results.json"
    solver_data = {}
    if res_path.exists():
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                solver_data = json.load(f)
        except Exception:
            pass

    return {
        "case_id": c["case_id"],
        "name": c["name"],
        "status": "SUCCESS" if rc == 0 else f"FAILED (code {rc})",
        "return_code": rc,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60.0, 2),
        "n_layers": c["n_layers"],
        "build_height_mm": round(c["n_layers"] * c["bead_height_mm"], 2),
        "solver_results": solver_data,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WAAM Parametric Sweep Runner")
    parser.add_argument("--run", action="store_true", help="Execute FreeFEM++ simulations")
    parser.add_argument("--case", type=int, default=0, help="Run specific case 1-6 (0 = all)")
    parser.add_argument("--from-case", type=int, default=1, help="Start from case N through 6")
    args = parser.parse_args()

    base_outdir = FRAMEWORK_ROOT / "waam_output" / "parametric_sweep_6cases"
    base_outdir.mkdir(parents=True, exist_ok=True)
    summary_file = base_outdir / "sweep_timing_summary.json"

    print("=" * 70)
    print(" WAAM PARAMETRIC SWEEP (6 EXPERIMENTAL CASES)")
    print(f" Target output: {base_outdir}")
    print("=" * 70)

    cases = []
    for c in PARAMETRIC_CASES:
        info = setup_case(c, base_outdir)
        cases.append(info)
        tot_h = c["n_layers"] * c["bead_height_mm"]
        print(f"[{c['case_id']}/6] {c['name']:<28} | {c['n_layers']} layers ({tot_h:.1f} mm) | "
              f"I={c['current_A']}A, U={c['voltage_V']}V, v={c['torch_speed_mm_min']} mm/min -> EDP ready")

    if args.run:
        if 1 <= args.case <= 6:
            target_cases = [cases[args.case - 1]]
        elif args.from_case > 1:
            target_cases = [c for c in cases if c["case_dict"]["case_id"] >= args.from_case]
        else:
            target_cases = cases

        # Load existing summary if present
        all_summaries = {}
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as sf:
                    all_summaries = json.load(sf)
            except Exception:
                all_summaries = {}

        print(f"\nStarting execution of {len(target_cases)} case(s)...")
        for c_info in target_cases:
            res_item = run_case(c_info)
            all_summaries[c_info["case_dict"]["name"]] = res_item
            with open(summary_file, "w", encoding="utf-8") as sf:
                json.dump(all_summaries, sf, indent=2)

        print("\n" + "=" * 70)
        print(" PARAMETRIC SWEEP SUMMARY (EXECUTION TIMES)")
        print("=" * 70)
        for k, v in all_summaries.items():
            print(f"  {k:<32} : {v.get('status', 'N/A')} in {v.get('elapsed_seconds', 0)} s ({v.get('elapsed_minutes', 0)} min)")
    else:
        print("\nAll 6 case EDP scripts and meshes generated successfully!")
        print("Execute all cases with : python3 parametric_sweep.py --run")
        print("Execute single case    : python3 parametric_sweep.py --run --case 1")
        print("Continue from case N   : python3 parametric_sweep.py --run --from-case 2")


if __name__ == "__main__":
    main()
