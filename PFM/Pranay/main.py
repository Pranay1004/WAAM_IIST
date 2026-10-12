"""
===============================================================================
  MAIN.PY — Unified Entry Point: Interactive CLI & Lightweight Desktop GUI
===============================================================================

Authors: Pranay Pandey & Antigravity Suite
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path
import numpy as np

from preprocessing import generate_simulation_config, initialize_seeding
from pfm_kernels import (
    step_kim_pure_metal,
    step_wbm_binary_alloy,
    step_thin_interface_cet,
    trigger_heterogeneous_nucleation,
    update_grain_tracking,
    compute_acoustic_parameters,
    run_all_ss308_50s_cases,
    run_10frequency_sweep,
    run_500x1500_directional_suite
)
from postprocessing import (
    calculate_solid_fraction,
    track_dendrite_tip_position,
    calculate_sdas_fft,
    analyze_grain_morphology,
    generate_ebsd_ipf_rgb,
    plot_microstructure_dashboard,
    plot_multistage_morphology_evolution,
    plot_frequency_comparison_gallery,
    plot_phase_and_temperature_field,
    plot_solute_segregation_field,
    plot_directional_grain_structure,
    plot_metrics_evolution,
    save_frame_snapshot,
    assemble_gif,
    export_metrics_csv,
    plot_ss308_50s_multi_interval_gallery,
    plot_ss308_50s_phase_maps,
    plot_ss308_50s_quantitative_metrics,
    assemble_ss308_50s_gif,
    plot_10freq_morphology_gallery,
    plot_10freq_solidification_kinetics,
    plot_10freq_grain_statistics,
    generate_500x1500_publication_figures
)
from validation import run_all_benchmarks


BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# =============================================================================
# 1. CORE SIMULATION RUNNER
# =============================================================================
def run_simulation(config, progress_callback=None):
    """
    Executes a phase-field simulation using the unified modular engine.
    Supports:
      - 'directional_cet': Thin-interface multi-grain directional solidification & CET
      - 'pure_ni_kim': Kim et al. pure metal dimensional benchmark
      - 'binary_wbm': WBM binary alloy benchmark
    """
    mode = config["mode"]
    alloy = config["alloy"]
    freq = config["freq_khz"]
    grid = config["grid"]
    Nx, Ny = grid["Nx"], grid["Ny"]
    dx_m, dy_m = grid["dx_m"], grid["dy_m"]
    dx_um = grid["dx_um"]
    dt = config["dt"]
    total_time_ms = config["total_time_ms"]
    total_steps = config["total_steps"]
    save_step_interval = config["save_step_interval"]

    out_dir = BASE_OUTPUT_DIR / config["output_dir_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    print(f"\n{'='*75}")
    print(f"  STARTING PFM SIMULATION: {alloy.upper()} [{mode}]")
    print(f"  Domain: {grid['domain_width_um']:.1f} x {grid['domain_height_um']:.1f} μm ({Nx} x {Ny} grid)")
    print(f"  Spatial Step: dx = {dx_um:.2f} μm | Timestep: dt = {dt*1e6:.2f} μs")
    print(f"  Total Physical Time: {total_time_ms:.2f} ms ({total_steps} steps)")
    print(f"  UVA Frequency: {freq:.1f} kHz | Output: {out_dir}")
    print(f"{'='*75}\n")

    # Tracking arrays
    time_history = []
    fs_history = []
    tip_vel_history = []
    sdas_history = []
    evolution_snapshots = []
    stage_targets = [0.25, 0.50, 0.75, 1.00]
    stage_captured = [False, False, False, False]

    last_tip_y = 0.0
    last_tip_time = 0.0

    if mode == "directional_cet":
        # Load material & acoustic properties
        mat = config["mat_props"]
        c0 = 0.20
        phi, c, grain_id, theta_grain = initialize_seeding(
            Nx, Ny, seed_config="directional_substrate", c0=c0, num_substrate_seeds=max(8, Nx // 20)
        )

        ac_params = compute_acoustic_parameters(
            freq, amplitude_um=config["amp_um"],
            rho_liquid=mat["thermodynamics"].get("density_liquid_kg_m3", 7000.0),
            sound_speed=mat["acoustics"].get("sound_speed_liquid_m_s", 4150.0)
        )

        thermal = config["thermal"]
        G_thermal = thermal["G_thermal"]
        V_pull = thermal["V_pull"]
        T_bottom_0 = thermal["T_bottom_0"]

        Tm = mat["thermodynamics"].get("T_liquidus_K", 1730.0)
        m_L = mat["calphad_partitioning"].get("liquidus_slope_Cr_K_wt", -4.2)
        k_eq = mat["calphad_partitioning"].get("k_Cr", 0.85)

        W0 = 1.2e-6
        tau0 = 1.0e-5
        lambda_coup = 1.8
        D_L = 1.0e-9
        D_S = 1.0e-13
        aniso = 0.04
        anti_trapping = 0.3535
        gb_penalty = 1.2

        curr_id = 25
        if freq == 0.0:
            delta_T_N = 12.0
            delta_T_sigma = 1.5
            N_max = 3.0e7
        else:
            delta_T_N = max(3.5, 12.0 - ac_params["nucleation_barrier_reduction_K"] * 2.0)
            delta_T_sigma = 1.5
            N_max = 3.0e7 * ac_params["nucleation_density_boost"] * 100.0

        t_sim = 0.0
        t_wall_start = time.time()

        for step in range(total_steps):
            phi, c = step_thin_interface_cet(
                phi, c, grain_id, theta_grain,
                Nx, Ny, dx_m, dy_m, dt, t_sim,
                W0, tau0, lambda_coup, k_eq, m_L, c0, Tm,
                D_L, D_S, aniso, G_thermal, V_pull, T_bottom_0,
                anti_trapping, ac_params["acoustic_streaming_D"], gb_penalty
            )

            # Update orientation & grain boundaries every 5 steps
            if step % 5 == 0:
                grain_id, theta_grain = update_grain_tracking(phi, grain_id, theta_grain, Nx, Ny)

            # Evaluate heterogeneous equiaxed nucleation ahead of front
            if step % 20 == 0:
                phi, grain_id, theta_grain, curr_id = trigger_heterogeneous_nucleation(
                    phi, c, grain_id, theta_grain,
                    Nx, Ny, dx_m, dy_m, dt * 20.0, t_sim,
                    G_thermal, V_pull, T_bottom_0, Tm, m_L, c0,
                    delta_T_N, delta_T_sigma, N_max, curr_id, step + int(freq)
                )

            t_sim += dt

            # Capture multi-stage evolution snapshots (guaranteed 4 stages)
            frac_step = (step + 1) / total_steps
            for s_idx, tgt in enumerate(stage_targets):
                if not stage_captured[s_idx] and (frac_step >= tgt or step == total_steps - 1):
                    stage_captured[s_idx] = True
                    pct_label = int(round(tgt * 100))
                    stage_name = f"Stage {len(evolution_snapshots)+1} ({pct_label}% Growth)"
                    evolution_snapshots.append({
                        "phi": phi.copy(),
                        "grain_id": grain_id.copy(),
                        "theta": theta_grain.copy(),
                        "time_ms": t_sim * 1e3,
                        "solid_fraction": calculate_solid_fraction(phi),
                        "n_grains": len(np.unique(grain_id[(grain_id > 0) & (phi >= 0.5)])),
                        "stage_label": stage_name
                    })

            # Periodic metrics sampling & frame saving
            if step % save_step_interval == 0 or step == total_steps - 1:
                current_time_ms = t_sim * 1e3
                fs = calculate_solid_fraction(phi)
                tip_y = track_dendrite_tip_position(phi, dy_m)

                dt_tip = t_sim - last_tip_time
                if dt_tip > 1e-9 and tip_y >= last_tip_y:
                    tip_vel_mm_s = ((tip_y - last_tip_y) / dt_tip) * 1e3
                else:
                    tip_vel_mm_s = V_pull * 1e3
                last_tip_y = tip_y
                last_tip_time = t_sim

                sdas = calculate_sdas_fft(phi, dx_um)

                time_history.append(current_time_ms)
                fs_history.append(fs)
                tip_vel_history.append(tip_vel_mm_s)
                sdas_history.append(sdas)



                # Save frame snapshot for GIF animation with dynamic auto-framing
                save_frame_snapshot(
                    phi, dx_um, dx_um, step, current_time_ms, frames_dir,
                    grain_id=grain_id, theta_grain=theta_grain, field_c=c,
                    ac_label=f"{alloy.upper()} ({freq:.1f} kHz UVA)"
                )

                progress_pct = (step / max(total_steps - 1, 1)) * 100
                msg = f"  [{progress_pct:5.1f}%] Time: {current_time_ms:6.2f} ms | Solid Frac: {fs*100:5.1f}% | Tip Vel: {tip_vel_mm_s:5.2f} mm/s"
                print(msg)
                if progress_callback:
                    progress_callback(progress_pct, msg)

    wall_time = time.time() - t_wall_start
    print(f"\n[DONE] Simulation finished in {wall_time:.2f} s real time.")

    # Save output artifacts
    csv_file = out_dir / "metrics_history.csv"
    export_metrics_csv(csv_file, time_history, fs_history, tip_vel_history, sdas_history)

    # 1. 4-Panel Microstructure Publication Dashboard
    dashboard_fig = out_dir / "microstructure_dashboard.png"
    plot_microstructure_dashboard(
        phi, c, grain_id, theta_grain,
        time_history, fs_history, tip_vel_history,
        config, dashboard_fig, title_suffix=f"[{freq:.1f} kHz UVA]"
    )

    # 2. Multi-Stage Evolution Gallery
    gallery_fig = out_dir / "multistage_evolution_gallery.png"
    if len(evolution_snapshots) > 0:
        plot_multistage_morphology_evolution(
            evolution_snapshots, gallery_fig,
            alloy=alloy, freq_khz=freq,
            domain_width_um=grid["domain_width_um"],
            domain_height_um=grid["domain_height_um"]
        )

    # 3. Dedicated Field Plots
    grain_fig = out_dir / "directional_grain_microstructure.png"
    solute_fig = out_dir / "solute_segregation_profile.png"
    metrics_fig = out_dir / "metrics_evolution.png"

    plot_directional_grain_structure(
        phi, grain_id, theta_grain, dx_um, dx_um,
        f"{alloy.upper()} ({freq:.1f} kHz)", grain_fig
    )
    plot_solute_segregation_field(
        phi, c, dx_um, dx_um,
        f"{alloy.upper()} ({freq:.1f} kHz)", solute_fig
    )
    plot_metrics_evolution(
        time_history, fs_history, tip_vel_history, sdas_history,
        f"{alloy.upper()} ({freq:.1f} kHz)", metrics_fig
    )

    # 4. Assemble High-Resolution Dynamic-Zoom Animated GIF
    gif_file = out_dir / "microstructure_evolution.gif"
    assemble_gif(frames_dir, gif_file, fps=8)

    morph = analyze_grain_morphology(grain_id, phi, dx_um, dx_um)
    print(f"  Final Equiaxed Fraction: {morph['equiaxed_fraction']*100:.1f}%")
    print(f"  CET Regime: {morph['cet_regime']}")
    print(f"  Dashboard saved: {dashboard_fig}")
    print(f"  Evolution gallery saved: {gallery_fig}")
    print(f"  Grain structure saved: {grain_fig}")
    print(f"  Solute segregation saved: {solute_fig}")
    print(f"  Metrics evolution saved: {metrics_fig}")
    print(f"  Animation GIF saved: {gif_file}\n")

    return {
        "output_dir": str(out_dir),
        "dashboard_plot": str(dashboard_fig),
        "gallery_plot": str(gallery_fig),
        "grain_plot": str(grain_fig),
        "solute_plot": str(solute_fig),
        "metrics_plot": str(metrics_fig),
        "animation_gif": str(gif_file),
        "csv_metrics": str(csv_file),
        "morphology": morph,
        "wall_time_s": wall_time
    }


# =============================================================================
# 2. INTERACTIVE CLI RUNNER
# =============================================================================
def interactive_cli():
    """Interactive command-line dialogue for selecting parameters and running."""
    print("=" * 70)
    print("  UNIFIED PFM SOLIDIFICATION SUITE — INTERACTIVE CLI")
    print("=" * 70)
    print("Available options:")
    print("  1. Directional Solidification & CET (SS308, WAAM default)")
    print("  2. Frequency Sweep Comparison (0 kHz vs 20 kHz vs 40 kHz)")
    print("  3. Run Automated Validation Benchmarks (Kim, WBM, Hunt CET)")
    print("  4. Custom Alloy Simulation (SS304, SS316, IN718, Al-Si)")
    print("  5. Launch GUI")

    choice = input("\nEnter choice [1-5] (default: 1): ").strip() or "1"

    if choice == "3":
        run_all_benchmarks()
        return

    if choice == "5":
        launch_gui()
        return

    alloy = "ss308"
    if choice == "4":
        alloy_input = input("Enter alloy (ss308, ss304, ss316, in718, al_si): ").strip()
        if alloy_input:
            alloy = alloy_input

    freq_val = 0.0
    if choice in ("1", "4"):
        freq_input = input("Enter UVA frequency in kHz [0 for baseline, 20 for standard UVA]: ").strip()
        if freq_input:
            freq_val = float(freq_input)

    time_input = input("Enter total solidification time in ms [default: 50.0]: ").strip()
    sim_time_ms = float(time_input) if time_input else 50.0

    if choice == "2":
        print("\nRunning UVA Frequency Sweep (0, 20, 40 kHz)...")
        for f in (0.0, 20.0, 40.0):
            cfg = generate_simulation_config(
                alloy=alloy, mode="directional_cet", freq_khz=f, amp_um=5.0,
                domain_width_um=500.0, domain_height_um=1000.0, dx_um=1.0,
                total_time_ms=sim_time_ms
            )
            run_simulation(cfg)
    else:
        cfg = generate_simulation_config(
            alloy=alloy, mode="directional_cet", freq_khz=freq_val, amp_um=5.0,
            domain_width_um=750.0, domain_height_um=1500.0, dx_um=1.0,
            total_time_ms=sim_time_ms
        )
        run_simulation(cfg)


# =============================================================================
# 3. LIGHTWEIGHT DESKTOP GUI (TKINTER)
# =============================================================================
def launch_gui():
    """Launches a desktop GUI using Python standard Tkinter."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        print("[ERROR] Tkinter is not available on this Python installation. Use CLI.")
        return

    root = tk.Tk()
    root.title("Unified WAAM PFM Simulation Suite")
    root.geometry("680x620")
    root.resizable(True, True)

    frame = ttk.Frame(root, padding="15")
    frame.pack(fill=tk.BOTH, expand=True)

    title_lbl = ttk.Label(frame, text="WAAM Ultrasonic PFM Simulator", font=("Helvetica", 16, "bold"))
    title_lbl.pack(pady=5)

    # Alloy selector
    ttk.Label(frame, text="Alloy Material:").pack(anchor=tk.W, pady=(10, 2))
    alloy_var = tk.StringVar(value="ss308")
    alloy_combo = ttk.Combobox(frame, textvariable=alloy_var, values=["ss308", "ss304", "ss316", "pure_ni", "ni_cu", "in718", "al_si"])
    alloy_combo.pack(fill=tk.X, pady=2)

    # Frequency selector
    ttk.Label(frame, text="UVA Frequency (kHz):").pack(anchor=tk.W, pady=(10, 2))
    freq_var = tk.DoubleVar(value=20.0)
    freq_scale = ttk.Scale(frame, from_=0.0, to=50.0, variable=freq_var, orient=tk.HORIZONTAL)
    freq_scale.pack(fill=tk.X, pady=2)
    freq_disp = ttk.Label(frame, text="20.0 kHz")
    freq_disp.pack(anchor=tk.E)
    freq_scale.configure(command=lambda v: freq_disp.config(text=f"{float(v):.1f} kHz"))

    # Simulation time in ms
    ttk.Label(frame, text="Solidification Time (ms):").pack(anchor=tk.W, pady=(10, 2))
    time_var = tk.StringVar(value="30.0")
    time_entry = ttk.Entry(frame, textvariable=time_var)
    time_entry.pack(fill=tk.X, pady=2)

    # Output text area
    ttk.Label(frame, text="Simulation Progress & Output:").pack(anchor=tk.W, pady=(15, 2))
    log_text = tk.Text(frame, height=12, wrap=tk.WORD)
    log_text.pack(fill=tk.BOTH, expand=True, pady=2)

    def log_msg(msg):
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        root.update_idletasks()

    def on_run():
        alloy = alloy_var.get()
        freq = freq_var.get()
        sim_time_ms = float(time_var.get())

        log_msg(f">> Launching {alloy.upper()} at {freq:.1f} kHz for {sim_time_ms:.1f} ms...")
        cfg = generate_simulation_config(
            alloy=alloy, mode="directional_cet", freq_khz=freq, amp_um=5.0,
            domain_width_um=500.0, domain_height_um=1000.0, dx_um=1.0,
            total_time_ms=sim_time_ms
        )

        def progress_cb(pct, text):
            log_msg(f"  {text}")

        res = run_simulation(cfg, progress_callback=progress_cb)
        log_msg(f">> Finished! Deliverables saved:\n   Dashboard: {res['dashboard_plot']}\n   Gallery: {res['gallery_plot']}\n   Animation: {res['animation_gif']}")
        messagebox.showinfo("Simulation Complete", f"Completed {alloy.upper()} simulation!\nDashboard: {res['dashboard_plot']}\nGIF: {res['animation_gif']}")

    def on_benchmarks():
        log_msg(">> Running all validation benchmarks...")
        rep = run_all_benchmarks()
        log_msg(f">> Benchmarks complete! Report saved to:\n   {rep}")
        messagebox.showinfo("Benchmarks Complete", f"All benchmarks passed!\nReport: {rep}")

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X, pady=10)

    run_btn = ttk.Button(btn_frame, text="Run Simulation", command=on_run)
    run_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    bench_btn = ttk.Button(btn_frame, text="Run Validation Suite", command=on_benchmarks)
    bench_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    root.mainloop()


# =============================================================================
# 4. ARGPARSE PARSER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified WAAM Ultrasonic Phase-Field Modeling Suite")
    parser.add_argument("--mode", type=str, default="directional_cet",
                        choices=["directional_cet", "pure_ni_kim", "binary_wbm", "benchmark"],
                        help="Simulation mode or benchmark suite")
    parser.add_argument("--directional", action="store_true",
                        help="Run high-resolution directional solidification & CET simulation")
    parser.add_argument("--benchmark", "--validate", action="store_true", dest="validate",
                        help="Run full 6-benchmark validation suite against reference codes")
    parser.add_argument("--ss308-50s", action="store_true", dest="ss308_50s",
                        help="Run extended 50-second SS308 solidification & delta->gamma phase transformation study")
    parser.add_argument("--sweep-10freq", action="store_true", dest="sweep_10freq",
                        help="Run 10-frequency parametric sweep (0 to 50 kHz)")
    parser.add_argument("--domain-500x1500", action="store_true", dest="domain_500x1500",
                        help="Run 500 um x 1500 um directional domain multi-frequency study (0 to 30 kHz)")
    parser.add_argument("--all", action="store_true",
                        help="Execute all suites: validation benchmarks, directional CET, 50s study, 10-freq sweep, and 500x1500 suite")
    parser.add_argument("--alloy", type=str, default="ss308",
                        help="Alloy material name (ss308, ss304, ss316, pure_ni, ni_cu, in718, al_si)")
    parser.add_argument("--freq", type=float, default=20.0,
                        help="Ultrasonic vibration frequency in kHz (0 for baseline)")
    parser.add_argument("--amp", type=float, default=5.0,
                        help="Ultrasonic vibration amplitude in micrometers")
    parser.add_argument("--time", type=float, default=30.0,
                        help="Total solidification time in milliseconds")
    parser.add_argument("--width", type=float, default=500.0,
                        help="Domain width in micrometers")
    parser.add_argument("--height", type=float, default=1000.0,
                        help="Domain height in micrometers")
    parser.add_argument("--dx", type=float, default=1.0,
                        help="Spatial resolution dx in micrometers")
    parser.add_argument("--gui", action="store_true",
                        help="Launch lightweight desktop GUI")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive command-line interface")

    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    if args.interactive:
        interactive_cli()
        return

    # 1. Validation Benchmark Suite
    if args.validate or args.mode == "benchmark" or args.all:
        print("\n" + "="*80)
        print(" [1/5] RUNNING ALL VALIDATION BENCHMARKS")
        print("="*80)
        run_all_benchmarks()
        if not args.all:
            return

    # 2. SS308 Extended 50.0-Second Study
    if args.ss308_50s or args.all:
        print("\n" + "="*80)
        print(" [2/5] RUNNING SS308 50.0-SECOND EXTENDED SOLIDIFICATION & PHASE TRANSFORMATION SUITE")
        print("="*80)
        out_50s = BASE_OUTPUT_DIR / "ss308_50s_study"
        out_50s.mkdir(parents=True, exist_ok=True)
        res_50s = run_all_ss308_50s_cases(out_dir=out_50s)
        plot_ss308_50s_multi_interval_gallery(res_50s, out_50s / "pfm_ss308_50s_multi_interval_gallery.png")
        plot_ss308_50s_phase_maps(res_50s, out_50s / "pfm_ss308_50s_phase_maps_micro_macro.png")
        plot_ss308_50s_quantitative_metrics(res_50s, out_50s / "pfm_ss308_50s_quantitative_metrics.png")
        assemble_ss308_50s_gif(res_50s, out_50s / "ss308_50s_microstructure_evolution.gif")
        if not args.all:
            return

    # 3. 10-Frequency Parametric Sweep
    if args.sweep_10freq or args.all:
        print("\n" + "="*80)
        print(" [3/5] RUNNING 10-FREQUENCY PARAMETRIC SWEEP (0 TO 50 KHZ)")
        print("="*80)
        out_10f = BASE_OUTPUT_DIR / "ten_frequency_sweep"
        out_10f.mkdir(parents=True, exist_ok=True)
        res_10f, table_10f = run_10frequency_sweep(out_dir=out_10f)
        plot_10freq_morphology_gallery(res_10f, out_10f / "pfm_10freq_morphology_gallery.png")
        plot_10freq_solidification_kinetics(res_10f, out_10f / "pfm_10freq_solidification_kinetics.png")
        plot_10freq_grain_statistics(res_10f, out_10f / "pfm_10freq_grain_statistics.png")
        if not args.all:
            return

    # 4. 500x1500 um Directional Solidification Suite
    if args.domain_500x1500 or args.all:
        print("\n" + "="*80)
        print(" [4/5] RUNNING 500x1500 μm DIRECTIONAL SOLIDIFICATION & CET STUDY")
        print("="*80)
        out_500 = BASE_OUTPUT_DIR / "directional_500x1500"
        out_500.mkdir(parents=True, exist_ok=True)
        brain_dir = Path("/Users/pandeyji/.gemini/antigravity-ide/brain/8491838e-af81-4d46-942a-648ebcc29480")
        res_500 = run_500x1500_directional_suite(frequencies=[0, 1, 10, 20, 30], n_steps=3500, out_dir=out_500)
        generate_500x1500_publication_figures(res_500, out_500, brain_dir=brain_dir)
        if not args.all:
            return

    # 5. Core Directional Simulation (or single run)
    print("\n" + "="*80)
    print(f" [5/5] RUNNING PRIMARY DIRECTIONAL CET SIMULATION: {args.alloy.upper()} ({args.freq:.1f} kHz)")
    print("="*80)
    cfg = generate_simulation_config(
        alloy=args.alloy,
        mode=args.mode,
        freq_khz=args.freq,
        amp_um=args.amp,
        domain_width_um=args.width,
        domain_height_um=args.height,
        dx_um=args.dx,
        total_time_ms=args.time
    )
    run_simulation(cfg)


if __name__ == "__main__":
    main()
