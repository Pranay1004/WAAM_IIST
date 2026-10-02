#!/usr/bin/env python3
"""
WAAM Multi-Physics Simulation Framework — Main Entry Point
===========================================================
Usage:
  python3 run.py                 # Launch interactive GUI
  python3 run.py --stl <file>    # FULL PIPELINE: slice -> mesh -> EDP -> solve -> report
  python3 run.py --cli           # Run default simulation in headless CLI mode
  python3 run.py --validate      # Run the physics validation suite
  python3 run.py --slice <file>  # Slice an STL/STEP file to G-code toolpath
  python3 run.py --openfoam      # Generate OpenFOAM 2412 weld pool CFD case
"""

import os
import sys
import argparse
import shutil
import subprocess
import datetime
from pathlib import Path


def run_command_with_tee(cmd: list, cwd: str, log_path: str, env: dict = None) -> int:
    """
    Run command while streaming unbuffered stdout/stderr to terminal AND writing to log_path in real time.
    Emulates OpenFOAM `| tee sim.log` execution.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n{'='*80}\n")
        log_f.write(f"EXECUTION: {' '.join(str(c) for c in cmd)}\n")
        log_f.write(f"CWD:       {cwd}\n")
        log_f.write(f"START:     {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"{'='*80}\n\n")
        log_f.flush()

        use_script = shutil.which("script") is not None and sys.platform != "win32"
        if use_script:
            exec_cmd = ["script", "-q", "/dev/null"] + cmd
        else:
            exec_cmd = cmd

        proc = subprocess.Popen(
            exec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env or os.environ.copy(),
            bufsize=1
        )

        for line in proc.stdout:
            clean_line = line.replace("\r", "")
            while clean_line and ord(clean_line[0]) < 32 and clean_line[0] not in "\n\t":
                clean_line = clean_line[1:]
            sys.stdout.write(clean_line)
            sys.stdout.flush()
            log_f.write(clean_line)
            log_f.flush()

        proc.wait()
        rc = proc.returncode
        log_f.write(f"\n[FINISHED] Process exited with code {rc} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.flush()
        return rc


def run_full_pipeline(geometry_file: str, args, base_waam_output: Path, here: Path) -> int:
    """
    End-to-end pipeline: slice -> toolpath artifacts -> mesh -> EDP -> FreeFEM solve -> postprocess.

    Guarantees the FreeFEM run directory receives the artifacts the generated
    .edp actually reads: toolpath.txt, layer_pass_times.txt, <stem>.mesh.
    """
    from mesh.slicer_api import slice_geometry_to_toolpath
    from mesh.stl_pipeline import stl_to_freefem_mesh
    from codegen.edp_builder import EDPBuilder

    geo = Path(geometry_file).resolve()
    if not geo.exists():
        print(f"[ERROR] Geometry file not found: {geo}")
        return 1

    now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    case_outdir = base_waam_output / now_ts
    run_outdir = case_outdir / "FreeFEM"
    run_outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[FULL-PIPELINE] Case: {case_outdir}")
    print(f"[FULL-PIPELINE] FreeFEM run dir: {run_outdir}")

    # ------------------------------------------------------------------
    # 1. Slice geometry -> G-code + toolpath.txt + per-layer pass times
    # ------------------------------------------------------------------
    layer_height = float(getattr(args, "layer_height", 3.0) or 3.0)
    bead_width = float(getattr(args, "bead_width", 8.0) or 8.0)
    gcode_path = run_outdir / "toolpath.gcode"
    print(f"[1/5] Slicing {geo.name} (h={layer_height} mm, w={bead_width} mm) ...")
    try:
        slice_res = slice_geometry_to_toolpath(
            str(geo), str(gcode_path),
            layer_height=layer_height, bead_width=bead_width)
    except Exception as ex:
        print(f"[ERROR] Slicing failed: {ex}")
        return 1
    n_layers_sliced = int(slice_res.get("n_layers", 0))
    print(f"       {n_layers_sliced} layers, {slice_res.get('n_points', 0)} points -> {gcode_path.name}")

    # slicer writes toolpath.txt next to the gcode; make sure both artifacts
    # sit exactly where the .edp reads them (outdir + "toolpath.txt")
    toolpath_txt = gcode_path.with_suffix(".txt")
    if not toolpath_txt.exists():
        print(f"[ERROR] Slicer did not produce {toolpath_txt.name}")
        return 1

    # per-layer times: slicer writes <stem>.layer_times.txt; edp reads layer_pass_times.txt
    layer_times_src = gcode_path.with_suffix(".layer_times.txt")
    layer_times_dst = run_outdir / "layer_pass_times.txt"
    if layer_times_src.exists():
        shutil.copy2(layer_times_src, layer_times_dst)

    # ------------------------------------------------------------------
    # 2. Mesh the geometry (Gmsh conformal mesh, mesh-rule compliant)
    # ------------------------------------------------------------------
    print(f"[2/5] Meshing {geo.name} ...")
    mesh_path = run_outdir / f"{geo.stem}.mesh"
    if not mesh_path.exists() or args.remesh:
        try:
            target_nodes = int(getattr(args, "target_nodes", 0) or 0)
            mesh_kwargs = {"layer_height": layer_height}
            if target_nodes > 0:
                mesh_kwargs["target_nodes"] = target_nodes
            mesh_res = stl_to_freefem_mesh(str(geo), str(mesh_path), **mesh_kwargs)
            print(f"       {mesh_res.get('n_nodes', '?')} nodes -> {mesh_path.name}")
        except Exception as ex:
            print(f"[ERROR] Meshing failed: {ex}")
            return 1
    else:
        print(f"       Reusing existing {mesh_path.name}")

    # ------------------------------------------------------------------
    # 3. Build parameters + generate .edp wired to the artifacts above
    # ------------------------------------------------------------------
    print("[3/5] Generating waam_simulation.edp ...")
    p = EDPBuilder.default_params()
    preset_name = getattr(args, "preset", "DEFAULT")
    try:
        from gui.app import PARAMETER_PRESETS, MM_DIMENSION_KEYS, MM_SPEED_KEYS
        if preset_name in PARAMETER_PRESETS:
            preset_data = dict(PARAMETER_PRESETS[preset_name])
            for k in MM_DIMENSION_KEYS:
                if k in preset_data and float(preset_data[k]) > 0.05:
                    preset_data[k] = float(preset_data[k]) / 1000.0
            for k in MM_SPEED_KEYS:
                if k in preset_data and float(preset_data[k]) > 0.1:
                    preset_data[k] = float(preset_data[k]) / 1000.0
            p.update(preset_data)
            print(f"       Preset: {preset_name}")
    except Exception as ex:
        print(f"       [WARN] Preset load skipped: {ex}")

    p["output_dir"] = str(run_outdir.resolve()) + "/"
    p["stl_file"] = str(geo)
    p["mesh_source"] = "cad_file"
    p["mesh_file"] = str(mesh_path)
    p["h_bead"] = layer_height / 1000.0
    p["h_bead_subseq"] = layer_height / 1000.0
    if n_layers_sliced > 0:
        p["n_layers"] = n_layers_sliced
    solver_name = getattr(args, "solver", "") or ""
    if solver_name:
        p["solver"] = solver_name

    # Materialize the toolpath under the canonical name the .edp reads
    shutil.copy2(toolpath_txt, run_outdir / "toolpath.txt")

    builder = EDPBuilder(p)
    edp_path = run_outdir / "waam_simulation.edp"
    builder.save(str(edp_path))

    # parameters.json — single source of truth for postprocessing
    try:
        import json as _json
        with open(run_outdir / "parameters.json", "w") as pf:
            _json.dump(p, pf, indent=2, default=str)
    except Exception as ex:
        print(f"       [WARN] parameters.json: {ex}")
    print(f"       {edp_path.name} ({n_layers_sliced} layers, solver={p.get('solver')})")

    # ------------------------------------------------------------------
    # 4. Solve
    # ------------------------------------------------------------------
    print("[4/5] Running FreeFEM++ (tee -> sim.log) ...")
    ff_bin = shutil.which("FreeFem++") or shutil.which("ff") or "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
    cmd = [ff_bin, str(edp_path.resolve()), str(run_outdir.resolve()) + "/"]
    rc = run_command_with_tee(cmd, cwd=str(run_outdir), log_path=str(run_outdir / "sim.log"))
    try:
        shutil.copy2(run_outdir / "sim.log", case_outdir / "sim.log")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Postprocess: plots + 3D renders + report
    # ------------------------------------------------------------------
    if rc == 0:
        print("[5/5] Postprocessing ...")
        try:
            from postprocess.plot_generator import generate_all_plots
            generate_all_plots(str(run_outdir.resolve()))
        except Exception as ex:
            print(f"       [WARN] Plots: {ex}")
        try:
            from postprocess.report_generator import generate_production_report
            report = generate_production_report(str(run_outdir.resolve()))
            print(f"       Report: {report}")
        except Exception as ex:
            print(f"       [WARN] Report: {ex}")
    else:
        print(f"[WARN] FreeFEM exited with code {rc}, skipping postprocessing")

    try:
        latest = base_waam_output / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(case_outdir.name)
    except Exception:
        pass

    print(f"\n[FULL-PIPELINE] Done. Case: {case_outdir}")
    return rc


def main():
    _here = Path(__file__).resolve().parent
    default_output_root = (_here / "waam_output").resolve()
    default_output_root.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="WAAM Multi-Physics Simulation Framework")
    parser.add_argument("--cli", action="store_true", help="Run simulation in headless CLI mode")
    parser.add_argument("--stl", type=str, metavar="GEOMETRY_FILE",
                        help="FULL PIPELINE: slice geometry -> mesh -> generate EDP -> solve -> report")
    parser.add_argument("--validate", action="store_true",
                        help="Run the physics validation suite (tests/physics_suite.py)")
    parser.add_argument("--slice", type=str, metavar="STL_FILE", help="Slice 3D geometry file to WAAM G-code")
    parser.add_argument("--openfoam", action="store_true", help="Generate OpenFOAM 2412 weld pool CFD case")
    parser.add_argument("--preset", type=str, default="DEFAULT", help="Parameter preset to load (default: DEFAULT)")
    parser.add_argument("--outdir", type=str, default=str(default_output_root), help="Output directory path")
    parser.add_argument("--uva", action="store_true", help="Run 3D UVA-WAAM SS308L on SS316 multi-physics simulation")
    parser.add_argument("--layer-height", type=float, default=3.0, help="Slice layer height [mm] (default 3.0)")
    parser.add_argument("--bead-width", type=float, default=8.0, help="Slice bead width [mm] (default 8.0)")
    parser.add_argument("--target-nodes", type=int, default=0, help="Mesh target node count (0 = auto)")
    parser.add_argument("--remesh", action="store_true", help="Regenerate mesh even if one exists")
    parser.add_argument("--solver", type=str, default="", help="Linear solver override (UMFPACK, CG, sparsesolver, ...)")

    args = parser.parse_args()

    # Always ensure target output directory is inside WAAM-Framework/waam_output
    if args.outdir == "./waam_output" or args.outdir == "waam_output" or not args.outdir:
        base_waam_output = default_output_root
    else:
        out_p = Path(args.outdir)
        if not out_p.is_absolute():
            base_waam_output = (_here / args.outdir).resolve()
        else:
            base_waam_output = out_p.resolve()
            # If user passed workspace root waam_output, redirect to WAAM-Framework/waam_output
            if str(base_waam_output) == str(_here.parent / "waam_output"):
                base_waam_output = default_output_root

    base_waam_output.mkdir(parents=True, exist_ok=True)

    if args.uva or args.preset in ["SS308L_UVA_WAAM", "SS308L on SS316 (UVA-WAAM 20kHz)"]:
        uva_candidates = [
            _here / "uva_waam_ss308l_ss316.edp",
            _here / "U-FSSW-CEL-3D-Simulation" / "uva_waam_ss308l_ss316.edp",
            _here.parent / "uva_waam_ss308l_ss316.edp"
        ]
        uva_edp = next((c for c in uva_candidates if c.exists()), None)
        if uva_edp:
            now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_outdir = base_waam_output / now_ts / "FreeFEM"
            run_outdir.mkdir(parents=True, exist_ok=True)
            print(f"\n[UVA-WAAM] Running authentic 3D simulation for SS308L on SS316: {uva_edp}")
            print(f"[OUTPUT] Dedicated timestamp-based run directory: {run_outdir}")
            
            ff_bin = shutil.which("FreeFem++") or "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
            cmd = [ff_bin, str(uva_edp.resolve()), str(run_outdir.resolve()) + "/"]
            sim_log_file = str(run_outdir / "sim.log")
            
            # Execute with real-time automatic tee logging
            rc = run_command_with_tee(cmd, cwd=str(uva_edp.parent), log_path=sim_log_file)
            
            # Postprocess: Generate 8 Native 2D Matplotlib Plots
            try:
                from postprocess.plot_generator import generate_all_plots
                plots = generate_all_plots(str(run_outdir.resolve()))
                print(f"[PLOTS] Generated {len(plots)} publication plots in: {run_outdir / 'plots'}")
            except Exception as ex:
                print(f"[PLOTS] Postprocessing warning: {ex}")
            sys.exit(rc)

    if args.validate:
        import subprocess
        suite = Path(__file__).parent / "tests" / "physics_suite.py"
        sys.exit(subprocess.call([sys.executable, str(suite)]))

    if args.stl:
        sys.exit(run_full_pipeline(args.stl, args, base_waam_output, _here))

    if args.slice:
        from mesh.slicer_api import slice_geometry_to_toolpath
        out_gcode = str(Path(args.slice).with_suffix(".gcode"))
        try:
            res = slice_geometry_to_toolpath(args.slice, out_gcode)
            print(f"[OK] Slicing finished: {res['n_layers']} layers -> {res['output_file']}")
        except Exception as ex:
            print(f"[ERROR] Slicing failed: {ex}")
            sys.exit(1)
        sys.exit(0)

    if args.openfoam:
        from core.fluid.openfoam_coupling import setup_openfoam_case
        from codegen.edp_builder import EDPBuilder
        p = EDPBuilder.default_params()
        case_dir = Path(base_waam_output) / "openfoam_weldpool"
        setup_openfoam_case(str(case_dir), p)
        print(f"[OK] OpenFOAM 2412 case setup complete at: {case_dir}")
        sys.exit(0)

    if args.cli:
        from codegen.edp_builder import EDPBuilder
        from gui.app import PARAMETER_PRESETS, MM_DIMENSION_KEYS, MM_SPEED_KEYS

        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        case_outdir = base_waam_output / now_ts
        run_outdir = case_outdir / "FreeFEM"
        run_outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[OUTPUT] Dedicated timestamp-based run directory: {run_outdir}")

        candidates = [
            _here / "DEFAULT.edp",
            _here.parent / "DEFAULT.edp",
            Path(os.environ.get("WAAM_DEFAULT_EDP", "")).expanduser() if os.environ.get("WAAM_DEFAULT_EDP") else None,
        ]
        default_edp = next((c for c in candidates if c is not None and c.exists()), None)

        if args.preset == "DEFAULT" and default_edp is not None and default_edp.exists():
            edp_path = default_edp
            print(f"[PRESET] Using authentic FreeFEM++ script: {edp_path}")
            run_cwd = str(default_edp.parent)
        else:
            p = EDPBuilder.default_params()
            if args.preset in PARAMETER_PRESETS:
                preset_data = dict(PARAMETER_PRESETS[args.preset])
                for k in MM_DIMENSION_KEYS:
                    if k in preset_data and float(preset_data[k]) > 0.05:
                        preset_data[k] = float(preset_data[k]) / 1000.0
                for k in MM_SPEED_KEYS:
                    if k in preset_data and float(preset_data[k]) > 0.1:
                        preset_data[k] = float(preset_data[k]) / 1000.0
                p.update(preset_data)
                print(f"[PRESET] Applied preset: {args.preset}")
            else:
                print(f"[WARNING] Unknown preset '{args.preset}'. Available presets: {list(PARAMETER_PRESETS.keys())}")
            p["output_dir"] = str(run_outdir.resolve()) + '/'
            builder = EDPBuilder(p)
            edp_path = run_outdir / "waam_simulation.edp"
            builder.save(str(edp_path))
            print(f"Generated script: {edp_path}")
            run_cwd = str(run_outdir)

        print("\nRunning FreeFEM++ with automatic real-time unbuffered logging to sim.log...")
        ff_bin = shutil.which("FreeFem++") or shutil.which("ff") or "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
        cmd = [ff_bin, str(edp_path.resolve()), str(run_outdir.resolve()) + "/"]
        sim_log_file = str(run_outdir / "sim.log")
        
        # Automatic real-time unbuffered tee logging
        rc = run_command_with_tee(cmd, cwd=run_cwd, log_path=sim_log_file)
        
        # Also copy sim.log to case root for quick top-level inspection
        try:
            shutil.copy2(sim_log_file, str(case_outdir / "sim.log"))
        except Exception:
            pass

        # Postprocess: Generate 8 Native 2D Matplotlib Plots
        if rc == 0:
            try:
                from postprocess.plot_generator import generate_all_plots
                plots = generate_all_plots(str(run_outdir.resolve()))
                print(f"[PLOTS] Generated {len(plots)} publication plots in: {run_outdir / 'plots'}")
            except Exception as ex:
                print(f"[PLOTS] Postprocessing warning: {ex}")
        else:
            print(f"[WARN] FreeFEM exited with code {rc}, skipping postprocessing")

        # Update latest symlink
        try:
            latest_link = base_waam_output / "latest"
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(case_outdir.name)
        except Exception:
            pass

        sys.exit(rc)

    # Default: launch GUI
    from gui.app import main as gui_main
    gui_main(preset_name=args.preset)


if __name__ == "__main__":
    main()
