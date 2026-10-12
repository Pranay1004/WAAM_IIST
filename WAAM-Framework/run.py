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
import re
import sys
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path

# FreeFEM / ParaView locations: PATH first, then the well-known macOS install
# dir. Override with FREEFEM_BIN / FF_MPIRUN / FREEFEM_MPI / PARAVIEW_BIN.
FF_APP = "/Applications/FreeFem++.app/Contents/ff-4.15/bin"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[@-Z\\-_]")


def find_freefem_serial() -> str:
    """Absolute path of the serial FreeFEM++ binary, or "" if not found."""
    return (shutil.which("FreeFem++") or shutil.which("ff")
            or (str(Path(FF_APP, "FreeFem++")) if Path(FF_APP, "FreeFem++").exists() else "")
            or os.environ.get("FREEFEM_BIN", ""))


def find_freefem_mpi() -> tuple:
    """(mpirun wrapper, mpi binary) or ("", "") if not found."""
    mpirun = (shutil.which("ff-mpirun")
              or (str(Path(FF_APP, "ff-mpirun")) if Path(FF_APP, "ff-mpirun").exists() else ""))
    mpibin = (shutil.which("FreeFem++-mpi")
              or (str(Path(FF_APP, "FreeFem++-mpi")) if Path(FF_APP, "FreeFem++-mpi").exists() else ""))
    return mpirun, mpibin


def _strip_ansi(line: str) -> str:
    """Remove terminal escape sequences (they corrupt a log file)."""
    return _ANSI_RE.sub("", line)


def run_command_with_tee(cmd: list, cwd: str, log_path: str, env: dict = None) -> int:
    """
    Run command while streaming unbuffered stdout/stderr to terminal AND writing
    to log_path in real time. Emulates OpenFOAM `| tee sim.log` execution.

    Uses a pty (via `script`) only to defeat child-side block buffering; the
    syntax differs per platform:
        macOS:  script -q /dev/null <cmd>...
        Linux:  script -q -c "<cmd>" /dev/null
    The child is killed with the parent, so an interrupted run cannot leave an
    orphaned FreeFEM behind.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    # One run = one log. Appending across invocations mixed several runs
    # (and repeated headers) into a single file.
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"{'='*80}\n")
        log_f.write(f"EXECUTION: {' '.join(str(c) for c in cmd)}\n")
        log_f.write(f"CWD:       {cwd}\n")
        log_f.write(f"START:     {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"{'='*80}\n\n")
        log_f.flush()

        use_script = shutil.which("script") is not None and sys.platform != "win32"
        if use_script:
            if sys.platform == "darwin":
                exec_cmd = ["script", "-q", "/dev/null"] + [str(c) for c in cmd]
            else:
                # util-linux: 'script -q -c CMD /dev/null'. The macOS argument
                # order raises "invalid option" on Linux.
                joined = " ".join('"%s"' % str(c) if " " in str(c) else str(c) for c in cmd)
                exec_cmd = ["script", "-q", "-c", joined, "/dev/null"]
        else:
            exec_cmd = [str(c) for c in cmd]

        proc = subprocess.Popen(
            exec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env or os.environ.copy(),
            bufsize=1
        )
        try:
            for raw in getattr(proc, "stdout", []) or []:
                clean_line = _strip_ansi(raw.replace("\r", ""))
                while clean_line and ord(clean_line[0]) < 32 and clean_line[0] not in "\n\t":
                    clean_line = clean_line[1:]
                sys.stdout.write(clean_line)
                sys.stdout.flush()
                log_f.write(clean_line)
                log_f.flush()
            proc.wait()
        except KeyboardInterrupt:
            # The wrapper (script) must die, and so must its child, otherwise
            # Ctrl-C leaves FreeFEM burning a core.
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            log_f.write("\n[INTERRUPTED] terminated by user\n")
            log_f.flush()
            return 130
        rc = proc.returncode
        log_f.write(f"\n[FINISHED] Process exited with code {rc} at "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.flush()
        return rc


def _framework_version() -> str:
    """Framework version, read from pyproject.toml (single source of truth)."""
    try:
        import re as _re
        txt = (_here_root() / "pyproject.toml").read_text(encoding="utf-8")
        m = _re.search(r'^version\s*=\s*"([^"]+)"', txt, _re.M)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "0.1.0"


def _here_root() -> Path:
    return Path(__file__).resolve().parent


def _shell_safe_dir(path: Path) -> str:
    """Reject output paths that would be interpolated into a shell command.

    The generated .edp builds system() strings like ("mkdir -p " + outdir),
    so a value containing ';' or '$' would execute. Rejecting here is cheaper
    than trying to escape it for FreeFEM's string concatenation.
    """
    bad = set(";|&`$><\n\\\"'*?!()[]{}\n\t")
    text = str(path)
    hits = sorted({ch for ch in text if ch in bad})
    if hits:
        raise SystemExit(
            f"--outdir contains characters that are unsafe to pass into the "
            f"generated FreeFEM script (which builds shell commands): {hits!r}. "
            f"Use a plain directory path.")
    return text


def apply_preset_to_params(p, preset_name: str, source: str, verbose: bool = True) -> None:
    """
    Overlay a GUI preset onto a parameter dict, converting the GUI's mm / mm/min
    units to the solver's SI base. One implementation for every entry point:
    the --cli and --stl paths previously carried copies that had already drifted
    (each warned differently and both crashed on a string-valued preset entry).
    """
    from gui.app import PARAMETER_PRESETS, MM_DIMENSION_KEYS, MM_SPEED_KEYS
    if preset_name not in PARAMETER_PRESETS:
        if verbose:
            print(f"[WARNING] Unknown preset '{preset_name}'. "
                  f"Available: {sorted(PARAMETER_PRESETS.keys())}")
        return
    preset_data = dict(PARAMETER_PRESETS[preset_name])
    for k in MM_DIMENSION_KEYS:
        if k in preset_data:
            try:
                v = float(preset_data[k])
            except (TypeError, ValueError):
                continue
            if v > 0.05:          # a mm value (a 0.05 m one is already SI)
                preset_data[k] = v / 1000.0
    for k in MM_SPEED_KEYS:
        if k in preset_data:
            try:
                v = float(preset_data[k])
            except (TypeError, ValueError):
                continue
            if v > 0.1:
                preset_data[k] = v / 1000.0
    p.update(preset_data)
    if verbose:
        print(f"       Preset: {preset_name} ({source})")


def _point_latest(base_waam_output: Path, case_outdir: Path) -> None:
    """Make <base>/latest resolve to case_outdir.

    A symlink is preferred, but it is not always available (Windows needs a
    developer-mode privilege, and the target may live on another volume), so
    the fallback is a small pointer FILE rather than a silent failure. The old
    behaviour leaked the opposite way: on this machine it produced a dangling
    symlink into a deleted /tmp directory, and the GUI's "open run folder"
    then opened nothing.
    """
    latest = base_waam_output / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(case_outdir.name)
        return
    except Exception:
        pass
    try:
        # Resolve inside <base> when possible: a relative pointer stays valid
        # if the tree is moved together.
        try:
            target = case_outdir.relative_to(base_waam_output)
        except ValueError:
            target = case_outdir
        with open(latest, "w", encoding="utf-8") as f:
            f.write(str(target) + "\n")
    except Exception:
        pass


def run_full_pipeline(geometry_file: str, args, base_waam_output: Path, here: Path) -> int:
    """
    End-to-end pipeline: slice -> toolpath artifacts -> mesh -> EDP -> FreeFEM solve -> postprocess.

    Guarantees the FreeFEM run directory receives the artifacts the generated
    .edp actually reads: toolpath.txt, layer_pass_times.txt, <stem>.mesh.
    """
    from mesh.slicer_api import slice_geometry_to_toolpath
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
    if float(slice_res.get("scale_applied", 1.0)) != 1.0:
        print(f"       [WARN] slicer auto-scaled the part by "
              f"{slice_res['scale_applied']:.4f}x to fit the build envelope")

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
    # 2. Mesh the geometry (Gmsh conformal mesh: substrate fused with the
    #    part, explicit boundary labels). Raises -- loudly -- if the CAD
    #    cannot be turned into a solvable volume mesh.
    # ------------------------------------------------------------------
    print(f"[2/5] Meshing {geo.name} ...")
    mesh_path = run_outdir / f"{geo.stem}.mesh"
    if not mesh_path.exists() or args.remesh:
        from mesh.stl_pipeline import stl_to_freefem_mesh
        p_hz = float(getattr(args, "hz_base_mm", 0.0) or 0.0)
        target_nodes = int(getattr(args, "target_nodes", 0) or 0)
        # The mesher is invoked HERE (not from inside the builder) so its
        # diagnostics and its hard failures are both visible at the right stage
        # of the pipeline instead of being swallowed by code generation.
        mesh_kwargs = dict(target_nodes=target_nodes or 2000,
                           layer_height=layer_height)
        if p_hz > 0:
            mesh_kwargs["sub_hz_mm"] = p_hz
        try:
            mesh_res = stl_to_freefem_mesh(str(geo), str(mesh_path), **mesh_kwargs)
        except Exception as ex:
            print(f"[ERROR] Meshing failed: {ex}")
            return 1
        print(f"       {mesh_res.get('n_nodes', '?')} nodes, "
              f"{mesh_res.get('n_tets', '?')} tets, {mesh_res.get('mesh_dim', '?')}D "
              f"-> {mesh_path.name}")
        for w in mesh_res.get("warnings", []) or []:
            print(f"       [MESH] {w}")
    else:
        print(f"       Reusing existing {mesh_path.name}")

    # ------------------------------------------------------------------
    # 3. Build parameters + generate .edp wired to the artifacts above
    # ------------------------------------------------------------------
    print("[3/5] Generating waam_simulation.edp ...")
    p = EDPBuilder.default_params()
    preset_name = getattr(args, "preset", "DEFAULT")
    if preset_name != "DEFAULT":
        apply_preset_to_params(p, preset_name, "run.py --stl")

    p["output_dir"] = str(run_outdir.resolve()) + "/"
    p["mesh_source"] = "cad_file"
    p["mesh_file"] = str(mesh_path)
    p.pop("stl_file", None)
    p["h_bead"] = layer_height / 1000.0
    p["h_bead_subseq"] = layer_height / 1000.0
    if getattr(args, "hz_base_mm", 0):
        p["hz_base"] = float(args.hz_base_mm) / 1000.0
    if n_layers_sliced > 0:
        p["n_layers"] = n_layers_sliced
    solver_name = getattr(args, "solver", "") or ""
    if solver_name:
        p["solver"] = solver_name
    n_ranks = max(1, int(getattr(args, "ranks", 1) or 1))
    if n_ranks > 1:
        # MPI flavor must be selected BEFORE the script is materialized: the
        # serial script must never reference mpirank (undefined there).
        p["mpi_safe"] = True

    # --- History output: probes given in mm, converted to metres for the solver ---
    def _parse_floats(raw, count=None, min_vals=None):
        if isinstance(raw, (int, float)):
            chunks = [str(raw)]
        else:
            # Accept comma, semicolon and plain-space separated input so
            # pasted coordinates from a spreadsheet do not silently parse as
            # one giant number and fail with an opaque message.
            chunks = re.split(r"[,;\s]+", str(raw).strip())
        vals = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                vals.append(float(chunk))
            except ValueError:
                raise ValueError(f"'{chunk}' is not a number")
        if count is not None and len(vals) % count != 0:
            raise ValueError(f"expected a multiple of {count} coordinates, got {len(vals)}")
        if min_vals is not None and len(vals) < min_vals:
            raise ValueError(f"a path needs at least {min_vals} values "
                             f"(X1,Y1,Z1,X2,Y2,Z2), got {len(vals)}")
        return vals

    probe_pts = []
    for raw in getattr(args, "probe", []) or []:
        try:
            v = _parse_floats(raw, count=3)
        except (ValueError, TypeError) as ex:
            raise SystemExit(f"--probe '{raw}': {ex}")
        probe_pts.append([c / 1000.0 for c in v])
    probe_paths = []
    for raw in getattr(args, "probe_path", []) or []:
        try:
            v = _parse_floats(raw, count=3, min_vals=6)
        except (ValueError, TypeError) as ex:
            raise SystemExit(f"--probe-path '{raw}': {ex}")
        probe_paths.append([c / 1000.0 for c in v])

    if probe_pts:
        p["probe_points"] = probe_pts
    if probe_paths:
        p["probe_paths"] = probe_paths
    p["path_samples"] = int(getattr(args, "path_samples", 40) or 40)
    p["probe_interval"] = max(1, int(getattr(args, "probe_interval", 1) or 1))

    # --- Restart / checkpoint ---
    restart_mode = getattr(args, "restart", "off") or "off"
    p["restart"] = restart_mode
    if restart_mode != "off":
        p["restart_every"] = max(1, int(getattr(args, "restart_every", 25) or 25))
        p["restart_resume"] = not bool(getattr(args, "no_resume", False))
        if p["restart_resume"]:
            ck = run_outdir / "checkpoint.bin"
            print(f"       restart={restart_mode} (resume enabled"
                  f"{'' if ck.exists() else '; no checkpoint present yet'})")

    # Materialize the toolpath under the canonical name the .edp reads. The
    # slicer writes "<gcode>.txt" beside the G-code, which for the fixed name
    # "toolpath.gcode" IS already "toolpath.txt" -- copying it onto itself
    # raised shutil.SameFileError and killed the pipeline at step 3/5, after
    # slicing and meshing had already been paid for.
    toolpath_dst = run_outdir / "toolpath.txt"
    if Path(toolpath_txt).resolve() != toolpath_dst.resolve():
        shutil.copy2(toolpath_txt, toolpath_dst)

    _shell_safe_dir(run_outdir.resolve())
    builder = EDPBuilder(p)
    edp_path = run_outdir / "waam_simulation.edp"
    builder.save(str(edp_path))

    # parameters.json — single source of truth for postprocessing
    try:
        import json as _json
        with open(run_outdir / "parameters.json", "w", encoding="utf-8") as pf:
            _json.dump(p, pf, indent=2, default=str)
    except Exception as ex:
        print(f"       [WARN] parameters.json: {ex}")
    for w in getattr(builder, "warnings", []) or []:
        print(f"       [EDP] {w}")
    print(f"       {edp_path.name} ({n_layers_sliced} layers, solver={p.get('solver')})")
    if getattr(args, "generate_only", False):
        _point_latest(base_waam_output, case_outdir)
        print(f"\n[GENERATE-ONLY] script + parameters ready at {run_outdir}; "
              f"solver not launched.")
        return 0

    # ------------------------------------------------------------------
    # 4. Solve
    # ------------------------------------------------------------------
    print("[4/5] Running FreeFEM++ (tee -> sim.log) ...")
    if n_ranks > 1:
        ff_mpi, ff_bin_mpi = find_freefem_mpi()
        if not (ff_mpi and ff_bin_mpi):
            print(f"[ERROR] --ranks {n_ranks} requested but no MPI FreeFEM was found "
                  f"(ff-mpirun / FreeFem++-mpi). Run serially or set FREEFEM_MPI.")
            return 1
        cmd = [ff_mpi, "-np", str(n_ranks), ff_bin_mpi,
               str(edp_path.resolve()), str(run_outdir.resolve()) + "/"]
        print(f"       MPI: {n_ranks} ranks via ff-mpirun (rank0/ is canonical; "
              f"mesh replicated in Phase 1)")
    else:
        ff_bin = find_freefem_serial()
        if not ff_bin:
            print("[ERROR] No FreeFEM++ binary found. Install FreeFEM++ or set "
                  "FREEFEM_BIN to its location.")
            return 1
        print(f"       solver: {ff_bin}")
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
            from postprocess.report_generator import (
                generate_production_report, generate_probe_history_plots)
            # Solver-sampled probe history (present only when --probe/--probe-path
            # was used); these carry true simulated times unlike VTU re-sampling.
            probe_figs = generate_probe_history_plots(
                Path(run_outdir.resolve()), Path(run_outdir.resolve()))
            if probe_figs:
                print(f"       Probe history: {len(probe_figs)} figure(s) from "
                      f"{'probe_points.csv' if (run_outdir / 'probe_points.csv').exists() else ''}"
                      f"{'probe_paths.csv' if (run_outdir / 'probe_paths.csv').exists() else ''}")
            report = generate_production_report(str(run_outdir.resolve()))
            print(f"       Report: {report}")
        except Exception as ex:
            print(f"       [WARN] Report: {ex}")
    else:
        print(f"[WARN] FreeFEM exited with code {rc}, skipping postprocessing")

    _point_latest(base_waam_output, case_outdir)
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
                        help="Run the fast validation suites (physics + J2 mechanics, "
                             "no solver execution)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Build the .edp + parameters.json and stop "
                             "before the FreeFEM solve (dry run / inspection)")
    parser.add_argument("--version", action="store_true",
                        help="Print the framework version and exit")
    parser.add_argument("--validate-full", action="store_true",
                        help="Run ALL validation suites, including the CAS/DED benchmark "
                             "which executes a real FreeFEM run (tens of minutes)")
    parser.add_argument("--slice", type=str, metavar="STL_FILE", help="Slice 3D geometry file to WAAM G-code")
    parser.add_argument("--openfoam", action="store_true", help="Generate OpenFOAM 2412 weld pool CFD case")
    parser.add_argument("--preset", type=str, default="DEFAULT", help="Parameter preset to load (default: DEFAULT)")
    parser.add_argument("--outdir", type=str, default=str(default_output_root), help="Output directory path")
    parser.add_argument("--uva", action="store_true", help="Run 3D UVA-WAAM SS308L on SS316 multi-physics simulation")
    parser.add_argument("--layer-height", type=float, default=3.0, help="Slice layer height [mm] (default 3.0)")
    parser.add_argument("--bead-width", type=float, default=8.0, help="Slice bead width [mm] (default 8.0)")
    parser.add_argument("--hz-base", dest="hz_base_mm", type=float, default=0.0,
                        help="Substrate thickness [mm] (default: the model default, "
                             "5 mm). The CAD mesh is built with the same plate, so "
                             "the two must agree or the deposit floats.")
    parser.add_argument("--target-nodes", type=int, default=0, help="Mesh target node count (0 = auto)")
    parser.add_argument("--remesh", action="store_true", help="Regenerate mesh even if one exists")
    parser.add_argument("--solver", type=str, default="", help="Linear solver override (UMFPACK, CG, sparsesolver, ...)")
    parser.add_argument("--ranks", type=int, default=1,
                        help="MPI ranks (default 1 = serial). >1 launches via ff-mpirun "
                             "with per-rank output trees (rank0/ is canonical). Phase 1: "
                             "I/O-safe, mesh still replicated; distributed solve is next.")

    # --- History output (Abaqus *OUTPUT,HISTORY equivalent) ---
    parser.add_argument("--probe", type=str, action="append", default=[], metavar="X,Y,Z",
                        help="Probe point in mm (repeatable), e.g. --probe 75,75,2. "
                             "The solver samples T, Tpeak, cooling rate and activation here "
                             "and writes probe_points.csv with true simulated times.")
    parser.add_argument("--probe-path", type=str, action="append", default=[],
                        metavar="X1,Y1,Z1,X2,Y2,Z2[,...]",
                        help="Probe path polyline in mm (repeatable). Resampled at equal "
                             "arc-length stations into probe_paths.csv.")
    parser.add_argument("--path-samples", type=int, default=40,
                        help="Stations sampled along each probe path (default 40)")
    parser.add_argument("--probe-interval", type=int, default=1,
                        help="Write history every N solver steps (default 1)")

    # --- Restart / checkpoint ---
    parser.add_argument("--restart", choices=["off", "layer", "step"], default="off",
                        help="Checkpointing: off | layer (after each layer) | step")
    parser.add_argument("--restart-every", type=int, default=25,
                        help="Steps between checkpoints when --restart step (default 25)")
    parser.add_argument("--no-resume", action="store_true",
                        help="With --restart, ignore any existing checkpoint and start clean")

    args = parser.parse_args()

    if args.version:
        print(f"waam-framework {_framework_version()}")
        sys.exit(0)

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
            _here / "generated_edps" / "uva_waam_ss308l_ss316.edp",
            _here / "uva_waam_ss308l_ss316.edp",
            _here / "U-FSSW-CEL-3D-Simulation" / "uva_waam_ss308l_ss316.edp",
            _here.parent / "uva_waam_ss308l_ss316.edp",
            _here.parent / "generated_edps" / "uva_waam_ss308l_ss316.edp",
        ]
        uva_edp = next((c for c in uva_candidates if c.exists()), None)
        if uva_edp is None:
            print("[ERROR] --uva requested but no uva_waam_ss308l_ss316.edp was found "
                  f"in any of:\n  " + "\n  ".join(str(c) for c in uva_candidates))
            sys.exit(1)
        if True:
            now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_outdir = base_waam_output / now_ts / "FreeFEM"
            run_outdir.mkdir(parents=True, exist_ok=True)
            print(f"\n[UVA-WAAM] Running authentic 3D simulation for SS308L on SS316: {uva_edp}")
            print(f"[OUTPUT] Dedicated timestamp-based run directory: {run_outdir}")
            
            ff_bin = find_freefem_serial()
            if not ff_bin:
                print("[ERROR] No FreeFEM++ binary found (set FREEFEM_BIN).")
                sys.exit(1)
            cmd = [ff_bin, str(uva_edp.resolve()), str(run_outdir.resolve()) + "/"]
            sim_log_file = str(run_outdir / "sim.log")

            # Execute with real-time automatic tee logging. The cwd is the
            # framework root, NOT the .edp's own folder: the script ends with
            # `system("python3 postprocess/plot_generator.py ...")`, which only
            # resolves when cwd is the project root.
            rc = run_command_with_tee(cmd, cwd=str(_here), log_path=sim_log_file)
            
            # Postprocess: Generate 8 Native 2D Matplotlib Plots
            try:
                from postprocess.plot_generator import generate_all_plots
                plots = generate_all_plots(str(run_outdir.resolve()))
                print(f"[PLOTS] Generated {len(plots)} publication plots in: {run_outdir / 'plots'}")
            except Exception as ex:
                print(f"[PLOTS] Postprocessing warning: {ex}")
            sys.exit(rc)

    if args.validate or args.validate_full:
        # physics_suite is the gate and validate_j2_mechanics the cross-check;
        # both are code-only and fast. validate_cas_ded_dimeen26 EXECUTES a real
        # FreeFEM run (tens of minutes), so it only runs under --validate-full.
        # Routing all three through plain --validate would make a routine
        # health check take longer than the simulation it is supposed to gate.
        suites = ["physics_suite.py", "validate_j2_mechanics.py"]
        if args.validate_full:
            suites.append("validate_cas_ded_dimeen26.py")
        overall, extra = 0, []
        for name in suites:
            suite = Path(__file__).parent / "tests" / name
            if not suite.exists():
                print(f"[VALIDATE] missing test file: {suite}")
                overall = 1
                continue
            print(f"\n[VALIDATE] === {name} ===")
            rc = subprocess.call([sys.executable, str(suite)])
            if rc != 0:
                overall = 1
                extra.append(name)
        if overall:
            print(f"\n[VALIDATE] FAILED: {', '.join(extra) or 'see above'}")
        else:
            print("\n[VALIDATE] all suites OK")
        sys.exit(overall)

    if args.stl:
        sys.exit(run_full_pipeline(args.stl, args, base_waam_output, _here))

    if args.slice:
        from mesh.slicer_api import slice_geometry_to_toolpath
        src = Path(args.slice).resolve()
        if not src.exists():
            print(f"[ERROR] Geometry file not found: {src}")
            sys.exit(1)
        # Write BESIDE the user's run directory, not into their source folder:
        # this used to drop <name>.gcode/.txt/.layer_times.txt next to the STL
        # (verified: running it against stl/7mmx80.stl polluted stl/).
        layer_h = float(getattr(args, "layer_height", 3.0) or 3.0)
        bead_w = float(getattr(args, "bead_width", 8.0) or 8.0)
        if args.outdir and args.outdir != str(default_output_root):
            out_root = Path(args.outdir)
            if not out_root.is_absolute():
                out_root = (_here / args.outdir).resolve()
        else:
            out_root = default_output_root
        slice_root = out_root / "sliced"
        slice_root.mkdir(parents=True, exist_ok=True)
        out_gcode = str(slice_root / (src.stem + ".gcode"))
        try:
            res = slice_geometry_to_toolpath(str(src), out_gcode,
                                             layer_height=layer_h, bead_width=bead_w)
        except Exception as ex:
            print(f"[ERROR] Slicing failed: {ex}")
            sys.exit(1)
        print(f"[OK] Slicing finished: {res['n_layers']} layers -> {res['output_file']}")
        if 'toolpath_path' in res:
            print(f"     toolpath.txt -> {res['toolpath_path']}")
        if 'layer_times_path' in res:
            print(f"     layer_pass_times.txt -> {res['layer_times_path']}")
        sys.exit(0)

    if args.openfoam:
        from core.fluid.openfoam_coupling import setup_openfoam_case
        from codegen.edp_builder import EDPBuilder
        p = EDPBuilder.default_params()
        apply_preset_to_params(p, args.preset, "run.py --openfoam")
        # Timestamped like every other case, so a new generation can never
        # overwrite an existing weld-pool case (and its results).
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        case_dir = Path(base_waam_output) / now_ts / "openfoam_weldpool"
        case_dir.parent.mkdir(parents=True, exist_ok=True)
        setup_openfoam_case(str(case_dir), p)
        print(f"[OK] OpenFOAM 2412 case setup complete at: {case_dir}")
        _point_latest(base_waam_output, case_dir.parent)
        sys.exit(0)

    if args.cli:
        from codegen.edp_builder import EDPBuilder

        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        case_outdir = base_waam_output / now_ts
        run_outdir = case_outdir / "FreeFEM"
        run_outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[OUTPUT] Dedicated timestamp-based run directory: {run_outdir}")

        p = EDPBuilder.default_params()
        apply_preset_to_params(p, args.preset, "run.py --cli")
        p["output_dir"] = str(run_outdir.resolve()) + '/'
        # Every solver/numerics/history flag below was previously accepted on
        # the command line and then silently dropped in --cli mode.
        if getattr(args, "solver", ""):
            p["solver"] = args.solver
        n_ranks = max(1, int(getattr(args, "ranks", 1) or 1))
        if n_ranks > 1:
            p["mpi_safe"] = True
        for raw in getattr(args, "probe", []) or []:
            v = [float(x) for x in re.split(r"[,;\s]+", str(raw).strip()) if x]
            if len(v) == 3:
                p.setdefault("probe_points", []).append([c / 1000.0 for c in v])
        for raw in getattr(args, "probe_path", []) or []:
            v = [float(x) for x in re.split(r"[,;\s]+", str(raw).strip()) if x]
            if len(v) >= 6:
                p.setdefault("probe_paths", []).append([c / 1000.0 for c in v])
        if getattr(args, "path_samples", None):
            p["path_samples"] = int(args.path_samples)
        if getattr(args, "probe_interval", None):
            p["probe_interval"] = max(1, int(args.probe_interval))
        if getattr(args, "restart", "off") and args.restart != "off":
            p["restart"] = args.restart
            p["restart_every"] = max(1, int(getattr(args, "restart_every", 25) or 25))
            p["restart_resume"] = not bool(getattr(args, "no_resume", False))

        _shell_safe_dir(run_outdir.resolve())
        builder = EDPBuilder(p)
        edp_path = run_outdir / "waam_simulation.edp"
        builder.save(str(edp_path))
        for w in getattr(builder, "warnings", []) or []:
            print(f"[EDP] {w}")
        print(f"Generated script: {edp_path}")
        if getattr(args, "generate_only", False):
            _point_latest(base_waam_output, case_outdir)
            print(f"[GENERATE-ONLY] script + parameters ready at {run_outdir}; "
                  f"solver not launched.")
            sys.exit(0)
        run_cwd = str(run_outdir)

        print("\nRunning FreeFEM++ with automatic real-time unbuffered logging to sim.log...")
        if n_ranks > 1:
            ff_mpi, ff_bin_mpi = find_freefem_mpi()
            if not (ff_mpi and ff_bin_mpi):
                print("[ERROR] MPI FreeFEM not found; run serially or set FREEFEM_MPI.")
                sys.exit(1)
            cmd = [ff_mpi, "-np", str(n_ranks), ff_bin_mpi,
                   str(edp_path.resolve()), str(run_outdir.resolve()) + "/"]
        else:
            ff_bin = find_freefem_serial()
            if not ff_bin:
                print("[ERROR] No FreeFEM++ binary found (set FREEFEM_BIN).")
                sys.exit(1)
            print(f"solver: {ff_bin}")
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

        _point_latest(base_waam_output, case_outdir)

        sys.exit(rc)

    # Default: launch GUI
    from gui.app import main as gui_main
    gui_main(preset_name=args.preset)


if __name__ == "__main__":
    main()
