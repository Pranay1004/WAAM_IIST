# WAAM_IIST

**Wire Arc Additive Manufacturing (WAAM) — multiphysics FEM simulation framework & laboratory tooling**

[![CI](https://github.com/Pranay1004/WAAM_IIST/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranay1004/WAAM_IIST/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

This repository bundles two complementary projects:

| Project | What it is |
|---|---|
| [`WAAM-Framework/`](WAAM-Framework/) | Production-grade **simulation pipeline**: STL/STEP → adaptive slicing → conformal tetrahedral meshing (5 mm substrate by default) → **FreeFEM++** transient thermal–mechanical FEM (Goldak double-ellipsoid heat source, quiet-element activation, MUMPS direct solver) → energy-balance audit → publication-grade postprocessing (plots, thermocouples, build animation, self-contained HTML report) + Tkinter GUI. |
| [`waam_lab/`](waam_lab/) | **Laboratory tooling**: a standalone STL → G-code **WAAM slicer** (perimeter + zigzag infill, DDCS V2.1 motion-controller dialect), CAD/geometry library, process documentation and calculator. |

---

## Highlights

- **End-to-end CLI**: `python run.py --stl <geometry.stl>` drives geometry ingest, slicing, meshing, code generation, solving, and postprocessing in one command.
- **Physics**: Goldak double-ellipsoid (plus Gaussian-volumetric / conical / uniform / prescribed-temperature options), latent-heat enthalpy method, convection + radiation boundaries, boiling-temperature cap, quiet-element activation for not-yet-deposited material, inherent-strain mechanical stage, phase-transformation → Vickers-hardness prediction.
- **Honest numerics**: every run prints an **energy-balance audit** (arc/prescribed energy → stored / surface / boiling-cap / rectifier, with residual %) instead of hiding quadrature error. The deposit/substrate interface is a single conforming mesh with shared nodes (always bonded); the heat source carries a per-element birth mask so no energy is deposited into not-yet-active material.
- **Performance**: MUMPS direct solver (≈3.8× faster than UMFPACK on an 83k-DoF thermal solve; 251k-DoF mechanical solves in <7 GB).
- **Machine-readable results**: the solver writes `solver_results.json` (stress, warpage, energy audit, Newton residuals) instead of the post-processor scraping console text. When the equilibrium solve does not converge the structural values are JSON `null` and `mechanics_converged` is `false` — an unconverged iterate can never be reported as a result.
- **History output**: user-defined **probe points and paths** (`--probe 75,75,2`, `--probe-path 0,75,2,150,75,2`) are sampled *by the solver* and written to `probe_points.csv` / `probe_paths.csv` tagged with the true simulated time — the Abaqus `*OUTPUT, HISTORY` equivalent, replacing hard-coded thermocouples and a guessed time axis.
- **Restart**: `--restart layer` checkpoints the full thermal + mechanical state after every layer, so an interrupted build resumes mid-deposition. Verified to reproduce an uninterrupted 8-layer run to within 0.02%.
- **Equilibrium control**: incremental thermo-elasto-plastic solve with a real force + displacement convergence test (`CONVERGED` / `NOT CONVERGED`); unconverged results are refused rather than published.
- **Adaptive time stepping**: `adaptive_dt="fourier"` (builder param / GUI) derives Δt from the diffusion number and the measured minimum element size, landing on the converged answer without the user guessing Δt. A coarse fixed `dt_weld` was measured to under-predict warpage by 23%.
- **External validation**: CAS-DEDD-IMEEN26 (Ansys Academic Program, AlSi10Mg) with prescribed-temperature activation at liquidus and in-weld mechanical cadence: `python tests/validate_cas_ded_dimeen26.py`. Distortion 0.17×/0.46× (max/mean) of the reference on a compliant table (both in the 0.1–10× regime band), 0.05×/0.12× rigidly clamped — the pair brackets the reference; a straight stripe springs back where a spiral accumulates constraint. Energy audit closes to <1% on resolved arc-source meshes.
- **Validation**: automated **279/279 physics & code-generation checks** (`tests/physics_suite.py`) runnable via `python run.py --validate`.
- **MPI (Phase 1)**: `--ranks N` launches via `ff-mpirun` with per-rank output trees (`rank0/` canonical); scripts are I/O-safe under multi-rank execution. Mesh is still replicated; true distributed solve (Dmesh) is next. Multi-rank *execution* is currently blocked by the vendor packaging (bundled `libpmpi` needs an MPI-IO symbol the bundled `libmpi` lacks; Homebrew mpich is an incompatible ABI) — root-caused in-repo, not fixable from here.
- **Higher-order mechanics**: `mechanics_order=2` selects P2 tets (C3D10-like) for warpage-grade bending without P1-tet locking; thermal stays P1 (its audit already closes to <1%). P2-elastic converges; P2+J2-Picard is BETA (it churns on sharp activation-front gradients — consistent-tangent Newton is the defined cure) and warns loudly. True hexahedra are impossible in FreeFEM 4.15 (`readmesh3` silently drops HEXAHEDRA, `nt=0`; no Q1 fespace): the 3D kernel is simplex-only, so P2 tets are the higher-order path, not hexes.
- **Postprocessing**: 5 publication-grade plots, broken-axis substrate thermocouple figure, layer-by-layer build GIF, uncut 3D field renders, and a self-contained `report.html`.

## Requirements

- Python ≥ 3.9 (`numpy`, `matplotlib`, `pyvista`, `imageio`, `trimesh`, `gmsh`, `vtk` — see [`WAAM-Framework/pyproject.toml`](WAAM-Framework/pyproject.toml))
- **FreeFEM++ ≥ 4.14** (4.15 recommended, built with MUMPS) — external solver, installed separately from <https://freefem.org>
- Optional: OpenFOAM (Marangoni melt-pool CFD coupling), ParaView (`.pvd` inspection), pywebview (GUI windowing)
- Tested on macOS (Apple Silicon); on Linux/Windows use the native or WSL2 builds of FreeFEM++.

## Quickstart

```bash
# 1. Install Python dependencies
pip install ./WAAM-Framework

# 2. Run the full multiphysics pipeline on the sample sphere
cd WAAM-Framework
python run.py --stl stl/sphere_R40mm.stl

# 3. Run the validation suite anytime (212 checks)
python run.py --validate

# 4. Laboratory slicer: STL -> layers -> toolpaths -> G-code
python ../waam_lab/waam_slicer/main.py     # CLI
python ../waam_lab/waam_slicer/gui.py      # Tkinter GUI
```

Outputs (VTU fields, CSV profiles, plots, `report.html`, logs) are written under `waam_output/<run>/`.

## Repository structure

```
WAAM_IIST/
├── WAAM-Framework/            # Multiphysics FEM simulation framework
│   ├── run.py                 # CLI entry point (--stl / --validate)
│   ├── codegen/               # FreeFEM++ EDP script generator (+ energy audit)
│   ├── core/                  # Heat sources, microstructure, multimaterial, CFD coupling
│   ├── mesh/                  # STL/STEP import, adaptive slicing, conformal tet meshing
│   ├── postprocess/           # Plot engine, HTML report engine, hex conversion
│   ├── gui/                   # Tkinter desktop GUI + 3D studio
│   ├── tests/physics_suite.py # 212 automated physics/codegen checks
│   └── stl/                   # Sample geometries (sphere, cube) + toolpaths
└── waam_lab/                  # Laboratory tooling
    ├── waam_slicer/           # STL -> G-code slicer (CLI + GUI, DDCS V2.1 dialect)
    ├── geometry/              # Part/fixture STL library
    └── docs/                  # Architecture notes, process calculator
```

## Citation

If this work contributes to academic publications, please cite it:

```bibtex
@software{waam_iist_2026,
  title  = {WAAM_IIST: Wire Arc Additive Manufacturing Multiphysics FEM Framework},
  author = {Pranay1004 and the WAAM_IIST contributors},
  year   = {2026},
  url    = {https://github.com/Pranay1004/WAAM_IIST},
  license = {GPL-3.0-or-later}
}
```

## Acknowledgments & Credits

This project stands on open-source shoulders — as with OpenFOAM and other community suites, keeping credits visible is part of the license:

- **FreeFEM++** (<https://freefem.org>) — the finite-element engine this framework drives. Invoked as an **external GPL solver binary**; not bundled with this repository.
- **Goldak double-ellipsoid** heat-source model — M. Goldak, A. Chakravarti & M. Bibby, *A new finite element model for welding heat sources*, Metall. Trans. B 17B (1986).
- **MUMPS** sparse direct solver (used through FreeFEM++), **METIS** partitioning, and the FreeFEM++ community.
- **three.js / OrbitControls** (MIT) — bundled in `WAAM-Framework/gui/assets/` for the in-GUI 3D viewer.
- **OpenFOAM** — the case-driven, solver-separated architecture of this framework (and its optional CFD coupling) follows the design philosophy pioneered by OpenFOAM.
- Scientific stack: **NumPy, Matplotlib, PyVista/VTK, trimesh, gmsh, imageio**.

## License

Licensed under **GNU GPL v3 or later** (see [LICENSE](LICENSE)) — the same license family as OpenFOAM.

| Use case | Allowed under GPL-3.0? |
|---|---|
| Academic research, teaching, internal research computing | ✅ Yes, free of charge |
| Commercial use **inside** a company (run it, don't distribute) | ✅ Yes, free of charge |
| Modifying and **distributing** the software (binaries or as a service bundle) | ✅ Yes — the derivative must remain GPL-3.0-or-later with notices & credits preserved |
| Proprietary closed-source redistribution of derived works | ❌ Not permitted by GPL — contact the authors for a separate commercial arrangement if required |
| Publishing results/derivatives in papers | ✅ Required to credit per *Citation* and *Acknowledgments* above |

Third-party components keep their own licenses (three.js/OrbitControls: MIT; FreeFEM++: GPL, invoked externally and not redistributed here).

```
WAAM_IIST — Copyright (C) 2026 Pranay1004 and the WAAM_IIST contributors.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
```
