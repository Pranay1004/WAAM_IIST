# WAAM_IIST

**Wire Arc Additive Manufacturing (WAAM) — multiphysics FEM simulation framework & laboratory tooling**

[![CI](https://github.com/Pranay1004/WAAM_IIST/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranay1004/WAAM_IIST/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

This repository bundles two complementary projects:

| Project | What it is |
|---|---|
| [`WAAM-Framework/`](WAAM-Framework/) | Production-grade **simulation pipeline**: STL/STEP → adaptive slicing → conformal tetrahedral meshing → **FreeFEM++** transient thermal–mechanical FEM (Goldak double-ellipsoid heat source, quiet-element activation, MUMPS direct solver) → energy-balance audit → publication-grade postprocessing (plots, thermocouples, build animation, self-contained HTML report) + Tkinter GUI. |
| [`waam_lab/`](waam_lab/) | **Laboratory tooling**: a standalone STL → G-code **WAAM slicer** (perimeter + zigzag infill, DDCS V2.1 motion-controller dialect), CAD/geometry library, process documentation and calculator. |

---

## Highlights

- **End-to-end CLI**: `python run.py --stl <geometry.stl>` drives geometry ingest, slicing, meshing, code generation, solving, and postprocessing in one command.
- **Physics**: Goldak double-ellipsoid (plus Gaussian-volumetric / conical / uniform / prescribed-temperature options), latent-heat enthalpy method, convection + radiation boundaries, boiling-temperature cap, quiet-element activation for not-yet-deposited material, inherent-strain mechanical stage, phase-transformation → Vickers-hardness prediction.
- **Honest numerics**: every run prints an **energy-balance audit** (arc energy → captured / lost / quiet-clamped / stored, with residual %) instead of hiding quadrature error.
- **Performance**: MUMPS direct solver (≈3.8× faster than UMFPACK on an 83k-DoF thermal solve; 251k-DoF mechanical solves in <7 GB).
- **Validation**: automated **62/62 physics & code-generation checks** (`tests/physics_suite.py`) runnable via `python run.py --validate`.
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

# 3. Run the validation suite anytime (62 checks)
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
│   ├── tests/physics_suite.py # 62 automated physics/codegen checks
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
