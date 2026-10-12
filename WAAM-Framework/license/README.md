# License — WAAM-Framework

**GNU General Public License v3.0 or later (GPL-3.0-or-later)**

Copyright (C) 2026 Pranay1004 and the WAAM_IIST contributors.

Full license text: [`LICENSE`](LICENSE) · Third-party and upstream notices: [`NOTICE`](NOTICE)

---

## 1. What is covered

| Path | License | Copyright holder |
|---|---|---|
| `codegen/`, `core/`, `mesh/`, `postprocess/`, `gui/`, `tests/`, `run.py`, `parametric_sweep.py` | **GPL-3.0-or-later** | Pranay1004 and the WAAM_IIST contributors |
| `license/LICENSE` | GPL-3.0 (verbatim license text) | Free Software Foundation |
| `original_git_file/` (`README.md`, `downloaded.edp`) | **CC BY-NC 4.0** — *retained verbatim, unmodified* | akshansh11 (upstream) |
| `gui/assets/three.min.js`, `gui/assets/OrbitControls.js` | MIT | three.js authors |
| FreeFEM++ (external solver binary) | GPL — **invoked, never redistributed** | FreeFEM++ community |
| OpenFOAM (optional CFD coupling) | GPL — **invoked, never redistributed** | OpenCFD Ltd. |
| NumPy, Matplotlib, PyVista/VTK, trimesh, gmsh, imageio | BSD/MIT-style | respective authors |

```
WAAM-Framework — Copyright (C) 2026 Pranay1004 and the WAAM_IIST contributors.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
```

---

## 2. Upstream provenance — read this first

This framework is a derivative work of **WAAM-MultiLayer-Simulation** by
**akshansh11**, preserved unmodified under `original_git_file/`:

```bibtex
@software{mishra2026waam,
  author       = {Mishra, Akshansh},
  title        = {WAAM-MultiLayer-Simulation},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20753459},
  url          = {https://doi.org/10.5281/zenodo.20753459}
}
```

Upstream is licensed **Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0)** — <https://creativecommons.org/licenses/by-nc/4.0/>.

### 2.1 Why the two licenses coexist

CC BY-NC 4.0 contains a **NonCommercial** field. The GPL deliberately permits
commercial use and forbids *additional* restrictions, so **CC BY-NC 4.0 and
GPL-3.0 are not license-compatible in the direction of relicensing upstream
code as GPL.** This repository therefore does **not** relicense upstream:

* `original_git_file/` is kept byte-for-byte as received and continues to
  carry **CC BY-NC 4.0** in full.
* Only the framework's *own* original Python/E `.edp` source is offered under
  **GPL-3.0-or-later**.
* The two sets remain **separate works distributed together**, and a recipient
  must honour *both* regimes: GPL for the framework code, CC BY-NC 4.0 for the
  upstream reference files.
* The algorithmic ideas (Goldak double-ellipsoid, quiet-element activation,
  progressive torch-triggered birth) are uncopyrightable *methods* and are
  reimplemented from scratch in `codegen/edp_builder.py`; no upstream source
  text is copied into the generator.

### 2.2 Attribution requirement (both regimes)

Any distribution, publication, or derivative must credit:

1. **Upstream** — akshansh11, *WAAM-MultiLayer-Simulation*, Zenodo,
   DOI `10.5281/zenodo.20753459`, CC BY-NC 4.0.
2. **This framework** — Pranay1004 and the WAAM_IIST contributors,
   GPL-3.0-or-later, `https://github.com/Pranay1004/WAAM_IIST`.

---

## 3. Commercial use

| Use case | Framework (GPL-3.0-or-later) | Upstream files (CC BY-NC 4.0) |
|---|---|---|
| Academic research, teaching | Allowed | Allowed with attribution |
| Run internally in a company (no distribution) | Allowed | **Not allowed** — NC field covers "noncommercial purposes" |
| Modify and redistribute (binaries or as a service) | Allowed if the derivative stays GPL-3.0-or-later with notices intact | Allowed only under CC BY-NC 4.0 terms |
| Proprietary closed-source redistribution of a derivative | Not permitted by GPL | Not permitted |
| Publishing results | Credit per §2.2 | Credit per §2.2 |

If a commercial deployment requires the upstream reference files to be
redistributed, contact the upstream author for a separate commercial grant —
this repository cannot relicense them.

---

## 4. Applying this license to new files

Add to the top of every new source file:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Pranay1004 and the WAAM_IIST contributors
```

Run `python run.py --validate` before release; the physics suite
(`tests/physics_suite.py`) is the quality gate.

---

## 5. Disclaimer

This program is distributed in the hope that it will be useful, but
**WITHOUT ANY WARRANTY**; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

Simulation output is a research aid, **not** a certified engineering
prediction. Process parameters must be verified against experiment before any
manufacturing decision is taken from them.
