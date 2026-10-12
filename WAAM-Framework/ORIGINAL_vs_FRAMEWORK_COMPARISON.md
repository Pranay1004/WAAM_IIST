# Original vs. Framework — Detailed Comparison

**Original:** `original_git_file/` — *WAAM-MultiLayer-Simulation* (akshansh11, Zenodo DOI 10.5281/zenodo.20753459, CC BY-NC 4.0)
→ `README.md` (434 lines) + `downloaded.edp` (534 lines) = **968 lines, 2 files**

**Framework:** `WAAM-Framework/` — `waam-framework` v0.1.0
→ **28,595 lines of Python** (40 source files) + **3,571 lines of `.edp` reference modules** + a code generator that emits up to ~3,150-line `.edp` scripts
→ Scale factor: **~40× the code**, ~45× the artifact count

---

## 1. Executive Summary

| | Original | Framework |
|---|---|---|
| **Nature** | One handwritten FreeFEM++ script + a README | Layered Python application that *generates* FreeFEM++ scripts |
| **Workflow** | Edit `.edp` → `FreeFem++ waam_multilayer.edp` → open `waam.pvd` in ParaView | GUI (12 tabs) or CLI → slice STL → mesh → generate `.edp` → solve (serial/MPI) → plots + HTML report |
| **Geometry** | Hardcoded: 26 × 3 mm wall on a 200×100×10 mm plate | Arbitrary STL/STEP/IGES/BREP, parametric walls, G-code driven toolpaths |
| **Heat source** | Goldak double-ellipsoid only | 7 models (Goldak, P48 arc+droplet, Gaussian vol/surf, conical frustum, uniform band, prescribed-T) |
| **Material** | Mild steel S235, hard-coded functions | 8-alloy JSON database (SS308L/316L/304, AlSi10Mg, ER70S-6, IN625, Ti-6Al-4V, Pure Al), dual substrate/deposit |
| **Mechanics** | Single post-hoc **elastic** solve, inherent strain | Incremental **J2 thermo-elasto-plasticity**, Newton w/ consistent tangent, TRIP, springback, fixture BCs |
| **Thermal extras** | Conduction + convection | + Stefan–Boltzmann radiation, boiling cap, enthalpy-exact latent heat, adaptive dt, Picard correctors, inter-pass *target-temperature* control |
| **Beyond original scope** | — | OpenFOAM VOF melt-pool CFD, ultrasonic (UVA/Blaha) assistance, microstructure & Vickers hardness prediction |
| **Verification** | None | 11-group physics suite, J2 cross-check, external CAS-DED benchmark, pytest suite |
| **Portability** | `D:\freefem++\...` hardcoded | Binary discovery + env overrides (macOS/Win/Linux), MPI, self-contained timestamped run dirs |

**In one sentence:** the original is a *research artifact* — a correct, minimal, self-documenting demonstration of quiet-element WAAM heat transfer; the framework is an *engineering platform* that keeps the original's physics as its inner core and rebuilds everything around it (geometry, toolpath, materials, plasticity, UI, verification, reporting).

---

## 2. Original: What It Actually Does

### 2.1 Physics (`downloaded.edp`)
- **3D transient heat equation**, θ-scheme with `thetaT = 0.5` (Crank–Nicolson), P1 elements, UMFPACK direct solver.
- **Goldak double-ellipsoid** moving source: `af=8, ar=20, bG=7, cG=4 mm`, `ff=0.6 / fr=1.4`, tanh front/rear split (`epsSplit=1e-3`), amplitude `Q0 = 6√3·f·P/(π√π·a·b·c)`.
- **Power**: `I=200 A × U=24 V × η=0.8 = 3840 W`, `v_scan = 10 mm/s`, `tPass = 20 s`, `tWait = 60 s`, `Ttotal = 2080 s`.
- **Quiet element activation**: inactive deposit nodes get `k_eff = 1e-4·k`, `Cp_eff = 1e-4·Cp`; progressive activation by `xTorch >= x_node` inside the weld loop; **source masked by `psiActive`** (the README flags this as the #1 correctness requirement — without it, `Cp≈0` gives 10⁷ °C spikes).
- **Temperature-dependent properties**: `kSteel(T)` 60→38→27 W/mK, `CpSteel(T)` with Curie peak of **5000 J/kgK at 720 °C**, solidus 1450 °C / liquidus 1520 °C.
- **Boundaries**: `hBot = 50` on face 1 (bottom), `hConv = 15` on faces 2–6. **No radiation, no latent heat, no boiling cap** — conduction + convection only.
- **Adaptive-but-fixed timestepping**: `dt_weld = 0.333 s` (v·dt ≈ 0.5·hx), `dt_wait = 10 s`, `dt_cool = 120 s × 20 steps`.
- **Mechanical stage (post-hoc)**: accumulate eigenstrain `epsTh += 2.1e-5·ΔT` every step, then one linear-elastic solve with `E(T)` 210→80 GPa, `ν=0.3`, bottom fixed → `waam_mech.vtu` with `Ux Uy Uz DispMag SigMises`.

### 2.2 Mesh
Two `cube()` meshes concatenated with `Th = ThBase + ThDep`:
- Base 30×20×8, deposit 30×6×26, scaled by linear `movemesh3` only (documented Windows/FreeFEM v4.15 workaround — no trig in `transfo`).
- **~11,687 nodes, ~56,880 tets**; one element per layer in Z.

### 2.3 Output
`waam.pvd` + `T_NNNN.vtu` (fields `TempC`, `Activation`) + `waam_mech.vtu` + `waam_data.csv` (8 columns: `time_s, layer, phase, xTorch_mm, zLayer_mm, Tmax_C, TinterPass_C, epsTh_max`), all written to a **hardcoded `D:\freefem++\arc_waam_simulation\`**.

### 2.4 Known quirks in the original
1. **Header/code mismatch** — comments say *"activation: gradual tanh ramp over 5 s"* and *"T = Tliq on activation"*; the code actually switches `psiActive` instantly at `xTorch + af` and sets `TdepInit = Tamb` (deliberately, to avoid double-heating — see the comment at line 92).
2. **Non-conforming interface** — `ThBase + ThDep` merges no nodes (different Y spacing: 5.0 mm vs 1.33 mm), so substrate and deposit are only weakly coupled through FreeFEM's mesh-union bookkeeping.
3. **No radiation / latent heat / evaporation** despite peak temperatures reaching ~1900 °C.
4. **Elastic-only residual stress** — no yield limit, so reported von Mises can exceed the material's strength.

---

## 3. Framework: What Is Different

### 3.1 Architecture — monolith → layered platform

```
ORIGINAL                          FRAMEWORK
────────                          ─────────
downloaded.edp  (534 ln)          gui/app.py (6,959)  ──┐
README.md       (434 ln)          codegen/edp_builder.py (4,992) ← single source of physics truth
                                  core/fluid/openfoam_coupling.py (1,958)
                                  postprocess/plot_generator.py (1,419)
                                  gui/widgets/waam_3d_studio.py (1,331)
                                  core/postprocess/output_manager.py (1,255)
                                  gui/widgets/mesh_viewer_3d.py (1,242)
                                  postprocess/report_generator.py (1,174)
                                  tests/physics_suite.py (1,464)
                                  run.py (785)  ← orchestrator / CLI
                                  mesh/* (2,856) ← slicer, mesher, G-code, CAD, units
                                  core/multimaterial/* (408 + 1,309-line JSON DB)
                                  tests/* (2,885), parametric_sweep.py (324), scratch/*
```

Key structural change: **the physics stopped being the source code and became an emission target.** `EDPBuilder.build()` (`codegen/edp_builder.py:1474`) assembles a 20+-section `.edp` (GEOMETRY → PROCESS → FEEDSTOCK → MATERIAL → TIMESTEP → G-CODE READER → MESH → ADAPTIVE dt → FE SPACES → **3D NONLINEAR THERMO-ELASTO-PLASTIC ENGINE** → ICs → FORMS → PROBES → OUTPUT → RESTART → DEPOSITION LOOP → FINAL COOLING → MECHANICAL → MICROSTRUCTURE → VALIDATION GATE → `solver_results.json`).

Two static `.edp` files survive as hand-maintained presets, not generator output:
- `DEFAULT.edp` / `generated_edps/DEFAULT.edp` (655 ln) — the original's architecture **with fixes**: enthalpy-exact latent Cp band over `[Tsol,Tliq]`, quiet method quenching **only k** (mass retained so diffusivity collapses instead of thermal mass), linearized Stefan–Boltzmann radiation (`emiss=0.7`), boiling cap at `Tboil=2862 °C` with an `EcapJ` energy audit, and an explicit caveat that the mechanical stage is *"post-hoc ELASTIC estimate (no plasticity)"*.
- `generated_edps/uva_waam_ss308l_ss316.edp` (859 ln) — SS316 substrate / SS308L deposit, 33 layers, **ultrasonic vibration assistance** (20 kHz, 15 µm, Blaha yield-softening).

### 3.2 Geometry & meshing — `cube()` → full mesh pipeline

| | Original | Framework |
|---|---|---|
| Input | None (hardcoded) | STL, STEP, STP, IGES, IGS, BREP, PLY, OBJ, G-code |
| Mesher | FreeFEM `cube()` ×2 | Gmsh OCCT volume meshing (`mesh/stl_pipeline.py:541`) **or** pure-Python graded structured mesh (`mesh/structured_conforming.py:464`) |
| Sizing | Fixed 30×20×8 / 30×6×26 | Rule-based engine with 3 strategies + node budget 1500/2500/5000/custom (`mesh/intelligent_mesher.py:29`) |
| Interface | `ThBase + ThDep` → **zero-width crack** | **Single conforming mesh, nodes shared by construction** |
| Labels | Implicit | Enforced convention: 1 = bottom (hBot + fixture spring), 5 = top, 2/3/4/6 = flanks (convection/radiation), shared by mesher and generated forms |
| Quality control | None | Signed-volume / orientation / outward-normal checks; **raises rather than emitting an unusable mesh** (`edp_builder.py:3168`) |

The conforming-mesh work is the framework's most consequential *correctness* fix: the crack at the original-style interface was measured at **19.9 °C mean / 92.9 °C peak** temperature discontinuity, i.e. no real conduction and no mechanical bond between substrate and deposit.

### 3.3 Toolpath — linear scan → slicer + G-code

- **Original:** torch goes `x = 0 → 200 mm`, one direction, every layer, constant `v_scan`.
- **Framework:** `mesh/slicer_api.py` (695) slices geometry into layers → 5 infill patterns (zigzag, meander, contour, crosshatch, unidirectional) with overlap/perimeters/bidirectional reversal → writes `toolpath.gcode` + `toolpath.txt` + `<stem>.layer_times.txt`. `mesh/gcode_parser.py` (411) parses G0/G1/**G2/G3 arcs** into solver arrays.
- The generated `.edp` reads `toolpath.txt` (`t x y z power`) and `layer_pass_times.txt`, gates the arc off on rapid travels (`pState`), supports `dirSign` zigzag, and clamps activation lookup with an "afterProgram" guard so an over-energized arc isn't re-applied after the program ends.

### 3.4 Thermal physics

| Aspect | Original | Framework |
|---|---|---|
| Scheme | θ = 0.5, single solve/step | θ-scheme + **Picard correctors** (3 passes, 0.5 °C tol) for nonlinearity |
| Timestepping | 3 fixed dts (0.333 / 10 / 120 s) | Fixed **or** adaptive Fourier-number controller (Fo target 0.45, torch-advance cap `src_travel_fraction`, runaway guard) |
| Latent heat | Cp spike to 5000 J/kgK at Curie point only; **no fusion latent** | **Enthalpy-exact PWL Cp band** over `[Tsol,Tliq]` preserving ∫Cp dT (fixes ~10 % energy error) |
| Radiation | None | Linearized Stefan–Boltzmann on all outer surfaces (`emiss` per alloy) |
| Evaporation | None | Boiling cap at `Tboil` + `EcapJ` energy accounting |
| Inter-pass control | Fixed `tWait = 60 s` | `time` mode (as original) **or** `target` mode — iterative cooldown to a target peak T (default 400 °C), convergence + iteration cap |
| Quiet elements | `k` **and** `Cp` × 1e-4 | Builder: quiet `k` quench with full mass (diffusivity collapses, thermal mass doesn't); plus optional **prescribed-T birth** (Ansys-style liquidus gate, `_prescribed_T_override`, `edp_builder.py:3290`) |
| Energy audit | `Tmax` printed to console | `solver_results.json` energy balance: `intended_arc`, `injected_arc`, `stored_enthalpy`, `boiling_cap_removed`, `surface_losses`, `quiet_node_discarded`, `tamb_floor_added`, `prescribed_imposed`, `numerical_residual` |

### 3.5 Heat sources

| Model | Original | Framework | Notes |
|---|:--:|:--:|---|
| Goldak double-ellipsoid | ✅ | ✅ | Same `Q0 = 6√3·f·P/(π√π·abc)` and tanh split — direct carry-over |
| Gaussian volumetric | — | ✅ | `Q0 = 3√3·P/(π√π·R³)` |
| Gaussian surface | — | ✅ | Neumann `int2d` BC |
| Conical frustum (keyhole) | — | ✅ | `R(ζ) = rt + (rb−rt)·ζ/dpen` |
| Uniform band | — | ✅ | tanh box |
| Prescribed temperature | — | ✅ | Post-solve `Temp = Tliq` inside activation radius |
| P48 arc/droplet partition | — | ✅ | *Progress in Additive Manufacturing* 2025 10:11647; empirical `σ_q = 6.205e-4·I^0.28` |

All are masked by `psiActive` (original's key insight, preserved). Auto source sizing keeps semi-axes ≥ element size with a fidelity floor (`_auto_size_source`, `edp_builder.py:468`).

`core/heat_sources/*.edp` (6 files) are **standalone reference modules** — explicitly marked *"NOT WIRED INTO THE PIPELINE"*; live physics is whatever `EDPBuilder` emits.

### 3.6 Materials

| | Original | Framework |
|---|---|---|
| Source | `kSteel(T)`, `CpSteel(T)` inline | `core/multimaterial/material_database.json` (1,309 lines) |
| Alloys | 1 (S235/A36 mild steel) | 8: **AlSi10Mg, ER70S-6, IN625, Pure Al, SS304, SS308L, SS316L, Ti-6Al-4V** |
| Data | k, Cp, ρ, Tsol, Tliq, E, ν | + ν, latent heat, emissivity, composition, T-tables for k/E/σy/α, phase transformation, hardness coefficients, wire diameters |
| Assignment | Single material for everything | **Dual**: substrate and deposit alloys independently, with a dissimilar-metal CTE/thermophysical compatibility report |
| Phase kinetics | None | `core/multimaterial/alloy_metallurgy.py` — 5 phase families × 4 phases: Burgers BOR (Ti), WRC-1992 + Hall-Petch (steel), Scheil/Laves (IN625), cellular eutectic (Al), Kirkaldy–Venugopalan + Koistinen–Marburger; IIW CE, WRC Cr/Ni equivalents |

### 3.7 Mechanics — elastic estimate → incremental plasticity

| | Original | Framework |
|---|---|---|
| Formulation | One post-hoc linear-elastic solve | **Incremental J2 thermo-elasto-plasticity**, backward-Euler radial return |
| Tangent | n/a | Consistent tangent Newton (default) or Picard, with force/displacement residuals recorded |
| Hardening | None | Isotropic `h_prime_iso` |
| TRIP | None | `trip_beta = 0.011` transformation strain |
| Remelt | Eigenstrain keeps growing unphysically | **Plastic memory reset on remelt**; above-solidus "slurry" stiffness cut avoids singular dofs |
| Eigenstrain | `epsTh += α·ΔT` accumulated from t=0 (can exceed yield → phantom stress) | Same accumulation **plus** correct eigenstress subtraction; `DEFAULT.edp` documents the phantom-stress fix |
| BCs | Bottom fixed only | Isostatic / fixture-clamped / cantilever, fixture stiffness, **springback with fixture-release regularization**, gravity |
| Discretization | P1 only | P1 or P2; legacy inherent-strain `chi_*` fallback retained but off |
| Output | `waam_mech.vtu`, max values to console | + PEEQ, von Mises per frame, Newton convergence in `solver_results.json`, checkpoint/restart |

### 3.8 New physics domains (absent from the original)

1. **OpenFOAM melt-pool CFD** — `core/fluid/openfoam_coupling.py` (1,958 lines) generates an OpenFOAM 2412 `beamWeldFoam` case: VOF two-phase (metal + atmosphere), diverging-oscillating-Gaussian melting source, Marangoni, vapor recoil, Darcy mushy zone; maps `I·U·η → MELTING/HS_Q`, `v_scan → HS_velocity`, `bG → HS_a`; writes `cfd_history.csv/png` + `cfd_report.html`.
2. **Ultrasonic vibration assistance (UVA)** — 20 kHz / 15 µm; four acoustic modules (streaming, heat generation, cavitation, momentum transfer), amplitude acting on the **yield surface** (Blaha softening), multi-material zone table.
3. **Microstructure & hardness** — G/R solidification map, JMAK kinetics, SDAS laws per alloy family, reheat anneal factors, discrete **Vickers indentation** predictions → `hardness_profile_z.csv`, `hardness_test_locations.csv`.
4. **Marangoni melt pool / vibration acoustics / solidification phases** — reference `.edp` modules under `core/`, not yet pipeline-wired.

### 3.9 Output & postprocessing

| | Original | Framework |
|---|---|---|
| Fields | `TempC`, `Activation`, `Ux Uy Uz DispMag SigMises` | + PEEQ, per-frame mechanics, UVA fields, `VickersHV` |
| Files | `waam.pvd`, `T_*.vtu`, `waam_mech.vtu`, `waam_data.csv` | + `solver_results.json`, `run_summary.json`, `parameters.json`, `probe_points.csv`, `probe_paths.csv`, `layer_pass_times.txt`, hex-VTK conversions, `waam_3d_studio.html` |
| `waam_data.csv` | 8 columns | 8 columns + `HV_layer`; probes recorded with **true simulated times** |
| Plots | None (user writes matplotlib by hand per README) | **8 publication figures** @ 300 dpi: thermal history & peaks, cooling rates/t85, distortion & warpage, von Mises, heat input & dwell energy, baseplate thermal cycle + elevation probes, spatial stress paths (A–A/B–B sections), normalized warpage index |
| Reports | None | `report.html` (2-page A4 "LaTeX PDF-style"), `run_summary.json`, ParaView loader script (`open_in_paraview.py`), `cfd_report.html`, toolpath GIF, uncut 3D renders |
| Visualization | Manual ParaView | Embedded PyVista viewport, standalone PyVista GPU window, **Three.js WebGL studio** (offline-inlined `three.min.js` + `OrbitControls.js`) with heat-source inspector, G-code playback 1×–50×, layer filter; ParaView still one click away |
| History/probes | None | Abaqus-style `*OUTPUT,HISTORY` equivalent: probe **points** and **paths** at solver time steps |

### 3.10 Workflow & UX

| | Original | Framework |
|---|---|---|
| Entry | Edit the file, run one command | `run.py` with **10 modes**: default → GUI; `--cli`, `--stl`, `--slice`, `--validate`, `--validate-full`, `--generate-only`, `--openfoam`, `--uva`, `--version` |
| Parameters | 40+ constants at the top of the file | Live GUI with 8 presets, units-aware mm↔SI, `.waam` project save/load |
| GUI | None | **Tkinter, 12 tabs**: Geometry, Process, Material, Mesh/CAD, Physics, Solver, OpenFOAM CFD, Parametric & DOE, 3D Viewport, Script Preview, Ultrasonic, Simulation Log — plus live console, progress parsing, themes, tooltips |
| Orchestration | Manual | 5-stage pipeline: slice → mesh → generate → solve (serial or `ff-mpirun` MPI) → postprocess, into a self-contained `waam_output/<timestamp>/` with a `latest` pointer |
| Persistence | Rewrite the file | `.waam` projects, `parameters.json` as single source of truth, checkpoint/restart (layer or every-N-steps) |
| Portability | `D:\freefem++\...` | `find_freefem_serial/mpi()` + `FREEFEM_BIN`/`FF_MPIRUN`/`PARAVIEW_BIN` env overrides, macOS app-bundle default, pty wrapper (`script -q /dev/null`) so FreeFEM's stdout flushes line-by-line |

### 3.11 Verification & validation — none → four suites

| Suite | What it checks |
|---|---|
| `tests/physics_suite.py` (1,464 ln) | 11 groups, ~213 assertions: heat-source normalization, latent enthalpy, VOF, ferrite number, builder markers, static-preset integrity, `.edp` module structure, incremental mechanics, history/restart, structured results, **conforming interface** |
| `tests/validate_j2_mechanics.py` (203) | Elastic baseline vs J2 plastic (radial return, TRIP, annealing) on an identical 2-layer mesh |
| `tests/validate_cas_ded_dimeen26.py` (599) | **External benchmark CAS-DEDD-IMEEN26** (Ansys/TU Wien 2026, AlSi10Mg DED): max deformation 0.25 mm, mean 4×10⁻² mm, process temp 580 °C, 50 mm³/s, bead radius 2.5 mm. Runs real FreeFEM → gated behind `--validate-full` |
| `tests/test_wall_probes_ultrasonic.py` (219) | pytest: deposition-pattern codegen, ultrasonic gating, probe stations, plot generation |
| Parametric sweep (324 ln) | 6 cases (I, U, WFS, v, bead dims, n_layers) with incremental `sweep_timing_summary.json` |

### 3.12 Engineering quality

| | Original | Framework |
|---|---|---|
| Packaging | None | `pyproject.toml`, `requirements.txt`, ruff config (line 120, py39) |
| Lint/CI | Manual | `ruff` gates (`E9`, `F63`, `F7`) |
| Error handling | README table of 7 common errors + fixes | Mesher raises instead of writing bad meshes; FreeFEM file-existence guards (`ifstream` throws on missing file); structural validity gate inside the generated `.edp`; MPI rank-0 canonical output |
| Docs | Excellent README (concept, physics, geometry, params, run steps, ParaView recipes, insights, error table, extension table, citation) | Docs are distributed: GUI educational info dialogs (~25), extensive code comments, HTML reports — but **no top-level README** |

---

## 4. Lineage: What Came From the Original

Direct carry-over (recognizably the same code/idea):

1. θ-scheme 3D transient heat variational form — same structure as `aThermal`/`lThermal` in `downloaded.edp:199–225`.
2. Goldak double-ellipsoid with tanh front/rear split and `ff=0.6/fr=1.4` — same algebra.
3. Quiet-element activation with `psiActive` source masking — the original's central correctness insight, preserved everywhere.
4. Progressive x-based node activation inside the weld loop.
5. Inter-pass cooling cadence and the weld/wait/cooldown stage structure of the deposition loop.
6. Cumulative thermal eigenstrain accumulation `epsTh += α·ΔT`.
7. Boundary label scheme (1 = bottom, 5 = top, 2/3/4/6 = sides) and the PVD/VTU/`waam_data.csv` output contract.
8. Console progress format (`--- Layer n/N`, `ws=… Tmax=…`, `Inter-pass`) — the GUI's progress bar parses *the original's log lines* (`gui/app.py:6112–6151`).
9. Default geometry retained: `n_layers = 26`, 200 mm length, 8 mm bead width, 200 A / 24 V / η 0.8 / 10 mm/s.

Evolved or corrected:

| Original | Framework |
|---|---|
| `Cp_eff` and `k_eff` both × 1e-4 | Quench only `k`, keep full mass (diffusivity collapses, thermal mass doesn't) |
| No latent heat | Enthalpy-exact PWL Cp band over `[Tsol, Tliq]` |
| No radiation / boiling cap | Stefan–Boltzmann + `Tboil` clip with `EcapJ` audit |
| Eigenstrain may exceed yield → phantom stress | Correct eigenstress subtraction / J2 plasticity |
| `ThBase + ThDep` non-conforming interface | Single conforming mesh |
| Hardcoded `D:\` output | Portable run dirs |

---

## 5. Gap Analysis — What the Framework Still Lacks

1. **No top-level README** comparable to the original's (which documents physics, geometry, ParaView recipes, error table, extensions, citation). Knowledge is inside GUI dialogs and code comments.
2. **`core/heat_sources/*.edp` and `core/{thermal,mechanical}` are unwired** — `core/thermal/` and `core/mechanical/` are empty; reference modules are never executed by the pipeline (only grepped by the physics suite).
3. **`waam_parametric_sweep_comparison_report.pdf` is not reproducible** — no code in the repo generates it.
4. **Two mm-detection heuristics coexist** — `mesh/units.py` per-key thresholds vs `run.py:apply_preset_to_params` blanket `>0.05`/`>0.1`.
5. **`test_wall_probes_ultrasonic.py` is not in `--validate`** — pytest-only, so routine health checks miss it.
6. **Multi-material per-layer alloy switching (`material_schedule`) is declared but unimplemented** — only zonal base/deposit split works.
7. **Report is HTML, not PDF** despite "LaTeX PDF-style" styling.
8. **License**: original is CC BY-NC 4.0 with attribution requirement. The framework derives from it but carries **no LICENSE or attribution file** — this should be reconciled.

---

## 6. Verdict

| Dimension | Original | Framework |
|---|---|---|
| Correctness of core thermal solve | ✅ Good (with 4 known physical omissions) | ✅ Stronger (conforming mesh, energy audit, latent heat, radiation) |
| Fidelity of mechanics | ⚠️ Elastic estimate only | ✅ Incremental J2 + validation |
| Breadth of physics | ⚠️ Heat + eigenstrain | ✅ + CFD, UVA, microstructure, hardness |
| Realism of geometry/toolpath | ❌ Fixed wall, straight scan | ✅ CAD/G-code, 5 infill patterns |
| Usability | ⚠️ Edit-and-run, ParaView manual | ✅ GUI, reports, 3D viewers, projects |
| Verification | ❌ None | ✅ 4 suites + external benchmark |
| Documentation | ✅ Excellent single README | ⚠️ Fragmented |
| Accessibility of the physics | ✅ Read 534 lines, understand everything | ❌ 4,992-line generator to trace one equation |
| Dependency weight | FreeFEM++ only | FreeFEM + Gmsh + PyVista + VTK + matplotlib + three.js (+ optional OpenFOAM, pywebview) |

**Bottom line:** the framework is a superset of the original — it retains the original's solver engine (FreeFEM++), Goldak/quiet-element physics, output contract, and even its log format, then adds everything a usable simulation platform needs. Its main regression relative to the original is *legibility*: 534 self-contained, heavily commented lines became a 4,992-line generator plus 40 Python modules, and the original's exemplary top-level README was not carried forward.
