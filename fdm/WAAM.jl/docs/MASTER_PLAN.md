# WAAMFlow.jl — Master Engineering Plan

**Status:** v0.1 design baseline · **Owner:** WAAMLab · **Date:** 2026-10-07

A from-scratch, open-source Julia implementation of a free-surface, multiphase,
finite-volume solver for **long-term Wire Arc Additive Manufacturing (WAAM / WA-DED)**
thin-wall deposition, reproducing and extending:

> H. H. Zargari, S. A. Khajeh Mohammadilar, V. Ziaei Laleh (2025),
> *A long-term and pragmatic numerical simulation of wire arc additively manufactured
> ER70S6 and ER316L thin-wall via finite difference method.*
> **Progress in Additive Manufacturing** 10:11647–11663.
> [doi:10.1007/s40964-025-01307-x](https://doi.org/10.1007/s40964-025-01307-x)

---

## 0. Executive Summary

Conventional WAAM simulation splits into two camps that each fail on something
important:

| Approach | Strengths | Fatal weakness |
| --- | --- | --- |
| **FEM** (thermal + J2 mechanics) | Multi-pass, long-duration, residual stress & distortion | **Cannot predict free-surface geometry, melt-pool flow, or sagging.** Bead shape must be assumed *a priori* via element birth/death. |
| **Production CFD** (Fluent / FLOW-3D / OpenFOAM) | Free-surface VOF, Marangoni, real bead geometry | **Millisecond-scale single tracks.** Multi-pass thin-wall histories are computationally out of reach. |

Zargari et al. bridge the gap: a *pragmatic* FDM + VOF model that sustains **25 passes
over 3400 s** and reproduces experimental wall thickness to within a few percent,
while still resolving the melt pool well enough to predict **sagging collapse**.

**The one-sentence thesis of this project:** make that capability fast, open,
testable, and *provably correct* at the operator level, so a user can run it on a
laptop in minutes and trust the number.

### 0.1 Non-negotiable design principles

1. **Verification before validation.** Every operator is proven correct against
   manufactured/analytical solutions *before* it is ever pointed at welding.
2. **Conservation is a feature, not a hope.** Mass and energy budgets are closed and
   *reported* at run time; violations are surfaced, not silently absorbed.
3. **No black boxes.** Every physical coefficient carries a provenance tag
   (paper equation / literature citation / digitized figure).
4. **Structured-grid first.** We do not attempt unstructured CFD until the structured
   solver is verified. See §2.1 for why this is the correct sequencing.
5. **The paper's internal inconsistencies are documented and configurable**, not
   papered over. See §3.3 — this is the single highest-value thing this project adds.

---

## 1. What the paper actually specifies (the source of truth)

Extracted from the full text. This section is the contract the code is measured
against; anything not traceable here or to primary literature must be labelled
`Assumption`.

### 1.1 Process parameters (experimental, Sec. 2)

| Symbol | Value | Note |
| --- | --- | --- |
| Process | P-GMAW | pulsed, on Merkle PU520 DW + robotic carrier |
| Nominal current `I` | 125 A | |
| Effective current | 90 A | |
| Polarity | DCEP | |
| Duty cycle | 30 % | |
| Frequency | 60 Hz | |
| Voltage | 16 → 22 V | |
| Average wire feed speed | 4.7 m/min | 1.2 mm solid wire |
| Torch travel speed `TS` | 0.17 m/min = 2.833 mm/s | |
| CTWD | 18 mm | |
| Shielding gas | 99 % Ar, 15 L/min | |
| Substrate | St37, 8 mm thick | |
| Per-pass height increment | ≈ 1.8 mm | imposed kinematically |
| Inter-pass rule | air-cool to 100 °C before next pass | |
| Dwell | 5 s … 180 s (flexible) | study grid: 5 s and 100 s |
| Pass time | 40 s per pass | |
| Total (25 passes) | 3400 s | |

### 1.2 Governing equations (Sec. 3.1, Eqs. 1–6)

Mass, momentum, energy, VOF solved **sequentially**:

```
(1)  ∇·V = ṁ_s / ρ
(2)  ∂V/∂t + V·∇V = -∇P/ρ + μ∇²V/ρ - k_d V + G_b + (ṁ_s/ρ)(V_s - V)
(3)  ∂h/∂t + V·∇h = ∇·(K∇T)/ρ + ḣ_s
(4)  ∂F/∂t + ∇·(V F) = Ṁ_s
(5)  k_d = s_d · f_s² / [(1 - f_s)³ + ε₀]
(6)  h(T) piecewise in [Ts, Tl] absorbing L_f
```

- Darcy drag (Eq. 5): `s_d ≈ 1e6 s⁻¹`, regularization `ε₀ = 1e-4`.
  *(Note: the paper's Eq. 5 prints `kd = sd (fs)³` in one rendering and `f_s²` in
  another; Carman–Kozeny mushy-zone literature universally uses **f_s²/(1-f_s)³**.
  We implement f_s²/(1-f_s)³ and expose the exponent.)*
- Buoyancy (Eq. 15): `G_b = -g β_t (T - T_l)` (Boussinesq).

### 1.3 Boundary conditions / sources (Sec. 3.2, Eqs. 7–16)

```
(7-9)   -K ∂T/∂n = h_c (T - T_0) + ε σ_SB (T⁴ - T_0⁴)
(10)    η Q = η Q_arc + Q_drop  →  η U I = η(1-η_d) U I + η η_d U I
(11)    q_arc = [η(1-η_d) U I / (ξ π σ_q²)] · exp( -[(x-xc)² + (y-yc)²] / (ξ σ_q²) )
(12)    Q_drop = (4/3)π r_d³ f_d [ ρ_l C_l (T_d - T_l) + ρ L_f + ρ_s C_s (T_s - T_0) ]
(14)    μ ∂V_t/∂n = -(∂γ/∂T)(∂T/∂t)          (Marangoni, tangential BC)
(16)    σ_q = 6.205e-4 · I^0.28      [m, I in A]
```

Numeric anchors: `η = 0.68`, `η_d = 0.25`, `U I ≈ 2165 W`, `σ_q(125 A) ≈ 2.40 mm`,
`T_d = 1900 K`, `v_d = 0.7 m/s` (down), droplet injection height 1.5 mm,
`f_d = 60 Hz`, `h_c = 20 W/(m²·K)`, `ε = 0.65…0.70`, `σ_SB = 5.67e-8`.

### 1.4 Material data (Sec. 3.3, Fig. 5)

| | ER70S6 | ER316L |
| --- | --- | --- |
| `T_sol` | 1698 K | 1675 K |
| `T_liq` | 1778 K | 1708 K |
| `K` range | 27 – 59 W/(m·K) | 14 – 32 W/(m·K) |
| `γ(T)` | Sahoo–DebRoy pseudo-binary **Fe–S** | Pichler, EMF levitation |
| `∂γ/∂T` sign | **sulfur-dependent**, may flip | **negative** over range |

```
ER70S6 (Sahoo & DebRoy 1988):
  γ = γ°_m - A (T - T_m) - R T Γ_s ln[1 + k₁ a_s exp(-ΔH°/RT)]
  γ°_m = 1.943 N/m, A = 4.3e-4 N/(m·K), Γ_s = 1.3e-5 mol/m²,
  k₁ = 3.18e-3, ΔH° = -1.66e5 J/mol, a_s ≈ 0.018 wt.% S

ER316L (Pichler et al. 2022):
  γ = 1.77 - 3.5e-4 (T - 1700)      [N/m]
```

### 1.5 Discretization (Sec. 3.3)

- **Structured, non-uniform Cartesian**, half-symmetry plane.
- 0.6 mm cuboid cells through the fusion zone; graded up to 1.8 mm far field.
- FLOW-3D free-surface engine + **in-house Fortran FDM** (Intel `ifort`) for the
  heat flux.
- **Lagrangian VOF advection** (Barkhudarov 2004, ref. 24).

### 1.6 Validation targets (Sec. 4)

| Metric | Paper result |
| --- | --- |
| S-B100 effective thickness | sim 6.2 mm vs exp 5.3 mm → **17 %** over-prediction (≈14 % excluding top third) |
| SS-B100 effective thickness | sim 6.3 mm vs exp 6.2 mm → **2 %** |
| Overall | abstract claims **< 10 % average** error |
| Sagging, ER316L bidirectional 5 s (SS-B5) | **pass 3, t = 119 s** — earliest failure |
| Sagging, ER70S6 5 s (S-U5/S-B5) | pass 4, t ≈ 140 – 150 s |
| Sagging, dwell = 100 s (all 4) | **no sagging** through 10 or 25 passes |
| Molten-pool volume S-B100 vs SS-B100 | S is **35–40 % smaller** |
| Sagging mechanism | finger-type → cylindrical pool; `P_st = γκ` spike at weld toes, ≈250–350 Pa |

---

## 2. Assessment of the Gemini plan

The Gemini plan (`julia_waam_framework_plan.md`) is a competent *generic CFD*
blueprint. Adopted verbatim it would **not** reproduce this paper, and would in
several places contradict its central thesis. Verdict per item:

| # | Gemini item | Verdict | Rationale & required change |
| --- | --- | --- | --- |
| 1 | *Unstructured collocated FVM* | ⚠️ **Change** | Paper validated on a **structured graded Cartesian** grid. Collocated storage needs **Rhie–Chow** to suppress checkerboard, which is fragile precisely in multiphase free-surface cells where it matters most. A **staggered MAC** grid removes the need for Rhie–Chow entirely and is *the* right choice for VOF advection (fluxes land on faces). Unstructured is deferred to Phase 4 (§7). |
| 2 | *Goldak double-ellipsoid volumetric heat source* | ❌ **Reject** | Directly opposed to the paper. The paper's §1 explicitly criticises volumetric sources: *"In fusion welding heat is transferred as a flux across the weld pool's surface, with no volumetric heating occurring."* The model uses a **Gaussian surface flux** (Eq. 11). Goldak is kept only as an optional *comparison mode*. |
| 3 | *CSF, κ = −∇·n* | ⚠️ **Extend** | CSF is a valid way to apply γκ in the bulk, but the paper applies surface tension as a **free-surface boundary condition** (Eq. 14) and computes an *equivalent surface tension pressure* as a **diagnostic**. We implement the BC form (faithful) **and** surface the diagnostic `P_st = γ(T)κ` as a first-class output — it is the paper's own proposed sagging criterion (§4.6). |
| 4 | *MULES explicit VOF advection* | ⚠️ **Offer both** | Paper uses **Lagrangian VOF**. MULES is an alternative. Ship Lagrangian (default, conservative by construction) and MULES-style explicit compression as an option; unit-test both to conserve volume to < 1e-6. |
| 5 | *PIMPLE outer/inner loops* | ⚠️ **Change** | PIMPLE is for transient incompressible CFD at large Courant numbers. Here the free surface is quasi-steady within a step. **PISO on a MAC grid** is simpler, cheaper, and standard for VOF solvers. |
| 6 | *TVD convection (van Leer/Gamma)* | ✅ **Accept** | Required to suppress temperature/velocity oscillations near the sharp thermal front. Implemented for the energy equation. |
| 7 | *Enthalpy–porosity + Darcy damping* | ✅ **Accept** | Matches Eqs. 5–6. |
| 8 | *Crank–Nicolson* | ⚠️ **Change** | CN is **A-unstable**: it rings for the stiff, near-discontinuous thermal front at the solidification front. Use **backward Euler** for the stiff frozen/cooling phase and **BDF2** for smooth phases. |
| 9 | *Droplet model* | ❌ **Severely incomplete** | The entire droplet sub-model (r_d, f_d, T_d, v_d, injection height, Eq. 12) is missing. It is ~25 % of the total heat input. |
| 10 | *Long-term multi-pass, dwell, deposition strategy* | ❌ **Missing** | The paper's headline capability. U/B strategies, dwell control, 3400 s timeline, per-pass 1.8 mm growth — all absent. |
| 11 | *Half-symmetry domain, graded mesh* | ❌ **Missing** | Halves the cost; directly enables the 3400 s claim. |
| 12 | *Provenance of coefficients* | ❌ **Missing** | No traceability from code to paper equation / citation. We add a `Provenance` system. |
| 13 | V&V: lid-driven cavity, static droplet, Stefan | ✅ **Accept, expand** | Good core; needs MMS, grid convergence, Rayleigh–Bénard, droplet-impact, 1-D advection, conservation audits, and the WAAM-specific validation matrix (§6). |
| 14 | `Documenter.jl`, GH Actions, General Registry, `WriteVTK.jl`, Python interop | ✅ **Accept** | Agreed, with caveats in §8 (registry name collision check, Julia ≥ 1.10 compat). |

### 2.1 Why structured-first is not a retreat

- The paper's 3400 s / 25-pass claim is only credible on a structured grid; that
  *is* the contribution. Re-deriving it on an unstructured mesh first would destroy
  the one result we can validate against.
- Collocated + unstructured + VOF + surface tension + mushy zone is a research-scale
  code. Structured MAC delivers 90 % of the engineering value at 30 % of the risk.
- Structured grids vectorize and thread trivially (the real cost driver for 3400 s).
- Unstructured becomes tractable *later* because the physics kernels (§5) are written
  against a small operator interface, not against a grid type.

---

## 3. Correcting the paper: internal inconsistencies we must handle

A public tool must not silently inherit errors from its source. Every one of these is
**encoded as a test** and **exposed as a configurable option**.

### 3.1 Darcy drag exponent
Eq. 5 is ambiguous between `f_s³` and `f_s²`. Carman–Kozeny is `f_s²/(1-f_s)³`.
→ `darcy_exponent` option, default Carman–Kozeny.

### 3.2 Drag coefficient magnitude
`s_d = 1e6 s⁻¹` with cell size 0.6 mm gives a damping velocity
`u ≈ Δx/s_d = 6e-7 m/s` — effectively a *hard freeze*, not a porous medium.
→ Exposed as `s_d`; a `porous_exponent_scaling` option allows `s_d ∝ μ/K₀/C²`
for a physically-scaled mushy zone. Documented in the codebook.

### 3.3 Energy budget does not close ⚠️ *most important finding*

Take the paper's own numbers and the two heat-split equations (10) and (12) are
**mutually inconsistent by a factor ≈ 2.5**:

```
Wire feed:  4.7 m/min × π(0.6 mm)² × 7850 kg/m³ = 6.95e-4 kg/s  → ṁ_droplet
ṁ (from droplet volume, Eq. 12)  = ρ_l (4/3)π r_d³ f_d
   self-consistent radius:       r_d ≈ 0.73 mm

Q_drop  = ṁ [ C_l(T_d − T_l) + L_f + C_s(T_s − T_0) ]
        = 6.95e-4 [ 800·(1900−1778) + 272000 + 680·(1698−298) ]
        = 6.95e-4 × 1.342e6   ≈ 922 W

η η_d U I = 0.68 × 0.25 × 2165 = 368 W

          ⇒ Q_drop / (η η_d U I) ≈ 2.5
```

Two defensible readings, so the solver supports both and **always reports closure
error** (`energy_budget` diagnostic):

- **Mode A — literature split (paper-faithful).** Fix `η_d = 0.25`; surface flux
  carries 1104 W, droplets 368 W. The droplet *thermal* source is then inconsistent
  with the droplet *mass* source. Reports the 2.5× discrepancy.
- **Mode B — energy-consistent (default).** Fix the mass source from the wire feed
  rate (this is what governs wall geometry), derive `Q_drop` from it, and set the
  surface flux to the remainder `η·UI − Q_drop`. Budget closes to round-off.

Mode B is the default because it is the only one where the deposited wall volume and
the thermal history are mutually consistent — but Mode A is one config flag away and
is required to reproduce the published numbers.

### 3.4 `Q_drop` dimensions
Eq. 12 as printed adds a mass-specific bracket (`C_l ΔT + L_f + C_s ΔT`) to
density-scaled terms. Only the mass-specific form is dimensionally homogeneous.
→ We implement the consistent form (above) and document the deviation.

### 3.5 `ξ` in the Gaussian flux
Eq. 11 carries an undefined `ξ` in both numerator and exponent. Cannot be
reconstructed from the text. Default `ξ = 1.0`, flagged as an unresolved parameter.

### 3.6 Thermophysical property tables
Fig. 5 is a figure; no machine-readable data ships with the paper. All `K`, `μ`, `C_p`,
`ρ` curves are built from the **cited primary sources** (Hamed Zargari et al. 2020
[17]; Pichler et al. 2020/2022 [52][54]) and each carries a provenance tag.
A `digitize` utility + notebook is provided so a user can override with their own
transcription of Fig. 5 and re-run validation.

### 3.7 Sagging timing is not reproducible from the paper
The paper itself concedes (§4.4) it *"remains unfeasible"* to pinpoint the precise
sagging moment. Validation therefore targets **pass index** and **qualitative
mechanism** as hard assertions, and **time to sag** only as a soft (±pass) check.

---

## 4. Architecture

### 4.1 Module map

```
WAAMFlow
├── mesh/          Grid (non-uniform Cartesian, MAC), geometry, symmetry, block structure
├── fields/        VolumeField / SurfaceField / VectorField, interpolation, flux operators
├── materials/     ThermophysicalDatabase (T-dependent, provenance-tagged),
│                  SurfaceTension (Sahoo–DebRoy, Pichler), Composition
├── physics/       Enthalpy·liquidFraction, DarcyDrag, Buoyancy, MarangoniBC, SurfaceTensionForce,
│                  VOF (advect + source), DropletSource, GaussianArcFlux, SurfaceLosses
├── solvers/       Pressure (PISO + void/air region), Energy (BE/BDF2), Momentum, VOFStep,
│                  SparseLDU (CSR + ILU0/ILUT + PCG), Multigrid
├── process/       DepositionSchedule (U/B, dwell, thermal-interpass rule), TorchPath, RunDriver
├── diagnostics/   WallHeight, EffectiveThickness, PoolVolume, SurfaceTensionPressure,
│                  SaggingDetector, Probes, EnergyBudget, MassBudget
├── io/            VTK (.vtu/.vtr/.pvd) readers & writers, STL/STEP import, CSV metrics
├── config/        TOML-driven `SimulationConfig` with validation + defaults
└── ext/           optional: HDF5, MPI (multi-node)
```

### 4.2 Grid and storage

- **Staggered MAC**: `u` on x-faces `(nx+1, ny, nz)`, `v` on y-faces, `w` on z-faces;
  `p`, `T`, `h`, `F`, `f_s` cell-centred `(nx, ny, nz)`.
- **Non-uniform but orthogonal**: per-axis node coordinates → face spacings `Δx[i]`,
  cell volumes `V = Δx·Δy·Δz`, exact face areas, exact face-normal distances for
  two-point flux coefficients on stretched grids.
- **Half-symmetry**: reflection BC on `y = 0` (`∂/∂y = 0`), halving cost.
- **Two-level blocking**: (a) *cartesian blocking* — decompose the half-domain into
  `4×4×4` blocks, boundary-face exchange is a memcpy; (b) *physical activity lists* —
  because the arc is a ~2 mm spot moving along a line, only O(10⁴) of O(10⁷) cells
  are ever liquid or near-solidified. Everything else is pure conduction handled by
  an implicit solve, so **flux-at-rest cells are skipped entirely**.

### 4.3 Time integration — the 3400 s problem

3400 s with a 0.6 mm cell and a free surface is ~10⁸ explicit steps. That is
infeasible, and the paper's own answer is "pragmatic simplification". Ours is
explicit and *auditable*:

| Regime | Condition | `dt` | Fields advanced |
| --- | --- | --- | --- |
| **Arc-on, liquid** | `T_max > T_sol` near torch | CFL-limited, ~1e-4 – 1e-3 s | all |
| **Solidification** | `f_l → 0` | CFL on residual motion | energy only, then freeze |
| **Frozen / dwell** | `T_max < T_sol` | diffusion-limited, **1 – 10 s** | energy only (implicit) |

`TimestepController` enforces, in priority order: CFL, viscous, surface-tension
(Capillary), phase-change (Stefan), diffusion, and truncation-error limits. The
regime is reported so a user knows when the solver was in "cheat mode".

### 4.4 Solver stack per step

```
1. update phase state      f_s, β, ρ from T  (enthalpy–porosity)
2. update material         K, μ, γ at faces  (T-dependent, face-interpolated)
3. external fluxes         Gaussian arc (11), convection+radiation (8,9)
4. droplet source          mass + momentum + enthalpy (12)  [Mode A/B, §3.3]
5. momentum predictor      PISO outer loop
6. pressure                PISO inner loop, void-region handling, γκ source
7. Marangoni / CSF         free-surface tangential stress (14), γκ bulk force
8. VOF advection           Lagrangian (default) | MULES (option), + source
9. energy                  BE (stiff) / BDF2 (smooth), enthalpy form
10. diagnostics & I/O      height, thickness, P_st, budgets
```

### 4.5 Pressure / free-surface treatment

Two modes, both implemented and tested against each other:

- **`partialVoid` (default, paper-faithful).** Single global Poisson solve; cells with
  `F < F_void` are excluded from the pressure system and pinned `p = p_atm`. Surface
  tension enters as the interface condition. This mirrors FLOW-3D.
- **`csfSource`.** Solve everywhere, add `σ κ ∇χ` as a body force (OpenFOAM
  `interFoam` style). Simpler, but drags the void air into the momentum equation.

Consistency between the two modes is a **verification test** (§6.2, T7).

### 4.6 `SurfaceTensionPressure` as first-class output

`P_st = γ(T) · κ` is computed every step at every interface cell and written to VTK.
The paper proposes its **critical value** as the wall-stability criterion, so it
gets a dedicated diagnostic, a CSV time series, and a `SaggingDetector` that raises
a warning when `max P_st` crosses a user threshold or when the wall footprint stops
monotonically growing. This turns a figure in the paper into an automated tripwire —
the most transferable single feature of the project.

---

## 5. Physics kernels to implement (traceable to paper)

| Module | Equation | Notes |
| --- | --- | --- |
| `enthalpy` | (6) | Piecewise `h(T)`, `T_sol`, `T_liq`, `L_f`; inverse `T(h)`; Boussinesq-consistent `f_l`. |
| `darcy_drag` | (5) | `k_d = s_d f_s^ε / ((1-f_s)³ + ε₀)`, added **implicitly** to the momentum diagonal. |
| `buoyancy` | (15) | Boussinesq, `β_t` per material. |
| `continuity` | (1) | `∇·V = ṁ_s/ρ`. |
| `momentum` | (2) | Divergence form, conservative face fluxes. |
| `energy` | (3) | Face-interpolated `K(T)`, `h`-form ⇒ exact energy conservation. |
| `vof_advection` | (4) | Lagrangian parcels + geometric (volume-conserving) reconstruction; MULES option. |
| `vof_source` | (4) | `Ṁ_s` from droplet mass rate / droplet volume. |
| `arc_flux` | (11) | Gaussian, moving centre `x_c = x_0 + V_w t`, `σ_q` from (16). |
| `droplet_source` | (12) | Mass, momentum `(ṁ/ρ)(V_s - V)`, enthalpy. `V_s = (0,0,-0.7)`. |
| `surface_losses` | (8,9) | Convection + Stefan–Boltzmann radiation, upward normals only. |
| `marangoni` | (14) | `μ ∂u_t/∂n = -(∂γ/∂T)(∇_s T)`, discretized as a face stress. |
| `surface_tension` | — | `κ = -∇·n̂`, `P_st = γ(T)κ`, surface stress `σ κ`. |
| `deposition` | — | Per-pass 1.8 mm growth, torch path, U/B alternation, dwell, inter-pass cool-to-100 °C. |

### 5.1 Material database design

```julia
struct Material
    name::String
    Tsol::Float64; Tliq::Float64
    Lf::Float64                       # J/kg
    rho_s::Float64; rho_l::Float64
    Cp_s::PropCurve; Cp_l::PropCurve   # monotone piecewise-linear in T, monotone-extended
    k_s::PropCurve;  k_l::PropCurve
    mu_l::PropCurve
    beta_T::Float64
    gamma_model::GammaModel            # SahooDebRoyFeS | Pichler316L | Linear | Constant
    provenance::Dict{Symbol,String}    # e.g. :gamma => "paper Eq.13 / Sahoo&DebRoy 1988"
end
```

`PropCurve` is a monotone piecewise-linear interpolant with **automatic tangent
continuation outside the data range** — essential because the arc exceeds the tabulated
range and an unclamped extrapolation silently kills the physics. `provenance` is
surfaced by `material_report()` and exported into every VTK run directory so results
are self-documenting.

---

## 6. Verification & Validation plan

The industry-standard framing (Roache, AIAA G-077-1998) is used:
*verification* = solving the equations right; *validation* = solving the right equations.

### 6.1 Verification (Phase 1 — no physics assumptions, exact answers)

| ID | Test | Method | Tolerance |
| --- | --- | --- | --- |
| V1 | Mesh invariants | `Σ_f Sf = 0` per cell; `V_c = Σ_f (Sf·x_f)/3`; face-count parity | exact (`< 1e-14` rel) |
| V2 | Operator convergence | MMS for `grad/div/laplacian` on the **stretched** grid | 2nd order, `>1.9` |
| V3 | Interpolation monotonicity | `lin`, `minmod`, `superbee`, `vanLeer` — TVD + bounded | `min ≤ ũ ≤ max` exact |
| V4 | Flux exactness | face flux of `u = x`, `u = y`, `u = z` on stretched grid | machine eps |
| V5 | 1-D linear advection | upwind + all TVD limiters, exact transport solution | `L1 < 1e-6` |
| V6 | Poisson Dirichlet | manufactured `p = x² + y² + z²`, exact rhs | `L∞ < 1e-10` |
| V7 | Poisson Neumann | compatibility `Σ rhs = 0` | `‖rhs‖₁ < 1e-12` |
| V8 | Linear solver | CSR mat-vec vs dense reference; PCG convergence on 1e6 dof | `< 1e-10` resid |
| V9 | Enthalpy ↔ temperature | round-trip `T(h(T)) = T` across the mush; `h` continuity at `T_sol`,`T_liq` | `< 1e-12` |
| V10 | Liquid fraction limits | `f_l(T_sol)=0`, `f_l(T_liq)=1`, monotonic, `[0,1]`-bounded | exact |
| V11 | **1-D Stefan problem** | classical Neumann similarity solution, latent-heat plateau | `< 2 %` interface pos. |
| V12 | **Static droplet / Laplace** | `Δp = σ/R` for a sphere; steady; spurious currents bounded | `Δp` within `1 %`; max spurious `‖u‖ < 1e-3 σ/√(ρR)` |
| V13 | **Lid-driven cavity** | Ghia et al. (1982) tabulated `u,v` at Re = 100 … 10⁴ | `< 3 %` L2 |
| V14 | **Rayleigh–Bénard** | onset `Ra_c = 1707.76`, Nusselt `Nu = 1` at onset | `< 2 %` |
| V15 | Droplet impact | Martin & Moylan (2002) spreading/bouncing | `< 5 %` spread factor |
| V16 | VOF volume conservation | closed box, pure advection, no sources | drift `< 1e-6` rel |
| V17 | VOF boundedness | `min ≥ −tol`, `max ≤ 1+tol` throughout | `tol = 1e-10` |
| V18 | Partial-void vs CSF | identical droplet setup, both pressure modes | `< 5 %` on `Δp`, `κ` |
| V19 | **Global mass budget** | closed domain, `Σ ṁ` vs `d(Σm)/dt` | `< 0.1 %` |
| V20 | **Global energy budget** | `Σ ḣ_in` vs `d(Σ H)/dt + losses` | `< 0.5 %` |
| V21 | Grid convergence | driver problem on 3 meshes, Richardson extrapolation | observed order `>1.8` |
| V22 | Time-step convergence | driver problem, halving `dt` | observed order `>1.8` |
| V23 | Thread-independence | `1/2/4/8` threads, identical setup | `< 1e-12` rel diff |
| V24 | Determinism | same seed & config, two runs | bitwise identical |

V11, V12, V13, V14, V15, V16 are the **gating suite** — no release without them.
V20 and V23/V24 are the ones that catch "works on my machine".

### 6.2 Validation (Phase 3 — against the paper and literature)

| ID | Test | Target | Tolerance |
| --- | --- | --- | --- |
| P1 | Rosenthal 3-D point/line source | bead width, depth, profile | `< 3 %` |
| P2 | Single-track GMAW geometry | width / height / penetration vs published measurements | `< 10 %` |
| P3 | Thermal cycles | t-T at fixed probes matches IR camera | `< 10 %` |
| P4 | **S-B100** effective thickness | 6.2 mm (paper) / 5.3 mm (exp) | ≤ 17 % |
| P5 | **SS-B100** effective thickness | 6.3 mm vs 6.2 mm | ≤ 5 % |
| P6 | Sagging — pass index | S-*: 4 · SS-B5: 3 · all 100 s: none | **hard** |
| P7 | Sagging mechanism | finger→cylindrical transition, `P_st` spike at toes | qualitative + `P_st` peak present |
| P8 | Pool volume ratio | S is 35–40 % smaller than SS | ±10 % abs |
| P9 | Average error budget | abstract's "< 10 % average" over the 8 cases | ≤ 10 % |
| P10 | Marangoni direction | positive `∂γ/∂T` → inward flow, deep & narrow; negative → outward, wide | depth ratio inversion |
| P11 | ER316L vs ER70S6 `K` sensitivity | ER316L pools wider & sags earlier | monotone ordering |
| P12 | Inter-pass temperature | cool-to-100 °C rule honoured | exact (schedule unit test) |

P1–P3 are **tier-1**: they must pass before any of P4–P12 is believed.
P6 is a **hard assertion** (discrete, unambiguous); P4/P5/P9 are statistical
statements about an ensemble and are reported as such, never as pass/fail on a
single run.

### 6.3 Reproducibility & provenance

Every run writes a `run_manifest.json` (or `.toml`) containing: git SHA, package
versions, full resolved config, mesh summary, material database hash + provenance,
solver tolerance settings, wall-clock, thread count, random seed, and the numeric
results of all budget/verification diagnostics. A run directory is
**self-describing**: months later you can still say exactly what produced a figure.

---

## 7. Phased delivery

| Phase | Scope | Exit criteria |
| --- | --- | --- |
| **P0** Scaffold | package, CI, docs, config system, style, pre-commit | `Pkg.test` runs green on empty suite |
| **P1** Kernel | grid, fields, operators, sparse solvers, materials, enthalpy | V1–V10 green |
| **P2** Free surface | VOF (Lagrangian + MULES), curvature, Marangoni, partial-void pressure | V11, V12, V16, V17 green |
| **P3** Physics | momentum, energy, buoyancy, Darcy, arc flux, droplets, deposition schedule | V13–V15, V18–V20 green |
| **P4** Validation | all 8 paper cases + tier-1 literature validation | P1–P12 reported |
| **P5** Performance | blocking, threading, multigrid, adaptive dt, restart | 3400 s / 25-pass case < 30 min on 8 cores |
| **P6** Release | docs, tutorials, General Registry, Zenodo DOI, Python wrapper | `] add WAAMFlow` works from a clean depot |
| **P7** Extensions | unstructured FVM, CAD/STEP→mesh, XPBD weld solid mechanics coupling, phase-field hookup | opt-in, covered by same V&V |

### 7.1 Performance budget for the 3400 s headline

| Lever | Expected gain |
| --- | --- |
| Frozen/dwell implicit large-`dt` (§4.3) | **10³ ×** — dominant; dwell is ~95 % of wall time |
| Activity-list masking (cells with `T<T_sol` and zero flux skip fluid work) | 20–50 × on the fluid stage |
| Cartesian blocking + threading | 4–8 × on 10 cores |
| Multigrid pressure | 5–10 × vs BiCGSTAB on PISO inner loop |
| Vectorized `@inbounds @simd` kernels; `LoopVectorization` where it pays | 1.5–3 × |
| Lower-order (BE) energy in stiff regime | already implied by large `dt` |

The honest structural lever is the **phase-aware timestep**: 3400 s of which only
~1000 s is arc-on, and most of *that* is a localized pool. Everything else is
conduction. Designing for that is the difference between 30 minutes and 30 hours.

---

## 8. Public release plan

### 8.1 Packaging

- **License:** MIT (permissive; maximum adoption for an academic/industrial tool).
  Code that ports a specific published formulation stays attributed via `NOTICE`.
- **Name:** `WAAMFlow` — chosen to avoid collision with any existing General Registry
  entry; **verify availability at registration time** (§8.4).
- **Compat:** Julia ≥ 1.10 (LTS); CI matrix 1.10/1.11/1.13 + nightly.
- **Deps, kept minimal on purpose:** `LinearAlgebra`, `SparseArrays`, `Printf`,
  `TOML`, `StaticArrays`. Optional extras: `WriteVTK`, `Plots`/`CairoMakie`,
  `HDF5`, `MPI`, `JSON3`. *Every visualization/output dependency is an extra*, so a
  headless CI install stays fast and the core stays auditable.
- **Reproducibility:** commit a `Manifest.toml` for the CI/bench environment only;
  libraries should not.

### 8.2 Repository hygiene

- `Project.toml` `[compat]` bounded **and** upper bounds on `julia` and stdlib deps.
- Pre-commit: `JuliaFormatter` (blue style), `JET` (type-stability smoke test),
  `Aqua.jl` (ambiguity, stale deps, piracy, unbound type params).
- `CODE_OF_CONDUCT.md` (Contributor Covenant), `CONTRIBUTING.md`, `SECURITY.md`.
- Issue templates: **bug** (must include a `run_manifest.json`) and
  **physics-validation request** (distinct template — science contributions get a
  different review path than bug fixes).

### 8.3 CI / CD

| Workflow | Trigger | Content |
| --- | --- | --- |
| `ci.yml` | push, PR | format check, Aqua, JET, unit tests, doc build, **fast verification subset (≤ 5 min)** |
| `verify.yml` | nightly, `workflow_dispatch` | full V1–V24 gating suite, artifact upload |
| `validate.yml` | weekly + manual | P1–P12 incl. paper reproduction; long timeout; artifacts (VTK + CSV + manifests) |
| `docs.yml` | push to main | Documenter build + deploy to GitHub Pages |
| `bench.yml` | weekly | performance regression vs `bench/`, comment on PRs |
| `release.yml` | tag `v*.*.*` | full suite, tag registry, build + upload docs |

Hard rule: **PR CI must stay under ~10 minutes.** The expensive suites are scheduled,
not blocking — a suite nobody runs is not a suite.

### 8.4 Making it genuinely installable

1. `Pkg.test` green on a **clean depot** (this catches undeclared dependencies).
2. Check the name is free in the General Registry:
   `julia -e 'using Pkg; Pkg.Registry.add("General"); ...'` or inspect
   `General/W/WAAMFlow/Package.toml`.
3. `Registrator.jl` → opens the registration PR; respond to automated review comments.
4. Tag `v0.1.0`; Documenter deploys docs to GitHub Pages.
5. **Zenodo** release → citable DOI. *This is what turns a repo into a research
   artifact*, and it is the step most often skipped.
6. Announce on Julia Discourse + Julia Slack `#help`, and to the WAAM research
   community (Meltio/Mazak/ESAB/voestalpine welding R&D, universities running GMAW
   programs).
7. Post a **reproduction notebook** that reproduces Fig. 6 of the paper end-to-end.
   Anyone must be able to get the paper's figures from the public package. That is
   the credibility test for the whole project.

### 8.5 Documentation

- **Documenter.jl**, versioned, with **doctests in CI** (never allow docs to rot).
- Tutorials, ordered by cost:
  1. `01_rosenthal.md` — 10 s, pure conduction, prints a profile.
  2. `02_static_droplet.md` — 60 s, surface tension.
  3. `03_single_track.md` — 5 min, one GMAW bead, real geometry.
  4. `04_multipass.md` — 15 min, 10 passes, U vs B, dwell study.
  5. `05_paper_reproduction.md` — the 8 cases, reproduces the paper's figures.
  6. `06_custom_material.md` — add your alloy / digitize Fig. 5.
- A **physics codebook** page mapping every config key → equation number → citation.
- A **"what this model does not do"** page, stated plainly: no keyhole, no
  electromagnetic pinch (paper neglects it at low current), no arc plasma
  hydrodynamics, no droplet spatter, no evaporation, laminar only, no solid-state
  mechanics, single-track geometry prescribed per pass. Overclaiming is the fastest
  way to lose a research community's trust.

---

## 9. Explicitly out of scope (v1)

Keyhole welding · electromagnetic/pinch forces · arc-plasma fluid dynamics ·
droplet spatter and satellite droplets · mass evaporation / fume · turbulence
modelling (laminar per the paper) · elastic–plastic residual stress (this belongs to
the **existing FEM framework**; coupling is P7) · multi-material dissimilar wires ·
full CAD → unstructured FVM (P7).

The last item deserves emphasis: **v1 answers "will this wall stand up?" — not
"what are the residual stresses?"** The existing `WAAM-Framework` (FreeFEM++, Goldak,
J2 mechanics) answers the second question. §10 is the bridge.

---

## 10. Integration with the existing `WAAM-Framework`

The two solvers are **complementary, not competing** — the paper makes this explicit
in its own §1 ("conventional FEM remains at the top when it comes to predicting
microstructure").

```
                  ┌──────────────────────────────┐
   G-code / STEP  │  WAAM-Framework (Python)     │
   deposition     │  slicer → G-code → contour   │
   plan           └──────────────┬───────────────┘
                                  │  toolpath (x,y,z per layer)
                                  ▼
                  ┌──────────────────────────────┐
                  │  WAAMFlow.jl  (this project) │
                  │  VOF + Marangoni + mushy     │
                  │  → real bead geometry        │
                  │  → sagging verdict            │
                  │  → P_st stability index       │
                  └──────────────┬───────────────┘
                                  │  measured free-surface mesh
                                  ▼
                  ┌──────────────────────────────┐
                  │  WAAM-Framework (Python)     │
                  │  FreeFEM++: as-built mesh,   │
                  │  thermal cycles, J2, residual │
                  │  stress, distortion           │
                  └──────────────────────────────┘
                                  │
                                  ▼
                  phase-field (PFM) ← cooling rate, thermal gradient
```

**The one-way coupling that matters most** is: *predicted sagging collapses an
element-birth/death FEM model silently, because FEM assumed the geometry.*
A collapse predicted here is a real finding; a collapse assumed away is a
manufacturing defect with a certificate of conformity.

Interoperability mechanism: `WAAMFlow` exports the as-built free-surface as
`.vtu` **and** a watertight `.stl`, consumed by the existing `mesh/cad_importer.py`
and `mesh/gcode_parser.py` stack. A thin `pywaamflow` Python wrapper makes the whole
pipeline callable from the existing GUI.

---

## 11. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 3400 s case does not finish in reasonable time | High | High | Phase-aware `dt` (§4.3) designed in from day one; activity lists; P5 perf gate |
| R2 | Sagging onset is mesh-sensitive → validation fails | High | High | Mesh-convergence study on SS-B5 specifically; report confidence band, not a point claim |
| R3 | `P_st` is grid-dependent (curvature is 2nd-order, noisy) | High | Medium | Smooth `F` before curvature; report **cell-averaged** `P_st` over a probe window; cross-check on 3 meshes |
| R4 | Property data cannot be reproduced from Fig. 5 | Certain | Medium | Accept; build from primary citations with provenance; ship `digitize` utility; document |
| R5 | Users misread outputs as mechanics | Medium | High | "Limitations" doc page; CLI requires explicit `--intent`; sagging verdict clearly separated from strength |
| R6 | SciPy/FLOW-3D comparison requests dominate issues | High | Low | FAQ + pinned comparison script; standard response = provenance-labelled side-by-side |
| R7 | Registry name collision | Low | Medium | Name chosen conservatively; verified pre-registration (§8.4) |
| R8 | Community wants unstructured/CAD immediately | High | Medium | P7 roadmap published from day one; unstructured is a real plan, not a "maybe" |

---

## 12. First milestone — concrete task list

**M1: "Does the kernel lie?"** (P0 + P1)

1. `Project.toml`, `src/WAAMFlow.jl`, module skeleton, `.gitignore`, `LICENSE`, `README.md`
2. `mesh/grid.jl` — non-uniform Cartesian grid, MAC geometry, invariants (V1)
3. `mesh/metrics.jl` — exact stretched-grid two-point coefficients
4. `fields/*.jl` — `VolumeField`, `SurfaceField`, `VectorField`
5. `fields/interpolation.jl` — linear + TVD limiters, face interpolation, gradients (V3, V4)
6. `solvers/sparse.jl` — CSR build, `mul!`, PCG, ILU0/ILUT (V8)
7. `solvers/poisson.jl` — variable-coefficient Dirichlet/Neumann Poisson (V6, V7)
8. `materials/properties.jl` — `PropCurve`, `Material`, `ThermophysicalDatabase`
9. `materials/er70s6.jl`, `materials/er316l.jl` — the two alloys + provenance (§3.6)
10. `materials/surfacetension.jl` — Sahoo–DebRoy, Pichler, linear, constant (V-none, analytic check)
11. `physics/enthalpy.jl` — `h(T)`, `T(h)`, `liquid_fraction` (V9, V10)
12. `test/verify/*` — V1–V10
13. CI workflow + Aqua + formatter

**M2: "Can it hold a liquid?"** (P2) — VOF, curvature, Marangoni, partial-void pressure → V11, V12, V16, V17.

**M3: "Does it weld?"** (P3) — momentum, energy, sources, schedule → V13–V15, V18–V20, then P1–P3.

**M4: "Does it match the paper?"** (P4) — P4–P12.

Each milestone ends with a short `docs/src/milestones/*.md` stating what was proven
and what remains unproven — including, explicitly, which validations did **not** pass.
---

# Appendix A — Implementation log: M0–M1 findings

Everything below was found by *running* the verification suite, not by reading code.
Each entry is a bug class this project will hit again, recorded so it is caught by a
test rather than by luck. Suite state at the end of this session: **41 625 assertions,
0 failures, deterministic**.

## A.1 Bugs found and fixed (all now regression-tested)

| # | Where | Bug | Why it mattered |
| --- | --- | --- | --- |
| B1 | `GridMetrics` | Face-area planes for index `n+1` were **never assigned** (`undef`) | Every face flux touching a domain boundary was multiplied by garbage. Silent, and it looked fine everywhere except the walls. |
| B2 | `volume_identity_error` | Used `xf[1]` instead of `xf[i]` for the low face; wrong sign | The invariant itself was wrong, so it could not catch B1. A broken yardstick. |
| B3 | `graded_axis` | Rescaling the cell list to fill the span perturbed the requested ratio and produced non-monotone widths | Broke the "3:1 graded mesh" premise the paper depends on. |
| B4 | `minmod`/`superbee`/`vanleer` | The `::Real` fallback coerced *after* dispatch, so two `Float64` arguments re-entered it | Infinite recursion — a `StackOverflowError` at module load. |
| B5 | `vanleer` | The commonly-quoted algebraic form `(a+b+√ab)/2` returns **1.5a** for `a == b`: it **amplifies** slopes by 50 % | Destroys TVD. The limiter was the direct cause of the advection instability in B7. |
| B6 | `harmonic_face_coeff` | Two successive wrong conventions before the right one | On a uniform slab this silently doubles or halves the thermal resistance of the whole model. |
| B7 | `face_fluxes` | The upwind pick used both reconstructions from the *same* stencil, so the profile transported backwards | L1 error ~0.9 on a problem whose answer is 0.0. Found only because the linear-ramp exactness test exists. |
| B8 | `face_fluxes` | Read `a[0]` / `a[n+1]` under `@inbounds` | Out-of-bounds read; may not fault in a release build. |
| B9 | `enthalpy` | `h(T)` used a *varying* `C_p` while `T(h)` assumed a constant | 66 K round-trip error over 800–2600 K. Now constant-per-phase, which makes the inverse exact and energy conservation exact. |
| B10 | `poisson7!` | Dirichlet rows kept a residual of `-rhs`, entering the CG inner products | Contaminated the first `(r,z)` product; the iteration wandered off an otherwise perfectly solvable system. |
| B11 | `poisson7!` | Suppressed Dirichlet neighbours *without* moving their prescribed value onto the RHS | Converged cleanly to the **wrong solution** — small residual, large L2 error. The most dangerous bug class here. |
| B12 | `poisson7!` | `r = L p − rhs` instead of `rhs − L p` | Converged to `−x` with a healthy-looking recursive residual. Only visible when the result is checked against something independent. |
| B13 | `apply7!` | The diagonal contribution was not multiplied by `p[c]` | Turned the operator into a per-row constant offset. |
| B14 | `apply7!` | Normalised by face width only, giving `(6p − Σp)/Δx` instead of `/Δx²` | Solved a different PDE by a factor of `Δx`. The MMS pinned it immediately. |

## A.2 Deliberate design decisions taken during implementation

- **Rusanov, not Lax–Wendroff.** LW on limited MUSCL states is stable only to `C ≈ 0.2`
  (measured), versus `C ≈ 0.6` for Rusanov. The smaller timestep LW needs costs more
  accuracy than its lower dissipation buys back. `MAX_STABLE_COURANT = 0.45` is measured,
  not assumed, and the suite asserts the scheme is stable at it and unstable above it.
- **Constant-phase `C_p` in the enthalpy method** (B9). Physically ~15 % over the range,
  and it buys *exact* `h ↔ T` inversion and therefore exact energy conservation.
- **`tail_floor` on viscosity.** The tangent continuation of an Arrhenius falloff crosses
  zero near 3300 K; a zero-viscosity cell makes the momentum equation singular.
- **Bounded extrapolation everywhere.** Property curves clamp `|T̄|` to a multiple of the
  end value and keep strictly-positive properties positive. Clamping to the tabulated end
  value instead would invent an isothermal plume above the table.

## A.3 Multigrid: gated OFF, with four bugs already diagnosed

`multigrid_poisson!` is implemented but **does not converge** (residual grows ≈ 3.2× per
cycle). Rather than ship a diverging solver, it throws unless explicitly overridden, and
a test asserts the gate exists so it cannot be quietly removed. `poisson7!` is the
verified reference.

Four real bugs were found and fixed on the way; they are worth recording because each is
a standard trap:

1. **The coarse RHS must be the restricted *residual*** (`A_c e = R(r)`), not the
   restricted source. Restricting `b` coincides with the correct thing only when `x == 0`.
2. **Restriction must average all eight children.** Averaging four (easy to do, since the
   `i` and `j` children come in symmetric pairs while the `k` pair is easy to overlook)
   biases the coarse problem toward one slab; paired with a prolongation that writes all
   eight, the cycle amplifies instead of reducing.
3. **Coarse Dirichlet cells must use invalidation** (any fine child fixed ⇒ coarse fixed).
   Requiring all eight leaves the coarse grid pure-Neumann — a singular coarse problem —
   whenever the fine grid has an isolated pinned cell, which is exactly the free-surface
   case (one ambient cell).
4. **Coarse face conductances need Galerkin scaling.** A plain two-face average is too
   weak by the area ratio, so `A_coarse` approximates a quarter of `A_fine`.

With all four fixed the cycle still diverges, so a fifth defect remains — most likely the
coarse-grid Jacobian or the smoother. This is P5 work, gated on a residual-reduction test
rather than on a wall-clock target.

## A.4 What the verification suite now proves

| Suite | Proves |
| --- | --- |
| V1 (40) | `Σ S_f = 0` and `V = ⅓ Σ(S_f·x_f)` hold to ~1e-16 on uniform **and** 3:1 graded grids |
| V2 (7) | Gradient/Laplacian are 2nd order on uniform and graded axes; divergence exact for linear MAC fields |
| V3 (971) | Every limiter is non-amplifying and monotone — with each limiter's own correct bound |
| V4 (16) | Two-point face conductivity reproduces an exact constant-flux gradient |
| V5 (26) | Advection is ~2nd order, TVD, exact for a linear ramp, conservative on a stretched axis; the measured Courant limit is asserted |
| V6 (19) | The 7-point Poisson operator reproduces `p = x²+y²+z²` **exactly** (constant and variable `k`) — which pins coefficient normalisation, Dirichlet RHS treatment and indexing simultaneously |
| V7 (5) | The pure-Neumann system is correctly reported as non-convergent |
| V8 (10) | Fused operator == independent sparse assembly to 1e-12; CG == dense direct solve to 1e-6; operator is SPD |
| V9/V10 (40 356) | `h ↔ T` is exactly invertible, latent heat is exactly `L_f`, `f_l ∈ [0,1]` and monotone, all properties positive over an extrapolated range |
| Darcy (116) | `k_d` finite at `f_s → 1`, monotone, and the paper's `s_d = 10⁶` is documented as a hard freeze rather than a porous medium |

## A.5 Next milestone (M2 — "can it hold a liquid?")

VOF advection with exact volume conservation (V16/V17), curvature `κ = −∇·n̂`, the
`P_st = γ(T)κ` diagnostic, Marangoni surface stress as a free-surface boundary condition,
and partial-void pressure coupling — gated on the **1-D Stefan problem (V11)** and the
**static droplet Laplace balance (V12)** passing before anything else in the fluid stack
is written.
