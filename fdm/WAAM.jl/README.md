# WAAMFlow.jl

A free-surface, multiphase finite-volume solver in Julia for **long-term Wire Arc
Additive Manufacturing (WAAM / WA-DED) thin-wall deposition** — the capability that
conventional FEM cannot provide (it cannot predict bead geometry or wall collapse) and
production CFD cannot reach (it cannot run 25 passes over 3400 s).

Reproduces and extends:

> H. H. Zargari, S. A. Khajeh Mohammadilar, V. Ziaei Laleh (2025),
> *A long-term and pragmatic numerical simulation of wire arc additively manufactured
> ER70S6 and ER316L thin-wall via finite difference method.*
> **Progress in Additive Manufacturing** 10:11647–11663.
> [doi:10.1007/s40964-025-01307-x](https://doi.org/10.1007/s40964-025-01307-x)

---

## Status

| Area | State | Evidence |
| --- | --- | --- |
| Grid, metrics, geometry invariants | **verified** | V1, V2 |
| Materials database, surface tension | **verified** | V9, V10 |
| Enthalpy / phase change / Darcy drag | **verified** | V9, V10 |
| TVD limiters, interpolation, advection | **verified** | V3, V4, V5 |
| Poisson solver (Jacobi-PCG) | **verified** | V6, V7, V8 |
| VOF advection | **verified** | volume conserved to 1e-15, bounded at any `dt` (V16, V17) |
| Energy solver | **verified, single-phase** | exact to 5.9e-11 vs the analytic adiabatic-box solution |
| Heat sources, energy split, losses | **verified vs paper** | σ_q = 2.398 mm vs 2.40 mm published |
| Single-track bead geometry | **runs** | real numbers; ER70S6/ER316L ordering reproduced |
| Energy budget across the mushy front | **NOT closing** | ~1.4 kW residual while a cell straddles [T_sol, T_liq] |
| VOF ↔ momentum ↔ Marangoni coupling | not done (P3) | — |
| Deposition schedule, sagging diagnostics | partial | bead diagnostics written; schedule not yet |
| Paper 8-case validation | not done (P4) | — |
| Multigrid | **experimental, gated off** | diverges; 4 bugs diagnosed, see below |

Everything shipped is covered by an automated test. Nothing is enabled that has not been
proven first; see [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) for the full plan,
including the physics the paper leaves ambiguous and how each ambiguity is handled.

## Installation

```julia
using Pkg
Pkg.develop(path = "/path/to/WAAM.jl")   # from a checkout
```

Requires Julia ≥ 1.10. The core has no dependencies beyond the standard library and
`StaticArrays`; every output/visualisation dependency is a weak dependency, so a headless
install stays small.

## Quick start

```julia
using WAAMFlow

# The paper's grid: 0.6 mm through the fusion zone grading to 1.8 mm far field,
# half-symmetry domain.
g = CartesianGrid(
    graded_axis(0.0, 0.020, 0.6e-3, 1.8e-3; h_fine_end = :first),
    graded_axis(0.0, 0.008, 0.6e-3, 1.2e-3; h_fine_end = :first),
    graded_axis(0.0, 0.006, 1.0e-3, 1.0e-3))

m = metrics(g)
println(grid_summary(g))
@assert face_closure_error(g, m) < 1e-14      # discrete divergence theorem
@assert volume_identity_error(g, m) < 1e-13

# Materials, with the Marangoni sign that decides pool shape
for name in ("ER70S6", "ER316L")
    println(material_report(get_material(name)))
end
```

## Design principles

1. **Verification before validation.** Every operator is proven against a manufactured or
   analytical solution *before* it is pointed at welding.
2. **Conservation is reported, not hoped for.** Mass and energy budgets are closed and
   checked at run time.
3. **Provenance.** Every physical coefficient traces to a paper equation or a cited
   primary source, and `material_report` prints it.
4. **Boundedness over accuracy where it matters.** TVD limiters, because a volume
   fraction that leaves `[0, 1]` is not a slower answer, it is a wrong one.
5. **Fail loudly.** A non-converged solve, a negative face coefficient, or a singular
   pressure system raises rather than returning a plausible-looking field.

## What this model does *not* do

Stated plainly, because overclaiming is the fastest way to lose a research community's
trust: no keyhole, no electromagnetic pinch forces, no arc-plasma hydrodynamics, no
droplet spatter, no mass evaporation, laminar only, no solid-state mechanics, and the
per-pass bead footprint is prescribed rather than self-generated.

v1 answers **"will this wall stand up?"** — not "what are the residual stresses?". The
existing `WAAM-Framework` (FreeFEM++, Goldak, J2 mechanics) answers the second question,
and the two are designed to couple one-way: predicted free-surface geometry in, residual
stress out. See `docs/MASTER_PLAN.md` §10.

## Reproducing the validation numbers

```bash
julia --project=. test/validate/run_validation.jl
```

Writes `validation_output/validation_run.txt`. Current results are tabulated in
`validation_output/RESULTS.md`.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite is deterministic (fixed seed) and thread-count independent. `verify.yml` runs
it nightly; `ci.yml` runs a fast subset on every pull request; `validate.yml` runs the
paper-reproduction cases weekly.

## Licence

MIT. See `LICENSE`. The paper-specific formulations carry attribution in the source and
in `docs/MASTER_PLAN.md`.