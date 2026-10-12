# WAAM.jl: High-Fidelity FVM Simulation Framework

## 1. Software Architecture (The Collocated FVM Engine)

The framework must be designed from scratch around an unstructured, collocated Finite Volume Method (FVM) architecture to ensure stability in multiphase geometries.

*   **Core Data Structures:**
    *   `Mesh`: Unstructured definition of `Cells`, `Faces`, and `Nodes`. Includes face-owner and face-neighbor addressing necessary for unstructured sparse matrix assembly.
    *   `Fields`: Cell-centered `VolumeField` (for Pressure, Temperature, Phase Fraction) and face-centered `SurfaceField` (for fluxes).
    *   `lduMatrix`: Sparse matrix representation (Lower-Diagonal-Upper) optimized for cache-coherent sparse matrix-vector multiplications.
*   **Discretization Schemes:**
    *   Convection: High-resolution Total Variation Diminishing (TVD) schemes (e.g., `vanLeer` or `Gamma`) to prevent unphysical oscillations in temperature and momentum.
    *   Time integration: Implicit 1st-order Euler or 2nd-order Crank-Nicolson for momentum and energy to allow for practical Courant numbers (CFL).
    *   VOF Advection: Explicit Multidimensional Universal Limiter with Explicit Solution (MULES). An artificial compression term ∇·(Ur α (1-α)) ensures strict phase conservation and interface sharpness.
*   **Pressure-Velocity Coupling (PIMPLE):**
    *   Implementation of the Rhie-Chow interpolation to evaluate face fluxes directly from pressure gradients, preventing collocated checkerboarding.
    *   Outer loops handle non-linear momentum/VOF coupling; inner PISO loops strictly enforce the divergence-free continuity constraint ∇·U = 0.

## 2. Physics Modeling

*   **Phase Change (Mushy Zone):**
    *   Darcy damping term added implicitly to the momentum matrix diagonal.
    *   Enthalpy-Porosity method for the energy equation, utilizing an iterative solver to update the liquid fraction and absorb the latent heat of fusion.
*   **Droplet Deposition (Sources):**
    *   Volumetric source terms representing the moving torch (e.g., Goldak double-ellipsoid for heat, volumetric mass/momentum sources for filler wire addition).
*   **Surface Tension & Marangoni Stress:**
    *   Continuum Surface Force (CSF) model. Curvature κ = -∇·n evaluated at cell centers.
    *   Temperature-dependent surface tension γ(T) driving Marangoni convective loops in the melt pool.

## 3. Verification & Validation (V&V) Strategy

A robust framework requires stringent, automated testing protocols.

*   **Unit Testing (`Test` module):**
    *   *Grid Operations*: Verify that the sum of face area vectors for any closed cell equals exactly zero (Σ Sf = 0).
    *   *Operator Accuracy*: Test gradient (`grad`), divergence (`div`), and laplacian (`laplacian`) operators against analytical functions (e.g., manufactured solutions like Taylor-Green vortices) to ensure 2nd-order spatial accuracy.
*   **Verification (Methodological Correctness):**
    *   *Lid-Driven Cavity*: Compare velocity profiles against standard Ghia benchmark data.
    *   *Static Droplet (Spurious Currents)*: Ensure the VOF surface tension model produces an exact pressure jump (ΔP = σ / R) with minimal spurious velocity currents.
    *   *Stefan Problem*: Validate the 1D phase-change/Stefan problem against the analytical Neumann solution to verify latent heat absorption.
*   **Validation (Physical Realism):**
    *   Calibrate the Goldak heat source and Marangoni coefficients against experimental WAAM macrographs (weld bead width, depth of penetration).
    *   Ensure the framework correctly captures the cyclic thermal history (thermal ratcheting) during multi-pass deposition.

## 4. Open-Source Deployment & Distribution

*   **Repository & CI/CD:**
    *   Host on GitHub. Utilize GitHub Actions for continuous integration (running the V&V test suite on every pull request).
*   **Documentation:**
    *   Auto-generate documentation using `Documenter.jl`.
    *   Provide comprehensive tutorials covering simple 1D benchmarks up to full 3D 25-pass WAAM simulations.
*   **Registry & Package Manager:**
    *   Format as a standard Julia package.
    *   Register `WAAM.jl` in the official Julia General Registry so users can install it globally via `] add WAAM`.
*   **Interoperability:**
    *   Provide native `.vtu` / `.vtr` outputs utilizing `WriteVTK.jl` for direct visualization in ParaView.
    *   Create a clean Python wrapper (`PyCall` or `PythonCall`) so engineers accustomed to Python can script simulations while utilizing the Julia backend.
