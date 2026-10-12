# In-Depth Analysis: FDM-VOF WAAM Thin-Wall Multi-Physics Simulation

**Paper Citation**:
Habib Hamed Zargari, Seyyed Ahmadreza Khajeh Mohammadilar, Vahid Ziaei Laleh (2025). *A long-term and pragmatic numerical simulation of wire arc additively manufactured ER70S6 and ER316L thin-wall via finite difference method*. **Progress in Additive Manufacturing**, 10:11647–11663. [DOI: 10.1007/s40964-025-01307-x](https://doi.org/10.1007/s40964-025-01307-x).

---

## 1. Executive Summary & Scientific Breakthrough

In Wire Arc Additive Manufacturing (WAAM / WA-DED), conventional numerical investigations have historically been split between two extremes:
1. **Finite Element Method (FEM)**: Capable of multi-layer, long-term thermomechanical calculations (residual stress, distortion), but **fundamentally incapable of predicting free-surface dynamic geometry, melt-pool hydrodynamic flow, surface tension forces, or wall collapse (sagging)**. Layer geometry must be assumed *a priori* (e.g., quiet/inactive element activation with fixed bead geometry).
2. **Computational Fluid Dynamics (CFD / OpenFOAM / Fluent)**: Capable of free-surface hydrodynamic modeling using Volume of Fluid (VOF), but **computationally restricted to millisecond-scale single-track simulations** (\(< 2\) seconds of physical deposition) due to heavy continuum surface force (CSF) and volumetric Navier-Stokes coupling overhead.

### The Breakthrough of this Paper
Zargari et al. formulate and validate an **in-house Pragmatic Finite Difference Method (FDM) + Volume of Fluid (VOF)** code running in conjunction with FLOW-3D that achieves:
- **Long-term simulation of \(25\) successive passes** spanning **\(3400\,\mathrm{s}\)** of physical deposition and inter-pass cooling time.
- **Accurate prediction of macroscopic sagging collapse**: Discovering that sagging is triggered by a sudden transition of the weld pool from *finger-type* to *cylindrical* geometry, accompanied by a **critical spike in surface tension pressure at the weld toes**.
- Direct experimental validation for two vastly different metallurgical alloys:
  - **ER70S6** (Ferritic/pearlitic carbon-manganese structural steel, high thermal conductivity, sulfur-dependent Marangoni flow).
  - **ER316L** (Austenitic stainless steel, low thermal conductivity, high heat accumulation, negative Marangoni coefficient).

---

## 2. Complete Mathematical Formulation

The physical model couples four primary partial differential equations governing mass, momentum, heat, and free-surface advection.

### 2.1 Mass Continuity Equation
For an incompressible liquid phase with droplet mass source injection:
\[
\nabla \cdot \vec{V} = \frac{\dot{m}_s}{\rho}
\]
where:
- \(\vec{V} = (u, v, w)\) is the fluid velocity vector \([\mathrm{m/s}]\).
- \(\rho\) is the temperature-dependent fluid density \([\mathrm{kg/m^3}]\).
- \(\dot{m}_s\) is the mass source rate per unit volume from wire droplet impingement \([\mathrm{kg/(m^3\,s)}]\).

### 2.2 Momentum Conservation (Navier-Stokes) with Mushy Zone Drag
\[
\frac{\partial \vec{V}}{\partial t} + \vec{V} \cdot \nabla \vec{V} = -\frac{1}{\rho}\nabla P + \frac{\mu}{\rho}\nabla^2 \vec{V} - k_d \vec{V} + \vec{G}_b + \frac{\dot{m}_s}{\rho}\left(\vec{V}_s - \vec{V}\right)
\]
where:
- \(P\) is pressure \([\mathrm{Pa}]\).
- \(\mu\) is temperature-dependent dynamic viscosity \([\mathrm{Pa\,s}]\).
- \(\vec{V}_s\) is the droplet injection velocity vector (downward at \(0.7\,\mathrm{m/s}\)).
- \(k_d\) is the **Darcy porous media drag coefficient** modeling flow damping in the solid-liquid mushy transition zone (Carman-Kozeny relation):
  \[
  k_d = s_d \frac{f_s^2}{(1 - f_s)^3 + \epsilon_0}
  \]
  with \(s_d \approx 10^6\,\mathrm{s^{-1}}\), local solid fraction \(f_s \in [0, 1]\), and numerical regularization parameter \(\epsilon_0 = 10^{-4}\).
- \(\vec{G}_b\) is the **Boussinesq thermal buoyancy acceleration**:
  \[
  \vec{G}_b = -\vec{g}\beta_T (T - T_{\mathrm{liq}})
  \]
  with gravitational acceleration \(\vec{g} = (0, 0, -9.81)\,\mathrm{m/s^2}\) and volumetric thermal expansion coefficient \(\beta_T\,[\mathrm{K^{-1}}]\).

### 2.3 Energy Conservation with Enthalpy Phase Change
\[
\frac{\partial h}{\partial t} + \vec{V} \cdot \nabla h = \frac{1}{\rho}\nabla \cdot (K \nabla T) + \dot{h}_s
\]
where \(h\) is the specific enthalpy \([\mathrm{J/kg}]\), \(K\) is thermal conductivity \([\mathrm{W/(m\,K)}]\), and \(\dot{h}_s\) is the droplet enthalpy source.

The non-isothermal phase transition incorporates the latent heat of fusion \(L_f\,[\mathrm{J/kg}]\) via the piecewise continuous function:
\[
h(T) = \begin{cases}
C_s T & T \le T_{\mathrm{sol}} \\
C_s T_{\mathrm{sol}} + L_f \left(\frac{T - T_{\mathrm{sol}}}{T_{\mathrm{liq}} - T_{\mathrm{sol}}}\right) & T_{\mathrm{sol}} < T \le T_{\mathrm{liq}} \\
h(T_{\mathrm{liq}}) + C_l (T - T_{\mathrm{liq}}) & T > T_{\mathrm{liq}}
\end{cases}
\]

### 2.4 Volume of Fluid (VOF) Free Surface Tracking
\[
\frac{\partial F}{\partial t} + \nabla \cdot (\vec{V} F) = \dot{F}_s
\]
where \(F \in [0, 1]\) is the fluid volume fraction (\(F=1\) metal fluid, \(F=0\) void/ambient air, \(0 < F < 1\) free surface boundary).

---

## 3. Boundary Conditions & Physical Driving Mechanisms

### 3.1 Arc Heat Flux Model
The surface arc heat flux is modeled via a Gaussian distribution on the workpiece surface:
\[
q_{\mathrm{arc}}(x, y) = \frac{\eta(1 - \eta_d) UI}{\xi \pi \sigma_q^2} \exp\left\{ - \frac{(x - x_{\mathrm{torch}})^2 + (y - y_{\mathrm{torch}})^2}{\xi \sigma_q^2} \right\}
\]
where:
- \(UI \approx 2165\,\mathrm{W}\) is the instantaneous average electrical power.
- \(\eta = 0.68\) is the overall arc efficiency.
- \(\eta_d = 0.25\) is the droplet heat partition coefficient.
- The effective Gaussian radius \(\sigma_q\) is governed by the empirical arc current relation:
  \[
  \sigma_q = 6.205 \times 10^{-4} \times I_{\mathrm{nominal}}^{0.28}
  \]
  yielding \(\sigma_q \approx 2.39\,\mathrm{mm}\) for \(I = 125\,\mathrm{A}\).

### 3.2 Droplet Impingement Model
The thermal power delivered directly by the molten droplets is:
\[
Q_{\mathrm{drop}} = \eta \eta_d UI = \frac{4}{3}\pi r_d^3 f_d \left[ \rho_l C_l (T_d - T_{\mathrm{liq}}) + \rho L_f + \rho_s C_s (T_{\mathrm{sol}} - T_0) \right]
\]
- Droplet temperature: \(T_d = 1900\,\mathrm{K}\).
- Initial vertical velocity: \(v_d = 0.7\,\mathrm{m/s}\) downward.
- Injection height: \(h_{\mathrm{inject}} = 1.5\,\mathrm{mm}\) above surface.
- Pulse frequency: \(f_d = 60\,\mathrm{Hz}\).

### 3.3 Surface Tension Models & Marangoni Convection
On the free surface, the temperature gradient creates shear stress balancing viscous traction:
\[
\mu \frac{\partial \vec{V}_t}{\partial \vec{n}} = - \frac{\partial \gamma}{\partial T} \frac{\partial T}{\partial \vec{t}}
\]

#### ER70S6: Sahoo-DebRoy Pseudo-Binary Fe-S Model
Sulfur acts as a potent surface-active element, changing the temperature coefficient \(\frac{\partial \gamma}{\partial T}\) from negative (outward flow) to positive (inward flow, deeper penetration):
\[
\gamma(T) = \gamma_m^\circ - A (T - T_m) - R T \Gamma_s \ln\left[ 1 + k_1 a_s \exp\left( - \frac{\Delta H^\circ}{R T} \right) \right]
\]
Parameters:
- \(\gamma_m^\circ = 1.943\,\mathrm{N/m}\)
- \(A = 4.3 \times 10^{-4}\,\mathrm{N/(m\,K)}\)
- \(\Gamma_s = 1.3 \times 10^{-5}\,\mathrm{mol/m^2}\)
- \(k_1 = 3.18 \times 10^{-3}\)
- \(\Delta H^\circ = -1.66 \times 10^5\,\mathrm{J/mol}\)
- \(a_s \approx 0.018\,\mathrm{wt.\%}\) (sulfur concentration).

#### ER316L: Pichler Electromagnetic Levitation Model
High chromium and alloy content in austenitic stainless steel produces a negative surface tension slope across the operating temperature range:
\[
\gamma(T) = 1.77 - 3.5 \times 10^{-4} (T - 1700) \quad [\mathrm{N/m}]
\]
This produces outward Marangoni flow, wider molten pools, and earlier susceptibility to wall sagging.

### 3.4 Surface Heat Dissipation
\[
- K \frac{\partial T}{\partial \vec{n}} = h_c (T - T_0) + \varepsilon \sigma_{\mathrm{SB}} (T^4 - T_0^4)
\]
with \(h_c = 20\,\mathrm{W/(m^2\,K)}\), \(\varepsilon = 0.65-0.70\), and \(\sigma_{\mathrm{SB}} = 5.67 \times 10^{-8}\,\mathrm{W/(m^2\,K^4)}\).

---

## 4. Key Findings & The Mechanism of Wall Sagging

### 4.1 Comparison Across All 8 Experimental & Simulated Cases

| Case ID | Filler Material | Deposition Strategy | Inter-Pass Dwell Time | Target Passes | Sagging Pass | Sagging Time | Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **S-U5** | ER70S6 | Unidirectional | \(5\,\mathrm{s}\) | \(10\) | Pass 4 | \(\approx 148\,\mathrm{s}\) | Sagged / Collapse |
| **SS-U5** | ER316L | Unidirectional | \(5\,\mathrm{s}\) | \(10\) | Pass 4 | \(\approx 142\,\mathrm{s}\) | Sagged / Collapse |
| **S-U100** | ER70S6 | Unidirectional | \(100\,\mathrm{s}\) | \(10\) | None | — | **Stable** |
| **SS-U100**| ER316L | Unidirectional | \(100\,\mathrm{s}\) | \(10\) | None | — | **Stable** |
| **S-B5** | ER70S6 | Bidirectional | \(5\,\mathrm{s}\) | \(10\) | Pass 4 | \(\approx 145\,\mathrm{s}\) | Sagged / Collapse |
| **SS-B5** | ER316L | Bidirectional | \(5\,\mathrm{s}\) | \(10\) | **Pass 3** | **\(119\,\mathrm{s}\)** | **Earliest Sagging** |
| **S-B100** | ER70S6 | Bidirectional | \(100\,\mathrm{s}\) | \(25\) | None | — | **Stable (3400 s)** |
| **SS-B100**| ER316L | Bidirectional | \(100\,\mathrm{s}\) | \(25\) | None | — | **Stable (3400 s)** |

### 4.2 The Physics of Sagging at Pass 3 in SS-B5
1. **Low Thermal Conductivity**: ER316L thermal conductivity (\(14-32\,\mathrm{W/(m\,K)}\)) is approximately half that of ER70S6 (\(27-59\,\mathrm{W/(m\,K)}\)). Heat cannot escape into the substrate quickly enough when dwell time is only \(5\,\mathrm{s}\).
2. **Molten Pool Expansion**: The molten pool volume in SS-B5 is **\(35\%\) to \(40\%\) larger** than in S-B5.
3. **Puddle Shape Transition**: In passes 1 and 2, the pool maintains a compact *finger-type* profile. In pass 3, heat accumulation broadens the pool into a *cylindrical* mode.
4. **Weld Toe Instability & Surface Tension Spike**: As the molten pool widens toward the thin-wall sides, the curvature at the weld toes increases sharply. The surface tension equivalent pressure \(P_{\mathrm{st}} = \gamma \kappa\) spikes above the critical stability limit (\(\approx 250-350\,\mathrm{Pa}\)), forcing the liquid to overflow the side and collapse.

---

## 5. Architectural Comparison: FDM-VOF vs FEM WAAM Framework

| Feature | Current WAAM Framework (FEM / FreeFEM++) | Zargari et al. (2025) FDM-VOF |
| :--- | :--- | :--- |
| **Primary Focus** | Thermomechanical distortion, residual stress, J2 plasticity | Fluid dynamics, free surface, layer geometry, sagging |
| **Mesh Type** | Unstructured tetrahedral / Gmsh adaptive mesh | Non-uniform structured Cartesian MAC grid |
| **Geometry Representation**| Explicit boundary mesh with quiet-element activation | Volume of Fluid (VOF) Eulerian fraction \(F(x,y,z,t)\) |
| **Melt Pool Fluid Flow**| Neglected (pure conduction or artificially enhanced \(K_{\mathrm{eff}}\)) | Fully solved Navier-Stokes + Marangoni + Darcy drag |
| **Sagging Prediction** | **Impossible** (geometry is predefined) | **Directly predicted** via free-surface instability |
| **Computational Scale** | Hours for thermomechanical stress across multi-pass | Fast explicit/SOR finite difference on structured grid |

---

## 6. How FDM-VOF Can Be Integrated into the WAAM Framework

A two-way loosely coupled multi-scale architecture offers the ultimate balance:
1. **Mesoscale Hydrodynamic Pre-Solver (FDM-VOF)**:
   - Runs fast multi-pass deposition for candidate process parameters (current, speed, dwell time).
   - Predicts exact layer bead geometry \((w, h)\) and checks sagging stability.
   - Outputs the true reconstructed wall surface mesh.
2. **Macroscale Structural FEM Solver (FreeFEM++ / WAAM Framework)**:
   - Takes the true reconstructed mesh from FDM-VOF.
   - Solves elastoplastic J2 mechanics, cooling contraction, substrate bending, and residual stresses.
