# Phase-Field Modeling (PFM) Theory, Formulation & Literature Citations

## Executive Summary & Architecture

The phase-field modeling (PFM) suite has been consolidated from scattered legacy scripts into five tightly integrated, high-performance modules:

1. `pfm_kernels.py`: Unified Numba-accelerated computational kernels for Kim et al. pure metal solidification, Wheeler-Boettinger-McFadden (WBM) binary alloy solidification, and Karma thin-interface multi-grain directional solidification with anti-trapping current and acoustic cavitation coupling.
2. `preprocessing.py`: Material thermodynamic/CALPHAD database loader (SS308, SS304, SS316, Pure Ni, Ni-Cu, IN718, Al-Si), domain mesh generation, and seeding configurations (single center, corner, and substrate directional seeding).
3. `postprocessing.py`: Solid fraction tracking, tip velocity kinetics, secondary dendrite arm spacing (SDAS) via FFT, grain aspect ratio distribution, Hunt Columnar-to-Equiaxed Transition (CET) volume fractions, and SEM-like Electron Backscatter Diffraction (EBSD) Inverse Pole Figure (IPF) orientation coloring with crisp grain boundary detection.
4. `validation.py`: Automated benchmark suite validating against IIST thesis SC19B032 C codes (`pure_omp.c`, `binary_serial.c`), experimental pure Ni tip velocities, and Hunt CET criteria.
5. `main.py`: Unified interactive CLI and lightweight desktop GUI (Tkinter) entry point.

---

## 1. Physical Solidification Models

### 1.1 Pure Metal Solidification: Kim et al. (1999) Model

The phase-field equation for single-component pure melt solidification is governed by a non-conserved order parameter phi, where phi = 1 represents solid and phi = 0 represents liquid:

```
tau * (dphi / dt) = div[eps(theta)^2 * grad(phi)] + t1 - t2 - [w * g'(phi) + 30 * g(phi) * (H / Tm) * (T - Tm)] + noise
```

where:
- Double-well potential function:
```
g(phi) = phi^2 * (1 - phi)^2
g'(phi) = 2 * phi * (1 - phi) * (1 - 2 * phi)
```
- Anisotropic gradient energy coefficient:
```
eps(theta) = eps0 * [1 + delta * cos(j * (theta - theta0))]
theta = arctan((dphi / dy) / (dphi / dx))
```
- Anisotropic torque variational terms:
```
t1 = d/dy [eps(theta) * (deps / dtheta) * (dphi / dx)]
t2 = d/dx [eps(theta) * (deps / dtheta) * (dphi / dy)]
```
- Coupled thermal diffusion equation with latent heat evolution:
```
dT / dt = D_th * laplacian(T) + (H / Cp) * 30 * g(phi) * (dphi / dt)
```

#### Physical Parameter Mapping:
```
eps0 = sqrt(6 * sqrt(2) * sigma * d)
w = 3 * sigma / (sqrt(2) * d)
M_phi = (Tm * mu) / (6 * sqrt(2) * H * d)
tau = 1 / M_phi
```

For pure nickel (Kim et al. 1999, Ferreira et al. 2006, SC19B032 thesis):
- Melting temperature: `Tm = 1728 K`
- Surface energy: `sigma = 0.37 J/m^2`
- Interface thickness: `d = 1.6 × 10^-7 m`
- Latent heat: `H = 2.35 × 10^9 J/m^3`
- Heat capacity: `Cp = 5.42 × 10^6 J/(m^3 K)`
- Kinetic coefficient: `mu = 2.0 m/(s K)`
- Thermal diffusivity: `D_th = 1.55 × 10^-5 m^2/s`
- Spatial resolution: `dx = 2.0 × 10^-8 m (20 nm)`
- Timestep: `dt = 1.0 × 10^-12 s (1 ps)`

---

### 1.2 Binary Alloy Solidification: Wheeler-Boettinger-McFadden (WBM 1993) Model

For binary alloy systems (e.g., Ni-Cu or Al-Zn), phase transformation involves both phase evolution and solute redistribution:

```
(1 / M_phi) * (dphi / dt) = div[eps^2 * grad(phi)] + (t1 - t2) - [(1 - c) * H_A(T, phi) + c * H_B(T, phi)] + noise
```

where the component thermodynamic driving forces are:
```
H_A(T, phi) = w_A * g'(phi) + 30 * g(phi) * L_A * (1/T - 1/Tm_A)
H_B(T, phi) = w_B * g'(phi) + 30 * g(phi) * L_B * (1/T - 1/Tm_B)
```

The solute concentration `c` is governed by the conserved Cahn-Hilliard equation:
```
dc / dt = div[ D_c(phi) * grad(c) + (Vm / R) * D_c(phi) * c * (1 - c) * (H_B - H_A) * grad(phi) ]
```

where:
```
D_c(phi) = D_s + p(phi) * (D_l - D_s)
p(phi) = phi^3 * (10 - 15 * phi + 6 * phi^2)
```

> **Correction of Legacy Bug**: In the original reference C code `binary_serial.c`, the interpolation function was written as `p = phi^3 * (6*phi^2 + 15*phi - 10)`, which produced non-physical negative values near `phi = 0`. This suite implements the mathematically rigorous monotonically increasing polynomial `p(phi) = phi^3 * (10 - 15*phi + 6*phi^2)`.

---

## 2. Thin-Interface Asymptotics & Millisecond Timescale Modeling

### 2.1 The Timescale Dilemma
In physical wire arc additive manufacturing (WAAM), solidification occurs over tens to hundreds of milliseconds (`10 - 100 ms`), with solute diffusivity `D_L ≈ 10^-9 m^2/s` and tip velocities `V_tip ≈ 1 - 10 mm/s`. 

Simulating `10 ms` with classical sharp-interface grid spacing (`dx = 20 nm, dt = 1 ps`) requires `10^10` timesteps, which is computationally intractable on workstations. Conversely, artificially enlarging `dx` to `1 μm` in classical models induces massive non-physical solute trapping.

### 2.2 Karma's Thin-Interface Formulation with Anti-Trapping Current
Based on Alain Karma (Phys. Rev. Lett. 87, 2001) and Provatas et al. (2021), the interface width `W0` is expanded to the mesoscale (`W0 ≈ 1.0 - 2.0 μm`), enabling grid steps `dx = 0.5 - 1.0 μm` and timesteps `dt = 5 - 20 μs`.

To eliminate artificial solute trapping, the **anti-trapping current** `j_at` is introduced into the solute conservation law:

```
dc / dt = div[ D_eff(phi) * grad(c) + j_at ] + (1 - k) * c * (dphi / dt)
```

where:
```
j_at = -a_at * W0 * (1 - k) * c * (dphi / dt) * (grad(phi) / |grad(phi)|)
a_at = 1 / (2 * sqrt(2)) ≈ 0.3535
```

The phase-field equation utilizes the smooth interpolation derivative `p'(phi) = 30 * phi^2 * (1 - phi)^2 = 30 * g(phi)`:
```
tau(n) * (dphi / dt) = W(n)^2 * laplacian(phi) - g'(phi) + 30 * g(phi) * lambda * [ Delta_T / Delta_T0 - (c - c0) / c0 ]
```

where:
- `lambda = a1 * W0 / d0`: Thin-interface coupling constant
- `tau0 = a1 * (W0^2 / D_L)`: Interface relaxation time

This formulation strictly reproduces the classical Stefan boundary problem without spurious kinetic undercooling and achieves exact convergence at millisecond timescales.

---

## 3. Columnar-to-Equiaxed Transition (CET) & Grain Tracking

### 3.1 Directional Solidification Profile
A moving linear thermal gradient is imposed along the vertical growth axis `y`:
```
T(y, t) = T_bottom_0 + G_thermal * (y - V_pull * t)
```
- Thermal gradient: `G_thermal ≈ 5.0 × 10^4 - 1.0 × 10^5 K/m (50 - 100 K/mm)`
- Pulling velocity: `V_pull ≈ 1.0 - 5.0 mm/s`
- Cooling rate: `R_cooling = G_thermal * V_pull ≈ 100 - 500 K/s`

### 3.2 Heterogeneous Nucleation Model (Hunt & Rappaz-Gandin)
Constitutional supercooling develops in the liquid ahead of the advancing columnar dendrite tips:
```
Delta_T_c(x, y) = T_liquidus(c(x, y)) - T(y, t)
```

Heterogeneous nucleants activate according to the continuous cumulative log-normal / Gaussian distribution:
```
n_active(Delta_T) = (N_max / 2) * [ 1 + erf( (Delta_T - Delta_T_N) / (sqrt(2) * Delta_T_sigma) ) ]
```
When `Delta_T_c > Delta_T_N`, new equiaxed nuclei trigger ahead of the columnar front with random crystallographic orientations `theta_k in [0, pi/2]`.

### 3.3 Hunt's CET Criterion
The volume fraction of equiaxed grains `Phi_eq` determines the microstructure regime:

| Equiaxed Fraction `Phi_eq` | Microstructure Regime | Physical Description |
| :--- | :--- | :--- |
| `Phi_eq < 0.0066 (0.66%)` | **Fully Columnar** | Columnar dendrites grow unhindered from substrate to top. |
| `0.0066 <= Phi_eq < 0.49` | **Mixed Columnar + Equiaxed** | Equiaxed grains nucleate but columnar front partially penetrates. |
| `Phi_eq >= 0.49 (49.0%)` | **Fully Equiaxed CET** | Equiaxed grains thermally and mechanically block columnar advance. |

---

## 4. Ultrasonic Vibration Assistance (UVA) Coupling

High-intensity ultrasound (20 kHz to 50 kHz) transmits acoustic waves into the liquid melt pool:

```
P_acoustic = 2 * pi * f_UVA * rho_liquid * sound_speed * A_vib
```

For `f_UVA = 20 kHz`, `A_vib = 5.0 μm`, `rho = 7000 kg/m^3`, `sound_speed = 4150 m/s`:
- Acoustic pressure amplitude: `P_acoustic ≈ 18.2 MPa`
- Blake cavitation threshold: `P_Blake ≈ 0.69 MPa`

Since `P_acoustic >> P_Blake`, transient acoustic cavitation bubbles nucleate, expand, and violently collapse.

### 4.1 Acoustic Grain Multiplication (Dendrite Arm Fragmentation)
Shockwave emissions from collapsing cavitation bubbles exert severe shear stress on secondary and tertiary dendrite arms, inducing mechanical arm detachment (fragmentation). Each fragment serves as a potent new equiaxed nucleus:
```
N_max(f_UVA) = N_max_0 * [ 1 + C_frag * (f_UVA / 20 kHz) * sqrt(I_cav) ]
```

### 4.2 Acoustic Pressure-Induced Nucleation (Clapeyron Shift)
Local gigapascal pressure spikes during bubble collapse raise the local equilibrium melting temperature via the Clapeyron relation (`dT_m / dP = Tm * Delta_V / Delta_H`), effectively reducing the required undercooling barrier:
```
Delta_T_N(f_UVA) = max[ 2.0 K, Delta_T_N_0 - C_cav * ln(1 + P_acoustic / P_Blake) ]
```

### 4.3 Acoustic Streaming
Acoustic radiation forces drive convective acoustic streaming, accelerating liquid solute homogenization:
```
D_eff = D_L + D_streaming, where D_streaming ≈ 2.5 × 10^-10 * (P_acoustic / 1 MPa)
```

**Net Result**: Ultrasonic vibration dramatically suppresses coarse columnar grain formation, shifts the Hunt criterion beyond 49%, and yields fine, equiaxed grains across the deposition layer.

---

## 5. Comprehensive Academic Literature Citations

| Citation # | Authors & Year | Publication Title & Journal / Source | Key Contributions & Relevance |
| :--- | :--- | :--- | :--- |
| **[1]** | **Kim, Kim, & Suzuki (1999)** | *Phase-field model for binary alloys.* Physical Review E, 60(6), 7186–7197. | Foundational KKS binary model removing interface energy excess. |
| **[2]** | **Kim, Kim, Lee, Ode, & Suzuki (1999)** | *Large scale simulation of dendritic growth in pure undercooled melt by phase-field model.* ISIJ International, 39(4), 335–340. | Dimensional pure melt model, 4-fold dendritic growth, pure Ni properties. |
| **[3]** | **Wheeler, Boettinger, & McFadden (1993)** | *Phase field model of solute trapping during solidification.* Physical Review E, 47(3), 1893–1909. | WBM binary alloy model, conserved Cahn-Hilliard solute diffusion. |
| **[4]** | **Karma, A. (2001)** | *Phase-field formulation for quantitative modeling of alloy solidification.* Physical Review Letters, 87(11), 115701. | Thin-interface limit, anti-trapping current, eliminates spurious kinetics. |
| **[5]** | **Echebarria, Folch, Karma, & Plapp (2004)** | *Quantitative phase-field model of alloy solidification.* Physical Review E, 70(6), 061604. | Complete asymptotic matching of thin-interface alloy solidification. |
| **[6]** | **Ferreira, da Silva, & de Castro (2006)** | *Simulation of the solidification of pure nickel via the phase-field method.* Materials Research, 9(4), 349–356. | Benchmark pure Ni tip velocity vs undercooling; corner seed validation. |
| **[7]** | **Warren, J. A. & Boettinger, W. J. (1995)** | *Prediction of dendritic growth and microsegregation patterns in a binary alloy using the phase field method.* Acta Metallurgica et Materialia, 43(2), 689–703. | Isothermal dendrite growth, solute microsegregation in binary alloys. |
| **[8]** | **Hunt, J. D. (1984)** | *Steady state columnar and equiaxed growth of dendrites and eutectic.* Materials Science and Engineering, 65(1), 75–83. | Analytical Columnar-to-Equiaxed Transition (CET) criterion. |
| **[9]** | **Rappaz, M. & Gandin, C. A. (1993)** | *Probabilistic modelling of microstructure formation in solidification processes.* Acta Metallurgica et Materialia, 41(2), 345–360. | Heterogeneous nucleation distribution in directional solidification. |
| **[10]** | **Provatas, Pinomaa, & Ofori-Opoku (2021)** | *Quantitative Phase Field Modelling of Solidification.* CRC Press. | Comprehensive reference on grand potential and thin-interface models. |
| **[11]** | **Sanas, V. R., Chakravarthy, P., & Girish, B. S. (2023)** | *Phase-field Modelling of Solidification Process.* B.Tech Thesis SC19B032, IIST Thiruvananthapuram. | Direct source of `pure_omp.c` and `binary_serial.c` reference implementations. |
| **[12]** | **Tennyson, P. G. & Phanikumar, G. (2014)** | *Phase-field Modelling of Microstructure Evolution During Solidification Incorporating CALPHAD Description.* Ph.D. Thesis, IIT Madras. | CALPHAD thermodynamic coupling, dendrite fragmentation mechanisms. |
| **[13]** | **Eskin, G. I. & Eskin, D. G. (2014)** | *Ultrasonic Treatment of Light Alloy Melts.* CRC Press. | Acoustic cavitation mechanics, Blake threshold, acoustic streaming. |
| **[14]** | **Jian, X., Meek, T. T., & Han, Q. (2006)** | *Refinement of eutectic silicon in Al-Si alloy by ultrasonic vibration.* Scripta Materialia, 54(5), 893–896. | Experimental validation of acoustic grain refinement in alloy melts. |
