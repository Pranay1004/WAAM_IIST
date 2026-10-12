# Unified Phase-Field Modeling (PFM) Suite for WAAM & Ultrasonic Processing

A high-performance, physically validated phase-field modeling suite developed for Wire Arc Additive Manufacturing (WAAM) and Ultrasonic Vibration Assistance (UVA).

Consolidated into **five core modular Python scripts** with Numba JIT acceleration:

```
uva_pranay/
├── pfm_kernels.py      # Core Numba JIT solvers: Kim Pure Melt, WBM Binary Alloy, Karma Thin-Interface CET
├── preprocessing.py    # Material database (SS308, SS304, SS316, Pure Ni, Ni-Cu, IN718, Al-Si) & domain setup
├── postprocessing.py   # Metrics, SEM-like EBSD IPF grain colormaps, Hunt CET fraction, publication figures
├── validation.py       # Automated benchmark suite against literature & IIST thesis SC19B032 C codes
├── main.py             # Unified entry point: Interactive CLI & Lightweight Tkinter Desktop GUI
├── docs/               # Comprehensive theoretical documentation & academic citations
│   └── PFM_THEORY_AND_CITATIONS.md
├── materials/          # Thermodynamic, CALPHAD, and acoustic JSON parameter databases
└── output/             # Output simulation results, CSV histories, and publication figures
```

---

## Quick Start

### 1. Interactive CLI Mode
```bash
python3 main.py --interactive
```

### 2. Desktop GUI Mode
```bash
python3 main.py --gui
```

### 3. Direct Command-Line Execution
Run SS308 directional solidification at 20 kHz UVA for 50 milliseconds:
```bash
python3 main.py --mode directional_cet --alloy ss308 --freq 20 --time 50.0 --width 750 --height 1500
```

Run baseline comparison (0 kHz, columnar growth):
```bash
python3 main.py --mode directional_cet --alloy ss308 --freq 0 --time 50.0 --width 750 --height 1500
```

### 4. Run Automated Validation Suite
Executes pure Ni Kim model benchmark, Ni-Cu WBM binary alloy benchmark, and directional CET benchmark:
```bash
python3 validation.py
```

---

## Physical Models Implemented

1. **Kim et al. (1999) Pure Metal Solidification**:
   - Exact dimensional formulation matching IIST thesis `pure_omp.c`.
   - Validated against pure Ni experimental dendrite tip velocities (`20 - 50 m/s`).

2. **Wheeler-Boettinger-McFadden (WBM 1993) Binary Alloy Model**:
   - Exact conserved Cahn-Hilliard solute diffusion matching IIST thesis `binary_serial.c`.
   - Corrected literature polynomial interpolation bug: `p(phi) = phi^3 * (10 - 15*phi + 6*phi^2)`.

3. **Karma (2001) Thin-Interface Limit & Anti-Trapping Current**:
   - Enables physical millisecond-scale (`10 - 100 ms`) directional solidification on mesoscale domains (`750 × 1500 μm`) without artificial solute trapping.

4. **Hunt & Rappaz-Gandin Columnar-to-Equiaxed Transition (CET)**:
   - Heterogeneous nucleation ahead of advancing columnar front.
   - Evaluates Hunt criteria (`Phi_eq < 0.66%` = Columnar; `Phi_eq >= 49%` = Equiaxed).

5. **Ultrasonic Vibration Assistance (UVA)**:
   - Acoustic cavitation pressure (`P_A = 2*pi*f*rho*c*A`).
   - Grain multiplication via acoustic dendrite fragmentation.
   - Nucleation barrier reduction via local cavitation shock pressure spikes.
   - Acoustic streaming convection accelerating liquid solute transport.

---

## Key Literature References

- **Kim, S. G., Kim, W. T., & Suzuki, T. (1999)**: Phase-field model for binary alloys. *Phys. Rev. E*, 60(6), 7186.
- **Wheeler, A. A., Boettinger, W. J., & McFadden, G. B. (1993)**: Phase field model of solute trapping. *Phys. Rev. E*, 47(3), 1893.
- **Karma, A. (2001)**: Phase-field formulation for quantitative modeling of alloy solidification. *Phys. Rev. Lett.*, 87(11), 115701.
- **Hunt, J. D. (1984)**: Steady state columnar and equiaxed growth of dendrites. *Mater. Sci. Eng.*, 65(1), 75.
- **Rappaz, M., & Gandin, C. A. (1993)**: Probabilistic modelling of microstructure formation. *Acta Metall. Mater.*, 41(2), 345.
- **Provatas, N., Pinomaa, T., & Ofori-Opoku, N. (2021)**: *Quantitative Phase Field Modelling of Solidification*. CRC Press.
- **Sanas, V. R., Chakravarthy, P., & Girish, B. S. (2023)**: *Phase-field Modelling of Solidification Process*. B.Tech Thesis SC19B032, IIST.
