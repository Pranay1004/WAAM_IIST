# Legacy Phase Field Modeling (PFM) Studies for UVA-WAAM

## 1. Executive Summary

This report aggregates the legacy phase field simulation studies conducted for Ultrasonic Vibration Assisted Wire Arc Additive Manufacturing (UVA-WAAM). The studies investigated the effect of ultrasonic acoustic energy on the solidification morphology of SS304/SS316 alloys. The core mechanism modeled is the acoustic cavitation-induced grain refinement, wherein ultrasonic pressure waves create transient cavitation bubbles that collapse, leading to dendrite fragmentation and enhanced equiaxed nucleation ahead of the solidification front.

## 2. Mathematical Framework (Governing Equations)

The legacy studies utilized a unified phase-field framework coupling heat transfer, phase evolution, and acoustic perturbation.

**Phase-Field Equation (Allen-Cahn):**
```markdown
∂ϕ/∂t = M_{ϕ} × [ ∇²ϕ - (1/ε²) × g'(ϕ) + (L / (T_m × ε)) × h'(ϕ) × (T - T_m) ] + S_{cavitation}
```
where `ϕ` is the phase field variable (0 for liquid, 1 for solid), `M_{ϕ}` is the mobility, `ε` is the interface thickness, and `S_{cavitation}` represents the stochastic nucleation source term induced by acoustic cavitation.

**Heat Transfer Equation:**
```markdown
∂T/∂t = α × ∇²T + (L / C_p) × (∂ϕ/∂t) + Q_{acoustic}
```
where `α` is thermal diffusivity, `L` is latent heat, `C_p` is specific heat, and `Q_{acoustic}` is the acoustic heating term.

**Acoustic Blake Threshold Pressure:**
```markdown
P_{B} = P_0 + (8 / 9) × [ (2 × σ) / (3 × R_0) ]^1.5 × (P_0 + 2 × σ / R_0)^{-0.5}
```
Cavitation (and consequent fragmentation/nucleation) occurs when the local acoustic pressure amplitude `P_{ac} > P_{B}`.

## 3. Parametric Frequency Sweep Study (0 kHz to 50 kHz)

A comprehensive frequency sweep was performed to understand the transition from pure conduction to optimal ultrasonic cavitation regimes. 

### 3.1. Simulation Parameters
*   **Domain Grid:** 500 × 1500 (Physical size: 500 µm × 1500 µm)
*   **Physical Time:** 23 seconds (approx.)
*   **Base Material:** SS304/SS316 equivalent properties
*   **SDAS (Secondary Dendrite Arm Spacing):** ~14.2 µm (constant proxy)

### 3.2. Morphological and Kinetic Results

The following table summarizes the quantitative metrics extracted across the 10-frequency parametric sweep.

| Frequency | Regime | Acoustic P (MPa) | Bubble R (µm) | Grains (N) | d_mean (µm) | Final Solid % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0 kHz** | Pure Conduction (No Acoustics) | 0.0 | 0.0 | 1 | 115.0 ± 0.0 | 27.38 % |
| **1 kHz** | Bulk Convection / Mild Mixing | 0.35 | 160.0 | 1 | 114.7 ± 0.0 | 27.27 % |
| **10 kHz** | Sub-Ultrasonic Transition | 0.85 | 55.0 | 6 | 52.3 ± 26.7 | 42.42 % |
| **15 kHz** | Low-Ultrasound Cavitation | 1.05 | 40.0 | 13 | 36.2 ± 22.2 | 47.70 % |
| **20 kHz** | Standard Industrial Power | 1.25 | 32.0 | 19 | 28.9 ± 20.8 | 49.19 % |
| **25 kHz** | High-Efficiency Inertial Cav | 1.35 | 27.0 | 23 | 26.6 ± 18.9 | 49.85 % |
| **30 kHz** | Peak Resonance / Max CET | 1.40 | 24.0 | 23 | 26.6 ± 18.9 | 49.81 % |
| **35 kHz** | High-Frequency Cavitation | 1.28 | 21.0 | 19 | 28.9 ± 20.8 | 49.13 % |
| **45 kHz** | Fine Dispersion / Micro-Streaming| 1.10 | 17.0 | 13 | 36.1 ± 22.1 | 47.62 % |
| **50 kHz** | Damped Processing Ultrasound | 0.95 | 15.0 | 6 | 52.1 ± 26.7 | 42.17 % |

**Observations:**
- **Optimal Frequency Window:** The peak refinement occurs between **25 kHz and 30 kHz**, where the mean grain diameter `d_mean` reaches a minimum of `26.6 µm` and the number of equiaxed grains peaks at 23.
- **Damping Effect:** Beyond 35 kHz, viscous damping and reduced bubble radii (`R_0 < 20 µm`) limit the acoustic pressure below optimal cavitation limits, resulting in a regression towards baseline columnar growth.

### 3.3. Parametric Visualizations

![Morphology Gallery](/Users/pandeyji/Desktop/WAAM/PFM/uva_pranay/legacy_studies/frequency_study_results/pfm_10freq_morphology_gallery.png)
*Figure 1: Evolution of microstructure morphology across the ultrasonic frequency spectrum (0-50 kHz).*

![Solidification Kinetics](/Users/pandeyji/Desktop/WAAM/PFM/uva_pranay/legacy_studies/frequency_study_results/pfm_10freq_solidification_kinetics.png)
*Figure 2: Impact of frequency on solid fraction kinetics and acoustic pressure.*

![Grain Statistics](/Users/pandeyji/Desktop/WAAM/PFM/uva_pranay/legacy_studies/frequency_study_results/pfm_10freq_grain_statistics.png)
*Figure 3: Quantitative analysis of mean grain diameter and total grain count as a function of ultrasonic frequency.*

---

## 4. Production Microstructure Comparison (Baseline vs 20 kHz UVA)

High-resolution production simulations were performed to closely analyze the Columnar-to-Equiaxed Transition (CET) dynamics induced by a standard 20 kHz industrial transducer.

### 4.1. Simulation Parameters
*   **Grid Size:** 400 × 400
*   **Time Steps:** 4000
*   **Time Step (dt):** 0.15 ms (`1.5e-04 s`)
*   **Acoustic Velocity:** 0.3 m/s (at 20 kHz)

### 4.2. Microstructural Refinement Metrics

| Metric | 0 kHz (Baseline) | 20 kHz (UVA) | Change / Impact |
| :--- | :--- | :--- | :--- |
| **Grain Count** | 1 | 10 | 10x Increase |
| **Mean Grain Diam.** | 80.82 µm | 30.15 µm | **62.7% Refinement** |
| **Max Tip Velocity** | 0.082 m/s | 0.052 m/s | Slower, more stable front |
| **Nucleation Density** | Low | High | 10x Density Increase |

**Conclusion:** The application of 20 kHz ultrasound induces a massive **62.7% reduction** in mean grain size, successfully triggering a sharp columnar-to-equiaxed transition.

### 4.3. Production Visualizations

![Baseline vs UVA Morphology](/Users/pandeyji/Desktop/WAAM/PFM/uva_pranay/legacy_studies/production_results/morphology_comparison.png)
*Figure 4: Direct comparison of final solid-phase morphology between 0 kHz (Left) and 20 kHz (Right).*

![CET Size Distribution](/Users/pandeyji/Desktop/WAAM/PFM/uva_pranay/legacy_studies/production_results/pfm_grain_size_distribution_cet.png)
*Figure 5: Statistical distribution of grain sizes, highlighting the shift from large columnar grains to fine equiaxed grains under UVA.*

![Multistage Evolution](/Users/pandeyji/Desktop/WAAM/PFM/uva_pranay/legacy_studies/production_results/pfm_multistage_morphology_evolution.png)
*Figure 6: Multi-stage temporal evolution of the solid-liquid interface under 20 kHz ultrasonic vibrations.*
