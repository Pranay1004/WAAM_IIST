"""
===============================================================================
  PREPROCESSING.PY — Unified Setup, Material Database, & Domain Initialization
===============================================================================

Authors: Pranay Pandey & Antigravity Suite
"""

import json
import math
from pathlib import Path
import numpy as np


MATERIALS_DIR = Path(__file__).resolve().parent / "materials"


# =============================================================================
# 1. MATERIAL DATABASE LOADER
# =============================================================================
def load_material_properties(alloy_name="ss308"):
    """
    Loads thermodynamic, partition, and acoustic properties for a specified alloy.
    Supports: ss308, ss304, ss316, pure_ni, ni_cu, al_si, in718.
    """
    clean_name = alloy_name.lower().replace("-", "_").replace(" ", "_")
    json_path = MATERIALS_DIR / f"{clean_name}.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    # Default fallback to SS308 if not found
    fallback_path = MATERIALS_DIR / "ss308.json"
    if fallback_path.exists():
        with open(fallback_path, "r") as f:
            return json.load(f)

    # Hardcoded minimum safety fallback
    return {
        "alloy_name": "SS308 (AWS ER308L)",
        "thermodynamics": {
            "T_liquidus_K": 1730.0,
            "T_solidus_K": 1675.0,
            "delta_T0_K": 55.0,
            "latent_heat_J_kg": 270000.0,
            "specific_heat_J_kg_K": 650.0,
            "density_liquid_kg_m3": 6980.0,
            "thermal_diffusivity_m2_s": 4.8e-6,
            "interfacial_energy_J_m2": 0.35,
            "molar_volume_m3_mol": 7.4e-6
        },
        "calphad_partitioning": {
            "k_Cr": 0.95,
            "k_Ni": 0.78,
            "liquidus_slope_Cr_K_wt": -4.2
        },
        "acoustics": {
            "sound_speed_liquid_m_s": 4150.0,
            "blake_threshold_mpa": 0.69
        }
    }


# =============================================================================
# 2. DOMAIN & MESH INITIALIZATION
# =============================================================================
def initialize_domain_grid(domain_width_um=750.0, domain_height_um=1500.0, dx_um=1.0):
    """
    Sets up Cartesian grid coordinates for the computational domain.
    Default: 750 x 1500 um domain at dx = 1.0 um -> 750 x 1500 grid cells.
    """
    Nx = int(round(domain_width_um / dx_um))
    Ny = int(round(domain_height_um / dx_um))

    dx_m = dx_um * 1e-6
    dy_m = dx_m

    x_coords = np.linspace(0, domain_width_um, Nx)
    y_coords = np.linspace(0, domain_height_um, Ny)

    return {
        "Nx": Nx,
        "Ny": Ny,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dx_um": dx_um,
        "domain_width_um": domain_width_um,
        "domain_height_um": domain_height_um,
        "x_coords_um": x_coords,
        "y_coords_um": y_coords
    }


# =============================================================================
# 3. SEEDING & BOUNDARY INITIALIZATION
# =============================================================================
def initialize_seeding(Nx, Ny, seed_config="directional_substrate",
                       c0=0.20, num_substrate_seeds=15):
    """
    Generates initial field arrays: phi, c, grain_id, theta_grain.

    Configurations:
      - 'directional_substrate': Multiple columnar seeds along y=0 with near-vertical orientations.
      - 'center_single': Single circular seed at (Nx//2, Ny//2) for 4-fold dendrite tests.
      - 'corner_single': Single seed at (0, 0) matching Ferreira (2006).
      - 'equiaxed_random': Randomly distributed seeds for equiaxed growth.
    """
    phi = np.zeros((Nx, Ny), dtype=np.float64)
    c = np.full((Nx, Ny), c0, dtype=np.float64)
    grain_id = np.zeros((Nx, Ny), dtype=np.int32)
    theta_grain = np.zeros((Nx, Ny), dtype=np.float64)

    if seed_config == "center_single":
        cx, cy = Nx // 2, Ny // 2
        r_seed = 4
        for i in range(max(0, cx - r_seed), min(Nx, cx + r_seed + 1)):
            for j in range(max(0, cy - r_seed), min(Ny, cy + r_seed + 1)):
                if (i - cx)**2 + (j - cy)**2 <= r_seed**2:
                    phi[i, j] = 1.0
                    grain_id[i, j] = 1
                    theta_grain[i, j] = 0.0

    elif seed_config == "corner_single":
        r_seed = 5
        for i in range(r_seed + 1):
            for j in range(r_seed + 1):
                if i**2 + j**2 <= r_seed**2:
                    phi[i, j] = 1.0
                    grain_id[i, j] = 1
                    theta_grain[i, j] = math.pi / 4.0  # 45 deg diagonal growth

    elif seed_config == "directional_substrate":
        # Distinct columnar dendrite seeds spaced along bottom substrate with liquid channels
        n_seeds = max(3, min(num_substrate_seeds, Nx // 20))
        spacing = Nx // (n_seeds + 1)
        current_id = 1
        np.random.seed(42)

        # Baseline thin substrate layer at bottom (y <= 2)
        for i in range(Nx):
            for j in range(2):
                phi[i, j] = 1.0
                c[i, j] = c0 * 0.85
                grain_id[i, j] = 1
                theta_grain[i, j] = 0.0

        for s in range(1, n_seeds + 1):
            cx = s * spacing
            cy = 5
            r_seed = max(4, min(7, spacing // 4))
            orientation = np.random.uniform(-math.pi / 24.0, math.pi / 24.0)  # Near-vertical <001> growth

            for i in range(max(0, cx - r_seed), min(Nx, cx + r_seed + 1)):
                for j in range(2, min(Ny, cy + r_seed + 1)):
                    if (i - cx)**2 + (j - cy)**2 <= r_seed**2:
                        phi[i, j] = 1.0
                        c[i, j] = c0 * 0.85  # Partitioned solid dendrite core
                        grain_id[i, j] = current_id
                        theta_grain[i, j] = orientation

            current_id += 1

    elif seed_config == "equiaxed_random":
        np.random.seed(42)
        num_seeds = 12
        for s in range(1, num_seeds + 1):
            rx = np.random.randint(10, Nx - 10)
            ry = np.random.randint(10, Ny - 10)
            rand_theta = np.random.uniform(0.0, math.pi / 2.0)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    phi[rx + di, ry + dj] = 1.0
                    grain_id[rx + di, ry + dj] = s
                    theta_grain[rx + di, ry + dj] = rand_theta

    return phi, c, grain_id, theta_grain


# =============================================================================
# 4. THERMAL CONDITIONS BUILDER
# =============================================================================
def build_thermal_profile(Ny, dy_m, G_thermal_K_m=1.5e4, V_pull_mm_s=75.0,
                          T_bottom_0_K=1720.0, cooling_rate_K_s=1500.0):
    """
    Builds thermal conditions for directional solidification:
      T(y, t) = T_bottom_0 + G_thermal * y - cooling_rate * t
    Ensures steady front progression across domain without artificial pinning.
    """
    V_pull_m_s = V_pull_mm_s * 1e-3

    return {
        "G_thermal": G_thermal_K_m,
        "V_pull": V_pull_m_s,
        "T_bottom_0": T_bottom_0_K,
        "cooling_rate_K_s": cooling_rate_K_s
    }


# =============================================================================
# 5. UNIFIED CONFIGURATION GENERATOR
# =============================================================================
def generate_simulation_config(alloy="ss308",
                               mode="directional_cet",
                               freq_khz=0.0,
                               amp_um=0.0,
                               domain_width_um=300.0,
                               domain_height_um=450.0,
                               dx_um=1.0,
                               total_time_ms=30.0,
                               save_interval_ms=None):
    """
    Assembles a complete, self-contained configuration dictionary for a simulation run.
    """
    mat_props = load_material_properties(alloy)
    grid = initialize_domain_grid(domain_width_um, domain_height_um, dx_um)

    dx_m = grid["dx_m"]
    D_phi = 1.44e-7
    dt_phi_cfl = 0.20 * (dx_m * dx_m) / D_phi
    dt = min(1.0e-6, dt_phi_cfl)

    total_time_s = total_time_ms * 1e-3
    total_steps = int(math.ceil(total_time_s / dt))
    if save_interval_ms is None or save_interval_ms >= total_time_ms:
        save_interval_ms = max(0.1, total_time_ms / 30.0)
    save_interval_s = save_interval_ms * 1e-3
    save_step_interval = max(1, int(round(save_interval_s / dt)))

    T_liq = mat_props["thermodynamics"].get("T_liquidus_K", 1730.0)
    thermal = build_thermal_profile(grid["Ny"], grid["dy_m"],
                                    G_thermal_K_m=1.5e4, V_pull_mm_s=75.0,
                                    T_bottom_0_K=T_liq - 8.0)

    return {
        "alloy": alloy,
        "mode": mode,
        "freq_khz": freq_khz,
        "amp_um": amp_um,
        "mat_props": mat_props,
        "grid": grid,
        "thermal": thermal,
        "dt": dt,
        "total_time_ms": total_time_ms,
        "total_time_s": total_time_s,
        "total_steps": total_steps,
        "save_step_interval": save_step_interval,
        "output_dir_name": f"{alloy}_{mode}_{int(freq_khz)}khz"
    }
