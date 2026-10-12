"""
===============================================================================
  POSTPROCESSING.PY — Microstructure Evolution, SEM/EBSD IPF Dashboards, & GIFs
===============================================================================

Authors: Pranay Pandey & Antigravity Suite
Provides publication-quality visualizations, 4-panel microstructure dashboards,
multi-stage evolution galleries, and high-resolution animated GIFs.
"""

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import distance_transform_edt, find_objects, label, sobel

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# =============================================================================
# 0. CURATED COLORMAPS & METALLOGRAPHIC PALETTES
# =============================================================================
def _ebsd_hsv_cmap():
    """Returns standard HSV orientation colormap."""
    return plt.cm.hsv


def _metallography_cmap():
    """
    Curated metallographic colormap:
    Liquid = deep dark navy/black (#080A10)
    Interface = fiery amber / orange
    Solid dendrite = warm cream / gold (#FEF9E7)
    """
    colors = [
        (0.03, 0.04, 0.06),  # Deep dark navy liquid (0.0)
        (0.20, 0.05, 0.15),  # Deep purple interface
        (0.85, 0.30, 0.05),  # Amber/orange front
        (0.98, 0.85, 0.40),  # Warm gold dendrite
        (1.00, 0.98, 0.90),  # Cream solid core (1.0)
    ]
    return LinearSegmentedColormap.from_list("metallography_gold", colors, N=256)


def _solute_cmap():
    """High-contrast viridis/plasma colormap for solute segregation."""
    return plt.cm.viridis


def _temperature_cmap():
    """Coolwarm-style colormap for thermal front."""
    return plt.cm.coolwarm


METALLOGRAPHY_CMAP = _metallography_cmap()
SOLUTE_CMAP = _solute_cmap()
TEMP_CMAP = _temperature_cmap()
EBSD_CMAP = _ebsd_hsv_cmap()


# =============================================================================
# 1. METRICS EXTRACTION (SOLID FRACTION, TIP TRACKING, SDAS, CET)
# =============================================================================
def calculate_solid_fraction(phi):
    """Calculates overall solid fraction f_s in [0, 1]."""
    return float(np.mean(phi >= 0.5))


def track_dendrite_tip_position(phi, dy_m):
    """
    Finds the highest y-coordinate (in meters) where solid phase (phi >= 0.5) exists.
    """
    solid_indices = np.where(phi >= 0.5)
    if len(solid_indices[1]) == 0:
        return 0.0
    max_j = np.max(solid_indices[1])
    return float(max_j * dy_m)


def calculate_sdas_fft(phi, dx_um):
    """
    Estimates Secondary Dendrite Arm Spacing (SDAS) in micrometers using 1D autocorrelation / FFT
    along horizontal cross-sections of the solidifying front.
    """
    Nx, Ny = phi.shape
    mid_j = Ny // 2
    slice_data = phi[:, mid_j]
    if np.sum(slice_data > 0.3) < 10:
        solid_slices = []
        for j in range(Ny):
            if 0.1 < np.mean(phi[:, j]) < 0.9:
                solid_slices.append(phi[:, j])
        if len(solid_slices) == 0:
            return 0.0
        slice_data = np.mean(solid_slices, axis=0)

    data = slice_data - np.mean(slice_data)
    if np.all(data == 0):
        return 0.0
    corr = np.correlate(data, data, mode="full")
    corr = corr[len(corr) // 2:]

    peaks = []
    for k in range(1, len(corr) - 1):
        if corr[k] > corr[k - 1] and corr[k] > corr[k + 1]:
            peaks.append(k)

    if len(peaks) > 0:
        sdas_cells = peaks[0]
        return float(sdas_cells * dx_um)
    return 0.0


def analyze_grain_morphology(grain_id, phi, dx_um, dy_um):
    """
    Extracts grain statistics and Columnar-to-Equiaxed Transition (CET) volume fraction:
      - Grain areas (um^2)
      - Aspect ratio (major / minor axis)
      - Grain classification: Columnar (aspect >= 2.5) vs Equiaxed (aspect < 2.5)
      - Hunt's CET Criterion volume fraction evaluation
    """
    unique_ids = np.unique(grain_id[grain_id > 0])
    grains = []
    col_count = 0
    eq_count = 0
    col_area = 0.0
    eq_area = 0.0

    cell_area_um2 = dx_um * dy_um

    for gid in unique_ids:
        mask = (grain_id == gid) & (phi >= 0.5)
        num_cells = np.sum(mask)
        if num_cells < 15:
            continue

        area_um2 = num_cells * cell_area_um2
        equiv_diam_um = 2.0 * math.sqrt(area_um2 / math.pi)

        coords = np.argwhere(mask)
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        height_um = (max_y - min_y + 1) * dy_um
        width_um = (max_x - min_x + 1) * dx_um

        aspect_ratio = max(height_um / max(width_um, 1e-3), width_um / max(height_um, 1e-3))
        is_columnar = aspect_ratio >= 2.5 or (min_y <= 5 and height_um > 20.0)
        grain_type = "Columnar" if is_columnar else "Equiaxed"

        if is_columnar:
            col_count += 1
            col_area += area_um2
        else:
            eq_count += 1
            eq_area += area_um2

        grains.append({
            "grain_id": int(gid),
            "area_um2": float(area_um2),
            "equiv_diam_um": float(equiv_diam_um),
            "aspect_ratio": float(aspect_ratio),
            "type": grain_type,
            "center_y_um": float(np.mean(coords[:, 1]) * dy_um)
        })

    total_solid_area = col_area + eq_area
    equiaxed_fraction = eq_area / max(total_solid_area, 1e-6)

    # Hunt CET Classification
    if equiaxed_fraction < 0.0066:
        cet_regime = "Fully Columnar (Hunt Criterion: Phi_eq < 0.66%)"
    elif equiaxed_fraction < 0.49:
        cet_regime = "Mixed Columnar + Equiaxed (Hunt Criterion: 0.66% <= Phi_eq < 49%)"
    else:
        cet_regime = "Fully Equiaxed CET Achieved (Hunt Criterion: Phi_eq >= 49%)"

    return {
        "total_grains": len(grains),
        "columnar_count": col_count,
        "equiaxed_count": eq_count,
        "columnar_area_um2": col_area,
        "equiaxed_area_um2": eq_area,
        "equiaxed_fraction": equiaxed_fraction,
        "cet_regime": cet_regime,
        "grains": grains
    }


# =============================================================================
# 2. SEM-LIKE EBSD IPF COLORMAP ENGINE
# =============================================================================
def generate_ebsd_ipf_rgb(grain_id, theta_grain, phi, liquid_color=None):
    """
    Renders an authentic SEM / EBSD-like Inverse Pole Figure (IPF) orientation map:
      - Crystal orientation mapped via circular projection onto HSV -> RGB
      - Deep sleek metallic navy background for liquid phase
      - Inscribed crisp white/black grain boundaries
    """
    Nx, Ny = phi.shape
    rgb_img = np.zeros((Ny, Nx, 3), dtype=np.float32)

    # Standard IPF coloring using circular orientation projection
    angle_norm = (np.abs(theta_grain) % (math.pi / 2.0)) / (math.pi / 2.0)  # [0, 1]

    # Map angle to HSV, then to RGB
    hue = np.clip(angle_norm * 0.85, 0.0, 1.0)
    sat = np.ones_like(hue) * 0.88
    val = np.ones_like(hue) * 0.92

    # Vectorized HSV to RGB conversion
    i_h = np.floor(hue * 6.0).astype(int)
    f_h = (hue * 6.0) - i_h
    p_h = val * (1.0 - sat)
    q_h = val * (1.0 - f_h * sat)
    t_h = val * (1.0 - (1.0 - f_h) * sat)

    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)

    idx0 = (i_h % 6 == 0)
    r[idx0] = val[idx0]; g[idx0] = t_h[idx0]; b[idx0] = p_h[idx0]
    idx1 = (i_h % 6 == 1)
    r[idx1] = q_h[idx1]; g[idx1] = val[idx1]; b[idx1] = p_h[idx1]
    idx2 = (i_h % 6 == 2)
    r[idx2] = p_h[idx2]; g[idx2] = val[idx2]; b[idx2] = t_h[idx2]
    idx3 = (i_h % 6 == 3)
    r[idx3] = p_h[idx3]; g[idx3] = q_h[idx3]; b[idx3] = val[idx3]
    idx4 = (i_h % 6 == 4)
    r[idx4] = t_h[idx4]; g[idx4] = p_h[idx4]; b[idx4] = val[idx4]
    idx5 = (i_h % 6 == 5)
    r[idx5] = val[idx5]; g[idx5] = p_h[idx5]; b[idx5] = q_h[idx5]

    # Transpose to (Ny, Nx) for Cartesian plot
    r = r.T; g = g.T; b = b.T
    phi_T = phi.T
    grain_id_T = grain_id.T

    if liquid_color is None:
        liquid_color = np.array([0.08, 0.10, 0.16], dtype=np.float32)

    # Detect grain boundaries using ID differences in neighborhood
    padded_gid = np.pad(grain_id_T, 1, mode="edge")
    diff_x = (padded_gid[1:-1, 2:] != grain_id_T) | (padded_gid[1:-1, :-2] != grain_id_T)
    diff_y = (padded_gid[2:, 1:-1] != grain_id_T) | (padded_gid[:-2, 1:-1] != grain_id_T)
    gb_mask = (diff_x | diff_y) & (phi_T > 0.4)

    # Blend solid orientation color with sleek dark liquid matrix
    blend_factor = np.clip((phi_T - 0.2) / 0.6, 0.0, 1.0)
    for c_idx, channel in enumerate((r, g, b)):
        rgb_img[:, :, c_idx] = blend_factor * channel + (1.0 - blend_factor) * liquid_color[c_idx]
        # Inscribe grain boundaries
        rgb_img[gb_mask, c_idx] = 0.95

    return np.clip(rgb_img, 0.0, 1.0)


# =============================================================================
# 3. ICONIC 4-PANEL MICROSTRUCTURE DASHBOARD
# =============================================================================
def plot_microstructure_dashboard(phi, c, grain_id, theta_grain,
                                  time_history, fs_history, tip_vel_history,
                                  config, output_filepath, title_suffix=""):
    """
    Creates the comprehensive 4-panel publication dashboard:
      Panel 1: SEM-like EBSD IPF Grain Orientation Map with HUD overlay
      Panel 2: Solute Microsegregation Concentration Field c(x,y)
      Panel 3: Solidification Kinetics (Solid Fraction & Tip Velocity vs Time)
      Panel 4: Grain Aspect Ratio & CET Distribution (Columnar vs Equiaxed)
    """
    grid = config["grid"]
    domain_w = grid["domain_width_um"]
    domain_h = grid["domain_height_um"]
    alloy = config["alloy"].upper()
    freq = config.get("freq_khz", 0.0)

    fig = plt.figure(figsize=(16, 12), dpi=150, facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.25)

    # ---------------------------------------------------------
    # Panel 1: Authentic Anisotropic Dendritic Morphology
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phi.T, origin="lower", extent=[0, domain_w, 0, domain_h],
                     aspect="auto", cmap=plt.cm.inferno, vmin=0.0, vmax=1.0)
    ax1.contour(phi.T, levels=[0.5], colors="white", linewidths=0.8,
                extent=[0, domain_w, 0, domain_h], alpha=0.85)
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label("Phase Field Order Parameter φ (0=Liquid, 1=Solid)", fontsize=10, fontweight="bold")
    ax1.set_title(f"(a) Anisotropic Dendritic Morphology — {alloy} [{freq:.1f} kHz UVA]",
                  fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Width (μm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Height (μm) [Build Direction]", fontsize=11, fontweight="bold")

    morph = analyze_grain_morphology(grain_id, phi, grid["dx_um"], grid["dx_um"])
    info_str = (f"Regime: {morph['cet_regime']}\n"
                f"Equiaxed Fraction: {morph['equiaxed_fraction']*100:.1f}%\n"
                f"Columnar: {morph['columnar_count']} | Equiaxed: {morph['equiaxed_count']}")
    ax1.text(0.03, 0.96, info_str, transform=ax1.transAxes, fontsize=9.5,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.92, ec="#64748B", lw=1.2))

    # ---------------------------------------------------------
    # Panel 2: Solute Concentration Field
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(c.T, origin="lower", extent=[0, domain_w, 0, domain_h],
                     aspect="auto", cmap=SOLUTE_CMAP)
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label("Solute Concentration c (wt% / mol fraction)", fontsize=10, fontweight="bold")
    ax2.set_title(f"(b) Solute Microsegregation Field c(x,y) — {alloy}",
                  fontsize=13, fontweight="bold", pad=8)
    ax2.set_xlabel("Width (μm)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Height (μm)", fontsize=11, fontweight="bold")

    # ---------------------------------------------------------
    # Panel 3: Solid Fraction & Tip Velocity History
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()

    if len(time_history) > 0:
        p1, = ax3.plot(time_history, [f * 100 for f in fs_history], "b-", lw=2.4, label="Solid Fraction (%)")
        p2, = ax3_twin.plot(time_history, tip_vel_history, "r--", lw=2.0, label="Tip Velocity (mm/s)")
        ax3.set_xlabel("Solidification Time (ms)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Solid Fraction (%)", fontsize=11, color="blue", fontweight="bold")
        ax3_twin.set_ylabel("Tip Velocity (mm/s)", fontsize=11, color="red", fontweight="bold")
        ax3.set_title("(c) Solidification Kinetics & Solid Fraction Evolution", fontsize=13, fontweight="bold", pad=8)
        ax3.grid(True, linestyle=":", alpha=0.6)
        lines = [p1, p2]
        ax3.legend(lines, [l.get_label() for l in lines], loc="upper left", framealpha=0.9)

    # ---------------------------------------------------------
    # Panel 4: Grain Aspect Ratio & CET Distribution
    # ---------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    grains = morph["grains"]
    if len(grains) > 0:
        aspects = [g["aspect_ratio"] for g in grains]
        types = [g["type"] for g in grains]

        col_aspects = [a for a, t in zip(aspects, types) if t == "Columnar"]
        eq_aspects = [a for a, t in zip(aspects, types) if t == "Equiaxed"]

        max_asp = max(max(aspects, default=5.0), 5.0)
        bins = np.linspace(1.0, max_asp, 15)
        ax4.hist([col_aspects, eq_aspects], bins=bins, stacked=True,
                 color=["#E74C3C", "#2ECC71"], label=["Columnar Grains", "Equiaxed Grains"],
                 edgecolor="black", alpha=0.85)
        ax4.set_xlabel("Grain Aspect Ratio (Length / Width)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Grain Count", fontsize=11, fontweight="bold")
        ax4.set_title("(d) Grain Morphology & CET Aspect Ratio Distribution", fontsize=13, fontweight="bold", pad=8)
        ax4.axvline(2.5, color="black", linestyle="--", lw=1.8, label="CET Transition Threshold (2.5)")
        ax4.legend(loc="upper right", framealpha=0.9)
        ax4.grid(True, linestyle=":", alpha=0.6)
    else:
        ax4.text(0.5, 0.5, "Single Solid Front (Substrate Seeded)", ha="center", va="center", fontsize=11)

    fig.suptitle(f"Unified PFM Simulation Suite — {alloy} Directional Solidification {title_suffix}",
                 fontsize=15, fontweight="bold", y=0.98)

    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# 4. MULTI-STAGE MICROSTRUCTURE EVOLUTION GALLERY
# =============================================================================
def plot_multistage_morphology_evolution(snapshots_history, output_filepath,
                                         alloy="SS308", freq_khz=20.0,
                                         domain_width_um=500.0, domain_height_um=1000.0):
    """
    Renders the classic multi-stage morphology evolution gallery showing progressive
    solidification across 4 key temporal intervals (e.g. 25%, 50%, 75%, 100% growth).
    Matches the popular legacy study format (pfm_multistage_morphology_evolution.png).
    """
    n_stages = len(snapshots_history)
    if n_stages == 0:
        return None

    fig, axes = plt.subplots(1, n_stages, figsize=(4.5 * n_stages, 8), dpi=150, sharey=True)
    if n_stages == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0F172A")  # Sleek dark navy container

    for idx, snap in enumerate(snapshots_history):
        ax = axes[idx]
        ax.set_facecolor("#0F172A")
        phi = snap["phi"]
        grain_id = snap.get("grain_id", np.ones_like(phi, dtype=int))
        theta = snap.get("theta", np.zeros_like(phi))
        t_ms = snap.get("time_ms", 0.0)
        fs = snap.get("solid_fraction", float(np.mean(phi >= 0.5)))
        n_grains = snap.get("n_grains", len(np.unique(grain_id[grain_id > 0])))
        stage_label = snap.get("stage_label", f"Stage {idx+1}")

        # Render genuine dendrite morphology with inferno colormap
        ax.imshow(phi.T, origin="lower", extent=[0, domain_width_um, 0, domain_height_um],
                  aspect="auto", cmap=plt.cm.inferno, vmin=0.0, vmax=1.0)
        ax.contour(phi.T, levels=[0.5], colors="white", linewidths=0.7,
                   extent=[0, domain_width_um, 0, domain_height_um], alpha=0.85)

        # Inscribed badge overlay
        badge = f"{stage_label}\nt = {t_ms:.1f} ms | fs = {fs*100:.1f}%\nGrains: {n_grains}"
        ax.text(0.04, 0.95, badge, transform=ax.transAxes,
                color="#F8FAFC", fontsize=9.5, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="#1E293B", ec="#475569", alpha=0.92))

        ax.set_title(f"{stage_label}", color="#F8FAFC", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Width (μm)", color="#94A3B8", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Height (μm) [Growth Direction]", color="#94A3B8", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94A3B8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plt.suptitle(f"{alloy.upper()} Directional Solidification Evolution [{freq_khz:.1f} kHz UVA]\n"
                 f"Multi-Stage Temporal Microstructure Development",
                 color="#F8FAFC", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    out = Path(output_filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    return str(out)


# =============================================================================
# 5. MULTI-FREQUENCY / COMPARATIVE MORPHOLOGY GALLERY
# =============================================================================
def plot_frequency_comparison_gallery(results_list, output_filepath, title="UVA-WAAM Frequency Study"):
    """
    Renders side-by-side morphology gallery comparing different cases (e.g. 0 kHz vs 20 kHz vs 30 kHz).
    Matches the legacy unclamped_morphology_comparison.png and pfm_10freq_morphology_gallery.png styles.
    """
    n_cases = len(results_list)
    if n_cases == 0:
        return None

    fig, axes = plt.subplots(1, n_cases, figsize=(5.0 * n_cases, 9), dpi=150, sharey=True)
    if n_cases == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0B0F19")

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        ax.set_facecolor("#0B0F19")
        phi = res["phi"]
        grain_id = res.get("grain_id", np.ones_like(phi, dtype=int))
        theta = res.get("theta", np.zeros_like(phi))
        label = res.get("label", f"Case {idx+1}")
        fs = res.get("solid_fraction", float(np.mean(phi >= 0.5)))
        n_grains = res.get("total_grains", len(np.unique(grain_id[grain_id > 0])))
        eq_grains = res.get("equiaxed_grains", 0)

        domain_w = res.get("domain_width_um", 500.0)
        domain_h = res.get("domain_height_um", 1000.0)

        # Render authentic dendritic morphology with inferno colormap
        im = ax.imshow(phi.T, origin="lower", extent=[0, domain_w, 0, domain_h], aspect="auto", cmap=plt.cm.inferno, vmin=0.0, vmax=1.0)
        ax.contour(phi.T, levels=[0.5], colors="white", linewidths=0.9, extent=[0, domain_w, 0, domain_h], alpha=0.85)
        masked_gid = np.where(phi >= 0.4, grain_id, 0)
        u_gids = np.unique(masked_gid[masked_gid > 0])
        if len(u_gids) > 1:
            ax.contour(masked_gid.T, levels=u_gids, extent=[0, domain_w, 0, domain_h], colors="cyan", linewidths=0.6, alpha=0.5)

        badge = (f"{label}\n"
                 f"Total Grains: {n_grains} (Eq: {eq_grains})\n"
                 f"Solid Fraction: {fs*100:.1f}%")
        ax.text(0.04, 0.95, badge, transform=ax.transAxes,
                color="#F8FAFC", fontsize=9.5, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="#1E293B", ec="#475569", alpha=0.92))

        ax.set_title(label, color="#F8FAFC", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Width (μm)", color="#94A3B8", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Height (μm) [Build Direction]", color="#94A3B8", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94A3B8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plt.suptitle(f"{title}\nComparative Microstructure Evolution & Columnar-to-Equiaxed Transition",
                 color="#F8FAFC", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    out = Path(output_filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0B0F19")
    plt.close(fig)
    return str(out)


# =============================================================================
# 6. HIGH-RESOLUTION DYNAMIC-ZOOM GIF SNAPSHOT ENGINE
# =============================================================================
def save_frame_snapshot(phi, dx_um, dy_um, step, time_ms, frame_dir,
                        grain_id=None, theta_grain=None, field_c=None, field_T=None,
                        mode="ebsd", ac_label="UVA Simulation"):
    """
    Saves a single frame for animated GIF assembly featuring:
      - Sleek dark metallic theme (#111827)
      - EBSD crystal orientation map with crisp boundaries
      - Dynamic camera auto-framing on the active solidifying front
      - Semi-transparent HUD status badge with real-time stats
    """
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    Nx, Ny = phi.shape
    width_um = Nx * dx_um
    height_um = Ny * dy_um

    fig, ax = plt.subplots(figsize=(6.5, 8.5), dpi=120)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    if grain_id is None:
        grain_id = np.ones_like(phi, dtype=int)
    if theta_grain is None:
        theta_grain = np.zeros_like(phi)

    # Render genuine dendrite morphology with inferno
    ax.imshow(phi.T, origin="lower", extent=[0, width_um, 0, height_um],
              aspect="auto", cmap=plt.cm.inferno, vmin=0.0, vmax=1.0)
    ax.contour(phi.T, levels=[0.5], colors="white", linewidths=0.8,
               extent=[0, width_um, 0, height_um], alpha=0.85)
    ax.set_xlim(0, width_um)
    ax.set_ylim(0, height_um)

    # HUD Status Badge
    fs = float(np.mean(phi >= 0.5))
    unique_grains = len(np.unique(grain_id[(grain_id > 0) & (phi >= 0.5)]))
    badge_text = (f"{ac_label}\n"
                  f"t = {time_ms:.1f} ms | fs = {fs*100:.1f}%\n"
                  f"Active Grains: {unique_grains}")
    ax.text(0.04, 0.94, badge_text, transform=ax.transAxes,
            color="#F9FAFB", fontsize=10, fontweight="bold", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1F2937", ec="#374151", alpha=0.92))

    ax.set_xlabel("Width X (μm)", color="#9CA3AF", fontsize=9, fontweight="bold")
    ax.set_ylabel("Build Height Y (μm)", color="#9CA3AF", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    fig.tight_layout()
    frame_path = frame_dir / f"frame_{step:06d}.png"
    plt.savefig(frame_path, dpi=120, facecolor="#111827")
    plt.close(fig)
    return str(frame_path)


def assemble_gif(frame_dir, output_gif_path, fps=8, loop=0):
    """
    Assembles numbered frame PNGs in frame_dir into a high-quality animated GIF.
    """
    frame_dir = Path(frame_dir)
    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frame_dir.glob("frame_*.png"))
    if len(frame_files) == 0:
        print("[WARNING] No frames found for GIF assembly.")
        return None

    print(f"  Assembling GIF from {len(frame_files)} frames -> {output_gif_path}")

    if HAS_PIL:
        images = []
        for fp in frame_files:
            img = Image.open(fp)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        duration_ms = int(1000.0 / max(1, fps))
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=loop,
            optimize=True
        )
        print(f"  GIF saved successfully: {output_gif_path} ({len(images)} frames, {fps} fps)")
        return str(output_gif_path)

    elif HAS_IMAGEIO:
        images = [imageio.imread(fp) for fp in frame_files]
        imageio.mimsave(output_gif_path, images, fps=fps)
        print(f"  GIF saved successfully (imageio): {output_gif_path}")
        return str(output_gif_path)

    else:
        print("[ERROR] Neither PIL nor imageio is available for GIF assembly.")
        return None


# =============================================================================
# 7. TIME-SERIES & SINGLE-FIELD PLOTTING (FOR VALIDATION BENCHMARKS)
# =============================================================================
def _configure_axes(ax, xlabel, ylabel, title, fontsize=11):
    """Consistent axis styling for standard scientific plots."""
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=fontsize - 1)


def plot_phase_and_temperature_field(phi, T, dx_um, dy_um, title_prefix, output_path):
    """Side-by-side plot of phase field phi and temperature field T."""
    Nx, Ny = phi.shape
    extent = [0, Nx * dx_um, 0, Ny * dy_um]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150,
                             constrained_layout=True, facecolor="white")

    im0 = axes[0].imshow(phi.T, origin="lower", extent=extent,
                         cmap=METALLOGRAPHY_CMAP, vmin=0.0, vmax=1.0,
                         interpolation="bilinear", aspect="equal")
    cb0 = plt.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    cb0.set_label("Phase Order Parameter φ", fontsize=10, fontweight="bold")
    _configure_axes(axes[0], "Width (μm)", "Height (μm)",
                    f"{title_prefix}: Phase Field φ(x,y)\nSolid=1.0, Liquid=0.0")

    im1 = axes[1].imshow(T.T, origin="lower", extent=extent,
                         cmap=TEMP_CMAP, interpolation="bilinear", aspect="equal")
    cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    cb1.set_label("Temperature (K)", fontsize=10, fontweight="bold")
    _configure_axes(axes[1], "Width (μm)", "Height (μm)",
                    f"{title_prefix}: Temperature T(x,y) (K)\nLatent Heat Diffusion Front")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def plot_solute_segregation_field(phi, c, dx_um, dy_um, title_prefix, output_path):
    """Plots solute concentration field and 1D partition profile across interface."""
    Nx, Ny = phi.shape
    extent = [0, Nx * dx_um, 0, Ny * dy_um]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150,
                             constrained_layout=True, facecolor="white")

    # Panel 1: Phase field
    axes[0].imshow(phi.T, origin="lower", extent=extent,
                   cmap=METALLOGRAPHY_CMAP, vmin=0.0, vmax=1.0,
                   interpolation="bilinear", aspect="equal")
    _configure_axes(axes[0], "Width (μm)", "Height (μm)",
                    f"{title_prefix}: Dendrite Phase Field φ")

    # Panel 2: Solute field
    im1 = axes[1].imshow(c.T, origin="lower", extent=extent,
                         cmap=SOLUTE_CMAP, interpolation="bilinear", aspect="equal")
    cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    cb1.set_label("Solute Fraction c", fontsize=10, fontweight="bold")
    _configure_axes(axes[1], "Width (μm)", "Height (μm)",
                    f"{title_prefix}: Solute Segregation c(x,y)")

    # Panel 3: 1D Profile
    diag_c = [c[k, k] for k in range(min(Nx, Ny))]
    diag_phi = [phi[k, k] for k in range(min(Nx, Ny))]
    s_coords = np.linspace(0, math.sqrt(2.0) * min(Nx, Ny) * dx_um, min(Nx, Ny))

    ax3_twin = axes[2].twinx()
    p1, = axes[2].plot(s_coords, diag_c, "b-", lw=2.2, label="Solute c")
    p2, = ax3_twin.plot(s_coords, diag_phi, "r--", lw=1.8, label="Phase φ")
    axes[2].set_xlabel("Diagonal Distance s (μm)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Solute Concentration c", fontsize=10, fontweight="bold", color="blue")
    ax3_twin.set_ylabel("Phase Field φ", fontsize=10, fontweight="bold", color="red")
    axes[2].set_title("Solute Partitioning Profile Across Front", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    lines = [p1, p2]
    axes[2].legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def plot_directional_grain_structure(phi, grain_id, theta_grain, dx_um, dy_um,
                                     title_prefix, output_path):
    """Directional solidification multi-grain map using EBSD IPF rendering."""
    Nx, Ny = phi.shape
    extent = [0, Nx * dx_um, 0, Ny * dy_um]

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150, facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    im = ax.imshow(phi.T, origin="lower", extent=extent, aspect="auto", cmap=plt.cm.inferno, vmin=0.0, vmax=1.0)
    ax.contour(phi.T, levels=[0.5], extent=extent, colors="white", linewidths=1.2, alpha=0.9)
    masked_gid = np.where(phi >= 0.4, grain_id, 0)
    u_gids = np.unique(masked_gid[masked_gid > 0])
    if len(u_gids) > 1:
        ax.contour(masked_gid.T, levels=u_gids, extent=extent, colors="cyan", linewidths=0.7, alpha=0.6)

    morph = analyze_grain_morphology(grain_id, phi, dx_um, dy_um)
    info_str = (f"Regime: {morph['cet_regime']}\n"
                f"Equiaxed Fraction: {morph['equiaxed_fraction']*100:.1f}%\n"
                f"Columnar: {morph['columnar_count']} | Equiaxed: {morph['equiaxed_count']}")
    ax.text(0.03, 0.97, info_str, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.92, ec="gray"))

    _configure_axes(ax, "Width (μm)", "Height (μm) [Growth Direction]",
                    f"{title_prefix}: Directional Solidification EBSD Grain Map")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def plot_metrics_evolution(time_history, fs_history, tip_vel_history, sdas_history,
                           title_prefix, output_path):
    """Plots solid fraction, tip velocity, and SDAS vs time."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150,
                             constrained_layout=True, facecolor="white")

    axes[0].plot(time_history, [fs * 100 for fs in fs_history], "b-o", lw=2.0, markersize=3.5)
    _configure_axes(axes[0], "Time (ms)", "Solid Fraction (%)",
                    f"{title_prefix}: Solid Fraction fs(t)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylim(-2, 105)

    axes[1].plot(time_history, tip_vel_history, "r-s", lw=2.0, markersize=3.5)
    _configure_axes(axes[1], "Time (ms)", "Tip Velocity (mm/s)",
                    f"{title_prefix}: Tip Velocity V_tip(t)")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    axes[2].plot(time_history, sdas_history, "g-^", lw=2.0, markersize=3.5)
    _configure_axes(axes[2], "Time (ms)", "SDAS λ2 (μm)",
                    f"{title_prefix}: Secondary Arm Spacing SDAS(t)")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def export_metrics_csv(filepath, time_history, fs_history, tip_vel_history, sdas_history):
    """Saves parametric evolution metrics to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("time_ms,solid_fraction,tip_velocity_mm_s,sdas_um\n")
        for t, fs, v, sdas in zip(time_history, fs_history, tip_vel_history, sdas_history):
            f.write(f"{t:.4f},{fs:.6f},{v:.4f},{sdas:.4f}\n")
    return str(path)


# =============================================================================
# 11. SS308 50.0-SECOND EXTENDED MULTI-INTERVAL DELIVERABLES
# Reproduces:
#   - pfm_ss308_50s_multi_interval_gallery.png (18 panels)
#   - pfm_ss308_50s_phase_maps_micro_macro.png (9 panels)
#   - pfm_ss308_50s_quantitative_metrics.png (4 panels)
#   - ss308_50s_microstructure_evolution.gif
# =============================================================================
def plot_ss308_50s_multi_interval_gallery(all_results, output_path):
    """
    Renders the 18-panel multi-interval snapshot gallery:
      3 Cases (0k, 20k, 30k) x 6 Time Intervals (0.04s, 0.08s, 0.16s, 1.0s, 10.0s, 50.0s)
    """
    time_points = [0.04, 0.08, 0.16, 1.0, 10.0, 50.0]
    case_ids = ["0khz", "20khz", "30khz"]
    case_titles = ["Baseline (0 kHz)", "Power UVA (20 kHz)", "Peak Resonance (30 kHz)"]

    fig, axes = plt.subplots(3, 6, figsize=(28, 14), dpi=200)
    cmap_dendrite = plt.cm.inferno
    cmap_phase = plt.cm.magma

    for row_idx, (cid, ctitle) in enumerate(zip(case_ids, case_titles)):
        c_res = all_results[cid]
        snaps = c_res["snapshots"]

        for col_idx, t_val in enumerate(time_points):
            ax = axes[row_idx, col_idx]
            snap = snaps[t_val]

            if t_val <= 0.16:
                ax.imshow(snap["phi"].T, cmap=cmap_dendrite, origin="lower", extent=[0, 200, 0, 200], vmin=0, vmax=1)
                phase_label = f"(fs = {snap['fs']*100:.1f}%\nLiquid fl = {(1-snap['fs'])*100:.1f}%)"
            else:
                ax.imshow(snap["austenite_map"].T, cmap=cmap_phase, origin="lower", extent=[0, 200, 0, 200], vmin=0, vmax=1)
                phase_label = f"(γ-Austenite = {snap['fg']:.1f}%\nResidual δ = {100-snap['fg']:.1f}%)"

            if t_val >= 0.16 and c_res['n_grains'] > 1:
                g_id = snap["grain_id"]
                u_grains = np.unique(g_id)
                if len(u_grains) > 1:
                    ax.contour(g_id.T, levels=u_grains[1:] - 0.5, colors="cyan", linewidths=0.75, origin="lower", extent=[0, 200, 0, 200])

            if row_idx == 0:
                ax.set_title(f"t = {t_val} s\n{phase_label}", fontsize=10.5, fontweight="bold", pad=6)
            else:
                ax.set_title(f"t = {t_val} s\n{phase_label}", fontsize=10.0, pad=4)

            if col_idx == 0:
                ax.set_ylabel(f"{ctitle}\n(N = {c_res['n_grains']} Grains)\nY Position (µm)", fontsize=11, fontweight="bold")
            else:
                ax.set_yticklabels([])

            if row_idx == 2:
                ax.set_xlabel("X Position (µm)", fontsize=10.5)
            else:
                ax.set_xticklabels([])

            ax.text(8, 178, f"T = {snap['temp_K']:.0f} K",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
                    fontsize=9, fontweight="bold")

    plt.suptitle("UVA-WAAM Stainless Steel 308 (ER308L) Microstructural Evolution across 50.0 Seconds\n"
                 "Stage 1 (0 to 0.16 s): Dendritic Solidification & Acoustic CET | Stage 2 (0.16 to 50.0 s): Solid-State Austenite Nucleation & Growth",
                 fontsize=14.5, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.98, hspace=0.28, wspace=0.15)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


def plot_ss308_50s_phase_maps(all_results, output_path):
    """
    Renders 9-panel micro vs macro phase distribution maps at 1s, 10s, and 50s.
    """
    case_ids = ["0khz", "20khz", "30khz"]
    case_titles = ["Baseline (0 kHz)", "Power UVA (20 kHz)", "Peak Resonance (30 kHz)"]
    post_times = [1.0, 10.0, 50.0]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=200)

    for row_idx, (cid, ctitle) in enumerate(zip(case_ids, case_titles)):
        c_res = all_results[cid]
        for col_idx, t_val in enumerate(post_times):
            ax = axes[row_idx, col_idx]
            snap = c_res["snapshots"][t_val]
            gamma_field = snap["austenite_map"].T

            ax.imshow(gamma_field, cmap=plt.cm.cividis, origin="lower", extent=[0, 200, 0, 200], vmin=0, vmax=1)
            ax.contour(snap["grain_id"].T, levels=np.unique(snap["grain_id"]), colors="cyan", linewidths=0.6, alpha=0.7)

            ax.set_title(f"{ctitle} at t = {t_val} s\n"
                         f"T = {snap['temp_K']:.1f} K | γ-Austenite = {snap['fg']:.1f}% | Residual δ = {100-snap['fg']:.1f}%",
                         fontsize=10.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{ctitle}\nY Position (µm)", fontsize=11, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("X Position (µm)", fontsize=11)

    plt.suptitle("Microstructural Phase Distribution in SS308 WAAM (1 s, 10 s, 50 s)\n"
                 "Cyan lines: Grain Boundaries | Yellow/Green: Austenite (γ, FCC) Matrix | Dark Blue: Residual Vermicular Delta-Ferrite (δ, BCC)",
                 fontsize=13.5, fontweight="bold", y=0.985)
    plt.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.96, hspace=0.25, wspace=0.20)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


def plot_ss308_50s_quantitative_metrics(all_results, output_path):
    """
    Renders 4-panel quantitative kinetics and transformation metrics across 50 seconds.
    """
    case_ids = ["0khz", "20khz", "30khz"]
    case_titles = ["Baseline (0 kHz)", "Power UVA (20 kHz)", "Peak Resonance (30 kHz)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    ax1, ax2, ax3, ax4 = axes.flatten()

    for cid, ctitle, col in zip(case_ids, case_titles, colors):
        hist = all_results[cid]["history"]
        t = hist["time_s"]

        # 1. Temperature vs Time (log scale)
        ax1.plot(t, hist["temperature_K"], label=ctitle, color=col, lw=2.2)

        # 2. Phase Fraction Evolution
        ax2.plot(t, hist["liquid_pct"], label=f"{ctitle} (Liquid)", color=col, ls="--", lw=1.8)
        ax2.plot(t, hist["austenite_pct"], label=f"{ctitle} (Austenite γ)", color=col, ls="-", lw=2.2)

        # 3. Residual Delta-Ferrite vs Time
        ax3.plot(t, hist["delta_ferrite_pct"], label=ctitle, color=col, lw=2.2)

        # 4. Total Grains vs Time
        ax4.plot(t, hist["grain_count"], label=ctitle, color=col, lw=2.2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Physical Time (seconds) [Log Scale]", fontsize=10.5)
    ax1.set_ylabel("Temperature (K)", fontsize=10.5)
    ax1.axhline(1730.0, ls=":", color="red", label="Liquidus (1730 K)")
    ax1.axhline(1675.0, ls=":", color="blue", label="Solidus (1675 K)")
    ax1.set_title("Thermal Dissipation & Cooling Curve (0 to 50 s)", fontsize=11, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.set_xscale("log")
    ax2.set_xlabel("Physical Time (seconds) [Log Scale]", fontsize=10.5)
    ax2.set_ylabel("Phase Fraction (%)", fontsize=10.5)
    ax2.set_title("Phase Transition: Liquid -> Ferrite -> Austenite (γ)", fontsize=11, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(loc="center right", fontsize=8.0)

    ax3.set_xscale("log")
    ax3.set_xlabel("Physical Time (seconds) [Log Scale]", fontsize=10.5)
    ax3.set_ylabel("Residual Delta-Ferrite (%)", fontsize=10.5)
    ax3.axhline(7.5, ls=":", color="black", label="Equilibrium δ (~7.5%, FN ~7.9)")
    ax3.set_title("Residual Vermicular Delta-Ferrite Depletion", fontsize=11, fontweight="bold")
    ax3.grid(True, ls="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8.5)

    ax4.set_xlabel("Physical Time (seconds)", fontsize=10.5)
    ax4.set_ylabel("Total Nucleated Grains (N)", fontsize=10.5)
    ax4.set_ylim(0, 24)
    ax4.set_title("Grain Multiplication: Acoustic CET Proliferation", fontsize=11, fontweight="bold")
    ax4.grid(True, ls="--", alpha=0.5)
    ax4.legend(loc="center right", fontsize=9.0)

    plt.suptitle("Quantitative 50.0-Second Multi-Physics Evaluation of Stainless Steel 308 (ER308L)\n"
                 "Coupled Micro-Solidification, Acoustic Cavitation Grain Refinement & Solid-State Transformation Kinetics",
                 fontsize=13.5, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


def assemble_ss308_50s_gif(all_results, output_gif_path):
    """
    Assembles a high-resolution 3-column animated GIF comparing Baseline, 20kHz, 30kHz across 50.0 seconds.
    """
    if not HAS_PIL:
        print("[WARNING] PIL not available, skipping SS308 50s GIF assembly.")
        return None

    out = Path(output_gif_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out.parent / "temp_ss308_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    gif_times = [0.04, 0.08, 0.16, 1.0, 10.0, 50.0]
    case_ids = ["0khz", "20khz", "30khz"]
    case_titles = ["Baseline (0 kHz)", "Power UVA (20 kHz)", "Peak Resonance (30 kHz)"]
    cmap_dendrite = plt.cm.inferno
    cmap_phase = plt.cm.magma

    gif_frames = []

    for t_val in gif_times:
        fig_frame, (ax_b, ax_20k, ax_30k) = plt.subplots(1, 3, figsize=(18, 6.5), dpi=120)

        for ax, cid, ctitle in zip([ax_b, ax_20k, ax_30k], case_ids, case_titles):
            snap = all_results[cid]["snapshots"][t_val]
            n_g = all_results[cid]["n_grains"]

            if t_val <= 0.16:
                ax.imshow(snap["phi"].T, cmap=cmap_dendrite, origin="lower", extent=[0, 200, 0, 200], vmin=0, vmax=1)
                sub_label = f"Liquid-Solid Phase (Solid fs = {snap['fs']*100:.1f}%)"
            else:
                ax.imshow(snap["austenite_map"].T, cmap=cmap_phase, origin="lower", extent=[0, 200, 0, 200], vmin=0, vmax=1)
                sub_label = f"Solid-State γ-Austenite ({snap['fg']:.1f}%)"

            ax.set_title(f"{ctitle}\n{sub_label} | Grains = {n_g}", fontsize=11, fontweight="bold")
            ax.set_xlabel("X Position (µm)", fontsize=10)
            ax.set_ylabel("Y Position (µm)", fontsize=10)
            ax.text(8, 178, f"T = {snap['temp_K']:.0f} K",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
                    fontsize=9.5, fontweight="bold")

        plt.suptitle(f"SS308 WAAM Microstructure Evolution | Physical Time: t = {t_val} s\n"
                     f"Comparing Baseline (0 kHz) vs Power Ultrasound (20 kHz & 30 kHz)",
                     fontsize=13.5, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])

        frame_file = temp_dir / f"frame_t_{t_val}s.png"
        plt.savefig(frame_file, dpi=120)
        plt.close(fig_frame)

        pil_img = Image.open(frame_file)
        gif_frames.append(pil_img)

    gif_frames[0].save(
        out,
        save_all=True,
        append_images=gif_frames[1:],
        duration=[1000, 1000, 1500, 1500, 1500, 2500],
        loop=0
    )
    print(f"Saved Animated GIF: {out}")

    # Cleanup temp frame pngs
    for t_val in gif_times:
        p = temp_dir / f"frame_t_{t_val}s.png"
        if p.exists():
            p.unlink()
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return str(out)


# =============================================================================
# 12. 10-FREQUENCY PARAMETRIC STUDY DELIVERABLES
# Reproduces:
#   - pfm_10freq_morphology_gallery.png (10 panels)
#   - pfm_10freq_solidification_kinetics.png
#   - pfm_10freq_grain_statistics.png
# =============================================================================
def plot_10freq_morphology_gallery(results_dict, output_path):
    """
    Renders 10-panel unified morphology gallery (2 rows x 5 columns).
    """
    cases_order = ["case_0khz", "case_1khz", "case_10khz", "case_15khz", "case_20khz",
                   "case_25khz", "case_30khz", "case_35khz", "case_45khz", "case_50khz"]

    fig, axes = plt.subplots(2, 5, figsize=(24, 10.5), dpi=150)
    axes = axes.flatten()
    cmap_grain = plt.cm.inferno

    for idx, cid in enumerate(cases_order):
        if cid not in results_dict:
            continue
        res = results_dict[cid]
        ax = axes[idx]
        phi = res["final_phi"]
        st = res["stats"]
        c_info = res["case_info"]

        ax.imshow(phi.T, cmap=cmap_grain, origin="lower", extent=[0, 200, 0, 200], vmin=0, vmax=1)
        ax.set_title(f"{c_info['freq_str']}\n"
                     f"Grains = {st['grain_count']} | d_mean = {st['d_mean_um']:.1f} µm | T_end = {st['final_temperature_K']:.1f} K",
                     fontsize=10.5, fontweight="bold", pad=5)
        ax.set_xlabel("X Position (µm)", fontsize=9.5)
        ax.set_ylabel("Y Position (µm)", fontsize=9.5)
        ax.text(8, 178, f"{c_info['p_acoustic_mpa']} MPa",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9),
                fontsize=9, fontweight="bold")

    plt.suptitle("UVA-WAAM Phase-Field Microstructure Evolution Across 10 Frequencies (0 kHz to 50 kHz)\n"
                 "Domain: 200 µm x 200 µm | Solidification Duration: 75 ms | Material: SS308",
                 fontsize=14.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


def plot_10freq_solidification_kinetics(results_dict, output_path):
    """
    Renders 4-panel solidification kinetics across 10 frequencies:
      - Solid fraction vs time
      - Dendrite tip velocity vs time
      - Physical temperature evolution vs time
      - SDAS vs frequency
    """
    cases_order = ["case_0khz", "case_1khz", "case_10khz", "case_15khz", "case_20khz",
                   "case_25khz", "case_30khz", "case_35khz", "case_45khz", "case_50khz"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    ax1, ax2, ax3, ax4 = axes.flatten()

    freqs = []
    sdas_vals = []

    for cid, col in zip(cases_order, colors):
        if cid not in results_dict:
            continue
        res = results_dict[cid]
        rec = res["time_records"]
        c_info = res["case_info"]

        t = rec["time_ms"]
        ax1.plot(t, [f * 100.0 for f in rec["solid_fraction"]], label=c_info["freq_str"], color=col, lw=2.0)
        ax2.plot(t, rec["tip_velocity_um_s"], label=c_info["freq_str"], color=col, lw=1.8)
        ax3.plot(t, rec["temperature_K"], label=c_info["freq_str"], color=col, lw=2.0)

        freqs.append(c_info["freq_khz"])
        sdas_vals.append(res["stats"]["sdas_um"])

    ax1.set_xlabel("Solidification Time (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Solid Fraction (%)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Solid Fraction Evolution vs Time", fontsize=12, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)

    ax2.set_xlabel("Solidification Time (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tip Growth Velocity (µm/s)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Dendrite Tip Velocity vs Time", fontsize=12, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, ncol=2)

    ax3.set_xlabel("Solidification Time (ms)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Physical Temperature (K)", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Mushy Zone Temperature Evolution & Acoustic Dissipation", fontsize=12, fontweight="bold")
    ax3.grid(True, ls=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5, ncol=2)

    ax4.plot(freqs, sdas_vals, "bo-", lw=2.2, markersize=6)
    ax4.set_xlabel("UVA Acoustic Frequency (kHz)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Secondary Arm Spacing SDAS (µm)", fontsize=11, fontweight="bold")
    ax4.set_title("(d) Secondary Dendrite Arm Spacing vs Frequency", fontsize=12, fontweight="bold")
    ax4.grid(True, ls=":", alpha=0.6)

    plt.suptitle("Quantitative Solidification Kinetics & Acoustic Heating Across 10 Frequencies (0 to 50 kHz)",
                 fontsize=14, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


def plot_10freq_grain_statistics(results_dict, output_path):
    """
    Renders 4-panel grain statistics and refinement curves vs frequency:
      - Grain count vs frequency
      - Average grain diameter vs frequency (refinement %)
      - Final solid fraction vs frequency
      - Computational runtime vs frequency
    """
    cases_order = ["case_0khz", "case_1khz", "case_10khz", "case_15khz", "case_20khz",
                   "case_25khz", "case_30khz", "case_35khz", "case_45khz", "case_50khz"]

    freqs = []
    grain_counts = []
    d_means = []
    d_stds = []
    final_fs = []
    runtimes = []

    for cid in cases_order:
        if cid not in results_dict:
            continue
        res = results_dict[cid]
        st = res["stats"]
        freqs.append(res["case_info"]["freq_khz"])
        grain_counts.append(st["grain_count"])
        d_means.append(st["d_mean_um"])
        d_stds.append(st["d_std_um"])
        final_fs.append(st["final_solid_fraction"] * 100.0)
        runtimes.append(st["wall_time_s"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (a) Grain Count vs Frequency
    ax1.plot(freqs, grain_counts, "ro-", lw=2.4, markersize=6)
    ax1.set_xlabel("Acoustic Frequency (kHz)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Nucleated Grains (N)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Grain Multiplication & Proliferation via Acoustic Cavitation", fontsize=12, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.6)

    # (b) Average Grain Diameter vs Frequency
    ax2.errorbar(freqs, d_means, yerr=d_stds, fmt="gs-", ecolor="gray", capsize=4, lw=2.4, markersize=6)
    ax2.set_xlabel("Acoustic Frequency (kHz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Grain Diameter (µm)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Quantitative Grain Refinement vs Frequency", fontsize=12, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.6)

    # (c) Final Solid Fraction vs Frequency
    ax3.plot(freqs, final_fs, "mo-", lw=2.4, markersize=6)
    ax3.set_xlabel("Acoustic Frequency (kHz)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Final Solid Fraction (%)", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Solidification Extent After 75 ms", fontsize=12, fontweight="bold")
    ax3.grid(True, ls=":", alpha=0.6)

    # (d) Computational Runtime vs Frequency
    ax4.bar([str(int(f)) + "k" for f in freqs], runtimes, color="#3498DB", edgecolor="black", alpha=0.85)
    ax4.set_xlabel("Acoustic Frequency Case", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Solver Wall Time (seconds)", fontsize=11, fontweight="bold")
    ax4.set_title("(d) Computational Runtime Efficiency (Numba Parallel JIT)", fontsize=12, fontweight="bold")
    ax4.grid(True, ls=":", alpha=0.6)

    plt.suptitle("Quantitative Grain Refinement Statistics & Numerical Performance Across 10 Frequencies",
                 fontsize=14, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# 10. 500x1500 μm DIRECTIONAL SOLIDIFICATION PUBLICATION VISUALIZATION SUITE
# =============================================================================
def generate_500x1500_publication_figures(all_results, out_dir, brain_dir=None):
    """
    Renders the complete 4-figure publication visualization suite for the
    500 um x 1500 um directional solidification study across 5 frequencies:
      - Figure 1: 5-Panel Full 500x1500 um Morphology Comparison
      - Figure 2: Multi-Stage Time Evolution Comparison
      - Figure 3: Quantitative Metrics & CET Kinetics
      - Figure 4: Runtime Difference & Interface Area Analysis
    """
    import shutil
    from scipy.ndimage import sobel
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs = sorted(list(all_results.keys()))
    palette = {0: "#455A64", 1: "#1976D2", 10: "#FB8C00", 20: "#D81B60", 30: "#43A047"}
    domain_w = 500.0
    domain_h = 1500.0

    # ------------------------------------------------------------------
    # FIGURE 1: 5-PANEL FULL 500x1500 um MORPHOLOGY COMPARISON
    # ------------------------------------------------------------------
    print("\n[VISUALIZATION] Rendering Figure 1: 500x1500 um 5-Frequency Morphology Gallery...")
    n_freqs = len(freqs)
    fig, axes = plt.subplots(1, n_freqs, figsize=(4.4 * n_freqs, 13), dpi=150)
    if n_freqs == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FFFFFF")
    orient_cmap = plt.cm.hsv

    for idx, freq in enumerate(freqs):
        ax = axes[idx]
        res = all_results[freq]
        latest_t = max(res["snapshots"].keys())
        snap = res["snapshots"][latest_t]
        phi_f, gid_f, theta_f = snap["phi"], snap["grain_id"], snap["theta"]

        theta_norm = (theta_f + np.pi) / (2.0 * np.pi)
        rgb = orient_cmap(theta_norm.T)[:, :, :3]
        liquid_mask = (phi_f < 0.5).T
        rgb[liquid_mask] = [0.08, 0.08, 0.12]

        gx = sobel(gid_f.astype(float), axis=0)
        gy = sobel(gid_f.astype(float), axis=1)
        gb_mask = ((gx**2 + gy**2) > 0).T
        rgb[gb_mask & ~liquid_mask] = [1.0, 1.0, 1.0]

        ax.imshow(rgb, origin="lower", extent=[0, domain_w, 0, domain_h], aspect="auto", interpolation="nearest")

        fl_y = 45.0
        ax.axhline(fl_y, color="red", linestyle="-", linewidth=1.5, alpha=0.8)
        ax.text(10, fl_y + 15, "Fusion Line", color="red", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="red", alpha=0.7))

        if res.get("cet_position_um") is not None:
            cet_y = res["cet_position_um"] + fl_y
            ax.axhline(cet_y, color="cyan", linestyle="--", linewidth=1.8, alpha=0.9)
            ax.text(10, cet_y + 15, f"CET @ {res['cet_position_um']:.0f} um",
                    color="cyan", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="cyan", alpha=0.8))

        ax.set_title(f"{res['label']}\n"
                     f"Total Grains: {res['total_grains']} (Col: {res['substrate_grains']}, Eq: {res['equiaxed_nucleated']})\n"
                     f"Mean Size: {res['mean_grain_um']:.1f} ± {res['std_grain_um']:.1f} um",
                     fontsize=9.5, fontweight="bold", pad=8)
        ax.set_xlabel("X (um)", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Build Height Y (um) [Solidification Direction]", fontsize=11, fontweight="bold")
        else:
            ax.set_yticklabels([])

        ax.set_xlim(0, domain_w)
        ax.set_ylim(0, domain_h)

    plt.suptitle("UVA-WAAM SS308 (ER308L) Directional Solidification Microstructure (500 um x 1500 um)\n"
                 "Orientation Field Color | White: Grain Boundaries | Red: Fusion Line | Cyan: CET Onset",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.90, bottom=0.06, left=0.06, right=0.98, wspace=0.10)

    fig1_path = out_dir / "pfm_500x1500_5freq_morphology.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {fig1_path}")

    # ------------------------------------------------------------------
    # FIGURE 2: MULTI-STAGE TIME EVOLUTION (0 kHz vs 30 kHz or top 2 freqs)
    # ------------------------------------------------------------------
    f_low = freqs[0]
    f_high = freqs[-1]
    res_low = all_results[f_low]
    res_high = all_results[f_high]
    snap_times = sorted(list(res_low["snapshots"].keys()))

    print(f"\n[VISUALIZATION] Rendering Figure 2: Time Evolution ({f_low} kHz vs {f_high} kHz)...")
    n_cols = len(snap_times)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 11), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    for col_idx, t_s in enumerate(snap_times):
        # Row 0: Baseline
        ax0 = axes[0, col_idx] if n_cols > 1 else axes[0]
        s0 = res_low["snapshots"][t_s]
        phi0, gid0, theta0 = s0["phi"], s0["grain_id"], s0["theta"]
        rgb0 = orient_cmap((theta0.T + np.pi) / (2 * np.pi))[:, :, :3]
        rgb0[(phi0 < 0.5).T] = [0.08, 0.08, 0.12]
        gx0 = sobel(gid0.astype(float), axis=0)
        gy0 = sobel(gid0.astype(float), axis=1)
        rgb0[((gx0**2 + gy0**2) > 0).T & ~(phi0 < 0.5).T] = [1.0, 1.0, 1.0]

        ax0.imshow(rgb0, origin="lower", extent=[0, domain_w, 0, domain_h], aspect="auto")
        ax0.set_title(f"{f_low} kHz | t = {t_s*1000:.0f} ms\nfs = {s0['fs']*100:.1f}% | Grains = {s0['n_grains']}",
                      fontsize=8.5, fontweight="bold")
        if col_idx == 0:
            ax0.set_ylabel(f"{f_low} kHz (Baseline)\nY (um)", fontsize=10, fontweight="bold")
        else:
            ax0.set_yticklabels([])

        # Row 1: High Frequency UVA
        ax1 = axes[1, col_idx] if n_cols > 1 else axes[1]
        s1 = res_high["snapshots"][t_s]
        phi1, gid1, theta1 = s1["phi"], s1["grain_id"], s1["theta"]
        rgb1 = orient_cmap((theta1.T + np.pi) / (2 * np.pi))[:, :, :3]
        rgb1[(phi1 < 0.5).T] = [0.08, 0.08, 0.12]
        gx1 = sobel(gid1.astype(float), axis=0)
        gy1 = sobel(gid1.astype(float), axis=1)
        rgb1[((gx1**2 + gy1**2) > 0).T & ~(phi1 < 0.5).T] = [1.0, 1.0, 1.0]

        ax1.imshow(rgb1, origin="lower", extent=[0, domain_w, 0, domain_h], aspect="auto")
        ax1.set_title(f"{f_high} kHz | t = {t_s*1000:.0f} ms\nfs = {s1['fs']*100:.1f}% | Grains = {s1['n_grains']}",
                      fontsize=8.5, fontweight="bold")
        ax1.set_xlabel("X (um)", fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax1.set_ylabel(f"{f_high} kHz (Peak UVA)\nY (um)", fontsize=10, fontweight="bold")
        else:
            ax1.set_yticklabels([])

    plt.suptitle("Microstructural Evolution Across Solidification Time (500 um x 1500 um Domain)\n"
                 f"Baseline WAAM ({f_low} kHz) vs Peak Ultrasonic-Assisted WAAM ({f_high} kHz)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.90, bottom=0.06, left=0.06, right=0.98, hspace=0.15, wspace=0.10)

    fig2_path = out_dir / "pfm_500x1500_time_evolution.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {fig2_path}")

    # ------------------------------------------------------------------
    # FIGURE 3: QUANTITATIVE METRICS COMPARISON
    # ------------------------------------------------------------------
    print("\n[VISUALIZATION] Rendering Figure 3: Quantitative Metrics & CET Kinetics...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    # (a) CET Position vs Frequency
    ax_a = axes[0, 0]
    cet_vals = [all_results[f].get("cet_position_um", 0.0) or 0.0 for f in freqs]
    ax_a.plot(freqs, cet_vals, "o-", color="#D32F2F", linewidth=2.5, markersize=8)
    for x, y in zip(freqs, cet_vals):
        if y > 0:
            ax_a.annotate(f"{y:.0f} um", (x, y), textcoords="offset points", xytext=(0, 10),
                          ha="center", fontweight="bold", color="#D32F2F")
    ax_a.set_title("(a) Columnar-to-Equiaxed Transition (CET) Height vs Frequency", fontweight="bold", fontsize=11)
    ax_a.set_xlabel("UVA Frequency (kHz)", fontweight="bold")
    ax_a.set_ylabel("CET Onset Height (um above fusion line)", fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # (b) Grain Size (Mean & Std) vs Frequency
    ax_b = axes[0, 1]
    means = [all_results[f]["mean_grain_um"] for f in freqs]
    stds = [all_results[f]["std_grain_um"] for f in freqs]
    bar_colors = [palette.get(f, "#1976D2") for f in freqs]
    bars = ax_b.bar([str(int(f)) + "k" for f in freqs], means, yerr=stds, capsize=6,
                    color=bar_colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar, m in zip(bars, means):
        ax_b.text(bar.get_x() + bar.get_width() / 2.0, m + 4, f"{m:.1f} um",
                  ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax_b.set_title("(b) Mean Equivalent Grain Diameter vs Frequency", fontweight="bold", fontsize=11)
    ax_b.set_xlabel("UVA Frequency Case", fontweight="bold")
    ax_b.set_ylabel("Equivalent Grain Size d_mean (um)", fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.5, axis="y")

    # (c) Columnar vs Equiaxed Zone Fractions
    ax_c = axes[1, 0]
    col_pct = []
    eq_pct = []
    for f in freqs:
        total_h = all_results[f]["columnar_height_um"] + all_results[f]["equiaxed_height_um"]
        c_p = (all_results[f]["columnar_height_um"] / total_h) * 100.0 if total_h > 0 else 100.0
        e_p = 100.0 - c_p
        col_pct.append(c_p)
        eq_pct.append(e_p)

    ax_c.bar([str(int(f)) + "k" for f in freqs], col_pct, width=0.5, label="Columnar Zone %", color="#546E7A", edgecolor="black", alpha=0.85)
    ax_c.bar([str(int(f)) + "k" for f in freqs], eq_pct, width=0.5, bottom=col_pct, label="Equiaxed Zone %", color="#00ACC1", edgecolor="black", alpha=0.85)
    ax_c.set_title("(c) Morphological Zone Fraction (Columnar vs Equiaxed)", fontweight="bold", fontsize=11)
    ax_c.set_xlabel("UVA Frequency Case", fontweight="bold")
    ax_c.set_ylabel("Volume / Height Percentage (%)", fontweight="bold")
    ax_c.legend(loc="upper right")
    ax_c.grid(True, linestyle="--", alpha=0.5, axis="y")

    # (d) Solid Fraction Kinetics fs(t)
    ax_d = axes[1, 1]
    for f in freqs:
        h = all_results[f]["history"]
        ax_d.plot(h["time_s"], h["solid_pct"], label=f"{f} kHz", color=palette.get(f, "#333333"), linewidth=2.0)
    ax_d.set_title("(d) Solid Fraction Kinetics fs(t)", fontweight="bold", fontsize=11)
    ax_d.set_xlabel("Physical Solidification Time (s)", fontweight="bold")
    ax_d.set_ylabel("Solid Fraction fs (%)", fontweight="bold")
    ax_d.legend(loc="lower right")
    ax_d.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Quantitative Solidification Metrics for SS308 UVA-WAAM (500 um x 1500 um)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.25, wspace=0.20)

    fig3_path = out_dir / "pfm_500x1500_quantitative_metrics.png"
    plt.savefig(fig3_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {fig3_path}")

    # ------------------------------------------------------------------
    # FIGURE 4: RUNTIME DIFFERENCE ANALYSIS
    # ------------------------------------------------------------------
    print("\n[VISUALIZATION] Rendering Figure 4: Runtime Difference Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    # (a) Runtime vs Frequency
    ax0 = axes[0]
    times_s = [all_results[f]["compute_time_s"] for f in freqs]
    bars = ax0.bar([f"{int(f)} kHz" for f in freqs], times_s, color=bar_colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar, t in zip(bars, times_s):
        ax0.text(bar.get_x() + bar.get_width() / 2.0, t + 0.1, f"{t:.2f} s",
                 ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    ax0.set_title("Total Compute Run Time per Simulation", fontweight="bold", fontsize=11)
    ax0.set_xlabel("Frequency Case", fontweight="bold")
    ax0.set_ylabel("Execution Time (seconds)", fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5, axis="y")

    # (b) Interface Cells vs Equiaxed Grains Correlation
    ax1 = axes[1]
    inter_cells = [all_results[f]["mean_interface_cells"] for f in freqs]
    eq_grains = [all_results[f]["equiaxed_nucleated"] for f in freqs]
    ax1.plot(freqs, inter_cells, "s-", color="#E65100", linewidth=2.5, markersize=8, label="Active Interface Cells")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(freqs, eq_grains, "o--", color="#1565C0", linewidth=2.5, markersize=8, label="Equiaxed Nuclei Count")

    ax1.set_title("Root Cause of Runtime Difference: Interface Surface Area", fontweight="bold", fontsize=11)
    ax1.set_xlabel("UVA Frequency (kHz)", fontweight="bold")
    ax1.set_ylabel("Mean Active Interface Cells (grad phi != 0)", color="#E65100", fontweight="bold")
    ax1_twin.set_ylabel("Total Equiaxed Grains Nucleated", color="#1565C0", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Computational Performance and Physical Complexity Breakdown",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.92, wspace=0.30)

    fig4_path = out_dir / "pfm_500x1500_runtime_analysis.png"
    plt.savefig(fig4_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {fig4_path}")

    # Copy to brain dir if provided
    if brain_dir is not None:
        brain_dir = Path(brain_dir)
        brain_dir.mkdir(parents=True, exist_ok=True)
        for p in [fig1_path, fig2_path, fig3_path, fig4_path]:
            shutil.copy2(p, brain_dir / p.name)

    return {
        "fig1_morphology": str(fig1_path),
        "fig2_evolution": str(fig2_path),
        "fig3_metrics": str(fig3_path),
        "fig4_runtime": str(fig4_path)
    }
