import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

# Journal typography configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 24,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(18.5, 11.5))
ax.set_xlim(80, 1850)
ax.set_ylim(80, 1080)
ax.invert_yaxis()
ax.axis("off")

# 1. Substrate
sub_top = Polygon([[380, 780], [1280, 780], [1510, 650], [610, 650]], 
                  facecolor="#EDF2F7", edgecolor="#1A202C", lw=1.5)
sub_front = Polygon([[380, 780], [1280, 780], [1280, 820], [380, 820]], 
                    facecolor="#E2E8F0", edgecolor="#1A202C", lw=1.5)
sub_right = Polygon([[1280, 780], [1510, 650], [1510, 690], [1280, 820]], 
                    facecolor="#CBD5E0", edgecolor="#1A202C", lw=1.5)
for patch in [sub_top, sub_front, sub_right]:
    ax.add_patch(patch)

# 2. Wall Domain
wall_front = Polygon([[480, 411], [1380, 411], [1380, 723], [480, 723]], 
                     facecolor="#EBF4FF", edgecolor="#1A202C", lw=1.5)
wall_right = Polygon([[1380, 411], [1410, 394], [1410, 706], [1380, 723]], 
                     facecolor="#D3E2F2", edgecolor="#1A202C", lw=1.5)
wall_top = Polygon([[480, 411], [1380, 411], [1410, 394], [510, 394]], 
                   facecolor="#FEEBC8", edgecolor="#1A202C", lw=1.5)
for patch in [wall_front, wall_right, wall_top]:
    ax.add_patch(patch)

# 3. 25 Layer Division Lines
y_layers = np.linspace(423, 710.5, 24)
for y_val in y_layers:
    ax.plot([480, 1380], [y_val, y_val], color="#718096", lw=0.75, ls=(0, (5, 3)))
    ax.plot([1380, 1410], [y_val, y_val - 17], color="#718096", lw=0.75, ls=(0, (5, 3)))

# 4. Bidirectional Deposition
ax.annotate("", xy=(1200, 398), xytext=(1045, 398),
            arrowprops=dict(arrowstyle="->", color="#C53030", lw=3.0))
ax.text(1120, 380, r"$\vec{v}_{\mathrm{scan}}\ (+X)$", color="#C53030", 
        fontsize=24, fontweight="bold", ha="center")

ax.annotate("", xy=(1045, 428), xytext=(1200, 428),
            arrowprops=dict(arrowstyle="->", color="#718096", lw=2.5, ls="--"))
ax.text(1120, 455, r"$\mathrm{Layer}\ i - 1\ (-X)$", color="#4A5568", fontsize=24, ha="center")

# Turnaround & Dwell
ax.text(1485, 405, "Bidirectional Turnaround", fontsize=24, fontweight="bold", color="#C05621")
ax.text(1485, 435, r"$t_{\mathrm{dwell}} = 20\ \mathrm{s}$", fontsize=24, color="#C05621")

# 5. Lateral Surface Callout (Convection + Radiation, 2-line gap, formula)
ax.annotate("", xy=(480, 560), xytext=(390, 560),
            arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.6))
ax.text(375, 525, "Lateral Faces:", fontsize=24, fontweight="bold", color="#2B6CB0", ha="right")
ax.text(375, 555, "Convection + Radiation", fontsize=24, fontweight="bold", color="#2B6CB0", ha="right")
ax.text(375, 625, r"$-k\nabla T \cdot \vec{n} = q_{\mathrm{conv}} + q_{\mathrm{rad}}$", 
        fontsize=24, color="#2B6CB0", ha="right")

# 6. Top Surface BC
ax.text(680, 245, "Top Surface Boundary Condition:", fontsize=24, fontweight="bold", color="#9B2C2C", ha="center")
ax.text(680, 278, r"$-k\nabla T \cdot \vec{n} = q_{\mathrm{arc}} - q_{\mathrm{conv}} - q_{\mathrm{rad}}$", 
        fontsize=24, color="#9B2C2C", ha="center")

# 7. Dimensions
ax.annotate("", xy=(1270, 880), xytext=(390, 880),
            arrowprops=dict(arrowstyle="<->", color="#2D3748", lw=1.2))
ax.text(830, 865, r"$L_x = 150\ \mathrm{mm}$", fontsize=24, fontweight="bold", ha="center")

ax.text(305, 808, r"$h_z = 5\ \mathrm{mm}$", fontsize=24, fontweight="bold", ha="right")
ax.text(1450, 800, r"$L_y = 150\ \mathrm{mm}$", fontsize=24, fontweight="bold")
ax.text(1470, 540, r"$H_{\mathrm{wall}} \approx 80\ \mathrm{mm}$", fontsize=24, fontweight="bold")
ax.text(1470, 570, "(25 Deposition Layers)", fontsize=24, color="#4A5568")
ax.text(440, 365, r"$w_{\mathrm{bead}}$", fontsize=24, fontweight="bold", color="#2B6CB0", ha="right")

# 8. Substrate Bottom Conduction (Cleaned, no crossing line)
ax.text(830, 955, "Substrate Bottom Heat Extraction:", fontsize=24, fontweight="bold", color="#2D3748", ha="center")
ax.text(830, 990, r"$-k\nabla T \cdot \vec{n} = h_{\mathrm{contact}}(T - T_{\mathrm{table}})$", 
        fontsize=24, color="#2D3748", ha="center")

plt.tight_layout()
# Download/Save outputs
plt.savefig("WAAM_Domain_Schematic.png", format="png", dpi=300, bbox_inches="tight")
plt.savefig("WAAM_Domain_Schematic.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()