import sys
import os
import glob
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# Find latest internal.vtu in VTK directory
vtk_base = "/Users/pandeyji/Desktop/WAAM/waamCase/VTK"
vtu_files = sorted(glob.glob(os.path.join(vtk_base, "waamCase_*/internal.vtu")), key=os.path.getmtime)
if not vtu_files:
    print("No VTU files found in", vtk_base)
    sys.exit(1)

vtu_path = vtu_files[-1]
print("Using latest VTU file:", vtu_path)

output_dir = "/Users/pandeyji/.gemini/antigravity-ide/brain/9931f6d8-ca6b-4074-a405-04685e7fa6fe/scratch/"
os.makedirs(output_dir, exist_ok=True)

reader = XMLUnstructuredGridReader(FileName=[vtu_path])
reader.PointArrayStatus = []
reader.CellArrayStatus = ['T', 'alpha.metal', 'fl', 'U', 'Q_arc', 'VOF_source']
c2p = CellDatatoPointData(Input=reader)

# Common render view setup
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1920, 1080]
view.Background = [0.95, 0.95, 0.98] # Clean publication studio background

# --- FIGURE 5: Deposition Morphology & Bead Shape (alpha.metal = 0.5)
contour_bead = Contour(Input=c2p)
contour_bead.ContourBy = ['POINTS', 'alpha.metal']
contour_bead.Isosurfaces = [0.5]

disp_bead = Show(contour_bead, view, 'GeometryRepresentation')
ColorBy(disp_bead, ('POINTS', 'T'))

t_lut = GetColorTransferFunction('T')
t_lut.ApplyPreset('Cool to Warm', True)
t_lut.RescaleTransferFunction(300.0, 2400.0)

t_bar = GetScalarBar(t_lut, view)
t_bar.Title = 'Temperature T [K]'
t_bar.TitleColor = [0, 0, 0]
t_bar.LabelColor = [0, 0, 0]
t_bar.Position = [0.85, 0.25]
t_bar.ScalarBarLength = 0.5

# Focus camera close to the deposited bead
view.CameraPosition = [0.016, -0.012, 0.013]
view.CameraFocalPoint = [0.007, 0.0, 0.0084]
view.CameraViewUp = [-0.35, 0.35, 0.87]
Render(view)
SaveScreenshot(os.path.join(output_dir, "fig5_morphology.png"), view)
Hide(contour_bead, view)

# --- FIGURE 6: Melt Pool Dimensions & Thermal Field (Cross-section at y = 0)
clip_y = Clip(Input=c2p)
clip_y.ClipType = 'Plane'
clip_y.ClipType.Origin = [0.007, 0.0, 0.0]
clip_y.ClipType.Normal = [0.0, 1.0, 0.0]

thresh_metal = Threshold(Input=clip_y)
thresh_metal.Scalars = ['POINTS', 'alpha.metal']
thresh_metal.LowerThreshold = 0.15
thresh_metal.UpperThreshold = 1.05

disp_cut = Show(thresh_metal, view, 'GeometryRepresentation')
ColorBy(disp_cut, ('POINTS', 'T'))

# Solidus / Liquidus line (fl = 0.5)
contour_mush = Contour(Input=thresh_metal)
contour_mush.ContourBy = ['POINTS', 'fl']
contour_mush.Isosurfaces = [0.5]
disp_mush = Show(contour_mush, view, 'GeometryRepresentation')
disp_mush.Representation = 'Wireframe'
disp_mush.AmbientColor = [0.0, 0.0, 0.0]
disp_mush.LineWidth = 4.0

# Tight longitudinal view of the deposited bead & melt pool
view.CameraPosition = [0.007, -0.010, 0.0085]
view.CameraFocalPoint = [0.007, 0.0, 0.0085]
view.CameraViewUp = [0, 0, 1]
Render(view)
SaveScreenshot(os.path.join(output_dir, "fig6_dimensions.png"), view)
Hide(contour_mush, view)
Hide(disp_cut, view)

# --- FIGURE 7: Marangoni Hydrodynamic Velocity Vectors in Melt Pool
thresh_liquid = Threshold(Input=clip_y)
thresh_liquid.Scalars = ['POINTS', 'fl']
thresh_liquid.LowerThreshold = 0.05
thresh_liquid.UpperThreshold = 1.05

glyphs = Glyph(Input=thresh_liquid, GlyphType='Arrow')
glyphs.OrientationArray = ['POINTS', 'U']
glyphs.ScaleArray = ['POINTS', 'U']
glyphs.ScaleFactor = 0.003
glyphs.MaximumNumberOfSamplePoints = 1200

disp_glyphs = Show(glyphs, view, 'GeometryRepresentation')
ColorBy(disp_glyphs, ('POINTS', 'U'))
u_lut = GetColorTransferFunction('U')
u_lut.ApplyPreset('Black-Body Radiation', True)
u_lut.RescaleTransferFunction(0.0, 1.0)

u_bar = GetScalarBar(u_lut, view)
u_bar.Title = 'Velocity |U| [m/s]'
u_bar.TitleColor = [0, 0, 0]
u_bar.LabelColor = [0, 0, 0]
u_bar.Position = [0.05, 0.25]
u_bar.ScalarBarLength = 0.5

# Side-view of Marangoni circulation in liquid pool
view.CameraPosition = [0.007, -0.016, 0.0088]
view.CameraFocalPoint = [0.007, 0.0, 0.0085]
view.CameraViewUp = [0, 0, 1]
Render(view)
SaveScreenshot(os.path.join(output_dir, "fig7_velocity.png"), view)

print("Render complete! Figures saved to", output_dir)
