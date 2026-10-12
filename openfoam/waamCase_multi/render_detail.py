import sys
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

vtu_path = "/Users/pandeyji/Desktop/WAAM/waamCase/VTK/waamCase_459/internal.vtu"
reader = XMLUnstructuredGridReader(FileName=[vtu_path])
reader.PointArrayStatus = []
reader.CellArrayStatus = ['T', 'alpha.metal', 'fl', 'U', 'Q_arc']

c2p = CellDatatoPointData(Input=reader)

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1920, 1080]
view.Background = [0.07, 0.08, 0.10] # Clean slate dark theme

# 1. 3D Metal Bead Surface (alpha.metal = 0.5)
contour_bead = Contour(Input=c2p)
contour_bead.ContourBy = ['POINTS', 'alpha.metal']
contour_bead.Isosurfaces = [0.5]

disp_bead = Show(contour_bead, view, 'GeometryRepresentation')
disp_bead.ColorArrayName = ['POINTS', 'T']
t_lut = GetColorTransferFunction('T')
t_lut.ApplyPreset('Warm to Cool', True) # Red-hot melt pool to cold substrate
t_lut.InvertTransferFunction()
t_lut.RescaleTransferFunction(300.0, 2600.0)
disp_bead.Opacity = 0.95

# 2. Longitudinal Cut-plane at Y = 0 to show Melt Pool Penetration & Thermal Field
clip_y = Clip(Input=c2p)
clip_y.ClipType = 'Plane'
clip_y.ClipType.Origin = [0.04, 0.0, 0.012]
clip_y.ClipType.Normal = [0.0, 1.0, 0.0]

# Threshold to only show metallic zone (alpha.metal > 0.1) on cut plane
thresh_metal = Threshold(Input=clip_y)
thresh_metal.Scalars = ['POINTS', 'alpha.metal']
thresh_metal.LowerThreshold = 0.1
thresh_metal.UpperThreshold = 1.05

disp_cut = Show(thresh_metal, view, 'GeometryRepresentation')
disp_cut.ColorArrayName = ['POINTS', 'T']
disp_cut.SetScalarBarVisibility(view, True)

# 3. Molten Pool Boundary (Solidus / Liquidus line fl = 0.5)
contour_mush = Contour(Input=thresh_metal)
contour_mush.ContourBy = ['POINTS', 'fl']
contour_mush.Isosurfaces = [0.5]
disp_mush = Show(contour_mush, view, 'GeometryRepresentation')
disp_mush.Representation = 'Wireframe'
disp_mush.AmbientColor = [0.0, 1.0, 0.4] # Vibrant green solidus line
disp_mush.LineWidth = 3.5

# 4. Velocity Vectors (Marangoni Circulatory Loops in Melt Pool)
# Filter cells where metal is liquid (fl > 0.1)
thresh_liquid = Threshold(Input=clip_y)
thresh_liquid.Scalars = ['POINTS', 'fl']
thresh_liquid.LowerThreshold = 0.2
thresh_liquid.UpperThreshold = 1.0

glyphs = Glyph(Input=thresh_liquid, GlyphType='Arrow')
glyphs.OrientationArray = ['POINTS', 'U']
glyphs.ScaleArray = ['POINTS', 'U']
glyphs.ScaleFactor = 0.003
glyphs.MaximumNumberOfSamplePoints = 1500

disp_glyphs = Show(glyphs, view, 'GeometryRepresentation')
disp_glyphs.ColorArrayName = ['POINTS', 'U']
u_lut = GetColorTransferFunction('U')
u_lut.ApplyPreset('Plasma (matplotlib)', True)
u_lut.RescaleTransferFunction(0.0, 0.8)

# Colorbars
t_bar = GetScalarBar(t_lut, view)
t_bar.Title = 'Temperature [K]'
t_bar.ComponentTitle = ''
t_bar.TitleColor = [0.95, 0.95, 0.95]
t_bar.LabelColor = [0.85, 0.85, 0.85]
t_bar.Position = [0.85, 0.52]
t_bar.ScalarBarLength = 0.42

u_bar = GetScalarBar(u_lut, view)
u_bar.Title = 'Marangoni Velocity [m/s]'
u_bar.ComponentTitle = ''
u_bar.TitleColor = [0.95, 0.95, 0.95]
u_bar.LabelColor = [0.85, 0.85, 0.85]
u_bar.Position = [0.85, 0.06]
u_bar.ScalarBarLength = 0.42

# 5. Domain Outline
outline = Outline(Input=reader)
disp_outline = Show(outline, view, 'GeometryRepresentation')
disp_outline.AmbientColor = [0.35, 0.38, 0.45]

# 6. Focused Close-up Camera on Molten Pool and Bead
view.CameraPosition = [0.06, -0.045, 0.028]
view.CameraFocalPoint = [0.024, 0.0, 0.009]
view.CameraViewUp = [-0.3, 0.4, 0.86]
view.CameraParallelProjection = 0

Render(view)
output_img = "/Users/pandeyji/.gemini/antigravity-ide/brain/9931f6d8-ca6b-4074-a405-04685e7fa6fe/scratch/waam_openfoam_physics_detail.png"
SaveScreenshot(output_img, view, ImageResolution=[1920, 1080])
print(f"Rendered detailed WAAM OpenFOAM cut-section to {output_img}")
