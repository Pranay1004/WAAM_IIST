import sys
from paraview.simple import *

# Disable automatic camera reset
paraview.simple._DisableFirstRenderCameraReset()

# 1. Load the internal mesh from the 0.3s solution
vtu_path = "/Users/pandeyji/Desktop/WAAM/waamCase/VTK/waamCase_304/internal.vtu"
reader = XMLUnstructuredGridReader(FileName=[vtu_path])
reader.PointArrayStatus = []
reader.CellArrayStatus = ['T', 'alpha.metal', 'fl', 'U', 'Q_arc']

# 2. Convert cell data to point data for smooth rendering
c2p = CellDatatoPointData(Input=reader)

# 3. Create active render view
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1600, 900]
view.Background = [0.08, 0.09, 0.11] # Dark elegant background

# 4. Extract free surface of metal (alpha.metal = 0.5)
contour_surf = Contour(Input=c2p)
contour_surf.ContourBy = ['POINTS', 'alpha.metal']
contour_surf.Isosurfaces = [0.5]

# Show the bead surface colored by Temperature
disp_surf = Show(contour_surf, view, 'GeometryRepresentation')
disp_surf.ColorArrayName = ['POINTS', 'T']
t_lut = GetColorTransferFunction('T')
t_lut.ApplyPreset('Cool to Warm (Extended)', True)
t_lut.RescaleTransferFunction(300.0, 2500.0)

# Add color bar
t_bar = GetScalarBar(t_lut, view)
t_bar.Title = 'Temperature [K]'
t_bar.ComponentTitle = ''
t_bar.TitleColor = [0.9, 0.9, 0.9]
t_bar.LabelColor = [0.8, 0.8, 0.8]
t_bar.Position = [0.82, 0.25]
t_bar.ScalarBarLength = 0.5

# 5. Extract molten pool (fl = 0.5) to show melting boundary
contour_melt = Contour(Input=c2p)
contour_melt.ContourBy = ['POINTS', 'fl']
contour_melt.Isosurfaces = [0.5]
disp_melt = Show(contour_melt, view, 'GeometryRepresentation')
disp_melt.Representation = 'Wireframe'
disp_melt.AmbientColor = [1.0, 0.2, 0.1]
disp_melt.LineWidth = 2.0

# 6. Outline of the domain
outline = Outline(Input=reader)
disp_outline = Show(outline, view, 'GeometryRepresentation')
disp_outline.AmbientColor = [0.4, 0.4, 0.5]

# 7. Camera setup (Isometric 3D view of the weld bead)
view.CameraPosition = [0.14, -0.07, 0.05]
view.CameraFocalPoint = [0.04, 0.0, 0.01]
view.CameraViewUp = [-0.2, 0.4, 0.9]
view.CameraParallelProjection = 0

# 8. Render and save
Render(view)
output_img = "/Users/pandeyji/.gemini/antigravity-ide/brain/9931f6d8-ca6b-4074-a405-04685e7fa6fe/scratch/waam_openfoam_physics.png"
SaveScreenshot(output_img, view, ImageResolution=[1600, 900])
print(f"Rendered WAAM OpenFOAM physics result to {output_img}")
