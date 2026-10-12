import os
import glob
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

foam_file = "/Users/pandeyji/Desktop/WAAM/waamCase/waamCase.foam"
output_dir = "/Users/pandeyji/.gemini/antigravity-ide/brain/9931f6d8-ca6b-4074-a405-04685e7fa6fe/scratch/"
os.makedirs(output_dir, exist_ok=True)

reader = OpenFOAMReader(registrationName='waamCase.foam', FileName=foam_file)
reader.MeshRegions = ['internalMesh']
reader.CellArrays = ['T', 'alpha.metal', 'fl', 'U']

reader.UpdatePipeline()
time_steps = reader.TimestepValues
if len(time_steps) > 0:
    latest_time = time_steps[-1]
    view = GetActiveViewOrCreate('RenderView')
    view.ViewTime = latest_time
else:
    print("No time steps found!")
    exit(1)

print(f"Rendering for time: {latest_time} s")

c2p = CellDatatoPointData(Input=reader)

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1600, 1000]
view.Background = [0.94, 0.95, 0.97]

thresh_sub = Threshold(Input=c2p)
thresh_sub.Scalars = ['POINTS', 'alpha.metal']
thresh_sub.LowerThreshold = 0.5
thresh_sub.UpperThreshold = 1.05

clip_sub = Clip(Input=thresh_sub)
clip_sub.ClipType = 'Plane'
clip_sub.ClipType.Origin = [0.0125, 0.0, 0.0080]
clip_sub.ClipType.Normal = [0.0, 0.0, 1.0]

disp_sub = Show(clip_sub, view, 'GeometryRepresentation')
disp_sub.DiffuseColor = [0.45, 0.48, 0.55]
disp_sub.Specular = 0.3
disp_sub.SpecularPower = 20

contour_metal = Contour(Input=c2p)
contour_metal.ContourBy = ['POINTS', 'alpha.metal']
contour_metal.Isosurfaces = [0.5]

disp_bead = Show(contour_metal, view, 'GeometryRepresentation')
ColorBy(disp_bead, ('POINTS', 'T'))

t_lut = GetColorTransferFunction('T')
t_lut.ApplyPreset('Cool to Warm', True)
t_lut.RescaleTransferFunction(300.0, 2400.0)

t_bar = GetScalarBar(t_lut, view)
t_bar.Title = 'Temperature T [K]'
t_bar.TitleColor = [0.1, 0.1, 0.1]
t_bar.LabelColor = [0.1, 0.1, 0.1]
t_bar.Position = [0.85, 0.2]
t_bar.ScalarBarLength = 0.6

# --- VIEW 1: 3D Isometric View
view.CameraPosition = [0.035, -0.025, 0.025]
view.CameraFocalPoint = [0.0125, 0.0, 0.008]
view.CameraViewUp = [-0.35, 0.35, 0.87]
Render(view)
SaveScreenshot(os.path.join(output_dir, "latest_view1_isometric.png"), view)

# --- VIEW 2: Top-Down Plan View (XY Plane)
view.CameraPosition = [0.0125, 0.0, 0.035]
view.CameraFocalPoint = [0.0125, 0.0, 0.008]
view.CameraViewUp = [1.0, 0.0, 0.0]
Render(view)
SaveScreenshot(os.path.join(output_dir, "latest_view2_top.png"), view)

# --- VIEW 3: Side Longitudinal Elevation (XZ Plane)
view.CameraPosition = [0.0125, -0.035, 0.010]
view.CameraFocalPoint = [0.0125, 0.0, 0.010]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render(view)
SaveScreenshot(os.path.join(output_dir, "latest_view3_side.png"), view)

print("Rendered 3 views successfully!")
