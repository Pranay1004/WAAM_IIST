#!/Applications/ParaView-5.13.3.app/Contents/bin/pvpython
# -*- coding: utf-8 -*-
"""
Automated ParaView Post-Processing Engine for WAAM Thermal & Cooling Rate Analysis.
Executes headlessly via ParaView Python API (pvpython).
Generates:
  1. 3D Render of Moving Melt Pool & Temperature Contour at Active Arc
  2. 3D Render of Solidification Front Cooling Rate [°C/s]
  3. Spatial Line Plot Over Height (Z) & Cooling Rate Evolution
  4. Time-series Extraction of Peak Transient Cooling Rates
"""

import os
import sys
from paraview.simple import *

# Disable interactive rendering warnings
paraview.simple._DisableFirstRenderCameraReset()

case_dir = "/Users/pandeyji/Desktop/WAAM/WAAM-Framework/waam_output/parametric_sweep_6cases/Case_1_120A_19V_v300"
pvd_file = os.path.join(case_dir, "visual_outputs/waam.pvd")
out_dir = os.path.join(case_dir, "paraview_postprocess")
os.makedirs(out_dir, exist_ok=True)

print(f"Loading WAAM simulation dataset: {pvd_file}")
waam_pvd = PVDReader(registrationName='waam_pvd', FileName=pvd_file)
waam_pvd.PointArrays = ['Temperature', 'TempC', 'Activation', 'Tpeak', 'CoolingRate_C_per_s']

# Get active animation scene and time information
anim_scene = GetAnimationScene()
anim_scene.UpdateAnimationUsingDataTimeSteps()
timesteps = waam_pvd.TimestepValues
print(f"Loaded {len(timesteps)} timesteps from t = {timesteps[0]:.1f} s to {timesteps[-1]:.1f} s")

# Choose an active deposition timestep with peak arc heating (e.g. t = 511.0 s)
target_t = 511.0
closest_t = min(timesteps, key=lambda t: abs(t - target_t))
print(f"Evaluating active arc state at t = {closest_t:.1f} s")
anim_scene.AnimationTime = closest_t

# Create a 3D Render View
render_view = CreateView('RenderView')
render_view.ViewSize = [1600, 1000]
render_view.OrientationAxesVisibility = 1
render_view.Background = [0.12, 0.14, 0.18]  # Professional dark slate

# Setup camera view focused closely on the deposit and arc
render_view.CameraPosition = [0.22, -0.05, 0.08]
render_view.CameraFocalPoint = [0.075, 0.075, 0.045]
render_view.CameraViewUp = [-0.15, 0.25, 0.95]
render_view.CameraParallelScale = 0.06

# -----------------------------------------------------------------------------
# 1. RENDER TEMPERATURE FIELD
# -----------------------------------------------------------------------------
waam_display = Show(waam_pvd, render_view, 'UnstructuredGridRepresentation')
ColorBy(waam_display, ('POINTS', 'Temperature'))
waam_display.RescaleTransferFunctionToDataRange(True, False)
waam_display.SetScalarBarVisibility(render_view, True)

# Custom color map for Temperature (Cool to Warm / Turbo)
temp_LUT = GetColorTransferFunction('Temperature')
temp_LUT.ApplyPreset('Turbo', True)
temp_LUT.RescaleTransferFunction(25.0, 2000.0)

# Format Scalar Bar
temp_bar = GetScalarBar(temp_LUT, render_view)
temp_bar.Title = 'Temperature [°C]'
temp_bar.ComponentTitle = ''
temp_bar.TitleFontSize = 14
temp_bar.LabelFontSize = 12
temp_bar.ScalarBarLength = 0.45
temp_bar.Position = [0.82, 0.50]

Render(render_view)
img_temp = os.path.join(out_dir, "01_paraview_temperature_field_3d.png")
SaveScreenshot(img_temp, render_view, ImageResolution=[1600, 1000])
print(f"Saved: {img_temp}")

# -----------------------------------------------------------------------------
# 2. RENDER SOLIDIFICATION FRONT COOLING RATE FIELD
# -----------------------------------------------------------------------------
# Threshold for mushy & hot deposit zone [300 °C to 2000 °C]
thresh = Threshold(registrationName='MeltPoolThreshold', Input=waam_pvd)
thresh.Scalars = ['POINTS', 'Temperature']
thresh.ThresholdMethod = 'Between'
thresh.LowerThreshold = 300.0
thresh.UpperThreshold = 2862.0

thresh_display = Show(thresh, render_view, 'UnstructuredGridRepresentation')
Hide(waam_pvd, render_view)

ColorBy(thresh_display, ('POINTS', 'CoolingRate_C_per_s'))
thresh_display.RescaleTransferFunctionToDataRange(True, False)

cr_LUT = GetColorTransferFunction('CoolingRate_C_per_s')
cr_LUT.ApplyPreset('Plasma', True)
cr_LUT.RescaleTransferFunction(0.0, 1500.0)

cr_bar = GetScalarBar(cr_LUT, render_view)
cr_bar.Title = 'Cooling Rate [°C/s]'
cr_bar.ComponentTitle = ''
cr_bar.TitleFontSize = 14
cr_bar.LabelFontSize = 12
cr_bar.ScalarBarLength = 0.45
cr_bar.Position = [0.82, 0.50]
temp_bar.Visibility = 0

Render(render_view)
img_cr = os.path.join(out_dir, "02_paraview_cooling_rate_solidification_3d.png")
SaveScreenshot(img_cr, render_view, ImageResolution=[1600, 1000])
print(f"Saved: {img_cr}")

# -----------------------------------------------------------------------------
# 3. EXTRACT 1D PROFILE OVER LINE ALONG Z (Melt Pool Center)
# -----------------------------------------------------------------------------
# Torch is centered around Y = 0.075 m (75 mm)
line_sample = PlotOverLine(registrationName='PlotOverLineZ', Input=waam_pvd)
line_sample.Point1 = [0.075, 0.075, 0.0]
line_sample.Point2 = [0.075, 0.075, 0.085]
line_sample.Resolution = 200

csv_line = os.path.join(out_dir, "paraview_profile_over_z.csv")
SaveData(csv_line, proxy=line_sample, Precision=5)
print(f"Saved: {csv_line}")

print("\nParaView automated Python pipeline finished successfully!")
