# WAAM G-code Slicer

STL -> sliced layers -> WAAM toolpaths (perimeters + zigzag infill) -> G-code,
targeted at a **Digital Dream DDCS V2.1** 3-axis motion controller driving an
**Ador Champ Pulse 500** (+ Pulse Feed 40 SFW wire feeder) WAAM torch.

Built to fail loudly rather than silently: a damaged mesh, a feature thinner
than your bead width, or a toolpath that doesn't fit your machine's travel
envelope will stop the run with a specific, actionable error - not produce a
"successful" G-code file that's actually wrong or empty.

## Machine defaults baked into this tool

Taken directly from the nameplate on the machine (Vashishtha Research 3-axis
Cartesian WAAM cell):

| Spec | Value |
|---|---|
| X travel | 800mm |
| Y travel | 800mm |
| Z travel | 400mm |
| Max speed per axis | 1000 mm/min |
| Controller | Digital Dream DDCS V2.1, 3-axis |
| G-code input | `.tap` file via USB drive (not streamed live) |
| Arc trigger | M3 (on) / M5 (off) - standard DDCS spindle-output pair |

**The arc trigger M-codes are a strong default, not a confirmed fact.** The
machine's own build sheet confirms torch ON/OFF is wired to the motion
controller for exactly this purpose, and M3/M5 is the standard, documented
way to drive a DDCS output - but the actual wiring inside the control
cabinet hasn't been traced or tested yet on this specific machine. See
**"Before your first real weld"** below.

## IMPORTANT: Before your first real weld

This machine has run a WAAM job successfully before, so the hardware path
works - you're confirming which G-code commands drive it, not building new
wiring. Do these in order:

1. **Look for an old `.tap` file first.** Check any USB drive normally left
   plugged into the DDCS, and the DDCS's own file browser (Program page). If
   a file from the earlier successful job still exists, open it in the
   DDCS's text/program viewer and look for `M3`, `M4`, `M5`, `M7`, or `M8`
   lines mixed in with the motion lines - whichever pair appears right
   before/after a run of weld moves is your confirmed trigger pair. This is
   more reliable than anything below.

2. **If no old file exists, do a cold dry run.** With the welder's main
   power OFF (or shielding gas and wire feed disconnected so nothing can
   actually strike an arc even if triggered), jog the torch well clear of
   the table, load a generated G-code file, and run it in AUTO mode while
   watching the DDCS's output indicators (or listening/watching for the
   relay click in the control cabinet). Confirm the output activates
   exactly when the file's `M3` lines run and deactivates on `M5`. If
   nothing visibly happens, the trigger code may be different (try M7/M8)
   or the wiring may need tracing by someone comfortable opening the
   control cabinet.

3. **Only once (1) or (2) confirms the trigger path**, run a real bead on
   scrap material, then a real test part, before trusting this for
   anything you care about.

If you find the actual M-codes differ from the M3/M5 default, pass
`--arc-on-mcode` / `--arc-off-mcode` (see Usage below) - nothing else in the
pipeline needs to change.

## Setup (macOS, e.g. M4 MacBook Air)

```bash
python3 -m venv venv
source venv/bin/activate
pip install trimesh numpy shapely manifold3d rtree
```

`manifold3d` and `rtree` are needed for mesh boolean ops and fast section
queries respectively - without them some meshes will hard-crash instead of
slicing. Add `matplotlib` too if you want to preview toolpaths.

## Supported input formats

| Format | How it's handled |
|---|---|
| STL, OBJ | Loaded directly (unitless - assumed mm) |
| 3MF | Loaded directly, incl. multi-object files (sliced as one multi-part job). Non-mm declared units (inch/cm/...) are converted to mm with a logged warning |
| STEP / IGES (`.step` `.stp` `.iges` `.igs`) | Tessellated to STL via headless `gmsh` (`pip install gmsh`, bundles OpenCASCADE), cached as `<name>_waam.stl` beside the source and reused while newer. `--step-mesh-size` caps triangle size (default 5mm - keep at/below bead width) |

## Usage

Machine bounds (`--x-min/max`, etc.) already default to this machine's real
travel (X/Y 800mm, Z 400mm), so you only need to override them if your
part's work-coordinate position needs different limits:

```bash
python main.py part.stl \
    --layer-height 2.0 \
    --bead-width 5.0 \
    --perimeters 2 \
    --travel-speed 800 \
    --weld-speed 300
```

Output defaults to `geometry/<inputname>.gcode` (sibling to the mesh file).
Override with `-o`:

```bash
python main.py part.stl -o /path/to/output.gcode
```

Position oversized or off-origin parts without touching CAD:

```bash
python main.py part.stl --scale 0.3 --auto-place --rotate-z 90
```

`--scale` shrinks uniformly, `--rotate-x/y/z` spin about the part centroid,
`--auto-place` centers XY and rests Z on the envelope floor. Still refuses if
the result doesn't fit.

## GUI

```bash
python gui.py
```

Attach a file, set parameters, press Slice. Tabs: Log/Report, **Orient & View**
(live 3D model with rotation sliders, manual shift, scale, and a FITS/TOO BIG
indicator against the machine envelope - drag to orbit, scroll to zoom; what
you see is what slices), 2D layer preview, and 3D kinematic twin replay.
Model preview loads in the background (large files stay interactive) and
draws a decimated subset - slicing always uses the full mesh. Closing the
window or Ctrl+C shuts down cleanly mid-draw.

The GUI includes process parameter fields: arc voltage (V), arc current (A),
wire efficiency η (0.7-0.9 typical for GMAW), wire diameter (mm), and material
selector (steel/aluminium/titanium). These feed into the job statistics
computation (heat input, mass deposition, etc.).

Notes: attaching a *different* file resets scale/orientation to defaults (a
stale scale from a previous part is the easiest way to mis-slice); every run
auto-clears the previous previews, and **Clear results** wipes them manually.
**Theme: light/dark** toggles the whole window including the 3D views.

Override the arc trigger M-codes once you've confirmed them (see "Before
your first real weld" above) if they differ from the M3/M5 default:

```bash
python main.py part.stl \
    --arc-on-mcode M7 --arc-off-mcode M9
```

**If your part's work zero isn't near machine (0,0,0)**, override the bounds
explicitly to match where the part actually sits in machine coordinates:

```bash
python main.py part.stl \
    --x-min -50 --x-max 50 --y-min -50 --y-max 50 --z-min 0 --z-max 100
```

If a part is positioned outside the real travel envelope, the tool refuses
to generate G-code and tells you exactly which points are out of bounds,
rather than letting the machine crash into a limit switch.

### Key WAAM-specific parameters

- `--layer-height`: typically 1.5-4mm for WAAM (much larger than FDM)
- `--bead-width`: typically 4-8mm depending on arc current/voltage/travel speed
- `--perimeters`: number of concentric wall beads before switching to infill -
  automatically reduces per-region if the part is too thin to fit the
  requested count (see warnings)
- `--infill-style`: hatching strategy - `zigzag` (one connected raster per
  region, single arc strike), `lines` (each scanline its own bead and
  arc start/stop - better around holes, more craters), `contour`
  (perimeters only, hollow - the classic WAAM thin-wall mode), `concentric`
  (inward offset loops following the part shape - even heat, good WAAM
  default for blocks), `grid` / `triangles` (2-3 crossing raster sets per
  layer - stiffer but double/triple heat input, watch interpass temp).
  Honeycomb-style sparse infills are deliberately excluded: a 5mm bead makes
  the cells enormous and every wall costs an arc strike.
- `--min-feature-area`: regions smaller than this (mm²) are skipped with a
  warning rather than fabricated; default is `0.5 × bead_width²`
- `--weld-speed` / `--travel-speed`: deposition (arc on) vs rapid (arc off)
  speed in mm/min (both unchecked against machine max - user responsibility)
- `--interlayer-dwell`: pause in seconds between layers for interpass cooling
  (0 = disabled; real WAAM often needs this to avoid heat buildup and slumping)
- `--lead-in-out`: mm of lead-in/lead-out offset on closed loops, staggers the
  seam position instead of stacking start/stop craters at the same XY every layer
- `--arc-on-mcode` / `--arc-off-mcode`: M-codes that trigger the arc (default
  M3/M5). **Confirm these against this machine's actual wiring before a real
  run** - see "Before your first real weld" above.
- `--wire-feed-speed`: wire feed speed in m/min (optional, comment-only, for logging)
- `--voltage`: arc voltage in V (default 28.0) - used for heat input calculation
- `--current`: arc current in A (default 180.0) - used for heat input calculation
- `--wire-efficiency`: wire feed efficiency η (default 0.8, typical 0.7-0.9 for GMAW)
- `--wire-diameter`: wire diameter in mm (default 1.2) - used for volume flow calc
- `--material`: wire material (steel/aluminium/titanium) - sets density, specific heat, thermal conductivity
- `--allow-damaged-mesh`: proceed even if the mesh is still non-watertight
  after auto-repair (off by default - damaged meshes usually mean missing
  material in the output)
- `--skip-bounds-check`: bypass machine travel limit validation (only for
  simulation/preview use, never for a real run)

## Output G-code

Marlin-style G0/G1/G28/G4 motion, with actual M-codes for arc control:

```
; === JOB STATISTICS ===
; Layers:              22   Layer height: 4.11mm   Bead width: 5.0mm
; Total beads:         65
; --- Time Breakdown ---
; Weld time:           10m 36s   (3180 mm path)
; Travel time:         2m 23s   (~954 mm rapids)
; Arc dwell time:      32s   (65 beads × 0.3s on + 0.2s off)
; TOTAL EST. BUILD:    13m 31s
; --- Wire / Material Consumption ---
; Wire from spool:     72.2 m   (641 g)
; Deposited volume:    65350 mm³   (65.4 cm³)
; Mass deposition:     1.936 kg/hr   (DR = η·Q·ρ)
; --- Heat Input ---
; Arc power:           P = V·I = 4840 W
; Net heat power:      P_net = η·V·I = 3872 W
; Linear heat input:   HI = P_net/V_torch = 12.9 J/mm
; Mass feed/length:    MFL = ...
;
; === LAYER PROCESS TABLE ===
; Layer | Z(mm)  | Weld(s) | Beads | Dist(mm) | HI(J/mm) | Est.T(°C) | Cool(s)
;     0 |    2.1 |     0.0 |     0 |      0.0 |     12.9 |       --- |   0.0
;     1 |    6.2 |    21.8 |     6 |    109.2 |     12.9 |       --- |   0.0
...
G4 P300   ; dwell for arc stabilization
M3        ; ARC_ON - starts torch/arc
G1 X.. Y.. F150
...
G4 P200   ; dwell to fill crater before arc off
M5        ; ARC_OFF - stops torch/arc
...
M2        ; program end (DDCS: stops AUTO mode)
```

Any toolpath warnings (thin features, adapted perimeter counts, skipped
regions) are written directly into the G-code as comments next to the
affected layer, so they travel with the file even if someone runs it without
seeing the console output.

## Job Statistics

The slicer computes and emits job statistics in the G-code header and CLI
output, cross-validated against the WAAM Process Calculator (AE645 course):

| Metric | Formula | Units |
|---|---|---|
| Arc power | P = V × I | W |
| Net heat power | P_net = η × V × I | W |
| Linear heat input | HI = P_net / V_torch | J/mm |
| Volume flow rate | Q = A_wire × V_wire | mm³/min |
| Mass deposition rate | DR = Q × ρ × 60 | kg/hr |
| Mass feed per length | MFL = mdot / V_torch | kg/mm |
| Bead width (calc) | w = 3π × d² × V_wire / (8 × h × V_torch) | mm |

Where: V = arc voltage (V), I = arc current (A), η = wire efficiency,
V_torch = travel speed (mm/min), A_wire = wire cross-section area (mm²),
V_wire = wire feed speed (mm/min), ρ = wire density (kg/m³), h = layer height (mm).

## Architecture

- `slicer_core.py` - loads/repairs mesh (trimesh), runs concrete diagnostics
  (non-manifold edge count, body count, degenerate/duplicate faces) via
  `inspect_mesh()`, slices at Z heights into shapely polygons. Raises
  `MeshValidationError` on unrecoverable input and `SlicingError` if slicing
  produces suspiciously few layers - both stop the pipeline before an empty
  or wrong result reaches G-code.
- `toolpath.py` - offsets contours inward by bead width for perimeters (with
  adaptive perimeter count and a centerline-bead fallback for
  thinner-than-one-bead features), generates rotating zigzag infill.
  Collects per-layer `warnings` on every `LayerToolpath` instead of dropping
  geometry silently.
- `gcode_gen.py` - validates params (positive finite speeds/dwells) and every
  emitted coordinate (finite, within `MachineLimits`) before writing any
  G-code; adds interlayer dwell and lead-in/out; refuses to write an empty
  file.
- `main.py` - CLI wiring all of the above together, printing mesh
  diagnostics and toolpath warnings before writing output, exiting non-zero
  with a specific message on any of the error classes above.

## File layout

```
waam_lab/
  geometry/          ← STL/STEP/3MF files live here, and G-code outputs here by default
    Main.stl
    j2.stl
    rocket_v2.stl
  waam_slicer/
    main.py          ← CLI entry point
    gui.py           ← Tkinter desktop GUI
    slicer_core.py   ← mesh load/repair/diagnostics
    toolpath.py      ← 6 infill styles (zigzag, lines, contour, concentric, grid, triangles)
    gcode_gen.py     ← G-code generation, machine limits, job statistics
```

Input files can be anywhere, but output defaults to `geometry/<inputname>.gcode`.
Override with `-o` if you want a different output path.

## Known limitations / next steps

- No thermal *simulation* - `--interlayer-dwell` is a fixed time, not a
  temperature-gated wait. If you get a thermocouple/pyrometer reading into
  the loop later, that's a natural next step (swap the fixed dwell for a
  poll-until-below-threshold hook).
- Bead ordering within a layer isn't travel-optimized (nearest-neighbor or
  similar) - beads are visited in the order shapely's buffer/intersection
  ops happen to return them. Fine correctness-wise (z-hop always applies
  between beads) but not time-optimal.
- The centerline fallback for thin features approximates the medial axis
  with a small inward buffer rather than a true skeleton - fine for roughly
  uniform-thickness thin walls, will be less accurate on a thin feature with
  varying width along its length.
- No G-code simulation/visualization built in - import the output into a
  viewer like CAMotics or ncviewer, or use the toolpath preview approach
  from the project's development (matplotlib plotting `LayerToolpath.beads`
  directly) to sanity-check before running on hardware.
- `MachineLimits` bounds check is axis-aligned bounding-box only - it
  doesn't model any non-rectangular work envelope, fixture obstructions, or
  torch/gantry collision geometry beyond the raw X/Y/Z travel limits.
