## Your equipment

- **Motion platform**: A purpose-built 3-axis Cartesian gantry for WAAM, made by **Vashishtha Research Pvt Ltd**. Travel: X 800mm / Y 800mm / Z 400mm, max 1000mm/min per axis. Steppers, not servos (visible on the rig). Has limit switches, mechanical stops, and an E-stop.
- **Controller**: **Digital Dream DDCS V2.1**, a 3-axis motion controller. It doesn't stream G-code live — you load a `.tap` file from a USB stick and run it in AUTO mode. Standard DDCS behavior: M3/M4 turn an output on, M5 turns it off (normally used for a spindle, repurposed here as the arc trigger).
- **MPG pendant**: a handheld jog wheel for manually moving the axes — not part of the G-code pipeline, just a setup/homing convenience.
- **Welding equipment**: **Ador Champ Pulse 500** (pulsed-MIG/MMA/TIG power source, 500A class) with an **Ador Pulse Feed 40 SFW** wire feeder and water cooling.
- **Critical fact from the machine's own nameplate**: it explicitly states "integration of welding torch ON/OFF with the electronic motion controller for 3D printing purposes" — meaning someone already wired the DDCS's output to trigger the welder. And you confirmed it's actually been used to run a weld job before. So the hardware path is proven; only the exact M-code needs confirming (no old file or person to ask, so I built in a safe way to find it yourself).

## What you asked for

You wanted to build a WAAM slicer + G-code pipeline: take a 3D model, slice it into weld layers, generate toolpaths, and produce G-code your machine can run.

## What I built

A Python pipeline, `/home/claude/waam_slicer` (packaged and sent to you), four files:

1. **`slicer_core.py`** — loads an STL, repairs common mesh problems, slices it into horizontal layers as 2D polygons. Runs real diagnostics (non-manifold edges, disconnected bodies, degenerate faces) and **refuses to proceed with a badly broken mesh** instead of silently producing a hollow or empty result.
2. **`toolpath.py`** — turns each layer's polygons into actual weld bead paths: perimeter loops (offset inward by your bead width) plus zigzag infill for the interior, rotating angle per layer. Falls back to a centerline bead for features too thin for a full bead, rather than dropping them.
3. **`gcode_gen.py`** — writes the actual G-code, now targeted specifically at your DDCS/Champ Pulse 500 setup: `M3`/`M5` arc triggers (configurable), your real 800×800×400mm travel limits enforced before any file is written, 1000mm/min feed cap, no unsafe auto-homing mid-job, and a mandatory dry-run warning printed into every file.
4. **`main.py`** — the command-line tool tying it together, with clear pass/fail output and specific error messages rather than a script that just crashes.

## How we got here

- Started generic (STL → G-code, Marlin-style placeholders).
- You told me the real controller (DDCS V2.1) — I looked up its actual G-code/M-code behavior and rewrote the generator to match instead of guessing.
- I stress-tested the pipeline myself against deliberately broken inputs (thin walls, non-manifold meshes, multi-body parts, a pinched hourglass shape) and fixed every silent-failure case I found — the most dangerous being a broken mesh that used to produce a "successful" but completely empty G-code file.
- You sent photos of the actual machine, which let me replace guesses with facts: real travel limits, real controller model, real welder identity, and proof the arc-trigger integration already exists and works.
- The one thing still unconfirmed is the exact trigger M-code — I didn't guess at that since it's a live-arc safety detail; the README tells you how to confirm it safely (find the old file, or a supervised cold dry-run) before ever running it live.



## Complete summary

### Your equipment (confirmed from photos/nameplates)

| Component                   | What it is                                                                                                                                                                                            |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gantry frame**      | Custom 3-axis Cartesian rig, built by**Vashishtha Research Pvt Ltd** , specifically for WAAM                                                                                                    |
| **Travel**            | X: 800mm, Y: 800mm, Z: 400mm                                                                                                                                                                          |
| **Max speed**         | 1000 mm/min per axis                                                                                                                                                                                  |
| **Motors**            | Steppers (visible on X-rail and Z-column)                                                                                                                                                             |
| **Motion controller** | Digital Dream**DDCS V2.1** , 3-axis                                                                                                                                                             |
| **Pendant**           | MPG (Manual Pulse Generator) — a jog wheel for manual positioning, not part of your G-code logic                                                                                                     |
| **G-code delivery**   | `.tap` files loaded via USB stick, run in AUTO mode (not streamed live)                                                                                                                             |
| **Welder**            | Ador**Champ Pulse 500** — pulsed-MIG/MMA/TIG power source                                                                                                                                      |
| **Wire feeder**       | Ador**Pulse Feed 40 SFW**                                                                                                                                                                       |
| **Torch integration** | Nameplate explicitly confirms:*"Integration of welding torch ON/OFF with the electronic motion controller for 3D printing purposes"* — i.e., this was built to do exactly what you're trying to do |
| **Prior use**         | Someone has already run a successful weld job on this machine — so the hardware path is proven, only the exact M-code driving it is unconfirmed                                                      |

### What I built: a WAAM slicer, from scratch, in 4 modules

```
Your STL file
    ↓
slicer_core.py   → loads mesh, checks it's actually solid/weldable, slices into flat layers
    ↓
toolpath.py      → turns each layer into weld bead paths (outer walls + zigzag fill)
    ↓
gcode_gen.py     → writes G-code your DDCS can actually run
    ↓
main.py          → the command you type to run the whole thing
```

### What each part actually does

* **`slicer_core.py`** — Opens your STL, checks for damage (holes, multi-body, bad geometry), tries to auto-repair it, and cuts it into horizontal layers at your chosen height. If the mesh is too broken to slice safely, it** ****stops and tells you why** instead of silently producing garbage.
* **`toolpath.py`** — For each layer, draws the outer wall(s) as concentric loops sized to your bead width, then fills the inside with a back-and-forth zigzag pattern (rotating angle each layer for strength). If a wall is thinner than one bead, it doesn't just vanish — it falls back to a single centerline bead and warns you.
* **`gcode_gen.py`** — Converts all that into real G-code lines your DDCS understands:** **`G0`/`G1` for movement,** **`G4` for pauses, and** **`M3`/`M5` to trigger your welder's arc. It checks every single move against your machine's real 800×800×400mm limits before writing anything, so it can't generate a file that crashes into a limit switch.
* **`main.py`** — The actual command you run:** **`python main.py yourpart.stl -o output.gcode`, with options for layer height, bead width, speeds, etc.

### The one thing still unconfirmed: the exact trigger code

Your machine's build sheet proves the DDCS can switch the arc on/off — but nobody has confirmed** ***which* M-code (M3? M7?) actually does it on this specific machine, since the old proof-of-concept file is lost. The generator defaults to** ** **M3 (on) / M5 (off)** , the standard choice for this controller, but the README tells you exactly how to verify it safely (check for an old file on the USB stick first; if not, a cold dry-run with the welder powered off) before ever running it with a live arc.

### Where things stand

A complete, tested pipeline exists, packaged and downloadable, targeted specifically at your hardware — not generic placeholder code. The only remaining step before a live weld is a 10-minute confirmation of the trigger M-code.
