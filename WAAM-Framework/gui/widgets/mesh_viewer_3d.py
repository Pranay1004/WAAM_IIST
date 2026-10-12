"""
Scientific 3D Finite Element Mesh & Geometry Viewport for WAAM Simulations

Provides advanced 3D rendering pipelines with PyVista as the primary default:
1. Embedded PyVista Viewport Tab (DEFAULT):
   - Off-screen hardware GPU renderer embedded directly inside the GUI tab.
   - Complete workpiece visualization: substrate plate, multi-layer bead wall,
     active Goldak double-ellipsoid heat source, torch vector, and toolpath.
   - Interactive mouse drag orbit and wheel zoom controls.
   - Quick orientation preset buttons (Isometric, Top, Side, Front, Reset).
2. Native Hardware GPU OpenGL/Metal Window via PyVista / VTK:
   - Full 60 FPS interactive PyVista window with full VTK controls.
"""

import os
import sys
import json
import math
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np

from PIL import Image, ImageTk

try:
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Viewport light/dark palettes so the embedded studio follows the app theme
# instead of staying always-dark. Keys mirror gui/app.py THEMES intent.
VIEWPORT_THEMES = {
    "light": {
        "ribbon_bg": "#e2e8f0",
        "title_fg": "#0d6efd",
        "cambar_bg": "#f1f5f9",
        "camdim_fg": "#475569",
        "preset_bg": "#ffffff",
        "preset_fg": "#0f172a",
        "reset_bg": "#e2e8f0",
        "reset_fg": "#0d6efd",
        "canvas_bg": "#f8fafc",
        "inspector_bg": "#ffffff",
        "grid_bg": "#f1f5f9",
        "grid_label_fg": "#475569",
        "grid_value_fg": "#0f172a",
        "hs_title_fg": "#b45309",
        "diag_title_fg": "#0d6efd",
        "diag_face": "#ffffff",
        "diag_tick": "#475569",
        "diag_spine": "#cbd5e1",
        "tp_title_fg": "#047857",
        "tp_bg": "#f1f5f9",
        "tp_fg": "#334155",
        "stats_fg": "#475569",
    },
    "dark": {
        "ribbon_bg": "#0f172a",
        "title_fg": "#38bdf8",
        "cambar_bg": "#1e293b",
        "camdim_fg": "#94a3b8",
        "preset_bg": "#334155",
        "preset_fg": "#f8fafc",
        "reset_bg": "#0f172a",
        "reset_fg": "#38bdf8",
        "canvas_bg": "#090d16",
        "inspector_bg": "#0f172a",
        "grid_bg": "#1e293b",
        "grid_label_fg": "#94a3b8",
        "grid_value_fg": "#f8fafc",
        "hs_title_fg": "#facc15",
        "diag_title_fg": "#38bdf8",
        "diag_face": "#0b1120",
        "diag_tick": "#94a3b8",
        "diag_spine": "#1e293b",
        "tp_title_fg": "#34d399",
        "tp_bg": "#1e293b",
        "tp_fg": "#cbd5e1",
        "stats_fg": "#94a3b8",
    },
}
try:
    from gui.fonts import (
        FONT_SMALL, FONT_SMALL_BOLD, FONT_MONO, FONT_TINY,
        FONT_SECTION, FONT_STATUS,
        UI_FONT_FAMILY, MONO_FONT_FAMILY, OS_TYPE,
    )
except ImportError:
    try:
        from ..fonts import (
            FONT_SMALL, FONT_SMALL_BOLD, FONT_MONO, FONT_TINY,
            FONT_SECTION, FONT_STATUS,
            UI_FONT_FAMILY, MONO_FONT_FAMILY, OS_TYPE,
        )
    except ImportError:
        UI_FONT_FAMILY = "Helvetica"
        MONO_FONT_FAMILY = "Menlo"
        OS_TYPE = "other"
        FONT_SMALL = (UI_FONT_FAMILY, 9)
        FONT_SMALL_BOLD = (UI_FONT_FAMILY, 9, "bold")
        FONT_MONO = (MONO_FONT_FAMILY, 9)
        FONT_TINY = (MONO_FONT_FAMILY, 8)
        FONT_SECTION = (UI_FONT_FAMILY, 10, "bold")
        FONT_STATUS = (UI_FONT_FAMILY, 11, "bold")


def _extract_geometry_params(params: dict):
    """Safely extracts geometry in millimeters and computes mesh statistics.

    Unit contract: canonical params are SI (m, m/s) per EDPBuilder.default_params().
    This helper accepts SI or mm for robustness: values >1.0 are treated as mm,
    values <=1.0 as m and scaled x1000. Callers should prefer SI; mm input is a
    legacy fallback. Sub-mm (<1mm) bead inputs must be passed in SI (e.g. 0.0005
    for 0.5mm), not as bare 0.5.
    """
    def get_dim_mm(key, default_val):
        val = float(params.get(key, default_val))
        return val * 1000.0 if val < 1.0 else val

    lx = get_dim_mm("lx", 150.0)           # mm
    ly_base = get_dim_mm("ly_base", 150.0) # mm
    hz_base = get_dim_mm("hz_base", 5.0)   # mm (substrate plate thickness)
    ty_dep = get_dim_mm("ty_dep", 8.0)     # mm
    h_bead_first = get_dim_mm("h_bead_first", get_dim_mm("h_bead", 4.0))     # mm
    remelt_ratio = float(params.get("remelt_ratio", 0.80))
    h_bead_subseq = float(params.get("h_bead_subseq", h_bead_first * remelt_ratio))
    if h_bead_subseq < 0.1:
        h_bead_subseq *= 1000.0
    n_layers = max(int(float(params.get("n_layers", 10))), 1)
    hz_dep = h_bead_first if n_layers <= 1 else (h_bead_first + (n_layers - 1) * h_bead_subseq)             # mm

    y_dep_centre = ly_base / 2.0
    y_dep_left = y_dep_centre - ty_dep / 2.0
    y_dep_right = y_dep_centre + ty_dep / 2.0

    nx_dep = max(int(params.get("nx_dep", 25)), 1)
    ny_dep = max(int(params.get("ny_dep", 3)), 1)
    nz_dep = max(int(params.get("nz_dep", n_layers)), 1)

    nx_base = max(int(params.get("nx_base", 25)), 1)
    ny_base = max(int(params.get("ny_base", 8)), 1)
    nz_base = max(int(params.get("nz_base", 2)), 1)

    hex_base = nx_base * ny_base * nz_base
    hex_dep = nx_dep * ny_dep * nz_dep
    total_hex = hex_base + hex_dep
    total_tets = 6 * total_hex
    nodes_base = (nx_base + 1) * (ny_base + 1) * (nz_base + 1)
    nodes_dep = (nx_dep + 1) * (ny_dep + 1) * (nz_dep + 1)
    total_nodes = nodes_base + nodes_dep

    dx_dep = lx / nx_dep
    dy_dep = ty_dep / ny_dep
    dz_dep = hz_dep / nz_dep
    dx_base = lx / nx_base
    dy_base = ly_base / ny_base
    dz_base = hz_base / nz_base

    # Heat Source Model & Dimensions (mm)
    heat_source = str(params.get("heat_source", "goldak")).lower()
    bG = float(params.get("bG", 7.0e-3))
    a_hs = bG * 1000.0 if bG < 1.0 else bG
    cG = float(params.get("cG", 3.0e-3))
    b_hs = cG * 1000.0 if cG < 1.0 else cG
    af = float(params.get("af", 4.0e-3))
    cf_hs = af * 1000.0 if af < 1.0 else af
    ar = float(params.get("ar", 16.0e-3))
    cr_hs = ar * 1000.0 if ar < 1.0 else ar

    R_beam = float(params.get("R_beam", 5.0e-3))
    R_beam_mm = R_beam * 1000.0 if R_beam < 1.0 else R_beam
    R_spot = float(params.get("R_spot", 3.0e-3))
    R_spot_mm = R_spot * 1000.0 if R_spot < 1.0 else R_spot
    r_top = float(params.get("rt", params.get("r_top", 3.0e-3)))
    rt_mm = r_top * 1000.0 if r_top < 1.0 else r_top
    r_bot = float(params.get("rb", params.get("r_bot", 1.0e-3)))
    rb_mm = r_bot * 1000.0 if r_bot < 1.0 else r_bot
    d_pen = float(params.get("d_pen", params.get("depth_cone", 5.0e-3)))
    d_pen_mm = d_pen * 1000.0 if d_pen < 1.0 else d_pen

    a_band = float(params.get("a_band", 10.0e-3))
    a_band_mm = a_band * 1000.0 if a_band < 1.0 else a_band
    b_band = float(params.get("b_band", 5.0e-3))
    b_band_mm = b_band * 1000.0 if b_band < 1.0 else b_band
    c_band = float(params.get("c_band", 3.0e-3))
    c_band_mm = c_band * 1000.0 if c_band < 1.0 else c_band
    R_activate = float(params.get("R_activate", 10.0e-3))
    R_act_mm = R_activate * 1000.0 if R_activate < 1.0 else R_activate

    I_weld = float(params.get("I_weld", 200.0))
    U_weld = float(params.get("U_weld", 24.0))
    eta_arc = float(params.get("eta_arc", 0.8))
    Q_net = I_weld * U_weld * eta_arc
    v_scan_m_s = float(params.get("v_scan", 10.0e-3))
    v_scan_mm_s = v_scan_m_s * 1000.0 if v_scan_m_s < 1.0 else v_scan_m_s
    heat_input = Q_net / max(v_scan_mm_s, 0.001)

    ff = float(params.get("ff", 0.6))
    fr = float(params.get("fr", 1.4))
    denom_f = math.pi * math.sqrt(math.pi) * a_hs * b_hs * cf_hs
    q0_f = (6.0 * math.sqrt(3.0) * ff * Q_net) / denom_f if denom_f > 0 else 0.0
    denom_r = math.pi * math.sqrt(math.pi) * a_hs * b_hs * cr_hs
    q0_r = (6.0 * math.sqrt(3.0) * fr * Q_net) / denom_r if denom_r > 0 else 0.0

    return {
        "lx": lx, "ly_base": ly_base, "hz_base": hz_base,
        "ty_dep": ty_dep, "h_bead": h_bead_first, "h_bead_first": h_bead_first,
        "remelt_ratio": remelt_ratio, "h_bead_subseq": h_bead_subseq,
        "n_layers": n_layers, "hz_dep": hz_dep,
        "y_dep_centre": y_dep_centre, "y_dep_left": y_dep_left, "y_dep_right": y_dep_right,
        "nx_dep": nx_dep, "ny_dep": ny_dep, "nz_dep": nz_dep,
        "nx_base": nx_base, "ny_base": ny_base, "nz_base": nz_base,
        "total_hex": total_hex, "total_tets": total_tets, "total_nodes": total_nodes,
        "dx_dep": dx_dep, "dy_dep": dy_dep, "dz_dep": dz_dep,
        "dx_base": dx_base, "dy_base": dy_base, "dz_base": dz_base,
        "heat_source": heat_source,
        "a_hs": a_hs, "b_hs": b_hs, "cf_hs": cf_hs, "cr_hs": cr_hs,
        "R_beam": R_beam_mm, "R_spot": R_spot_mm,
        "rt": rt_mm, "rb": rb_mm, "d_pen": d_pen_mm,
        "a_band": a_band_mm, "b_band": b_band_mm, "c_band": c_band_mm,
        "R_activate": R_act_mm,
        "Q_net": Q_net, "heat_input": heat_input, "q0_f": q0_f, "q0_r": q0_r,
        "v_scan": v_scan_mm_s
    }


def load_any_cad_geometry(filepath: str, sub_hz_mm: float = None):
    """Safely loads any CAD geometry (.stl, .step, .stp, .iges, .igs, .ply, .obj).

    Args:
        filepath: CAD file to load.
        sub_hz_mm: substrate plate thickness [mm] the part is placed on top of.
            None -> 5 mm, the framework default (matches the solver's hz_base).
            Callers should pass the solver's hz_base so the preview sits
            exactly where the mesh/solver puts it.

    Returns:
        (verts, faces, info) where verts is (N,3) vertex array, faces is (M,3)
        face index array, info is dict with bbox_mm/format. Returns
        (None, None, None) tuple on failure so callers can unpack safely.
        Kept as tuple (not bare trimesh) to match gui/app.py unpacking.
    """
    if not filepath or not os.path.exists(filepath):
        return None, None, None
    import trimesh
    import numpy as np
    p = str(filepath)
    suffix = os.path.splitext(p)[1].lower()

    def _mesh_to_tuple(mesh, fmt: str):
        try:
            if mesh is None:
                return None, None, None
            # trimesh.load may return Scene; resolve to single mesh
            if hasattr(mesh, "to_mesh"):
                try:
                    mesh = mesh.to_mesh()
                except Exception:
                    pass
            if isinstance(mesh, trimesh.Scene):
                try:
                    combined = trimesh.util.concatenate(tuple(mesh.dump()))
                    mesh = combined
                except Exception:
                    return None, None, None
            verts = np.asarray(getattr(mesh, "vertices", []))
            faces = np.asarray(getattr(mesh, "faces", []))
            if verts.size == 0 or faces.size == 0:
                return None, None, None
            bmin = verts.min(axis=0)
            bmax = verts.max(axis=0)
            spans = bmax - bmin
            v_bbox = float(spans[0] * spans[1] * spans[2])
            orig_bbox = {
                "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
                "xmin": float(bmin[0]), "ymin": float(bmin[1]), "zmin": float(bmin[2]),
                "xmax": float(bmax[0]), "ymax": float(bmax[1]), "zmax": float(bmax[2]),
            }

            # Enclosure constraints: the substrate the part is placed on, plus
            # a 150 mm build height above it. SUB_HZ must match the solver's
            # hz_base, otherwise the preview floats above (or sinks into) the
            # plate the mesh was built on.
            SUB_LX = 150.0
            SUB_LY = 150.0
            SUB_HZ = float(sub_hz_mm) if sub_hz_mm else 5.0
            BUILD_HEIGHT = 150.0
            V_ENCLOSURE = SUB_LX * SUB_LY * BUILD_HEIGHT  # 3,375,000 mm^3
            MAX_OCCUPIED_VOL = 0.75 * V_ENCLOSURE       # 2,531,250 mm^3 (75% max volume)

            scale_applied = 1.0
            # If larger than enclosure or exceeds 75% volume limit, shrink uniformly
            if spans[0] > SUB_LX or spans[1] > SUB_LY or spans[2] > BUILD_HEIGHT or v_bbox > MAX_OCCUPIED_VOL:
                s_vol = (MAX_OCCUPIED_VOL / max(v_bbox, 1e-6)) ** (1.0 / 3.0)
                s_dim = min(SUB_LX / max(spans[0], 1e-6), SUB_LY / max(spans[1], 1e-6), BUILD_HEIGHT / max(spans[2], 1e-6))
                scale_applied = min(s_vol, s_dim)
                verts = verts * scale_applied

            # Auto-place correctly on top of the substrate plate:
            # Centered in X and Y, bottom resting on top of it (Z = SUB_HZ).
            bmin = verts.min(axis=0)
            bmax = verts.max(axis=0)
            center_x = (bmin[0] + bmax[0]) / 2.0
            center_y = (bmin[1] + bmax[1]) / 2.0
            min_z = bmin[2]

            tx = (SUB_LX / 2.0) - center_x
            ty = (SUB_LY / 2.0) - center_y
            tz = SUB_HZ - min_z

            verts = verts.copy()
            verts[:, 0] += tx
            verts[:, 1] += ty
            verts[:, 2] += tz

            bmin = verts.min(axis=0)
            bmax = verts.max(axis=0)
            spans = bmax - bmin
            v_occupied = float(spans[0] * spans[1] * spans[2])
            vol_ratio_pct = (v_occupied / V_ENCLOSURE) * 100.0

            info = {
                "bbox_mm": {
                    "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
                    "xmin": float(bmin[0]), "ymin": float(bmin[1]), "zmin": float(bmin[2]),
                    "xmax": float(bmax[0]), "ymax": float(bmax[1]), "zmax": float(bmax[2]),
                },
                "orig_bbox_mm": orig_bbox,
                "scale_applied": float(scale_applied),
                "volume_occupied_mm3": v_occupied,
                "enclosure_volume_mm3": V_ENCLOSURE,
                "max_allowed_vol_mm3": MAX_OCCUPIED_VOL,
                "vol_ratio_pct": vol_ratio_pct,
                "substrate_mm": {"lx": SUB_LX, "ly": SUB_LY, "hz": SUB_HZ},
                "format": fmt,
                "n_verts": int(verts.shape[0]),
                "n_faces": int(faces.shape[0]),
            }
            return verts, faces, info
        except Exception as e:
            print(f"[CAD Load Error] {e}")
            return None, None, None

    if suffix in ('.stl', '.ply', '.obj'):
        try:
            mesh = trimesh.load(p, force='mesh')
            return _mesh_to_tuple(mesh, suffix.upper().lstrip("."))
        except Exception as e:
            print(f"[CAD Load Error] {e}")
            return None, None, None
    elif suffix in ('.step', '.stp', '.iges', '.igs', '.brep'):
        try:
            cache_dir = "./waam_output"
            os.makedirs(cache_dir, exist_ok=True)
            cached_stl = os.path.join(cache_dir, os.path.basename(p) + ".preview.stl")
            if not os.path.exists(cached_stl) or os.path.getmtime(cached_stl) < os.path.getmtime(p):
                import subprocess
                cad_script = f"""
import gmsh
gmsh.initialize(interruptible=False)
gmsh.option.setNumber('General.Terminal', 0)
gmsh.model.add('cad_preview')
gmsh.option.setNumber('Geometry.OCCAutoFix', 1)
gmsh.model.occ.importShapes({repr(p)})
gmsh.model.occ.synchronize()
gmsh.option.setNumber('Mesh.MeshSizeMin', 1.0)
gmsh.option.setNumber('Mesh.MeshSizeMax', 4.0)
gmsh.model.mesh.generate(2)
gmsh.write({repr(cached_stl)})
gmsh.finalize()
"""
                subprocess.run([sys.executable, "-c", cad_script], check=True, capture_output=True)
            mesh = trimesh.load(cached_stl, force='mesh')
            return _mesh_to_tuple(mesh, suffix.upper().lstrip("."))
        except Exception as e:
            print(f"[CAD Load Error] {e}")
            return None, None, None
    return None, None, None


def _sweep_stale_pyvista_params(max_age_days: float = 7.0):
    """Delete orphaned runner JSON/NPZ files from crashed viewer launches."""
    import tempfile as _tf
    import time as _time
    try:
        cutoff = _time.time() - max_age_days * 86400.0
        for name in os.listdir(_tf.gettempdir()):
            if (name.startswith("waam_pyvista_")
                    and name.endswith((".json", ".npz"))):
                full = os.path.join(_tf.gettempdir(), name)
                try:
                    if os.path.getmtime(full) < cutoff:
                        os.remove(full)
                except OSError:
                    pass
    except OSError:
        pass


def launch_pyvista_gpu_window(params: dict):
    """Launch native 60fps hardware-accelerated OpenGL/Metal 3D window via PyVista/VTK in an isolated process."""
    import tempfile
    import atexit

    # Normalise dimensions to SI metres BEFORE serialising: the runner script
    # used to re-apply the old `< 1.0 means metres` guess to the raw dict,
    # which rescaled genuine sub-mm SI values by x1000 a second time.
    try:
        from mesh.units import dim_to_m
        norm = dict(params)
        for _k in ("lx", "ly_base", "hz_base", "ty_dep", "h_bead",
                   "h_bead_first", "h_bead_subseq"):
            if _k in norm and norm[_k] is not None:
                try:
                    norm[_k] = dim_to_m(norm[_k], _k)
                except ValueError:
                    pass
    except Exception:
        norm = params

    # Package root (the directory that CONTAINS gui/, mesh/, ...). The child
    # is started with `python -c`, whose sys.path[0] is the shell CWD -- not
    # this file's directory -- so the framework root must be injected there.
    fw_root = str(Path(__file__).resolve().parents[2])

    _sweep_stale_pyvista_params()
    tmp_cad_path = None
    # Pre-load the CAD surface in THIS process and hand the child plain
    # verts/faces arrays. The old child-side `from gui.widgets.mesh_viewer_3d
    # import load_any_cad_geometry` failed with "No module named 'gui'"
    # whenever the GUI was launched from outside the framework directory (the
    # `python -c` subprocess inherits the shell CWD, not the package root), so
    # the PyVista window silently fell back to the parametric bead wall and
    # never showed the imported geometry.
    hz_base_mm = None
    try:
        hz_base_mm = float(norm.get("hz_base", 0.005)) * 1000.0
    except (TypeError, ValueError):
        _cad_file = (norm.get("stl_file") or norm.get("mesh_file") or "")
    _cad_exts = ('.stl', '.step', '.stp', '.iges', '.igs', '.ply', '.obj')
    if _cad_file and str(_cad_file).lower().endswith(('.mesh', '.mesh3')):
        _cand_stem = Path(_cad_file).stem
        if _cand_stem.startswith("imported_"):
            _cand_stem = _cand_stem[len("imported_"):]
        for _fld in [Path(_cad_file).parent, Path("./stl"), Path(fw_root) / "stl"]:
            for _ext in _cad_exts:
                _cand = _fld / (_cand_stem + _ext)
                if _cand.exists():
                    _cad_file = str(_cand)
                    break
            if str(_cad_file).lower().endswith(_cad_exts):
                break
    if _cad_file and os.path.exists(str(_cad_file)) and \
            str(_cad_file).lower().endswith(_cad_exts):
        try:
            verts, faces, cad_info = load_any_cad_geometry(
                str(_cad_file), sub_hz_mm=hz_base_mm)

            if verts is not None and faces is not None:
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False,
                                                 prefix="waam_pyvista_") as tf_cad:
                    np.savez_compressed(tf_cad, verts=np.asarray(verts, dtype=np.float64),
                                        faces=np.asarray(faces, dtype=np.int64))
                    tmp_cad_path = tf_cad.name
                atexit.register(lambda p=tmp_cad_path: os.path.exists(p) and os.remove(p))
                norm["viewer_cad_npz"] = tmp_cad_path
                norm["viewer_cad_bbox_mm"] = cad_info.get("bbox_mm")
                norm["viewer_cad_name"] = os.path.basename(str(_cad_file))
        except Exception as _cad_err:
            print("[CAD Runner View Error]", _cad_err)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     prefix="waam_pyvista_") as tf:
        json.dump(norm, tf)
        tmp_params_path = tf.name
    # The child removes this file when its window closes; a crash before that
    # point used to leak one JSON per launch into the temp dir forever.
    atexit.register(lambda p=tmp_params_path: os.path.exists(p) and os.remove(p))

    runner_script = f'''
import sys
import json
import os
import math
import numpy as np

with open({repr(tmp_params_path)}) as f:
    params = json.load(f)

try:
    import pyvista as pv
except ImportError:
    print("[PyVista] Not installed. Install via pip install pyvista")
    sys.exit(0)

plotter = pv.Plotter(title="WAAM 3D Workpiece & Heat Source — Hardware GPU (PyVista / VTK)")
plotter.set_background("#090d16", top="#0f172a")

def _get_dim(key, default_m):
    # params arrive normalised to SI metres (see the parent); display is mm.
    try:
        return float(params.get(key, default_m)) * 1000.0
    except (TypeError, ValueError):
        return float(default_m) * 1000.0

lx = _get_dim("lx", 0.150)
ly_base = _get_dim("ly_base", 0.150)
hz_base = _get_dim("hz_base", 0.005)
build_height = 150.0
ty_dep = _get_dim("ty_dep", 0.008)
h_bead_first = _get_dim("h_bead", 0.004)
if "h_bead_first" in params:
    h_bead_first = _get_dim("h_bead_first", 0.004)
remelt_ratio = float(params.get("remelt_ratio", 0.80))
h_bead_subseq = _get_dim("h_bead_subseq", h_bead_first / 1000.0 * remelt_ratio)
n_layers = max(int(float(params.get("n_layers", 10))), 1)
hz_dep = h_bead_first if n_layers <= 1 else (h_bead_first + (n_layers - 1) * h_bead_subseq)
y_dep_centre = ly_base / 2.0
y_dep_left = y_dep_centre - ty_dep / 2.0
y_dep_right = y_dep_centre + ty_dep / 2.0

# 1. Substrate Plate Solid (lx x ly_base x hz_base, default 150x150x5mm)
sub_box = pv.Box(bounds=(0, lx, 0, ly_base, 0, hz_base))
plotter.add_mesh(sub_box, color="#38bdf8", opacity=0.75, show_edges=True, edge_color="#0284c7", line_width=1.0)

# Build Volume Enclosure (150x150x150mm above substrate, max 75% volume)
enc_box = pv.Box(bounds=(0, lx, 0, ly_base, hz_base, hz_base + build_height))
plotter.add_mesh(enc_box, color="#38bdf8", opacity=0.10, style="wireframe", line_width=1.2)

# 2. Deposited Geometry (CAD Model or Parametric Beads)
# Preferred source: the verts/faces the parent already loaded (immune to CWD /
# sys.path problems). Fallback: load the file here, with the framework root
# forced onto sys.path so `import gui...` works from any launch directory.
has_cad_runner = False
_cad_npz = params.get("viewer_cad_npz")
_cad_bbox = params.get("viewer_cad_bbox_mm")
if _cad_npz and os.path.exists(_cad_npz):
    try:
        _data = np.load(_cad_npz)
        verts = np.asarray(_data["verts"], dtype=np.float64)
        faces = np.asarray(_data["faces"], dtype=np.int64)
        if verts.size and faces.size:
            faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
            cad_poly = pv.PolyData(verts, faces_pv)
            cad_poly['Z (mm)'] = verts[:, 2]
            plotter.add_mesh(cad_poly, scalars='Z (mm)', cmap='plasma', opacity=0.90, show_edges=True, edge_color='#7c2d12', line_width=0.8, scalar_bar_args={{"title": "CAD Height Z (mm)"}})
            has_cad_runner = True
            if _cad_bbox:
                bb = _cad_bbox
                x_torch = (bb["xmin"] + bb["xmax"]) / 2.0
                y_torch = (bb["ymin"] + bb["ymax"]) / 2.0
                z_torch = bb["zmax"]
    except Exception as err:
        print("[CAD Runner View Error]", err)

if not has_cad_runner:
    cad_path = params.get("stl_file") or params.get("mesh_file")
    if cad_path and os.path.exists(cad_path):
        ext = os.path.splitext(cad_path)[1].lower()
        if ext in ('.stl', '.step', '.stp', '.iges', '.igs', '.ply', '.obj'):
            try:
                if {repr(fw_root)} not in sys.path:
                    sys.path.insert(0, {repr(fw_root)})
                from gui.widgets.mesh_viewer_3d import load_any_cad_geometry
                verts, faces, info = load_any_cad_geometry(cad_path)
                if verts is not None and faces is not None:
                    faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
                    cad_poly = pv.PolyData(verts, faces_pv)
                    cad_poly['Z (mm)'] = verts[:, 2]
                    plotter.add_mesh(cad_poly, scalars='Z (mm)', cmap='plasma', opacity=0.90, show_edges=True, edge_color='#7c2d12', line_width=0.8, scalar_bar_args={{"title": "CAD Height Z (mm)"}})
                    has_cad_runner = True
                    bb = info["bbox_mm"]
                    x_torch = (bb["xmin"] + bb["xmax"]) / 2.0
                    y_torch = (bb["ymin"] + bb["ymax"]) / 2.0
                    z_torch = bb["zmax"]
            except Exception as err:
                print("[CAD Runner View Error]", err)

if not has_cad_runner:
    layer_colors = ["#f97316", "#fb923c", "#ea580c", "#d97706"]
    for lay in range(n_layers):
        if lay == 0:
            z0 = hz_base
            z1 = hz_base + h_bead_first
        else:
            z0 = hz_base + h_bead_first + (lay - 1) * h_bead_subseq
            z1 = z0 + h_bead_subseq
        col = layer_colors[lay % len(layer_colors)]
        bead_box = pv.Box(bounds=(0, lx, y_dep_left, y_dep_right, z0, z1))
        plotter.add_mesh(bead_box, color=col, opacity=0.90, show_edges=True, edge_color="#7c2d12", line_width=1.0)
    x_torch = lx * 0.55
    y_torch = y_dep_centre
    z_torch = hz_base + hz_dep

# Active Heat Source Semi-Axes
a_hs = _get_dim("bG", 7.0)
b_hs = _get_dim("cG", 4.0)
cf_hs = _get_dim("af", 8.0)
cr_hs = _get_dim("ar", 20.0)

# Front semi-ellipsoid
sph_f = pv.Sphere(radius=1.0, phi_resolution=24, theta_resolution=24)
sph_f.scale((cf_hs, a_hs, b_hs), inplace=True)
sph_f.translate((x_torch, y_torch, z_torch), inplace=True)
plotter.add_mesh(sph_f, color="#facc15", opacity=0.65, show_edges=True, edge_color="#ca8a04")

# Rear semi-ellipsoid
sph_r = pv.Sphere(radius=1.0, phi_resolution=24, theta_resolution=24)
sph_r.scale((cr_hs, a_hs, b_hs), inplace=True)
sph_r.translate((x_torch, y_torch, z_torch), inplace=True)
plotter.add_mesh(sph_r, color="#ef4444", opacity=0.55, show_edges=True, edge_color="#991b1b")

# Torch Nozzle Downward Arrow
arrow = pv.Arrow(start=(x_torch, y_torch, z_torch + 18.0), direction=(0, 0, -1), scale=15.0)
plotter.add_mesh(arrow, color="#ef4444")

# Travel Velocity Vector
v_arrow = pv.Arrow(start=(x_torch, y_torch, z_torch + 3.0), direction=(1, 0, 0), scale=28.0)
plotter.add_mesh(v_arrow, color="#38bdf8")

# Coordinate Axes & Dimensional Grid
plotter.add_axes()
plotter.show_bounds(color="#64748b", xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", grid="front", location="outer")

plotter.view_isometric()
plotter.reset_camera()
plotter.show()

try:
    os.remove({repr(tmp_params_path)})
except Exception:
    pass
try:
    if params.get("viewer_cad_npz") and os.path.exists(params["viewer_cad_npz"]):
        os.remove(params["viewer_cad_npz"])
except Exception:
    pass
'''

    subprocess.Popen([sys.executable, "-c", runner_script])


class EmbeddedMeshViewport(tk.Frame):
    """
    Scientific 3D finite element mesh viewport and Heat Source Inspector.
    PyVista hardware GPU off-screen renderer is used by default with mouse orbit/zoom.
    """

    def __init__(self, parent, params: dict = None, app=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.params = params or {}
        self.app = app

        # Camera view state
        self.cam_preset = "iso"
        self.cam_azimuth = 0.0
        self.cam_elevation = 0.0
        self.cam_zoom = 1.0
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._rerender_job = None
        self._photo_image = None
        self.canvas_width = 800
        self.canvas_height = 550

        self._build_ui()
        if self.params:
            self.update_mesh(self.params)

    def _build_ui(self):
        # Keep refs so apply_theme(mode) can recolor without rebuild.
        self._vp_buttons = []
        self._vp_preset_btns = []
        self.viewport_theme = getattr(self, "viewport_theme", "dark")
        # 1. Top High-End Control Ribbon
        self.ribbon = tk.Frame(self, bg="#0f172a", padx=8, pady=6)
        self.ribbon.pack(fill=tk.X)

        title_lbl = tk.Label(
            self.ribbon, text="3D Viewport and Heat Source Studio (PyVista)",
            fg="#38bdf8", bg="#0f172a", font=FONT_STATUS
        )
        title_lbl.pack(side=tk.LEFT, padx=(4, 10))
        self.vp_title_lbl = title_lbl

        # Primary Action: Launch Full PyVista Window (text labels, no emoji-only)
        gpu_btn = tk.Button(
            self.ribbon, text="Open PyVista 3D Window",
            font=FONT_SMALL_BOLD, bg="#10b981", fg="#111827",
            relief=tk.FLAT, padx=10, pady=3, command=self._launch_gpu
        )
        gpu_btn.pack(side=tk.LEFT, padx=3)
        self._vp_buttons.append(("accent", gpu_btn))

        refresh_btn = tk.Button(
            self.ribbon, text="Refresh 3D",
            font=FONT_SMALL_BOLD, bg="#0284c7", fg="#111827",
            relief=tk.FLAT, padx=8, pady=3, command=self._redraw
        )
        refresh_btn.pack(side=tk.LEFT, padx=3)
        self._vp_buttons.append(("info", refresh_btn))

        # 2. Camera Controls & Status Bar
        self.cam_bar = tk.Frame(self, bg="#1e293b", padx=8, pady=4)
        self.cam_bar.pack(fill=tk.X)

        lbl_cam = tk.Label(self.cam_bar, text="Camera View:", bg="#1e293b", fg="#94a3b8", font=FONT_SMALL_BOLD)
        lbl_cam.pack(side=tk.LEFT, padx=(4, 6))
        self.vp_cam_label = lbl_cam

        presets = [
            ("Isometric", "iso"),
            ("Top (X-Y)", "top"),
            ("Side (X-Z)", "side"),
            ("Front (Y-Z)", "front"),
        ]
        for name, preset in presets:
            btn = tk.Button(
                self.cam_bar, text=name, font=(UI_FONT_FAMILY, 8),
                bg="#334155", fg="#f8fafc", relief=tk.FLAT, padx=6, pady=2,
                command=lambda p=preset: self._set_cam_preset(p)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self._vp_preset_btns.append(btn)

        reset_btn = tk.Button(
            self.cam_bar, text="Reset View", font=(UI_FONT_FAMILY, 8, "bold"),
            bg="#0f172a", fg="#38bdf8", relief=tk.FLAT, padx=6, pady=2,
            command=self._reset_camera
        )
        reset_btn.pack(side=tk.LEFT, padx=4)
        self.vp_reset_btn = reset_btn

        zin_btn = tk.Button(
            self.cam_bar, text="+", font=(UI_FONT_FAMILY, 8, "bold"),
            bg="#334155", fg="#f8fafc", relief=tk.FLAT, padx=4, pady=2,
            command=lambda: self._zoom(1.15)
        )
        zin_btn.pack(side=tk.LEFT, padx=1)
        self.vp_zin_btn = zin_btn

        zout_btn = tk.Button(
            self.cam_bar, text="-", font=(UI_FONT_FAMILY, 8, "bold"),
            bg="#334155", fg="#f8fafc", relief=tk.FLAT, padx=4, pady=2,
            command=lambda: self._zoom(0.87)
        )
        zout_btn.pack(side=tk.LEFT, padx=1)
        self.vp_zout_btn = zout_btn

        self.stats_lbl = tk.Label(
            self.cam_bar, text="Goldak: a=7.0, b=4.0, cf=8.0, cr=20.0 mm | Q=3840 W",
            fg="#94a3b8", bg="#1e293b", font=FONT_MONO
        )
        self.stats_lbl.pack(side=tk.RIGHT, padx=6)

        # 3. Main Workspace Split: Left = Embedded PyVista Viewport, Right = Heat Source HUD Card
        self.main_split = tk.Frame(self, bg="#090d16")
        self.main_split.pack(fill=tk.BOTH, expand=True)

        # Left 3D Viewport Frame
        self.canvas_frame = tk.Frame(self.main_split, bg="#090d16")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pv_canvas = tk.Canvas(self.canvas_frame, bg="#090d16", highlightthickness=0)
        self.pv_canvas.pack(fill=tk.BOTH, expand=True)

        # Interactive Mouse Bindings on Canvas
        self.pv_canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.pv_canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.pv_canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.pv_canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.pv_canvas.bind("<Button-5>", lambda e: self._zoom(0.9))
        self.pv_canvas.bind("<Configure>", self._on_canvas_configure)

        # Right Inspector Card (Heat Source Dimensions & Toolpath)
        self.side_inspector = tk.Frame(self.main_split, bg="#0f172a", width=340, padx=10, pady=8)
        self.side_inspector.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_inspector.pack_propagate(False)

        # Heat Source Title
        hs_title = tk.Label(
            self.side_inspector, text="Active Heat Source Dimensions",
            font=FONT_STATUS, fg="#facc15", bg="#0f172a"
        )
        hs_title.pack(anchor="w", pady=(0, 6))
        self.vp_hs_title = hs_title

        # Dimension Grid Container
        self.hs_grid_frame = tk.Frame(self.side_inspector, bg="#1e293b", padx=8, pady=8, relief=tk.SOLID, bd=1)
        self.hs_grid_frame.pack(fill=tk.X, pady=(0, 10))

        self.hs_labels = {}
        fields = [
            ("Half-Width (a = bG):", "a_val", "7.00 mm"),
            ("Penetration Depth (b = cG):", "b_val", "4.00 mm"),
            ("Front Semi-axis (c_f = af):", "cf_val", "8.00 mm"),
            ("Rear Semi-axis (c_r = ar):", "cr_val", "20.00 mm"),
            ("Net Arc Power (Q_net):", "q_net_val", "3840 W"),
            ("Heat Input (H_net):", "h_net_val", "384.0 J/mm"),
            ("Peak Power Density Front:", "q0f_val", "19.2 W/mm³"),
            ("Peak Power Density Rear:", "q0r_val", "17.9 W/mm³"),
        ]
        for row, (label_txt, key, default_val) in enumerate(fields):
            lbl = tk.Label(self.hs_grid_frame, text=label_txt, font=FONT_SMALL, fg="#94a3b8", bg="#1e293b")
            lbl.grid(row=row, column=0, sticky="w", pady=2)
            val_lbl = tk.Label(self.hs_grid_frame, text=default_val, font=(MONO_FONT_FAMILY, 9, "bold"), fg="#f8fafc", bg="#1e293b")
            val_lbl.grid(row=row, column=1, sticky="e", pady=2)
            self.hs_labels[key] = val_lbl

        # 2D Heat Source Cross-Section Canvas
        diag_title = tk.Label(
            self.side_inspector, text="Analytical Longitudinal Profile (X-Z)",
            font=FONT_SECTION, fg="#38bdf8", bg="#0f172a"
        )
        diag_title.pack(anchor="w", pady=(4, 4))
        self.vp_diag_title = diag_title

        self.diag_fig, self.diag_ax = plt.subplots(figsize=(3.4, 2.0), facecolor="#0b1120")
        self.diag_ax.set_facecolor("#0b1120")
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=self.side_inspector)
        self.diag_canvas.get_tk_widget().pack(fill=tk.X, pady=(0, 10))

        # Toolpath and G-code Operations Card
        tp_title = tk.Label(
            self.side_inspector, text="G-code and Operations",
            font=FONT_SECTION, fg="#34d399", bg="#0f172a"
        )
        tp_title.pack(anchor="w", pady=(4, 4))
        self.vp_tp_title = tp_title

        self.tp_summary_lbl = tk.Label(
            self.side_inspector,
            text="G-code: Ready\nTotal Length: ~2.00 m | Feedrate: 600 mm/min\nDeposition Path: Unidirectional (+X)",
            font=FONT_TINY, fg="#cbd5e1", bg="#1e293b", justify=tk.LEFT, padx=8, pady=6, relief=tk.SOLID, bd=1
        )
        self.tp_summary_lbl.pack(fill=tk.X)
        self.viewport_theme = "dark"
        self.apply_theme("dark")

    def apply_theme(self, mode: str):
        """Recolor viewport chrome for light/dark app theme. Idempotent."""
        pal = VIEWPORT_THEMES.get(mode, VIEWPORT_THEMES["dark"])
        self.viewport_theme = mode
        try:
            self.ribbon.configure(bg=pal["ribbon_bg"])
            self.vp_title_lbl.configure(bg=pal["ribbon_bg"], fg=pal["title_fg"])
            self.cam_bar.configure(bg=pal["cambar_bg"])
            self.vp_cam_label.configure(bg=pal["cambar_bg"], fg=pal["camdim_fg"])
            for b in getattr(self, "_vp_preset_btns", []):
                b.configure(bg=pal["preset_bg"], fg=pal["preset_fg"])
            self.vp_reset_btn.configure(bg=pal["reset_bg"], fg=pal["reset_fg"])
            self.vp_zin_btn.configure(bg=pal["preset_bg"], fg=pal["preset_fg"])
            self.vp_zout_btn.configure(bg=pal["preset_bg"], fg=pal["preset_fg"])
            self.stats_lbl.configure(bg=pal["cambar_bg"], fg=pal["stats_fg"])
            self.main_split.configure(bg=pal["canvas_bg"])
            self.canvas_frame.configure(bg=pal["canvas_bg"])
            self.pv_canvas.configure(bg=pal["canvas_bg"])
            self.side_inspector.configure(bg=pal["inspector_bg"])
            self.vp_hs_title.configure(bg=pal["inspector_bg"], fg=pal["hs_title_fg"])
            self.hs_grid_frame.configure(bg=pal["grid_bg"])
            for child in self.hs_grid_frame.winfo_children():
                try:
                    if child in self.hs_labels.values():
                        child.configure(bg=pal["grid_bg"], fg=pal["grid_value_fg"])
                    else:
                        child.configure(bg=pal["grid_bg"], fg=pal["grid_label_fg"])
                except Exception:
                    pass
            self.vp_diag_title.configure(bg=pal["inspector_bg"], fg=pal["diag_title_fg"])
            self.vp_tp_title.configure(bg=pal["inspector_bg"], fg=pal["tp_title_fg"])
            self.tp_summary_lbl.configure(bg=pal["tp_bg"], fg=pal["tp_fg"])
            # Matplotlib diagram follows theme; redrawn on next update_mesh.
            try:
                self.diag_fig.set_facecolor(pal["diag_face"])
                self.diag_ax.set_facecolor(pal["diag_face"])
                self._vp_diag_palette = pal
                if hasattr(self, "params") and self.params:
                    self._draw_2d_diagram()
                else:
                    self.diag_canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    def _set_cam_preset(self, preset: str):
        self.cam_preset = preset
        self.cam_azimuth = 0.0
        self.cam_elevation = 0.0
        self._redraw()

    def _reset_camera(self):
        self.cam_preset = "iso"
        self.cam_azimuth = 0.0
        self.cam_elevation = 0.0
        self.cam_zoom = 1.0
        self._redraw()

    def _zoom(self, factor: float):
        self.cam_zoom *= factor
        self.cam_zoom = max(0.2, min(5.0, self.cam_zoom))
        self._redraw()

    def _on_mouse_down(self, event):
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def _on_mouse_drag(self, event):
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        self.cam_azimuth += dx * 0.6
        self.cam_elevation -= dy * 0.6

        # Debounce re-rendering during dragging
        if self._rerender_job:
            self.after_cancel(self._rerender_job)
        self._rerender_job = self.after(15, self._render_pyvista)

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self._zoom(1.1)
        elif event.delta < 0:
            self._zoom(0.9)

    def _on_canvas_configure(self, event):
        w = max(event.width, 200)
        h = max(event.height, 200)
        if abs(w - self.canvas_width) > 30 or abs(h - self.canvas_height) > 30:
            self.canvas_width = w
            self.canvas_height = h
            self._redraw()

    def update_mesh(self, params: dict):
        self.params = params
        self._update_inspector_data()
        self._draw_2d_diagram()
        self._redraw()

    def _update_inspector_data(self):
        """Update numerical values in the Heat Source inspector."""
        g = _extract_geometry_params(self.params)

        if "a_val" in self.hs_labels:
            self.hs_labels["a_val"].config(text=f"{g['a_hs']:.2f} mm")
            self.hs_labels["b_val"].config(text=f"{g['b_hs']:.2f} mm")
            self.hs_labels["cf_val"].config(text=f"{g['cf_hs']:.2f} mm")
            self.hs_labels["cr_val"].config(text=f"{g['cr_hs']:.2f} mm")
            self.hs_labels["q_net_val"].config(text=f"{g['Q_net']:.0f} W")
            self.hs_labels["h_net_val"].config(text=f"{g['heat_input']:.1f} J/mm")
            self.hs_labels["q0f_val"].config(text=f"{g['q0_f']:.1f} W/mm³")
            self.hs_labels["q0r_val"].config(text=f"{g['q0_r']:.1f} W/mm³")

        self.stats_lbl.config(
            text=f"Goldak: a={g['a_hs']:.1f}, b={g['b_hs']:.1f}, cf={g['cf_hs']:.1f}, cr={g['cr_hs']:.1f} mm | Q={g['Q_net']:.0f} W"
        )

        # Update Toolpath Summary Text
        gcode_file = self.params.get("gcode_file") or self.params.get("mesh_file")
        if gcode_file and os.path.exists(gcode_file) and str(gcode_file).endswith(('.gcode', '.nc')):
            fname = os.path.basename(gcode_file)
            self.tp_summary_lbl.config(
                text=f"Custom G-code: {fname}\nLayers: {g['n_layers']} | Travel: {g['v_scan']:.0f} mm/min\nDirect simulation preview ready."
            )
        elif self.params.get("stl_file") or (self.params.get("mesh_file") and str(self.params.get("mesh_file")).lower().endswith(('.stl', '.step', '.stp', '.iges', '.igs', '.ply', '.obj', '.mesh'))):
            cad_f = self.params.get("stl_file") or self.params.get("mesh_file")
            fname = os.path.basename(cad_f)
            self.tp_summary_lbl.config(
                text=f"Loaded CAD: {fname}\nLayers: {g['n_layers']} | Base: {g['lx']:.0f}x{g['ly_base']:.0f}x{g['hz_base']:.0f} mm\n3D CAD Viewport active."
            )
        else:
            self.tp_summary_lbl.config(
                text=f"Parametric WAAM Pass: {g['n_layers']} Layers\nTravel Length: {g['lx']:.0f} mm/layer\nTotal Weld Length: {g['lx']*g['n_layers']/1000.0:.2f} m"
            )

    def _draw_2d_diagram(self):
        """Draw 2D analytical Goldak cross-section diagram (follows viewport theme)."""
        g = _extract_geometry_params(self.params)
        cf = g["cf_hs"]
        cr = g["cr_hs"]
        b = g["b_hs"]
        pal = getattr(self, "_vp_diag_palette", None) or VIEWPORT_THEMES.get(
            getattr(self, "viewport_theme", "dark"), VIEWPORT_THEMES["dark"])

        self.diag_ax.cla()
        self.diag_fig.set_facecolor(pal["diag_face"])
        self.diag_ax.set_facecolor(pal["diag_face"])

        # Ellipse coordinates: front (X >= 0) and rear (X <= 0)
        theta_f = np.linspace(0, np.pi / 2, 50)
        xf = cf * np.cos(theta_f)
        zf = -b * np.sin(theta_f)

        theta_r = np.linspace(np.pi / 2, np.pi, 50)
        xr = cr * np.cos(theta_r)
        zr = -b * np.sin(theta_r)

        # Substrate line
        self.diag_ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.2, label="Bead Surface (Z=0)")

        # Fill Front Ellipsoid
        self.diag_ax.fill_between(xf, 0, zf, color="#facc15", alpha=0.55, label=f"Front (cf={cf:.1f})")
        self.diag_ax.plot(xf, zf, color="#facc15", linewidth=2.0)

        # Fill Rear Ellipsoid
        self.diag_ax.fill_between(xr, 0, zr, color="#ef4444", alpha=0.45, label=f"Rear (cr={cr:.1f})")
        self.diag_ax.plot(xr, zr, color="#ef4444", linewidth=2.0)

        # Torch travel arrow
        self.diag_ax.annotate(
            "v_scan",
            xy=(cf * 0.7, b * 0.3), xytext=(cf * 0.1, b * 0.3),
            arrowprops=dict(facecolor="#38bdf8", edgecolor="#38bdf8", width=1.5, headwidth=6),
            color="#38bdf8", fontsize=9, fontweight="bold"
        )

        self.diag_ax.set_xlabel("X (Travel) [mm]", color=pal["diag_tick"], fontsize=7)
        self.diag_ax.set_ylabel("Z (Depth) [mm]", color=pal["diag_tick"], fontsize=7)
        self.diag_ax.tick_params(colors=pal["diag_tick"], labelsize=7)
        for spine in self.diag_ax.spines.values():
            spine.set_color(pal["diag_spine"])

        self.diag_ax.set_xlim(-cr * 1.25, cf * 1.35)
        self.diag_ax.set_ylim(-b * 1.4, b * 0.6)
        self.diag_fig.tight_layout()
        self.diag_canvas.draw_idle()

    def _launch_gpu(self):
        """Launch native standalone PyVista hardware-accelerated VTK window."""
        params = self._current_params()
        launch_pyvista_gpu_window(params)

    def _current_params(self) -> dict:
        """Latest parameters for the viewers: live app params + last-known
        CAD/toolpath paths. `_collect_params()` re-derives stl_file from the
        app state, but a mesh selected outside it (or a gcode loaded earlier)
        must not be dropped, or the 3D window silently reverts to the
        parametric bead wall instead of the custom geometry."""
        params = dict(self.params or {})
        app = getattr(self, "app", None) or getattr(self.master, "app", None)
        if app and hasattr(app, "_collect_params"):
            try:
                collected = app._collect_params()
                if collected:
                    params = collected
            except Exception:
                pass
        for key in ("stl_file", "mesh_file", "gcode_file"):
            if not params.get(key):
                params[key] = (self.params or {}).get(key)
        if not params.get("stl_file") and app and hasattr(app, "loaded_stl_path") and app.loaded_stl_path:
            params["stl_file"] = app.loaded_stl_path
        self.params = params
        return params

    def _redraw(self):
        if not self.params:
            return
        self._render_pyvista()

    def _render_pyvista(self):
        """Render complete WAAM workpiece via PyVista off-screen GPU renderer."""
        try:
            import pyvista as pv
        except ImportError:
            self._render_fallback_matplotlib()
            return

        g = _extract_geometry_params(self.params)
        w = max(self.canvas_width, 400)
        h = max(self.canvas_height, 300)

        plotter = pv.Plotter(off_screen=True, window_size=(w, h))
        plotter.set_background("#090d16", top="#0f172a")

        # 1. Substrate Plate Solid
        sub_box = pv.Box(bounds=(0, g["lx"], 0, g["ly_base"], 0, g["hz_base"]))
        plotter.add_mesh(sub_box, color="#38bdf8", opacity=0.75, show_edges=True, edge_color="#0284c7", line_width=1.0)

        # Build Volume Enclosure (150x150x150mm wireframe)
        enc_box = pv.Box(bounds=(0, g["lx"], 0, g["ly_base"], g["hz_base"], g["hz_base"] + 150.0))
        plotter.add_mesh(enc_box, color="#38bdf8", opacity=0.10, style="wireframe", line_width=1.0)

        # 2. Deposited Geometry (CAD Model or Parametric Beads)
        cad_path = self.params.get("stl_file") or self.params.get("mesh_file")
        # A Medit .mesh cannot be previewed as a surface; if its source STL
        # sits next to it, fall back to that instead of the parametric beads.
        if cad_path and str(cad_path).lower().endswith((".mesh", ".mesh3")):
            _sib = str(Path(cad_path).with_suffix(".stl"))
            if os.path.exists(_sib):
                cad_path = _sib
        has_cad = False
        if cad_path and os.path.exists(cad_path):
            ext = os.path.splitext(cad_path)[1].lower()
            if ext in ('.stl', '.step', '.stp', '.iges', '.igs', '.ply', '.obj'):
                try:
                    verts, faces, info = load_any_cad_geometry(cad_path, sub_hz_mm=g["hz_base"])
                    if verts is not None and faces is not None:
                        faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
                        cad_poly = pv.PolyData(verts, faces_pv)
                        cad_poly["Z (mm)"] = verts[:, 2]
                        plotter.add_mesh(
                            cad_poly, scalars="Z (mm)", cmap="plasma", opacity=0.90,
                            show_edges=True, edge_color="#7c2d12", line_width=0.8,
                            scalar_bar_args={"title": "CAD Height Z (mm)", "color": "white", "title_font_size": 10, "label_font_size": 8}
                        )
                        has_cad = True
                        bb = info["bbox_mm"]
                        x_torch = (bb["xmin"] + bb["xmax"]) / 2.0
                        y_torch = (bb["ymin"] + bb["ymax"]) / 2.0
                        z_torch = bb["zmax"]
                        
                        # Add transparent slicing planes to visualize layer resolution
                        h1 = g.get("h_bead_first", g.get("h_bead", 4.0))
                        hsub = g.get("h_bead_subseq", h1 * g.get("remelt_ratio", 1.0))
                        n_layers = g.get("n_layers", 10)
                        
                        for lay in range(1, n_layers + 1):
                            z_plane = g["hz_base"] + h1 + (lay - 1) * hsub
                            if z_plane <= bb["zmax"] + 1.0:
                                plane = pv.Plane(center=(x_torch, y_torch, z_plane),
                                                 direction=(0, 0, 1),
                                                 i_size=g["lx"]*1.1, j_size=g["ly_base"]*1.1)
                                plotter.add_mesh(plane, color="white", opacity=0.25, style="wireframe")
                except Exception as err:
                    print(f"[CAD View Error] {err}")

        if not has_cad:
            layer_colors = ["#f97316", "#fb923c", "#ea580c", "#d97706"]
            h1 = g.get("h_bead_first", g.get("h_bead", 4.0))
            hsub = g.get("h_bead_subseq", h1 * g.get("remelt_ratio", 0.80))
            for lay in range(g["n_layers"]):
                if lay == 0:
                    z0 = g["hz_base"]
                    z1 = g["hz_base"] + h1
                else:
                    z0 = g["hz_base"] + h1 + (lay - 1) * hsub
                    z1 = z0 + hsub
                col = layer_colors[lay % len(layer_colors)]
                bead_box = pv.Box(bounds=(0, g["lx"], g["y_dep_left"], g["y_dep_right"], z0, z1))
                plotter.add_mesh(bead_box, color=col, opacity=0.90, show_edges=True, edge_color="#7c2d12", line_width=1.0)
            x_torch = g["lx"] * 0.55
            y_torch = g["y_dep_centre"]
            z_torch = g["hz_base"] + g["hz_dep"]
        hs_type = g.get("heat_source", "goldak")
        if hs_type == "gaussian_vol":
            r_b = max(g.get("R_beam", 5.0), 0.5)
            cyl = pv.Cylinder(center=(x_torch, y_torch, z_torch - 2.0), direction=(0, 0, -1), radius=r_b, height=12.0, resolution=24)
            plotter.add_mesh(cyl, color="#a855f7", opacity=0.70, show_edges=True, edge_color="#7e22ce")
        elif hs_type == "gaussian_surf":
            r_s = max(g.get("R_spot", 3.0), 0.5)
            disc = pv.Cylinder(center=(x_torch, y_torch, z_torch), direction=(0, 0, 1), radius=r_s, height=0.4, resolution=24)
            plotter.add_mesh(disc, color="#a855f7", opacity=0.85, show_edges=True, edge_color="#7e22ce")
        elif hs_type == "conical":
            rt = max(g.get("rt", 3.0), 0.5)
            rb = max(g.get("rb", 1.0), 0.2)
            d_p = max(g.get("d_pen", 5.0), 1.0)
            cone = pv.Cone(center=(x_torch, y_torch, z_torch - d_p / 2.0), direction=(0, 0, -1), height=d_p, radius=rt, resolution=24)
            plotter.add_mesh(cone, color="#ec4899", opacity=0.70, show_edges=True, edge_color="#be185d")
        elif hs_type == "uniform":
            ab = max(g.get("a_band", 10.0), 1.0)
            bb = max(g.get("b_band", 5.0), 1.0)
            cb = max(g.get("c_band", 3.0), 1.0)
            u_box = pv.Box(bounds=(x_torch - ab, x_torch + ab, y_torch - bb, y_torch + bb, z_torch - cb, z_torch))
            plotter.add_mesh(u_box, color="#10b981", opacity=0.65, show_edges=True, edge_color="#047857")
        elif hs_type == "prescribed":
            r_act = max(g.get("R_activate", 10.0), 1.0)
            p_sph = pv.Sphere(radius=r_act, center=(x_torch, y_torch, z_torch), phi_resolution=24, theta_resolution=24)
            plotter.add_mesh(p_sph, color="#6366f1", opacity=0.55, show_edges=True, edge_color="#4338ca")
        elif hs_type in ("p48_surface", "p48"):
            i_w = max(g.get("I_weld", 150.0), 10.0)
            sig_q = 6.205e-4 * (i_w ** 0.28) * 1000.0
            arc_disc = pv.Cylinder(center=(x_torch, y_torch, z_torch), direction=(0, 0, 1), radius=sig_q, height=0.6, resolution=24)
            plotter.add_mesh(arc_disc, color="#38bdf8", opacity=0.75, show_edges=True, edge_color="#0284c7")
            drop_sph = pv.Sphere(radius=sig_q * 0.4, center=(x_torch, y_torch, z_torch - 1.5), phi_resolution=16, theta_resolution=16)
            plotter.add_mesh(drop_sph, color="#f97316", opacity=0.85, show_edges=True, edge_color="#c2410c")
        else:
            a = max(g.get("a_hs", 7.0), 0.5)
            b = max(g.get("b_hs", 4.0), 0.5)
            cf = max(g.get("cf_hs", 8.0), 0.5)
            cr = max(g.get("cr_hs", 20.0), 0.5)

            sph_f = pv.Sphere(radius=1.0, phi_resolution=24, theta_resolution=24)
            sph_f.scale((cf, a, b), inplace=True)
            sph_f.translate((x_torch, y_torch, z_torch), inplace=True)
            plotter.add_mesh(sph_f, color="#facc15", opacity=0.65, show_edges=True, edge_color="#ca8a04")

            sph_r = pv.Sphere(radius=1.0, phi_resolution=24, theta_resolution=24)
            sph_r.scale((cr, a, b), inplace=True)
            sph_r.translate((x_torch, y_torch, z_torch), inplace=True)
            plotter.add_mesh(sph_r, color="#ef4444", opacity=0.55, show_edges=True, edge_color="#991b1b")

        # Torch Nozzle Downward Arrow
        arrow = pv.Arrow(start=(x_torch, y_torch, z_torch + 18.0), direction=(0, 0, -1), scale=15.0)
        plotter.add_mesh(arrow, color="#ef4444")

        # Travel Velocity Vector
        v_arrow = pv.Arrow(start=(x_torch, y_torch, z_torch + 3.0), direction=(1, 0, 0), scale=28.0)
        plotter.add_mesh(v_arrow, color="#38bdf8")

        # Coordinate Axes & Dimensional Grid
        plotter.add_axes()
        plotter.show_bounds(
            color="#64748b",
            xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)",
            grid="front", location="outer"
        )

        # Apply Camera View Preset
        if self.cam_preset == "top":
            plotter.view_xy(render=False)
            plotter.camera.up = (0.0, 1.0, 0.0)
        elif self.cam_preset == "side":
            plotter.view_xz(render=False)
            plotter.camera.up = (0.0, 0.0, 1.0)
        elif self.cam_preset == "front":
            plotter.view_yz(render=False)
            plotter.camera.up = (0.0, 0.0, 1.0)
        else: # "iso"
            plotter.view_isometric(render=False)

        # Apply Interactive Orbit & Zoom
        if self.cam_azimuth != 0.0:
            plotter.camera.azimuth = self.cam_azimuth
        if self.cam_elevation != 0.0:
            plotter.camera.elevation = self.cam_elevation
        if self.cam_zoom != 1.0:
            plotter.camera.zoom(self.cam_zoom)

        # Offscreen Snapshot
        img_array = plotter.screenshot(return_img=True)
        plotter.close()

        # Update Tkinter Canvas
        img = Image.fromarray(img_array)
        self._photo_image = ImageTk.PhotoImage(img)

        self.pv_canvas.delete("all")
        self.pv_canvas.create_image(
            w // 2, h // 2,
            image=self._photo_image,
            anchor=tk.CENTER
        )

    def _render_fallback_matplotlib(self):
        """Fallback rendering using Matplotlib if PyVista is not available."""
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import numpy as np
            g = _extract_geometry_params(self.params)
            fig_w = max(self.canvas_width, 400) / 100.0
            fig_h = max(self.canvas_height, 300) / 100.0
            fig = plt.Figure(figsize=(fig_w, fig_h), dpi=100, facecolor="#0b1120")
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#0b1120")
            # Substrate box corners
            lx, ly, hz = g["lx"], g["ly_base"], g["hz_base"]
            xs = [0, lx, lx, 0, 0, lx, lx, 0]
            ys = [0, 0, ly, ly, 0, 0, ly, ly]
            zs = [0, 0, 0, 0, hz, hz, hz, hz]
            ax.scatter(xs, ys, zs, c="#38bdf8", s=2)
            ax.text(lx / 2, ly / 2, hz + g["hz_dep"] + 5,
                    f"PyVista missing\nLx={lx:.0f} N={g['n_layers']}",
                    color="white", fontsize=8, ha="center")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            fig.tight_layout()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            for w in self.canvas_frame.winfo_children():
                if isinstance(w, FigureCanvasTkAgg):
                    pass
            # Render to Tk canvas as text notice (keep embedded viewport alive)
            self.pv_canvas.delete("all")
            self.pv_canvas.create_text(
                max(self.canvas_width, 400) // 2, max(self.canvas_height, 300) // 2,
                text="PyVista not installed.\nInstall: pip install pyvista\nShowing 2D diagram + params.",
                fill="white", font=(UI_FONT_FAMILY, 11), justify="center")
        except Exception:
            try:
                self.pv_canvas.delete("all")
                self.pv_canvas.create_text(
                    200, 150, text="3D unavailable (no PyVista)",
                    fill="white", font=(UI_FONT_FAMILY, 11))
            except Exception:
                pass


def show_3d_mesh_window(parent, params: dict):
    """Open standalone popup window with embedded viewport."""
    win = tk.Toplevel(parent)
    win.title("3D Viewport & Heat Source Studio (PyVista) — WAAM Multi-Physics")
    win.geometry("1100x760")
    vp = EmbeddedMeshViewport(win, params=params)
    vp.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Test WAAM 3D Viewport (PyVista Default)")
    vp = EmbeddedMeshViewport(root, params={
        "lx": 0.200, "ly_base": 0.100, "hz_base": 0.010,
        "ty_dep": 0.008, "h_bead": 0.003, "n_layers": 10,
        "I_weld": 200.0, "U_weld": 24.0, "eta_arc": 0.8,
        "v_scan": 10.0e-3, "bG": 7.0e-3, "cG": 3.0e-3, "af": 4.0e-3, "ar": 16.0e-3
    })
    vp.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
