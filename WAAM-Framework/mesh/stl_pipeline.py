"""
mesh/stl_pipeline.py

STL -> Volume Mesh Pipeline using Gmsh (OCCT kernel).
Produces a FreeFEM-compatible .mesh file from any watertight STL surface.

Pipeline:
  1. Load STL via trimesh (surface repair / watertight check)
  2. Feed repaired surface to Gmsh (OCC kernel)
  3. Generate 3D tetrahedral volume mesh inside Gmsh
  4. Export as .mesh (Medit format that FreeFEM reads natively)
  5. Return mesh path + statistics
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def _stl_to_freefem_mesh_in_process(stl_path: str, output_mesh_path: str = None,
                                    target_nodes: int = 2000,
                                    layer_height: float = 3.0) -> dict:
    """
    Core mesher running in an isolated process.
    """
    import trimesh
    import gmsh

    warnings = []
    stl_path = str(Path(stl_path).resolve())

    if output_mesh_path is None or not str(output_mesh_path).strip():
        output_mesh_path = str(Path(stl_path).with_suffix(".mesh"))

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)

    # Step 1: Load + repair surface via trimesh
    mesh_surf = trimesh.load_mesh(stl_path, force="mesh")
    if not mesh_surf.is_watertight:
        warnings.append("[STL-REPAIR] Surface not watertight — attempting auto-repair.")
        import trimesh.repair as tr
        tr.fix_normals(mesh_surf)
        tr.fill_holes(mesh_surf)
        if not mesh_surf.is_watertight:
            warnings.append("[STL-REPAIR] Partial repair applied. Proceeding with Gmsh.")

    # Detect STL units: if max dimension > 100, assume meters and convert to mm
    # STL files are typically in mm for WAAM, but some CAD exports use meters
    temp_bounds = mesh_surf.bounds
    temp_spans = temp_bounds[1] - temp_bounds[0]
    max_span = max(temp_spans)
    if max_span < 1.0:
        # Likely meters - convert to mm
        mesh_surf.apply_scale(1000.0)
        warnings.append(f"[STL-UNITS] Detected meter units (max span {max_span:.4f}). Scaled to mm.")
    elif max_span > 1000.0:
        # Likely already in mm or larger units - no conversion needed
        warnings.append(f"[STL-UNITS] Max span {max_span:.1f} - assuming mm units.")

    # Build enclosure limits: Substrate 150x150x15mm, Build volume 150x150x150mm above substrate
    SUB_LX = 150.0
    SUB_LY = 150.0
    SUB_HZ = 15.0
    BUILD_HEIGHT = 150.0
    V_ENCLOSURE = SUB_LX * SUB_LY * BUILD_HEIGHT  # 3,375,000 mm^3
    MAX_OCCUPIED_VOL = 0.75 * V_ENCLOSURE       # 2,531,250 mm^3 (75% max volume)

    bbox = mesh_surf.bounds
    spans = bbox[1] - bbox[0]
    v_bbox = float(spans[0] * spans[1] * spans[2])
    warnings.append(f"[STL-BBOX-RAW] X={spans[0]:.1f}  Y={spans[1]:.1f}  Z={spans[2]:.1f} (STL units, Vol={v_bbox:,.0f} mm³)")

    # Auto-scale if larger than enclosure or exceeds 75% volume limit
    scale_applied = 1.0
    if spans[0] > SUB_LX or spans[1] > SUB_LY or spans[2] > BUILD_HEIGHT or v_bbox > MAX_OCCUPIED_VOL:
        s_vol = (MAX_OCCUPIED_VOL / max(v_bbox, 1e-6)) ** (1.0 / 3.0)
        s_dim = min(SUB_LX / max(spans[0], 1e-6), SUB_LY / max(spans[1], 1e-6), BUILD_HEIGHT / max(spans[2], 1e-6))
        scale_applied = min(s_vol, s_dim)
        mesh_surf.apply_scale(scale_applied)
        v_after = v_bbox * (scale_applied ** 3)
        vol_pct = (v_after / V_ENCLOSURE) * 100.0
        warnings.append(
            f"[STL-AUTO-SCALE] Model exceeded 150x150x150mm enclosure. Scaled by {scale_applied:.4f}x "
            f"so occupied volume is <= 75% of enclosure ({v_after:,.1f} mm³ / {vol_pct:.1f}%)."
        )

    # Auto-place correctly on top of 150x150x15mm base plate:
    # Centered in X and Y, bottom resting on top of substrate (Z = 15.0 mm)
    bbox = mesh_surf.bounds
    spans = bbox[1] - bbox[0]
    center_x = (bbox[0][0] + bbox[1][0]) / 2.0
    center_y = (bbox[0][1] + bbox[1][1]) / 2.0
    min_z = bbox[0][2]

    tx = (SUB_LX / 2.0) - center_x
    ty = (SUB_LY / 2.0) - center_y
    tz = SUB_HZ - min_z
    mesh_surf.apply_translation([tx, ty, tz])
    warnings.append(
        f"[STL-PLACEMENT] Correctly placed on 150x150x15mm base plate: "
        f"Centered at X={SUB_LX/2.0:.1f}, Y={SUB_LY/2.0:.1f} mm, resting flush at Z={SUB_HZ:.1f} mm."
    )

    bbox = mesh_surf.bounds
    spans = bbox[1] - bbox[0]
    warnings.append(f"[STL-BBOX-PLACED] X={spans[0]:.1f}  Y={spans[1]:.1f}  Z={spans[2]:.1f} mm on [0..150, 0..150, 15..{15.0+spans[2]:.1f}]")

    tmp_stl = output_mesh_path + "_tmp_repaired.stl"
    mesh_surf.export(tmp_stl)

    # Step 2: Gmsh volume meshing
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("waam_volume")
    gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
    gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)

    try:
        gmsh.merge(tmp_stl)
        angle = 40.0 * (3.14159265358979 / 180.0)
        gmsh.model.mesh.classifySurfaces(angle, True, True)
        gmsh.model.mesh.createGeometry()
        surfs = gmsh.model.getEntities(2)
        if surfs:
            sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
            vol = gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
    except Exception as e:
        warnings.append(f"[STL-GMSH] Built-in volume formation warning: {e}. Trying OCC fallback.")
        try:
            gmsh.model.occ.importShapes(tmp_stl, highestDimOnly=False)
            gmsh.model.occ.synchronize()
            s = gmsh.model.getEntities(2)
            if s:
                sl = gmsh.model.occ.addSurfaceLoop([e[1] for e in s])
                gmsh.model.occ.addVolume([sl])
                gmsh.model.occ.synchronize()
        except Exception:
            pass

    # Calibrate element size based on layer height:
    # Rule: cell thickness/size of volume element must be <= 0.5 * layer_height (e.g. 1.5mm for h=3mm)
    # and capped from below at 0.10 * layer_height (0.3 mm, i.e. 10%)
    h_val = float(layer_height) if float(layer_height) > 0 else 3.0
    lc_layer_max = 0.50 * h_val
    lc_layer_min = max(0.10 * h_val, 0.30)

    vol = float(spans[0] * spans[1] * spans[2])
    if target_nodes > 0 and vol > 0:
        lc_nodes = (vol / max(target_nodes, 1)) ** (1.0 / 3.0)
    else:
        lc_nodes = lc_layer_max

    lc_max = min(lc_nodes, lc_layer_max)
    lc_min = min(lc_layer_min, lc_max * 0.5)

    # Node budget guard: cap maximum elements to prevent memory explosion
    # For V > 5e5 mm^3 with lc <= 1.5mm, elements can exceed 500k
    max_elements = 500000
    if vol > 0:
        est_elements = vol / (lc_max ** 3)
        if est_elements > max_elements:
            # Coarsen mesh to stay within budget
            lc_max = (vol / max_elements) ** (1.0 / 3.0)
            lc_min = min(lc_layer_min, lc_max * 0.5)
            warnings.append(
                f"[STL-MESH] Node budget guard: coarsened lc_max from {lc_layer_max:.3f}mm to {lc_max:.3f}mm "
                f"to stay under {max_elements} elements (est. {est_elements:.0f})."
            )

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.Optimize", 1)

    # Check if we have 3D volumes to mesh
    vols_now = gmsh.model.getEntities(3)
    if vols_now:
        gmsh.model.mesh.generate(3)
        mesh_dim = 3
    else:
        # Fallback to 2D surface if volume couldn't be stitched
        warnings.append("[STL-MESH] No closed volume detected. Generating surface (triangular) mesh.")
        gmsh.model.mesh.generate(2)
        mesh_dim = 2

    gmsh.write(output_mesh_path)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    n_tets = 0
    n_tris = 0

    if mesh_dim == 3:
        tet_type, tet_tags, _ = gmsh.model.mesh.getElements(3)
        n_tets = sum(len(t) for t in tet_tags) if tet_tags else 0
    else:
        tri_type, tri_tags, _ = gmsh.model.mesh.getElements(2)
        n_tris = sum(len(t) for t in tri_tags) if tri_tags else 0
        n_tets = n_tris  # Report triangles as 'elements'
        warnings.append(f"[STL-MESH] Surface mesh: {n_nodes:,} nodes  |  {n_tris:,} triangles")

    gmsh.finalize()


    try:
        os.remove(tmp_stl)
    except Exception:
        pass

    return {
        "mesh_path": output_mesh_path,
        "n_nodes": n_nodes,
        "n_tets": n_tets,
        "bbox_mm": {
            "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
            "xmin": float(bbox[0][0]), "ymin": float(bbox[0][1]), "zmin": float(bbox[0][2]),
            "xmax": float(bbox[1][0]), "ymax": float(bbox[1][1]), "zmax": float(bbox[1][2]),
        },
        "warnings": warnings,
    }


def stl_to_freefem_mesh(stl_path: str, output_mesh_path: str = None,
                         target_nodes: int = 2000,
                         layer_height: float = 3.0,
                         in_process: bool = False) -> dict:
    """
    Convert a STL file to a FreeFEM-compatible .mesh file via Gmsh.

    By default, executes Gmsh inside an isolated worker subprocess. This prevents
    macOS OpenMP (libgmsh) fatal SIGBUS / KERN_PROTECTION_FAILURE crashes when called
    from Tkinter or Cocoa event loops.
    """
    if in_process:
        return _stl_to_freefem_mesh_in_process(stl_path, output_mesh_path, target_nodes, layer_height)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(stl_path),
        str(output_mesh_path or ""),
        str(target_nodes),
        str(layer_height),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out_text = proc.stdout.strip()
        json_start = out_text.find("{")
        json_end = out_text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return json.loads(out_text[json_start:json_end+1])
        else:
            # Fallback to in-process if JSON could not be parsed
            print(f"[STL-PIPELINE] Warning: Could not parse worker JSON, falling back to in-process: {out_text}")
            return _stl_to_freefem_mesh_in_process(stl_path, output_mesh_path, target_nodes)
    except Exception as e:
        print(f"[STL-PIPELINE] Subprocess runner error: {e}, attempting in-process execution.")
        return _stl_to_freefem_mesh_in_process(stl_path, output_mesh_path, target_nodes)


if __name__ == "__main__":
    default_stl = str(Path(__file__).resolve().parent.parent / "Main.stl")
    stl = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else default_stl
    out_mesh = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
    target_nodes = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].strip() else 2000
    lh = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].strip() else 3.0

    result = _stl_to_freefem_mesh_in_process(stl, out_mesh, target_nodes=target_nodes, layer_height=lh)
    print(json.dumps(result, indent=2))

