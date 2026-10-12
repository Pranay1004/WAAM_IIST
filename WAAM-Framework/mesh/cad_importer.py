"""
CAD importer: STEP/IGES/BREP → fused-domain FreeFEM .mesh.

STEP/IGES/BREP files are imported with the Gmsh OCC kernel, their surface skin
is exported to a temporary STL, and mesh/stl_pipeline.stl_to_freefem_mesh()
builds the SOLVER domain from it: substrate plate fused with the part into ONE
conformal tetrahedral volume, with the boundary labels the generated .edp
expects (see mesh/structured_conforming._BOUNDARY_LABEL set B):

    1 -> substrate underside (hBot + fixture spring)
    5 -> top, 2 -> every other surface (all convective)

The previous implementation meshed ONLY the imported part and tagged it with
its own private scheme (surfaces 1/2/3, volume 10). None of those labels exist
in the solver's forms, so convection, radiation, base transfer and the fixture
spring integrated over the wrong sets -- and the part was later glued onto a
separate plate with `Th = ThBase + ThCad`, an interface FreeFEM never merges
(zero-width crack: no conduction, no mechanical bond). Both failure modes are
structurally impossible through the delegated path below.

build_parametric_wall() is kept for API compatibility and now delegates to
mesh/structured_conforming.generate_waam_mesh (same geometry, correct labels)
instead of building a second, differently-labelled Gmsh model.
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

# Boundary label set shared with mesh/stl_pipeline.py and the solver.
LABEL_BOTTOM = 1
LABEL_TOP = 5
LABEL_SIDE = 2


def import_cad_and_mesh(
    cad_file: str,
    output_mesh: str,
    mesh_size_min: float = 1.0,
    mesh_size_max: float = 5.0,
    substrate_z_max: float = 15.0,
    deposit_z_max: Optional[float] = None,
    refine_deposit: bool = True,
    deposit_mesh_factor: float = 0.5,
    verbose: bool = True,
    target_nodes: int = 2000,
    layer_height: float = 3.0,
    sub_hz_mm: Optional[float] = None,
    sub_lx_mm: Optional[float] = None,
    sub_ly_mm: Optional[float] = None,
) -> dict:
    """
    Import a CAD file (STEP/IGES/BREP) and generate a 3D tetrahedral mesh.

    The OCC import only extracts the part SKIN to a temporary STL; all sizing,
    unit handling, substrate fusion, labelling and failure reporting are owned
    by mesh/stl_pipeline.stl_to_freefem_mesh (documented above). The legacy
    mesh_size_min/max, refine_deposit and deposit_mesh_factor arguments are
    accepted for compatibility and mapped onto layer_height/target_nodes.

    sub_hz_mm MUST match the solver's hz_base (default 5 mm): the imported
    substrate_z_max default (15 mm) predates that and is only kept when no
    explicit sub_hz_mm is given.

    Returns:
        Dictionary with mesh statistics and label assignments
    """
    cad_path = Path(cad_file)
    if not cad_path.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_file}")

    suffix = cad_path.suffix.lower()
    supported = {'.step', '.stp', '.iges', '.igs', '.brep'}
    if suffix not in supported:
        raise ValueError(f"Unsupported format '{suffix}'. Use: {sorted(supported)}")

    # The substrate plate inside the fused domain uses the solver's hz_base.
    # The legacy substrate_z_max default (15 mm) is honoured only when the
    # caller did not pass an explicit sub_hz_mm.
    plate_hz = float(sub_hz_mm) if sub_hz_mm else float(substrate_z_max)
    if sub_hz_mm is None and abs(plate_hz - 5.0) > 1e-9 and verbose:
        print(f"[CAD-IMPORT] NOTE: substrate plate {plate_hz:.1f} mm differs "
              f"from the solver default hz_base (5 mm). Pass sub_hz_mm "
              f"explicitly to keep the mesh and the .edp in agreement.")

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("waam_part")
    try:
        # OCC import; the part skin is triangulated and handed to the fused
        # domain mesher below (which owns repair, scaling and labelling).
        gmsh.model.occ.importShapes(str(cad_path))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            raise RuntimeError("No 3D volumes found in CAD file")

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
            volumes[0][0], volumes[0][1])
        for vol in volumes[1:]:
            x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(vol[0], vol[1])
            xmin, ymin, zmin = min(xmin, x0), min(ymin, y0), min(zmin, z0)
            xmax, ymax, zmax = max(xmax, x1), max(ymax, y1), max(zmax, z1)
        if verbose:
            print(f"Bounding box: [{xmin:.1f},{ymin:.1f},{zmin:.1f}] "
                  f"to [{xmax:.1f},{ymax:.1f},{zmax:.1f}] (CAD units)")

        # A moderate skin triangulation; the fused-domain mesher re-meshes the
        # volume to its own sizing afterwards, so this only needs to capture
        # the shape, not resolve the physics.
        skin_lc = max(float(mesh_size_max), float(layer_height) * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(float(mesh_size_min), 0.2))
        gmsh.option.setNumber("Mesh.MeshSizeMax", skin_lc)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        tmp_stl = str(Path(output_mesh).with_suffix("")) + "_cad_skin_tmp.stl"
        gmsh.model.mesh.generate(2)
        gmsh.write(tmp_stl)
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

    from mesh.stl_pipeline import stl_to_freefem_mesh
    try:
        result = stl_to_freefem_mesh(
            tmp_stl, output_mesh,
            target_nodes=int(target_nodes),
            layer_height=float(layer_height),
            sub_hz_mm=plate_hz,
            sub_lx_mm=sub_lx_mm, sub_ly_mm=sub_ly_mm,
        )
    finally:
        try:
            if os.path.exists(tmp_stl):
                os.remove(tmp_stl)
        except Exception:
            pass

    result["cad_file"] = str(cad_path)
    result["format"] = suffix.upper()[1:]
    result["cad_bbox"] = {"x": [xmin, xmax], "y": [ymin, ymax], "z": [zmin, zmax]}
    if verbose:
        print(f"CAD domain mesh: {result['n_nodes']:,} nodes, "
              f"{result['n_tets']:,} tets -> {output_mesh}")
        print(f"Boundary labels: {result.get('labels')}")
    return result


def build_parametric_wall(
    output_mesh: str,
    lx: float = 150.0,
    ly_base: float = 150.0,
    hz_base: float = 5.0,
    ty_dep: float = 8.0,
    n_layers: int = 26,
    h_bead: float = 3.0,
    nx_base: int = 30, ny_base: int = 20, nz_base: int = 8,
    nx_dep: int = 30, ny_dep: int = 6, nz_dep: int = 26,
    verbose: bool = True
) -> dict:
    """
    Build a parametric WAAM wall mesh (base plate + deposit zone).

    All dimensions are millimetres, matching the historical signature. The mesh
    itself is built by mesh/structured_conforming.generate_waam_mesh -- one
    conformal grid with shared interface nodes and solver-compatible boundary
    labels -- instead of a second Gmsh model whose private label scheme
    (volumes 1/2, surfaces 10/11/12) matched nothing the solver integrates.

    Returns:
        Dictionary with mesh statistics and label assignments
    """
    from mesh.structured_conforming import generate_waam_mesh
    hz_dep = float(n_layers) * float(h_bead) / 1000.0
    info = generate_waam_mesh(
        output_mesh,
        lx=float(lx) / 1000.0, ly_base=float(ly_base) / 1000.0,
        hz_base=float(hz_base) / 1000.0, ty_dep=float(ty_dep) / 1000.0,
        hz_dep=hz_dep,
        nx_dep=int(nx_dep), ny_dep=int(ny_dep), nz_dep=int(nz_dep),
        nx_base=int(nx_base), ny_base=int(ny_base), nz_base=int(nz_base),
    )
    stats = {
        "output_mesh": str(output_mesh),
        "n_nodes": info["n_nodes"],
        "n_tets": info["n_tets"],
        "boundary_faces_by_label": info.get("boundary_faces_by_label", {}),
        "geometry": {
            "lx": lx, "ly_base": ly_base, "hz_base": hz_base,
            "ty_dep": ty_dep, "n_layers": n_layers, "h_bead": h_bead,
            "hz_dep": hz_dep * 1000.0,
        },
    }
    if verbose:
        print(f"Parametric wall mesh: {info['n_nodes']:,} nodes, "
              f"{info['n_tets']:,} tets -> {output_mesh}")
        print(f"Boundary labels: {stats['boundary_faces_by_label']}")
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cad_importer.py <file.step> [output.mesh]")
        print("  python cad_importer.py --parametric [output.mesh]")
        sys.exit(1)

    if sys.argv[1] == "--parametric":
        out = sys.argv[2] if len(sys.argv) > 2 else "waam_wall.mesh"
        stats = build_parametric_wall(out)
        print(json.dumps(stats, indent=2))
    else:
        cad = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else cad.rsplit('.', 1)[0] + ".mesh"
        stats = import_cad_and_mesh(cad, out)
        print(json.dumps(stats, indent=2))
