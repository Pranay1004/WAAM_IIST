"""
CAD importer: STEP/IGES → Gmsh → FreeFEM .mesh

Uses the Gmsh Python API to import CAD geometry, mesh it with
3D tetrahedra, assign physical groups, and export to INRIA .mesh
format for FreeFEM's readmesh3().
"""

import gmsh
import json
import math
from pathlib import Path
from typing import Optional


def import_cad_and_mesh(
    cad_file: str,
    output_mesh: str,
    mesh_size_min: float = 1.0,
    mesh_size_max: float = 5.0,
    substrate_z_max: float = 15.0,
    deposit_z_max: Optional[float] = None,
    refine_deposit: bool = True,
    deposit_mesh_factor: float = 0.5,
    verbose: bool = True
) -> dict:
    """
    Import a CAD file (STEP/IGES/BREP) and generate a 3D tetrahedral mesh.

    Args:
        cad_file: Path to STEP, IGES, or BREP file
        output_mesh: Output .mesh file path
        mesh_size_min: Minimum element size [mm]
        mesh_size_max: Maximum element size [mm]
        substrate_z_max: Z-coordinate separating substrate from deposit [mm]
        deposit_z_max: Z-coordinate of deposit top (auto-detected if None)
        refine_deposit: Whether to refine mesh in the deposit region
        deposit_mesh_factor: Mesh size factor for deposit region (0-1)
        verbose: Print progress messages

    Returns:
        Dictionary with mesh statistics and label assignments
    """
    cad_path = Path(cad_file)
    if not cad_path.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_file}")

    suffix = cad_path.suffix.lower()
    supported = {'.step', '.stp', '.iges', '.igs', '.brep'}
    if suffix not in supported:
        raise ValueError(f"Unsupported format '{suffix}'. Use: {supported}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("waam_part")

    # Import CAD
    if verbose:
        print(f"Importing CAD: {cad_file}")

    if suffix in ('.step', '.stp'):
        gmsh.model.occ.importShapes(str(cad_path))
    elif suffix in ('.iges', '.igs'):
        gmsh.model.occ.importShapes(str(cad_path))
    elif suffix == '.brep':
        gmsh.model.occ.importShapes(str(cad_path))

    gmsh.model.occ.synchronize()

    # Get bounding box
    volumes = gmsh.model.getEntities(dim=3)
    if not volumes:
        gmsh.finalize()
        raise RuntimeError("No 3D volumes found in CAD file")

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
        volumes[0][0], volumes[0][1]
    )
    for vol in volumes[1:]:
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(vol[0], vol[1])
        xmin, ymin, zmin = min(xmin, x0), min(ymin, y0), min(zmin, z0)
        xmax, ymax, zmax = max(xmax, x1), max(ymax, y1), max(zmax, z1)

    if deposit_z_max is None:
        deposit_z_max = zmax

    if verbose:
        print(f"Bounding box: [{xmin:.1f},{ymin:.1f},{zmin:.1f}] "
              f"to [{xmax:.1f},{ymax:.1f},{zmax:.1f}] mm")

    # Set mesh sizes
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    # Refine deposit region with a mesh size field
    if refine_deposit:
        box_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box_field, "VIn", mesh_size_min * deposit_mesh_factor)
        gmsh.model.mesh.field.setNumber(box_field, "VOut", mesh_size_max)
        gmsh.model.mesh.field.setNumber(box_field, "XMin", xmin)
        gmsh.model.mesh.field.setNumber(box_field, "XMax", xmax)
        gmsh.model.mesh.field.setNumber(box_field, "YMin", ymin)
        gmsh.model.mesh.field.setNumber(box_field, "YMax", ymax)
        gmsh.model.mesh.field.setNumber(box_field, "ZMin", substrate_z_max)
        gmsh.model.mesh.field.setNumber(box_field, "ZMax", deposit_z_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(box_field)

    # Assign physical groups based on z-coordinate
    # Surfaces: detect boundary faces and classify by position
    surfaces = gmsh.model.getEntities(dim=2)

    bottom_surfaces = []
    top_surfaces = []
    side_surfaces = []

    for surf in surfaces:
        sx0, sy0, sz0, sx1, sy1, sz1 = gmsh.model.occ.getBoundingBox(surf[0], surf[1])
        face_zmin = sz0
        face_zmax = sz1
        face_dz = face_zmax - face_zmin

        # Bottom face: z ≈ zmin and flat
        if abs(face_zmin - zmin) < 0.1 and abs(face_dz) < 0.1:
            bottom_surfaces.append(surf[1])
        # Top face: z ≈ zmax and flat
        elif abs(face_zmax - zmax) < 0.1 and abs(face_dz) < 0.1:
            top_surfaces.append(surf[1])
        else:
            side_surfaces.append(surf[1])

    # Create physical groups
    labels = {}
    if bottom_surfaces:
        tag = gmsh.model.addPhysicalGroup(2, bottom_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, tag, "Bottom_Clamp")
        labels[1] = "Bottom_Clamp"

    if top_surfaces:
        tag = gmsh.model.addPhysicalGroup(2, top_surfaces, tag=2)
        gmsh.model.setPhysicalName(2, tag, "Top_Surface")
        labels[2] = "Top_Surface"

    if side_surfaces:
        tag = gmsh.model.addPhysicalGroup(2, side_surfaces, tag=3)
        gmsh.model.setPhysicalName(2, tag, "Side_Convection")
        labels[3] = "Side_Convection"

    # Volume physical group
    vol_tags = [v[1] for v in volumes]
    vol_group = gmsh.model.addPhysicalGroup(3, vol_tags, tag=10)
    gmsh.model.setPhysicalName(3, vol_group, "Domain")
    labels[10] = "Domain"

    # Generate mesh
    if verbose:
        print("Generating 3D mesh...")
    gmsh.model.mesh.generate(3)

    # Get mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, _, _ = gmsh.model.mesh.getElements(dim=3)
    n_nodes = len(node_tags)
    n_elements = sum(len(tags) for _, tags, _ in
                     [gmsh.model.mesh.getElements(dim=3)])

    if verbose:
        print(f"Mesh: {n_nodes} nodes, {n_elements} tetrahedra")

    # Export
    out_path = Path(output_mesh)
    if out_path.suffix == '.mesh':
        gmsh.write(str(out_path))
    elif out_path.suffix == '.msh':
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out_path))
    elif out_path.suffix == '.vtu':
        gmsh.write(str(out_path))
    else:
        # Default to .mesh
        gmsh.write(str(out_path.with_suffix('.mesh')))

    gmsh.finalize()

    stats = {
        "cad_file": str(cad_path),
        "output_mesh": str(out_path),
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "bounding_box": {
            "x": [xmin, xmax], "y": [ymin, ymax], "z": [zmin, zmax]
        },
        "physical_labels": labels,
        "mesh_size_range": [mesh_size_min, mesh_size_max],
        "substrate_z_max": substrate_z_max
    }

    if verbose:
        print(f"Mesh exported to: {out_path}")
        print(f"Physical labels: {labels}")

    return stats


def build_parametric_wall(
    output_mesh: str,
    lx: float = 150.0,
    ly_base: float = 150.0,
    hz_base: float = 15.0,
    ty_dep: float = 8.0,
    n_layers: int = 26,
    h_bead: float = 3.0,
    nx_base: int = 30, ny_base: int = 20, nz_base: int = 8,
    nx_dep: int = 30, ny_dep: int = 6, nz_dep: int = 26,
    verbose: bool = True
) -> dict:
    """
    Build a parametric WAAM wall geometry (base plate + deposit zone)
    directly in Gmsh without requiring a CAD file.

    This replicates the FreeFEM cube+movemesh3 approach but with
    proper physical group labels.

    Returns:
        Dictionary with mesh statistics and label assignments
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("waam_wall")

    hz_dep = n_layers * h_bead
    y_centre = ly_base / 2.0
    y_left = y_centre - ty_dep / 2.0
    y_right = y_centre + ty_dep / 2.0

    # Create base plate box
    base = gmsh.model.occ.addBox(0, 0, 0, lx, ly_base, hz_base)

    # Create deposit zone box
    dep = gmsh.model.occ.addBox(0, y_left, hz_base, lx, ty_dep, hz_dep)

    # Fragment (merge shared faces)
    out_dimtags, out_map = gmsh.model.occ.fragment(
        [(3, base)], [(3, dep)]
    )
    gmsh.model.occ.synchronize()

    # Identify volumes
    all_vols = gmsh.model.getEntities(dim=3)

    substrate_vols = []
    deposit_vols = []
    for vol in all_vols:
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(vol[0], vol[1])
        z_mid = (z0 + z1) / 2.0
        if z_mid < hz_base + 0.1:
            substrate_vols.append(vol[1])
        else:
            deposit_vols.append(vol[1])

    # Physical volumes
    if substrate_vols:
        gmsh.model.addPhysicalGroup(3, substrate_vols, tag=1)
        gmsh.model.setPhysicalName(3, 1, "Substrate")
    if deposit_vols:
        gmsh.model.addPhysicalGroup(3, deposit_vols, tag=2)
        gmsh.model.setPhysicalName(3, 2, "Deposit")

    # Identify boundary surfaces
    all_surfs = gmsh.model.getEntities(dim=2)
    bottom_s, top_s, side_s = [], [], []

    for surf in all_surfs:
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(surf[0], surf[1])
        dz = abs(z1 - z0)
        if abs(z0) < 0.01 and dz < 0.01:
            bottom_s.append(surf[1])
        elif abs(z1 - (hz_base + hz_dep)) < 0.1 and dz < 0.1:
            top_s.append(surf[1])
        elif abs(z0 - hz_base) < 0.01 and dz < 0.01:
            pass  # Internal interface
        else:
            # Check if it's a boundary surface (not internal)
            up, down = gmsh.model.getAdjacencies(surf[0], surf[1])
            if len(up) <= 1:  # Only one adjacent volume = boundary
                side_s.append(surf[1])

    if bottom_s:
        gmsh.model.addPhysicalGroup(2, bottom_s, tag=10)
        gmsh.model.setPhysicalName(2, 10, "Bottom_Clamp")
    if top_s:
        gmsh.model.addPhysicalGroup(2, top_s, tag=11)
        gmsh.model.setPhysicalName(2, 11, "Top_Surface")
    if side_s:
        gmsh.model.addPhysicalGroup(2, side_s, tag=12)
        gmsh.model.setPhysicalName(2, 12, "Side_Surfaces")

    # Mesh sizing
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max(lx/nx_base, ly_base/ny_base, hz_base/nz_base))

    # Generate mesh
    gmsh.model.mesh.generate(3)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)

    # Export
    out_path = Path(output_mesh)
    if out_path.suffix == '.msh':
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(out_path))

    gmsh.finalize()

    stats = {
        "output_mesh": str(out_path),
        "n_nodes": n_nodes,
        "geometry": {
            "lx": lx, "ly_base": ly_base, "hz_base": hz_base,
            "ty_dep": ty_dep, "n_layers": n_layers, "h_bead": h_bead,
            "hz_dep": hz_dep
        }
    }

    if verbose:
        print(f"Parametric wall mesh: {n_nodes} nodes → {out_path}")

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
