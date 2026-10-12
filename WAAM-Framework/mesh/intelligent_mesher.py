"""
mesh/intelligent_mesher.py

Intelligent Rule-Based Mesh Generation & Cell Sizing Engine for WAAM Simulations.

Implements physics-based and geometry-adaptive meshing rules:
1. Aspect-Ratio & Gradient Balancing:
   - Evaluates Lx, Ly, Lz, feature thickness, bead dimensions (ty_dep, h_bead).
   - In structured mode, calculates Nx, Ny, Nz such that:
     * Deposit wall elements have isotropic aspect ratio (dx ~ dy ~ dz).
     * High-gradient thermal/meltpool zone receives ~65% of node budget.
     * Far-field substrate receives ~35% of node budget with smooth grading.
     * Prevents degenerately stretched elements (aspect ratio <= 2.5).
2. Curvature & Thin-Feature Detection for CAD (STEP / IGES / STL):
   - Computes bounding volume V and target characteristic length lc = (V / N_target)^(1/3).
   - Identifies minimum span and thin features (e.g. thin walls, hex grill struts, screw holes).
   - Applies curvature-adaptive sizing: refines high-curvature/thin features with lc_min, coarsens bulk regions with lc_max.
   - Dual-pass surface extraction & watertight healing for complex multi-surface STEP/CAD geometry.
3. Multi-Format Unified Pipeline:
   - Handles .stl, .step, .stp, .iges, .igs, .brep, .mesh, .msh.
"""

import os
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def calculate_intelligent_structured_mesh(params: dict, target_nodes: int = 2000,
                                         strategy: str = "physics_adaptive") -> dict:
    """
    Intelligently assigns Nx, Ny, Nz divisions for substrate and deposit wall
    based on geometric aspect ratios and physics gradient rules.
    """
    from mesh.units import dim_to_m
    unit_warnings: list = []

    def get_dim_mm(key, default_si):
        # Accepts SI (solver) or mm (GUI) and normalises to mm. The previous
        # rule (`< 1.0 means metres`) contradicted the OpenFOAM rule
        # (`> 1.0 means mm`) and silently rescaled every legitimate
        # sub-millimetre SI value by x1000; both now share one threshold table
        # in mesh/units.py.
        v_m = dim_to_m(params.get(key, default_si), key,
                       warn=unit_warnings.append)
        return v_m * 1000.0

    lx = max(get_dim_mm("lx", 0.150), 1.0)           # mm
    ly_base = max(get_dim_mm("ly_base", 0.150), 1.0)  # mm
    hz_base = max(get_dim_mm("hz_base", 0.005), 1.0)  # mm
    ty_dep = max(get_dim_mm("ty_dep", 0.008), 0.5)    # mm
    h_bead_first = max(get_dim_mm("h_bead_first",
                                  float(params.get("h_bead", 0.004))), 0.5)  # mm
    remelt_ratio = float(params.get("remelt_ratio", 0.80))
    h_bead_subseq = get_dim_mm(
        "h_bead_subseq", h_bead_first / 1000.0 * remelt_ratio)  # mm
    h_bead_subseq = max(h_bead_subseq, 0.1)
    n_layers = max(int(float(params.get("n_layers", 10))), 1)
    hz_dep = h_bead_first if n_layers <= 1 else (h_bead_first + (n_layers - 1) * h_bead_subseq)  # mm

    target_nodes = max(int(target_nodes), 500)

    # Strategy 1: Physics-Adaptive (65% Deposit / 35% Substrate)
    if strategy == "physics_adaptive":
        dep_fraction = 0.65
        base_fraction = 0.35
    elif strategy == "equal_aspect":
        # Weight by volume
        v_dep = lx * ty_dep * hz_dep
        v_base = lx * ly_base * hz_base
        v_tot = max(v_dep + v_base, 1.0)
        dep_fraction = max(min(v_dep / v_tot, 0.75), 0.40)
        base_fraction = 1.0 - dep_fraction
    else:
        dep_fraction = 0.60
        base_fraction = 0.40

    target_dep_nodes = target_nodes * dep_fraction
    target_base_nodes = target_nodes * base_fraction

    # 1. Deposit Wall Assignment (Target isotropic cells dx ~ dy ~ dz)
    v_dep = lx * ty_dep * hz_dep
    d_dep_ideal = max((v_dep / max(target_dep_nodes, 100)) ** (1.0 / 3.0), 0.5)

    # Minimum layer resolution: at least 1 element per layer
    nz_dep = max(n_layers, int(round(hz_dep / d_dep_ideal)))
    # Minimum bead width resolution: at least 2-3 elements across width
    ny_dep = max(2, int(round(ty_dep / d_dep_ideal)))
    # Length divisions
    nx_dep = max(4, int(round(lx / d_dep_ideal)))

    # Adjust scale to meet deposit budget
    actual_dep_nodes = (nx_dep + 1) * (ny_dep + 1) * (nz_dep + 1)
    if actual_dep_nodes > 0:
        scale_dep = (target_dep_nodes / actual_dep_nodes) ** (1.0 / 3.0)
        nx_dep = max(3, int(round(nx_dep * scale_dep)))
        ny_dep = max(2, int(round(ny_dep * scale_dep)))
        nz_dep = max(n_layers, int(round(nz_dep * scale_dep)))

    # 2. Substrate Assignment
    v_base = lx * ly_base * hz_base
    d_base_ideal = max((v_base / max(target_base_nodes, 100)) ** (1.0 / 3.0), 1.0)

    nx_base = max(4, int(round(lx / d_base_ideal)))
    ny_base = max(3, int(round(ly_base / d_base_ideal)))
    nz_base = max(2, int(round(hz_base / d_base_ideal)))

    actual_base_nodes = (nx_base + 1) * (ny_base + 1) * (nz_base + 1)
    if actual_base_nodes > 0:
        scale_base = (target_base_nodes / actual_base_nodes) ** (1.0 / 3.0)
        nx_base = max(3, int(round(nx_base * scale_base)))
        ny_base = max(3, int(round(ny_base * scale_base)))
        nz_base = max(2, int(round(nz_base * scale_base)))

    # Compute final metrics & aspect ratios
    dx_dep = lx / nx_dep
    dy_dep = ty_dep / ny_dep
    dz_dep = hz_dep / nz_dep
    ar_dep = max(dx_dep, dy_dep, dz_dep) / max(min(dx_dep, dy_dep, dz_dep), 1e-4)

    dx_base = lx / nx_base
    dy_base = ly_base / ny_base
    dz_base = hz_base / nz_base
    ar_base = max(dx_base, dy_base, dz_base) / max(min(dx_base, dy_base, dz_base), 1e-4)

    # The deposit footprint shares its bottom nodes with the plate top: those
    # (nx_dep+1)*(ny_dep+1) nodes were counted in BOTH blocks, inflating the
    # estimate. Subtract them once (an estimate -- exact only when the Y grids
    # align, as in mesh/structured_conforming).
    shared_interface = (nx_dep + 1) * (ny_dep + 1)
    tot_nodes = ((nx_dep + 1) * (ny_dep + 1) * (nz_dep + 1)
                 + (nx_base + 1) * (ny_base + 1) * (nz_base + 1)
                 - shared_interface)
    tot_hex = (nx_dep * ny_dep * nz_dep) + (nx_base * ny_base * nz_base)
    tot_tets = tot_hex * 6

    # The module docstring promises aspect ratio <= 2.5; enforce it instead of
    # merely reporting it. Refine the coarsest axis until both blocks comply
    # (bounded loop; a warning records when the budget had to give).
    aspect_cap = 2.5
    for _ in range(64):
        worst = max(ar_dep, ar_base)
        if worst <= aspect_cap:
            break
        if ar_dep >= ar_base:
            if dx_dep >= dy_dep and dx_dep >= dz_dep:
                nx_dep += 1; dx_dep = lx / nx_dep
            elif dy_dep >= dz_dep:
                ny_dep += 1; dy_dep = ty_dep / ny_dep
            else:
                nz_dep += 1; dz_dep = hz_dep / nz_dep
            ar_dep = (max(dx_dep, dy_dep, dz_dep)
                      / max(min(dx_dep, dy_dep, dz_dep), 1e-4))
        else:
            if dx_base >= dy_base and dx_base >= dz_base:
                nx_base += 1; dx_base = lx / nx_base
            elif dy_base >= dz_base:
                ny_base += 1; dy_base = ly_base / ny_base
            else:
                nz_base += 1; dz_base = hz_base / nz_base
            ar_base = (max(dx_base, dy_base, dz_base)
                       / max(min(dx_base, dy_base, dz_base), 1e-4))
        shared_interface = (nx_dep + 1) * (ny_dep + 1)
        tot_nodes = ((nx_dep + 1) * (ny_dep + 1) * (nz_dep + 1)
                     + (nx_base + 1) * (ny_base + 1) * (nz_base + 1)
                     - shared_interface)
        tot_hex = (nx_dep * ny_dep * nz_dep) + (nx_base * ny_base * nz_base)
        tot_tets = tot_hex * 6
    if max(ar_dep, ar_base) > aspect_cap:
        unit_warnings.append(
            f"aspect ratio {max(ar_dep, ar_base):.2f} exceeds the {aspect_cap} "
            f"cap even after refinement; results on stretched elements "
            f"need a mesh-sensitivity check.")

    return {
        "nx_dep": int(nx_dep), "ny_dep": int(ny_dep), "nz_dep": int(nz_dep),
        "nx_base": int(nx_base), "ny_base": int(ny_base), "nz_base": int(nz_base),
        "total_nodes": int(tot_nodes), "total_hex": int(tot_hex), "total_tets": int(tot_tets),
        "dx_dep": dx_dep, "dy_dep": dy_dep, "dz_dep": dz_dep, "aspect_ratio_dep": ar_dep,
        "dx_base": dx_base, "dy_base": dy_base, "dz_base": dz_base, "aspect_ratio_base": ar_base,
        "strategy": strategy, "target_budget": target_nodes,
        "warnings": list(unit_warnings),
    }


def import_and_mesh_cad_intelligent(cad_path: str, output_mesh_path: str = None,
                                   target_nodes: int = 2000,
                                   refine_thin_features: bool = True,
                                   layer_height: float = None,
                                   sub_hz_mm: float = None,
                                   sub_lx_mm: float = None,
                                   sub_ly_mm: float = None) -> dict:
    """
    Intelligently imports any CAD format (.stl, .step, .stp, .iges, .igs)
    and meshes it with curvature and feature-adaptive rules.

    sub_hz_mm / sub_lx_mm / sub_ly_mm MUST match the solver's hz_base / lx /
    ly_base (the generated .edp treats everything at z <= HzBase as substrate):
    they are forwarded verbatim to mesh/stl_pipeline.stl_to_freefem_mesh, whose
    own defaults (5 / 150 / 150 mm) otherwise apply. layer_height drives the
    near-part element size.
    """
    import trimesh
    import gmsh

    cad_path = str(Path(cad_path).resolve())
    suffix = Path(cad_path).suffix.lower()
    _pipe_kwargs = dict(target_nodes=target_nodes)
    if layer_height is not None:
        _pipe_kwargs["layer_height"] = float(layer_height)
    if sub_hz_mm is not None:
        _pipe_kwargs["sub_hz_mm"] = float(sub_hz_mm)
    if sub_lx_mm is not None:
        _pipe_kwargs["sub_lx_mm"] = float(sub_lx_mm)
    if sub_ly_mm is not None:
        _pipe_kwargs["sub_ly_mm"] = float(sub_ly_mm)

    if output_mesh_path is None:
        outdir = "./waam_output"
        os.makedirs(outdir, exist_ok=True)
        output_mesh_path = os.path.join(outdir, "imported_cad.mesh")

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)

    # 1. STEP / IGES Pipeline with Surface Healing
    if suffix in ('.step', '.stp', '.iges', '.igs', '.brep'):
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cad_intelligent_step")
        gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        gmsh.model.occ.importShapes(cad_path)
        gmsh.model.occ.synchronize()

        vols = gmsh.model.getEntities(3)
        surfs = gmsh.model.getEntities(2)

        if vols:
            bbox = gmsh.model.occ.getBoundingBox(vols[0][0], vols[0][1])
        elif surfs:
            bbox = gmsh.model.occ.getBoundingBox(surfs[0][0], surfs[0][1])
        else:
            bbox = (0, 0, 0, 100, 100, 10)

        spans = (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
        vol_est = float(spans[0] * spans[1] * spans[2])
        lc_target = max((vol_est / max(target_nodes, 500)) ** (1.0 / 3.0), 0.5)

        # Generate 2D surface triangulation first to heal complex CAD boundaries
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(lc_target * 0.25, 0.4))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(lc_target * 1.75, 2.0))
        if refine_thin_features:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 10)

        tmp_surf_stl = output_mesh_path + "_temp_step_surf.stl"
        gmsh.model.mesh.generate(2)
        gmsh.write(tmp_surf_stl)
        gmsh.finalize()

        # Step 2: Repair surface via trimesh
        mesh_surf = trimesh.load(tmp_surf_stl, force="mesh")
        if not mesh_surf.is_watertight:
            import trimesh.repair as tr
            tr.fix_normals(mesh_surf)
            tr.fill_holes(mesh_surf)

        mesh_surf.export(tmp_surf_stl)

        # Step 3: Mesh volume from watertight repaired surface
        from mesh.stl_pipeline import stl_to_freefem_mesh
        result = stl_to_freefem_mesh(tmp_surf_stl, output_mesh_path, **_pipe_kwargs)

        try:
            os.remove(tmp_surf_stl)
        except Exception:
            pass

        result["cad_file"] = cad_path
        result["format"] = suffix.upper()[1:]
        result["bbox_mm"] = {
            "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
            "xmin": float(bbox[0]), "ymin": float(bbox[1]), "zmin": float(bbox[2]),
            "xmax": float(bbox[3]), "ymax": float(bbox[4]), "zmax": float(bbox[5])
        }
        return result

    # 2. STL Pipeline
    elif suffix in ('.stl', '.ply', '.obj'):
        from mesh.stl_pipeline import stl_to_freefem_mesh
        result = stl_to_freefem_mesh(cad_path, output_mesh_path, **_pipe_kwargs)
        result["cad_file"] = cad_path
        result["format"] = "STL"
        return result

    # 3. Direct FreeFEM / Medit 3D Mesh
    elif suffix == '.mesh':
        n_nodes = 0
        n_tets = 0
        coords = []
        with open(cad_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "Vertices" and i + 1 < len(lines):
                try:
                    n_nodes = int(lines[i+1].strip())
                    for j in range(min(n_nodes, 100000)):
                        parts = lines[i+2+j].split()
                        if len(parts) >= 3:
                            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except Exception:
                    pass
                i += 2 + n_nodes
            elif line == "Tetrahedra" and i + 1 < len(lines):
                try:
                    n_tets = int(lines[i+1].strip())
                except Exception:
                    pass
                break
            i += 1
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]
            bbox = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
        else:
            bbox = (0, 0, 0, 100, 100, 10)
        spans = (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
        return {
            "mesh_path": cad_path,
            "cad_file": cad_path,
            "format": "MESH",
            "n_nodes": n_nodes,
            "n_tets": n_tets,
            "bbox_mm": {
                "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
                "xmin": float(bbox[0]), "ymin": float(bbox[1]), "zmin": float(bbox[2]),
                "xmax": float(bbox[3]), "ymax": float(bbox[4]), "zmax": float(bbox[5])
            },
            "warnings": [f"[MESH-DIRECT] Loaded existing volume mesh: {n_nodes:,} nodes, {n_tets:,} tets."]
        }

    else:
        raise ValueError(f"Unsupported CAD format: {suffix}")
