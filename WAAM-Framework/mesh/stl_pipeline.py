"""
mesh/stl_pipeline.py

STL/STEP -> FreeFEM volume-mesh pipeline (Gmsh OCCT kernel).

WHAT THIS PRODUCES AND WHY
--------------------------
The solver needs ONE conforming tetrahedral mesh of the WHOLE domain --
substrate plate plus deposited part -- with boundary labels the generated
.edp understands (see mesh/structured_conforming._BOUNDARY_LABEL, set B):

    1 -> z MIN  (substrate underside: hBot transfer + isostatic fixture spring)
    5 -> z MAX  (top surface)
    2/3/4/6 -> remaining flanks (all treated identically by the solver)

The previous implementation meshed ONLY the imported shell and let the .edp
glue it onto a separate plate with `Th = ThBase + ThCad`. Two things were
wrong with that:

  * FreeFEM's `+` concatenates WITHOUT merging nodes (verified: the combined
    node count is exactly the sum of the two), so the plate/part interface was
    a zero-width crack -- no conduction into the plate, no mechanical bond.
  * When shell watertightness failed, the mesher silently fell back to a 2D
    surface mesh, reported the triangle count as "n_tets", and wrote that
    surface to the .mesh path. readmesh3 then ABORTS (SIGBUS, exit 138) long
    after slicing and meshing had already been paid for.

Now the substrate is built as a solid OCC box inside Gmsh, fused with the
imported part, meshed once as a single volume, and the boundary surfaces are
tagged with explicit physical groups before export. If the volume cannot be
formed the call RAISES instead of writing an unusable file.
"""

import os
import sys
import json
import math
import subprocess
from pathlib import Path


# Boundary label set (must stay in lock-step with mesh/structured_conforming.py
# _BOUNDARY_LABEL set B and codegen/edp_builder.py). The solver only
# distinguishes label 1 (underside) from labels 2..6 (everything else), so all
# remaining surfaces may safely share tag 2.
LABEL_BOTTOM = 1
LABEL_TOP = 5
LABEL_SIDE = 2


def _stl_to_freefem_mesh_in_process(stl_path: str, output_mesh_path: str = None,
                                    target_nodes: int = 2000,
                                    layer_height: float = 3.0,
                                    sub_hz_mm: float = None,
                                    sub_lx_mm: float = None,
                                    sub_ly_mm: float = None,
                                    in_process: bool = False) -> dict:
    """
    Core mesher. Builds substrate + part as one conforming volume mesh.

    sub_hz_mm: substrate thickness [mm]. MUST match the solver's `hz_base`
        (the generated .edp computes HzBase from the same parameter and marks
        everything at z <= HzBase as substrate). None -> 5.0 mm, the builder
        default.
    sub_lx_mm / sub_ly_mm: substrate plan dimensions [mm]. MUST match the
        solver's lx / ly_base, otherwise the torch would traverse past the end
        of the meshed plate. None -> 150 mm.
    """
    import trimesh
    import gmsh

    warnings = []
    stl_path = str(Path(stl_path).resolve())
    # Substrate geometry: the plate the .edp models on top of which the CAD
    # part is deposited. Default 5 mm matches EDPBuilder.default_params().
    SUB_HZ = float(sub_hz_mm) if sub_hz_mm else 5.0
    SUB_LX = float(sub_lx_mm) if sub_lx_mm else 150.0
    SUB_LY = float(sub_ly_mm) if sub_ly_mm else 150.0
    BUILD_HEIGHT = 150.0

    if output_mesh_path is None or not str(output_mesh_path).strip():
        output_mesh_path = str(Path(stl_path).with_suffix(".mesh"))
    # A bare filename has an empty dirname, and os.makedirs("") raises
    # FileNotFoundError -- resolve it first.
    output_mesh_path = str(Path(output_mesh_path).resolve())
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)

    # Step 1: Load + repair surface via trimesh
    mesh_surf = trimesh.load_mesh(stl_path, force="mesh")
    if not mesh_surf.is_watertight:
        warnings.append("[STL-REPAIR] Surface not watertight — attempting auto-repair.")
        import trimesh.repair as tr
        tr.fix_normals(mesh_surf)
        tr.fill_holes(mesh_surf)
        if not mesh_surf.is_watertight:
            warnings.append(
                "[STL-REPAIR] Surface still not watertight after repair. "
                "Volume meshing will fail unless Gmsh can stitch the shell.")

    # Detect STL units: convert only a clearly-metre-scale model (span < 1 unit)
    # to mm. A WAAM coupon in mm can legitimately be < 1 m across, so the
    # heuristic is a guess, never a silent fact -- it is always reported.
    temp_bounds = mesh_surf.bounds
    temp_spans = temp_bounds[1] - temp_bounds[0]
    max_span = max(temp_spans)
    if max_span < 1.0:
        mesh_surf.apply_scale(1000.0)
        warnings.append(
            f"[STL-UNITS] Span {max_span:.4f} < 1.0 -- treated as METRES and "
            f"scaled x1000 to mm. Verify the CAD export units.")
    elif max_span > 1000.0:
        warnings.append(f"[STL-UNITS] Max span {max_span:.1f} - assumed mm.")

    # Auto-scale to fit the build envelope (never silent: returned as
    # scale_applied and consumed by callers that report mass/energy).
    V_ENCLOSURE = SUB_LX * SUB_LY * BUILD_HEIGHT
    MAX_OCCUPIED_VOL = 0.75 * V_ENCLOSURE
    bbox = mesh_surf.bounds
    spans = bbox[1] - bbox[0]
    v_bbox = float(spans[0] * spans[1] * spans[2])
    warnings.append(
        f"[STL-BBOX-RAW] X={spans[0]:.1f}  Y={spans[1]:.1f}  Z={spans[2]:.1f} "
        f"(STL units, Vol={v_bbox:,.0f} mm³)")
    scale_applied = 1.0
    if (spans[0] > SUB_LX or spans[1] > SUB_LY or spans[2] > BUILD_HEIGHT
            or v_bbox > MAX_OCCUPIED_VOL):
        s_vol = (MAX_OCCUPIED_VOL / max(v_bbox, 1e-6)) ** (1.0 / 3.0)
        s_dim = min(SUB_LX / max(spans[0], 1e-6), SUB_LY / max(spans[1], 1e-6),
                    BUILD_HEIGHT / max(spans[2], 1e-6))
        scale_applied = min(s_vol, s_dim)
        mesh_surf.apply_scale(scale_applied)
        warnings.append(
            f"[STL-AUTO-SCALE] Model exceeded the {SUB_LX:.0f}x{SUB_LY:.0f}x{BUILD_HEIGHT:.0f} "
            f"mm envelope. Scaled by {scale_applied:.4f}x so the occupied volume "
            f"stays under 75% of it. Every reported mass/energy figure must be "
            f"corrected by this factor.")

    # Place flush on top of the substrate: centred in X/Y, bottom at z = SUB_HZ.
    bbox = mesh_surf.bounds
    spans = bbox[1] - bbox[0]
    tx = (SUB_LX / 2.0) - (bbox[0][0] + bbox[1][0]) / 2.0
    ty = (SUB_LY / 2.0) - (bbox[0][1] + bbox[1][1]) / 2.0
    tz = SUB_HZ - bbox[0][2]
    mesh_surf.apply_translation([tx, ty, tz])
    # Re-read the bounds AFTER placement: bbox_mm is reported in the frame the
    # solver (and every viewer) actually uses, i.e. flush on the substrate at
    # X/Y = plate centre -- not the raw STL coordinates.
    bbox = mesh_surf.bounds
    spans = bbox[1] - bbox[0]
    warnings.append(
        f"[STL-PLACEMENT] Placed flush on a {SUB_LX:.0f}x{SUB_LY:.0f}x{SUB_HZ:.1f} mm "
        f"substrate at X={SUB_LX/2.0:.1f}, Y={SUB_LY/2.0:.1f} mm (z=[{SUB_HZ:.1f}, "
        f"{bbox[1][2]+tz:.1f}] mm).")
    z_cad_max = float(mesh_surf.bounds[1][2])

    # Coarsen a surface tessellation that is far finer than the volume mesh
    # will ever be. Gmsh's classifySurfaces()/createGeometry() turns every STL
    # vertex into a geometry POINT that CANNOT be removed, so a 20k-triangle
    # export silently forced ~10k surface nodes -- and therefore ~200k tets --
    # no matter how small the requested node budget was. Clustering the
    # vertices onto a grid of the target element size makes the tetrahedron
    # count depend on the BUDGET, not on how finely the CAD was tessellated.
    lc_fine_mm = 0.5 * (float(layer_height) if float(layer_height) > 0 else 3.0)
    mean_edge = (float(mesh_surf.edges_unique_length.mean())
                 if len(getattr(mesh_surf, "edges_unique", ())) else 0.0)
    mesh_surf_full = mesh_surf
    if 0.0 < mean_edge < 0.5 * lc_fine_mm:
        # Not required to be watertight here: vertex clustering routinely leaves
        # pinched vertices. The union with the substrate is the watertightness
        # that matters, and it is retried on the ORIGINAL tessellation below if
        # the simplified one cannot be booleaned.
        simplified = _cluster_decimate(mesh_surf, lc_fine_mm, require_watertight=False)
        if simplified is not mesh_surf:
            warnings.append(
                f"[STL-DECIMATE] Surface tessellation (mean edge {mean_edge:.2f} mm) was "
                f"much finer than the {lc_fine_mm:.2f} mm element size; simplified "
                f"{len(mesh_surf.faces):,} -> {len(simplified.faces):,} triangles so the "
                f"volume mesh follows the node budget instead of the STL density.")
            mesh_surf = simplified

    # Step 2: Geometry = substrate box UNION the imported part.
    #
    # The union is done with trimesh's CSG boolean (manifold backend), NOT inside
    # Gmsh: gmsh.model.occ.importShapes() reads BREP/STEP only ("Unknown file
    # type" for STL), and gmsh.model.mesh.classifySurfaces() does not terminate
    # on genuinely complex parts (measured: >100 s on a genus-28 shell). Both
    # halves are meshed as ONE shell, so the plate/part interface is conformal
    # by construction instead of the zero-width crack that `Th = ThBase +
    # ThCad` produced.
    substrate = None
    export_mesh = None
    # The simplified surface is tried first; if it cannot be booleaned into a
    # watertight solid the ORIGINAL tessellation is retried before giving up.
    surface_candidates = [mesh_surf]
    if mesh_surf is not mesh_surf_full:
        surface_candidates.append(mesh_surf_full)
    try:
        substrate = _box_mesh(SUB_LX, SUB_LY, SUB_HZ)
    except Exception as ex:
        warnings.append(f"[STL-UNION] Substrate construction failed ({ex}).")
        substrate = None
    if substrate is not None:
        for _surf in surface_candidates:
            try:
                union = trimesh.boolean.union([_surf, substrate], engine="manifold")
            except ImportError as ex:
                warnings.append(f"[STL-UNION] CSG backend unavailable ({ex}).")
                break
            except Exception as ex:
                warnings.append(f"[STL-UNION] Boolean union failed ({ex}).")
                continue
            _faces = getattr(union, "faces", None)
            if union is None or _faces is None or len(_faces) == 0:
                warnings.append("[STL-UNION] Boolean union produced an empty mesh.")
                continue
            if not union.is_watertight:
                # A non-watertight union cannot be meshed as a solid; fall back
                # to the next candidate (the original tessellation), then to the
                # part shell only (the caller still gets a solvable volume mesh,
                # just without the plate bonded to it).
                if _surf is not surface_candidates[-1]:
                    warnings.append(
                        "[STL-UNION] Union of the simplified surface is not "
                        "watertight; retrying with the original tessellation.")
                    continue
                warnings.append(
                    "[STL-UNION] Boolean union is not watertight; meshing the part "
                    "shell alone (substrate will be added by the solver's own "
                    "parametric plate, which does NOT bond to it).")
                export_mesh = _surf
                break
            export_mesh = union
            warnings.append(
                f"[STL-UNION] Substrate {SUB_LX:.0f}x{SUB_LY:.0f}x{SUB_HZ:.1f} mm "
                f"fused with the part into one watertight solid "
                f"({len(union.faces)} faces): conformal interface.")
            break
    if export_mesh is None:
        export_mesh = mesh_surf_full

    # ------------------------------------------------------------------
    # EXPORT IN METRES. The generated .edp works in SI (HzBase, TyDep, the
    # Goldak axes are all metres), so the mesh it reads must be metres too.
    # Gmsh's Medit output is unitless, and the previous pipeline handed the
    # solver a millimetre mesh while compensating with an x/1000 rescale in
    # the .edp. That compensation is gone (the mesh is now one conformal
    # domain, not a CAD patch to be repositioned), so the units are made
    # right here, once, at the point where they are known.
    # ------------------------------------------------------------------
    export_m = export_mesh.copy()
    export_m.apply_scale(1.0e-3)
    tmp_stl = output_mesh_path + "_tmp_repaired.stl"
    export_m.export(tmp_stl)

    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("waam_domain")

    mesh_dim = 0
    n_nodes = 0
    n_tets = 0
    # Defaults for the sizing report; overwritten by the element-size block.
    lc_fine = 0.5 * lc_fine_mm
    lc_coarse = lc_fine
    try:
        # Closed shell -> geo volume. The route below is the only one that
        # terminates and honours the requested element size:
        #   classifySurfaces() + createGeometry() builds real geo geometry
        #   (points/curves/surfaces) out of the imported discrete STL, which is
        #   what makes a re-meshable volume possible. Two alternatives were
        #   measured and rejected: geo addSurfaceLoop() on the raw merged
        #   surfaces creates an entity that cannot be re-meshed ("Could not
        #   recover boundary mesh"), and Mesh.CharacteristicLengthMax alone is
        #   ignored once a discrete surface mesh is present.
        #   The element size is then pushed onto the geometry POINTS, which is
        #   the size source that reliably governs a geo-kernel remesh.
        gmsh.merge(tmp_stl)
        gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)

        stage = "surface"
        vol_route = "none"
        try:
            gmsh.model.mesh.classifySurfaces(math.radians(40.0), True, True)
            gmsh.model.mesh.createGeometry()
            surfs = [t for d, t in gmsh.model.getEntities(2) if d == 2]
            if surfs:
                loop = gmsh.model.geo.addSurfaceLoop(surfs)
                gmsh.model.geo.addVolume([loop])
                gmsh.model.geo.synchronize()
                if [t for d, t in gmsh.model.getEntities(3) if d == 3]:
                    stage, vol_route = "volume", "classifySurfaces"
        except Exception:
            pass

        if stage != "volume":
            warnings.append(
                "[STL-MESH] Could not build a re-meshable 3D volume from the "
                "imported shell (Gmsh classifySurfaces rejected the boundary "
                "topology -- typical causes: non-manifold edges, zero-thickness "
                "walls, or self-intersections in the CAD export). No solvable "
                "mesh was produced.")

        # --- element size ----------------------------------------------------
        # The node BUDGET (target_nodes) and the layer cap (0.5 * layer_height)
        # are two different requirements that used to fight each other: lc_max
        # took min(lc_budget, layer cap), so on a 150x150x5 mm plate the layer
        # cap ALWAYS won and a 2000-node request silently produced a ~45k-node /
        # ~196k-tet mesh -- identical for a 1k- and a 20k-triangle STL.
        # They now govern DIFFERENT regions:
        #   lc_fine   = 0.5 * layer_height  -> a box around the deposited part
        #   lc_coarse = (domain volume / target_nodes)^(1/3) -> everywhere else
        h_val = float(layer_height) if float(layer_height) > 0 else 3.0
        lc_fine = 0.50 * h_val
        lc_layer_min = max(0.10 * h_val, 0.30)
        vol = float(spans[0] * spans[1] * spans[2])
        # The MESHED domain is the union of the part and the substrate plate,
        # so the budget must count both. Using only the part bbox let a
        # 3000-node request produce 210k nodes (the plate was free).
        vol_domain = vol + SUB_LX * SUB_LY * SUB_HZ
        # Fine-region margin: the part plus a band that reaches a little way
        # into the substrate (the heat-affected zone).
        margin_mm = min(max(0.35 * min(spans[0], spans[1], spans[2]), 3.0), 20.0)
        fine_box_vol = ((spans[0] + 2 * margin_mm) * (spans[1] + 2 * margin_mm)
                        * (spans[2] + 2 * margin_mm))
        if target_nodes and target_nodes > 0 and vol_domain > 0:
            lc_budget = (vol_domain / max(target_nodes, 1)) ** (1.0 / 3.0)
            # Budget coarser than the layer cap -> coarse far field, fine near
            # the part. Budget finer than the layer cap -> honour the budget
            # uniformly (the user asked for a high-resolution mesh).
            lc_coarse = max(lc_budget, lc_fine)
            if lc_budget < lc_fine:
                lc_fine = lc_budget
            # The fine region alone must not blow the budget: when the layer
            # cap cannot be met inside the budget, the cap is relaxed (never
            # past the far-field size) so the request degrades gracefully.
            est_fine = fine_box_vol / max(lc_fine, 1e-6) ** 3
            budget_fine = 0.6 * float(target_nodes)
            if est_fine > budget_fine > 0:
                lc_fine = min(lc_coarse, lc_fine * (est_fine / budget_fine) ** (1.0 / 3.0))
        else:
            lc_coarse = lc_fine
        lc_min = min(lc_layer_min, lc_coarse * 0.5, lc_fine * 0.5)
        max_elements = 500000
        if vol_domain > 0 and lc_coarse > 0:
            est_elements = vol_domain / (lc_coarse ** 3)
            if est_elements > max_elements:
                lc_coarse = (vol_domain / max_elements) ** (1.0 / 3.0)
                lc_fine = min(lc_fine, lc_coarse)
                lc_min = min(lc_layer_min, lc_coarse * 0.5)
                warnings.append(
                    f"[STL-MESH] Node budget guard: coarsened the far-field size to "
                    f"{lc_coarse:.3f} mm to stay under {max_elements} elements "
                    f"(est. {est_elements:.0f}).")

        # lc_* are calibrated in millimetres above; the exported geometry is
        # in metres, so the sizes follow.
        lc_max_m = lc_coarse / 1000.0
        lc_fine_m = lc_fine / 1000.0
        lc_min_m = lc_min / 1000.0
        if stage == "volume":
            # Sizes are pushed through a background FIELD, not onto the points:
            # a merged STL carries its own (arbitrary) discretisation sizes, and
            # both MeshSizeExtendFromBoundary and MeshSizeFromCurvature propagate
            # them into the volume. Point sizes alone also could not express the
            # two-region sizing above.
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min_m)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(lc_max_m, lc_fine_m))
            for d, tag in gmsh.model.getEntities(0):
                gmsh.model.mesh.setSize([(d, tag)], lc_max_m)
            # Fine box: the part plus a margin that reaches a little way into
            # the substrate (the heat-affected zone), coarse everywhere else.
            fld = gmsh.model.mesh.field
            fbox = fld.add("Box")
            fld.setNumber(fbox, "VIn", lc_fine_m)
            fld.setNumber(fbox, "VOut", lc_max_m)
            fld.setNumber(fbox, "XMin", (bbox[0][0] - margin_mm) / 1000.0)
            fld.setNumber(fbox, "XMax", (bbox[1][0] + margin_mm) / 1000.0)
            fld.setNumber(fbox, "YMin", (bbox[0][1] - margin_mm) / 1000.0)
            fld.setNumber(fbox, "YMax", (bbox[1][1] + margin_mm) / 1000.0)
            fld.setNumber(fbox, "ZMin", (bbox[0][2] - margin_mm) / 1000.0)
            fld.setNumber(fbox, "ZMax", (bbox[1][2] + margin_mm) / 1000.0)
            fld.setNumber(fbox, "Thickness", margin_mm / 1000.0)
            gmsh.model.mesh.field.setAsBackgroundMesh(fbox)
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            warnings.append(
                f"[STL-MESH] Two-region sizing: {lc_fine:.2f} mm near the part, "
                f"{lc_coarse:.2f} mm far field (from the ~{target_nodes:,}-node budget; "
                f"layer cap {h_val * 0.5:.2f} mm).")

        # --- boundary labelling ----------------------------------------------
        # DONE IN PYTHON, NOT IN GMSH. A merged STL arrives as ONE discrete
        # surface entity, so there is nothing to classify per-surface inside
        # Gmsh, and the Medit writer stores elementary entity tags (Mesh.
        # SaveElementTagType=1 by default), not physical tags -- so even
        # correctly created physical groups silently came out as a single
        # label. Here every boundary TRIANGLE is classified by its own
        # geometry, which is exact and kernel-independent:
        #     all three vertices on zmin -> 1 (substrate underside)
        #     all three vertices on zmax -> 5 (top)
        #     anything else               -> 2 (flanks; solver treats 2..6 alike)
        # (runs after gmsh.write, below, on the file that will be read back)

        if stage == "volume":
            gmsh.model.mesh.generate(3)
            mesh_dim = 3
        else:
            warnings.append(
                "[STL-MESH] Generating a 2D surface mesh as a LAST RESORT so the "
                "file exists for inspection. It is not solvable.")
            gmsh.model.mesh.generate(2)
            mesh_dim = 2

        gmsh.write(output_mesh_path)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        if mesh_dim == 3:
            tet_type, tet_tags, _ = gmsh.model.mesh.getElements(3)
            n_tets = sum(len(t) for t in tet_tags) if tet_tags else 0
        else:
            tri_type, tri_tags, _ = gmsh.model.mesh.getElements(2)
            n_tris = sum(len(t) for t in tri_tags) if tri_tags else 0
            warnings.append(
                f"[STL-MESH] Surface mesh: {n_nodes:,} nodes | {n_tris:,} triangles "
                f"-- NO TETRAHEDRA. This mesh cannot be solved.")
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_stl):
                os.remove(tmp_stl)
        except Exception:
            pass

    if mesh_dim == 3 and n_tets > 0:
        zmin_f, zmax_f = _mesh_z_extent(output_mesh_path)
        n_lab = _relabel_mesh_boundary(output_mesh_path, zmin_f, zmax_f)
        warnings.append(
            f"[STL-LABELS] Boundary triangles relabelled by geometry "
            f"(z in [{zmin_f:.2f}, {zmax_f:.2f}] mm): {n_lab} face(s) across "
            f"labels {sorted(_count_mesh_labels(output_mesh_path).keys())}.")

    # Post-write verification: a .mesh without tetrahedra is what made the
    # solver SIGBUS. Refuse to hand it over as a success.
    if mesh_dim != 3 or n_tets <= 0:
        raise RuntimeError(
            f"STL volume meshing failed for {stl_path}: produced "
            f"{'a 2D surface mesh' if mesh_dim == 2 else 'no elements'} "
            f"({n_nodes} nodes, {n_tets} tets).\n"
            f"Diagnostics collected while meshing:\n  "
            + "\n  ".join(warnings[-8:])
            + f"\nDiagnose the CAD with:\n"
            f"    python3 -c \"import trimesh; m=trimesh.load('{stl_path}'); "
            f"print(m.is_watertight, m.is_winding_consistent)\"")

    # Sanity-check the labels actually written (gmsh is free to ignore a
    # physical tag if the group was empty), so a silently untagged mesh can
    # never reach the solver.
    labels = _count_mesh_labels(output_mesh_path)
    if LABEL_BOTTOM not in labels or labels.get(LABEL_BOTTOM, 0) == 0:
        raise RuntimeError(
            f"Exported mesh carries no label-{LABEL_BOTTOM} (substrate underside) "
            f"faces (found {sorted(labels)}), so the fixture spring and hBot "
            f"transfer would integrate over an empty set.")
    if not any(l in (2, 3, 4, 5, 6) and c for l, c in labels.items()):
        raise RuntimeError(
            f"Exported mesh carries no label 2..6 faces (found {sorted(labels)}), "
            f"so no surface could convect or radiate.")

    # The budget is a TARGET, not a hard cap (gmsh will not honour a cap on a
    # curved surface), but a large overshoot must be reported instead of
    # looking like the budget was ignored.
    if target_nodes and target_nodes > 0 and n_nodes > 2.0 * float(target_nodes):
        warnings.append(
            f"[STL-MESH] Achieved {n_nodes:,} nodes / {n_tets:,} tets against the "
            f"~{int(target_nodes):,}-node budget: the near-part size had to stay at "
            f"{lc_fine:.2f} mm (layer cap {h_val * 0.5:.2f} mm) to resolve the part. "
            f"Raise the budget, or increase the layer height, for a coarser mesh.")

    return {
        "mesh_path": output_mesh_path,
        "n_nodes": n_nodes,
        "n_tets": n_tets,
        "mesh_dim": mesh_dim,
        "scale_applied": scale_applied,
        "sub_hz_mm": SUB_HZ,
        "target_nodes": int(target_nodes) if target_nodes else None,
        "lc_fine_mm": float(lc_fine),
        "lc_coarse_mm": float(lc_coarse),
        "labels": labels,
        "bbox_mm": {
            "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
            "xmin": float(bbox[0][0]), "ymin": float(bbox[0][1]), "zmin": float(bbox[0][2]),
            "xmax": float(bbox[1][0]), "ymax": float(bbox[1][1]), "zmax": float(bbox[1][2]),
            "cad_zmax": z_cad_max,
        },
        "warnings": warnings,
    }


def _box_mesh(dx: float, dy: float, dz: float):
    """Axis-aligned substrate box as a trimesh mesh, centred in X/Y at the
    origin and sitting on z in [0, dz] (same frame as the placed CAD part)."""
    import trimesh
    return trimesh.creation.box(extents=[dx, dy, dz],
                                transform=trimesh.transformations.translation_matrix(
                                    [dx / 2.0, dy / 2.0, dz / 2.0]))


def _cluster_decimate(mesh, cell_mm: float, require_watertight: bool = True):
    """Vertex-clustering decimation onto a `cell_mm` grid (fail-safe).

    Merges vertices that fall into the same grid cell and drops the triangles
    that collapse. Returns the input mesh UNCHANGED whenever the result looks
    worse than what went in: it did not actually get smaller, or it is
    topologically broken (not watertight when required / inconsistent winding).
    No external decimation backend is required, so this works in every
    environment the framework runs in.
    """
    import numpy as np
    import trimesh
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if cell_mm <= 0 or len(F) < 50:
        return mesh
    origin = V.min(axis=0)
    keys = np.floor((V - origin) / float(cell_mm)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    new_V = np.zeros((len(first), 3), dtype=np.float64)
    np.add.at(new_V, inverse, V)
    counts = np.bincount(inverse, minlength=len(first)).reshape(-1, 1)
    new_V /= np.maximum(counts, 1)
    new_F = inverse[F]
    keep = ((new_F[:, 0] != new_F[:, 1])
            & (new_F[:, 1] != new_F[:, 2])
            & (new_F[:, 0] != new_F[:, 2]))
    new_F = new_F[keep]
    if len(new_F) < 4 or len(new_F) >= len(F):
        return mesh
    try:
        out = trimesh.Trimesh(vertices=new_V, faces=new_F, process=True)
    except Exception:
        return mesh
    if not out.is_winding_consistent:
        return mesh
    if require_watertight and not out.is_watertight:
        return mesh
    return out


def _mesh_z_extent(mesh_path: str) -> tuple:
    """(zmin, zmax) over the vertex coordinates of a Medit .mesh file.

    Read back from the file rather than from live Gmsh state so the labelling
    step needs no kernel and can never disagree with what was written.
    """
    with open(mesh_path, "r", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    i = lines.index("Vertices")
    n_vert = int(lines[i + 1].split()[0])
    zs = []
    for ln in lines[i + 2:i + 2 + n_vert]:
        parts = ln.split()
        if len(parts) >= 3:
            try:
                zs.append(float(parts[2]))
            except ValueError:
                pass
    if not zs:
        raise RuntimeError(f"{mesh_path} has no readable vertex coordinates")
    return min(zs), max(zs)


def _relabel_mesh_boundary(mesh_path: str, zmin: float, zmax: float) -> int:
    """Rewrite the triangle refs of a Medit .mesh by geometric classification.

    Returns the number of triangles relabelled. The mesh is rewritten through a
    temporary file and atomically replaced, so a crash mid-rewrite can never
    leave a half-written mesh behind.

    Label scheme (must match mesh/structured_conforming.py set B):
        1 -> z == zmin  (substrate underside: hBot + fixture spring)
        5 -> z == zmax  (top surface)
        2 -> everything else (flanks; the solver treats labels 2..6 alike)
    """
    path = Path(mesh_path)
    tol = 1.0e-3 * max(1.0, abs(zmax - zmin))
    with open(path, "r", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    def section_bounds(name):
        for i, ln in enumerate(lines):
            if ln.strip() == name:
                return i
        return -1

    # --- vertices: implicit 1-based id -> z --------------------------------
    # A Medit vertex row is `x y z ref` with NO leading index: the id is the
    # position in the section. (An earlier revision parsed parts[0] as an id,
    # which silently worked while x was integral and broke the moment it was
    # not.)
    i = section_bounds("Vertices")
    if i < 0:
        raise RuntimeError(f"{mesh_path} has no Vertices section")
    n_vert = int(lines[i + 1].split()[0])
    vz = {}
    for k, ln in enumerate(lines[i + 2:i + 2 + n_vert]):
        parts = ln.split()
        if len(parts) >= 3:
            try:
                vz[k + 1] = float(parts[2])
            except ValueError:
                pass

    # --- triangles: classify by all three vertex z -------------------------
    i = section_bounds("Triangles")
    if i < 0:
        raise RuntimeError(f"{mesh_path} has no Triangles section")
    n_tri = int(lines[i + 1].split()[0])
    n_labelled = 0
    for k in range(n_tri):
        j = i + 2 + k
        if j >= len(lines):
            break
        parts = lines[j].split()
        if len(parts) < 4:
            continue
        try:
            zs = [vz[int(p)] for p in parts[:3]]
        except (KeyError, ValueError):
            continue
        if all(abs(z - zmin) <= tol for z in zs):
            lab = LABEL_BOTTOM
        elif all(abs(z - zmax) <= tol for z in zs):
            lab = LABEL_TOP
        else:
            lab = LABEL_SIDE
        lines[j] = "%s %s %s %d" % (parts[0], parts[1], parts[2], lab)
        n_labelled += 1

    if n_labelled == 0:
        raise RuntimeError(
            f"Could not classify any boundary triangle in {mesh_path}; the mesh "
            f"is empty or not a Medit volume mesh.")

    tmp = path.with_suffix(path.suffix + ".relabelled")
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    tmp.replace(path)
    return n_labelled



def _count_mesh_labels(mesh_path: str) -> dict:
    """Histogram of triangle refs in a Medit .mesh file ({} if unreadable)."""
    labels = {}
    in_tri = False
    expect = 0
    try:
        with open(mesh_path, "r", errors="replace") as fh:
            for line in fh:
                tok = line.strip()
                if not tok:
                    continue
                if tok == "Triangles":
                    in_tri, expect = True, 0
                    continue
                if in_tri and expect == 0 and tok.isdigit():
                    expect = int(tok)
                    continue
                if in_tri:
                    if tok.startswith("End"):
                        in_tri = False
                        continue
                    lab = tok.split()[-1]
                    try:
                        lab = int(lab)
                    except ValueError:
                        continue
                    labels[lab] = labels.get(lab, 0) + 1
                    expect -= 1
                    if expect <= 0:
                        in_tri = False
    except Exception:
        return {}
    return labels


def stl_to_freefem_mesh(stl_path: str, output_mesh_path: str = None,
                        target_nodes: int = 2000,
                        layer_height: float = 3.0,
                        sub_hz_mm: float = None,
                        sub_lx_mm: float = None,
                        sub_ly_mm: float = None,
                        in_process: bool = False) -> dict:
    """
    Convert a STL file to a FreeFEM-compatible .mesh file via Gmsh.

    The default execution path runs Gmsh inside an isolated worker subprocess.
    This prevents macOS OpenMP (libgmsh) fatal SIGBUS / KERN_PROTECTION_FAILURE
    crashes when called from Tkinter or Cocoa event loops.

    Raises RuntimeError when a solvable volume mesh cannot be produced. A mesh
    that is not solvable used to be returned as a success (a two-dimensional
    surface mesh with its triangle count reported as "n_tets"), which made the
    solver abort with SIGBUS only after the whole pipeline had been paid for.

    sub_hz_mm MUST match the solver's `hz_base` (in mm): the generated .edp
    treats everything at z <= hz_base as the substrate.
    """
    args = dict(stl_path=stl_path, output_mesh_path=output_mesh_path,
                target_nodes=target_nodes, layer_height=layer_height,
                sub_hz_mm=sub_hz_mm, sub_lx_mm=sub_lx_mm, sub_ly_mm=sub_ly_mm)
    if in_process:
        return _stl_to_freefem_mesh_in_process(**args)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(stl_path),
        str(output_mesh_path or ""),
        str(target_nodes),
        str(layer_height),
        str(sub_hz_mm if sub_hz_mm is not None else ""),
        str(sub_lx_mm if sub_lx_mm is not None else ""),
        str(sub_ly_mm if sub_ly_mm is not None else ""),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as ex:
        # Worker died. Surface its own error message (the mesher raises with a
        # diagnosis) rather than retrying silently with different defaults.
        detail = (ex.stderr or ex.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no worker output"
        raise RuntimeError(
            f"STL meshing worker failed (exit {ex.returncode}): {tail}") from ex

    out_text = proc.stdout.strip()
    json_start = out_text.find("{")
    json_end = out_text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return json.loads(out_text[json_start:json_end + 1])

    # No JSON on stdout: the worker may have crashed before printing. Retry
    # in-process with EVERY argument preserved (an earlier revision dropped
    # layer_height here and silently meshed with the wrong element size).
    print("[STL-PIPELINE] Worker produced no JSON; retrying in-process with the "
          "same arguments.")
    return _stl_to_freefem_mesh_in_process(**args)


if __name__ == "__main__":
    default_stl = str(Path(__file__).resolve().parent.parent / "Main.stl")
    stl = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else default_stl
    out_mesh = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
    target_nodes = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].strip() else 2000
    lh = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].strip() else 3.0
    shz = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].strip() else 5.0
    slx = float(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].strip() else 150.0
    sly = float(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7].strip() else 150.0
    result = _stl_to_freefem_mesh_in_process(
        stl, out_mesh, target_nodes=target_nodes, layer_height=lh,
        sub_hz_mm=shz, sub_lx_mm=slx, sub_ly_mm=sly)
    print(json.dumps(result, indent=2))

