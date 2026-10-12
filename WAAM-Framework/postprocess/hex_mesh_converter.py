"""
hex_mesh_converter.py
======================
Converts FreeFEM++ tetrahedral VTU outputs into pure hexahedral (rectangular brick)
VTK UnstructuredGrid files (VTK_HEXAHEDRON, cell type 12).

Eliminates all tetrahedral diagonal artifacts in ParaView, ensuring the user sees
a 100% rectangular structured mesh with exact nodal field preservation.
"""

import os
import glob
import vtk
import numpy as np


def convert_all_output_vtus(directory="waam_output",
                            LxW=0.150, LyBase=0.150, HzBase=0.005,
                            NxB=25, NyB=8, NzB=2,
                            xDepStart=None, xDepEnd=None,
                            yDepLeft=None, yDepRight=None,
                            HzDep=0.104, NxD=25, NyD=3, NzD=26):
    """
    Convert a run's tetrahedral VTUs onto a hexahedral template.

    Geometry defaults MUST come from the same source as the run's parameters.json
    whenever it is available (see convert_dir_with_params below): the values
    above are only the framework's current defaults, and the deprecated
    150x150x15 / 32-layer numbers made the converted grid disagree with the
    mesh that had just been solved.
    """
    if xDepStart is None:
        xDepStart = (LxW - 0.120) / 2.0
    if xDepEnd is None:
        xDepEnd = xDepStart + 0.120
    if yDepLeft is None:
        yDepLeft = LyBase / 2.0 - 0.004
    if yDepRight is None:
        yDepRight = yDepLeft + 0.008
    vtu_files = sorted(glob.glob(os.path.join(directory, "*.vtu")))
    if not vtu_files:
        print(f"[Hex Converter] No .vtu files found in '{directory}'.")
        return

    # 1. Build template hexahedral mesh connectivity from first file
    first_vtu = vtu_files[0]
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(first_vtu)
    reader.Update()
    grid0 = reader.GetOutput()
    nPts = grid0.GetNumberOfPoints()
    pts0 = grid0.GetPoints()

    # Spatial lookup map
    coord_map = {}
    for i in range(nPts):
        p = pts0.GetPoint(i)
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        coord_map[key] = i

    hex_cells_list = []

    # Substrate
    xB = np.linspace(0.0, LxW, NxB + 1)
    yB = np.linspace(0.0, LyBase, NyB + 1)
    zB = np.linspace(0.0, HzBase, NzB + 1)
    for iz in range(NzB):
        for iy in range(NyB):
            for ix in range(NxB):
                c = [
                    (round(xB[ix], 6),   round(yB[iy], 6),   round(zB[iz], 6)),
                    (round(xB[ix+1], 6), round(yB[iy], 6),   round(zB[iz], 6)),
                    (round(xB[ix+1], 6), round(yB[iy+1], 6), round(zB[iz], 6)),
                    (round(xB[ix], 6),   round(yB[iy+1], 6), round(zB[iz], 6)),
                    (round(xB[ix], 6),   round(yB[iy], 6),   round(zB[iz+1], 6)),
                    (round(xB[ix+1], 6), round(yB[iy], 6),   round(zB[iz+1], 6)),
                    (round(xB[ix+1], 6), round(yB[iy+1], 6), round(zB[iz+1], 6)),
                    (round(xB[ix], 6),   round(yB[iy+1], 6), round(zB[iz+1], 6)),
                ]
                if all(k in coord_map for k in c):
                    hex_cells_list.append([coord_map[k] for k in c])

    # Deposit wall
    xD = np.linspace(xDepStart, xDepEnd, NxD + 1)
    yD = np.linspace(yDepLeft, yDepRight, NyD + 1)
    zD = np.linspace(HzBase, HzBase + HzDep, NzD + 1)
    for iz in range(NzD):
        for iy in range(NyD):
            for ix in range(NxD):
                c = [
                    (round(xD[ix], 6),   round(yD[iy], 6),   round(zD[iz], 6)),
                    (round(xD[ix+1], 6), round(yD[iy], 6),   round(zD[iz], 6)),
                    (round(xD[ix+1], 6), round(yD[iy+1], 6), round(zD[iz], 6)),
                    (round(xD[ix], 6),   round(yD[iy+1], 6), round(zD[iz], 6)),
                    (round(xD[ix], 6),   round(yD[iy], 6),   round(zD[iz+1], 6)),
                    (round(xD[ix+1], 6), round(yD[iy], 6),   round(zD[iz+1], 6)),
                    (round(xD[ix+1], 6), round(yD[iy+1], 6), round(zD[iz+1], 6)),
                    (round(xD[ix], 6),   round(yD[iy+1], 6), round(zD[iz+1], 6)),
                ]
                if all(k in coord_map for k in c):
                    hex_cells_list.append([coord_map[k] for k in c])

    print(f"[Hex Converter] Built template with {len(hex_cells_list)} hexahedral cells for {len(vtu_files)} files.")

    # 2. Fast conversion for all VTU files
    writer = vtk.vtkXMLUnstructuredGridWriter()
    count = 0
    for vtu_path in vtu_files:
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(vtu_path)
        r.Update()
        in_grid = r.GetOutput()

        hex_grid = vtk.vtkUnstructuredGrid()
        hex_grid.SetPoints(in_grid.GetPoints())

        # Insert precomputed hex cells
        for node_ids in hex_cells_list:
            cell = vtk.vtkHexahedron()
            for idx, nid in enumerate(node_ids):
                cell.GetPointIds().SetId(idx, nid)
            hex_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

        # Copy point data arrays (the hex grid keeps the SAME node set).
        pd_in = in_grid.GetPointData()
        pd_out = hex_grid.GetPointData()
        for i in range(pd_in.GetNumberOfArrays()):
            pd_out.AddArray(pd_in.GetArray(i))

        # Result fields are commonly written as CELL data (e.g. activation
        # averaged per element). Dropping it -- as this did until now -- threw
        # away the very arrays the conversion exists to preserve, leaving a
        # mesh that looked converted but was blank.
        cd_in = in_grid.GetCellData()
        cd_out = hex_grid.GetCellData()
        if cd_in.GetNumberOfArrays() and hex_grid.GetNumberOfCells() == in_grid.GetNumberOfCells():
            for i in range(cd_in.GetNumberOfArrays()):
                cd_out.AddArray(cd_in.GetArray(i))
        elif cd_in.GetNumberOfArrays():
            print(f"[Hex Converter] WARNING: cell data dropped for "
                  f"{os.path.basename(vtu_path)}: the hex template has "
                  f"{hex_grid.GetNumberOfCells()} cells against the source's "
                  f"{in_grid.GetNumberOfCells()} (geometry mismatch).")

        # Write through a temporary file and replace atomically: a crash
        # mid-write used to leave a truncated .vtu in place of the data.
        tmp_path = vtu_path + ".tmp"
        writer.SetFileName(tmp_path)
        writer.SetInputData(hex_grid)
        writer.Write()
        os.replace(tmp_path, vtu_path)
        count += 1

    print(f"[Hex Converter] Converted {count} VTU files in '{directory}' to pure rectangular hexahedra (VTK_HEXAHEDRON).")


def convert_dir_with_params(directory: str) -> None:
    """Convert a run directory using ITS OWN parameters.json for the geometry.

    Reading the run's geometry instead of a hardcoded default is the difference
    between a converted grid that overlays the solved mesh and one that does
    not. The run's parameters.json is the single source of truth written by
    run.py / the GUI; when it is missing the framework defaults are used and the
    caller is told so.
    """
    import json
    from pathlib import Path
    cand = [Path(directory) / "parameters.json",
            Path(directory).parent / "parameters.json"]
    params = {}
    for c in cand:
        if c.exists():
            try:
                with open(c) as fh:
                    params = json.load(fh) or {}
                print(f"[Hex Converter] geometry from {c}")
                break
            except Exception as ex:
                print(f"[Hex Converter] could not read {c}: {ex}")
    if not params:
        print("[Hex Converter] no parameters.json found; using framework "
              "default geometry (verify the overlay matches the solved mesh).")
    kw = dict(
        LxW=float(params.get("lx", 0.150)),
        LyBase=float(params.get("ly_base", 0.150)),
        HzBase=float(params.get("hz_base", 0.005)),
        NxB=int(params.get("nx_base", 25)),
        NyB=int(params.get("ny_base", 8)),
        NzB=int(params.get("nz_base", 2)),
        NxD=int(params.get("nx_dep", 25)),
        NyD=int(params.get("ny_dep", 3)),
        NzD=int(params.get("nz_dep", 26)),
    )
    nl = int(params.get("n_layers", 26))
    h1 = float(params.get("h_bead_first", params.get("h_bead", 0.004)))
    hs = float(params.get("h_bead_subseq", h1))
    kw["HzDep"] = nl * h1 if nl <= 1 else h1 + (nl - 1) * hs
    convert_all_output_vtus(directory, **kw)


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "waam_output"
    convert_dir_with_params(target_dir)
