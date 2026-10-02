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
                            LxW=0.150, LyBase=0.150, HzBase=0.015,
                            NxB=15, NyB=15, NzB=3,
                            xDepStart=0.015, xDepEnd=0.135,
                            yDepLeft=0.0725, yDepRight=0.0775,
                            HzDep=0.100, NxD=15, NyD=3, NzD=33):
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

        # Copy all point data arrays
        pd_in = in_grid.GetPointData()
        pd_out = hex_grid.GetPointData()
        for i in range(pd_in.GetNumberOfArrays()):
            pd_out.AddArray(pd_in.GetArray(i))

        writer.SetFileName(vtu_path)
        writer.SetInputData(hex_grid)
        writer.Write()
        count += 1

    print(f"[Hex Converter] Converted {count} VTU files in '{directory}' to pure rectangular hexahedra (VTK_HEXAHEDRON).")


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "waam_output"
    convert_all_output_vtus(target_dir)
