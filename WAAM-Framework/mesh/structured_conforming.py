"""Conforming structured mesh generator for the WAAM plate + deposit domain.

WHY THIS EXISTS
---------------
The builder previously built the substrate and the deposit as two independent
FreeFEM `cube()` meshes and concatenated them (`Th = ThBase + ThDep`). Because
their element sizes differ, the two node distributions along Y do not match even
though the faces are coplanar at z = HzBase:

    base top face  y = [0, 18.75, 37.5, 56.25, 75.0, 93.75, 112.5, 131.25, 150.0] mm
    deposit bottom y = [71.0, 73.6667, 76.3333, 79.0] mm
    coincident coordinates: 0

FreeFEM's concatenation does NOT merge them, so the interface was a crack of zero
width: no shared node, no constraint. Measured consequence on a running model:
a mean 19.9 C and peak 92.9 C temperature discontinuity across the interface, no
heat conduction from the bead into the plate, and no mechanical bonding at all
(the deposit could not transmit stress or a bending moment).

For an additive process the track is FUSED to the plate, so that is simply wrong.

WHAT THIS DOES
--------------
Builds ONE conforming structured mesh over the whole domain, with:
  * uniform NxD divisions in X,
  * GRADED divisions in Y -- fine (NyD cells across the bead width) inside the
    deposit band, coarse (NyB cells) outside it,
  * nzBase divisions over the substrate plus nzDep over the deposit in Z.

Grading Y is what lets the deposit keep its fine resolution without forcing the
whole 150 mm plate to be meshed at bead scale. Because it is a single structured
grid, the interface nodes are shared by construction and the interface is bonded
with no special case.

Output is a Medit `.mesh` file (the format `readmesh3` consumes).

Element type: first-order tetrahedra, each hexahedron split into 6 tets about the
body diagonal. That subdivision is provably face-consistent between neighbouring
cells, which is what guarantees the shared interface nodes form a watertight,
bonded surface.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

# Body-diagonal subdivision of a hexahedron into 6 tetrahedra.
#
# The cube's corners are indexed with bit flags (dx, dy, dz):
#   index = 4*dz + 2*dy + dx
# so corner 0 is (0,0,0) and corner 7 is (1,1,1). The 6 tets all share the
# main body diagonal 0-7, which is what makes the subdivision conforming: the
# four tets around one corner of the shared face between two cubes induce the
# same diagonal on that face, so neighbouring cubes agree.
#
# This is deliberately the 6-tet split rather than a 5-tet "Kuhn" split: the
# 6-tet version is the standard construction and is provably face-consistent,
# whereas a hand-written 5-tet table is easy to get subtly wrong (an earlier
# version of this file did, producing a mesh with 33330 non-manifold faces and
# a missing x-min boundary, which FreeFEM silently tolerated).
_HEX_CORNERS = (
    (0, 0, 0),  # 0
    (1, 0, 0),  # 1
    (1, 1, 0),  # 2
    (0, 1, 0),  # 3
    (0, 0, 1),  # 4
    (1, 0, 1),  # 5
    (1, 1, 1),  # 6
    (0, 1, 1),  # 7
)

_BODY_DIAGONAL_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def graded_axis(breaks: Sequence[float], cells: Sequence[int]) -> List[float]:
    """Build a 1-D node coordinate list from breakpoints and per-segment cells.

    `breaks` are the segment boundaries in ascending order; `cells[i]` is the
    number of uniform cells to place inside segment i. Nodes at shared boundaries
    are emitted once, so the result is strictly increasing and continuous.
    """
    if len(breaks) != len(cells) + 1:
        raise ValueError("breaks must have exactly one more entry than cells")
    coords: List[float] = [float(breaks[0])]
    for i, n in enumerate(cells):
        n = int(n)
        if n < 1:
            raise ValueError(f"segment {i} needs at least 1 cell, got {n}")
        a, b = float(breaks[i]), float(breaks[i + 1])
        if not b > a:
            raise ValueError(f"segment {i} is not ascending: {a} -> {b}")
        for k in range(1, n + 1):
            coords.append(a + (b - a) * k / n)
    return coords


def build_waam_nodes(
    lx: float,
    ly_base: float,
    hz_base: float,
    ty_dep: float,
    hz_dep: float,
    nx_dep: int,
    ny_dep: int,
    nz_dep: int,
    nx_base: int,
    ny_base: int,
    nz_base: int,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int, int]]]:
    """Return (nodes, tets) for the conforming plate + deposit domain.

    Node indexing is row-major over (z, y, x) so the file stays cache friendly.
    """
    if hz_dep <= 0:
        raise ValueError("deposit height must be positive")
    if not (0 < ty_dep <= ly_base):
        raise ValueError("deposit width must be positive and fit within the plate")

    xs = graded_axis([0.0, lx], [nx_dep])

    # Y grading: the bead band is centred on the plate and gets ny_dep cells;
    # the two flanks split the remaining width and get ny_base cells each so the
    # plate keeps a reasonable element count. Integer split that always sums to
    # exactly ny_base: floor to the left flank, remainder to the right. Python's
    # round() is banker's rounding (round(2.5) == 2), which silently dropped the
    # odd cell for every odd ny_base and produced fewer flank cells than asked
    # for. The band stays centred because its position comes from the y_left /
    # y_right coordinates, not from the cell counts.
    y_left = 0.5 * ly_base - 0.5 * ty_dep
    y_right = y_left + ty_dep
    n_left = max(1, int(ny_base) // 2)
    n_right = max(1, int(ny_base) - n_left)
    ys = graded_axis([0.0, y_left, y_right, ly_base], [n_left, ny_dep, n_right])
    # Y node indices spanning the bead band (cells j0..j1-1). graded_axis emits
    # shared breakpoints once, so the band starts exactly after the n_left
    # left-flank cells.
    j0 = n_left
    j1 = n_left + ny_dep

    # Z: substrate then deposit, sharing the plane at hz_base. Deposit cells are
    # those at and above level index nz_base.
    zs = graded_axis([0.0, hz_base, hz_base + hz_dep], [nz_base, nz_dep])
    k_split = nz_base

    nx = len(xs)
    ny = len(ys)
    nz = len(zs)

    def node_id(i: int, j: int, k: int) -> int:
        return k * ny * nx + j * nx + i

    nodes: List[Tuple[float, float, float]] = []
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                nodes.append((x, y, z))

    # Cells: the substrate spans the full plate width, but above hz_base only
    # the bead band exists. There is deliberately NO material above the plate
    # outside the track: an earlier revision meshed the full tensor product,
    # which put a phantom 4 mm "deposit" blanket over the whole plate. It was
    # quiet (never activated), so it soaked conducted heat, stored it, and --
    # worse -- turned the plate's entire top surface into an interior face, so
    # the plate could not convect from its top at all.
    tets: List[Tuple[int, int, int, int]] = []
    for k in range(nz - 1):
        in_deposit = k >= k_split
        for j in range(ny - 1):
            if in_deposit and not (j0 <= j < j1):
                continue
            for i in range(nx - 1):
                c = [0] * 8
                c[0] = node_id(i, j, k)
                c[1] = node_id(i + 1, j, k)
                c[3] = node_id(i, j + 1, k)
                c[2] = node_id(i + 1, j + 1, k)
                c[4] = node_id(i, j, k + 1)
                c[5] = node_id(i + 1, j, k + 1)
                c[7] = node_id(i, j + 1, k + 1)
                c[6] = node_id(i + 1, j + 1, k + 1)
                for tet in _BODY_DIAGONAL_TETS:
                    tets.append((c[tet[0]], c[tet[1]], c[tet[2]], c[tet[3]]))

    # Compact: drop grid nodes no element references (the full-width levels
    # above the substrate outside the band) and renumber.
    used = sorted({v for t in tets for v in t})
    remap = {old: new for new, old in enumerate(used)}
    nodes = [nodes[old] for old in used]
    tets = [
        (remap[a], remap[b], remap[c], remap[d]) for a, b, c, d in tets
    ]

    _fix_orientation(nodes, tets)
    _assert_positive_volume(nodes, tets)
    return nodes, tets


def _signed_volume(
    nodes: Sequence[Tuple[float, float, float]],
    tet: Tuple[int, int, int, int],
) -> float:
    """Signed volume of a tetrahedron (positive when right-handed)."""
    p0, p1, p2, p3 = (nodes[i] for i in tet)
    u = [p1[i] - p0[i] for i in range(3)]
    v = [p2[i] - p0[i] for i in range(3)]
    w = [p3[i] - p0[i] for i in range(3)]
    det = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return det / 6.0


def _fix_orientation(
    nodes: Sequence[Tuple[float, float, float]],
    tets: List[Tuple[int, int, int, int]],
    eps: float = 1e-20,
) -> int:
    """Flip any negatively-oriented tet in place; return how many were flipped.

    FreeFEM rejects "badly oriented element" at mesh-read time, and the six-tet
    body-diagonal split below contains left-handed tets by construction (a corner
    tet such as (0,2,3,6) is negatively oriented), so orientation has to be
    repaired rather than assumed. Swapping two vertices reverses the sign of the
    signed volume, which is all that is needed.
    """
    flipped = 0
    for n, t in enumerate(tets):
        v = _signed_volume(nodes, t)
        if v < 0.0:
            a, b, c, d = t
            tets[n] = (a, b, d, c)
            flipped += 1
        elif abs(v) <= eps:
            raise ValueError(f"degenerate tetrahedron #{n} {t}: |V| = {abs(v):.3e}")
    return flipped


def _assert_positive_volume(
    nodes: Sequence[Tuple[float, float, float]],
    tets: Sequence[Tuple[int, int, int, int]],
) -> None:
    """Fail loudly if any tetrahedron is degenerate or still inverted.

    An inverted element surfaces much later as a mysterious non-convergence, so
    it is worth catching here. Checking only |V| would be insufficient: a whole
    block of consistently negative elements passes an absolute-value test and is
    still rejected by the solver.
    """
    for n, t in enumerate(tets):
        v = _signed_volume(nodes, t)
        if v <= 0.0:
            raise ValueError(
                f"non-positive tetrahedron #{n} {t}: signed volume = {v:.3e}"
            )


def _tet_boundary_triangles(tet: Tuple[int, int, int, int]):
    """The four faces of a tetrahedron, as index triples.

    Exactly the four 3-subsets, each once: (a,b,c), (a,b,d), (a,c,d), (b,c,d).
    An earlier revision returned (a,d,b) and (a,b,d) -- the same face twice
    with opposite winding -- and never returned (a,c,d), so exterior acd faces
    lost their boundary label while abd faces were labelled twice.
    """
    a, b, c, d = tet
    return ((a, b, c), (a, b, d), (a, c, d), (b, c, d))


# ---------------------------------------------------------------------------
# BOUNDARY LABEL CONVENTION -- READ THIS BEFORE TOUCHING ANY LABEL
# ---------------------------------------------------------------------------
# Two DIFFERENT numberings exist in this workshop and mixing them up silently
# misplaces every boundary condition. They were verified by executing
# FreeFEM++ 4.15 and asking it which faces each int2d(Th,L) set contains
# (face areas of a 10 x 30 x 50 mm box plus value-weighted integrals of x, y, z):
#
#   A) FreeFEM's own cube() mesh generator:
#        1 -> y MIN      2 -> x MAX     3 -> y MAX
#        4 -> x MIN      5 -> z MIN (bottom)   6 -> z MAX (top)
#      An earlier revision of this table asserted 1 -> zmin and 5 -> zmax /
#      6 -> ymin, i.e. 1/5/6 cyclically shifted. That is NOT what FreeFEM 4.15
#      builds, and any script that meshes with cube() and then integrates on
#      label 1 expecting the underside gets a vertical flank instead.
#
#   B) The labels THIS MODULE WRITES into the .mesh file (and therefore the
#      only numbering that matters for scripts that `readmesh3` it):
#        1 -> z MIN  (substrate underside)
#        2 -> x MAX      3 -> y MAX      4 -> x MIN
#        5 -> z MAX  (top / exposed plate top)
#        6 -> y MIN
#
# `readmesh3` reads the triangle refs straight out of the file, so set B is
# self-consistent with the solver: int2d(Th,1) is exactly the underside this
# writer tags 1. That is what the generated .edp relies on -- label 1 carries
# hBot on the substrate underside plus the isostatic fixture spring, and labels
# 2..6 carry the convective/radiative surface loss (they are all identical
# there, so their individual identities do not matter).
#
# Files that build their mesh with FreeFEM cube() directly (the hand-written
# legacy DEFAULT.edp) live in world A, so their int2d(Th,1) is a flank and
# their on(1,...) clamps a flank -- that is a bug in those scripts, not here.
# Do NOT "harmonise" by renumbering B to match A: it would move the fixture
# spring and hBot off the substrate underside for every generated run.
_BOUNDARY_LABEL = {
    ("z", "min"): 1,   # set B only (FreeFEM cube() labels this face 5)
    ("x", "max"): 2,   # set A: 2  | set B: 2
    ("y", "max"): 3,   # set A: 3  | set B: 3
    ("x", "min"): 4,   # set A: 4  | set B: 4
    ("z", "max"): 5,   # set B only (FreeFEM cube() labels this face 6)
    ("y", "min"): 6,   # set B only (FreeFEM cube() labels this face 1)
}


def _outward_normal(
    nodes: Sequence[Tuple[float, float, float]],
    tri: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Unit normal of a triangle from its winding (right-hand rule)."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = (nodes[i] for i in tri)
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    n = (nx * nx + ny * ny + nz * nz) ** 0.5
    if n <= 0.0:
        raise ValueError(f"degenerate boundary triangle {tri}")
    return (nx / n, ny / n, nz / n)


def classify_boundary_faces(
    nodes: Sequence[Tuple[float, float, float]],
    tets: Sequence[Tuple[int, int, int, int]],
    y_left: float,
    y_right: float,
    hz_base: float,
    tol: float = 1e-9,
) -> List[Tuple[int, int, int, int]]:
    """Return exterior faces of the domain with the label set documented in
    _BOUNDARY_LABEL (set B: 1 = substrate underside, 5 = top, 2/3/4/6 = flanks).

    Exteriority is decided by face COUNTING (a face used by exactly one tet is
    exterior), not by plane tests, so the substrate/deposit interface and the
    base-cell/deposit-cell junctions can never leak into the boundary set. Each
    exterior face is then labelled by which plane it lies on:

      * the six outer planes (labels 1..6, set-B numbering above);
      * the deposit flanks at y_left / y_right above the substrate (6 / 3);
      * the exposed plate top at hz_base outside the bead band (5).

    Every exterior face must match exactly one plane, otherwise generation
    fails loudly -- an unlabeled face silently deletes convection, and a
    doubly-labelled one double-counts it. Each triangle is wound so its normal
    points OUT of the domain (checked geometrically, not assumed from the tet
    vertex order), because an inward normal negates every surface integral.
    """
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    zs = [n[2] for n in nodes]
    lo = {"x": min(xs), "y": min(ys), "z": min(zs)}
    hi = {"x": max(xs), "y": max(ys), "z": max(zs)}
    scale = max(hi["x"], hi["y"], hi["z"], abs(lo["x"]), 1.0)
    t = tol * scale

    count: dict = {}
    for tet in tets:
        for tri in _tet_boundary_triangles(tet):
            key = tuple(sorted(tri))
            count[key] = count.get(key, 0) + 1
    bad = [k for k, v in count.items() if v > 2]
    if bad:
        raise ValueError(f"{len(bad)} non-manifold faces (used by >2 tets)")

    # (plane axis, outward sign, extra condition, label)
    faces: List[Tuple[int, int, int, int]] = []
    for key, v in count.items():
        if v != 1:
            continue
        tri = (key[0], key[1], key[2])
        px = [nodes[i][0] for i in tri]
        py = [nodes[i][1] for i in tri]
        pz = [nodes[i][2] for i in tri]
        cx, cy, cz = sum(px) / 3.0, sum(py) / 3.0, sum(pz) / 3.0

        def flat(vals: list, ref: float) -> bool:
            return max(vals) - min(vals) <= t and abs(vals[0] - ref) <= t

        match = None
        if flat(pz, lo["z"]):
            match = ("z", -1.0, 1)
        elif flat(px, hi["x"]):
            match = ("x", 1.0, 2)
        elif flat(py, hi["y"]):
            match = ("y", 1.0, 3)
        elif flat(px, lo["x"]):
            match = ("x", -1.0, 4)
        elif flat(pz, hi["z"]):
            match = ("z", 1.0, 5)
        elif flat(py, lo["y"]):
            match = ("y", -1.0, 6)
        elif flat(py, y_left) and cz > hz_base + t:
            match = ("y", -1.0, 6)
        elif flat(py, y_right) and cz > hz_base + t:
            match = ("y", 1.0, 3)
        elif flat(pz, hz_base) and (cy < y_left - t or cy > y_right + t):
            match = ("z", 1.0, 5)
        if match is None:
            raise ValueError(
                f"exterior face {tri} at ({cx:.6g},{cy:.6g},{cz:.6g}) "
                f"matches no boundary plane"
            )
        axis, outward, lab = match
        nx, ny, nz = _outward_normal(nodes, tri)
        n_out = nx if axis == "x" else (ny if axis == "y" else nz)
        if n_out * outward < 0:
            tri = (tri[0], tri[2], tri[1])
        faces.append((tri[0], tri[1], tri[2], lab))
    return faces


def write_medit_mesh(path: str | Path, nodes, tets, faces=()) -> Path:
    """Write nodes/tets/faces as a Medit `.mesh` file for FreeFEM's readmesh3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension\n3\n")
        f.write("Vertices\n")
        f.write(f"{len(nodes)}\n")
        for x, y, z in nodes:
            f.write(f"{x:.12g} {y:.12g} {z:.12g} 1\n")
        f.write("Tetrahedra\n")
        f.write(f"{len(tets)}\n")
        for a, b, c, d in tets:
            f.write(f"{a + 1} {b + 1} {c + 1} {d + 1} 1\n")
        if faces:
            f.write("Triangles\n")
            f.write(f"{len(faces)}\n")
            for a, b, c, lab in faces:
                f.write(f"{a + 1} {b + 1} {c + 1} {lab}\n")
        f.write("End\n")
    return path


def _mesh_volume(nodes, tets) -> float:
    """Total mesh volume (sum of tet volumes); must equal plate + bead."""
    vol = 0.0
    for t in tets:
        vol += abs(_signed_volume(nodes, t))
    return vol


def generate_waam_mesh(path: str | Path, **kw) -> dict:
    """Generate and write the conforming mesh; return a small summary dict."""
    nodes, tets = build_waam_nodes(**kw)
    lx = float(kw["lx"])
    ly_base = float(kw["ly_base"])
    hz_base = float(kw["hz_base"])
    ty_dep = float(kw["ty_dep"])
    hz_dep = float(kw["hz_dep"])
    y_left = 0.5 * ly_base - 0.5 * ty_dep
    y_right = y_left + ty_dep
    faces = classify_boundary_faces(nodes, tets, y_left, y_right, hz_base)
    write_medit_mesh(path, nodes, tets, faces)
    by_label: dict = {}
    for _a, _b, _c, lab in faces:
        by_label[lab] = by_label.get(lab, 0) + 1
    ys = sorted({round(n[1], 12) for n in nodes})
    zs = sorted({round(n[2], 12) for n in nodes})
    return {
        "path": str(path),
        "n_nodes": len(nodes),
        "n_tets": len(tets),
        "nodes": [list(map(float, n)) for n in nodes],
        "tets": [list(map(int, t)) for t in tets],
        "boundary_faces": [[int(a), int(b), int(c)] for a, b, c, _l in faces],
        "boundary_faces_labels": [int(l) for _a, _b, _c, l in faces],
        "n_boundary_faces": len(faces),
        "boundary_faces_by_label": dict(sorted(by_label.items())),
        "n_x": len({round(n[0], 12) for n in nodes}),
        "n_y": len(ys),
        "n_z": len(zs),
        "y_levels": ys,
        "z_levels": zs,
        "domain_volume": _mesh_volume(nodes, tets),
        "expected_volume": lx * ly_base * hz_base + lx * ty_dep * hz_dep,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    info = generate_waam_mesh(
        "/tmp/waam_conforming.mesh",
        lx=0.150, ly_base=0.150, hz_base=0.005,
        ty_dep=0.008, hz_dep=0.104,
        nx_dep=25, ny_dep=3, nz_dep=26,
        nx_base=25, ny_base=8, nz_base=2,
    )
    print(f"nodes={info['n_nodes']} tets={info['n_tets']} "
          f"nx={info['n_x']} ny={info['n_y']} nz={info['n_z']}")
    print(f"boundary faces={info['n_boundary_faces']} by label="
          f"{info['boundary_faces_by_label']}")
    print(f"y levels (mm): {[round(v * 1000, 3) for v in info['y_levels']]}")
