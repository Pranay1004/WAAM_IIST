"""
    CartesianGrid

Non-uniform, orthogonal Cartesian grid with **staggered (MAC)** storage.

Node coordinates `xf, yf, zf` are supplied at the *faces* (length `n+1`), and every
other metric is derived exactly, so a stretched mesh is a first-class citizen rather
than a smeared uniform one. This matters: the paper resolves the fusion zone with
0.6 mm cells and grades to 1.8 mm in the far field, i.e. a 3:1 expansion ratio.

Storage layout follows Julia's column-major convention with the **first index
contiguous**, so the canonical sweep order is `for k, for j, for i`. Both velocity
staggering and cell-centred scalars are provided.

    Field            Array shape          Location
    -------------------------------------------------------------------
    u                (nx+1, ny,    nz)   x-faces,  (xf[i], yc[j], zc[k])
    v                (nx,    ny+1, nz)   y-faces,  (xc[i], yf[j], zc[k])
    w                (nx,    ny,    nz+1) z-faces,  (xc[i], yc[j], zf[k])
    p,T,h,F,f_s      (nx,    ny,    nz)   cell centres (xc[i], yc[j], zc[k])
"""
struct CartesianGrid
    xf::Vector{Float64}   # x-node coordinates, length nx+1
    yf::Vector{Float64}
    zf::Vector{Float64}
    nx::Int
    ny::Int
    nz::Int
    function CartesianGrid(xf::AbstractVector{<:Real},
                          yf::AbstractVector{<:Real},
                          zf::AbstractVector{<:Real})
        xf = collect(Float64, xf)
        yf = collect(Float64, yf)
        zf = collect(Float64, zf)
        for (nm, f) in (("xf", xf), ("yf", yf), ("zf", zf))
            length(f) >= 2 || throw(ArgumentError("$nm must have >= 2 nodes"))
            @inbounds for i in 2:length(f)
                f[i] > f[i-1] || throw(ArgumentError("$nm must be strictly increasing (got $(f[i-1]), $(f[i]) at $i)"))
            end
        end
        new(xf, yf, zf, length(xf) - 1, length(yf) - 1, length(zf) - 1)
    end
end

Base.size(g::CartesianGrid) = (g.nx, g.ny, g.nz)
Base.length(g::CartesianGrid) = g.nx * g.ny * g.nz
Base.ndims(::CartesianGrid) = 3

"""
    cell_centers(f) -> Vector

Midpoints of a node coordinate vector.
"""
cell_centers(f::Vector{Float64}) = @. 0.5 * (f[1:end-1] + f[2:end])

"""
    widths(f) -> Vector

Cell widths (node spacings) of a node coordinate vector.
"""
widths(f::Vector{Float64}) = (f[2:end] .- f[1:end-1])

# ---------------------------------------------------------------------------
# Derived geometry (computed once, cached by the caller via `GridMetrics`)
# ---------------------------------------------------------------------------

"""
    GridMetrics

Pre-computed exact geometry for a `CartesianGrid`. Constructing this once and
threading it through the kernels keeps hot loops free of allocation and of
re-derivation.
"""
struct GridMetrics
    xc::Vector{Float64}
    yc::Vector{Float64}
    zc::Vector{Float64}
    dx::Vector{Float64}      # cell widths, length nx
    dy::Vector{Float64}
    dz::Vector{Float64}
    dxf::Vector{Float64}     # x-face spacings, length nx+1 (dxf[i+1] == dx[i])
    dyf::Vector{Float64}
    dzf::Vector{Float64}
    dxc::Vector{Float64}     # centre-to-centre distance ACROSS face f, length nx+1
    dyc::Vector{Float64}
    dzc::Vector{Float64}
    vol::Array{Float64,3}    # cell volumes
    ax::Array{Float64,3}     # x-face area at (i,j,k), i in 2:nx
    ay::Array{Float64,3}     # y-face area, j in 2:ny
    az::Array{Float64,3}     # z-face area, k in 2:nz
end

function GridMetrics(g::CartesianGrid)
    xc, yc, zc = cell_centers(g.xf), cell_centers(g.yf), cell_centers(g.zf)
    dx, dy, dz = widths(g.xf), widths(g.yf), widths(g.zf)
    dxf = similar(g.xf); dxf[1] = dx[1]; dxf[end] = dx[end]
    @inbounds @simd for i in 2:g.nx
        dxf[i] = dx[i-1]
    end
    dyf = similar(g.yf); dyf[1] = dy[1]; dyf[end] = dy[end]
    @inbounds @simd for j in 2:g.ny
        dyf[j] = dy[j-1]
    end
    dzf = similar(g.zf); dzf[1] = dz[1]; dzf[end] = dz[end]
    @inbounds @simd for k in 2:g.nz
        dzf[k] = dz[k-1]
    end

    # Centre-to-centre distance ACROSS each interior face f (between cells f-1 and f):
    #   d_c[f] = ( width(cell f-1) + width(cell f) ) / 2 = ( dxf[f] + dxf[f+1] ) / 2
    #
    # This is the distance the two-point flux approximation must divide by. Dividing by
    # ONE cell width instead (which is what this code used to do) is exact on a uniform
    # axis, where both widths are Δx, and wrong by up to ~37 % on the paper's 3:1 graded
    # mesh -- the only kind of mesh the WAAM cases ever run on. It made a linear steady
    # profile fail to be a discrete solution in the grading-transition region, i.e. the
    # conduction operator was inconsistent exactly where the fusion zone grades out.
    #
    # The boundary entries have no second cell; they are half a width and are never read
    # (only interior faces carry a two-point flux), but they are filled rather than left
    # undefined -- see the `ax` note below for why that matters.
    dxc = similar(g.xf)
    dyc = similar(g.yf)
    dzc = similar(g.zf)
    @inbounds for i in 1:g.nx
        dxc[i] = 0.5 * (dxf[i] + dxf[i + 1])
    end
    @inbounds for j in 1:g.ny
        dyc[j] = 0.5 * (dyf[j] + dyf[j + 1])
    end
    @inbounds for k in 1:g.nz
        dzc[k] = 0.5 * (dzf[k] + dzf[k + 1])
    end
    dxc[g.nx + 1] = 0.5 * dxf[end]
    dyc[g.ny + 1] = 0.5 * dyf[end]
    dzc[g.nz + 1] = 0.5 * dzf[end]

    vol  = Array{Float64,3}(undef, g.nx, g.ny, g.nz)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        vol[i, j, k] = dx[i] * dy[j] * dz[k]
    end

    # NOTE: the n+1-th plane must be filled too. Leaving it undefined silently
    # poisons every face flux touching the domain boundary.
    ax = Array{Float64,3}(undef, g.nx + 1, g.ny, g.nz)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx+1
        ax[i, j, k] = dy[j] * dz[k]
    end

    ay = Array{Float64,3}(undef, g.nx, g.ny + 1, g.nz)
    @inbounds for k in 1:g.nz, j in 1:g.ny+1, i in 1:g.nx
        ay[i, j, k] = dx[i] * dz[k]
    end

    az = Array{Float64,3}(undef, g.nx, g.ny, g.nz + 1)
    @inbounds for k in 1:g.nz+1, j in 1:g.ny, i in 1:g.nx
        az[i, j, k] = dx[i] * dy[j]
    end

    GridMetrics(xc, yc, zc, dx, dy, dz, dxf, dyf, dzf, dxc, dyc, dzc, vol, ax, ay, az)
end

metrics(g::CartesianGrid) = GridMetrics(g)

# ---------------------------------------------------------------------------
# Uniform grids & grading
# ---------------------------------------------------------------------------

"""
    uniform_grid(nx, ny, nz, Lx, Ly, Lz; origin=(0,0,0))

A uniform grid spanning `[origin, origin+L]`. `L` may be omitted for each axis,
in which case the extent defaults to 1.
"""
function uniform_grid(nx::Integer, ny::Integer, nz::Integer,
                      Lx::Real, Ly::Real, Lz::Real;
                      origin::NTuple{3,<:Real} = (0.0, 0.0, 0.0))
    x0, y0, z0 = Float64.(origin)
    xf = collect(range(x0, x0 + Lx; length = nx + 1))
    yf = collect(range(y0, y0 + Ly; length = ny + 1))
    zf = collect(range(z0, z0 + Lz; length = nz + 1))
    CartesianGrid(xf, yf, zf)
end
uniform_grid(nx::Integer, ny::Integer, nz::Integer) = uniform_grid(nx, ny, nz, 1, 1, 1)

"""
    graded_axis(x0, x1, h_fine, h_coarse; ratio=1.3, h_fine_end=:last)

Build a 1-D node distribution from `x0` to `x1` whose cell width grows geometrically
from `h_fine` to `h_coarse`.

This is the paper's mesh strategy in one function: `h_fine = 0.6 mm` through the
fusion zone expanding to `h_coarse = 1.8 mm` in the far field. `ratio` is the
preferred geometric growth factor; it is automatically relaxed if honouring it
would leave the cells too coarse.

`h_fine_end` selects which side stays fine: `:last` keeps `h_fine` adjacent to `x1`
(useful when `x1` is the weld axis), `:first` keeps it adjacent to `x0` (the
substrate surface), `:both` fine at both ends.

If `h_fine > h_coarse` the arguments are silently swapped — grading is a *coarsening*
away from the feature of interest and getting the direction wrong is an easy and
expensive mistake.
"""
function graded_axis(x0::Real, x1::Real, h_fine::Real, h_coarse::Real;
                     ratio::Real = 1.3, h_fine_end::Symbol = :last)
    a, b = Float64(x0), Float64(x1)
    lo, hi = minmax(a, b)
    L = hi - lo
    L > 0 || throw(ArgumentError("graded_axis needs a non-degenerate span"))
    hf, hc = Float64(h_fine), Float64(h_coarse)
    # grading is a *coarsening* away from the feature of interest; getting the
    # direction wrong is an easy and expensive mistake, so normalise it away here
    hf, hc = minmax(hf, hc)
    hf > 0 && hc > 0 || throw(ArgumentError("cell widths must be positive"))
    r = max(Float64(ratio), 1.0 + 1e-12)
    h_fine_end in (:last, :first, :both) ||
        throw(ArgumentError("h_fine_end must be :last, :first or :both"))

# Emit the cell widths spanning `L`: geometric from h_fine up to h_coarse, then
    # uniform coarse cells to close.
    #
    # The subtle part is the closing region. Those cells must be no *finer* than the
    # last graded cell, otherwise the widths are non-monotone and the grading is
    # meaningless. Rather than accept that, back the geometric prefix off one cell at a
    # time until the remainder can be tiled by cells that are both <= h_coarse and
    # >= the last graded width. This needs no rescaling afterwards, so the requested
    # ratio r survives exactly.
    function _graded_widths(L::Float64, hf::Float64, hc::Float64, r::Float64)
        cells = Float64[]
        rem = L
        w = hf
        while rem - w >= hf
            push!(cells, w)
            rem -= w
            w = min(w * r, hc)
        end
        while !isempty(cells)
            n = max(1, ceil(Int, rem / hc - 1e-12))
            if rem / n >= w * (1 - 1e-12)
                break
            end
            w = pop!(cells)          # hand the last graded cell back to the remainder
            rem += w
        end
        n = max(1, ceil(Int, rem / hc - 1e-12))
        append!(cells, fill(rem / n, n))
        cells
    end

    cells = if h_fine_end === :both
        # Fine at both ends, coarsest at the seam: grade one half and mirror it, then
        # widen the central cell to absorb the seam so the total span is exact.
        half = _graded_widths(L / 2, hf, hc, r)
        seam = L - 2 * sum(half)
        if seam > 0
            half[end] += seam
        end
        vcat(half, reverse(half[1:end-1]))
    else
        w = _graded_widths(L, hf, hc, r)
        h_fine_end === :last ? reverse!(w) : w
    end

    isempty(cells) && (cells = [L])
    any(c -> c <= 0, cells) &&
        throw(ArgumentError("graded_axis produced a non-positive cell width"))
    maximum(cells) <= hc * (1 + 1e-9) ||
        throw(ArgumentError("graded_axis exceeded h_coarse: $(maximum(cells)) > $hc"))

    nodes = Vector{Float64}(undef, length(cells) + 1)
    nodes[1] = 0.0
    c = 0.0
    @inbounds for i in eachindex(cells)
        c += cells[i]
        nodes[i + 1] = c
    end
    nodes .*= L / nodes[end]          # kill accumulated rounding
    nodes .+= lo
    a > b && reverse!(nodes)
    nodes[end] = hi                   # exact endpoint
    nodes
end

# ---------------------------------------------------------------------------
# Geometry invariants  (verification test V1)
# ---------------------------------------------------------------------------

"""
    face_closure_error(g, m) -> Float64

Maximum over cells of ``|Σ_f Sf| / V_c``: the relative violation of the discrete
divergence theorem on the unit cell. Must be at machine epsilon for a correct
mesh; a non-zero value means the face-area bookkeeping is wrong and every
divergence in the solver is silently biased.

Face normals on an axis-aligned Cartesian grid are signed axis vectors with
magnitude equal to the *transverse* face area, and the outward orientation at a
cell's low face is `-e_axis`.
"""
function face_closure_error(g::CartesianGrid, m::GridMetrics = GridMetrics(g))
    worst = 0.0
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        sfx = m.ax[1, j, k] - m.ax[i + 1, j, k]
        sfy = m.ay[i, 1, k] - m.ay[i, j + 1, k]
        sfz = m.az[i, j, 1] - m.az[i, j, k + 1]
        worst = max(worst, abs(sfx), abs(sfy), abs(sfz))
    end
    worst / maximum(m.vol)
end

"""
    volume_identity_error(g, m) -> Float64

Maximum relative violation of ``V_c == (1/3) Σ_f (S_f · x_f)``, the exact discrete
divergence theorem applied to ``r``. A correct Cartesian grid satisfies this to
machine precision.
"""
function volume_identity_error(g::CartesianGrid, m::GridMetrics = GridMetrics(g))
    worst = 0.0
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        # Discrete divergence theorem applied to the field r = (x,y,z):
        #   ∇·r = 3  ⇒  3 V_c = Σ_f (S_f · x_f)
        # Outward normals: low face is -e, high face is +e.
        sx = -m.ax[i, j, k] * g.xf[i] + m.ax[i + 1, j, k] * g.xf[i + 1]
        sy = -m.ay[i, j, k] * g.yf[j] + m.ay[i, j + 1, k] * g.yf[j + 1]
        sz = -m.az[i, j, k] * g.zf[k] + m.az[i, j, k + 1] * g.zf[k + 1]
        vdiv3 = (sx + sy + sz) / 3
        worst = max(worst, abs(vdiv3 - m.vol[i, j, k]) / m.vol[i, j, k])
    end
    worst
end

"""
    uniform_step_error(g) -> Float64

Max relative deviation of cell widths from `dx[1]`; zero for a uniform grid.
"""
function uniform_step_error(g::CartesianGrid)
    worst = 0.0
    for (f, n) in ((g.xf, g.nx), (g.yf, g.ny), (g.zf, g.nz))
        h = widths(f)
        for i in 2:n
            worst = max(worst, abs(h[i] - h[1]) / h[1])
        end
    end
    worst
end

"""
    grid_summary(g) -> NamedTuple

Human-readable grid description for run manifests.

Named `grid_summary` rather than `summary` so it cannot be confused with
`Statistics.summary` at a call site.
"""
function grid_summary(g::CartesianGrid)
    dx, dy, dz = widths(g.xf), widths(g.yf), widths(g.zf)
    (nx = g.nx, ny = g.ny, nz = g.nz, cells = length(g),
     extent = (g.xf[1], g.xf[end], g.yf[1], g.yf[end], g.zf[1], g.zf[end]),
     dx_min = minimum(dx), dx_max = maximum(dx),
     dy_min = minimum(dy), dy_max = maximum(dy),
     dz_min = minimum(dz), dz_max = maximum(dz),
     expansion = max(maximum(dx) / minimum(dx), maximum(dy) / minimum(dy),
                     maximum(dz) / minimum(dz)))
end

Base.show(io::IO, g::CartesianGrid) =
    print(io, "CartesianGrid($(g.nx)×$(g.ny)×$(g.nz) = $(length(g)) cells, ",
          "expansion ", round(maximum(widths(g.xf)) / minimum(widths(g.xf)), digits = 2), "×)")