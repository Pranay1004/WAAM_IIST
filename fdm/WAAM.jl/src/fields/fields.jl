"""
Fields: cell-centred scalars, face-centred fluxes, and staggered vectors.

Kept deliberately thin and concrete (no abstract field hierarchy). For a solver whose
hot loops must run for hours, dispatch overhead and type instability matter more than
elegance: these are concrete `Array` wrappers with hand-written kernels, so every
operation is inlined and `@inbounds @simd` is effective.
"""

"""
    VolumeField{T}

Cell-centred scalar field of shape `(nx, ny, nz)`.
"""
struct VolumeField{T<:AbstractFloat}
    data::Array{T,3}
    grid::CartesianGrid
    VolumeField{T}(data::Array{T,3}, grid::CartesianGrid) where {T} = new{T}(data, grid)
end
VolumeField(data::Array{T,3}, grid::CartesianGrid) where {T<:AbstractFloat} =
    VolumeField{T}(data, grid)
VolumeField(a::AbstractArray{T,3}, g::CartesianGrid) where {T<:AbstractFloat} =
    VolumeField{T}(convert(Array{T,3}, a), g)
VolumeField(g::CartesianGrid) = VolumeField(zeros(g.nx, g.ny, g.nz), g)
similar_field(f::VolumeField) = VolumeField(similar(f.data), f.grid)

Base.size(f::VolumeField) = size(f.data)
Base.getindex(f::VolumeField, i::Int, j::Int, k::Int) = f.data[i, j, k]
Base.setindex!(f::VolumeField{T}, v, i::Int, j::Int, k::Int) where {T} = (f.data[i, j, k] = T(v))
Base.fill!(f::VolumeField{T}, v) where {T} = fill!(f.data, T(v))
Base.copy(f::VolumeField) = VolumeField(copy(f.data), f.grid)

# Fallback 3-arg getindex so a VolumeField can index another VolumeField.
Base.getindex(f::VolumeField, g::VolumeField, I::Vararg{Int,3}) = getindex(f.data, g.data, I...)
Base.setindex!(f::VolumeField{T}, v, g::VolumeField, I::Vararg{Int,3}) where {T} =
    setindex!(f.data, T(v), g.data, I...)

"""
    SurfaceField

Face-centred scalar flux field. Axis `1` carries the x-normal faces (shape
`nx+1 × ny × nz`), axis `2` the y-faces, axis `3` the z-faces. Storing all three
axes in one array keeps the "apply the same physics to every axis" kernels a single
`@simd` loop instead of three near-duplicates.
"""
struct SurfaceField{T<:AbstractFloat, A<:AbstractArray{T,3}}
    data::A
    grid::CartesianGrid
    axis::Int      # which face direction this instance represents
end
function SurfaceField(g::CartesianGrid, axis::Integer)
    axis in 1:3 || throw(ArgumentError("axis must be 1, 2 or 3"))
    n = (g.nx, g.ny, g.nz)[axis]
    SurfaceField(zeros(n + 1, (g.nx, g.ny, g.nz)[mod1(axis + 1, 3)],
                       (g.nx, g.ny, g.nz)[mod1(axis + 2, 3)]), g, Int(axis))
end

"""
    VectorField

Staggered MAC velocity: `u` on x-faces, `v` on y-faces, `w` on z-faces.
"""
struct VectorField{T<:AbstractFloat}
    u::Array{T,3}
    v::Array{T,3}
    w::Array{T,3}
    grid::CartesianGrid
end
VectorField(g::CartesianGrid) = VectorField(zeros(g.nx + 1, g.ny, g.nz),
                                            zeros(g.nx, g.ny + 1, g.nz),
                                            zeros(g.nx, g.ny, g.nz + 1), g)
VectorField(::Type{T}, g::CartesianGrid) where {T<:AbstractFloat} =
    VectorField(zeros(T, g.nx + 1, g.ny, g.nz), zeros(T, g.nx, g.ny + 1, g.nz),
                zeros(T, g.nx, g.ny, g.nz + 1), g)
Base.copy(f::VectorField{T}) where {T} = VectorField(copy(f.u), copy(f.v), copy(f.w), f.grid)
Base.zero(f::VectorField{T}) where {T} = VectorField(zero(f.u), zero(f.v), zero(f.w), f.grid)

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

"""
    cell_volume_sum(g, a) -> Float64

``Σ_c a_c V_c`` — the volume integral of a cell-centred field. The universal building
block for mass, energy and volume budgets.
"""
function cell_volume_sum(g::CartesianGrid, a::AbstractArray, m::GridMetrics = GridMetrics(g))
    s = 0.0
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        s += a[i, j, k] * m.vol[i, j, k]
    end
    s
end
cell_volume_sum(g::CartesianGrid, f::VolumeField, m::GridMetrics = GridMetrics(g)) =
    cell_volume_sum(g, f.data, m)

"""
    cell_mean(g, a) -> Float64

Volume-weighted mean of a cell-centred field.
"""
function cell_mean(g::CartesianGrid, a::AbstractArray, m::GridMetrics = GridMetrics(g))
    cell_volume_sum(g, a, m) / sum(m.vol)
end
cell_mean(g::CartesianGrid, f::VolumeField, m::GridMetrics = GridMetrics(g)) =
    cell_mean(g, f.data, m)

"""
    field_range(f) -> (min, max)

Extrema of a field. Used by the VOF boundedness test (V17).
"""
function field_range(a::AbstractArray)
    mn, mx = Inf, -Inf
    @inbounds for x in a
        mn = min(mn, x)
        mx = max(mx, x)
    end
    (mn, mx)
end
field_range(f::VolumeField) = field_range(f.data)

# ---------------------------------------------------------------------------
# Global shift / clamping helpers used by TVD reconstruction
# ---------------------------------------------------------------------------

"""
    shift!(dst, src, s)

`dst = src .+ s` elementwise, for the 0th-order upwind flux
``F^{n+1} = F^n (a^{n+1} + s)`` where ``s = 1 - dt (div u)``.
"""
function shift!(dst::AbstractArray, src::AbstractArray, s::Real)
    T = float(eltype(dst))
    v = T(s)
    @inbounds @simd for idx in eachindex(dst, src)
        dst[idx] = src[idx] + v
    end
    dst
end

"""
    clamp_bounded!(f, lo, hi, tol)

Clip a field into `[lo - tol, hi + tol]`. Applied after every VOF update; ``tol = 0``
for a strictly conservative scheme, small positive for explicit schemes where
round-off can push a fraction a few ULPs out of range.
"""
function clamp_bounded!(a::AbstractArray, lo::Real, hi::Real, tol::Real = 0.0)
    T = float(eltype(a))
    l = T(lo - tol); h = T(hi + tol)
    @inbounds @simd for idx in eachindex(a)
        a[idx] = min(max(a[idx], l), h)
    end
    a
end
clamp_bounded!(f::VolumeField, args...) = clamp_bounded!(f.data, args...)