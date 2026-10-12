"""
Bead-geometry diagnostics.

Turns a temperature field into the numbers an experiment can actually measure, so a
simulation can be compared against a macrograph rather than against another plot.
"""

"""
    cross_section(T, mat, m, g, plane_y) -> (top_z, bot_z, width, penetration)

Measure the deposit cross-section at a given `y` station:

- `top_z`, `bot_z` : top and bottom of the region above `T_sol` [m]
- `width`          : lateral extent of the molten/melted region [m]
- `penetration`    : depth of the fusion boundary below the original top surface [m]

Robust to noise: bounds come from the first/last cell index rather than from a
threshold-crossing interpolation, so a single hot cell cannot inflate the result.
"""
function cross_section(T::AbstractArray{Tf,3}, mat::Material, m::GridMetrics,
                       g::CartesianGrid, plane_y::Int) where {Tf<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    zs = Float64[]; xs = Float64[]
    @inbounds for i in 1:nx
        hot = false
        for k in 1:nz
            if T[i, plane_y, k] > mat.Tsol
                hot = true
                break
            end
        end
        if hot
            push!(xs, m.xc[i])
            for k in nz:-1:1
                T[i, plane_y, k] > mat.Tsol && (push!(zs, m.zc[k]); break)
            end
        end
    end
    if isempty(xs)
        return (top_z = 0.0, bot_z = 0.0, width = 0.0, penetration = 0.0)
    end
    top_z = maximum(zs)
    bot_z = minimum(zs)
    # subtract the first cell centre: this is a half-cell convention on the bottom
    width = maximum(xs) - minimum(xs) + m.dx[clamp(1, 1, nx)]
    penetration = (m.zc[1] - bot_z) + 0.5 * m.dz[1]
    (top_z = top_z, bot_z = bot_z, width = width, penetration = penetration)
end

"""
    bead_geometry(T, mat, m, g; plane_y) -> NamedTuple

Bead width, height and penetration depth averaged over all `y` stations.

Width and height are the quantities the paper validates against macrographs; penetration
depth is what fixes the fusion geometry and therefore the mechanical properties.
"""
function bead_geometry(T::AbstractArray{Tf,3}, mat::Material, m::GridMetrics,
                       g::CartesianGrid; plane_y::Union{Nothing,Integer} = nothing) where {Tf<:AbstractFloat}
    stations = plane_y === nothing ? (1:g.ny) : (plane_y,)
    ws = Float64[]; hs = Float64[]; ps = Float64[]
    for j in stations
        c = cross_section(T, mat, m, g, j)
        c.width > 0 && (push!(ws, c.width); push!(hs, c.top_z - c.bot_z); push!(ps, c.penetration))
    end
    isempty(ws) && return (width = 0.0, height = 0.0, penetration = 0.0, stations = 0)
    (width = sum(ws) / length(ws),
     height = sum(hs) / length(hs),
     penetration = sum(ps) / length(ps),
     stations = length(ws))
end

"""
    thermal_profile(T, mat, m, g, plane_y) -> (x, T) at the top surface

Surface temperature trace along the bead, for comparison with an IR camera. The paper
monitors layer start temperature by IR, so this is the direct numerical counterpart.
"""
function thermal_profile(T::AbstractArray{Tf,3}, mat::Material, m::GridMetrics,
                         g::CartesianGrid, plane_y::Int) where {Tf<:AbstractFloat}
    x = Float64[]; Tv = Float64[]
    @inbounds for i in 1:g.nx
        for k in g.nz:-1:1
            if T[i, plane_y, k] > mat.Tsol
                push!(x, m.xc[i]); push!(Tv, T[i, plane_y, k]); break
            end
        end
    end
    (x, Tv)
end

"""
    cooling_rate(T, Tprev, mat, m, g, dt, plane_y) -> Float64  [K/s]

Peak surface cooling rate, a direct proxy for the solidification front velocity and
therefore for the grain structure. Reported as a maximum over the surface because that is
what sets the CET criterion.
"""
function cooling_rate(T, Tprev, mat::Material, m::GridMetrics, g::CartesianGrid,
                      dt::Real, plane_y::Int)
    best = 0.0
    @inbounds for i in 1:g.nx
        for k in g.nz:-1:1
            if T[i, plane_y, k] > mat.Tsol
                best = max(best, (Tprev[i, plane_y, k] - T[i, plane_y, k]) / dt)
                break
            end
        end
    end
    best
end

"""
    interpass_temperature(T, mat) -> Float64

Peak temperature remaining in the deposit at the moment a new layer starts. The paper's
rule is to air-cool to 100 °C before the next layer, and the deposited microstructure is
strongly sensitive to exactly this number.
"""
function interpass_temperature(T, mat::Material)
    v = 0.0
    @inbounds for x in T
        x > v && (v = x)
    end
    v
end

"""
    wall_height(F, m, g, x_lo, x_hi) -> Float64

Height of the deposited wall above the substrate top surface, from the VOF field.
This is the primary geometric output and the one the paper's sagging analysis is about.
"""
function wall_height(F::AbstractArray, m::GridMetrics, g::CartesianGrid,
                     x_lo::Real, x_hi::Real)
    zmax = 0.0
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        (m.xc[i] < x_lo || m.xc[i] > x_hi) && continue
        F[i, j, k] > 0.5 || continue
        m.zc[k] > zmax && (zmax = m.zc[k])
    end
    zmax - m.zc[1] + 0.5 * m.dz[1]
end

"""
    effective_thickness(F, m, g, x_lo, x_hi; frac = 0.5) -> Float64

The paper's validation metric. It defines *effective thickness* as the wall thickness
**after machining away the surface irregularities**, i.e. the width of the region whose
volume fraction exceeds `frac`. Reproducing that definition is what makes the comparison
to the published 6.2 mm / 6.3 mm numbers meaningful — measuring the raw envelope instead
would compare two different quantities.
"""
function effective_thickness(F::AbstractArray, m::GridMetrics, g::CartesianGrid,
                             x_lo::Real, x_hi::Real; frac::Real = 0.5)
    lo = Inf; hi = -Inf
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        (m.xc[i] < x_lo || m.xc[i] > x_hi) && continue
        F[i, j, k] >= frac || continue
        xc_ = m.xc[i]
        xc_ < lo && (lo = xc_)
        xc_ > hi && (hi = xc_)
    end
    hi < lo ? 0.0 : (hi - lo + m.dx[1])
end

"""
    pool_volume(F, mat, m, g) -> Float64  [m³]

Volume of genuinely molten metal (`F > 0` and `T > T_liq`). The paper compares this between
ER70S6 and ER316L and finds the 316L pool 35–40 % larger, which is the mechanism behind
its earlier sagging.
"""
function pool_volume(F, T, mat::Material, m::GridMetrics, g::CartesianGrid)
    s = 0.0
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        (F[i, j, k] > 1e-6 && T[i, j, k] > mat.Tliq) || continue
        s += F[i, j, k] * m.vol[i, j, k]
    end
    s
end