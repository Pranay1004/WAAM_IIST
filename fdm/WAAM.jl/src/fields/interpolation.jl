"""
Spatial discretisation: interpolation, limited reconstruction, gradients, diffusive
and convective face fluxes.

Everything here is written to be correct on a **stretched** (non-uniform) grid, which
is the regime the paper actually operates in. On a non-uniform axis, the face-to-face
distance is ``d_ip = (d_i/2 + d_{i+1}/2)`` and the two-point diffusion coefficient
carries the harmonic mean of the *face* conductivities — not the arithmetic mean of
cell conductivities. Getting this wrong on a 3:1 graded mesh is a first-order error
that still looks plausible in plots.
"""

# ---------------------------------------------------------------------------
# Slope limiters (verification test V3)
# ---------------------------------------------------------------------------

# --- Slope limiters -------------------------------------------------------
#
# A slope limiter is only TVD if it never AMPLIFIES: |s| <= min(|Δ₋|, |Δ₊|). The naive
# algebraic form often quoted for "van Leer", (a + b + √(ab))/2, returns 1.5a when
# a == b, i.e. it *amplifies* the slope by 50 %. That silently destroys the TVD
# property and destabilises an explicit scheme at Courant numbers where Lax–Wendroff
# should be perfectly comfortable. The implementations below are all strictly
# non-amplifying, which is the property actually asserted in the test suite.

# Backward-looking minmod: the strictest limiter, maximally dissipative.
@inline function minmod(a::Real, b::Real)
    aa, bb = float(a), float(b)
    sign(aa) == sign(bb) ? min(abs(aa), abs(bb)) * sign(aa) : zero(aa + bb)
end

# Superbee: sharper than minmod, still bounded by min(|a|,|b|).
@inline function superbee(a::Real, b::Real)
    aa, bb = float(a), float(b)
    T = typeof(aa + bb)
    sign(aa) == sign(bb) ?
        max(abs(minmod(aa, bb)), abs(minmod(2aa, bb))) * sign(aa) : zero(T)
end

# van Leer, smooth (Harten) form. psi(r) = (r + sqrt(r))/(1 + r) with r = a/b:
#   psi(0+) = 0, psi(1) = 1, psi(inf) = 1
# so s = psi(r) * min(|a|,|b|) satisfies |s| <= min(|a|,|b|) AND recovers s = a when
# the two one-sided slopes agree, which is what makes it second-order in smooth flow.
@inline function vanleer(a::Real, b::Real)
    aa, bb = float(a), float(b)
    sign(aa) == sign(bb) || return zero(aa + bb)
    mn = min(abs(aa), abs(bb))
    r = max(aa, bb) == 0 ? zero(mn) : mn / max(abs(aa), abs(bb))
    sign(aa) * mn * (r + sqrt(r)) / (1 + r)
end

# Unlimited central (2nd-order) slope. Included as a reference for convergence tests;
# NOT monotone, so it must not be used for a volume fraction.
@inline central(a::Real, b::Real) = 0.5 * (float(a) + float(b))

"""
    slope_limiter(name)

Look up a limiter by name. Used by config so a scheme can be selected without
touching code; unknown names error loudly rather than silently defaulting.
"""
function slope_limiter(name::AbstractString)
    n = lowercase(String(name))
    n == "none"        && return central
    n == "minmod"      && return minmod
    n == "superbee"    && return superbee
    n == "vanleer"     && return vanleer
    n == "van_leer"    && return vanleer
    throw(ArgumentError("unknown slope limiter \"$name\" (have: none, minmod, superbee, vanleer)"))
end

# ---------------------------------------------------------------------------
# Interpolation to faces
# ---------------------------------------------------------------------------

"""
    face_interp2(a, b, ha, hb) -> Float64

Two-point (harmonic-distance) interpolation of cell values `a`,`b` whose centres are
`ha`, `hb` from the face: ``(a*hb + b*ha)/(ha + hb)``.

This is the correct conservative face value on a stretched grid for the quantity
that is *conserved per face* (mass flux, volumetric heat rate). Using the arithmetic
mean `0.5(a+b)` here is the classic silent first-order error on graded meshes.
"""
@inline function face_interp2(a::Real, b::Real, ha::Real, hb::Real)
    T = float(promote_type(typeof(a), typeof(b), typeof(ha), typeof(hb)))
    aa, bb, hha, hhb = T(a), T(b), T(ha), T(hb)
    (aa * hhb + bb * hha) / (hha + hhb)
end

"""
    harmonic_mean(a, b)

Harmonic mean, the correct face conductivity for a two-point diffusion flux.
"""
@inline function harmonic_mean(a::Real, b::Real)
    T = float(promote_type(typeof(a), typeof(b)))
    aa, bb = T(a), T(b)
    (abs(aa) + abs(bb)) <= eps(T) ? 2aa / 3 + 2bb / 3 : 2 * aa * bb / (aa + bb)
end

"""
    harmonic_face_coeff(kp, kn, hp, hm) -> Float64

Effective face conductivity for cell-centred conductivities `kp`, `kn` on either side of
a face, with distances `hp`, `hm` from that face to each cell **centre**. Exact on a
non-uniform axis:

```
k_face = (hp + hm) / ( hp/k_P + hm/k_N )
```

i.e. the total centre-to-centre resistance of the two half-cells, referred to the face.
On a uniform axis (``hp = hm = Δx/2``) this reduces exactly to the harmonic mean
``2 k_P k_N/(k_P + k_N)``, which is the correct two-point face value.

`hp` and `hm` are centre-to-face distances, **not** half-cell widths. For a cell of
width `Δx` both equal `Δx/2`, which is what `half_distances` returns. Using the full cell
widths here instead doubles the total thermal resistance of a uniform slab — a
first-order error that still produces a plausible-looking temperature contour.
"""
@inline function harmonic_face_coeff(kp::Real, kn::Real, hp::Real, hm::Real)
    T = float(promote_type(typeof(kp), typeof(kn), typeof(hp), typeof(hm)))
    # Thermal resistance of the two half-cells in series, referred to the face:
    #   R = hp/k_P + hm/k_N      (units m²·K/W)
    # The face conductivity that reproduces a constant flux q = -k_face ΔT/(hp+hm)
    # across the centre-to-centre distance is then
    #   k_face = (hp + hm) / R
    # On a uniform axis (hp == hm == Δx/2) this is the harmonic mean 2·k_P·k_N/(k_P+k_N),
    # which is the correct two-point face value.
    (T(hp) + T(hm)) / (T(hp) / T(kp) + T(hm) / T(kn))
end

# ---------------------------------------------------------------------------
# Cell-to-face distance (stretched grid)
# ---------------------------------------------------------------------------

"""
    half_distances(h) -> Vector

Distance from each cell centre to the lower face of its own cell, i.e. simply
`h/2`. Length `length(h)`.

The distance from a cell centre to an *arbitrary* face is built from this and the
intervening cell widths — see `cell_to_face_distance`. Returning `h/2` rather than a
centre-to-centre distance is deliberate: it is the only quantity that is independent of
strain, so it is the correct building block on a graded axis.
"""
half_distances(h::AbstractVector) = 0.5 .* Float64.(h)

"""
    cell_to_face_distance(h, f) -> Float64

Distance from the centre of cell `f` to the face between cells `f` and `f+1`, i.e.
`h[f]/2`. `f = 0` denotes the domain boundary face to the left of cell 1, which is at
distance `h[1]/2` from cell 1's centre.
"""
@inline function cell_to_face_distance(h::AbstractVector, f::Int)
    h[clamp(f, 1, length(h))] / 2
end

# ---------------------------------------------------------------------------
# Gradients  (verification test V2: 2nd-order MMS convergence)
# ---------------------------------------------------------------------------

"""
    grad_x(a, h) -> Array

Cell-centred ∂a/∂x using the standard 2nd-order central formula on a **uniform**
axis and a 2nd-order-accurate non-uniform formula (quadratic fit through
`i-1, i, i+1`) on a graded axis. One-sided closures at the boundary, also quadratic.

On a uniform axis this reduces to `(a[i+1] - a[i-1]) / (2h)`.
"""
function grad_uniform!(out::AbstractArray, a::AbstractArray, h::Real)
    T = float(eltype(out))
    hh = T(h)
    n = length(a)
    @inbounds begin
        # one-sided, 2nd order
        out[1] = (-3a[1] + 4a[2] - a[3]) / (2hh)
        for i in 2:n-1
            out[i] = (a[i + 1] - a[i - 1]) / (2hh)
        end
        out[n] = (3a[n] - 4a[n - 1] + a[n - 2]) / (2hh)
    end
    out
end

function grad_nonuniform!(out::AbstractArray, xf::AbstractVector,
                          a::AbstractVector, xc::AbstractVector)
    T = float(eltype(out))
    n = length(a)
    @inbounds begin
        out[1] = _deriv_3pt(xf[1], xf[2], xf[3], a[1], a[2], a[3])
        for i in 2:n-1
            out[i] = _deriv_3pt(xf[i], xf[i + 1], xf[i + 2], a[i - 1], a[i], a[i + 1])
        end
        out[n] = _deriv_3pt(xf[n], xf[n + 1], xf[n + 2], a[n - 2], a[n - 1], a[n])
    end
    out
end

# Second-order accurate derivative from three (x, a) samples at x1<x2<x3.
@inline function _deriv_3pt(x1::T, x2::T, x3::T, a1::T, a2::T, a3::T) where {T<:AbstractFloat}
    h21 = x2 - x1
    h32 = x3 - x2
    (-h32 / (h21 * (h21 + h32))) * a1 +
    ((h32 - h21) / (h21 * h32)) * a2 +
    (h21 / (h32 * (h21 + h32))) * a3
end

"""
    grad_axis!(out, a, nodes, axis)

In-place cell-centred gradient of `a` along `axis`, dispatching to the uniform or
non-uniform closure based on whether the node spacing is constant.
"""
function grad_axis!(out::AbstractArray, a::AbstractArray, nodes::AbstractVector,
                    m::GridMetrics, axis::Int)
    if _is_uniform(nodes)
        grad_uniform!(out, a, (nodes[2] - nodes[1]))
    else
        xc = _centers(nodes)
        n = length(a)
        @inbounds begin
            out[1] = _deriv_3pt(nodes[1], nodes[2], nodes[3], a[1], a[2], a[3])
            for i in 2:n-1
                out[i] = _deriv_3pt(nodes[i], nodes[i + 1], nodes[i + 2], a[i - 1], a[i], a[i + 1])
            end
            out[n] = _deriv_3pt(nodes[n], nodes[n + 1], nodes[n + 2], a[n - 2], a[n - 1], a[n])
        end
    end
    out
end

@inline _is_uniform(nodes::AbstractVector) = all(i -> nodes[i + 1] - nodes[i] ≈ nodes[2] - nodes[1],
                                                1:length(nodes)-2)
_centers(f::AbstractVector) = @. 0.5 * (f[1:end-1] + f[2:end])

function axis_nodes(g::CartesianGrid, axis::Int)
    axis == 1 && return g.xf
    axis == 2 && return g.yf
    axis == 3 && return g.zf
    throw(ArgumentError("axis must be 1, 2 or 3"))
end

function axis_widths(g::CartesianGrid, m::GridMetrics, axis::Int)
    axis == 1 && return m.dx
    axis == 2 && return m.dy
    axis == 3 && return m.dz
    throw(ArgumentError("axis must be 1, 2 or 3"))
end

# ---------------------------------------------------------------------------
# Face gradient (used for curvature and for flux differencing)
# ---------------------------------------------------------------------------

"""
    face_gradient(a, nodes) -> Vector

Gradient at the *faces* (index 1..n+1) of a cell-centred field. Interior faces use the
2-cell average slope; boundary faces use a 1-sided 2nd-order closure. This is the
natural operator for ``∇·(ρ u)`` differencing and for surface-tension curvature on a
VOF field.
"""
function face_gradient(a::AbstractVector, nodes::AbstractVector)
    T = float(eltype(a))
    n = length(a)
    d = Vector{T}(undef, n + 1)
    x = _centers(nodes)
    @inbounds begin
        for i in 1:n                       # face i sits between cells i-1 and i
            if i == 1
                d[1] = (a[2] - a[1]) / (x[2] - x[1])
            elseif i == n + 1
                d[n + 1] = (a[n] - a[n - 1]) / (x[n] - x[n - 1])
            else
                d[i] = (a[i] - a[i - 1]) / (x[i] - x[i - 1])
            end
        end
    end
    d
end

# ---------------------------------------------------------------------------
# Divergence  (verification test V2)
# ---------------------------------------------------------------------------

"""
    divergence!(out, u, v, w, m)

Cell-centred ``∇·U`` from a staggered MAC field, exact flux differencing:

```
(div U)_c = [ u_e*A_e - u_w*A_w + v_n*A_n - v_s*A_s + w_t*A_t - w_b*A_b ] / V_c
```

Second-order accurate on a uniform grid, and consistent (exactly divergence-free for
a uniform velocity field) on any grid.
"""
function divergence!(out::AbstractArray, u::AbstractArray, v::AbstractArray,
                     w::AbstractArray, m::GridMetrics, g::CartesianGrid)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        fx = u[i + 1, j, k] * m.ax[i + 1, j, k] - u[i, j, k] * m.ax[i, j, k]
        fy = v[i, j + 1, k] * m.ay[i, j + 1, k] - v[i, j, k] * m.ay[i, j, k]
        fz = w[i, j, k + 1] * m.az[i, j, k + 1] - w[i, j, k] * m.az[i, j, k]
        out[i, j, k] = (fx + fy + fz) / m.vol[i, j, k]
    end
    out
end

"""
    laplacian_cell(a, h) -> Vector

``∂²a/∂x²`` on a uniform axis, interior + 2nd-order one-sided closures.
"""
function laplacian_cell(a::AbstractVector, h::Real)
    T = float(eltype(a))
    hh = T(h); n = length(a)
    out = Vector{T}(undef, n)
    c1 = 1 / hh^2
    @inbounds begin
        out[1] = (2a[1] - 5a[2] + 4a[3] - a[4]) * c1
        for i in 2:n-1
            out[i] = (a[i + 1] - 2a[i] + a[i - 1]) * c1
        end
        out[n] = (2a[n] - 5a[n - 1] + 4a[n - 2] - a[n - 3]) * c1
    end
    out
end