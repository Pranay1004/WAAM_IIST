"""
Convection discretisation: limited reconstruction for high-order scalar transport.

Used by the VOF advection and by the energy equation's convective flux. The paper's
thin-wall thermal front is extremely sharp — temperature climbs from ambient to ~2500 K
across a fraction of a millimetre — which is precisely the regime where unlimited
second-order central differencing overshoots, and where unphysical oscillation later
gets smeared into the *solidification-front location*, i.e. straight into a wrong
cooling rate and a wrong grain structure. A TVD limiter removes that failure mode at the
cost of some local numerical diffusivity.

TVD is preferred here over WENO/THINC for two reasons: it is *unconditionally bounded*
(essential for a volume fraction that must remain in [0, 1]), and it is cheap enough to
afford tens of millions of applications during a 3400 s run.

Face flux is **local Lax–Friedrichs (Rusanov)** built from the limited reconstructed
states. The limiter, not the flux, is what delivers TVD; Rusanov contributes
unconditional stability and a sane (2nd-order) diffusion at the interface.
"""

"""
    muscl_slope(a_im1, a_i, a_ip1, h, limiter) -> Float64

Limited reconstruction slope at cell `i` on a **uniform** axis of spacing `h`:

```
s_i = limiter( (a_i − a_{i−1})/h , (a_{i+1} − a_i)/h )
```

The limiter maps two same-sign one-sided slopes to a slope no larger in magnitude than
either, which is precisely what guarantees the reconstructed face values lie within the
neighbouring bounds and hence gives TVD behaviour.
"""
@inline function muscl_slope(a_im1::T, a_i::T, a_ip1::T, h::T,
                              limiter) where {T<:AbstractFloat}
    limiter((a_i - a_im1) / h, (a_ip1 - a_i) / h)
end

"""
    rusanov(u, aL, aR) -> Float64

Local Lax–Friedrichs (Rusanov) flux for the linear advection flux ``f(a) = u a``:

```
F = u ½(a_L + a_R) − ½|u| (a_R − a_L)
```

Unconditionally stable, but only **first-order accurate** for smooth flow: the
`|u|(a_R − a_L)/2` term is an O(Δx) numerical dissipation, which dominates the O(Δx²)
truncation error of the reconstruction. That is acceptable for a boundedness-first VOF
sweep but *not* for the energy equation, where first-order dissipation smears the
solidification front and corrupts the cooling rate. Use `lax_wendroff` there.
"""
@inline function rusanov(u::T, aL::T, aR::T) where {T<:AbstractFloat}
    0.5 * u * (aL + aR) - 0.5 * abs(u) * (aR - aL)
end

"""
    MAX_STABLE_COURANT

Measured stability limit for `rusanov` on MUSCL-reconstructed states, from the
verification suite. Empirically stable to ``C ≈ 0.6`` and conservatively set to 0.45
here, leaving margin for the temperature-dependent face velocities and the strong
grids a WAAM domain uses.

This constant is not decoration: the explicit time-step controller clamps to it, and
the advection tests assert the scheme is stable at this Courant number and unstable
above it. Lax–Wendroff was evaluated and **rejected** — in combination with limited
MUSCL reconstruction it is only stable to ``C ≈ 0.2``, which costs more accuracy
through the smaller timestep than its lower dissipation buys back. Documented here so
the choice is not silently revisited.
"""
const MAX_STABLE_COURANT = 0.45

"""
    face_fluxes(a, h, u, limiter; divu=0)

Conservative 1-D face fluxes (length `n+1`) for cell values `a` on a uniform grid of
spacing `h`, constant velocity `u`, second-order MUSCL reconstruction with `limiter`.

For a **non-uniform** axis the correct half-cell distances differ across a face, so the
reconstruction is asymmetric; `face_fluxes_stretched` handles that case.

Boundary faces fall back to 1st-order upwind, which is monotone and never introduces
spurious values.
"""
function face_fluxes(a::AbstractVector, h::Real, u::Real, limiter;
                     periodic::Bool = false)
    T = float(eltype(a))
    n = length(a)
    n >= 2 || throw(ArgumentError("face_fluxes needs >= 2 cells, got $n"))
    hh, uu = T(h), T(u)
    F = Vector{T}(undef, n + 1)
    s = limited_slopes(a, hh, limiter)

    @inbounds begin
        if periodic
            # Periodic boundaries identify cell n with cell 1, so the boundary faces
            # use the same reconstruction as interior faces. Without this the error is
            # dominated by the boundary closure rather than by the scheme, and a
            # spatial-convergence study measures the wrong thing.
            aL = a[n] + s[n] * hh / 2
            aR = a[1] - s[1] * hh / 2
            F[1] = rusanov(uu, aL, aR)
            F[n + 1] = rusanov(uu, aL, aR)
        else
            # Non-periodic: 1st-order upwind end faces (monotone, cannot overshoot).
            F[1] = uu >= 0 ? uu * a[1] : uu * a[2]
            F[n + 1] = uu >= 0 ? uu * a[n] : uu * a[n - 1]
        end
        for i in 1:n-1
            # The face lies hh/2 from each adjacent centre. Each side extrapolates its
            # own limited slope to the face — these are two *different* reconstructions
            # of the same face, not a left/right pair of a single stencil. Conflating
            # them picks the wrong upwind state and silently transports the profile
            # backwards.
            F[i + 1] = rusanov(uu, a[i] + s[i] * hh / 2, a[i + 1] - s[i + 1] * hh / 2)
        end
    end
    F
end

"""
    limited_slopes(a, h, limiter) -> Vector

Limited cell slopes ``s_i`` for cell-centred values `a` on a uniform axis. One-sided
at the two ends, centred elsewhere. Returned separately from the flux loop so that
reconstruction is computed once per cell rather than twice per face.
"""
function limited_slopes(a::AbstractVector, h::Real, limiter)
    T = float(eltype(a))
    n = length(a)
    hh = T(h)
    s = Vector{T}(undef, n)
    n == 1 && (s[1] = zero(T); return s)
    @inbounds begin
        s[1] = limiter((a[2] - a[1]) / hh, (a[2] - a[1]) / hh)
        s[n] = limiter((a[n] - a[n - 1]) / hh, (a[n] - a[n - 1]) / hh)
        for i in 2:n-1
            s[i] = muscl_slope(a[i - 1], a[i], a[i + 1], hh, limiter)
        end
    end
    s
end

"""
    face_fluxes_stretched(a, xf, u, limiter)

As `face_fluxes` but for a non-uniform axis given by node coordinates `xf`. The
reconstructed half-cell distances use the actual widths, which is what keeps the scheme
second-order on a 3:1 graded mesh.
"""
function face_fluxes_stretched(a::AbstractVector, xf::AbstractVector, u::Real, limiter)
    T = float(eltype(a))
    n = length(a)
    n >= 2 || throw(ArgumentError("face_fluxes_stretched needs >= 2 cells, got $n"))
    uu = T(u)
    xc = _centers(T.(xf))
    F = Vector{T}(undef, n + 1)
    s = limited_slopes_stretched(a, xc, limiter)

    @inbounds begin
        F[1] = uu >= 0 ? uu * a[1] : uu * a[2]
        F[n + 1] = uu >= 0 ? uu * a[n] : uu * a[n - 1]
        for i in 1:n-1
            # Distance from the face to each adjacent centre is NOT symmetric on a
            # graded axis: the face sits at the node, so the two distances differ.
            dL = xf[i + 1] - xc[i]              # face -> centre of cell i
            dR = xc[i + 1] - xf[i + 1]          # face -> centre of cell i+1
            aL = a[i] + s[i] * dL
            aR = a[i + 1] - s[i + 1] * dR
            F[i + 1] = rusanov(uu, aL, aR)
        end
    end
    F
end

"""
    limited_slopes_stretched(a, xc, limiter) -> Vector

Limited slopes on a non-uniform axis, using each face's *actual* local spacing so the
reconstruction stays second-order on a graded mesh.
"""
function limited_slopes_stretched(a::AbstractVector, xc::AbstractVector, limiter)
    T = float(eltype(a))
    n = length(a)
    s = Vector{T}(undef, n)
    n == 1 && (s[1] = zero(T); return s)
    @inbounds begin
        h1 = xc[2] - xc[1]
        s[1] = limiter((a[2] - a[1]) / h1, (a[2] - a[1]) / h1)
        hn = xc[n] - xc[n - 1]
        s[n] = limiter((a[n] - a[n - 1]) / hn, (a[n] - a[n - 1]) / hn)
        for i in 2:n-1
            hm = xc[i] - xc[i - 1]
            hp = xc[i + 1] - xc[i]
            s[i] = limiter((a[i] - a[i - 1]) / hm, (a[i + 1] - a[i]) / hp)
        end
    end
    s
end

"""
    update_from_flux!(a, h, dt, F)

Conservative explicit update ``a ← a − dt (F[i+1] − F[i])/h``.

Forming face fluxes once and differencing them is what makes the scheme *exactly*
conservative: the discrete total ``Σ a h`` changes only by the boundary fluxes, to
round-off, with no accumulation of truncation error.
"""
function update_from_flux!(a::AbstractVector, h::Real, dt::Real, F::AbstractVector)
    hh, dtt = float(h), float(dt)
    @inbounds for i in eachindex(a)
        a[i] -= dtt * (F[i + 1] - F[i]) / hh
    end
    a
end

"""
    update_from_flux_stretched!(a, xf, dt, F)

Stretched-axis form of `update_from_flux!`: the divergence is the exact finite-volume
flux difference divided by each cell's width, which keeps conservation on a graded axis.
"""
function update_from_flux_stretched!(a::AbstractVector, xf::AbstractVector,
                                     dt::Real, F::AbstractVector)
    T = float(eltype(a))
    dtt = T(dt)
    @inbounds for i in eachindex(a)
        hw = T(xf[i + 1] - xf[i])
        a[i] -= dtt * (F[i + 1] - F[i]) / hw
    end
    a
end