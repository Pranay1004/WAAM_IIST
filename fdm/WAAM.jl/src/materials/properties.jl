"""
Property curves: monotone piecewise-linear interpolation of a thermophysical property
against temperature, with **automatic tangent continuation** outside the data range.

Why tangent continuation rather than clamping or polynomial extrapolation:

- **Clamping** to the end value means `K`, `μ`, `C_p` are frozen outside the table. In a
  welding arc the peak temperature routinely exceeds the top of a published table, so
  clamping silently invents an isothermal plume — the single most common way a
  "physics" code stops being physics.
- **Polynomial extrapolation** oscillates wildly, and can drive `μ < 0`.
- **Tangent continuation** (`value + slope * (T - T_end)`) is monotone by construction,
  degrades gracefully, and is exactly correct for the many properties that really are
  linear in `T` over the relevant range (surface tension, roughly).

A safety rail caps the continuation at a configurable multiple of the last tabulated
value, so a pathological extrapolation cannot produce negative density or viscosity.
"""

"""
    PropCurve

Monotone piecewise-linear temperature property.

Fields
- `T`     : sorted node temperatures (K), strictly increasing
- `val`   : values at `T`
- `slope` : nodal slopes (K⁻¹), precomputed
- `tail_max` : multiplicative cap applied to tangent continuation
"""
struct PropCurve{T<:AbstractFloat}
    T::Vector{T}
    val::Vector{T}
    slope::Vector{T}
    tail_max::T        # cap on |extrapolation| as a multiple of the end value
    tail_floor::T      # lower clamp (positive for strictly-positive properties)
end

function PropCurve(Tv::AbstractVector{<:Real}, vv::AbstractVector{<:Real};
                   tail_max::Real = 3.0, tail_floor::Real = 0.0,
                   require_positive::Bool = false)
    length(Tv) == length(vv) || throw(ArgumentError(
        "PropCurve: $(length(Tv)) temperatures vs $(length(vv)) values"))
    perm = sortperm(collect(Float64, Tv))
    Tv2 = collect(Float64, Tv)[perm]
    vv2 = collect(Float64, vv)[perm]
    length(Tv2) >= 2 || throw(ArgumentError("PropCurve needs >= 2 nodes"))
    @inbounds for i in 2:length(Tv2)
        Tv2[i] > Tv2[i - 1] || throw(ArgumentError(
            "PropCurve temperatures must be distinct (got $(Tv2[i-1]) and $(Tv2[i]))"))
    end
    require_positive && minimum(vv2) <= 0 && throw(ArgumentError(
        "PropCurve: require_positive=true but a value is $(minimum(vv2))"))
    n = length(Tv2)
    s = Vector{Float64}(undef, n)
    @inbounds begin
        s[1] = (vv2[2] - vv2[1]) / (Tv2[2] - Tv2[1])
        s[n] = (vv2[n] - vv2[n - 1]) / (Tv2[n] - Tv2[n - 1])
        for i in 2:n-1
            s[i] = (vv2[i + 1] - vv2[i - 1]) / (Tv2[i + 1] - Tv2[i - 1])
        end
    end
    PropCurve(Tv2, vv2, s, Float64(tail_max), Float64(tail_floor))
end

@inline function Base.getindex(c::PropCurve, T::Real)
    Tc = float(T)
    t, v, s, cmax, cflr = c.T, c.val, c.slope, c.tail_max, c.tail_floor
    n = length(t)
    # --- below the table: tangent continuation from the first node, bounded ---
    Tc <= t[1] && return _tail(v[1], s[1], Tc - t[1], cmax, cflr)
    Tc >= t[n] && return _tail(v[n], s[n], Tc - t[n], cmax, cflr)
    # --- interior: linear within the bracketing interval ---
    i = searchsortedlast(t, Tc)
    h = t[i + 1] - t[i]
    v[i] + (v[i + 1] - v[i]) * (Tc - t[i]) / h
end

"""
    _tail(vn, sn, dT, cmax)

Bounded tangent continuation from the last tabulated node. Extrapolation follows the
local slope but is clamped so it can never change sign or exceed `cmax` times the
tabulated magnitude — a property must never become negative (viscosity, density,
conductivity) no matter how far outside its data range it is evaluated.
"""
@inline function _tail(vn::Float64, sn::Float64, dT::Float64, cmax::Float64,
                        cflr::Float64 = 0.0)
    val = vn + sn * dT
    if vn >= 0
        clamp(val, cflr, cmax * vn)
    elseif vn < 0
        clamp(val, cmax * vn, zero(vn))
    else
        val
    end
end

value(c::PropCurve, T::Real) = c[T]

"""
    is_monotone(c; tol) -> Bool

Test that a curve is non-decreasing (or non-increasing) — used by verification tests to
assert that no property curve in the database introduces non-physical behaviour.
"""
function is_monotone(c::PropCurve; increasing = true, tol = 1e-12)
    increasing ? all(i -> c.val[i + 1] >= c.val[i] - tol, 1:length(c.val)-1) :
                  all(i -> c.val[i + 1] <= c.val[i] + tol, 1:length(c.val)-1)
end

# ---------------------------------------------------------------------------
# Surface tension models  (paper Sec. 3.2, Eqs. 13 & the Pichler fit)
# ---------------------------------------------------------------------------

abstract type SurfaceTensionModel end

"""
    dgamma_dT(model, T)

Temperature slope of the surface tension at `T` (N/(m·K)); the quantity whose *sign*
decides whether Marangoni flow is inward (deep, narrow pool) or outward (wide, shallow
pool). The generic fallback is zero: a model that declares no temperature dependence
must report it rather than error.
"""
dgamma_dT(m::SurfaceTensionModel, T) = 0.0

"""
    ConstantGamma(gamma)

Constant surface tension, σ(T) = γ₀ N/m.
"""
struct ConstantGamma <: SurfaceTensionModel
    gamma::Float64      # N/m
end
@inline (c::ConstantGamma)(T) = c.gamma
dgamma_dT(c::ConstantGamma, T) = 0.0
Base.show(io::IO, c::ConstantGamma) = print(io, "ConstantGamma(γ=", c.gamma, " N/m)")

"""
    LinearGamma(gamma0, Tref, k)

Linear surface tension, σ(T) = γ₀ + k (T - T_ref).
`k = ∂γ/∂T < 0` for ER316L (Pichler et al. 2022), giving outward Marangoni flow and
wide shallow pools.
"""
struct LinearGamma <: SurfaceTensionModel
    gamma0::Float64     # N/m at Tref
    Tref::Float64       # K
    k::Float64          # N/(m·K), negative
end
(c::LinearGamma)(T) = c.gamma0 + c.k * (T - c.Tref)
dgamma_dT(c::LinearGamma, T) = c.k
Base.show(io::IO, c::LinearGamma) =
    print(io, "LinearGamma(γ₀=", c.gamma0, " N/m @ ", c.Tref, " K, dγ/dT=", c.k, ")")

"""
    SahooDebRoyFeS

Sahoo–DebRoy pseudo-binary Fe–S surface tension, Eq. 13 of the paper (ref. 51):

```
γ = γ°_m − A (T − T_m) − R T Γ_s ln[ 1 + k₁ a_s exp(−ΔH° / (R T)) ]
```

Sulfur is the dominant surface-active element in ER70S6; oxygen is neglected. The
`ln[·]` term is what makes `∂γ/∂T` capable of **changing sign** with sulfur content —
the physical basis for the paper's observation that ER70S6 exhibits compact, deep,
"finger-type" pools while ER316L spreads wide.
"""
struct SahooDebRoyFeS <: SurfaceTensionModel
    gamma_m::Float64     # γ°_m  N/m, surface tension of pure liquid iron
    A::Float64           # −∂γ/∂T for pure Fe, N/(m·K)
    Tm::Float64          # melting point of pure Fe, K
    R::Float64           # 8.314 J/(mol·K)
    Gamma_s::Float64     # Γ_s surface excess at saturation, mol/m²
    k1::Float64          # entropy factor, dimensionless
    dH::Float64          # ΔH° heat of adsorption, J/mol
    a_s::Float64         # sulfur activity ≈ wt.% S
end

function SahooDebRoyFeS(; a_s::Real = 0.018,
                        gamma_m::Real = 1.943, A::Real = 4.3e-4,
                        Tm::Real = 1811.0, R::Real = 8.314,
                        Gamma_s::Real = 1.3e-5, k1::Real = 3.18e-3,
                        dH::Real = -1.66e5)
    SahooDebRoyFeS(gamma_m, A, Tm, R, Gamma_s, k1, dH, a_s)
end

@inline function (m::SahooDebRoyFeS)(T)
    R, G, k1, dH, as = m.R, m.Gamma_s, m.k1, m.dH, m.a_s
    Tc = float(T)
    bracket = 1 + k1 * as * exp(-dH / (R * Tc))
    m.gamma_m - m.A * (Tc - m.Tm) - R * Tc * G * log(bracket)
end

@inline function dgamma_dT(m::SahooDebRoyFeS, T)
    R, G, as, dH = m.R, m.Gamma_s, m.a_s, m.dH
    Tc = float(T)
    # γ(T) = γ°_m − A(T − T_m) − R T Γ_s ln(1 + x),  x = k₁ a_s e^{−ΔH°/RT}
    x = m.k1 * as * exp(-dH / (R * Tc))
    b = 1 + x
    dxdT = x * (dH / (R * Tc^2))          # = x · ΔH°/(R T²)
    -m.A - R * G * (log(b) + Tc * (dxdT / b))
end

"""
    marangoni_source(model, T, dTdx, dTdy, dTdz)

Gradient of surface tension along the surface, ``∇_s γ = (∂γ/∂T) ∇_s T``.
Returns the vector driving tangential surface flow; its divergence drives the
characteristic surface circulation. For `SahooDebRoyFeS`, ``∂γ/∂T`` is evaluated
analytically (not by finite differences), which matters because the sign change at
low sulfur is a steep, narrow feature.
"""
function marangoni_source(m::LinearGamma, T, dTdx, dTdy, dTdz)
    k = m.k
    (k * dTdx, k * dTdy, k * dTdz)
end
function marangoni_source(m::ConstantGamma, T, dTdx, dTdy, dTdz)
    (0.0, 0.0, 0.0)
end
function marangoni_source(m::SahooDebRoyFeS, T, dTdx, dTdy, dTdz)
    k = dgamma_dT(m, T)
    (k * dTdx, k * dTdy, k * dTdz)
end

function Base.show(io::IO, m::SahooDebRoyFeS)
    print(io, "SahooDebRoyFeS(a_S=", m.a_s, " wt.%, ∂γ/∂T at 1800K = ",
          round(dgamma_dT(m, 1800.0), digits = 6), " N/(m·K))")
end

# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------

"""
    Material

A thermophysical material record. Solid and liquid properties are separate curves so
the mushy-zone blend ``(1−f_s) K_s + f_s K_l`` is explicit rather than emergent.

Fields (SI throughout)
- `Tsol`, `Tliq` : solidus / liquidus, K
- `Lf`          : latent heat of fusion, J/kg
- `rho_s`, `rho_l` : densities, kg/m³
- `Cp_s`, `Cp_l` : specific heats, J/(kg·K)
- `k_s`, `k_l`  : thermal conductivities, W/(m·K)
- `mu_l`        : dynamic viscosity (liquid), Pa·s
- `beta_T`      : thermal expansion, 1/K
- `gamma`       : a `SurfaceTensionModel`
"""
Base.@kwdef struct Material
    name::String
    Tsol::Float64
    Tliq::Float64
    Lf::Float64
    rho_s::Float64
    rho_l::Float64
    Cp_s::PropCurve
    Cp_l::PropCurve
    k_s::PropCurve
    k_l::PropCurve
    mu_l::PropCurve
    beta_T::Float64
    gamma::SurfaceTensionModel = ConstantGamma(1.2)
    composition::Dict{Symbol,Float64} = Dict{Symbol,Float64}()

    function Material(name, Tsol, Tliq, Lf, rho_s, rho_l, Cp_s, Cp_l, k_s, k_l,
                      mu_l, beta_T, gamma, composition)
        Tsol < Tliq || throw(ArgumentError("$name: Tsol($Tsol) must be < Tliq($Tliq)"))
        all(p -> p isa PropCurve, (Cp_s, Cp_l, k_s, k_l, mu_l)) ||
            throw(ArgumentError("$name: property curves must be PropCurve"))
        Lf > 0 || throw(ArgumentError("$name: latent heat must be positive"))
        new(name, Tsol, Tliq, Lf, rho_s, rho_l, Cp_s, Cp_l, k_s, k_l, mu_l, beta_T,
            gamma, composition)
    end
end

# --- derived, temperature-dependent accessors -------------------------------

@inline liquid_fraction(m::Material, T) = clamp((float(T) - m.Tsol) / (m.Tliq - m.Tsol), 0.0, 1.0)
@inline solid_fraction(m::Material, T) = 1 - liquid_fraction(m, T)
@inline rho(m::Material, T) = m.rho_s + (m.rho_l - m.rho_s) * liquid_fraction(m, T)
@inline Cp(m::Material, T) = (1 - liquid_fraction(m, T)) * m.Cp_s[T] + liquid_fraction(m, T) * m.Cp_l[T]
@inline kcond(m::Material, T) = (1 - liquid_fraction(m, T)) * m.k_s[T] + liquid_fraction(m, T) * m.k_l[T]
@inline visc(m::Material, T) = m.mu_l[T]

"""
    darcy_drag(m, f_s, sd; exponent=2, eps0=1e-4)

Mushy-zone Darcy drag coefficient, Eq. 5 of the paper:

```
k_d = s_d · f_s^exponent / [ (1 − f_s)³ + ε₀ ]
```

``ε₀ = 1e-4`` regularises the ``f_s → 1`` limit so the drag stays finite in fully
solid cells; without it the momentum matrix becomes singular as the pool freezes.
Carman–Kozeny (the physically correct form for flow through a solidifying dendritic
matrix) is ``exponent = 2``.
"""
@inline function darcy_drag(m::Material, f_s::Real, sd::Real;
                            exponent::Int = 2, eps0::Real = 1e-4)
    fs = float(f_s)
    sd * fs^exponent / ((1 - fs)^3 + eps0)
end

# Default drag coefficient: paper states s_d ≈ 1e6 1/s.
DEFAULT_SD = 1.0e6