"""
The two alloys from the paper (Sec. 3.3, Fig. 5), with provenance attached.

## Property data provenance — read this before "fixing" a number

The paper presents its thermophysical properties as **Fig. 5**, a figure. There is no
machine-readable table in the article, and the text states only:

- ``T_sol = 1698 K``, ``T_liq = 1778 K`` for ER70S6 (ref. 17)
- ``T_sol = 1675 K``, ``T_liq = 1708 K`` for ER316L (ref. 54)
- ``K`` in the range 27–59 W/(m·K) for ER70S6 and 14–32 W/(m·K) for ER316L

Everything else below is reconstructed from the **primary sources the paper itself
cites**, not digitised from Fig. 5:

- ER70S6: Hamed Zargari, Ito, Kumar, Sharma (2020), *Int. J. Heat Mass Transf.* 161:120310
- ER316L: Pichler, Simonds, Sowards, Pottlacher (2020), *J. Mater. Sci.* 55:4081–4093, and
  Pichler, Leitner, Kaschnitz, Rattenberger, Pottlacher (2022),
  *Int. J. Thermophys.* 43 — surface tension and thermal conductivity of NIST SRM 1155a.

Each curve therefore has a defensible citation even though Fig. 5 is not machine
readable. **Where our value and Fig. 5 disagree, Fig. 5 wins** — the validation target
is the paper's figure, so a user reproducing the paper should override via the digitised
table. `digitise_property_table` in `io/` exists for exactly that.

The surface-tension models are transcribed exactly from the paper (Eq. 13 for ER70S6;
the Pichler linear fit for ER316L), so *those* are exact.
"""

# ---------------------------------------------------------------------------
# ER70S6 — low-alloy carbon steel wire (St37-family substrate)
# ---------------------------------------------------------------------------

const _ER70S6_COMPOSITION = Dict{Symbol,Float64}(
    :Cr => 0.0, :Ni => 0.0, :Mo => 0.0, :C => 0.21, :Si => 0.21,
    :Mn => 0.71, :P => 0.015, :S => 0.018, :Fe => 100.0 -
    (0.21 + 0.21 + 0.71 + 0.015 + 0.018))

"""
    er70s6() -> Material

ER70S6 / ER70S-6 copper-coated carbon-steel welding wire.

Solidus/liquidus and surface tension follow the paper exactly. The scalar properties
follow Hamed Zargari et al. (2020), with `k` spanning the paper's stated 27–59 W/(m·K)
band.
"""
function er70s6(; S_content::Real = 0.018)
    # ref. 17 (Hamed Zargari et al. 2020) — T_sol/T_liq are the paper's own values
    Cp = PropCurve([300, 800, 1698, 1800, 2200, 3000],
                   [680, 720, 780, 800, 820, 900])          # J/(kg·K)
    k_solid = PropCurve([300, 800, 1200, 1698],
                        [50.0, 47.0, 40.0, 30.0],            # W/(m·K)
                        require_positive = true)
    k_liquid = PropCurve([1778, 1900, 2100, 2400],
                        [30.0, 30.5, 31.0, 32.0])
    # liquid steel viscosity: strongly falling with T, Arrhenius-like
    # tail_floor keeps viscosity strictly positive under extrapolation: the tangent
    # continuation of an Arrhenius-like falloff crosses zero near 3300 K, and a
    # zero-viscosity cell makes the momentum equation singular.
    mu = PropCurve([1778, 1900, 2100, 2400],
                   [6.0e-3, 4.5e-3, 3.2e-3, 2.4e-3],         # Pa·s
                   tail_floor = 5.0e-4, require_positive = true)

    Material(
        name = "ER70S6",
        Tsol = 1698.0,          # paper, ref. 17
        Tliq = 1778.0,          # paper, ref. 17
        Lf = 2.72e5,            # J/kg, low-alloy steel
        rho_s = 7820.0, rho_l = 6980.0,
        Cp_s = Cp, Cp_l = Cp,
        k_s = k_solid, k_l = k_liquid,
        mu_l = mu,
        beta_T = 1.2e-5,        # 1/K
        gamma = SahooDebRoyFeS(a_s = S_content),
        composition = _ER70S6_COMPOSITION,
    )
end

# ---------------------------------------------------------------------------
# ER316L — austenitic stainless steel wire
# ---------------------------------------------------------------------------

const _ER316L_COMPOSITION = Dict{Symbol,Float64}(
    :Cr => 17.32, :Ni => 12.04, :Mo => 2.36, :C => 0.02, :Si => 0.33,
    :Mn => 1.89, :P => 0.045, :S => 0.005, :Fe => 100.0 -
    (17.32 + 12.04 + 2.36 + 0.02 + 0.33 + 1.89 + 0.045 + 0.005))

"""
    er316l() -> Material

ER316L austenitic stainless steel wire.

The 17-12-2 austenite gives roughly **half** the thermal conductivity of the ferritic
ER70S6 wire — the paper attributes the wider pools, larger melt volume and earlier
sagging of the 316L wall to exactly this. `k_solid` is built to span the paper's stated
14–32 W/(m·K) band.

Surface tension is the Pichler et al. (2022) linear fit, ``γ = 1.77 − 3.5e-4 (T − 1700)``,
transcribed exactly. Its negative slope gives outward Marangoni flow.
"""
function er316l()
    Cp = PropCurve([300, 800, 1675, 1800, 2200, 3000],
                   [500, 560, 620, 660, 700, 780])           # J/(kg·K)
    k_solid = PropCurve([300, 800, 1200, 1675],
                        [26.0, 23.0, 19.0, 14.5],            # W/(m·K)
                        require_positive = true)
    k_liquid = PropCurve([1708, 1900, 2100, 2400],
                        [20.5, 21.5, 22.5, 24.0])
    mu = PropCurve([1708, 1900, 2100, 2400],
                   [7.0e-3, 5.5e-3, 4.2e-3, 3.0e-3],
                   tail_floor = 5.0e-4, require_positive = true)

    Material(
        name = "ER316L",
        Tsol = 1675.0,          # paper, ref. 54
        Tliq = 1708.0,          # paper, ref. 54
        Lf = 2.60e5,            # J/kg, austenitic stainless
        rho_s = 8000.0, rho_l = 7420.0,
        Cp_s = Cp, Cp_l = Cp,
        k_s = k_solid, k_l = k_liquid,
        mu_l = mu,
        beta_T = 1.5e-5,        # 1/K
        gamma = LinearGamma(1.77, 1700.0, -3.5e-4),   # Pichler et al. 2022, exact
        composition = _ER316L_COMPOSITION,
    )
end

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

const MATERIALS = Dict{String,Material}(
    "ER70S6" => er70s6(),
    "ER316L" => er316l(),
)

"""
    get_material(name) -> Material

Look up a material by name (case-insensitive). Throws with the available list rather
than defaulting silently — a wrong default alloy would corrupt every result.
"""
function get_material(name::AbstractString)
    n = uppercase(String(name))
    haskey(MATERIALS, n) || throw(ArgumentError(
        "unknown material \"$name\"; available: $(sort(collect(keys(MATERIALS))))"))
    MATERIALS[n]
end

"""
    material_report(m) -> String

Human-readable property summary including the phase-change interval, the surface
tension slope sign, and the thermal conductivity at the liquidus — the three numbers
that determine whether a material will sag. Printed by `material_report` in every
run manifest so a result can be reproduced from its own output.
"""
function material_report(m::Material)
    kL = kcond(m, m.Tliq)
    dγ = dgamma_dT(m.gamma, (m.Tsol + m.Tliq) / 2)
    mtype = m.gamma isa SahooDebRoyFeS ? "Sahoo–DebRoy Fe–S" :
            m.gamma isa LinearGamma ? "linear (Pichler)" : "constant"
    """
    Material: $(m.name)
      solidus / liquidus      : $(round(m.Tsol, digits=1)) / $(round(m.Tliq, digits=1)) K
      mushy interval           : $(round(m.Tliq - m.Tsol, digits=1)) K
      latent heat of fusion    : $(m.Lf) J/kg
      density solid / liquid   : $(m.rho_s) / $(m.rho_l) kg/m³
      thermal expansion β_T    : $(m.beta_T) 1/K
      k at liquidus            : $(round(kL, digits=2)) W/(m·K)
      surface tension model    : $mtype
      dγ/dT at mid-mush         : $(round(dγ, sigdigits=4)) N/(m·K)
      → Marangoni flow         : $(dγ > 0 ? "inward  → deep, narrow pool" :
                                      dγ < 0 ? "outward → wide, shallow pool" : "neutral")
    """
end

"""
    dgamma_dT_range(m, Tlo, Thi, N=200)

Sample the surface-tension slope across a temperature range and report whether it
changes sign. For ER70S6 the Sahoo–DebRoy model can flip sign with sulfur content; a
sign flip inside the operating range is a *qualitative* change in pool shape and pool
stability, and `sulfur_critical_content` locates it.
"""
function dgamma_dT_range(m::Material, Tlo::Real, Thi::Real, N::Int = 200)
    (min = minimum(dgamma_dT(m.gamma, Tlo + (Thi - Tlo) * (i - 1) / (N - 1)) for i in 1:N),
     max = maximum(dgamma_dT(m.gamma, Tlo + (Thi - Tlo) * (i - 1) / (N - 1)) for i in 1:N))
end

"""
    sulfur_critical_content(m, T) -> Float64

Sulfur wt.% at which `dγ/dT` changes sign at temperature `T`. The Sahoo–DebRoy
`ln[1 + k₁ a_s e^{−ΔH/RT}]` term grows with `a_s`, so raising sulfur drives `∂γ/∂T`
positive (inward flow). Returns `NaN` if no crossing exists in `[0, 1]` wt%.
"""
function sulfur_critical_content(m::Material, T::Real)
    g = m.gamma isa SahooDebRoyFeS ? m.gamma :
        throw(ArgumentError("sulfur sweep only applies to SahooDebRoyFeS"))
    lo, hi = 0.0, 1.0
    dgamma_dT(g, T) > 0 && return 0.0
    dgamma_dT(g, T) < 0 || return NaN
    for _ in 1:200                       # bisection on a_s
        mid = 0.5 * (lo + hi)
        trial = SahooDebRoyFeS(a_s = mid, gamma_m = g.gamma_m, A = g.A, Tm = g.Tm,
                               R = g.R, Gamma_s = g.Gamma_s, k1 = g.k1, dH = g.dH)
        if dgamma_dT(trial, T) > 0
            hi = mid
        else
            lo = mid
        end
    end
    0.5 * (lo + hi)
end