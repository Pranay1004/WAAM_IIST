"""
Enthalpy–porosity phase change.

Implements Eqs. (5)–(6) of the paper:

```
(6)  h(T) = C_s T                                    T ≤ T_sol
        = C_s T_sol + L_f (T−T_sol)/(T_liq−T_sol)     T_sol < T ≤ T_liq
        = h(T_liq) + C_l (T − T_liq)                  T > T_liq
(5)  k_d = s_d f_s^2 / [ (1−f_s)³ + ε₀ ]
```

Enthalpy is the conserved variable, not temperature. That is what makes the phase
change *exactly* energetically conservative: latent heat is absorbed over the mushy
interval by construction, so there is no need for an artificial mushy-zone smearing
constant and no energy is created at the solidification front.

The `T(h)` inverse is required because the solvers update `h` and the fluid equations
need `T`. It is solved by the piecewise-explicit inverses below rather than by
iteration, so it costs nothing and cannot fail to converge.
"""

"""
    Cp_s_ref(m) / Cp_l_ref(m) -> Float64

Phase-representative specific heats: ``C_p`` evaluated at solidus and liquidus.

The enthalpy method deliberately uses a **constant** ``C_p`` per phase rather than the
full temperature-dependent curve. That is what makes `enthalpy` and `temperature` exact
mutual inverses; using the varying curve in `enthalpy` while the inverse assumes a
constant silently destroys round-trip consistency (a 66 K error at 800–2600 K in
ER70S6 before this was fixed). Specific heats for ferritic steel and austenitic stainless
change only ~15 % across the relevant range, so the constant-phase assumption costs
little accuracy while guaranteeing exact invertibility and hence exact energy
conservation.
"""
@inline Cp_s_ref(m::Material) = m.Cp_s[m.Tsol]
@inline Cp_l_ref(m::Material) = m.Cp_l[m.Tliq]

"""
    enthalpy(m::Material, T) -> Float64  [J/kg]

Specific enthalpy. Continuous and strictly monotone increasing in `T`;
``∂h/∂T = C_s`` below solidus, ``= L_f/(T_liq − T_sol)`` across the mush, and
``= C_l`` above liquidus.

Exact inverse: `temperature(m, enthalpy(m, T)) == T` to machine precision.
"""
@inline function enthalpy(m::Material, T)
    Tc = float(T)
    Ts, Tl = m.Tsol, m.Tliq
    Cs, Cl = Cp_s_ref(m), Cp_l_ref(m)
    if Tc <= Ts
        Cs * Tc
    elseif Tc <= Tl
        Cs * Ts + m.Lf * (Tc - Ts) / (Tl - Ts)
    else
        (Cs * Ts + m.Lf) + Cl * (Tc - Tl)
    end
end

"""
    temperature(m::Material, h) -> Float64  [K]

Exact piecewise inverse of `enthalpy`.
"""
@inline function temperature(m::Material, h)
    Cs, Cl = Cp_s_ref(m), Cp_l_ref(m)
    hs = Cs * m.Tsol
    hl = hs + m.Lf
    hc = float(h)
    if hc <= hs
        hc / Cs
    elseif hc <= hl
        m.Tsol + (hc - hs) * (m.Tliq - m.Tsol) / m.Lf
    else
        m.Tliq + (hc - hl) / Cl
    end
end

"""
    cp_effective(m::Material, T) -> Float64

``∂h/∂T``, i.e. the plateau heat capacity ``L_f/(T_liq − T_sol)`` across the mush. Used
by the energy equation's implicit diagonal, where a huge ``C_p`` across the mush is the
correct representation of latent heat.
"""
@inline function cp_effective(m::Material, T)
    Tc = float(T)
    if Tc <= m.Tsol
        Cp_s_ref(m)
    elseif Tc <= m.Tliq
        m.Lf / (m.Tliq - m.Tsol)
    else
        Cp_l_ref(m)
    end
end

"""
    latent_heat_rate(m::Material, T_old, T_new) -> Float64  [J/m³/s]

Volumetric latent-heat source implied by a temperature change, for operator-split
bookkeeping and for the energy-budget diagnostic (verification test V20).
"""
@inline function latent_heat_rate(m::Material, Told, Tnew)
    (enthalpy(m, Tnew) - enthalpy(m, Told)) * rho(m, (Told + Tnew) / 2)
end

# ---------------------------------------------------------------------------
# Porosity (mushy-zone) drag
# ---------------------------------------------------------------------------

"""
    darcy_coefficient(m, T, sd; exponent, eps0) -> Float64  [1/s]

Darcy drag coefficient evaluated directly from temperature via the liquid fraction.
See `darcy_drag` in `materials/properties.jl` for the formula and the `sd` caveat:
with ``s_d = 1e6`` and a 0.6 mm cell the damping velocity ``Δx/s_d ≈ 0.6 nm/s`` is
effectively a hard freeze rather than a porous medium. Both regimes are supported; see
`porosity_scale`.
"""
@inline function darcy_coefficient(m::Material, T, sd::Real = DEFAULT_SD;
                                   exponent::Int = 2, eps0::Real = 1e-4)
    darcy_drag(m, solid_fraction(m, T), sd; exponent = exponent, eps0 = eps0)
end

"""
    porosity_scale(m, cell_size, sd) -> Float64

Physically-scaled `s_d` from Carman–Kozeny using the mushy-zone permeability scale.

```
s_d = μ / (K₀ ρ C²),   C = mushy-zone constant (Brinkman, C = 1e5 … 1e7 1/m)
```

With a realistic ``C = 1e7 1/m`` this gives ``s_d ~ 1e3 1/s``, i.e. a damping velocity
``Δx/s_d ~ 1e-6 m/s`` — genuinely a *porous* mush, not a freeze. Use this instead of
the paper's ``1e6`` when quantitative mushy-zone *flow* matters rather than merely
suppressing motion.

Defaults are deliberately explicit; the returned value is written into every run
manifest so the regime is never ambiguous.
"""
function porosity_scale(m::Material, cell_size::Real, C_brinkman::Real = 1e7;
                        mu_ref::Real = 6e-3, fs_ref::Real = 0.5)
    mu_ref / (2.0 * mu_ref * fs_ref^2 * C_brinkman^2 * cell_size)
end