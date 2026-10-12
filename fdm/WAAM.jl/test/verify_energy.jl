using Test, WAAMFlow, Statistics

# ---------------------------------------------------------------------------
# Analytic reference for the classical one-phase Stefan problem
# ---------------------------------------------------------------------------

"""
    _erf(x)

Error function. `SpecialFunctions` is deliberately **not** a test dependency — the
Project's test target must run on a clean offline depot — so the Abramowitz & Stegun
7.1.26 rational approximation is used instead. Its |ε| < 1.5e-7 is three orders of
magnitude below the 2 % that V11 is allowed, so it cannot be what decides the test.
"""
@inline function _erf(x::Real)
    ax = abs(float(x))
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
    x < 0 ? -y : y
end

"""
    stefan_lambda(ste) -> Float64

Neumann similarity constant: the root of

```
λ · erf(λ) · exp(λ²) = Ste / √π
```

which is the interfacial energy balance `L_f ρ Ẋ = -k ∂T/∂x|_X` for a solid initially at
the fusion temperature and a wall held above it. The left-hand side is strictly increasing
for λ > 0, so bisection on a doubling bracket is safe and needs no derivative.
"""
function stefan_lambda(ste::Real)
    f(λ) = λ * _erf(λ) * exp(λ * λ) - ste / sqrt(π)
    lo, hi = 1e-8, 1.0
    n = 0
    while f(hi) < 0.0 && n < 200
        hi *= 2
        n += 1
    end
    n < 200 || throw(ArgumentError("failed to bracket the Neumann root for Ste=$ste"))
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) < 0.0 ? (lo = mid) : (hi = mid)
    end
    0.5 * (lo + hi)
end

"""
    stefan_material()

Constant-property material for V11. The Neumann similarity solution assumes ρ, C_p and k
are all constant and the interface is isothermal, so the test alloy has to be too; a
temperature-dependent `k` would move the front for reasons that have nothing to do with
whether the solver is right.

The mushy interval is 1 K rather than zero only because `Material` requires `Tsol < Tliq`.
That is ~0.3 % of the 300 K superheat, so the smearing contributes far below the 2 % the
test is allowed while still exercising the latent-heat plateau for real.
"""
function stefan_material()
    Tv = [200.0, 3000.0]
    c = PropCurve(Tv, fill(1000.0, 2); require_positive = true)
    k = PropCurve(Tv, fill(30.0, 2); require_positive = true)
    μ = PropCurve(Tv, fill(6.0e-3, 2); require_positive = true)
    Material(name = "StefanReference", Tsol = 1700.0, Tliq = 1701.0, Lf = 272000.0,
             rho_s = 7000.0, rho_l = 7000.0, Cp_s = c, Cp_l = c, k_s = k, k_l = k,
             mu_l = μ, beta_T = 1.0e-4)
end

"""
    stefan_front(T, m, g, Tm) -> Float64

Sub-cell interface position: where the profile first crosses `Tm`, linearly interpolated
between cell centres. Measuring at a cell centre instead would quantise the answer to
±Δx/2 — larger than the 2 % being asserted on a 75 μm mesh.
"""
function stefan_front(T::AbstractArray{<:Real,3}, m::GridMetrics, g::CartesianGrid, Tm)
    @inbounds for i in 1:g.nx-1
        a, b = T[i, 1, 1], T[i + 1, 1, 1]
        if a > Tm >= b
            θ = (a - Tm) / (a - b)
            return m.xc[i] + θ * (m.xc[i + 1] - m.xc[i])
        end
    end
    NaN
end

"""
    run_stefan(nx, dt, tf; dt_record) -> NamedTuple

Integrate the melting-front problem on `[0, L]` with the left face held at `T_w` and every
other face adiabatic, starting from a solid at exactly the solidus. Returns the front
position, the profile, and the energy audit.
"""
function run_stefan(nx::Int, dt::Float64, tf::Float64)
    mat = stefan_material()
    Tw, Tm, L = 2000.0, mat.Tsol, 0.03
    g = CartesianGrid(collect(range(0.0, L; length = nx + 1)), [0.0, 1e-3], [0.0, 1e-3])
    m = metrics(g)
    ws = EnergyWorkspace(g)

    T = fill(float(Tm), nx, 1, 1)
    Told = copy(T)
    E0 = 7000.0 * sum(enthalpy(mat, T[i, 1, 1]) * m.vol[i, 1, 1] for i in 1:nx)

    nt = round(Int, tf / dt)
    Qin = 0.0
    converged = true
    for _ in 1:nt
        copyto!(Told, T)
        bal, iters, _ = solve_energy!(ws, T, Told, mat, nothing, nothing, nothing, nothing,
                                      m, g, dt; dirichlet = (xlo = Tw,))
        converged &= bal <= 1e-6
        # Energy crossing the interior face of the boundary cell. Cell 1's own equation is
        # replaced by the Dirichlet constraint, so the wall delivers its 0.6 J of sensible
        # heat through the constraint rather than through a flux; both parts are counted.
        Qin += mat.k_s[T[1, 1, 1]] * m.ax[2, 1, 1] * (T[1, 1, 1] - T[2, 1, 1]) / m.dxc[2] * dt
    end
    E1 = 7000.0 * sum(enthalpy(mat, T[i, 1, 1]) * m.vol[i, 1, 1] for i in 1:nx)
    wall_cell_gain = 7000.0 * (enthalpy(mat, T[1, 1, 1]) - enthalpy(mat, Tm)) * m.vol[1, 1, 1]

    Cp = Cp_s_ref(mat)
    α = mat.k_s[Tm] / (mat.rho_s * Cp)
    Ste = Cp * (Tw - Tm) / mat.Lf
    λ = stefan_lambda(Ste)

    (mat = mat, Tw = Tw, Tm = Tm, L = L, g = g, m = m, T = T, t = nt * dt,
     Ste = Ste, λ = λ, α = α,
     X_num = stefan_front(T, m, g, Tm + 0.5),
     X_an = 2 * λ * sqrt(α * nt * dt),
     Qin = Qin, dE = E1 - E0, wall_gain = wall_cell_gain, converged = converged)
end

@testset "V11 1-D Stefan problem" begin
    # The gating assertion of V11: interface position within 2 % of Neumann.
    fine = run_stefan(400, 0.05, 10.0)
    err = fine.X_num / fine.X_an - 1

    @test fine.converged                       # the Newton iteration actually settled
    @test abs(err) < 0.02                      # <-- V11's stated tolerance

    # The similarity solution is a statement about the whole profile, not just the front:
    # behind it T must follow erf(x/2√(αt)) down to the fusion temperature.
    @test fine.X_num > 0 && isfinite(fine.X_num)
    Tw, Tm, λ, α = fine.Tw, fine.Tm, fine.λ, fine.α
    denom = _erf(λ)
    @inbounds for i in 1:fine.g.nx
        x = fine.m.xc[i]
        x >= fine.X_num && break
        Tan = Tw + (Tm - Tw) * _erf(x / (2 * sqrt(α * fine.t))) / denom
        @test abs(fine.T[i, 1, 1] - Tan) < 0.02 * (Tw - Tm)
    end

    # Latent-heat plateau: the interface is pinned at the fusion temperature while it
    # advances, which is exactly what makes X(t) ∝ √t. A solver that releases the latent
    # heat in the wrong place lets the front run away instead.
    Tfront = fine.T[findfirst(i -> fine.m.xc[i] >= fine.X_num, 1:fine.g.nx), 1, 1]
    @test Tfront <= Tm + 3.0                   # never superheated at the interface
    @test Tfront >= Tm - 1.0                   # never subcooled either

    # Energy conservation: everything the wall put in must be in the domain. Cell 1's
    # equation is replaced by the Dirichlet constraint, so its share of the input arrives
    # as an imposed sensible gain rather than as a face flux — count both.
    @test fine.dE ≈ fine.Qin + fine.wall_gain rtol = 1e-8

    # The far field must be untouched: an adiabatic end that silently leaks would move the
    # front without ever showing up in the residual.
    @test fine.T[end, 1, 1] ≈ Tm atol = 1e-9

    # Spatial convergence. The Neumann solution is not a discrete solution, so the error
    # must fall as the mesh is refined — otherwise the 2 % above is a coincidence of the
    # chosen resolution rather than evidence the scheme is consistent.
    coarse = run_stefan(200, 0.1, 10.0)
    @test abs(coarse.X_num / coarse.X_an - 1) > abs(err)
    @test abs(coarse.X_num / coarse.X_an - 1) < 0.02

    # Self-similarity in time: X(t) ∝ √t, so the front at t/4 must be half the front at t.
    early = run_stefan(400, 0.05, 2.5)
    @test abs(early.X_num / early.X_an - 1) < 0.02
    @test early.X_num ≈ fine.X_num / 2 rtol = 0.05
end
