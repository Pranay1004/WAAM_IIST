"""
Energy equation solver, enthalpy form — the paper's Eq. (3).

```
∂h/∂t + ∇·(u h) = ∇·(K∇T)/ρ + ḣ_s
```

Discretised as **backward Euler** on the cell-centred grid with harmonic face
conductivities and exact face-flux differencing:

* **implicit** → stable at any timestep, which is what allows the large `dt` the frozen
  and cooling phases of a 3400 s run need;
* **conservative** → conduction is a flux difference that telescopes, so the discrete
  energy balance closes to round-off and the budget diagnostic means something;
* **enthalpy-based** → latent heat is absorbed across the mushy interval by construction,
  with no artificial smearing constant and no energy created at the solidification front.

Crank–Nicolson was rejected: it is A-unstable, and the thermal front at the solidification
boundary is stiff enough that CN rings — which appears directly as a wrong cooling rate,
i.e. a wrong grain structure prediction.

The only nonlinearity is `T = T(h)` (the enthalpy inverse), resolved by a Picard sweep
whose initial guess is the previous temperature field. Two sweeps normally suffice because
the conductivity varies weakly with `T`.

**Units note.** `h` is a *specific* enthalpy in J/kg, so the storage term is
`rho V (h_new - h_old)/dt`. The volumetric sources are powers in W/m³, so they enter as
`V q`. Mixing the two — dropping `rho` from the storage term — makes every cell's
apparent heat capacity a factor of ~7800 too small and heats the model off the scale.

## Workspace

`EnergyWorkspace` holds every buffer. A 3400 s run performs tens of thousands of solves,
and allocating ~20 MB per solve makes the garbage collector, not the physics, the
bottleneck — which is exactly the failure mode that makes "pragmatic" CFD code slow.
"""

"""
    EnergyWorkspace

Preallocated buffers for `solve_energy!`. Construct once, reuse for every step.
"""
struct EnergyWorkspace{T<:AbstractFloat}
    K::Array{T,3}
    Kfx::Array{T,3}
    Kfy::Array{T,3}
    Kfz::Array{T,3}
    src::Array{T,3}
    diag::Vector{T}
    off::Array{T,2}
    rhs::Vector{T}
    x::Vector{T}
    r::Vector{T}
    z::Vector{T}
    d::Vector{T}
    q::Vector{T}
    Told::Array{T,3}
    hout::Array{T,3}
    hlin::Array{T,3}      # enthalpy from the previous Picard sweep (linearisation base)
    Tsw::Array{T,3}       # scratch temperature for the sweep
    Tacc::Array{T,3}      # accumulated temperature update across sweeps
    resid::Array{T,3}     # per-cell nonlinear residual F_c of the current iterate
    Tcand::Array{T,3}     # line-search trial temperature
    hcand::Array{T,3}     # enthalpy of the line-search trial temperature
    fixed::Array{Bool,3}  # true on cells whose face is prescribed (Dirichlet)
    fixedT::Array{T,3}    # the prescribed value on those cells
end

function EnergyWorkspace(g::CartesianGrid)
    T = Float64
    nx, ny, nz = g.nx, g.ny, g.nz
    N = nx * ny * nz
    EnergyWorkspace{T}(
        zeros(T, nx, ny, nz),
        zeros(T, nx + 1, ny, nz),
        zeros(T, nx, ny + 1, nz),
        zeros(T, nx, ny, nz + 1),
        zeros(T, nx, ny, nz),
        zeros(T, N),
        zeros(T, 6, N),
        zeros(T, N),
        zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        zeros(T, nx, ny, nz),
        fill(false, nx, ny, nz),
        zeros(T, nx, ny, nz),
    )
end

"""
    _set_dirichlet!(fixed, fixedT, dirichlet, g) -> Bool

Fill the Dirichlet mask from a `NamedTuple` of face => temperature, e.g.
`(xlo = 6000.0, xhi = 300.0)`. Recognised faces are `:xlo :xhi :ylo :yhi :zlo :zhi`;
any face not named keeps the default adiabatic (zero-flux) condition. Returns `true`
when at least one face was prescribed, so callers can skip the masking work otherwise.

A prescribed value is imposed at the domain face itself, not at the first cell centre:
the boundary cell sees a conductance `k A / (Δx_b/2)` across its own half-cell, which is
exactly the ghost-cell condition `T = T_wall` located at the face. Placing the value at
the cell centre instead would shift the effective boundary by half a cell — an O(Δx)
error in the interface position, which is the entire quantity V11 measures.
"""
function _set_dirichlet!(fixed::Array{Bool,3}, fixedT::Array{T,3}, dirichlet,
                         g::CartesianGrid) where {T<:AbstractFloat}
    fill!(fixed, false)
    dirichlet === nothing && return false
    nx, ny, nz = g.nx, g.ny, g.nz
    @inbounds for (side, val) in pairs(dirichlet)
        v = T(val)
        if side === :xlo
            for k in 1:nz, j in 1:ny; fixed[1, j, k] = true; fixedT[1, j, k] = v; end
        elseif side === :xhi
            for k in 1:nz, j in 1:ny; fixed[nx, j, k] = true; fixedT[nx, j, k] = v; end
        elseif side === :ylo
            for k in 1:nz, i in 1:nx; fixed[i, 1, k] = true; fixedT[i, 1, k] = v; end
        elseif side === :yhi
            for k in 1:nz, i in 1:nx; fixed[i, ny, k] = true; fixedT[i, ny, k] = v; end
        elseif side === :zlo
            for j in 1:ny, i in 1:nx; fixed[i, j, 1] = true; fixedT[i, j, 1] = v; end
        elseif side === :zhi
            for j in 1:ny, i in 1:nx; fixed[i, j, nz] = true; fixedT[i, j, nz] = v; end
        else
            throw(ArgumentError("unknown energy boundary face :$side; expected one of " *
                                "xlo xhi ylo yhi zlo zhi"))
        end
    end
    true
end

"""
    solve_energy!(ws, T, p_old, mat, U, src_arc, src_droplet, src_loss, m, g, dt)

Advance `T` by one implicit energy step, in place. Returns
`(cgresidual, cgiterations, totalpower_in_W)`.

`p_old` is the temperature at the start of the step (used for the enthalpy increment).
Any of the source arrays may be `nothing`. `U` may be `nothing` for conduction-only.

The CG initial guess is `p_old` itself. For a small timestep the change is tiny, so this
starts the iteration essentially at the answer; starting from zero instead would spend
tens of iterations re-deriving a field that is already known.
"""
function solve_energy!(ws::EnergyWorkspace{Tf}, T::Array{Tf,3}, p_old::Array{Tf,3},
                       mat::Material, U, src_arc, src_droplet, src_loss,
                       m::GridMetrics, g::CartesianGrid, dt::Real;
                       rhofac::Real = 1, max_cg::Integer = 200,
                       tol::Real = 1e-7, max_sweeps::Integer = 12,
                       max_line_search::Integer = 8, dirichlet = nothing) where {Tf<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    N = nx * ny * nz
    dtt = Tf(dt)
    fixed, fixedT = ws.fixed, ws.fixedT
    has_bc = _set_dirichlet!(fixed, fixedT, dirichlet, g)
    K, Kfx, Kfy, Kfz, src = ws.K, ws.Kfx, ws.Kfy, ws.Kfz, ws.src
    diag, off, rhs = ws.diag, ws.off, ws.rhs

    # enthalpy at the start of the step; `hlin` is the linearisation base that the
    # Picard sweep updates, and `h` is the final enthalpy written back for the budget
    h = ws.hout
    hlin = ws.hlin
    Tsw = ws.Tsw
    Tacc = ws.Tacc
    @inbounds for idx in eachindex(h)
        h[idx] = enthalpy(mat, p_old[idx])
    end
    @inbounds for idx in eachindex(hlin)
        hlin[idx] = h[idx]
    end
    fill!(Tacc, zero(Tf))   # Newton correction starts from zero
    @inbounds for idx in eachindex(Tsw)
        Tsw[idx] = p_old[idx]
    end
    if has_bc
        @inbounds for idx in eachindex(Tsw)
            fixed[idx] && (Tsw[idx] = fixedT[idx])
        end
    end

    cgres = zero(Tf)
    cgit = 0
    pin = zero(Tf)
    sweeps_done = 0
    balance = Inf
    prev = Inf
    rnorm = zero(Tf)
    stalled = 0
    for sweep in 1:max_sweeps
    # Conductivities follow the ITERATE, not the caller's array: `T` is only written at
    # the very end of solve_energy!, so reading it here freezes the conductivity at the
    # step's starting temperature for every sweep and the Newton iteration then stalls
    # exactly where the conductivity is changing fastest -- across the mush.
    @inbounds for idx in eachindex(K)
        K[idx] = kcond(mat, Tsw[idx])
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        # face i sits between cells i-1 and i. `m.dxf[i]` is the width of cell i-1 and
        # `m.dxf[i+1]` the width of cell i. `harmonic_face_coeff` is homogeneous of degree
        # zero in the two distances (the factor 2 between a width and a half-width
        # cancels), so passing the full widths is exact and gives the same K_face as the
        # half-widths would.
        Kfx[i, j, k] = harmonic_face_coeff(K[i - 1, j, k], K[i, j, k],
                                           m.dxf[i], m.dxf[i + 1]) * m.ax[i, j, k]
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        Kfy[i, j, k] = harmonic_face_coeff(K[i, j - 1, k], K[i, j, k],
                                           m.dyf[j], m.dyf[j + 1]) * m.ay[i, j, k]
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        Kfz[i, j, k] = harmonic_face_coeff(K[i, j, k - 1], K[i, j, k],
                                           m.dzf[k], m.dzf[k + 1]) * m.az[i, j, k]
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        src[i, j, k] = (src_arc === nothing ? zero(Tf) : src_arc[i, j, k]) +
                       (src_droplet === nothing ? zero(Tf) : src_droplet[i, j, k]) -
                       (src_loss === nothing ? zero(Tf) : src_loss[i, j, k])
    end

    pinv = Tf(rhofac)
    pscale = zero(Tf)
    @inbounds for idx in eachindex(src)
        pscale += abs(src[idx]) * m.vol[idx]
    end
    # Reset per sweep. Carrying the previous sweep's value forward makes the line-search
    # acceptance test compare against a stale, far larger number, so a step that increases
    # the true residual is accepted as an improvement.
    rnorm = zero(Tf)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        V = m.vol[i, j, k]
        # rho on the storage term: h is a SPECIFIC enthalpy (J/kg), so the stored energy
        # is rho*V*h. Omitting rho makes the heat capacity of the cell a factor of rho
        # too small, which heats the model ~7800x too fast. Sources stay as V*src
        # because they are volumetric POWER (W/m^3), not specific.
        #
        # cp is evaluated at the PREVIOUS sweep's temperature, and rhs carries the
        # previous sweep's enthalpy, so the sweep is a genuine Newton/Picard step on
        # h(T) rather than a frozen linearisation.
        d = mat.rho_s * V * cp_effective(mat, Tsw[i, j, k]) / dtt
        if U !== nothing
            d += pinv / V * (_fc(U.u[i + 1, j, k], m.ax[i + 1, j, k], hlin, c) +
                             _fc(U.v[i, j + 1, k], m.ay[i, j + 1, k], hlin, c) +
                             _fc(U.w[i, j, k + 1], m.az[i, j, k + 1], hlin, c))
        end
        # Conductance is K_face * A / d_centre-to-centre. `m.dxc[f]` is that distance for
        # face f; dividing by a single cell width (`m.dxf[f]`) instead is exact only on a
        # uniform axis and wrong by up to ~37 % on the 3:1 graded mesh these cases run on.
        if i < nx; a = pinv * Kfx[i + 1, j, k] / m.dxc[i + 1]; d += a; off[1, c] = -a; end
        if i > 1;  a = pinv * Kfx[i, j, k] / m.dxc[i];         d += a; off[2, c] = -a; end
        if j < ny; a = pinv * Kfy[i, j + 1, k] / m.dyc[j + 1]; d += a; off[3, c] = -a; end
        if j > 1;  a = pinv * Kfy[i, j, k] / m.dyc[j];         d += a; off[4, c] = -a; end
        if k < nz; a = pinv * Kfz[i, j, k + 1] / m.dzc[k + 1]; d += a; off[5, c] = -a; end
        if k > 1;  a = pinv * Kfz[i, j, k] / m.dzc[k];         d += a; off[6, c] = -a; end
        if has_bc && fixed[i, j, k]
            # The prescribed cell's correction is identically zero, so its row is
            # replaced by `1·δ = 0`. Its neighbours keep their diagonal (the
            # conductance is still part of ∂F_n/∂T_n) and their coupling to this cell
            # is dropped in the pass below.
            diag[c] = one(Tf)
            @inbounds for r in 1:6
                off[r, c] = zero(Tf)
            end
        else
            diag[c] = d
        end
        # Newton iteration on the fully implicit residual
        #     F(T) = ρV(h(T) − h_old)/dt + Σa(T − T_nb) − V·src = 0
        # The linear system solved is the NEWTON SYSTEM for the CORRECTION δ = T − T_k:
        #     F'(T_k)·δ = −F(T_k)
        # with F'(T) = ρV·cp(T)/dt + Σa, i.e. exactly the matrix assembled above.
        #
        # Assembling the system for T directly (rather than for the correction) makes
        # every sweep re-apply the source from the *updated* iterate, so the temperature
        # climbs by one increment per sweep instead of converging — measured at 2x and
        # then 3x the correct rise. Solving for the correction converges in one sweep for
        # a linear property and two or three at the mushy front.
        Fk = mat.rho_s * V * (hlin[c] - h[c]) / dtt
        if i < nx; Fk += (Kfx[i + 1, j, k] / m.dxc[i + 1]) * (Tsw[i, j, k] - Tsw[i + 1, j, k]); end
        if i > 1;  Fk += (Kfx[i, j, k] / m.dxc[i])         * (Tsw[i, j, k] - Tsw[i - 1, j, k]); end
        if j < ny; Fk += (Kfy[i, j + 1, k] / m.dyc[j + 1]) * (Tsw[i, j, k] - Tsw[i, j + 1, k]); end
        if j > 1;  Fk += (Kfy[i, j, k] / m.dyc[j])         * (Tsw[i, j, k] - Tsw[i, j - 1, k]); end
        if k < nz; Fk += (Kfz[i, j, k + 1] / m.dzc[k + 1]) * (Tsw[i, j, k] - Tsw[i, j, k + 1]); end
        if k > 1;  Fk += (Kfz[i, j, k] / m.dzc[k])         * (Tsw[i, j, k] - Tsw[i, j, k - 1]); end
        # The advection term enters the residual exactly as it enters the diagonal, so the
        # two stay in lockstep and the line search measures the system actually being
        # solved. Note the upwind divergence is currently one-sided (only the +x/+y/+z
        # faces) and its diagonal coefficient is not the true Newton derivative; that is
        # pending `step_fluid!`, and every live call site passes `U = nothing` today.
        if U !== nothing
            Fk += pinv * (_fc(U.u[i + 1, j, k], m.ax[i + 1, j, k], hlin, c) +
                          _fc(U.v[i, j + 1, k], m.ay[i, j + 1, k], hlin, c) +
                          _fc(U.w[i, j, k + 1], m.az[i, j, k + 1], hlin, c))
        end
        if has_bc && fixed[i, j, k]
            rhs[c] = zero(Tf)
            ws.resid[c] = zero(Tf)
        else
            rhs[c] = V * src[i, j, k] - Fk
            # Fk IS the true per-cell nonlinear residual of the fully implicit enthalpy
            # equation, so it is recorded rather than recomputed later. `rnorm` is
            # therefore the honest L-infinity measure the Newton iteration drives to zero.
            ws.resid[c] = Fk
            rnorm = max(rnorm, abs(Fk))
        end
    end

    # Drop every matrix entry that touches a prescribed cell. The prescribed row is
    # already `1·δ = 0`, so a neighbour's coupling to it evaluates to `off·0 = 0` and
    # would be harmless on its own — but zeroing only that side leaves J asymmetric,
    # and CG is only valid for a symmetric operator, so both sides go.
    if has_bc
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            fixed[i, j, k] || continue
            c = i + (j - 1) * nx + (k - 1) * nx * ny
            for r in 1:6
                off[r, c] = zero(Tf)
            end
            if i < nx; off[2, (i + 1) + (j - 1) * nx + (k - 1) * nx * ny] = zero(Tf); end
            if i > 1; off[1, (i - 1) + (j - 1) * nx + (k - 1) * nx * ny] = zero(Tf); end
            if j < ny; off[4, i + j * nx + (k - 1) * nx * ny] = zero(Tf); end
            if j > 1; off[3, i + (j - 2) * nx + (k - 1) * nx * ny] = zero(Tf); end
            if k < nz; off[6, i + (j - 1) * nx + k * nx * ny] = zero(Tf); end
            if k > 1; off[5, i + (j - 1) * nx + (k - 2) * nx * ny] = zero(Tf); end
        end
    end

    # the correction system is homogeneous in the source, so start from zero
    cgres, cgit = _cg!(ws, rhs, diag, off, nx, ny, nz, ws.Tacc, tol, max_cg)

    # Apply the Newton correction, with a backtracking line search.
    #
    # h(T) is piecewise LINEAR with a slope that jumps by L_f/(T_liq−T_sol) at each phase
    # boundary, so F(T) is monotone but not C1. Undamped Newton steps straight over a
    # kink and cycles: measured on a single molten cell at 1738 K surrounded by 300 K
    # cells, the residual froze at 4.9 W instead of converging. Backtracking guarantees
    # the residual decreases monotonically inside a linear region; when no step shortens
    # it we are standing exactly on a kink, so the full step is taken to cross into the
    # next linear region, after which Newton converges finitely. This changes nothing
    # about the converged answer — F = 0 is still solved exactly — only the path to it.
    ω = one(Tf)
    accepted = false
    rnew = rnorm
    for _ls in 1:max_line_search
        @inbounds for idx in eachindex(ws.Tcand)
            ws.Tcand[idx] = Tsw[idx] + ω * ws.x[idx]
            if has_bc && fixed[idx]
                ws.Tcand[idx] = fixedT[idx]
            end
            ws.hcand[idx] = enthalpy(mat, ws.Tcand[idx])
        end
        # `fixed` is mandatory here: without it the line search also measures the
        # Dirichlet cell's *replaced* equation, whose residual is the (large, and never
        # satisfiable) imbalance of a cell the system does not solve for. Every step then
        # fails the `rnew < rnorm` test and the solver falls back to undamped full steps.
        rnew = _residual_norm(ws.Tcand, ws.hcand, p_old, mat, U, src, m, g, dtt, ws.resid;
                              rhofac = pinv, fixed = fixed)
        if rnew < rnorm
            copyto!(Tsw, ws.Tcand)
            accepted = true
            break
        end
        ω *= Tf(0.5)
    end
    if !accepted
        # On a kink: no shorter step helps, so take the full Newton step.
        ω = one(Tf)
        @inbounds for idx in eachindex(Tsw)
            Tsw[idx] += ws.x[idx]
        end
        @inbounds for idx in eachindex(hlin)
            hlin[idx] = enthalpy(mat, Tsw[idx])
        end
        rnew = _residual_norm(Tsw, hlin, p_old, mat, U, src, m, g, dtt, ws.resid;
                              rhofac = pinv, fixed = fixed)
    end
    @inbounds for idx in eachindex(hlin)
        hlin[idx] = enthalpy(mat, Tsw[idx])
    end
    sweeps_done = sweep
    balance = rnew
    rnorm = rnew

    # Convergence is the residual MAGNITUDE against the source scale — not the change in
    # the residual. Judging convergence by `|balance − prev|` declares success on a
    # solver that has merely stopped moving, which is precisely how an unconverged mushy
    # front used to be reported as done. A stagnation guard is kept alongside it so a
    # genuinely stuck sweep does not burn all `max_sweeps` doing nothing.
    if balance <= Tf(tol) * max(pscale, 1.0)
        break
    end
    if prev !== Inf && balance >= prev * (1 - 1e-3)
        stalled += 1
        stalled >= 3 && break
    else
        stalled = 0
    end
    prev = balance
    end

    copyto!(T, Tsw)
    if has_bc
        @inbounds for idx in eachindex(T)
            fixed[idx] && (T[idx] = fixedT[idx])
        end
    end
    @inbounds for idx in eachindex(h)
        h[idx] = enthalpy(mat, T[idx])
    end

    # total power entering the domain, for the energy audit
    pin = 0.0
    @inbounds for idx in eachindex(src)
        pin += src[idx] * m.vol[idx]
    end
    (balance, sweeps_done, pin)
end

# Upwind enthalpy flux through a face: `c` is the current cell's linear index, and the
# donor is c (outflow) or c+1 (inflow) depending on the face velocity sign.
@inline function _fc(uf::Tf, Af::Tf, h, c::Int) where {Tf<:AbstractFloat}
    uf > 0 ? uf * Af * h[c] : uf * Af * h[c + 1]
end

"""
    _residual_norm(T, h, p_old, mat, U, src, m, g, dt, resid; rhofac) -> Float64  [W]

L-infinity norm of the **per-cell** nonlinear residual of the implicit enthalpy equation,

```
F_c = ρV_c (h(T_c) − h_old,c)/dt + Σ_f a_f (T_c − T_nb) + advection − V_c src_c
```

which is exactly the right-hand side the Newton correction system is built from. Fills
`resid` with the per-cell values so a caller can locate *where* the solve is failing.

The L-infinity norm is deliberate. The earlier diagnostic returned
`|Σ_c F_c|` — a global sum — which is zero by cancellation whenever the residuals have
mixed signs, so it can report a perfectly converged solve while whole cells are far from
satisfied, and it reports a large number for a converged solve whose only asymmetry is a
temperature-dependent conductivity. Only the per-cell norm answers "did the iteration
converge".
"""
function _residual_norm(T::Array{Tf,3}, h, p_old::Array{Tf,3}, mat::Material, U, src,
                        m::GridMetrics, g::CartesianGrid, dtt, resid;
                        rhofac::Real = 1, fixed = nothing) where {Tf<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    nrm = zero(Tf)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        if fixed !== nothing && fixed[i, j, k]
            # The equation in this cell IS `T = T_wall`, and it is satisfied by
            # construction, so it contributes nothing to the Newton residual.
            resid[c] = zero(Tf)
            continue
        end
        V = m.vol[i, j, k]
        Kc = kcond(mat, T[i, j, k])
        F = Tf(rhofac) * mat.rho_s * V * (h[c] - enthalpy(mat, p_old[c])) / dtt
        if i < nx
            a = Tf(rhofac) * harmonic_face_coeff(Kc, kcond(mat, T[i + 1, j, k]),
                                                 m.dxf[i + 1], m.dxf[i + 2]) *
                m.ax[i + 1, j, k] / m.dxc[i + 1]
            F += a * (T[i, j, k] - T[i + 1, j, k])
        end
        if i > 1
            a = Tf(rhofac) * harmonic_face_coeff(kcond(mat, T[i - 1, j, k]), Kc,
                                                 m.dxf[i], m.dxf[i + 1]) *
                m.ax[i, j, k] / m.dxc[i]
            F += a * (T[i, j, k] - T[i - 1, j, k])
        end
        if j < ny
            a = Tf(rhofac) * harmonic_face_coeff(Kc, kcond(mat, T[i, j + 1, k]),
                                                 m.dyf[j + 1], m.dyf[j + 2]) *
                m.ay[i, j + 1, k] / m.dyc[j + 1]
            F += a * (T[i, j, k] - T[i, j + 1, k])
        end
        if j > 1
            a = Tf(rhofac) * harmonic_face_coeff(kcond(mat, T[i, j - 1, k]), Kc,
                                                 m.dyf[j], m.dyf[j + 1]) *
                m.ay[i, j, k] / m.dyc[j]
            F += a * (T[i, j, k] - T[i, j - 1, k])
        end
        if k < nz
            a = Tf(rhofac) * harmonic_face_coeff(Kc, kcond(mat, T[i, j, k + 1]),
                                                 m.dzf[k + 1], m.dzf[k + 2]) *
                m.az[i, j, k + 1] / m.dzc[k + 1]
            F += a * (T[i, j, k] - T[i, j, k + 1])
        end
        if k > 1
            a = Tf(rhofac) * harmonic_face_coeff(kcond(mat, T[i, j, k - 1]), Kc,
                                                 m.dzf[k], m.dzf[k + 1]) *
                m.az[i, j, k] / m.dzc[k]
            F += a * (T[i, j, k] - T[i, j, k - 1])
        end
        if U !== nothing
            F += Tf(rhofac) * (_fc(U.u[i + 1, j, k], m.ax[i + 1, j, k], h, c) +
                               _fc(U.v[i, j + 1, k], m.ay[i, j + 1, k], h, c) +
                               _fc(U.w[i, j, k + 1], m.az[i, j, k + 1], h, c))
        end
        F -= V * src[i, j, k]
        resid[c] = F
        nrm = max(nrm, abs(F))
    end
    nrm
end

"""
    global_energy_budget(T, h, p_old, mat, U, src, m, g, dt; rhofac) -> Float64  [W]

Global energy-budget closure error: `|Σ_c F_c|`, the sum of the per-cell residuals.

For a closed domain every internal conduction face is counted from both sides with
opposite signs, so this reduces to `|Σ ρV(h−h_old)/dt − Σ V src|` — the quantity
verification test V20 asks for. It is a **budget** audit, not a convergence measure: it
is blind to cells that are individually wrong with opposite signs. `_residual_norm` is
the convergence measure; this is the conservation measure. Reporting one as the other is
how a leaking scheme gets a clean bill of health.
"""
function global_energy_budget(T::Array{Tf,3}, h, p_old::Array{Tf,3}, mat::Material, U,
                              src, m::GridMetrics, g::CartesianGrid, dt;
                              rhofac::Real = 1, resid = nothing,
                              fixed = nothing) where {Tf<:AbstractFloat}
    r = resid === nothing ? Array{Tf,3}(undef, size(T)...) : resid
    _residual_norm(T, h, p_old, mat, U, src, m, g, Tf(dt), r; rhofac = rhofac,
                   fixed = fixed)
    s = zero(Tf)
    @inbounds for idx in eachindex(r)
        s += r[idx]
    end
    abs(s)
end

"""
    _cg!(ws, b, diag, off, nx, ny, nz, x0, tol, maxit) -> (residual, iterations)

Jacobi-preconditioned CG on the 7-point energy matrix. The `V·C_p/dt` block makes this
strongly diagonally dominant, so convergence is fast and the Jacobi preconditioner is
adequate — a stronger preconditioner would add cost for nothing here (unlike the pressure
Poisson system, which genuinely needs multigrid).
"""
function _cg!(ws::EnergyWorkspace{Tf}, b, diag, off, nx, ny, nz, x0, tol, maxit) where {Tf<:AbstractFloat}
    N = nx * ny * nz
    x, r, z, d, q = ws.x, ws.r, ws.z, ws.d, ws.q
    @inbounds for idx in 1:N
        x[idx] = x0[idx]
    end
    _applyE!(r, x, diag, off, nx, ny, nz)
    @inbounds for idx in 1:N
        r[idx] = b[idx] - r[idx]
        z[idx] = diag[idx] > 0 ? r[idx] / diag[idx] : zero(Tf)
        d[idx] = z[idx]
    end
    bnorm = sqrt(sum(abs2, b))
    bnorm == 0 && return (zero(Tf), 0)
    rz = dot(r, z)
    for it in 1:maxit
        _applyE!(q, d, diag, off, nx, ny, nz)
        dq = dot(d, q)
        dq > 0 || return (sqrt(dot(r, r)), it)
        α = rz / dq
        @inbounds @simd for idx in 1:N
            x[idx] += α * d[idx]
            r[idx] -= α * q[idx]
        end
        rn = sqrt(dot(r, r))
        rn <= Tf(tol) * bnorm && return (rn, it)
        @inbounds @simd for idx in 1:N
            z[idx] = diag[idx] > 0 ? r[idx] / diag[idx] : zero(Tf)
        end
        rz_new = dot(r, z)
        β = rz_new / rz
        @inbounds @simd for idx in 1:N
            d[idx] = z[idx] + β * d[idx]
        end
        rz = rz_new
    end
    (sqrt(dot(r, r)), maxit)
end

function _applyE!(y, x, diag, off, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        s = diag[c] * x[c]
        i < nx && (s += off[1, c] * x[c + 1])
        i > 1  && (s += off[2, c] * x[c - 1])
        j < ny && (s += off[3, c] * x[c + nx])
        j > 1  && (s += off[4, c] * x[c - nx])
        k < nz && (s += off[5, c] * x[c + nx * ny])
        k > 1  && (s += off[6, c] * x[c - nx * ny])
        y[c] = s
    end
    y
end

"""
    energy_residual(T, h, p_old, mat, U, src_arc, src_droplet, src_loss, m, g, dt)

Global energy-budget closure error [W]: `|Σ_c F_c|` for the implicit enthalpy equation.

The audit number behind verification test **V20**. A conservative scheme drives this to
round-off; a leaking one makes it grow without bound. Comparing it against the total
power in over a run is the honest check that a simulation conserved energy rather than
merely appearing to.

It is a *budget*, not a convergence measure — see `global_energy_budget`, which explains
why the two must never be reported as each other. For "did this timestep's Newton
iteration converge", use the return value of `solve_energy!`, which is the per-cell
L-infinity residual.
"""
function energy_residual(T::Array{Tf,3}, h::Array{Tf,3}, p_old::Array{Tf,3},
                         mat::Material, U, src_arc, src_droplet, src_loss,
                         m::GridMetrics, g::CartesianGrid, dt::Real;
                         rhofac::Real = 1, fixed = nothing) where {Tf<:AbstractFloat}
    src = Array{Tf,3}(undef, size(T)...)
    @inbounds for idx in eachindex(src)
        src[idx] = (src_arc === nothing ? zero(Tf) : Tf(src_arc[idx])) +
                   (src_droplet === nothing ? zero(Tf) : Tf(src_droplet[idx])) -
                   (src_loss === nothing ? zero(Tf) : Tf(src_loss[idx]))
    end
    global_energy_budget(T, h, p_old, mat, U, src, m, g, dt; rhofac = rhofac, fixed = fixed)
end

max_temperature(T) = maximum(T)
min_temperature(T) = minimum(T)

"""
    latent_heat_fraction(T, mat) -> Float64

Volume fraction currently molten. During solidification the *rate of change* of this is the
solidification front velocity, which is what sets the grain structure — so it is a cheap
and directly useful monitor.
"""
function latent_heat_fraction(T::AbstractArray, mat::Material)
    s = 0.0
    @inbounds for v in T
        s += liquid_fraction(mat, v)
    end
    s / length(T)
end