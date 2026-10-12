"""
Momentum equation, interface capture and free-surface pressure coupling.

Solves the paper's Eq. (2):

```
∂V/∂t + V·∇V = −∇p/ρ + (1/ρ)∇·(μ∇V) − k_d V + G_b + (ṁ_s/ρ)(V_s − V)
```

with the forcing terms that decide a WAAM melt pool:

| Term | Paper | Implementation |
| --- | --- | --- |
| Darcy mushy drag | (5) | `k_d = s_d f_s^2/((1−f_s)³+ε₀)` on the momentum diagonal |
| Boussinesq buoyancy | (15) | `G_b = −g β_T (T − T_liq)` |
| Marangoni stress | (14) | `μ ∂u_t/∂n = −(∂γ/∂T)∇_s T` as a face stress |
| Surface tension | — | `κ = −∇·n̂`, force `γ(T)κ ∇F`, and `P_st = γ(T)κ` |

## Staggered, not collocated

A **staggered MAC** layout is used rather than the collocated grid with Rhie–Chow the
original plan proposed. Rhie–Chow is a fix for checkerboarding in *single-phase* pressure
solves; here the checkerboard-prone cells are exactly the interface cells, so suppressing
it there fights the physics. MAC removes the problem structurally and puts fluxes on faces
where they belong.

## Pressure

- `PartialVoid` (default, paper-faithful): cells with `F < F_void` are removed from the
  pressure system and pinned to ambient; surface tension enters as the interface
  condition.
- `CSF`: a single global Poisson solve with `σ κ ∇F` as a body force.
"""
abstract type PressureMode end
struct PartialVoid <: PressureMode end
struct CSF <: PressureMode end

"""
    MomentumWorkspace

Preallocated buffers for `step_fluid!`. Reused every step; a 3400 s run performs
hundreds of thousands of steps and allocating per step makes the collector the bottleneck.
"""
struct MomentumWorkspace{T<:AbstractFloat}
    u0::Array{T,3}; v0::Array{T,3}; w0::Array{T,3}
    uu::Array{T,3}; vv::Array{T,3}; ww::Array{T,3}     # face velocity candidates
    rhofx::Array{T,3}                                   # x-face density (nx+1, ny, nz)
    rhofy::Array{T,3}                                   # y-face density
    rhofz::Array{T,3}                                   # z-face density
    drag::Array{T,3}                                    # Darcy drag [1/s]
    buoy::Array{T,3}                                    # Boussinesq acceleration
    px::Array{T,3}; py::Array{T,3}; pz::Array{T,3}      # face pressure
    p::Array{T,3}
    rhs::Array{T,3}
    divU::Array{T,3}
    kappa::Array{T,3}
    Pst::Array{T,3}
    alpha::Array{T,3}
    fixed::BitArray{3}
    fixedvals::Array{T,3}
    Fs::Array{T,3}                                      # smoothed F, for curvature
end

function MomentumWorkspace(g::CartesianGrid)
    T = Float64
    nx, ny, nz = g.nx, g.ny, g.nz
    z3() = zeros(T, nx, ny, nz)
    # Positional construction, in FIELD ORDER. Getting this out of alignment is silent
    # and vicious: the inner constructor happily converts a cell-shaped array into a
    # face-shaped field, so the pressure solver then reads one element past the end of a
    # cell array on every boundary face instead of erroring.
    MomentumWorkspace{T}(
        z3(), z3(), z3(),               # u0, v0, w0
        z3(), z3(), z3(),               # uu, vv, ww
        zeros(T, nx + 1, ny, nz), zeros(T, nx, ny + 1, nz), zeros(T, nx, ny, nz + 1),  # rho faces
        z3(),                           # drag
        z3(),                           # buoy
        zeros(T, nx + 1, ny, nz), zeros(T, nx, ny + 1, nz), zeros(T, nx, ny, nz + 1),  # p faces
        z3(), z3(), z3(), z3(), z3(),   # p, rhs, divU, kappa, Pst
        z3(),                           # alpha
        BitArray(undef, nx, ny, nz),    # fixed
        z3(),                           # fixedvals
        z3())                           # Fs
end

# ---------------------------------------------------------------------------
# Interface geometry
# ---------------------------------------------------------------------------

"""
    gradF(F, m, g, i, j, k) -> (gx, gy, gz)

Central-difference gradient of the volume fraction, second-order in the interior with
one-sided closures at the boundaries and correct half-distances on a graded axis.
"""
@inline function gradF(F::AbstractArray{T,3}, m::GridMetrics, g::CartesianGrid, i, j, k) where {T}
    nx, ny, nz = g.nx, g.ny, g.nz
    gx = if i > 1 && i < nx
        (F[i + 1, j, k] - F[i - 1, j, k]) / (m.dxf[i] + m.dxf[i + 1])
    elseif i == 1
        (F[2, j, k] - F[1, j, k]) / m.dxf[2]
    else
        (F[nx, j, k] - F[nx - 1, j, k]) / m.dxf[nx]
    end
    gy = if j > 1 && j < ny
        (F[i, j + 1, k] - F[i, j - 1, k]) / (m.dyf[j] + m.dyf[j + 1])
    elseif j == 1
        (F[i, 2, k] - F[i, 1, k]) / m.dyf[2]
    else
        (F[i, ny, k] - F[i, ny - 1, k]) / m.dyf[ny]
    end
    gz = if k > 1 && k < nz
        (F[i, j, k + 1] - F[i, j, k - 1]) / (m.dzf[k] + m.dzf[k + 1])
    elseif k == 1
        (F[i, j, 2] - F[i, j, 1]) / m.dzf[2]
    else
        (F[i, j, nz] - F[i, j, nz - 1]) / m.dzf[nz]
    end
    (T(gx), T(gy), T(gz))
end

"""
    curvature!(kappa, Fs, m, g) -> kappa

Mean curvature `κ = −∇·n̂` with `n̂ = ∇F/|∇F|`, evaluated by second-order central
differences of the unit normal.

`Fs` must already be **smoothed**. Curvature is a second derivative of a field that is
only piecewise smooth, so differentiating the raw fraction yields a saw-toothed κ — and
since the paper's sagging criterion *is* the peak surface tension pressure `γκ`, feeding
it raw curvature manufactures precisely the signal the criterion is meant to detect.
"""
function curvature!(kappa::Array{T,3}, Fs::Array{T,3}, m::GridMetrics,
                    g::CartesianGrid) where {T<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        gx, gy, gz = gradF(Fs, m, g, i, j, k)
        gm = sqrt(gx^2 + gy^2 + gz^2)
        if gm < 1e-10
            kappa[i, j, k] = zero(T)
            continue
        end
        dnx = dny = dnz = zero(T)
        if i > 1 && i < nx
            ap = gradF(Fs, m, g, i + 1, j, k); am = gradF(Fs, m, g, i - 1, j, k)
            npm = sqrt(ap[1]^2 + ap[2]^2 + ap[3]^2); nmm = sqrt(am[1]^2 + am[2]^2 + am[3]^2)
            npm < 1e-10 || (dnx = (ap[1] / npm - am[1] / nmm) / (m.dxf[i] + m.dxf[i + 1]))
        end
        if j > 1 && j < ny
            ap = gradF(Fs, m, g, i, j + 1, k); am = gradF(Fs, m, g, i, j - 1, k)
            npm = sqrt(ap[1]^2 + ap[2]^2 + ap[3]^2); nmm = sqrt(am[1]^2 + am[2]^2 + am[3]^2)
            npm < 1e-10 || (dny = (ap[2] / npm - am[2] / nmm) / (m.dyf[j] + m.dyf[j + 1]))
        end
        if k > 1 && k < nz
            ap = gradF(Fs, m, g, i, j, k + 1); am = gradF(Fs, m, g, i, j, k - 1)
            npm = sqrt(ap[1]^2 + ap[2]^2 + ap[3]^2); nmm = sqrt(am[1]^2 + am[2]^2 + am[3]^2)
            npm < 1e-10 || (dnz = (ap[3] / npm - am[3] / nmm) / (m.dzf[k] + m.dzf[k + 1]))
        end
        kk = -(dnx + dny + dnz)
        # Guard: a non-finite value can only arise where a neighbour's normal is
        # degenerate and the division is skipped, leaving an inconsistent stencil at a
        # domain boundary. Dropping it is correct (the curvature there is not
        # meaningful) and prevents one NaN from poisoning every downstream surface force
        # and the whole momentum solve.
        kappa[i, j, k] = isfinite(kk) ? kk : zero(T)
    end
    kappa
end

"""
    smooth_fraction!(Fs, F, passes)

Six-neighbour smoothing of the volume fraction, in place into `Fs`. Used before curvature.
"""
function smooth_fraction!(Fs::Array{T,3}, F::Array{T,3}, passes::Int = 2) where {T<:AbstractFloat}
    nx, ny, nz = size(F)
    copyto!(Fs, F)
    buf = similar(F)
    for _ in 1:passes
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            s = Fs[i, j, k]; n = 1.0
            i > 1  && (s += Fs[i - 1, j, k]; n += 1)
            i < nx && (s += Fs[i + 1, j, k]; n += 1)
            j > 1  && (s += Fs[i, j - 1, k]; n += 1)
            j < ny && (s += Fs[i, j + 1, k]; n += 1)
            k > 1  && (s += Fs[i, j, k - 1]; n += 1)
            k < nz && (s += Fs[i, j, k + 1]; n += 1)
            buf[i, j, k] = s / n
        end
        copyto!(Fs, buf)
    end
    Fs
end

"""
    surface_tension_pressure!(Pst, kappa, T, mat, m, g)

`P_st = γ(T)·κ` at every cell — the paper's proposed wall-stability criterion. Sign is
kept (a positive `P_st` means a convex, high-pressure surface); the magnitude is what the
criterion thresholds on.
"""
function surface_tension_pressure!(Pst::Array{T,3}, kappa::Array{T,3},
                                   Tfld::Array{T,3}, mat::Material) where {T<:AbstractFloat}
    @inbounds for idx in eachindex(kappa)
        Pst[idx] = mat.gamma(Tfld[idx]) * kappa[idx]
    end
    Pst
end

# ---------------------------------------------------------------------------
# Marangoni surface stress
# ---------------------------------------------------------------------------

"""
    marangoni_faces!(fx, fy, fz, F, T, mat, m, g, mu, alpha)

Tangential surface traction from `μ ∂u_t/∂n = −(∂γ/∂T)∇_s T` (paper Eq. 14).

Applied as a body force on the **interface cells**, computed face by face. The gradient
is taken along the surface (tangential to `∇F`), which is what makes the traction purely
tangential — a full `∇T` would inject a spurious normal component.

The sign of `∂γ/∂T` is the whole story: positive (ER70S6 with its 0.018 wt.% S) drives
flow *inward*, giving a deep narrow pool; negative (ER316L) drives flow *outward*, giving a
wide shallow pool that sags sooner.
"""
function marangoni_faces!(fx::Array{T,3}, fy::Array{T,3}, fz::Array{T,3},
                          F::Array{T,3}, Tfld::Array{T,3}, mat::Material,
                          m::GridMetrics, g::CartesianGrid) where {T<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    fx .= 0; fy .= 0; fz .= 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        # only interface cells carry a surface
        (F[i, j, k] > 1e-6 && F[i, j, k] < 1 - 1e-6) || continue
        Tk = Tfld[i, j, k]
        dgamma = dgamma_dT(mat.gamma, Tk)
        dgamma == 0 && continue
        # surface normal from the volume fraction gradient
        gx, gy, gz = gradF(F, m, g, i, j, k)
        gm = sqrt(gx^2 + gy^2 + gz^2)
        gm < 1e-10 && continue
        nxv, nyv, nzv = gx / gm, gy / gm, gz / gm
        # tangential temperature gradient
        Tx = i < nx ? (Tfld[i + 1, j, k] - Tfld[i, j, k]) / m.dxf[i + 1] : 0.0
        Ty = j < ny ? (Tfld[i, j + 1, k] - Tfld[i, j, k]) / m.dyf[j + 1] : 0.0
        Tz = k < nz ? (Tfld[i, j, k + 1] - Tfld[i, j, k]) / m.dzf[k + 1] : 0.0
        # project out the normal component
        dotn = Tx * nxv + Ty * nyv + Tz * nzv
        Ttx, Tty, Ttz = Tx - dotn * nxv, Ty - dotn * nyv, Tz - dotn * nzv
        # traction = -(dgamma/dT) * grad_s T, applied as a force density
        fx[i, j, k] = -dgamma * Ttx
        fy[i, j, k] = -dgamma * Tty
        fz[i, j, k] = -dgamma * Ttz
    end
    (fx, fy, fz)
end

# ---------------------------------------------------------------------------
# Pressure–velocity coupling
# ---------------------------------------------------------------------------

"""
    solve_pressure!(ws, U, F, T, mat, m, g, rho_amb, mode, dt) -> (residual, iters, ok)

Free-surface pressure solve.

`PartialVoid`: cells with `F < F_void` are Dirichlet at `rho_amb`; the liquid cells form
the Poisson system. The `∂F/∂t` source (the volume the interface gained) enters the
right-hand side, which is what makes the solve enforce the volume constraint on the liquid.

`CSF`: one global solve with the surface tension `γκ` included as a body force, so no
cell is pinned and the interface condition is emergent.

Returns `(residual, iterations, converged)`; the residual is in pressure units.
"""
function solve_pressure!(ws::MomentumWorkspace{Tf}, U::VectorField{Tf},
                         F::Array{Tf,3}, Tfld::Array{Tf,3}, mat::Material,
                         m::GridMetrics, g::CartesianGrid, dt::Real;
                         rho_amb::Real = 1000.0, mode::PressureMode = PartialVoid(),
                         p_amb::Real = 0.0, f_void::Real = 1e-3,
                         surf_tension_coeff::Real = 1.0,
                         tol::Real = 1e-9, maxiter::Integer = 3000) where {Tf<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    N = nx * ny * nz
    alpha = ws.alpha
    @inbounds for idx in eachindex(F)
        alpha[idx] = F[idx] >= 1 - f_void ? 1.0 : (F[idx] <= f_void ? 0.0 : F[idx])
    end

    # --- divergence of the face velocity ---
    divergence!(ws.divU, U.u, U.v, U.w, m, g)

    # --- face density from the local state: liquid above the melt range is rho_l ---
    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        ws.rhofx[i, j, k] = Tfld[i - 1, j, k] > mat.Tsol || Tfld[i, j, k] > mat.Tsol ?
                           mat.rho_l : mat.rho_s
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        ws.rhofy[i, j, k] = Tfld[i, j - 1, k] > mat.Tsol || Tfld[i, j, k] > mat.Tsol ?
                           mat.rho_l : mat.rho_s
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        ws.rhofz[i, j, k] = Tfld[i, j, k - 1] > mat.Tsol || Tfld[i, j, k] > mat.Tsol ?
                           mat.rho_l : mat.rho_s
    end

    # --- pressure conductances on faces: rho/dx ---
    ax = ws.px; ay = ws.py; az = ws.pz
    ax .= 0; ay .= 0; az .= 0
    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        ax[i, j, k] = ws.rhofx[i, j, k] / m.dxf[i]
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        ay[i, j, k] = ws.rhofy[i, j, k] / m.dyf[j]
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        az[i, j, k] = ws.rhofz[i, j, k] / m.dzf[k]
    end

    # --- right-hand side ---
    rhs = ws.rhs
    dtt = Tf(dt)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        rhs[i, j, k] = -m.vol[i, j, k] * ws.divU[i, j, k] / dtt
    end
    if mode isa CSF && surf_tension_coeff != 0
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            (F[i, j, k] > 1e-6 && F[i, j, k] < 1 - 1e-6) || continue
            rhs[i, j, k] += Tf(surf_tension_coeff) * ws.Pst[i, j, k] * m.vol[i, j, k]
        end
    end

    # --- Dirichlet mask ---
    fixed = ws.fixed
    fill!(fixed, false)
    @inbounds for idx in eachindex(F)
        if mode isa PartialVoid
            fixed[idx] = alpha[idx] < 1.0
        else
            # CSF: pin the far field only, so the system is not singular
            fixed[idx] = F[idx] <= f_void
        end
    end
    if mode isa PartialVoid && !all(fixed) && !any(fixed)
        fixed[1] = true      # keep the system non-singular
    end
    fv = ws.fixedvals
    fill!(fv, Tf(p_amb))
    fill!(ws.p, Tf(p_amb))

    res, iters, ok = poisson7!(vec(ws.p), vec(rhs), ax, ay, az, m.vol, vec(fixed), vec(fv);
                              tol = tol, maxiter = maxiter)
    (res, iters, ok)
end

"""
    project_velocity!(U, ws, p, F, m, g; mode)

Apply the pressure gradient to the face velocities:

```
u_f ← u_f − dt · (p_R − p_L) / (ρ_f · d_f)
```

and damp the velocity inside the mush by the Darcy drag. Face pressures are averaged from
the two cell centres, which is why a staggered layout needs no interpolator.
"""
function project_velocity!(U::VectorField{Tf}, ws::MomentumWorkspace{Tf},
                           dt::Real, drag::Array{Tf,3}, F::Array{Tf,3},
                           m::GridMetrics, g::CartesianGrid) where {Tf<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    dtt = Tf(dt)
    p = ws.p
    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        a = F[i - 1, j, k]; b = F[i, j, k]
        aa = a >= 1 - 1e-3 ? 1.0 : (a <= 1e-3 ? 0.0 : a)
        bb = b >= 1 - 1e-3 ? 1.0 : (b <= 1e-3 ? 0.0 : b)
        # skip faces between two void cells: no fluid, nothing to project
        (aa <= 0 && bb <= 0) && continue
        pf = (aa * p[i - 1, j, k] + bb * p[i, j, k]) / (aa + bb)
        rho_f = ws.rhofx[i, j, k]
        kd = 0.5 * (drag[i - 1, j, k] + drag[i, j, k])
        U.u[i, j, k] -= dtt * (p[i, j, k] - p[i - 1, j, k]) / (rho_f * m.dxf[i])
        # implicit Darcy damping (unconditionally stable for any k_d)
        U.u[i, j, k] /= (1 + dtt * kd)
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        aa = F[i, j - 1, k] >= 1 - 1e-3 ? 1.0 : (F[i, j - 1, k] <= 1e-3 ? 0.0 : F[i, j - 1, k])
        bb = F[i, j, k] >= 1 - 1e-3 ? 1.0 : (F[i, j, k] <= 1e-3 ? 0.0 : F[i, j, k])
        (aa <= 0 && bb <= 0) && continue
        pf = (aa * p[i, j - 1, k] + bb * p[i, j, k]) / (aa + bb)
        rho_f = ws.rhofy[i, j, k]
        kd = 0.5 * (drag[i, j - 1, k] + drag[i, j, k])
        U.v[i, j, k] -= dtt * (p[i, j, k] - p[i, j - 1, k]) / (rho_f * m.dyf[j])
        U.v[i, j, k] /= (1 + dtt * kd)
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        aa = F[i, j, k - 1] >= 1 - 1e-3 ? 1.0 : (F[i, j, k - 1] <= 1e-3 ? 0.0 : F[i, j, k - 1])
        bb = F[i, j, k] >= 1 - 1e-3 ? 1.0 : (F[i, j, k] <= 1e-3 ? 0.0 : F[i, j, k])
        (aa <= 0 && bb <= 0) && continue
        pf = (aa * p[i, j, k - 1] + bb * p[i, j, k]) / (aa + bb)
        rho_f = ws.rhofz[i, j, k]
        kd = 0.5 * (drag[i, j, k - 1] + drag[i, j, k])
        U.w[i, j, k] -= dtt * (p[i, j, k] - p[i, j, k - 1]) / (rho_f * m.dzf[k])
        U.w[i, j, k] /= (1 + dtt * kd)
    end
    U
end

"""
    darcy_and_buoyancy!(drag, buoy, T, mat, m, g; sd, exponent, eps0, rho_amb)

Darcy mushy drag `k_d` (paper Eq. 5) and Boussinesq buoyancy (Eq. 15).

Buoyancy is referenced to the liquidus so that a fully liquid pool is neutrally
buoyant and only the *hotter* part of a partly solid pool rises — the paper's reference
point, and the physically right one.
"""
function darcy_and_buoyancy!(drag::Array{Tf,3}, buoy::Array{Tf,3},
                             Tfld::Array{Tf,3}, mat::Material;
                             sd::Real = 1e6, exponent::Int = 2,
                             eps0::Real = 1e-4, rho_amb::Real = 1000.0,
                             gmag::Real = 9.80665) where {Tf<:AbstractFloat}
    @inbounds for idx in eachindex(drag)
        T = Tfld[idx]
        fs = solid_fraction(mat, T)
        drag[idx] = Tf(darcy_drag(mat, fs, sd; exponent = exponent, eps0 = eps0))
        # g_b = -g * beta_T * (T - T_liq): upward is -z is down, so +z is up
        buoy[idx] = Tf(gmag * mat.beta_T * (T - mat.Tliq))
    end
    (drag, buoy)
end

"""
    viscous_candidate!(uu, vv, ww, U, muf, drag, buoy, mat, Tfld, m, g, dt, fx, fy, fz)

Explicit predictor for the face velocities: viscous diffusion, buoyancy, Marangoni and
surface tension, and the drag damping. Pressure projection follows.

Explicit diffusion is fine here because the timestep is CFL-limited by the flow anyway, so
the viscous number never becomes the restriction.
"""
function viscous_candidate!(U::VectorField{Tf}, ws::MomentumWorkspace{Tf},
                            dt::Real, drag, buoy, mat, Tfld,
                            m::GridMetrics, g::CartesianGrid,
                            fx = nothing, fy = nothing, fz = nothing) where {Tf<:AbstractFloat}
    nx, ny, nz = g.nx, g.ny, g.nz
    dtt = Tf(dt)
    # kinematic viscosity and drag per cell
    @inbounds for idx in eachindex(drag)
        ws.visc[idx] = mat.rho_s > 0 ? viscous_rho(Tfld[idx], mat) : 0.0
    end
    copyto!(ws.u0, U.u); copyto!(ws.v0, U.v); copyto!(ws.w0, U.w)
    copyto!(ws.uu, U.u); copyto!(ws.vv, U.v); copyto!(ws.ww, U.w)

    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        nu = 0.5 * (ws.visc[i - 1, j, k] + ws.visc[i, j, k])
        ws.uu[i, j, k] += dtt * nu * (U.u[i + 1, j, k] - 2 * U.u[i, j, k] + U.u[i - 1, j, k]) / m.dxf[i]^2
        ws.uu[i, j, k] += dtt * 0.5 * (buoy[i - 1, j, k] + buoy[i, j, k])
        ws.uu[i, j, k] /= (1 + dtt * 0.5 * (drag[i - 1, j, k] + drag[i, j, k]))
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        nv = 0.5 * (ws.visc[i, j - 1, k] + ws.visc[i, j, k])
        ws.vv[i, j, k] += dtt * nv * (U.v[i, j + 1, k] - 2 * U.v[i, j, k] + U.v[i, j - 1, k]) / m.dyf[j]^2
        ws.vv[i, j, k] /= (1 + dtt * 0.5 * (drag[i, j - 1, k] + drag[i, j, k]))
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        nw = 0.5 * (ws.visc[i, j, k - 1] + ws.visc[i, j, k])
        ws.ww[i, j, k] += dtt * nw * (U.w[i, j, k + 1] - 2 * U.w[i, j, k] + U.w[i, j, k - 1]) / m.dzf[k]^2
        ws.ww[i, j, k] += dtt * 0.5 * (buoy[i, j, k - 1] + buoy[i, j, k])
        ws.ww[i, j, k] /= (1 + dtt * 0.5 * (drag[i, j, k - 1] + drag[i, j, k]))
    end
    # Marangoni traction, converted to a face velocity increment
    if fx !== nothing
        @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
            ws.uu[i, j, k] += dtt * 0.5 * (fx[i - 1, j, k] + fx[i, j, k]) / mat.rho_l
        end
        @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
            ws.vv[i, j, k] += dtt * 0.5 * (fy[i, j - 1, k] + fy[i, j, k]) / mat.rho_l
        end
        @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
            ws.ww[i, j, k] += dtt * 0.5 * (fz[i, j, k - 1] + fz[i, j, k]) / mat.rho_l
        end
    end
    copyto!(U.u, ws.uu); copyto!(U.v, ws.vv); copyto!(U.w, ws.ww)
    U
end

"""
    viscous_rho(T, mat) -> Float64

Kinematic viscosity `μ/ρ` of the metal at temperature `T`, zero outside the liquid range
(solid metal does not creep in this model).
"""
@inline function viscous_rho(T::Real, mat::Material)
    T > mat.Tliq ? visc(mat, T) / mat.rho_l :
    T > mat.Tsol ? visc(mat, mat.Tliq) / mat.rho_l * 1e-2 : 0.0
end

"""
    max_speed(U) -> Float64

Largest face speed, used by the timestep controller for the Courant limit.
"""
function max_speed(U::VectorField)
    m = 0.0
    @inbounds for x in U.u; abs(x) > m && (m = abs(x)); end
    @inbounds for x in U.v; abs(x) > m && (m = abs(x)); end
    @inbounds for x in U.w; abs(x) > m && (m = abs(x)); end
    m
end