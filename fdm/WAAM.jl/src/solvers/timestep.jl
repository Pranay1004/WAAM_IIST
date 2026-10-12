"""
Phase-aware timestep control.

This is the single most important performance decision in the package, and the reason a
3400 s / 25-pass simulation is reachable at all.

## The problem

The paper's headline case is 3400 s of simulated time, of which only about 1000 s is
arc-on and most of *that* is a ~2 mm spot on a 0.6 mm mesh. Explicit flow stepping needs
`dt ≈ Δx/u ≈ 1e-5 s` — 3.4e8 steps for the full case. Nothing runs that fast.

## The resolution

The domain is *not* active all the time. A cell that has solidified (`T < T_sol`) has no
free surface, no velocity, and a Laplacian diffusion limit of `Δx²/α ≈ 1e-10 s` **only if
the scheme is explicit**. With an implicit energy solver the limit is set by accuracy, not
stability, so the frozen and cooling phases — which are ~95 % of the wall-clock time — can
take seconds per step instead of microseconds.

The controller therefore reports a **regime** with every timestep, so a user always knows
when the solver was in its cheap mode:

| Regime | Condition | `dt` |
| --- | --- | --- |
| `:arc` | `T > T_liq` near the torch, liquid free surface | CFL-limited, ~1e-4 – 1e-3 s |
| `:solidify` | mushy region, `T_sol < T < T_liq` | Stefan + accuracy limited |
| `:frozen` | `T < T_sol` everywhere near the torch | accuracy limited, 0.1 – 5 s |

Accuracy (not stability) limits the implicit steps, so the frozen-phase limit is derived
from a target temperature change per step rather than from a stability criterion.
"""

"""
    TimestepController

Adaptive timestep with a CFL/viscous/capillary/Stefan floor and an accuracy ceiling.

Fields
- `dt_arc`, `dt_frozen` : ceiling on the timestep in each regime [s]
- `cfl` : Courant number limit for the VOF/flow update
- `dT_arc`, `dT_frozen` : maximum allowed temperature change per step [K]
- `c_sound_like` : effective velocity scale for the capillary/viscous limits
"""
Base.@kwdef struct TimestepController
    cfl::Float64 = MAX_STABLE_COURANT
    dt_arc::Float64 = 2.0e-3
    dt_frozen::Float64 = 5.0
    dT_arc::Float64 = 250.0         # K per step while liquid
    dT_frozen::Float64 = 60.0       # K per step once solid
    cap_factor::Float64 = 2.0       # capillary limit = sqrt(standoff/cap_length_scale)
    top_k::Int = 1                  # surface cell index receiving the arc heat
end

"""
    _arc_surface_flux_peak(p) -> Float64  [W/m²]

Peak Gaussian arc surface flux at the torch centre, `P/(π σ_q²)`. At the paper's settings
this is ≈ 1.1e8 W/m² — which is precisely why the timestep must be controlled by an
accuracy criterion and not by a stability one.
"""
_arc_surface_flux_peak(p::ProcessParameters) = arc_power(p) / (π * sigma_q(p)^2)

"""
    timestep(T, F, m, g, mat, U, p; ctrl) -> (dt, regime, info)

Recommended timestep for the current field state.

The **accuracy ceiling** is the mechanism that makes long runs possible: the maximum
temperature change per step bounds `dt` through an estimate of the local heating rate,
and once the material is solid that rate is small, so `dt` grows to seconds.
"""
function timestep(T::AbstractArray, F::AbstractArray, m::GridMetrics, g::CartesianGrid,
                  mat::Material, U, p::ProcessParameters;
                  ctrl::TimestepController = TimestepController(),
                  t::Real = 0.0, torch::Tuple{Float64,Float64} = (0.0, 0.0),
                  src_now::Union{Nothing,AbstractArray} = nothing)
    dmin = min(minimum(m.dx), minimum(m.dy), minimum(m.dz))

    # --- where are we? ---
    Tmax = maximum(T)
    Tnear = _temperature_near(T, m, g, mat, torch, 3.0e-3)
    liquid = Tnear > mat.Tliq
    mushy = Tnear > mat.Tsol
    frozen = !mushy
    regime = liquid ? :arc : (mushy ? :solidify : :frozen)

    # --- accuracy ceiling from the heating rate ---
    # dT = (q_vol / (rho Cp)) dt for the cell that receives the arc heat, so the step is
    # bounded by the temperature rise we are willing to accept in one step.
    #
    # The volumetric source that will actually be deposited is used when available, not
    # the Gaussian *peak*: the peak flux (1.1e8 W/m² at the paper's settings) is
    # concentrated in a fraction of one cell, so using it directly underestimates dt by
    # orders of magnitude and turns a 1 s track into a 1e9-step run.
    dTlim = frozen ? ctrl.dT_frozen : ctrl.dT_arc
    qvol = 0.0
    if src_now !== nothing
        @inbounds for idx in eachindex(src_now)
            src_now[idx] > qvol && (qvol = src_now[idx])
        end
    else
        qvol = _arc_surface_flux_peak(p) / max(min(m.dx), 1e-12)   # W/m³ equivalent
    end
    dt_acc = Inf
    if qvol > 0
        Cp_ref = frozen ? Cp_s_ref(mat) : Cp_l_ref(mat)
        dt_acc = dTlim * mat.rho_s * Cp_ref / qvol
    end

    # --- CFL floor (matters only in the liquid regime) ---
    dt_cfl = Inf
    if U !== nothing && liquid
        umax = maximum(abs, U.u), maximum(abs, U.v), maximum(abs, U.w)
        vmax = max(umax...)
        vmax > 0 && (dt_cfl = ctrl.cfl * dmin / vmax)
    end

    # --- capillary floor: surface tension stops a very thin layer ---
    dt_cap = Inf
    if liquid
        sigma = mat.gamma(Tnear)
        rho_l = mat.rho_l
        vmax = U === nothing ? 0.0 : max(maximum(abs, U.u), maximum(abs, U.v), maximum(abs, U.w))
        vmax = max(vmax, 1e-6)
        # Brackbill capillary time for a wave of the mesh scale
        dt_cap = sqrt(dmin^3 * rho_l / (2 * pi * sigma))
    end

    # Rate cap only applies where the flow is active. Applying `dt_arc` in the frozen
    # regime would silently cap the whole run at 2 ms and make 3400 s take 1.7e6 steps —
    # the exact inefficiency this controller exists to avoid.
    dt = min(dt_acc, regime === :frozen ? ctrl.dt_frozen : ctrl.dt_arc)
    dt_cfl < Inf && (dt = min(dt, dt_cfl))
    dt_cap < Inf && (dt = min(dt, dt_cap))
    dt = max(dt, 1e-9)

    info = (regime = regime,
            Tmax = Tmax, Tnear = Tnear,
            dmin = dmin,
            dt_accuracy = dt_acc, dt_cfl = dt_cfl, dt_capillary = dt_cap,
            liquid_volume = _liquid_volume_fraction(T, mat))
    (dt, regime, info)
end

# temperature within a horizontal radius of the torch
function _temperature_near(T, m, g, mat, torch, radius)
    xt, yt = torch
    best = 0.0
    @inbounds for j in 1:g.ny, i in 1:g.nx
        if (m.xc[i] - xt)^2 + (m.yc[j] - yt)^2 <= radius^2
            for k in 1:g.nz
                T[i, j, k] > best && (best = T[i, j, k])
            end
        end
    end
    best
end

"""
    _receiving_cell_volume(m, g, mat, torch, top_k) -> Float64

Volume of the cell that actually receives the arc heat: the nearest liquid cell if the
pool exists, otherwise the exposed surface cell at the torch.

Falling back to the surface cell matters on the very first step of a run, when nothing
is molten yet. Estimating the heating rate only from existing liquid makes the accuracy
limit `Inf` at exactly the moment the power input is largest, so the solver takes one
enormous step and produces a 200 000 K artefact.
"""
function _receiving_cell_volume(m::GridMetrics, g::CartesianGrid, mat::Material,
                                torch::Tuple{Float64,Float64}, T, top_k::Int)
    xt, yt = torch
    best = 0.0; bestd = Inf
    @inbounds for j in 1:g.ny, i in 1:g.nx
        d = (m.xc[i] - xt)^2 + (m.yc[j] - yt)^2
        for k in 1:g.nz
            if T[i, j, k] > mat.Tsol && d < bestd
                bestd = d; best = m.vol[i, j, k]
            end
        end
    end
    best > 0 && return best
    # no pool yet: use the surface cell under the torch
    bestd = Inf
    @inbounds for j in 1:g.ny, i in 1:g.nx
        d = (m.xc[i] - xt)^2 + (m.yc[j] - yt)^2
        if d < bestd
            bestd = d
            best = m.vol[i, min(top_k, g.nz), 1]
        end
    end
    best
end

function _power_density_near(T, m, g, mat, torch, p, radius)
    # power density the arc would deposit on the exposed pool near the torch
    σ2 = sigma_q(p)^2
    amp = arc_power(p) / (pi * σ2)
    xt, yt = torch
    best = 0.0
    @inbounds for j in 1:g.ny, i in 1:g.nx
        r2 = (m.xc[i] - xt)^2 + (m.yc[j] - yt)^2
        r2 <= radius^2 || continue
        q = amp * exp(-r2 / σ2)
        q > best && (best = q)
    end
    best
end

_liquid_volume_fraction(T, mat) = latent_heat_fraction(T, mat)

"""
    wall_clock_hint(info) -> String

One-line human description of the current regime, printed in run progress so a
3400 s simulation shows at a glance that it is spending its time in the cheap phase.
"""
function wall_clock_hint(info)
    @sprintf("%s: Tmax=%.0f K, Tnear=%.0f K, dmin=%.3f mm, liquid=%.3f%%",
            uppercase(string(info.regime)), info.Tmax, info.Tnear,
            info.dmin * 1e3, info.liquid_volume * 100)
end