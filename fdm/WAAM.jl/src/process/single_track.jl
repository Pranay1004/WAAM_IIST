"""
Single-track GMAW driver.

Runs a moving Gaussian arc over a cold substrate with the paper's P-GMAW parameters,
reporting bead geometry, thermal history and the energy audit.

This is the tier-1 validation case P2/P3 of the master plan.
"""

"""
    RunConfig

Everything a single-track case needs. Defaults follow the paper's Sec. 2.
"""
Base.@kwdef struct RunConfig
    material::String = "ER70S6"
    Lx::Float64 = 0.020          # m
    Ly::Float64 = 0.004          # m (half-symmetry width)
    Lz::Float64 = 0.008          # m
    h_fine::Float64 = 0.4e-3     # m
    h_coarse::Float64 = 1.6e-3   # m
    ny::Int = 10
    arc_power_W::Float64 = 1472.0
    travel_speed::Float64 = 0.0028333   # m/s  (0.17 m/min)
    track_length::Float64 = 0.008       # m
    substrate_T::Float64 = 298.0        # K
    energy_mode::Symbol = :consistent
    verbose::Bool = true
end

"""
    SingleTrackResult

Outputs plus the audit numbers needed to judge the run.
"""
struct SingleTrackResult
    T::Array{Float64,3}
    width::Float64
    height::Float64
    penetration::Float64
    peak_T::Float64
    molten_fraction::Float64
    energy_in::Float64
    energy_residual::Float64
    steps::Int
    wall::Float64
    sim_time::Float64
    track_length::Float64
    ncell::Int
end

"""
    run_track(cfg) -> SingleTrackResult

Integrate the thermal field along a single pass. Conduction + enthalpy phase change only —
the fluid stack (VOF/Marangoni) is layered on top in `simulate.jl`; this case exists to
validate the thermal and geometric kernel first, which is the tier-1 gate.
"""
function run_track(cfg::RunConfig = RunConfig())
    mat = get_material(cfg.material)
    proc = ProcessParameters()

    xf = graded_axis(0.0, cfg.Lx, cfg.h_fine, cfg.h_coarse; h_fine_end = :first)
    zf = graded_axis(0.0, cfg.Lz, cfg.h_fine, cfg.h_coarse; h_fine_end = :first)
    yf = collect(range(0.0, cfg.Ly; length = cfg.ny))
    g = CartesianGrid(xf, yf, zf)
    m = metrics(g)
    ws = EnergyWorkspace(g)

    nx, ny, nz = g.nx, g.ny, g.nz
    T = fill(cfg.substrate_T, nx, ny, nz)
    Told = copy(T)
    z3 = zeros(nx, ny, nz)
    qb = zeros(nx, ny, nz); sb = zeros(nx, ny, nz); lb = zeros(nx, ny, nz)

    es = EnergySplit(proc, mat; mode = cfg.energy_mode)
    ctrl = TimestepController()

    x_start = cfg.h_fine * 2
    x_torch = x_start
    t = 0.0
    steps = 0
    peak = cfg.substrate_T
    e_in = 0.0
    e_res = 0.0
    max_cg = 0
    t0 = time()

    while (x_torch - x_start) < cfg.track_length
        torch = (x_torch, yf[1])
        arc_flux!(qb, T, proc, mat, m, g, torch; q_surface_power = cfg.arc_power_W)
        arc_source!(sb, qb, T, mat, m, g; top_k = 1)
        surface_loss!(lb, T, proc, mat, m, g)

        dt, regime, info = timestep(T, fill(1.0, nx, ny, nz), m, g, mat, nothing, proc;
                                    ctrl = ctrl, t = t, torch = torch, src_now = sb)
        dt = clamp(dt, 1e-9, 0.05)

        copyto!(Told, T)
        res, iters, pin = solve_energy!(ws, T, Told, mat, nothing, sb, z3, lb, m, g, dt)
        e_res += energy_residual(T, ws.hout, Told, mat, nothing, sb, z3, lb, m, g, dt) * dt
        e_in += (abs(arc_power_delivered(sb, m)) + abs(sum(lb .* m.vol))) * dt

        t += dt
        steps += 1
        peak = max(peak, maximum(T))
        max_cg = max(max_cg, iters)
        x_torch += cfg.travel_speed * dt

        if cfg.verbose && (steps % 500 == 1)
            @printf("    t=%6.3f s  dt=%8.2e  [%s]  Tmax=%6.0f K  Qarc=%6.1f W  res=%.2e\n",
                    t, dt, uppercase(string(regime)), maximum(T),
                    arc_power_delivered(sb, m), res)
        end
        x_torch > xf[end] - cfg.h_fine && break
    end
    wall = time() - t0

    bg = bead_geometry(T, mat, m, g)
    if cfg.verbose
        @printf("\n  %s single track — %d x %d x %d = %d cells\n",
                cfg.material, nx, ny, nz, length(T))
        @printf("  track deposited   : %.2f mm in %.3f s  (%d steps, %.1f s wall, %.0fx realtime)\n",
                (x_torch - x_start) * 1e3, t, steps, wall, t / max(wall, 1e-9))
        @printf("  peak temperature   : %.0f K\n", peak)
        @printf("  bead width         : %.2f mm\n", bg.width * 1e3)
        @printf("  bead height        : %.2f mm\n", bg.height * 1e3)
        @printf("  penetration depth  : %.3f mm\n", bg.penetration * 1e3)
        @printf("  molten fraction    : %.5f\n", latent_heat_fraction(T, mat))
        @printf("  energy in          : %.1f J\n", e_in)
        @printf("  energy residual    : %.3e J  (%.2e relative)\n", e_res, e_res / max(e_in, 1e-30))
        @printf("  max CG iterations  : %d\n", max_cg)
        @printf("  energy split       : surface %.0f W, droplets %.0f W (mode %s, closure %.1e)\n",
                es.q_surface, es.q_droplet, es.mode, es.closure)
    end

    SingleTrackResult(T, bg.width, bg.height, bg.penetration, peak,
                      latent_heat_fraction(T, mat), e_in, e_res, steps, wall, t,
                      x_torch - x_start, length(T))
end

