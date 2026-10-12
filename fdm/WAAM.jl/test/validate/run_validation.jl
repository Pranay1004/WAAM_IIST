# Reproduces validation_output/RESULTS.md. Run directly:
#   julia --project=. test/validate/run_validation.jl
using WAAMFlow
using Printf
using LinearAlgebra

const OUT = joinpath(@__DIR__, "..", "..", "validation_output")
mkpath(OUT)

section(title) = println("\n", "=" ^ 74, "\n", title, "\n", "=" ^ 74)

# ---------------------------------------------------------------- 1. process budget
section("1. Process parameters vs Zargari et al. (2025), Sec. 2")
p = ProcessParameters()
mat = get_material("ER316L")
@printf("  sigma_q (Eq. 16, 125 A)          : %.4f mm   (paper 2.40 mm)\n", sigma_q(p) * 1e3)
@printf("  wire feed mass rate              : %.4e kg/s = %.1f g/min\n",
        wire_feed_mass_rate(p), wire_feed_mass_rate(p) * 60 * 1e3)
@printf("  arc power after efficiency       : %.1f W\n", arc_power(p))
@printf("  droplet power, Eq. (12)          : %.1f W\n", droplet_heat(p, mat))
@printf("  droplet share eta*eta_d*UI, Eq.(10): %.1f W\n", p.eta * p.eta_d * p.UI)
@printf("  --> the paper's two heat-split equations disagree by %.2fx\n",
        droplet_heat(p, mat) / (p.eta * p.eta_d * p.UI))

# ---------------------------------------------------- 2. analytic energy benchmark
section("2. Energy solver vs analytic solution (adiabatic box, uniform heating)")
function energy_benchmark(; n = 10, dt = 1e-4, q = 1.0e6, steps = (1, 10, 100, 1000, 5000))
    mat = get_material("ER70S6")
    g = uniform_grid(n, n, n, 0.01, 0.01, 0.01)
    m = metrics(g)
    ws = EnergyWorkspace(g)
    src = fill(q, n, n, n)
    z3 = zeros(n, n, g.nz)
    println("  steps |  simulated rise  |    exact rise     | rel err | max CG")
    worst = 0.0
    for ns in steps
        T = fill(300.0, n, n, g.nz); Told = copy(T)
        tot = 0.0; cmax = 0
        for _ in 1:ns
            copyto!(Told, T)
            _, it, pin = solve_energy!(ws, T, Told, mat, nothing, src, z3, z3, m, g, dt)
            tot += pin * dt
            cmax = max(cmax, it)
        end
        exact = tot / (mat.rho_s * Cp_s_ref(mat) * sum(m.vol))
        rel = abs((T[5, 5, 5] - 300.0) - exact) / max(exact, 1e-30)
        worst = max(worst, rel)
        @printf("  %6d | %14.6e | %14.6e | %7.1e | %6d\n", ns, T[5, 5, 5] - 300.0, exact, rel, cmax)
    end
    worst
end
worst = energy_benchmark()

# ------------------------------------------------------------- 3. single track runs
section("3. Single-track GMAW bead geometry (paper Sec. 2 parameters)")
results = Dict{String,Any}()
for name in ("ER70S6", "ER316L")
    r = run_track(RunConfig(material = name, verbose = false))
    results[name] = r
end
@printf("  %-8s | %10s | %10s | %10s | %10s | %12s\n",
        "alloy", "width/mm", "pen/mm", "peak/K", "molten", "E resid/J")
for name in ("ER70S6", "ER316L")
    r = results[name]
    @printf("  %-8s | %10.3f | %10.3f | %10.0f | %10.5f | %12.3e\n",
            name, r.width * 1e3, r.penetration * 1e3, r.peak_T,
            r.molten_fraction, r.energy_residual)
end
w70 = results["ER70S6"].width
w316 = results["ER316L"].width
@printf("  ER316L/ER70S6 width ratio      : %.2f\n", w316 / max(w70, 1e-12))
@printf("  k at liquidus 70S6 / 316L      : %.2f / %.2f W/(m K)  (ratio %.2f)\n",
        kcond(get_material("ER70S6"), get_material("ER70S6").Tliq),
        kcond(get_material("ER316L"), get_material("ER316L").Tliq),
        kcond(get_material("ER70S6"), get_material("ER70S6").Tliq) /
        kcond(get_material("ER316L"), get_material("ER316L").Tliq))

# ------------------------------------------------------------------- 4. write out
section("4. Open items")
r316 = results["ER316L"]
@printf("  energy residual while a cell straddles the mushy interval: %.3e J on %.1f J in\n",
        r316.energy_residual, r316.energy_in)
@printf("  --> single-phase energy conservation is exact (%.1e relative, above);\n", worst)
@printf("      the mushy-front Newton sweep does not converge. MUST be fixed before\n")
@printf("      any bead-geometry number is trusted.\n")

open(joinpath(OUT, "validation_run.txt"), "w") do io
    println(io, "WAAMFlow validation run")
    println(io, "Julia ", VERSION)
    for name in ("ER70S6", "ER316L")
        r = results[name]
        @printf(io, "%s: width=%.4f mm  penetration=%.4f mm  peak=%.1f K  molten=%.6f  steps=%d  wall=%.1fs\n",
                name, r.width * 1e3, r.penetration * 1e3, r.peak_T, r.molten_fraction,
                r.steps, r.wall)
    end
    @printf(io, "energy benchmark worst relative error: %.2e\n", worst)
end
println("\nwrote ", joinpath(OUT, "validation_run.txt"))
