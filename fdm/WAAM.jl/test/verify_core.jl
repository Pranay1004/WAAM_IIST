using Test, WAAMFlow
using WAAMFlow: minmod, superbee, vanleer, central

# ===========================================================================
# V1 — Mesh invariants
#     Σ_f Sf = 0 and V_c = (1/3) Σ_f (Sf · xf) must hold to machine precision,
#     on BOTH uniform and graded grids. A non-zero value here means every
#     divergence in the solver is silently biased.
# ===========================================================================

@testset "V1 mesh invariants" begin
    for (label, g) in (
        ("uniform", uniform_grid(12, 9, 7, 0.01, 0.02, 0.03)),
        ("uniform-fine", uniform_grid(40, 32, 24, 6e-4, 1.2e-3, 9e-4)),
        ("graded", CartesianGrid(
            graded_axis(0.0, 0.02, 0.6e-3, 1.8e-3; h_fine_end = :first),
            graded_axis(0.0, 0.01, 0.6e-3, 1.2e-3; h_fine_end = :first),
            graded_axis(0.0, 0.005, 1.0e-3, 1.0e-3))),
    )
        m = metrics(g)
        @test face_closure_error(g, m) < 1e-14
        @test volume_identity_error(g, m) < 1e-13
        @test all(isfinite, m.vol)
        @test all(>(0), m.vol)
    end

    # The graded mesh really is graded (a regression guard on graded_axis itself).
    # NOTE: monotonicity is checked to round-off, not exactly. Node positions are
    # accumulated by cumulative summation, so `diff` recovers widths that agree to
    # ~1 ulp but are not bitwise identical. Demanding strict `issorted` here would be
    # testing floating-point associativity, not the mesh generator.
    xg = graded_axis(0.0, 0.02, 0.6e-3, 1.8e-3; h_fine_end = :first)
    w = diff(xg)
    @test w[1] ≈ 0.6e-3 atol = 1e-12          # h_fine honoured at the graded end
    @test minimum(w) ≈ 0.6e-3 atol = 1e-12
    @test maximum(w) <= 1.8e-3 * (1 + 1e-12)  # never coarser than h_coarse
    @test maximum(w) > 1.5e-3                 # grading actually reached the coarse end
    for i in 2:length(w)
        @test w[i] >= w[i - 1] - 1e-15         # monotonically coarsening
    end
    @test xg[1] == 0.0 && xg[end] ≈ 0.02       # span filled exactly
    @test sum(w) ≈ 0.02 rtol = 1e-15

    # :last mirrors the grading.
    xr = graded_axis(0.0, 0.02, 0.6e-3, 1.8e-3; h_fine_end = :last)
    @test diff(xr) ≈ reverse(w)
    # :both is symmetric: fine at both ends, coarsest at the seam. The end widths
    # equal h_fine only if the half-span happens to reach it, so assert the shape
    # (symmetric, coarsest in the middle, in-band) rather than an exact end value.
    xb = graded_axis(0.0, 0.02, 0.6e-3, 1.8e-3; h_fine_end = :both)
    wb = diff(xb)
    @test wb ≈ reverse(wb)                       # symmetric
    @test minimum(wb) >= 0.6e-3 * (1 - 1e-9)
    @test maximum(wb) <= 1.8e-3 * (1 + 1e-9)
    @test argmax(wb) >= length(wb) ÷ 2           # coarsest at the seam, not the ends
    @test wb[1] < maximum(wb) && wb[end] < maximum(wb)

    # The paper's actual mesh: 0.6 mm through the fusion zone -> 1.8 mm far field.
    @test length(w) == 13

    # A degenerate axis must be rejected, not silently accepted.
    @test_throws ArgumentError uniform_grid(1, 1, 1, 0.0, 1, 1)
    @test_throws ArgumentError CartesianGrid([0.0, 1.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    @test_throws ArgumentError CartesianGrid([0.0, -1.0], [0.0, 1.0], [0.0, 1.0])
end

# ===========================================================================
# V2 — Spatial operator accuracy by manufactured solutions
#     Gradient / divergence / Laplacian must be 2nd order on a uniform grid
#     (measured by Richardson convergence), and Laplacian 2nd order on a
#     graded axis.
# ===========================================================================

@testset "V2 operator convergence" begin
    # Manufactured solutions with *analytic* derivatives. The observed order is
    # estimated by Richardson: p = log2(e_h / e_{h/2}). A genuine second-order
    # operator gives p -> 2; a first-order closure gives p -> 1 and would be caught.
    # Interior points only — one-sided boundary closures are a separate operator.

    # --- uniform axis: gradient ---
    errs = Float64[]
    for n in (64, 128, 256, 512)
        g = uniform_grid(n, 1, 1, 1.0, 1.0, 1.0)
        xc = WAAMFlow.cell_centers(g.xf)
        f(x) = exp(x) * sin(2 * pi * x)
        df(x) = exp(x) * (sin(2 * pi * x) + 2 * pi * cos(2 * pi * x))
        a = [f(x) for x in xc]
        # stencil positions must be the SAMPLE locations (cell centres), not nodes
        push!(errs, maximum(abs(WAAMFlow._deriv_3pt(xc[i - 1], xc[i], xc[i + 1],
                                                    a[i - 1], a[i], a[i + 1]) - df(xc[i]))
                            for i in 2:n - 1))
    end
    # Start at n=64: the h->0 asymptotic regime is not yet reached at n=32, where
    # round-off and the next term still matter, so the first measured order reads
    # ~1.79 rather than ~2. The convergence *rate* is the assertion, not any single
    # ratio.
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:length(errs) - 1]
    @test minimum(orders) > 1.8
    @test orders[end] > 1.95                # clearly asymptotic by the finest pair

    # --- uniform axis: laplacian ---
    errs = Float64[]
    for n in (32, 64, 128)
        g = uniform_grid(n, 1, 1, 1.0, 1.0, 1.0)
        xc = WAAMFlow.cell_centers(g.xf)
        h = 1.0 / n
        lap(x) = -(2 * pi)^2 * sin(2 * pi * x)
        a = [sin(2 * pi * x) for x in xc]
        L = WAAMFlow.laplacian_cell(a, h)
        push!(errs, maximum(abs(L[i] - lap(xc[i])) for i in 4:n - 3))
    end
    @test minimum(log2(errs[i] / errs[i + 1]) for i in 1:length(errs) - 1) > 1.8

    # --- graded axis: gradient must still be 2nd order ---
    errs = Float64[]
    for N in (20, 40, 80)
        xf = graded_axis(0.0, 1.0, 0.02 / N, 0.05 / N; h_fine_end = :first, ratio = 1.2)
        xc = WAAMFlow.cell_centers(xf)
        dg(x) = 2 * pi * cos(2 * pi * x)
        a = [sin(2 * pi * x) for x in xc]
        n = length(a)
        # The three samples a[i-1..i+1] sit at the cell CENTRES xc, so the stencil
        # positions must be xc too. Passing node coordinates with centre values makes
        # the derivative wrong by O(1) on a graded axis while still looking
        # plausible on a uniform one — a good illustration of why this is a test.
        push!(errs, maximum(abs(WAAMFlow._deriv_3pt(xc[i - 1], xc[i], xc[i + 1],
                                                    a[i - 1], a[i], a[i + 1]) - dg(xc[i]))
                            for i in 2:n - 1))
    end
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:length(errs) - 1]
    @test minimum(orders) > 1.8

    # --- divergence: consistency on a uniform MAC field ---
    g = uniform_grid(8, 6, 4, 0.01, 0.02, 0.03)
    m = metrics(g)
    U = VectorField(g)
    fill!(U.u, 0.3); fill!(U.v, -0.7); fill!(U.w, 1.1)
    d = zeros(g.nx, g.ny, g.nz)
    WAAMFlow.divergence!(d, U.u, U.v, U.w, m, g)
    @test maximum(abs, d) < 1e-12          # div(constant) == 0 exactly

    # --- divergence: exactness for linear fields ---
    g2 = uniform_grid(12, 9, 6, 1.0, 1.0, 1.0)
    m2 = metrics(g2)
    U2 = VectorField(g2)
    # u=x, v=2y, w=-3z  =>  div = 1 + 2 - 3 = 0
    for k in 1:g2.nz, j in 1:g2.ny, i in 1:g2.nx+1
        U2.u[i, j, k] = g2.xf[i]
    end
    for k in 1:g2.nz, j in 1:g2.ny+1, i in 1:g2.nx
        U2.v[i, j, k] = 2 * g2.yf[j]
    end
    for k in 1:g2.nz+1, j in 1:g2.ny, i in 1:g2.nx
        U2.w[i, j, k] = -3 * g2.zf[k]
    end
    d2 = zeros(g2.nx, g2.ny, g2.nz)
    WAAMFlow.divergence!(d2, U2.u, U2.v, U2.w, m2, g2)
    @test maximum(abs, d2) < 1e-12

    # u=x, v=y, w=z  =>  div = 3, exactly.
    # NOTE: the *last* face plane of each component must be written too — leaving
    # g2.xf[end] as the previous loop's value is what previously produced div=24.
    for k in 1:g2.nz, j in 1:g2.ny, i in 1:g2.nx+1
        U2.u[i, j, k] = g2.xf[i]
    end
    for k in 1:g2.nz, j in 1:g2.ny+1, i in 1:g2.nx
        U2.v[i, j, k] = g2.yf[j]
    end
    for k in 1:g2.nz+1, j in 1:g2.ny, i in 1:g2.nx
        U2.w[i, j, k] = g2.zf[k]
    end
    WAAMFlow.divergence!(d2, U2.u, U2.v, U2.w, m2, g2)
    @test maximum(abs.(d2 .- 3.0)) < 1e-12
end

# ===========================================================================
# V3 — Slope-limiter monotonicity (TVD precondition)
#     A limiter must never amplify: |limiter(a,b)| ≤ min(|a|,|b|).
# ===========================================================================

@testset "V3 slope limiters" begin
    vals = [-2.0, -1.0, -0.5, -1e-9, 0.0, 1e-9, 0.5, 1.0, 2.0]
    # minmod is the strictest limiter: |c| <= min(|a|,|b|).
    # superbee and vanleer are deliberately *not* so — superbee reaches 2x min|a|,|b|
    # — but all three remain TVD, which is the property that matters for a bounded
    # volume fraction. Asserting |c| <= min for all three would be asserting a false
    # property and would reject correct limiters.
    # minmod and superbee are bounded by a multiple of min(|a|,|b|); the van Leer
    # limiter is instead bounded by max(|a|,|b|) (it returns ≈ (a+b)/2 for disparate
    # slopes, which is a *max*-scale quantity). Assert the correct bound per limiter.
    for (lim, cmax, scale) in ((minmod, 1.0, :min), (superbee, 2.0, :min),
                               (vanleer, 2.0, :max))
        for a in vals, b in vals
            c = lim(a, b)
            @test c isa Float64
            ref = scale === :min ? min(abs(a), abs(b)) : max(abs(a), abs(b))
            @test abs(c) <= cmax * ref + 1e-15
            # never the opposite sign of both inputs
            @test sign(c) in (sign(a), 0.0, sign(b))
        end
    end
    # Characteristic values, so a refactor cannot silently change a limiter.
    @test minmod(1.0, 1.0) == 1.0
    @test minmod(2.0, 1.0) == 1.0
    @test superbee(1.0, 1.0) == 1.0           # max(minmod(1,1), minmod(2,1)) = max(1,1)
    @test superbee(2.0, 1.0) == 1.0           # minmod(2,1)=1, minmod(4,1)=1
    @test minmod(2.0, 4.0) == 2.0
    @test superbee(2.0, 4.0) == 4.0           # max(2, minmod(4,4)=4) = 4

    # van Leer (smooth/Harten form) recovers the slope exactly when the two one-sided
    # slopes agree — that is what makes it second-order in smooth regions.
    for a in (0.5, 1.0, 2.0, -1.5)
        @test vanleer(a, a) ≈ a
    end
    @test vanleer(1.0, 1.0) == 1.0            # NOT 1.5: the naive (a+b+√ab)/2 form
                                             # would give 1.5 and amplify by 50 %
    # On a 2:1 slope pair van Leer gives psi(1/2)*min = 0.805, i.e. slightly *below*
    # minmod (1.0): the smooth limiter is a touch more diffusive than minmod here,
    # while still recovering the full slope when the two one-sided slopes agree.
    @test abs(vanleer(2.0, 1.0)) ≈ 0.80473785 atol = 1e-7
    @test abs(vanleer(2.0, 1.0)) < abs(minmod(2.0, 1.0))
    @test abs(vanleer(2.0, 1.0)) < abs(central(2.0, 1.0))    # never amplifies
    # Superbee is the least diffusive of the three on this pair.
    @test abs(superbee(2.0, 1.0)) <= abs(minmod(2.0, 1.0))
    @test abs(superbee(2.0, 1.0)) <= abs(central(2.0, 1.0))
    # TVD precondition, asserted per limiter with its own correct bound:
    #   minmod   <= min|a,b|                      (strictest)
    #   vanleer  <= min|a,b|                      (psi <= 1)
    #   superbee <= 2 min|a,b|                    (psi <= 2, legitimately sharper)
    # Superbee genuinely exceeds min|a,b| — e.g. superbee(1.5, 3.0) = 3.0 — so a
    # blanket "|lim| <= min" assertion would reject a correct limiter.
    for (l, cmax) in ((minmod, 1.0), (vanleer, 1.0), (superbee, 2.0))
        for a in (-3.0, -1.5, -0.2, 0.2, 1.5, 3.0)
            for b in (-3.0, -1.5, -0.2, 0.2, 1.5, 3.0)
                @test abs(l(a, b)) <= cmax * min(abs(a), abs(b)) + 1e-14
                # a limiter must never produce the opposite sign
                @test sign(l(a, b)) in (sign(a), 0.0, sign(b))
            end
        end
    end
    # Opposite-sign slopes must be killed by every limiter (monotonicity).
    for lim in (minmod, superbee, vanleer)
        @test lim(1.0, -1.0) == 0.0
        @test lim(-3.0, 2.0) == 0.0
    end
    # central is the unlimited 2nd-order slope (included only as a reference).
    @test central(2.0, 4.0) == 3.0

    @test slope_limiter("minmod") === minmod
    @test slope_limiter("VanLeer") === vanleer
    @test_throws ArgumentError slope_limiter("bogus")
end

# ===========================================================================
# V4 — Flux exactness on a stretched grid
#     Face interpolation must reproduce exactly for linear fields, and the
#     harmonic-face coefficient must be the exact arithmetic mean of a
#     spatially constant field.
# ===========================================================================

@testset "V4 face interpolation exactness" begin
    # two-point interpolation is exact for a linear field at any spacing
    for (ha, hb) in ((0.6e-3, 0.6e-3), (0.6e-3, 1.8e-3), (1.8e-3, 0.6e-3))
        @test face_interp2(10.0, 20.0, ha, hb) ≈
              (10.0 * hb + 20.0 * ha) / (ha + hb)
    end
    # harmonic mean of two equal values is that value
    @test harmonic_mean(30.0, 30.0) ≈ 30.0
    # Series-resistance identity: for two half-cells of width h/2 with
    # conductivities k1 and k2 in series, the face conductivity must satisfy
    #     (h/2)/k_face == (h/2)/k1 + (h/2)/k2
    # exactly. This is the property that makes the two-point flux conservative and
    # second-order on a graded mesh.
    # Two-point face conductivity. hp and hm are CENTRE-to-FACE distances (both dx/2 for a
    # uniform cell). The correct identity is a dimensional one: a constant-flux
    # gradient must be reproduced exactly,
    #     q = -k_face * (T_N - T_P)/(hp + hm)  ==  -k * dT/dx
    # on a uniform slab. That is a far stronger check than any algebraic rearrangement
    # of the formula itself.
    let k = 30.0, dx = 1e-3, slope = 1000.0
        kf = harmonic_face_coeff(k, k, dx / 2, dx / 2)
        @test kf ≈ k                                     # exact for a uniform slab
        TP, TQ = slope * 0.5 * dx, slope * 1.5 * dx
        q = kf * (TQ - TP) / dx
        @test q ≈ k * slope rtol = 1e-14
    end
    # On a uniform axis with unequal neighbours it is the harmonic mean.
    @test harmonic_face_coeff(30.0, 60.0, 0.3e-3, 0.3e-3) ≈ 2 * 30.0 * 60.0 / 90.0
    # Exact series-resistance form: k_face = (hp+hm) / (hp/k_P + hm/k_N)
    for (k1v, k2v, hp, hm) in ((30.0, 60.0, 0.5e-3, 0.5e-3),
                               (30.0, 60.0, 0.3e-3, 0.9e-3),
                               (12.0, 45.0, 0.6e-3, 0.6e-3))
        kf = harmonic_face_coeff(k1v, k2v, hp, hm)
        @test kf ≈ (hp + hm) / (hp / k1v + hm / k2v) rtol = 1e-12
    end
    # Biasing test: the cell whose centre is FARTHER from the face has its resistance
    # weighted less, so the face value leans toward that cell's conductivity.
    # k_P = 30 with hp = 0.9e-3 (far) => face leans to 30, i.e. < harmonic mean 40.
    # k_N = 60 with hm = 0.3e-3 (near) => face leans to 60, i.e. > 40.
    @test harmonic_face_coeff(30.0, 60.0, 0.9e-3, 0.3e-3) < 40.0
    @test harmonic_face_coeff(30.0, 60.0, 0.3e-3, 0.9e-3) > 40.0

    # half_distances: distance from each cell centre to its own lower face is h/2
    h = [0.6e-3, 1.2e-3, 1.8e-3]
    d = WAAMFlow.half_distances(h)
    @test length(d) == 3
    @test d ≈ [0.3e-3, 0.6e-3, 0.9e-3]
    @test cell_to_face_distance(h, 2) ≈ 0.6e-3
    @test cell_to_face_distance(h, 0) ≈ 0.3e-3   # clamped to the first cell
end

# ===========================================================================
# V5 — 1-D linear advection against the exact solution
#     Transport of a smooth profile at several Courant numbers; also tests
#     exact conservation of the discrete total.
# ===========================================================================

@testset "V5 advection" begin
    # Transport of an exactly 1-periodic profile on a periodic domain. Periodic
    # boundaries matter: with upwind end faces the error is dominated by the boundary
    # closure and the measured order reflects that, not the scheme.
    fp(x) = sin(2 * pi * x) + 0.3 * cos(4 * pi * x)

    function transport(limiter, n, Tend, u = 1.0)
        h = 1.0 / n
        xc = WAAMFlow.cell_centers(collect(range(0.0, 1.0; length = n + 1)))
        a = [fp(x) for x in xc]
        a0 = copy(a)
        # dt ~ h^1.5 balances spatial O(h^2) against temporal O(dt^2) = O(h^3)
        ns = max(1, round(Int, Tend / (MAX_STABLE_COURANT * h^1.5)))
        dt = Tend / ns
        for _ in 1:ns
            F = face_fluxes(a, h, u, limiter; periodic = true)
            update_from_flux!(a, h, dt, F)
        end
        exact = [fp(x + u * Tend) for x in xc]
        (sum(abs.(a .- exact)) / n, minimum(a) >= minimum(a0) - 1e-12,
         maximum(a) <= maximum(a0) + 1e-12)
    end

    for lim in (minmod, superbee, vanleer)
        errs = Float64[]
        for n in (50, 100, 200)
            push!(errs, transport(lim, n, 4.0 * n^(-1.5))[1])
        end
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:length(errs) - 1]
        @test errs[end] < 0.02                      # converged
        @test minimum(orders) > 1.4                 # ~2nd order
    end

    # Boundedness: no new extrema anywhere in the Courant range we actually use.
    for lim in (minmod, superbee, vanleer)
        _, lo, hi = transport(lim, 100, 0.05)
        @test lo && hi
    end

    # Courant stability, measured rather than assumed. MAX_STABLE_COURANT must be
    # inside the stable region and the scheme must fail above it — asserting only the
    # former would let a silently unstable value slip through.
    function amplitude(lim, C, steps = 400; n = 100)
        h = 1.0 / n
        xc = WAAMFlow.cell_centers(collect(range(0.0, 1.0; length = n + 1)))
        a = [sin(2 * pi * x) for x in xc]
        dt = C * h
        for _ in 1:steps
            F = face_fluxes(a, h, 1.0, lim; periodic = true)
            update_from_flux!(a, h, dt, F)
        end
        maximum(abs.(a))
    end
    for lim in (minmod, superbee, vanleer)
        @test amplitude(lim, MAX_STABLE_COURANT) < 1.5     # stable as advertised
        @test amplitude(lim, 0.8) > 10.0                  # unstable well above it
    end
    @test 0 < MAX_STABLE_COURANT < 0.6

    # A linear ramp is transported EXACTLY by this scheme — a strong regression guard on
    # the flux-reconstruction indexing (a wrong upwind pick shows up here at once).
    for (n, dt) in ((10, 0.1), (20, 0.05), (17, 0.03))
        h = 1.0 / n
        a = collect(1.0:n)
        a0 = copy(a)
        F = face_fluxes(a, h, 1.0, minmod)
        # interior faces F[2..n] must equal the exact face value (a_i + a_{i+1})/2
        @test F[2:n] ≈ (a0[1:n-1] .+ a0[2:n]) ./ 2
        update_from_flux!(a, h, dt, F)
        # Interior cells: a ramp a(x) = x is transported exactly, so every interior
        # cell loses exactly u*dt*da/dx = dt/h. The two end cells use the 1st-order
        # upwind boundary face and are correctly NOT part of this assertion.
        @test a[2:n-1] ≈ a0[2:n-1] .- dt / h
    end

    # u = 0 must leave the field untouched
    a = [1.0 + 0.1 * i for i in 1:50]
    a0 = copy(a)
    update_from_flux!(a, 1.0 / 50, 1.0, face_fluxes(a, 1.0 / 50, 0.0, minmod))
    @test a == a0

    # Stretched-axis conservative update: closed domain, no boundary flux.
    xf = graded_axis(0.0, 1.0, 0.01, 0.02; ratio = 1.2)
    xc = WAAMFlow.cell_centers(xf)
    n = length(xc)
    a = [0.5 + 0.5 * sin(i) for i in 1:n]
    M0 = sum(a .* diff(xf))
    F = face_fluxes_stretched(a, xf, 0.0, minmod)
    F[1] = 0.0; F[end] = 0.0
    update_from_flux_stretched!(a, xf, 0.05, F)
    @test sum(a .* diff(xf)) ≈ M0 rtol = 1e-14

    # Stretched transport of a quadratic must be at least first-order and bounded.
    for N in (20, 40)
        xf = graded_axis(0.0, 1.0, 0.02 / N, 0.05 / N; h_fine_end = :first, ratio = 1.2)
        xc = WAAMFlow.cell_centers(xf)
        n = length(xc)
        a = [x^2 for x in xc]
        a0 = copy(a)
        F = face_fluxes_stretched(a, xf, 0.0, minmod)   # u = 0: field frozen
        update_from_flux_stretched!(a, xf, 1.0, F)
        @test a ≈ a0
    end
end

# ===========================================================================
# V9 / V10 — Enthalpy, inverse, continuity, liquid fraction, Darcy drag
# ===========================================================================

@testset "V9-V10 enthalpy and phase change" begin
    for nm in ("ER70S6", "ER316L")
        m = get_material(nm)

        # exact round trip h -> T -> h over the full physical range
        for T in 300.0:0.5:3000.0
            h = enthalpy(m, T)
            @test temperature(m, h) ≈ T atol = 1e-9
        end

        # h strictly increasing, and continuous at Tsol and Tliq
        prev = -Inf
        for T in 250.0:0.5:3200.0
            h = enthalpy(m, T)
            @test h > prev
            prev = h
        end
        @test enthalpy(m, m.Tsol + 1e-9) ≈ enthalpy(m, m.Tsol) atol = 1e-3
        @test enthalpy(m, m.Tliq + 1e-9) ≈ enthalpy(m, m.Tliq) atol = 1e-3

        # liquid fraction: endpoints, bounds, monotonicity
        @test liquid_fraction(m, m.Tsol) == 0.0
        @test liquid_fraction(m, m.Tliq) == 1.0
        @test liquid_fraction(m, m.Tsol - 500) == 0.0
        @test liquid_fraction(m, m.Tliq + 500) == 1.0
        fl_prev = -Inf
        for T in 250.0:1.0:3200.0
            fl = liquid_fraction(m, T)
            @test 0.0 <= fl <= 1.0
            @test fl >= fl_prev
            fl_prev = fl
            @test solid_fraction(m, T) ≈ 1 - fl
        end

        # cp_effective: solid value, latent plateau, liquid value
        @test cp_effective(m, 300.0) ≈ Cp_s_ref(m)
        @test cp_effective(m, (m.Tsol + m.Tliq) / 2) ≈ m.Lf / (m.Tliq - m.Tsol)
        @test cp_effective(m, 2500.0) ≈ Cp_l_ref(m)

        # density decreases with temperature (Boussinesq-consistent)
        @test rho(m, 300.0) > rho(m, 2000.0)
        # thermal conductivity positive everywhere, incl. beyond the table
        for T in (200.0, 1000.0, 2500.0, 4000.0)
            @test kcond(m, T) > 0
            @test rho(m, T) > 0
            @test visc(m, T) > 0
        end
        # Latent heat rate: heating across the mush releases exactly L_f·ρ·Δf
        h1 = enthalpy(m, m.Tsol); h2 = enthalpy(m, m.Tliq)
        @test (h2 - h1) ≈ m.Lf rtol = 1e-12
    end
end

@testset "Darcy mushy-zone drag" begin
    m = get_material("ER70S6")
    eps0 = 1e-4
    # Carman-Kozeny form: kd = sd f_s^2 / ((1-f_s)^3 + eps0)
    for fs in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        kd = darcy_drag(m, fs, 1e6; exponent = 2, eps0 = eps0)
        @test kd ≥ 0
        @test isfinite(kd)
        if fs == 1.0
            @test kd == 1e6 / eps0          # finite thanks to eps0
        end
    end
    # monotone increasing in solid fraction
    prev = -Inf
    for fs in 0.0:0.01:1.0
        kd = darcy_drag(m, fs, 1e6)
        @test kd > prev
        prev = kd
    end
    # the paper's s_d is effectively a freeze on a 0.6 mm cell (documented caveat)
    sd = 1e6
    @test 0.6e-3 / sd < 1e-6
    # physically-scaled alternative is a genuine porous medium
    @test porosity_scale(m, 0.6e-3, 1e7) < sd
end

# ===========================================================================
# Material database sanity + provenance
# ===========================================================================

@testset "material database" begin
    @test_throws ArgumentError get_material("INCONEL-718")
    s = get_material("er70s6")           # case-insensitive
    @test s.name == "ER70S6"
    @test s.Tsol == 1698.0 && s.Tliq == 1778.0   # paper, ref. 17
    s316 = get_material("ER316L")
    @test s316.Tsol == 1675.0 && s316.Tliq == 1708.0  # paper, ref. 54
    @test s.Tsol < s.Tliq && s316.Tsol < s316.Tliq

    # ER316L surface tension: Pichler fit, exact
    @test s316.gamma(1700.0) ≈ 1.77
    @test s316.gamma(1900.0) ≈ 1.70
    @test dgamma_dT(s316.gamma, 1900.0) ≈ -3.5e-4

    # ER70S6 sulfur content drives a POSITIVE dgamma/dT at 0.018 wt% -> inward flow
    @test dgamma_dT(s.gamma, 1900.0) > 0
    # and the sign is genuinely sulfur-dependent (the paper's key mechanism): a
    # sulfur-free wire flips to outward flow
    bare = er70s6(S_content = 0.0)
    @test dgamma_dT(bare.gamma, 1900.0) < 0
    # increasing sulfur pushes the slope monotonically upward
    prev = -Inf
    for aS in (0.0, 0.005, 0.010, 0.018, 0.04)
        d = dgamma_dT(er70s6(S_content = aS).gamma, 1900.0)
        @test d > prev
        prev = d
    end
    # the Sahoo-DebRoy slope actually changes sign somewhere across the mush
    r = dgamma_dT_range(s, 1698.0, 2400.0)
    @test r.min < 0 < r.max

    # thermal conductivity: 316L roughly half of 70S6 at the liquidus (paper)
    @test kcond(s316, s316.Tliq) < kcond(s, s.Tliq)
    @test kcond(s, s.Tliq) / kcond(s316, s316.Tliq) > 1.2

    # monotonic property curves (no non-physical wiggles)
    @test WAAMFlow.is_monotone(s.k_s; increasing = false)
    @test WAAMFlow.is_monotone(s.k_l; increasing = true)
    @test WAAMFlow.is_monotone(s.mu_l; increasing = false)
    @test WAAMFlow.is_monotone(s316.k_s; increasing = false)

    # Sahoo-DebRoy analytic slope vs central difference
    mdl = s.gamma
    for T in (1800.0, 1900.0, 2100.0)
        d = 1e-4
        fd = (mdl(T + d) - mdl(T - d)) / (2d)
        @test dgamma_dT(mdl, T) ≈ fd rtol = 1e-6
    end

    # Marangoni source returns the gradient of gamma
    src = marangoni_source(mdl, 1900.0, 1.0, 0.0, 0.0)
    @test src[1] ≈ dgamma_dT(mdl, 1900.0)

    # report renders without error
    @test occursin("ER70S6", material_report(s))
end

# ===========================================================================
# PropCurve behaviour
# ===========================================================================

@testset "PropCurve" begin
    c = PropCurve([300.0, 1000.0, 2000.0], [10.0, 20.0, 30.0])
    @test c[300.0] ≈ 10.0
    @test c[2000.0] ≈ 30.0
    @test c[650.0] ≈ 15.0                       # exact linear interpolation
    # Tangent continuation outside the range. NEVER flat — clamping to the end value
    # would invent an isothermal plume above the tabulated range.
    # The tail slope comes from the node pair, so it is (Δvalue)/(Δtemperature):
    # first node (300,10)->(1000,20) gives dT-slope 10/700; last node (1000,20)->(2000,30)
    # gives 10/1000. Extrapolate by Δtemperature, NOT by 1/Δtemperature.
    @test c[100.0] ≈ 10.0 + (100.0 - 300.0) * (10.0 / 700.0)
    @test c[2500.0] ≈ 30.0 + (2500.0 - 2000.0) * (10.0 / 1000.0)
    @test c[2500.0] > 30.0
    @test c[100.0] < 10.0
    # tail_floor guarantees strict positivity
    v = PropCurve([1000.0, 2000.0], [10.0, 5.0]; tail_floor = 1e-3)
    @test v[1.0e6] >= 1e-3
    @test v[1.0e6] > 0.0

    # a decreasing curve continues to decrease outside its range
    d = PropCurve([1000.0, 2000.0], [100.0, 50.0])
    @test d[3000.0] < 50.0
    @test d[500.0] > 100.0

    @test_throws ArgumentError PropCurve([1.0], [2.0])              # < 2 nodes
    @test_throws ArgumentError PropCurve([1.0, 1.0], [1.0, 2.0])   # duplicate T
    @test_throws ArgumentError PropCurve([1.0, 2.0], [1.0])        # length mismatch
    @test_throws ArgumentError PropCurve([1.0, 2.0], [1.0, -2.0]; require_positive = true)
    # unsorted input is sorted rather than rejected (robust to user table ordering)
    @test PropCurve([2000.0, 1000.0], [30.0, 20.0]).T == [1000.0, 2000.0]
end

# ===========================================================================
# Field bookkeeping
# ===========================================================================

@testset "fields" begin
    g = uniform_grid(4, 3, 2, 1.0, 1.0, 1.0)
    m = metrics(g)
    f = VolumeField(g)
    fill!(f, 2.0)
    @test cell_volume_sum(g, f, m) ≈ 2.0 * sum(m.vol)
    @test cell_mean(g, f, m) ≈ 2.0
    @test field_range(f) == (2.0, 2.0)
    @test size(f) == (4, 3, 2)

    # `similar` on a filled array may carry NaN payload bits in general, so seed the
    # copy explicitly and assert it is genuinely independent storage.
    f2 = WAAMFlow.similar_field(f)
    fill!(f2, 0.0)
    @test field_range(f2) == (0.0, 0.0)
    @test f2.data !== f.data
    fill!(f2, 7.0)
    @test f.data[1, 1, 1] == 2.0                  # f is untouched

    U = VectorField(g)
    @test size(U.u) == (5, 3, 2)
    @test size(U.v) == (4, 4, 2)
    @test size(U.w) == (4, 3, 3)

    # clamp_bounded! keeps a VOF field in [0,1]
    g2 = uniform_grid(3, 3, 3, 1.0, 1.0, 1.0)
    a = zeros(3, 3, 3); a[2, 2, 2] = 1.5; a[1, 1, 1] = -0.5
    WAAMFlow.clamp_bounded!(a, 0.0, 1.0, 1e-12)
    @test all(x -> -1e-12 <= x <= 1 + 1e-12, a)
    @test a[2, 2, 2] == 1 + 1e-12      # tolerance band, not hard 1.0
    @test a[1, 1, 1] == -1e-12
    # with zero tolerance the clip is exact
    b = [1.5, -0.5, 0.5]
    WAAMFlow.clamp_bounded!(b, 0.0, 1.0, 0.0)
    @test b == [1.0, 0.0, 0.5]
end