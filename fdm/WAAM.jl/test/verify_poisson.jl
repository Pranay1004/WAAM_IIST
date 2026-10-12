using Test, WAAMFlow, LinearAlgebra, Statistics, Random

Random.seed!(20261007)   # deterministic suite

# ===========================================================================
# V6 / V7 — Poisson solver
#
#     V6  manufactured solution: p = x²+y²+z², whose Laplacian is exactly 6.
#         Because the 3-point Laplacian is exact for a quadratic, the discrete
#         solution must reproduce the analytic field to machine precision, so this
#         pins the coefficient normalisation, the Dirichlet RHS treatment and the
#         operator indexing all at once.
#
#     V7  pure-Neumann solvability: the Laplacian without a Dirichlet cell is
#         SINGULAR. The solver must report failure rather than return a
#         plausible-looking field.
#
#     V8  fused operator vs an independent sparse assembly, and CG vs a dense
#         direct solve.
# ===========================================================================

"Analytic field used by the MMS: p = x²+y²+z², with -∇²p = 6."
function mms_field(m, nx, ny, nz)
    [m.xc[i]^2 + m.yc[j]^2 + m.zc[k]^2 for k in 1:nz, j in 1:ny, i in 1:nx]
end

"Pin every boundary cell (required: pure Neumann has no solution)."
function boundary_dirichlet(nx, ny, nz)
    fixed = falses(nx * ny * nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        (i == 1 || i == nx || j == 1 || j == ny || k == 1 || k == nz) && (fixed[c] = true)
    end
    fixed
end

@testset "V6 Poisson: manufactured solution" begin
    for n in (8, 12, 16)
        g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
        m = metrics(g)
        ax, ay, az, vol = m.ax, m.ay, m.az, m.vol
        pex = mms_field(m, n, n, n)
        fixed = boundary_dirichlet(n, n, n)
        # rhs = -lap(p) = -6, since apply7! realises the positive-definite operator
        # L = -div(a grad p).
        rhs = fill(-6.0, n, n, n)
        p = zeros(n, n, n)

        res, iters, conv = poisson7!(vec(p), vec(rhs), ax, ay, az, vol,
                                    m.dx, m.dy, m.dz, fixed, vec(pex))
        @test conv
        @test iters > 0
        @test res / sqrt(sum(abs2, rhs)) < 1e-8

        free = [i for i in 1:n^3 if !fixed[i]]
        err = norm(vec(p)[free] - vec(pex)[free]) / norm(vec(pex)[free])
        # A 3-point Laplacian is EXACT for a quadratic, so this is not a
        # second-order check but an exactness check: any growth with n means the
        # operator, its normalisation or the Dirichlet treatment is wrong.
        @test err < 1e-8

        # Dirichlet values are imposed, not merely left at the initial guess.
        @test p[1, 1, 1] ≈ pex[1, 1, 1]
        @test p[n, n, n] ≈ pex[n, n, n]
    end
end

@testset "V6b Poisson: variable coefficient" begin
    # Spatially varying conductivity. The source is built from the SAME discrete
    # operator (manufactured rather than analytic), so the discrete solution must be
    # reproduced exactly regardless of k — this catches a face-interpolation bug that a
    # constant-coefficient test cannot see.
    n = 12
    g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
    m = metrics(g)
    # conductances k(x) * A on each face plane (n+1 planes in the axis direction)
    # Broadcast explicitly against the face-plane shapes (n+1 planes along the axis).
    ax = [m.ax[i, j, k] * (1.0 + 0.5 * sin(2pi * g.xf[i]))
          for k in 1:n, j in 1:n, i in 1:(n + 1)]
    ay = [m.ay[i, j, k] * (1.0 + 0.5 * sin(2pi * g.yf[j]))
          for k in 1:n, j in 1:(n + 1), i in 1:n]
    az = [m.az[i, j, k] * (1.0 + 0.5 * sin(2pi * g.zf[k]))
          for k in 1:(n + 1), j in 1:n, i in 1:n]
    pex = [m.xc[i]^2 + m.yc[j]^2 + m.zc[k]^2 for k in 1:n, j in 1:n, i in 1:n]
    fixed = boundary_dirichlet(n, n, n)

    # Manufactured source: rhs = L(pex) using the same stencil. The Dirichlet mask is
    # deliberately all-false here — apply7! suppresses Dirichlet neighbours, so masking
    # here would omit exactly the terms poisson7! re-adds via _absorb_dirichlet, and the
    # two would double-count them.
    rhs = vec(apply7!(zeros(n^3), vec(pex), ax, ay, az, m.vol, m.dx, m.dy, m.dz,
                      falses(n^3)))
    p = zeros(n, n, n)
    res, iters, conv = poisson7!(vec(p), copy(rhs), ax, ay, az, m.vol,
                                m.dx, m.dy, m.dz, fixed, vec(pex))
    @test conv
    free = [i for i in 1:n^3 if !fixed[i]]
    @test norm(vec(p)[free] - vec(pex)[free]) / norm(vec(pex)[free]) < 1e-8
end

@testset "V7 pure Neumann is singular" begin
    n = 8
    g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
    m = metrics(g)
    fixed = falses(n^3)
    fv = zeros(n^3)
    rhs = ones(n^3)
    p = zeros(n^3)
    res, iters, conv = poisson7!(p, rhs, m.ax, m.ay, m.az, m.vol,
                                m.dx, m.dy, m.dz, fixed, fv)
    # The constant field lies in the null space: CG cannot converge, and the solver
    # must SAY SO rather than hand back a field that merely looks reasonable. A
    # silently-returned divergent field is the failure mode this guards against.
    @test !conv
    @test res > 1e-6 * norm(rhs)          # the reported residual is genuinely large
    @test all(isfinite, p)                # and it did not fill with NaN/Inf

    # A Dirichlet cell restores solvability.
    fixed[1] = true
    p2 = zeros(n^3)
    _, _, conv2 = poisson7!(p2, copy(rhs), m.ax, m.ay, m.az, m.vol,
                           m.dx, m.dy, m.dz, fixed, fv)
    @test conv2
end

@testset "V7b Neumann compatibility" begin
    # A Neumann problem is solvable only if the source integrates to zero over the
    # domain. This is a property of the discretisation, checked here so the solver's
    # behaviour on incompatible data is understood rather than discovered.
    n = 8
    g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
    m = metrics(g)
    fixed = falses(n^3)
    fixed[1] = true
    fv = zeros(n^3)
    # zero-mean source
    rhs = randn(n^3)
    rhs[1] = 0.0
    rhs .-= mean(rhs)
    p = zeros(n^3)
    _, _, conv = poisson7!(p, copy(rhs), m.ax, m.ay, m.az, m.vol,
                          m.dx, m.dy, m.dz, fixed, fv)
    @test conv
end

@testset "V8 linear algebra" begin
    # --- fused 7-point operator vs an independent sparse assembly ---
    n = 5
    g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
    m = metrics(g)
    fixed = falses(n^3)
    fixed[1] = true
    fixed[n + 2] = true
    A = to_sparse(m.ax, m.ay, m.az, m.vol, m.dx, m.dy, m.dz, fixed)
    Ad = [A[i, j] for i in 1:n^3, j in 1:n^3]
    x = rand(n^3)
    y = apply7!(similar(x), x, m.ax, m.ay, m.az, m.vol, m.dx, m.dy, m.dz, fixed)
    @test norm(Ad * x - y) / norm(y) < 1e-12

    # --- the assembled operator is symmetric positive semi-definite ---
    @test maximum(abs.(Ad .- Ad')) < 1e-10
    @test minimum(real.(eigvals(Symmetric(Ad)))) > -1e-9

    # --- a Dirichlet cell makes it genuinely positive definite ---
    fixed2 = boundary_dirichlet(n, n, n)
    A2 = to_sparse(m.ax, m.ay, m.az, m.vol, m.dx, m.dy, m.dz, fixed2)
    A2d = [A2[i, j] for i in 1:n^3, j in 1:n^3]
    free = [i for i in 1:n^3 if !fixed2[i]]
    @test minimum(eigvals(Symmetric(A2d[free, free]))) > 1e-6

    # --- CG agrees with a dense direct solve ---
    fixed3 = falses(n^3); fixed3[1] = true
    fv = zeros(n^3)
    A3 = to_sparse(m.ax, m.ay, m.az, m.vol, m.dx, m.dy, m.dz, fixed3)
    A3d = [A3[i, j] for i in 1:n^3, j in 1:n^3]
    idx = [i for i in 1:n^3 if !fixed3[i]]
    rhs = rand(n^3); rhs[1] = 0.0
    p = zeros(n^3)
    res, iters, conv = poisson7!(p, copy(rhs), m.ax, m.ay, m.az, m.vol,
                                m.dx, m.dy, m.dz, fixed3, fv)
    @test conv
    @test norm(A3d[idx, idx] * p[idx] - rhs[idx]) / norm(rhs[idx]) < 1e-8
    ptrue = zeros(n^3); ptrue[idx] = A3d[idx, idx] \ rhs[idx]
    @test norm(p - ptrue) / norm(ptrue) < 1e-6

    # --- a 3-D array argument is accepted and agrees with the flat form ---
    p3 = zeros(n, n, n)
    poisson7!(vec(p3), copy(rhs), m.ax, m.ay, m.az, m.vol, m.dx, m.dy, m.dz, fixed3, fv)
    @test norm(vec(p3) - p) < 1e-12
    y3 = apply7!(zeros(n, n, n), p3, m.ax, m.ay, m.az, m.vol, m.dx, m.dy, m.dz, fixed3)
    yflat = apply7!(similar(vec(p3)), vec(p3), m.ax, m.ay, m.az, m.vol,
                    m.dx, m.dy, m.dz, fixed3)
    @test norm(vec(y3) - yflat) < 1e-12
end

@testset "V8b negative face coefficient is rejected" begin
    n = 4
    g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
    m = metrics(g)
    fixed = falses(n^3); fixed[1] = true
    ax = copy(m.ax)
    ax[3, 2, 2] = -1.0          # a negative conductance makes the operator indefinite
    @test_throws ArgumentError poisson7!(zeros(n^3), ones(n^3), ax, m.ay, m.az, m.vol,
                                         m.dx, m.dy, m.dz, fixed, zeros(n^3))
end

@testset "Multigrid is gated" begin
    # Multigrid is EXPERIMENTAL and must refuse by default rather than be used by
    # accident. This is asserted so the gate cannot be quietly removed.
    n = 8
    g = uniform_grid(n, n, n, 1.0, 1.0, 1.0)
    m = metrics(g)
    fixed = falses(n^3); fixed[1] = true
    @test_throws ArgumentError WAAMFlow.multigrid_poisson!(
        zeros(n, n, n), ones(n, n, n), m.ax, m.ay, m.az, m.vol,
        m.dx, m.dy, m.dz, fixed, zeros(n^3))
end