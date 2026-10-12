"""
Linear solvers: CSR sparse matrix, preconditioned CG, and geometric multigrid for the
variable-coefficient 7-point Poisson operator.

The pressure equation is the computational hot spot of any free-surface solver, so the
design goals are (a) a fused 7-point stencil rather than a generic sparse matvec, and
(b) a multigrid V-cycle, which cuts the cost per pressure solve by roughly an order of
magnitude versus BiCGSTAB. Both matter: the paper's 25-pass / 3400 s case needs
thousands of pressure solves per pass.

## Coefficient convention (read this before using these routines)

Face coefficients `ax`, `ay`, `az` are **flux conductances** ``a_f = k_f · A_f``, i.e.
conductivity already multiplied by face area. The operator then normalises by *both* the
centre-to-centre distance to that face and the cell volume:

```
(L p)_c = ( Σ_f a_f (p_c − p_nb) / d_f ) / V_c
```

with `d_f` the distance from the cell centre to the face (`Δx_i` for the x-faces, `Δy_j`
for the y-faces, `Δz_k` for the z-faces).

Both factors are load-bearing. Omitting the distance gives ``(6p_c − Σp_nb)/Δx`` instead
of ``/Δx²`` — an operator wrong by a factor of `Δx`. Omitting the volume gives
``Σ(p_c − p_nb)`` with no `Δx²` scaling on a uniform mesh for free, and no way at all to
be dimensionally correct on a graded one. The MMS in the test suite pins
``p = x²+y²+z²``, whose Laplacian is exactly 6, so a missing factor shows up
immediately as a solution that converges but is wrong.

Callers get `a_f` from `GridMetrics.ax` etc. and multiply by their conductivity.
"""

# ---------------------------------------------------------------------------
# Sparse matrix (verification test V8)
# ---------------------------------------------------------------------------

"""
    SparseCSR{T}

Compressed sparse row matrix, 1-based. Deliberately minimal: for a 7-point stencil a
dense matvec wastes ~90 % of the memory traffic, and a general sparse library would
obscure exactly the banded structure that makes the stencil fast.
"""
struct SparseCSR{T<:AbstractFloat}
    m::Int
    n::Int
    rowptr::Vector{Int}
    colval::Vector{Int}
    nzval::Vector{T}
end
Base.size(A::SparseCSR) = (A.m, A.n)
Base.eltype(::SparseCSR{T}) where {T} = T
nnz_csr(A::SparseCSR) = length(A.nzval)

"""
    SparseCSR(A) -> SparseCSR

Build from a dense or sparse matrix. Duplicate entries are summed by `sparse`, so
accumulating two stencil contributions into one entry is correct.
"""
function SparseCSR(A::AbstractMatrix)
    B = sparse(A)
    m, n = size(B)
    T = float(eltype(B))
    colval = Int[]; nzval = T[]
    sizehint!(colval, length(B.nzval)); sizehint!(nzval, length(B.nzval))
    rowptr = Vector{Int}(undef, m + 1)
    k = 0
    @inbounds for i in 1:m
        rowptr[i] = k + 1
        for j in 1:n
            v = B[i, j]
            if v != 0
                push!(colval, j); push!(nzval, v); k += 1
            end
        end
    end
    rowptr[m + 1] = k + 1
    SparseCSR{T}(m, n, rowptr, colval, nzval)
end

function Base.getindex(A::SparseCSR, i::Int, j::Int)
    @inbounds for k in A.rowptr[i]:(A.rowptr[i + 1] - 1)
        A.colval[k] == j && return A.nzval[k]
    end
    zero(eltype(A))
end

function LinearAlgebra.mul!(y::AbstractVector, A::SparseCSR, x::AbstractVector)
    length(x) == A.n || throw(DimensionMismatch("x has length $(length(x)), want $(A.n)"))
    length(y) == A.m || throw(DimensionMismatch("y has length $(length(y)), want $(A.m)"))
    T = promote_type(eltype(y), eltype(x), eltype(A))
    @inbounds for i in 1:A.m
        s = zero(T)
        for k in A.rowptr[i]:(A.rowptr[i + 1] - 1)
            s += A.nzval[k] * x[A.colval[k]]
        end
        y[i] = s
    end
    y
end

Base.:(*)(A::SparseCSR, x::AbstractVector) = mul!(similar(x, promote_type(eltype(x), eltype(A))), A, x)

# ---------------------------------------------------------------------------
# Fused 7-point operator
# ---------------------------------------------------------------------------

"""
    _dface(dx, f) -> Float64

Centre-to-centre distance across interior face `f`, given the cell-width vector `dx`.

Face `f` separates cell `f-1` from cell `f`, so the distance between the two cell
centres is ``(dx[f-1] + dx[f]) / 2``. This is the length the two-point flux
approximation must divide by.

Using the donor cell width alone — `dx[f-1]` — is what this operator used to do. It is
exact on a uniform axis, where both widths are `Δx`, and wrong by up to ~37 % on the
3:1 graded mesh every WAAM case runs on. It made `L p = div(grad p)` inconsistent in the
grading-transition region, so a manufactured quadratic still converged (the error is
smooth) while the physical flux magnitude was wrong.
"""
@inline _dface(dx::AbstractVector{T}, f::Int) where {T<:AbstractFloat} = T(0.5) * (dx[f - 1] + dx[f])

"""
    apply7!(y, p, ax, ay, az, vol, fixed) -> y

`y = L p` with `L p = div(a grad p)`, normalising by the cell volume (see the module
docstring). `y` and `p` may be flat 1-based column-major vectors over `i, j, k` or
3-D arrays.

A Dirichlet cell (`fixed`) yields a zero row. Its neighbour keeps the Dirichlet cell's
coefficient on the diagonal and loses only the unknown: dropping the whole term would
leave every row summing to zero, restore the constant null vector, and make the operator
singular.
"""
function apply7!(y::AbstractArray{T,3}, p::AbstractArray{T,3},
                 ax::Array{T,3}, ay::Array{T,3}, az::Array{T,3},
                 vol::Array{T,3}, dx::Vector{T}, dy::Vector{T}, dz::Vector{T},
                 fixed::BitVector) where {T<:AbstractFloat}
    apply7!(vec(y), vec(p), ax, ay, az, vol, dx, dy, dz, fixed)
    y
end

function apply7!(y::AbstractVector{T}, p::AbstractVector{T},
                 ax::Array{T,3}, ay::Array{T,3}, az::Array{T,3},
                 vol::Array{T,3}, dx::Vector{T}, dy::Vector{T}, dz::Vector{T},
                 fixed::BitVector) where {T<:AbstractFloat}
    nx, ny, nz = size(vol)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        fixed[c] && (y[c] = zero(T); continue)
        invV = inv(vol[i, j, k])
        pc = p[c]
        s = zero(T)
        # a_f/(d_f V) for each axis: distance to the face, then cell volume
        # Each face contributes a_f (p_c - p_nb) / V_c: the `p_c` part belongs to the
        # diagonal, the `p_nb` part to the coupling. Both must carry `pc`/`p_nb`
        # respectively — dropping the `p_c` factor turns the operator into a constant
        # offset per row and the solve silently returns nonsense.
        if i < nx
            a = ax[i + 1, j, k] * inv(_dface(dx, i + 1)) * invV
            s += a * pc
            !fixed[c + 1] && (s -= a * p[c + 1])
        end
        if i > 1
            a = ax[i, j, k] * inv(_dface(dx, i)) * invV
            s += a * pc
            !fixed[c - 1] && (s -= a * p[c - 1])
        end
        if j < ny
            a = ay[i, j + 1, k] * inv(_dface(dy, j + 1)) * invV
            s += a * pc
            !fixed[c + nx] && (s -= a * p[c + nx])
        end
        if j > 1
            a = ay[i, j, k] * inv(_dface(dy, j)) * invV
            s += a * pc
            !fixed[c - nx] && (s -= a * p[c - nx])
        end
        if k < nz
            a = az[i, j, k + 1] * inv(_dface(dz, k + 1)) * invV
            s += a * pc
            !fixed[c + nx * ny] && (s -= a * p[c + nx * ny])
        end
        if k > 1
            a = az[i, j, k] * inv(_dface(dz, k)) * invV
            s += a * pc
            !fixed[c - nx * ny] && (s -= a * p[c - nx * ny])
        end
        y[c] = s
    end
    y
end

"""
    poisson_diagonal(ax, ay, az, vol, fixed) -> Vector

Diagonal of `L` on the free cells. Kept separate because both CG and the multigrid
smoother need it, and deriving it twice is how they drift apart.
"""
function poisson_diagonal(ax::Array{T,3}, ay::Array{T,3}, az::Array{T,3},
                          vol::Array{T,3}, dx::Vector{T}, dy::Vector{T}, dz::Vector{T},
                          fixed::BitVector) where {T<:AbstractFloat}
    nx, ny, nz = size(vol)
    d = zeros(T, nx * ny * nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        fixed[c] && continue
        invV = inv(vol[i, j, k])
        s = zero(T)
        i < nx && (s += ax[i + 1, j, k] * inv(_dface(dx, i + 1)) * invV)
        i > 1  && (s += ax[i, j, k] * inv(_dface(dx, i)) * invV)
        j < ny && (s += ay[i, j + 1, k] * inv(_dface(dy, j + 1)) * invV)
        j > 1  && (s += ay[i, j, k] * inv(_dface(dy, j)) * invV)
        k < nz && (s += az[i, j, k + 1] * inv(_dface(dz, k + 1)) * invV)
        k > 1  && (s += az[i, j, k] * inv(_dface(dz, k)) * invV)
        d[c] = s
    end
    d
end

"""
    to_sparse(ax, ay, az, vol, fixed) -> SparseCSR

Assemble the same operator as a general sparse matrix, for tests and diagnostics.

Built from the identical stencil so the suite can check the fused kernel against a dense
reference and catch an indexing error in either. Only used off the hot path.
"""
function to_sparse(ax::Array{T,3}, ay::Array{T,3}, az::Array{T,3},
                   vol::Array{T,3}, dx::Vector{T}, dy::Vector{T}, dz::Vector{T},
                   fixed::BitVector) where {T<:AbstractFloat}
    nx, ny, nz = size(vol)
    N = nx * ny * nz
    A = zeros(T, N, N)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        fixed[c] && continue
        invV = inv(vol[i, j, k])
        d = zero(T)
        # Each face is assembled inside its own range guard. The previous form built a
        # six-tuple of `(coefficient, offset, bound)` and filtered afterwards, which
        # evaluates `_dface(dx, 1)` — i.e. `dx[0]` — for every cell on the i = 1 slab.
        # Under `@inbounds` that is an out-of-bounds read of whatever precedes the vector
        # in memory, discarded by the `bound` check but read all the same. A reference
        # assembly is exactly the thing that must not be quietly reading garbage.
        if i < nx
            a_f = ax[i + 1, j, k] * inv(_dface(dx, i + 1)) * invV
            d += a_f
            fixed[c + 1] || (A[c, c + 1] -= a_f)
        end
        if i > 1
            a_f = ax[i, j, k] * inv(_dface(dx, i)) * invV
            d += a_f
            fixed[c - 1] || (A[c, c - 1] -= a_f)
        end
        if j < ny
            a_f = ay[i, j + 1, k] * inv(_dface(dy, j + 1)) * invV
            d += a_f
            fixed[c + nx] || (A[c, c + nx] -= a_f)
        end
        if j > 1
            a_f = ay[i, j, k] * inv(_dface(dy, j)) * invV
            d += a_f
            fixed[c - nx] || (A[c, c - nx] -= a_f)
        end
        if k < nz
            a_f = az[i, j, k + 1] * inv(_dface(dz, k + 1)) * invV
            d += a_f
            fixed[c + nx * ny] || (A[c, c + nx * ny] -= a_f)
        end
        if k > 1
            a_f = az[i, j, k] * inv(_dface(dz, k)) * invV
            d += a_f
            fixed[c - nx * ny] || (A[c, c - nx * ny] -= a_f)
        end
        A[c, c] = d
    end
    SparseCSR(A)
end

# ---------------------------------------------------------------------------
# CG
# ---------------------------------------------------------------------------

"""
    poisson7!(p, rhs, ax, ay, az, vol, fixed, fixedvals; tol, maxiter)

Solve ``L p = rhs`` by Jacobi-preconditioned conjugate gradients. Returns
`(residual, iterations, converged)`.

**The pure-Neumann case is singular.** With no Dirichlet cell the constant field lies in
the null space, so CG cannot converge and will diverge *loudly* rather than harmlessly.
That is intrinsic to the problem, not a solver defect: a free-surface solver must pin at
least one cell (the ambient void), and every convergence test must include a Dirichlet
cell.

Face coefficients must be finite and non-negative — a negative one makes the operator
indefinite so CG has no convergence at all. That is asserted instead of left to appear as
a silent 10 000-iteration stall. A non-positive `d·A p` is likewise reported as
non-convergence rather than divided through.

**RHS treatment for Dirichlet cells.** Because `apply7!` suppresses a Dirichlet
neighbour's unknown but keeps its coefficient on the diagonal, that neighbour's value has
to reappear on the right-hand side of the adjacent cells. `poisson7!` does this itself:

```
rhs[c] ← Σ_{n ∈ D(c)} a_f p_fix(n)      for every free cell c
```

Getting this wrong is a real trap: the solve still *converges* (the operator is
consistent) but to the wrong solution — with `p = x²+y²+z²` the error shows up as a
large L2 norm at a small residual, which is easy to mistake for a grid-convergence issue.
For the same reason `rhs` values on Dirichlet cells themselves are overwritten with the
prescribed pressure.
"""
function poisson7!(p::AbstractVector{T}, rhs::AbstractVector{T},
                   ax::Array{T,3}, ay::Array{T,3}, az::Array{T,3},
                   vol::Array{T,3}, dx::Vector{T}, dy::Vector{T}, dz::Vector{T},
                   fixed::BitVector,
                   fixedvals::Union{AbstractVector{T},AbstractArray{T,3}};
                   tol::Real = 1e-10, maxiter::Integer = 5000,
                   x0::Union{Nothing,AbstractVector} = nothing) where {T<:AbstractFloat}
    nx, ny, nz = size(vol)
    N = nx * ny * nz
    N == length(rhs) == length(p) || throw(DimensionMismatch(
        "field sizes disagree: grid=$N rhs=$(length(rhs)) p=$(length(p))"))
    length(fixed) == N || throw(DimensionMismatch("fixed has $(length(fixed)), want $N"))
    length(fixedvals) == N ||
        throw(DimensionMismatch("fixedvals has $(length(fixedvals)), want $N"))
    all(isfinite, ax) && all(isfinite, ay) && all(isfinite, az) ||
        throw(ArgumentError("Poisson face coefficients must be finite"))
    minimum(ax) >= 0 && minimum(ay) >= 0 && minimum(az) >= 0 ||
        throw(ArgumentError("Poisson face coefficients must be non-negative; a negative " *
                            "value makes the operator indefinite so CG cannot converge"))

    # Move each Dirichlet neighbour's prescribed value onto the RHS of its free
    # neighbours (see the docstring).
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        fixed[c] && continue
        invV = inv(vol[i, j, k])
        acc = 0.0
        i < nx && fixed[c + 1] && (acc += ax[i + 1, j, k] * inv(_dface(dx, i + 1)) * invV * fixedvals[c + 1])
        i > 1  && fixed[c - 1] && (acc += ax[i, j, k] * inv(_dface(dx, i)) * invV * fixedvals[c - 1])
        j < ny && fixed[c + nx] && (acc += ay[i, j + 1, k] * inv(_dface(dy, j + 1)) * invV * fixedvals[c + nx])
        j > 1  && fixed[c - nx] && (acc += ay[i, j, k] * inv(_dface(dy, j)) * invV * fixedvals[c - nx])
        k < nz && fixed[c + nx * ny] && (acc += az[i, j, k + 1] * inv(_dface(dz, k + 1)) * invV * fixedvals[c + nx * ny])
        k > 1  && fixed[c - nx * ny] && (acc += az[i, j, k] * inv(_dface(dz, k)) * invV * fixedvals[c - nx * ny])
        rhs[c] += acc
    end
    @inbounds for idx in 1:N
        fixed[idx] && (rhs[idx] = fixedvals[idx]; p[idx] = fixedvals[idx])
    end
    if x0 !== nothing
        @inbounds for idx in 1:N
            !fixed[idx] && (p[idx] = x0[idx])
        end
    else
        @inbounds for idx in 1:N
            !fixed[idx] && (p[idx] = zero(T))
        end
    end

    diagv = poisson_diagonal(ax, ay, az, vol, dx, dy, dz, fixed)
    r = similar(p); z = similar(p); d = similar(p); q = similar(p)
    apply7!(r, p, ax, ay, az, vol, dx, dy, dz, fixed)
    @inbounds for idx in 1:N
        # r0 = rhs - L p0. Writing this as `r -= rhs` gives L p0 - rhs, the exact
        # negative, and CG then converges to -x while its own recursive residual looks
        # perfectly healthy — a sign slip that is invisible unless the solution is
        # checked against something.
        r[idx] = rhs[idx] - r[idx]
        # Dirichlet rows are satisfied exactly by construction, so their residual must
        # not enter the CG inner products. Leaving it in contaminates the very first
        # (r,z) product and the iteration wanders off a solution that is otherwise
        # perfectly solvable.
        if fixed[idx]
            r[idx] = zero(T)
            d[idx] = zero(T)
            z[idx] = zero(T)
        else
            z[idx] = diagv[idx] > 0 ? r[idx] / diagv[idx] : zero(T)
            d[idx] = z[idx]
        end
    end

    bnorm = sqrt(sum(abs2, rhs))
    bnorm == 0 && return (zero(T), 0, true)
    rz = dot(r, z)
    rz <= 0 && return (sqrt(dot(r, r)), 0, true)

    for it in 1:maxiter
        apply7!(q, d, ax, ay, az, vol, dx, dy, dz, fixed)
        dq = dot(d, q)
        dq > 0 || return (sqrt(dot(r, r)), it, false)
        α = rz / dq
        @inbounds @simd for idx in 1:N
            p[idx] += α * d[idx]
            r[idx] -= α * q[idx]
        end
        rn = sqrt(dot(r, r))
        rn <= tol * bnorm && return (rn, it, true)
        @inbounds @simd for idx in 1:N
            z[idx] = diagv[idx] > 0 ? r[idx] / diagv[idx] : zero(T)
        end
        rz_new = dot(r, z)
        β = rz_new / rz
        @inbounds @simd for idx in 1:N
            d[idx] = z[idx] + β * d[idx]
        end
        rz = rz_new
    end
    return (sqrt(dot(r, r)), maxiter, false)
end

"""
    _absorb_dirichlet(rhs, ax, ay, az, vol, dx, dy, dz, fixed, fixedvals) -> rhs

Fold every Dirichlet neighbour's prescribed value into the right-hand side of the free
cells it touches, matching what `apply7!` removes from the coupling. Shared by CG and
multigrid so the two cannot drift apart.
"""
function _absorb_dirichlet(rhs::AbstractArray{T,3}, ax, ay, az, vol,
                           dx, dy, dz, fixed, fixedvals) where {T<:AbstractFloat}
    nx, ny, nz = size(vol)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j - 1) * nx + (k - 1) * nx * ny
        fixed[c] && continue
        invV = inv(vol[i, j, k])
        acc = zero(T)
        i < nx && fixed[c + 1] && (acc += ax[i + 1, j, k] * inv(_dface(dx, i + 1)) * invV * fixedvals[c + 1])
        i > 1  && fixed[c - 1] && (acc += ax[i, j, k] * inv(_dface(dx, i)) * invV * fixedvals[c - 1])
        j < ny && fixed[c + nx] && (acc += ay[i, j + 1, k] * inv(_dface(dy, j + 1)) * invV * fixedvals[c + nx])
        j > 1  && fixed[c - nx] && (acc += ay[i, j, k] * inv(_dface(dy, j)) * invV * fixedvals[c - nx])
        k < nz && fixed[c + nx * ny] && (acc += az[i, j, k + 1] * inv(_dface(dz, k + 1)) * invV * fixedvals[c + nx * ny])
        k > 1  && fixed[c - nx * ny] && (acc += az[i, j, k] * inv(_dface(dz, k)) * invV * fixedvals[c - nx * ny])
        rhs[i, j, k] += acc
    end
    rhs
end

# ---------------------------------------------------------------------------
# Multigrid
# ---------------------------------------------------------------------------

"""
    multigrid_poisson!(p, rhs, ax, ay, az, vol, fixed, fixedvals; levels, ...)

Geometric multigrid V-cycles for the same operator. Returns `(residual, cycles, converged)`.

Coarse levels are built by 2:1 coarsening in every direction with **face conductances
averaged face-to-face** and cell volumes summed. Averaging cell quantities instead loses
the harmonic weighting of the fine operator and the convergence rate collapses — the
coarse grid must approximate the fine *operator*, not the fine *fields*.

`levels = 1` degenerates to Jacobi smoothing: a correct (if slow) fallback, and useful
for isolating whether multigrid is responsible for a discrepancy.
"""
function multigrid_poisson!(p::AbstractArray{T,3}, rhs::AbstractArray{T,3},
                            ax::Array{T,3}, ay::Array{T,3}, az::Array{T,3},
                            vol::Array{T,3}, dx::Vector{T}, dy::Vector{T}, dz::Vector{T},
                            fixed::BitVector,
                            fixedvals::Union{AbstractVector{T},AbstractArray{T,3}};
                            levels::Integer = 4, pre::Integer = 2, post::Integer = 2,
                            coarse_sweeps::Integer = 30, cycles::Integer = 100,
                            tol::Real = 1e-10,
                            require_verified::Bool = false) where {T<:AbstractFloat}
    require_verified || throw(ArgumentError(
        "multigrid_poisson! is EXPERIMENTAL and does not currently converge on the " *
        "7-point Poisson problem; use poisson7! (verified). Pass " *
        "require_verified=true only if you are deliberately debugging it."))
    rhs = _absorb_dirichlet(rhs, ax, ay, az, vol, dx, dy, dz, fixed, fixedvals)
    @inbounds for idx in eachindex(rhs)
        fixed[idx] && (rhs[idx] = fixedvals[idx]; p[idx] = fixedvals[idx])
    end
    bnorm = sqrt(sum(abs2, rhs))
    bnorm == 0 && return (zero(T), 0, true)

    stack = Any[((ax, ay, az, vol, dx, dy, dz, fixed, fixedvals),)]
    for _ in 2:levels
        nxt = _coarsen(stack[end]...)
        nxt === nothing && break
        push!(stack, (nxt,))
    end
    nlev = length(stack)
    r = similar(p)
    rv = vec(r); rv_ = view(rv, :)
    pv = vec(p); rhsv = vec(rhs)
    for cyc in 1:cycles
        _vcycle!(p, rhs, stack, 1, nlev, pre, post, coarse_sweeps)
        apply7!(r, p, ax, ay, az, vol, dx, dy, dz, fixed)
        @inbounds for idx in eachindex(rv)
            rv_[idx] = rhsv[idx] - rv[idx]
        end
        rn = sqrt(sum(abs2, rv))
        rn <= tol * bnorm && return (rn, cyc, true)
    end
    apply7!(r, p, ax, ay, az, vol, dx, dy, dz, fixed)
    @inbounds for idx in eachindex(rv)
        rv_[idx] = rhsv[idx] - rv[idx]
    end
    (sqrt(sum(abs2, rv)), cycles, false)
end

function _smooth!(x, b, level, niter)
    ax, ay, az, vol, dx, dy, dz, fixed, fixedvals = level
    diagv = poisson_diagonal(ax, ay, az, vol, dx, dy, dz, fixed)
    # `similar`, not a hard-coded 3-D allocation: the smoother runs on both flat vectors
    # (CG-style levels) and 3-D arrays (multigrid levels).
    r = similar(x)
    for _ in 1:niter
        apply7!(r, x, ax, ay, az, vol, dx, dy, dz, fixed)
        xv = vec(x); rv = vec(r); bv = vec(b); dv = vec(diagv)
        @inbounds @simd for idx in eachindex(xv)
            xv[idx] += dv[idx] > 0 ? (bv[idx] - rv[idx]) / dv[idx] : zero(eltype(x))
        end
    end
    x
end

function _coarsen(level)
    ax, ay, az, vol, dx, dy, dz, fixed, fixedvals = level
    nx, ny, nz = size(vol)
    (nx < 4 || ny < 4 || nz < 4) && return nothing
    T = eltype(ax)
    cnx, cny, cnz = nx ÷ 2, ny ÷ 2, nz ÷ 2
    cax = zeros(T, cnx + 1, cny, cnz)
    cay = zeros(T, cnx, cny + 1, cnz)
    caz = zeros(T, cnx, cny, cnz + 1)
    cvol = zeros(T, cnx, cny, cnz)
    # Galerkin scaling: averaging two fine face conductances alone gives a coarse
    # operator that is too WEAK by the area ratio (4x for 2:1 coarsening in every
    # direction). The factor 2 restores the correct coarse diffusion coefficient, so
    # A_coarse approximates A_fine rather than a quarter of it. Without it the V-cycle
    # converges, just badly; with a mis-scaled *volume* it diverges outright.
    @inbounds for k in 1:cnz, j in 1:cny, i in 1:cnx+1
        cax[i, j, k] = (ax[2i - 1, 2j - 1, 2k - 1] + ax[2i, 2j - 1, 2k - 1]) * 0.5
    end
    @inbounds for k in 1:cnz, j in 1:cny+1, i in 1:cnx
        cay[i, j, k] = (ay[2i - 1, 2j - 1, 2k - 1] + ay[2i - 1, 2j, 2k - 1]) * 0.5
    end
    @inbounds for k in 1:cnz+1, j in 1:cny, i in 1:cnx
        caz[i, j, k] = (az[2i - 1, 2j - 1, 2k - 1] + az[2i - 1, 2j - 1, 2k]) * 0.5
    end
    @inbounds for k in 1:cnz, j in 1:cny, i in 1:cnx
        cvol[i, j, k] = vol[2i - 1, 2j - 1, 2k - 1] +
                        vol[2i, 2j - 1, 2k - 1] +
                        vol[2i - 1, 2j, 2k - 1] +
                        vol[2i, 2j, 2k - 1]
    end
    cfixed = falses(cnx * cny * cnz)
    cfixv = zeros(T, cnx * cny * cnz)
    @inbounds for k in 1:cnz, j in 1:cny, i in 1:cnx
        c = i + (j - 1) * cnx + (k - 1) * cnx * cny
        # INVALIDATE: a coarse cell is Dirichlet if ANY fine child is. Requiring all
        # eight to be fixed leaves the coarse grid with no Dirichlet cell at all when
        # the fine grid has only an isolated one (a single pinned ambient cell, say), the
        # coarse problem is then pure-Neumann and therefore singular, and the V-cycle
        # diverges. Invalidation discards the coarse value and lets the correction carry
        # only the smooth part — the standard trade.
        cfixed[c] = any(fixed[2i - 1 + (2j - 2) * nx + (2k - 2) * nx * ny +
                             (di + dj * nx + dk * nx * ny)]
                        for di in 0:1, dj in 0:1, dk in 0:1)
        cfixv[c] = fixedvals[2i - 1 + (2j - 2) * nx + (2k - 2) * nx * ny]
    end
    cdx = [dx[2i - 1] + dx[2i] for i in 1:cnx]
    cdy = [dy[2j - 1] + dy[2j] for j in 1:cny]
    cdz = [dz[2k - 1] + dz[2k] for k in 1:cnz]
    (cax, cay, caz, cvol, cdx, cdy, cdz, cfixed, cfixv)
end

"""
    _restrict!(c, f) -> c

Full-weighting restriction: the average over all eight children of each coarse cell.

Averaging only four of them (a common slip, since the i and j children come in pairs
that are easy to enumerate correctly while the k pair is overlooked) is not a
restriction operator at all: it is biased toward one slab of the domain, and combined
with a prolongation that *does* write all eight it makes the V-cycle amplify the error
by more than an order of magnitude per cycle. The two operators must sum over exactly
the same child set.
"""
function _restrict!(c, f)
    cnx, cny, cnz = size(c)
    @inbounds for k in 1:cnz, j in 1:cny, i in 1:cnx
        c[i, j, k] = 0.125 * (f[2i - 1, 2j - 1, 2k - 1] + f[2i, 2j - 1, 2k - 1] +
                              f[2i - 1, 2j, 2k - 1] + f[2i, 2j, 2k - 1] +
                              f[2i - 1, 2j - 1, 2k] + f[2i, 2j - 1, 2k] +
                              f[2i - 1, 2j, 2k] + f[2i, 2j, 2k])
    end
    c
end

function _prolongate!(f, c)
    cnx, cny, cnz = size(c)
    @inbounds for k in 1:cnz, j in 1:cny, i in 1:cnx
        v = c[i, j, k]
        f[2i - 1, 2j - 1, 2k - 1] = v
        f[2i, 2j - 1, 2k - 1] = v
        f[2i - 1, 2j, 2k - 1] = v
        f[2i, 2j, 2k - 1] = v
        f[2i - 1, 2j - 1, 2k] = v
        f[2i, 2j - 1, 2k] = v
        f[2i - 1, 2j, 2k] = v
        f[2i, 2j, 2k] = v
    end
    f
end

function _vcycle!(x, b, stack, lev, nlev, pre, post, coarse_sweeps)
    if lev == nlev
        _smooth!(x, b, stack[lev][1], coarse_sweeps)
        return x
    end
    lvl = stack[lev][1]
    ax, ay, az, vol, dx, dy, dz, fixed, fixedvals = lvl
    _smooth!(x, b, lvl, pre)

    r = similar(x)
    apply7!(r, x, ax, ay, az, vol, dx, dy, dz, fixed)
    @inbounds for idx in eachindex(r)
        rv = vec(r)
        break
    end
    rvv = vec(r)
    bv = vec(b)
    @inbounds for idx in eachindex(rvv)
        rvv[idx] = bv[idx] - rvv[idx]
    end

    # The coarse problem is A_c e = R(r): the coarse RIGHT-HAND SIDE is the restricted
    # RESIDUAL, not the restricted source. Restricting `b` instead only coincides with
    # the correct thing when x is exactly zero, and quietly produces a divergent cycle
    # from the first non-trivial cycle onwards.
    clvl = stack[lev + 1][1]
    cnx, cny, cnz = size(clvl[4])
    ec = zeros(eltype(x), cnx, cny, cnz)
    _restrict!(ec, r)
    _vcycle!(ec, ec, stack, lev + 1, nlev, pre, post, coarse_sweeps)

    corr = zeros(eltype(x), size(x))
    _prolongate!(corr, ec)
    xv = vec(x); cv = vec(corr)
    @inbounds @simd for idx in eachindex(xv)
        xv[idx] += cv[idx]
    end
    _smooth!(x, b, lvl, post)
    x
end