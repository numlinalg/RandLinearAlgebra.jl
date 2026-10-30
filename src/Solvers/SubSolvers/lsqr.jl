"""
    NoPreconditioner

A singleton type indicating that no preconditioner is applied. This is the default
    preconditioner for the `LSQRSolver` and causes the solver to dispatch to an
    implementation that performs no preconditioner solves and allocates no additional
    buffers.

# Fields
- None
"""
struct NoPreconditioner end

"""
    LSQRSolver{P} <: KrylovSubSolver

A type containing information relevant to solving the linear subsystems created by the
    Solver routines with the LSQR method [paige1982lsqr](@cite).

# Mathematical Description
Let ``A`` be an ``m \\times n`` matrix and ``b`` an ``m``-dimensional vector. LSQR
    computes the minimum-norm solution of ``\\min_x \\|Ax - b\\|_2`` by applying
    Golub-Kahan bidiagonalization to ``A`` starting from ``b``. After ``k`` steps this
    produces orthonormal bases ``U_{k+1}`` and ``V_k`` with ``A V_k = U_{k+1} B_k``, where
    ``B_k`` is a ``(k+1) \\times k`` lower bidiagonal matrix. The iterate
    ``x_k = V_k y_k`` is then chosen with ``y_k`` minimizing
    ``\\|\\beta_1 e_1 - B_k y\\|_2``, which is updated cheaply with Givens rotations.
    The method is mathematically equivalent to conjugate gradient applied to the normal
    equations, but is more numerically stable.

If a left preconditioner ``N`` is supplied through the `precond` field, the
    bidiagonalization is applied to ``\\hat A = N^{-1} A`` starting from
    ``\\hat b = N^{-1} b`` instead, so that the method computes
```math
    \\min_x \\|N^{-1}(Ax - b)\\|_2.
```
The preconditioner is applied through `ldiv!`, so it is the inverse of ``N`` that acts on
    the system and ``N`` itself that should approximate ``A``. Because the preconditioner
    is applied from the left, the iterates are produced directly in the coordinates of
    ``x`` and no back transformation is required.

# Fields
- `max_it::Int64`, the maximum number of LSQR iterations.
- `atol::Float64`, the tolerance on the relative error in the matrix ``A``.
- `btol::Float64`, the tolerance on the relative error in the vector ``b``.
- `zero_init::Bool`, whether the solution vector supplied to `ldiv!` is set to zero before
    the iteration begins. If `false` the vector is used as the initial iterate ``x_0``.
- `precond::P`, the left preconditioner ``N``, whose inverse is applied to the system, or
    `NoPreconditioner()` if none is applied.

# Constructor
    LSQRSolver(;
        max_it::Int64 = 0,
        atol::Float64 = 1e-8,
        btol::Float64 = 1e-8,
        zero_init::Bool = true,
        precond = NoPreconditioner()
    )

## Keywords
- `max_it::Int64`, the maximum number of LSQR iterations. If set to `0` then the number of
    columns of the matrix is used. By default this value is `0`.
- `atol::Float64`, the tolerance on the relative error in the matrix ``A``. By default
    this value is `1e-8`.
- `btol::Float64`, the tolerance on the relative error in the vector ``b``. By default
    this value is `1e-8`.
- `zero_init::Bool`, whether the solution vector supplied to `ldiv!` is set to zero before
    the iteration begins. By default this value is `true`.
- `precond`, the left preconditioner ``N``. Any object `N` may be supplied provided that
    it implements the two in-place solves `ldiv!(y, N, x)`, which computes
    ``y = N^{-1} x``, and `ldiv!(y, N', x)`, which computes ``y = N^{-*} x``. The
    factorization and structured matrix types of `LinearAlgebra`, such as `Diagonal`,
    `UpperTriangular`, `lu`, `qr` and `cholesky`, all satisfy this requirement. An
    unwrapped matrix does not and must be wrapped in one of them. By default no
    preconditioner is applied.

## Returns
- A `LSQRSolver` object.

# Throws
- `ArgumentError` if `max_it` is negative, or if `atol` or `btol` are negative.
- `ArgumentError` from `complete_sub_solver` if `precond` is not a square operator with as
    many rows as the coefficient matrix, or if it does not implement the two in-place
    solves listed above.

!!! info
    The iteration terminates when either
    ``\\|\\hat r_k\\|_2 \\leq \\text{btol} \\|\\hat b\\|_2
    + \\text{atol} \\|\\hat A\\|_F \\|d_k\\|_2``
    (a compatible system has been solved) or
    ``\\|\\hat A^\\top \\hat r_k\\|_2 \\leq
    \\text{atol} \\|\\hat A\\|_F \\|\\hat r_k\\|_2`` (a least squares
    solution has been found), where ``\\|\\hat A\\|_F`` is estimated from the
    bidiagonalization and ``d_k = x_k - x_0`` is the correction computed by the
    iteration. Without a preconditioner ``\\hat A = A`` and ``\\hat b = b``, so these are
    the usual criteria. With a preconditioner every quantity is measured in the
    ``N^{-1}``-weighted norm, and the tolerances should be interpreted accordingly.

!!! note
    A left preconditioner changes the quantity being minimized from
    ``\\|Ax - b\\|_2`` to ``\\|N^{-1}(Ax - b)\\|_2``. These have the same set of
    solutions
    when the system is consistent, or when ``N`` is unitary. For an
    inconsistent least squares problem the two minimizers differ, and ``N`` must be chosen
    with this in mind.

!!! note
    The vector `b` supplied to `ldiv!` is always the full constant vector of the system
    and never a residual. When `zero_init` is `false` the initial residual
    ``b - A x_0`` is formed internally from the incoming solution vector.
"""
struct LSQRSolver{P} <: KrylovSubSolver
    max_it::Int64
    atol::Float64
    btol::Float64
    zero_init::Bool
    precond::P
    function LSQRSolver(max_it, atol, btol, zero_init, precond::P) where {P}
        if max_it < 0
            throw(ArgumentError("Field `max_it` must be non-negative."))
        elseif atol < 0
            throw(ArgumentError("Field `atol` must be non-negative."))
        elseif btol < 0
            throw(ArgumentError("Field `btol` must be non-negative."))
        end

        return new{P}(max_it, atol, btol, zero_init, precond)
    end

end

function LSQRSolver(;
    max_it::Int64 = 0,
    atol::Float64 = 1e-8,
    btol::Float64 = 1e-8,
    zero_init::Bool = true,
    precond = NoPreconditioner()
)
    return LSQRSolver(max_it, atol, btol, zero_init, precond)
end

"""
    LSQRSolverRecipe{
        T<:Number,
        V<:AbstractVector,
        M<:AbstractArray,
        P
    } <: KrylovSubSolverRecipe

A mutable type containing information relevant to solving the linear subsystems created by
    the Solver routines with the LSQR method.

# Fields
- `A::M`, the matrix in the linear system that will be solved with the LSQR solver.
- `max_it::Int64`, the maximum number of LSQR iterations. If `0` then the number of
    columns of `A` is used.
- `atol::Float64`, the tolerance on the relative error in the matrix ``A``.
- `btol::Float64`, the tolerance on the relative error in the vector ``b``.
- `zero_init::Bool`, whether the solution vector supplied to `ldiv!` is set to zero before
    the iteration begins. If `false` the vector is used as the initial iterate ``x_0``.
- `u::V`, a buffer for the left Lanczos vector, of length equal to the rows of `A`.
- `v::V`, a buffer for the right Lanczos vector, of length equal to the columns of `A`.
- `w::V`, a buffer for the search direction, of length equal to the columns of `A`.
- `precond::P`, the left preconditioner ``N``, whose inverse is applied to the system, or
    `NoPreconditioner()` if none is applied.
- `t::V`, a buffer for the preconditioner solves, of length equal to the rows of `A`. It
    has length zero when no preconditioner is applied.
- `s::V`, a second buffer for the preconditioner solves, of length equal to the rows of
    `A`. It has length zero when no preconditioner is applied.
"""
mutable struct LSQRSolverRecipe{
    T<:Number,
    V<:AbstractVector,
    M<:AbstractArray,
    P
} <: KrylovSubSolverRecipe
    A::M
    max_it::Int64
    atol::Float64
    btol::Float64
    zero_init::Bool
    u::V
    v::V
    w::V
    precond::P
    t::V
    s::V
end

# An alias for the recipes that apply no preconditioner. It carries the field bounds
# declared by `LSQRSolverRecipe` so that the kernels specialized on it are strictly more
# specific than their general counterparts.
const LSQRNoPrecRecipe = LSQRSolverRecipe{
    <:Number,
    <:AbstractVector,
    <:AbstractArray,
    NoPreconditioner
}

# Verify that a preconditioner can be applied to vectors of type `V` from the left and
# that it has dimensions consistent with an `m` by `n` coefficient matrix.
function check_preconditioner(precond, m::Int, ::Type{V}) where {V<:AbstractVector}
    if hasmethod(size, Tuple{typeof(precond)})
        dims = size(precond)
        if length(dims) != 2 || dims[1] != m || dims[2] != m
            throw(
                ArgumentError(
                    "Field `precond` must be a square operator with $m rows and columns \
                    to be applied to a matrix with $m rows, but has size $dims."
                )
            )
        end

    end

    check_left_solve(precond, precond, "precond", V)
    if !hasmethod(adjoint, Tuple{typeof(precond)})
        throw(
            ArgumentError(
                "Field `precond` of type $(typeof(precond)) must implement `adjoint` so \
                that the adjoint of the preconditioner can be applied."
            )
        )
    end

    check_left_solve(precond, adjoint(precond), "precond'", V)
    return nothing
end

# Verify that `operator`, which is either the preconditioner or its adjoint, can be
# applied to vectors of type `V` through the three-argument `ldiv!`.
function check_left_solve(
    precond,
    operator,
    label::String,
    ::Type{V}
) where {V<:AbstractVector}
    if !hasmethod(ldiv!, Tuple{V, typeof(operator), V})
        throw(
            ArgumentError(
                "Field `precond` of type $(typeof(precond)) must implement \
                `ldiv!(y::$V, $label, x::$V)`."
            )
        )
    end

    # The generic three-argument `ldiv!` of `LinearAlgebra` forwards to the two-argument
    # form, so a type that matches only that fallback and provides no two-argument method,
    # such as an unwrapped matrix, would pass the check above but fail when applied.
    fallback = which(ldiv!, Tuple{V, AbstractMatrix, V})
    if which(ldiv!, Tuple{V, typeof(operator), V}) === fallback &&
        !hasmethod(ldiv!, Tuple{typeof(operator), V})
        throw(
            ArgumentError(
                "Field `precond` of type $(typeof(precond)) only matches the generic \
                `ldiv!` fallback of `LinearAlgebra` and cannot be applied to a vector. \
                Wrap it in a factorization or structured matrix type, such as `lu`, \
                `qr`, `cholesky`, `UpperTriangular` or `Diagonal`."
            )
        )
    end

    return nothing
end

check_preconditioner(::NoPreconditioner, m::Int, ::Type{V}) where {V<:AbstractVector} =
    nothing

# The buffers for the preconditioner solves are only required when a preconditioner is
# applied, so they are given length zero otherwise.
preconditioner_buffer_size(::NoPreconditioner, m::Int) = 0
preconditioner_buffer_size(precond, m::Int) = m

function complete_sub_solver(solver::LSQRSolver, A::AbstractMatrix)
    T = eltype(A)
    m, n = size(A)
    u = zeros(T, m)
    v = zeros(T, n)
    w = zeros(T, n)
    check_preconditioner(solver.precond, m, typeof(u))
    work_size = preconditioner_buffer_size(solver.precond, m)
    t = zeros(T, work_size)
    s = zeros(T, work_size)
    return LSQRSolverRecipe{T, typeof(u), typeof(A), typeof(solver.precond)}(
        A,
        solver.max_it,
        solver.atol,
        solver.btol,
        solver.zero_init,
        u,
        v,
        w,
        solver.precond,
        t,
        s
    )
end

# Resize the preconditioner buffers and revalidate the preconditioner against the new row
# dimension. This is only called when the number of rows of the matrix has changed.
update_work_buffers!(solver::LSQRNoPrecRecipe, m::Int) = nothing

function update_work_buffers!(solver::LSQRSolverRecipe, m::Int)
    check_preconditioner(solver.precond, m, typeof(solver.u))
    resize!(solver.t, m)
    resize!(solver.s, m)
    return nothing
end

function update_sub_solver!(solver::LSQRSolverRecipe, A::AbstractMatrix)
    solver.A = A
    m, n = size(A)
    # Only resize the buffers if the dimensions of the new matrix have changed
    if length(solver.u) != m
        resize!(solver.u, m)
        update_work_buffers!(solver, m)
    end

    if length(solver.v) != n
        resize!(solver.v, n)
        resize!(solver.w, n)
    end

    return nothing
end

###########################################
# Bidiagonalization kernels
#
# These are the only steps of the iteration that the preconditioner enters, so they are
# the only ones specialized on the preconditioner type. The iteration itself is shared.
###########################################

# Form the initial residual, normalize it into the left Lanczos vector, and return the
# norms of the initial residual and of the (preconditioned) constant vector.
@inline function init_system!(
    solver::LSQRNoPrecRecipe,
    x::AbstractVector,
    b::AbstractVector
)
    T = eltype(solver.A)
    u = solver.u
    if solver.zero_init
        fill!(x, zero(eltype(x)))
        copyto!(u, b)
    else
        # u = b - A * x
        copyto!(u, b)
        mul!(u, solver.A, x, -one(T), one(T))
    end

    beta = norm(u)
    beta > 0 && (u ./= beta)
    return beta, norm(b)
end

@inline function init_system!(
    solver::LSQRSolverRecipe,
    x::AbstractVector,
    b::AbstractVector
)
    T = eltype(solver.A)
    u = solver.u
    t = solver.t
    if solver.zero_init
        fill!(x, zero(eltype(x)))
        # u = N \ b, which is both the initial residual and the vector whose norm the
        # `btol` criterion is relative to
        ldiv!(u, solver.precond, b)
        bnorm = norm(u)
    else
        # u = N \ (b - A * x)
        copyto!(t, b)
        mul!(t, solver.A, x, -one(T), one(T))
        ldiv!(u, solver.precond, t)
        ldiv!(t, solver.precond, b)
        bnorm = norm(t)
    end

    beta = norm(u)
    beta > 0 && (u ./= beta)
    return beta, bnorm
end

# Continue the bidiagonalization in the left Lanczos vector, beta * u = A * v - alpha * u,
# and return the resulting beta.
@inline function bidiag_u!(solver::LSQRNoPrecRecipe, alpha::Real)
    T = eltype(solver.A)
    u = solver.u
    mul!(u, solver.A, solver.v, one(T), -alpha)
    beta = norm(u)
    beta > 0 && (u ./= beta)
    return beta
end

@inline function bidiag_u!(solver::LSQRSolverRecipe, alpha::Real)
    T = eltype(solver.A)
    u = solver.u
    t = solver.t
    s = solver.s
    # u = N \ (A * v) - alpha * u, which needs two buffers because the previous value of
    # u is required after the product has been formed
    mul!(t, solver.A, solver.v)
    ldiv!(s, solver.precond, t)
    axpby!(one(T), s, -alpha, u)
    beta = norm(u)
    beta > 0 && (u ./= beta)
    return beta
end

# Continue the bidiagonalization in the right Lanczos vector,
# alpha * v = A' * u - beta * v, and return the resulting alpha. A `beta` of zero
# overwrites `v` and is used to initialize the iteration.
@inline function bidiag_v!(solver::LSQRNoPrecRecipe, beta::Real)
    T = eltype(solver.A)
    v = solver.v
    mul!(v, solver.A', solver.u, one(T), -beta)
    alpha = norm(v)
    alpha > 0 && (v ./= alpha)
    return alpha
end

@inline function bidiag_v!(solver::LSQRSolverRecipe, beta::Real)
    T = eltype(solver.A)
    v = solver.v
    # v = A' * (N' \ u) - beta * v
    ldiv!(solver.t, adjoint(solver.precond), solver.u)
    mul!(v, solver.A', solver.t, one(T), -beta)
    alpha = norm(v)
    alpha > 0 && (v ./= alpha)
    return alpha
end

function ldiv!(
    x::AbstractVector,
    solver::LSQRSolverRecipe{<:Number, <:AbstractVector, <:AbstractMatrix},
    b::AbstractVector
)
    T = eltype(solver.A)
    v = solver.v
    w = solver.w
    atol = solver.atol
    btol = solver.btol
    max_it = solver.max_it == 0 ? size(solver.A, 2) : solver.max_it
    # Initialize the bidiagonalization: beta_1 u_1 = b - A x_0, alpha_1 v_1 = A' u_1
    beta, bnorm = init_system!(solver, x, b)
    beta == 0 && return nothing
    alpha = bidiag_v!(solver, zero(real(T)))
    alpha == 0 && return nothing
    copyto!(w, v)
    phibar = beta
    rhobar = alpha
    # Frobenius norm estimate of A built from the bidiagonal entries
    anorm_sq = zero(alpha)
    # Squared norm of the correction x_k - x_0, which equals the squared norm of the
    # coordinates of the iterate in the orthonormal basis V_k
    dnorm_sq = zero(alpha)
    for _ in 1:max_it
        # Continue the bidiagonalization: beta u = A v - alpha u
        beta = bidiag_u!(solver, alpha)
        anorm_sq += alpha^2 + beta^2
        # alpha v = A' u - beta v
        alpha = bidiag_v!(solver, beta)
        # Apply a Givens rotation to eliminate the subdiagonal of the bidiagonal matrix
        rho = hypot(rhobar, beta)
        c = rhobar / rho
        s = beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        # Update the solution and the search direction
        zeta = phi / rho
        x .+= zeta .* w
        w .= v .- (theta / rho) .* w
        dnorm_sq += zeta^2
        # Check the stopping criteria
        anorm = sqrt(anorm_sq)
        rnorm = phibar
        arnorm = phibar * alpha * abs(c)
        if rnorm <= btol * bnorm + atol * anorm * sqrt(dnorm_sq) ||
            arnorm <= atol * anorm * rnorm ||
            alpha == 0
            break
        end

    end

    return nothing
end
