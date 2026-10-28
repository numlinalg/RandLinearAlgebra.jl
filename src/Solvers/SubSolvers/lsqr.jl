"""
    LSQRSolver <: KrylovSubSolver

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

# Fields
- `max_it::Int64`, the maximum number of LSQR iterations.
- `atol::Float64`, the tolerance on the relative error in the matrix ``A``.
- `btol::Float64`, the tolerance on the relative error in the vector ``b``.

# Constructor
    LSQRSolver(;max_it::Int64 = 0, atol::Float64 = 1e-8, btol::Float64 = 1e-8)

## Keywords
- `max_it::Int64`, the maximum number of LSQR iterations. If set to `0` then the number of
    columns of the matrix is used. By default this value is `0`.
- `atol::Float64`, the tolerance on the relative error in the matrix ``A``. By default
    this value is `1e-8`.
- `btol::Float64`, the tolerance on the relative error in the vector ``b``. By default
    this value is `1e-8`.

## Returns
- A `LSQRSolver` object.

# Throws
- `ArgumentError` if `max_it` is negative, or if `atol` or `btol` are negative.

!!! info
    The iteration terminates when either
    ``\\|r_k\\|_2 \\leq \\text{btol} \\|b\\|_2 + \\text{atol} \\|A\\|_F \\|x_k\\|_2``
    (a compatible system has been solved) or
    ``\\|A^\\top r_k\\|_2 \\leq \\text{atol} \\|A\\|_F \\|r_k\\|_2`` (a least squares
    solution has been found), where ``\\|A\\|_F`` is estimated from the bidiagonalization.
"""
struct LSQRSolver <: KrylovSubSolver
    max_it::Int64
    atol::Float64
    btol::Float64
    function LSQRSolver(max_it, atol, btol)
        if max_it < 0
            throw(ArgumentError("Field `max_it` must be non-negative."))
        elseif atol < 0
            throw(ArgumentError("Field `atol` must be non-negative."))
        elseif btol < 0
            throw(ArgumentError("Field `btol` must be non-negative."))
        end

        return new(max_it, atol, btol)
    end

end

function LSQRSolver(; max_it::Int64 = 0, atol::Float64 = 1e-8, btol::Float64 = 1e-8)
    return LSQRSolver(max_it, atol, btol)
end

"""
    LSQRSolverRecipe{T<:Number, V<:AbstractVector, M<:AbstractArray} <: KrylovSubSolverRecipe

A mutable type containing information relevant to solving the linear subsystems created by
    the Solver routines with the LSQR method.

# Fields
- `A::M`, the matrix in the linear system that will be solved with the LSQR solver.
- `max_it::Int64`, the maximum number of LSQR iterations. If `0` then the number of
    columns of `A` is used.
- `atol::Float64`, the tolerance on the relative error in the matrix ``A``.
- `btol::Float64`, the tolerance on the relative error in the vector ``b``.
- `u::V`, a buffer for the left Lanczos vector, of length equal to the rows of `A`.
- `v::V`, a buffer for the right Lanczos vector, of length equal to the columns of `A`.
- `w::V`, a buffer for the search direction, of length equal to the columns of `A`.
"""
mutable struct LSQRSolverRecipe{
    T<:Number,
    V<:AbstractVector,
    M<:AbstractArray
} <: KrylovSubSolverRecipe
    A::M
    max_it::Int64
    atol::Float64
    btol::Float64
    u::V
    v::V
    w::V
end

function complete_sub_solver(solver::LSQRSolver, A::AbstractMatrix)
    T = eltype(A)
    m, n = size(A)
    u = zeros(T, m)
    v = zeros(T, n)
    w = zeros(T, n)
    return LSQRSolverRecipe{T, typeof(u), typeof(A)}(
        A,
        solver.max_it,
        solver.atol,
        solver.btol,
        u,
        v,
        w
    )
end

function update_sub_solver!(solver::LSQRSolverRecipe, A::AbstractMatrix)
    solver.A = A
    m, n = size(A)
    # Only resize the buffers if the dimensions of the new matrix have changed
    length(solver.u) != m && resize!(solver.u, m)
    if length(solver.v) != n
        resize!(solver.v, n)
        resize!(solver.w, n)
    end

    return nothing
end

function ldiv!(
    x::AbstractVector,
    solver::LSQRSolverRecipe{<:Number, <:AbstractVector, <:AbstractMatrix},
    b::AbstractVector
)
    A = solver.A
    u = solver.u
    v = solver.v
    w = solver.w
    atol = solver.atol
    btol = solver.btol
    max_it = solver.max_it == 0 ? size(A, 2) : solver.max_it
    # Initialize the bidiagonalization: beta_1 u_1 = b, alpha_1 v_1 = A' u_1
    copyto!(u, b)
    beta = norm(u)
    bnorm = beta
    beta == 0 && return nothing
    u ./= beta
    mul!(v, A', u)
    alpha = norm(v)
    alpha == 0 && return nothing
    v ./= alpha
    copyto!(w, v)
    phibar = beta
    rhobar = alpha
    # Frobenius norm estimate of A built from the bidiagonal entries
    anorm_sq = zero(alpha)
    for _ in 1:max_it
        # Continue the bidiagonalization: beta u = A v - alpha u
        mul!(u, A, v, one(eltype(u)), -alpha)
        beta = norm(u)
        beta > 0 && (u ./= beta)
        anorm_sq += alpha^2 + beta^2
        # alpha v = A' u - beta v
        mul!(v, A', u, one(eltype(v)), -beta)
        alpha = norm(v)
        alpha > 0 && (v ./= alpha)
        # Apply a Givens rotation to eliminate the subdiagonal of the bidiagonal matrix
        rho = hypot(rhobar, beta)
        c = rhobar / rho
        s = beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        # Update the solution and the search direction
        x .+= (phi / rho) .* w
        w .= v .- (theta / rho) .* w
        # Check the stopping criteria
        anorm = sqrt(anorm_sq)
        rnorm = phibar
        arnorm = phibar * alpha * abs(c)
        if rnorm <= btol * bnorm + atol * anorm * norm(x) ||
            arnorm <= atol * anorm * rnorm ||
            alpha == 0
            break
        end

    end

    return nothing
end
