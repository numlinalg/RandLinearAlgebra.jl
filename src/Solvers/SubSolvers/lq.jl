"""
    LQSolver <: SubSolver

A type containing information relevant to solving the linear subsystems created by the
    Solver routines with the LQ factorization. As there are no user controlled parameters,
    if the user wishes to use this method they can simply specify `LQSolver()`.
"""
struct LQSolver <: SubSolver end

"""
    LQSolverRecipe{M<:AbstractMatrix, V<:AbstractVector, MT<:AbstractMatrix} <: SubSolverRecipe

A mutable type containing information relevant to solving the linear subsystems created by
    the Solver routines with the LQ factorization, using pre-allocated LAPACK workspace.
    Internally stores the transpose of the matrix to perform QR on `A'`.

# Fields
- `A::M`, a buffer storing `A'` (destroyed by the factorization in `ldiv!`).
- `T::MT`, the block reflector workspace for `geqrt3!`.
- `work::V`, a workspace holding the right-hand side while it is transformed.

!!! info
    For the BLAS element types (`Float32`, `Float64`, `ComplexF32` and `ComplexF64`),
    with `x` of the same element type, `ldiv!` uses the LAPACK routines `geqrt3!`,
    `trtrs!` and `gemqrt!` on the preallocated workspace and does not allocate. For
    other element types (e.g., `BigFloat` or `Float16`), it falls back to the generic
    `qr!`, which allocates the factorization.
"""
mutable struct LQSolverRecipe{
    M<:AbstractMatrix, V<:AbstractVector, MT<:AbstractMatrix
} <: SubSolverRecipe
    A::M
    T::MT
    work::V
end

function complete_sub_solver(solver::LQSolver, A::AbstractMatrix)
    k = min(size(A)...)
    T = zeros(eltype(A), k, k)
    work = zeros(eltype(A), size(A, 1))
    At = Matrix(A')
    return LQSolverRecipe{typeof(At), typeof(work), typeof(T)}(At, T, work)
end

function update_sub_solver!(solver::LQSolverRecipe, A::AbstractMatrix)
    copyto!(solver.A, A')
    return nothing
end

# Allocation-free LAPACK path for BLAS element types
function ldiv!(
    x::StridedVector{F},
    solver::LQSolverRecipe{Matrix{F}},
    b::AbstractVector,
) where {F<:BlasFloat}
    n, m = size(solver.A)
    trans_char = _qt_char(eltype(solver.A))
    copyto!(solver.work, b)
    @inbounds LAPACK.geqrt3!(solver.A, solver.T)
    @inbounds LAPACK.trtrs!('U', trans_char, 'N', @view(solver.A[1:m, :]), solver.work)
    fill!(x, zero(eltype(x)))
    copyto!(@view(x[1:m]), solver.work)
    @inbounds LAPACK.gemqrt!('L', 'N', solver.A, solver.T, x)
    return nothing
end

# Generic fallback for element types not supported by LAPACK (e.g., BigFloat, Float16).
# solver.A stores A', so the adjoint of its QR factorization is a factorization of A.
function ldiv!(
    x::AbstractVector,
    solver::LQSolverRecipe{<:AbstractMatrix},
    b::AbstractVector,
)
    # this overwrites solver.A, as the LAPACK path does
    ldiv!(x, qr!(solver.A)', b)
    return nothing
end
