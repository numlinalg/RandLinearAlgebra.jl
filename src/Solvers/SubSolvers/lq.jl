"""
    LQSolver <: SubSolver

A type containing information relevant to solving the linear subsystems created by the
    Solver routines with the LQ factorization. As there are no user controlled parameters,
    if the user wishes to use this method they can simply specify `LQSolver()`.
"""
struct LQSolver <: SubSolver end

"""
    LQSolverRecipe{M<:AbstractArray} <: SubSolverRecipe

A mutable type containing information relevant to solving the linear subsystems created by
    the Solver routines with the LQ factorization, using pre-allocated LAPACK workspace.
    Internally stores the transpose of the matrix to perform QR on `A'`.

# Fields
- `A::M`, a buffer storing `A'` (destroyed by `geqrt3!`).
- `T::Matrix`, the block reflector workspace for `geqrt3!`.
"""
mutable struct LQSolverRecipe{M<:AbstractArray, V<:AbstractVector} <: SubSolverRecipe
    A::M
    T::Matrix
    work::V
end

function complete_sub_solver(solver::LQSolver, A::AbstractMatrix)
    k = min(size(A)...)
    T = zeros(eltype(A), k, k)
    work = zeros(eltype(A), size(A, 1))
    return LQSolverRecipe{Matrix{eltype(A)}, typeof(work)}(Matrix(A'), T, work)
end

function update_sub_solver!(solver::LQSolverRecipe, A::AbstractMatrix)
    copyto!(solver.A, A')
    return nothing
end

function ldiv!(
    x::AbstractVector,
    solver::LQSolverRecipe{<:AbstractMatrix},
    b::AbstractVector,
)
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
