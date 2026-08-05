"""
    QRSolver <: SubSolver

A type containing information relevant to solving the linear subsystems created by the
    Solver routines with the QR factorization. As there are no user controlled parameters,
    if the user wishes to use this method they can simply specify `QRSolver()`.

# Fields
-  None
"""
struct QRSolver <: SubSolver end

"""
    QRSolverRecipe{M<:AbstractArray} <: SubSolverRecipe

A mutable type containing information relevant to solving the linear subsystems created by
    the Solver routines with the QR factorization, using pre-allocated LAPACK workspace.

# Fields
- `A::M`, a working copy of the matrix (destroyed by `geqrt3!`).
- `T::Matrix`, the block reflector workspace for `geqrt3!`.
"""
mutable struct QRSolverRecipe{M<:AbstractArray, V<:AbstractVector} <: SubSolverRecipe
    A::M
    T::Matrix
    work::V
end

function complete_sub_solver(solver::QRSolver, A::AbstractMatrix)
    k = min(size(A)...)
    T = zeros(eltype(A), k, k)
    work = zeros(eltype(A), size(A, 1))
    return QRSolverRecipe{Matrix{eltype(A)}, typeof(work)}(Matrix(A), T, work)
end

function update_sub_solver!(solver::QRSolverRecipe, A::AbstractMatrix)
    copyto!(solver.A, A)
    return nothing
end

function ldiv!(
    x::AbstractVector,
    solver::QRSolverRecipe{<:AbstractMatrix},
    b::AbstractVector,
)
    m, n = size(solver.A)
    copyto!(solver.work, b)
    @inbounds LAPACK.geqrt3!(solver.A, solver.T)
    @inbounds LAPACK.gemqrt!(
        'L', _qt_char(eltype(solver.A)), solver.A, solver.T, solver.work
    )
    @inbounds LAPACK.trtrs!(
        'U', 'N', 'N', @view(solver.A[1:n, :]), @view(solver.work[1:n])
    )
    copyto!(x, @view(solver.work[1:n]))
    return nothing
end
