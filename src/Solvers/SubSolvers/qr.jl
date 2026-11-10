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
    QRSolverRecipe{M<:AbstractMatrix, V<:AbstractVector, MT<:AbstractMatrix} <: SubSolverRecipe

A mutable type containing information relevant to solving the linear subsystems created by
    the Solver routines with the QR factorization, using pre-allocated LAPACK workspace.

# Fields
- `A::M`, a working copy of the matrix (destroyed by the factorization in `ldiv!`).
- `T::MT`, the block reflector workspace for `geqrt3!`.
- `work::V`, a workspace holding the right-hand side while it is transformed.

!!! info
    For the BLAS element types (`Float32`, `Float64`, `ComplexF32` and `ComplexF64`),
    with `x` of the same element type, `ldiv!` uses the LAPACK routines `geqrt3!`,
    `gemqrt!` and `trtrs!` on the preallocated workspace and does not allocate. For
    other element types (e.g., `BigFloat` or `Float16`), it falls back to the generic
    `qr!`, which allocates the factorization.
"""
mutable struct QRSolverRecipe{
    M<:AbstractMatrix, V<:AbstractVector, MT<:AbstractMatrix
} <: SubSolverRecipe
    A::M
    T::MT
    work::V
end

function complete_sub_solver(solver::QRSolver, A::AbstractMatrix)
    k = min(size(A)...)
    T = zeros(eltype(A), k, k)
    work = zeros(eltype(A), size(A, 1))
    A_copy = Matrix(A)
    return QRSolverRecipe{typeof(A_copy), typeof(work), typeof(T)}(A_copy, T, work)
end

function update_sub_solver!(solver::QRSolverRecipe, A::AbstractMatrix)
    copyto!(solver.A, A)
    return nothing
end

# Allocation-free LAPACK path for BLAS element types
function ldiv!(
    x::StridedVector{F},
    solver::QRSolverRecipe{Matrix{F}},
    b::AbstractVector,
) where {F<:BlasFloat}
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

# Generic fallback for element types not supported by LAPACK (e.g., BigFloat, Float16)
function ldiv!(
    x::AbstractVector,
    solver::QRSolverRecipe{<:AbstractMatrix},
    b::AbstractVector,
)
    # this overwrites solver.A, as the LAPACK path does
    ldiv!(x, qr!(solver.A), b)
    return nothing
end
