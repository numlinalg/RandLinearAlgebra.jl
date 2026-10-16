"""
    LQSolver <: SubSolver

A type containing information relevant to solving the linear subsystems created by the 
    Solver routines with the LQ factorization. As there are no user controlled parameters, 
    if the user wishes to use this method they can simply specify `LQSolver()`.
"""
struct LQSolver <: SubSolver

end

"""
    LQSolverRecipe <: SubSolverRecipe{M<:AbstractArray}

A mutable type containing informtation relevant to solving the linear subsytems created by 
    the Solver routines with the LQ factorization.

# Fields
 - `A::M`, The matrix in the linear system that will be solved with the LQ solver.
"""
mutable struct LQSolverRecipe{M<:AbstractArray} <: SubSolverRecipe
    A::M
end

function complete_sub_solver(solver::LQSolver, A::AbstractMatrix)
    return LQSolverRecipe{typeof(A)}(A) 
end

function update_sub_solver!(solver::LQSolverRecipe, A::AbstractMatrix)
    # This will overwrite the block of the matrix so do not reuse the block values
    solver.A = A
    return nothing
end

function ldiv!(
    x::AbstractVector,
    solver::LQSolverRecipe{<:AbstractMatrix},
    b::AbstractVector
)
    fill!(x, zero(eltype(b)))
    # Note: qr! (in-place) is used here for efficiency on CPU. GPU arrays do not support
    # qr! — GPU block Kaczmarz/ColumnProjection requires a follow-up PR.
    # using qr here on the transpose of the matrix will work for sparse and dense matrices
    # while the lq would have only worked for dense matrices
    ldiv!(x, qr!(solver.A')', b)
    return nothing
end
