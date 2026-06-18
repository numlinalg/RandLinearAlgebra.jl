"""
    LSGradient <: SolverError

A `SolverError` structure for monitoring the residual of the normal equations 
    of a linear system (equivalently, the gradient of the least squares problem),
    ``-A^\\top (b - Ax)``.

# Fields
- None
"""
struct LSGradient <: SolverError end

"""
    LSGradientRecipe <: SolverErrorRecipe

A `SolverErrorRecipe` structure for storing the residual of the normal equations 
    of a linear system (equivalently, the gradeint of the least squares problem),
    ``-A^\\intercal (b - Ax)``.

# Fields
- `gradient::AbstractVector`, ``-A^\\top (b - Ax)``.
"""
mutable struct LSGradientRecipe{V<:AbstractVector} <: SolverErrorRecipe
    gradient::V
end

function complete_error(
    error::LSGradient, 
    solver::Solver, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    gradient = fill!(similar(A, eltype(A), size(A,2)), zero(eltype(A)))
    return LSGradientRecipe{typeof(gradient)}(gradient)
end

function compute_error(
    error::LSGradientRecipe, 
    solver::SolverRecipe, 
    A::AbstractMatrix, 
    b::AbstractVector
)::Float64
    # solver.residual_vec = b - A x
    # error.gradient <-- 0*error.gradient + (-1.0)*A'(solver.residual_vec) 
    #                <-- -A'(b - Ax)
    mul!(error.gradient, A', solver.residual_vec, -1.0, 0.0)
    return norm(error.gradient)
end
