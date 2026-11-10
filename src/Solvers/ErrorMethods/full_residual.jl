"""
    FullResidual <: SolverError

A `SolverError` structure for computing the norm of the full residual, ``b-Ax``.

# Fields
- None
"""
struct FullResidual <: SolverError end

"""
    FullResidualRecipe{V<:AbstractVector} <: SolverErrorRecipe

A `SolverErrorRecipe` structure for computing the norm of the full residual, ``b-Ax``.

# Fields
- `residual::V`, a container for the residual ``b-Ax``.
"""
mutable struct FullResidualRecipe{V<:AbstractVector} <: SolverErrorRecipe
    residual::V
end

function complete_error(
    error::FullResidual,
    solver::Solver, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    # Type the recipe by the allocated buffer, not by `b`, which may be a view
    residual = zeros(eltype(b), size(b, 1))
    return FullResidualRecipe{typeof(residual)}(residual)
end

function compute_error(
        error::FullResidualRecipe, 
        solver::SolverRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )::Float64
    copyto!(error.residual, b)
    mul!(error.residual, A, solver.solution_vec, -1.0, 1.0)
    return norm(error.residual)
end
