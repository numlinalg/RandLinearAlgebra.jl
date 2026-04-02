"""
    CompressedResidual <: ErrorRecipe

A structure for the compressed residual, `Sb-SAx`.

# Fields
- None
"""
struct CompressedResidual <: SolverError end

"""
    CompressedResidualRecipe <: ErrorRecipe

A structure containing the preallocated structures for the compressed residual, `Sb-SAx`.

# Fields
- `residual::AbstractVector`, a container for the compressed residual, `Sb-SAx`.
- `residual_view::SubArray`, a view of the residual container to handle varying compression
    sizes.
"""
mutable struct CompressedResidualRecipe{V<:AbstractVector, S<:SubArray} <: SolverErrorRecipe
    residual::V
    residual_view::S
end

function complete_error(
    error::CompressedResidual, 
    solver::Solver,
    A::AbstractMatrix, 
    b::AbstractVector
)
    residual = fill!(similar(A, eltype(A), size(b,1)), zero(eltype(A)))
    residual_view = view(residual, 1:1)
    return CompressedResidualRecipe{typeof(residual),typeof(residual_view)}(
        residual, 
        residual_view
    )

end

function compute_error(
        error::CompressedResidualRecipe, 
        solver::SolverRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )
    rows_s = size(solver.compressor, 1)
    error.residual_view = view(error.residual, 1:rows_s)
    copyto!(error.residual_view, solver.vec_view)
    mul!(error.residual_view, solver.mat_view, solver.solution_vec, -1.0, 1.0)
    return norm(error.residual_view)
end
