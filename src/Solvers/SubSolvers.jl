"""
    SubSolver

An abstract supertype for structures specifying solution methods for a linear system or
least squares problem.
"""
abstract type SubSolver end

"""
    SubSolverRecipe

An abstract supertype for structures with pre-allocated memory for methods that solve a
linear system or least squares problem.
"""
abstract type SubSolverRecipe end

# Docstring Components
sub_solver_arg_list = Dict{Symbol,String}(
    :sub_solver => "`solver::SubSolver`, a user-specified sub-solving method.",
    :sub_solver_recipe => "`solver::SubSolverRecipe`, a fully initialized realization for a
    linear sub-solver.",
    :A => "`A::AbstractArray`, a coefficient matrix or vector.",
    :b => "`b::AbstractArray`, a constant matrix or vector.",
)

sub_solver_output_list = Dict{Symbol,String}(
    :sub_solver_recipe => "A `SubSolverRecipe` object."
)

sub_solver_method_description = Dict{Symbol,String}(
    :complete_sub_solver => "A function that generates a `SubSolverRecipe` given the 
    arguments.",
    :update_sub_solver => "A function that updates the `SubSolver` in place given 
    arguments.",
    :ldiv => "A function that solves a linear system using the `SubSolverRecipe` and stores 
    the result in `x`.",
)

sub_solver_error_list = Dict{Symbol,String}(
    :complete_sub_solver => "`ArgumentError` if no method for completing the sub-solver exists for the given sub-solver type.",
    :update_sub_solver => "`ArgumentError` if no method for updating the sub-solver exists for the given sub-solver type.",
    :ldiv => "`ArgumentError` if no method for solving with the sub-solver exists for the given sub-solver type."
)

"""
    complete_sub_solver(solver::SubSolver, A::AbstractArray)

$(sub_solver_method_description[:complete_sub_solver])

# Arguments
- $(sub_solver_arg_list[:sub_solver])
- $(sub_solver_arg_list[:A]) 

# Returns 
- $(sub_solver_output_list[:sub_solver_recipe])

# Throws
- $(sub_solver_error_list[:complete_sub_solver])
"""
function complete_sub_solver(solver::SubSolver, A::AbstractArray)
    return throw(
        ArgumentError(
            "No `complete_sub_solver!` method defined for a solver of type \
            $(typeof(solver)) and $(typeof(A))."
        )
    )
end

"""
    complete_sub_solver(solver::SubSolver, A::AbstractArray, b::AbstractArray)

$(sub_solver_method_description[:complete_sub_solver])

# Arguments
- $(sub_solver_arg_list[:sub_solver])
- $(sub_solver_arg_list[:A]) 
- $(sub_solver_arg_list[:b]) 

# Returns 
- $(sub_solver_output_list[:sub_solver_recipe])

# Throws
- $(sub_solver_error_list[:complete_sub_solver])
"""
function complete_sub_solver(solver::SubSolver, A::AbstractArray, b::AbstractArray)
    complete_sub_solver(solver, A)
end

"""
    update_sub_solver!(solver::SubSolverRecipe, A::AbstractArray)

$(sub_solver_method_description[:update_sub_solver])

# Arguments
- $(sub_solver_arg_list[:sub_solver_recipe])
- $(sub_solver_arg_list[:A]) 

# Returns
- Modifies the `SubSolverRecipe` in place given the arguments.and returns nothing.

# Throws
- $(sub_solver_error_list[:update_sub_solver])
"""
function update_sub_solver!(solver::SubSolverRecipe, A::AbstractArray)
    return  throw(
        ArgumentError(
            "No `update_sub_solver!` method defined for a solver of type $(typeof(solver))\
            and $(typeof(A))."
        )
    )
end

"""
    ldiv!(x::AbstractVector, solver::SubSolverRecipe, b::AbstractVector)

$(sub_solver_method_description[:ldiv])

# Arguments
- `x::AbstractVector`, the output vector to store the solution.
- $(sub_solver_arg_list[:sub_solver_recipe])
- `b::AbstractVector`, the right-hand side vector.

# Returns
- Modifies `x` in place to contain the solution and returns `x`.

# Throws
- $(sub_solver_error_list[:ldiv])
"""
function ldiv!(x::AbstractVector, solver::SubSolverRecipe, b::AbstractVector)
    return throw(
        ArgumentError(
            "No `ldiv!` method defined for a solver of type $(typeof(solver)), \
            $(typeof(x)), and $(typeof(b))."
        )
    )
end

###########################################
# Include SubSolver files
###########################################
include("SubSolvers/lq.jl")
include("SubSolvers/qr.jl")
