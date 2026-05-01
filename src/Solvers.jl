###################################
# Abstract Types
###################################
"""
    Solver

An abstract supertype for structures that contain the user-controlled parameters for
methods that solve linear systems and least squares problems.
"""
abstract type Solver end

"""
    SolverRecipe

An abstract supertype specifying a solver method with pre-allocated data structures given a
coefficient matrix and constant vector.
"""
abstract type SolverRecipe end

###################################
# Docstring Components  
###################################
solver_arg_list = Dict{Symbol,String}(
    :solver => "`solver::Solver`, a user-specified solver method.",
    :solver_recipe => "`solver::SolverRecipe`, a fully initialized realization for a 
    solver method for a specific linear system.",
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
    :b => "`b::AbstractVector`, a constant vector.",
    :x => "`x::AbstractVector`, a vector for the proposed solution.",
)

solver_output_list = Dict{Symbol,String}(
    :solver_recipe => "A `SolverRecipe` object.",
    :solver_error_recipe => "A `SolverErrorRecipe` object.",
    :x => "`x::AbstractVector`, a proposed solution to a linear system or least squares 
    problem.",
)

solver_method_description = Dict{Symbol,String}(
    :complete_solver => "A function that generates a `SolverRecipe` given the 
    arguments.",
    :rsolve => "A function that solves a linear system given the arguments.",
)

solver_error_list = Dict{Symbol,String}(
    :complete_solver => "`ArgumentError` if no method for completing the solver \
    exists for the given solver type.",
    :rsolve => "`ArgumentError` if no method for solving the linear system exists \
    for the given solver type."
)

###################################
# Complete Solver
###################################
"""
    complete_solver(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)

$(solver_method_description[:complete_solver])

# Arguments
- $(solver_arg_list[:solver])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

# Returns
- $(solver_output_list[:solver_recipe])

# Throws
- $(solver_error_list[:complete_solver])
"""
function complete_solver(
    solver::Solver, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    return throw(
        ArgumentError(
            "No `complete_solver` method defined for a $(typeof(solver)) solver,\
             $(typeof(x)), $(typeof(A)), and $(typeof(b))."
        )
    )
end

###################################
# rsolve Interface 
###################################
"""
    rsolve!(
        solver::SolverRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

$(solver_method_description[:rsolve])

# Arguments
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

# Returns
- Returns `nothing`. Updates the `SolverRecipe` and `x` in place.

# Throws
- $(solver_error_list[:rsolve])
"""
function rsolve!(
    solver::SolverRecipe, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    return throw(
        ArgumentError(
            "No `rsolve!` method defined for $(typeof(solver)) solver,\
             $(typeof(x)), $(typeof(A)), and $(typeof(b))."
            )
        )
end

"""
    rsolve!(
        solver::Solver, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

$(solver_method_description[:rsolve])

# Arguments
- $(solver_arg_list[:solver])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

# Returns
- $(solver_output_list[:x])
- $(solver_output_list[:solver_recipe])
"""
function rsolve!(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    solver_method = complete_solver(solver, x, A, b)
    rsolve!(solver_method, x, A, b)
    return x, solver_method
end

############################
# The Loggers
############################
include("Solvers/Loggers.jl")
#############################
# The sub solvers
#############################
include("Solvers/SubSolvers.jl")
############################
# The solver error
###########################
include("Solvers/ErrorMethods.jl")
#############################
# The Solver Routine Files
############################
include("Solvers/col_projection.jl")
include("Solvers/ihs.jl")
include("Solvers/kaczmarz.jl")
############################
# Helper functions
###########################
include("Solvers/helpers/dotu.jl")
