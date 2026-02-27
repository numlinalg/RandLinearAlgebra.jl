"""
    Logger

An abstract supertype for structures that record the progress of a `SolverRecipe` applied to
a coefficient matrix and constant vector.
"""
abstract type Logger end

"""
    LoggerRecipe

An abstract supertype for a structure that contains pre-allocated memory for a method that
records the progress of a `SolverRecipe`.
"""
abstract type LoggerRecipe end

# Docstring Components
logger_arg_list = Dict{Symbol,String}(
    :logger => "`logger::Logger`, a user-specified logging method.",
    :logger_recipe => "`logger::LoggerRecipe`, a fully initialized realization for a 
    logging method for a specific linear or least squares solver.",
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
    :b => "`b::AbstractVector`, a constant vector.",
    :err => "`err::Real`, an error value to be logged.",
    :iteration => "`iteration::Int64`, the iteration of the solver.",
)

logger_output_list = Dict{Symbol,String}(:logger_recipe => "A `LoggerRecipe` object.")

logger_method_description = Dict{Symbol,String}(
    :complete_logger => "A function that generates a `LoggerRecipe` given the 
    arguments.",
    :update_logger => "A function that updates the `LoggerRecipe` in place given 
    arguments.",
    :reset_logger => "A function that resets the `LoggerRecipe` in place.", 
)

logger_error_list = Dict{Symbol, String}(
    :complete_logger => "`ArgumentError` if no method for completing the logger exists \
    for the given logger type.",
    :update_logger => "`ArgumentError` if no method for updating the logger exists \
    for the given logger type.",
    :reset_logger => "`ArgumentError` if no method for resetting the logger exists \
    for the given logger type."
)

"""
    complete_logger(logger::Logger)

$(logger_method_description[:complete_logger])

### Arguments
- $(logger_arg_list[:logger])

### Returns 
- $(logger_output_list[:logger_recipe])

### Throws
- $(logger_error_list[:complete_logger])
"""
function complete_logger(logger::Logger)
    throw(ArgumentError("No `complete_logger` method defined for logger of type \
          $(typeof(logger))."))
    return nothing
end

"""
    update_logger!(logger::LoggerRecipe, err::Float64, iteration::Int64)

$(logger_method_description[:update_logger])

### Arguments
- $(logger_arg_list[:logger_recipe])
- $(logger_arg_list[:err]) 
- $(logger_arg_list[:iteration]) 

### Returns 
- Performs an inplace update to the `LoggerRecipe` and returns nothing.

### Throws
- $(logger_error_list[:update_logger])
"""
function update_logger!(logger::LoggerRecipe, err::Real, iteration::Int64)
    throw(ArgumentError("No `update_logger!` method defined for a logger of type \
    $(typeof(logger)), $(typeof(err)), and $(typeof(iteration))."))
    return nothing
end


"""
    reset_logger!(logger::LoggerRecipe)

$(logger_method_description[:reset_logger])

### Arguments
- $(logger_arg_list[:logger_recipe])

### Returns 
- Performs an inplace update to the `LoggerRecipe` and returns nothing.

### Throws
- $(logger_error_list[:reset_logger])
"""
function reset_logger!(logger::LoggerRecipe)
    throw(ArgumentError("No `reset_logger!` method defined for a logger of type \
    $(typeof(logger))."))
    return nothing
end

##############################
# Include Logger Files
###############################
include("Loggers/basic_logger.jl")
