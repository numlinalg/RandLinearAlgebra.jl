###################################
# Abstract Types
###################################
"""
    Approximator

An abstract supertype for structures that store user-controlled parameters corresponding to
techniques that form low-rank approximations of the matrix `A`.
"""
abstract type Approximator end

"""
    ApproximatorRecipe

An abstract supertype for structures that store user-controlled parameters, linear system
dependent parameters and preallocated memory corresponding to techniques that form low-rank
approximations of the matrix `A`.
"""
abstract type ApproximatorRecipe end

"""
    ApproximatorError

An abstract supertype for structures containing user-controlled parameters corresponding to
methods that evaluate the quality of a low-rank approximation of a matrix `A`.
"""
abstract type ApproximatorError end

"""
    ApproximatorErrorRecipe

An abstract supertype for structures containing user-controlled parameters, matrix
dependent parameters and preallocated memory corresponding to methods that evaluate the
quality of a low-rank approximation of a matrix `A`.
"""
abstract type ApproximatorErrorRecipe end

"""
    RangeApproximator

An abstract type for the structures that contain the user-controlled parameters 
corresponding to the Approximator methods that produce an orthogonal approximation to the 
range of a matrix A. This includes methods like the RandomizedSVD and 
randomized rangefinder.
"""
abstract type RangeApproximator <: Approximator end

"""
    RangeApproximatorRecipe

An abstract type for the structures that contain the user-controlled parameters, 
linear system information, and preallocated memory for methods
corresponding to the Approximator methods that produce an orthogonal approximation to the
range of a matrix A. This includes methods like the RandomizedSVD and 
randomized rangefinder.
"""
abstract type RangeApproximatorRecipe <: ApproximatorRecipe end


###################################
# Docstring Components  
###################################
approx_arg_list = Dict{Symbol,String}(
    :approximator => "`approximator::Approximator`, a data structure containing the
    user-defined parameters associated with a particular low-rank approximation.",
    :approximator_recipe => "`approximator::ApproximatorRecipe`, a fully initialized
    realization for a low rank approximation method for a particular matrix.",
    :approximator_error => "`error::ApproximatorError`, a data structure containing
    the user-defined parameters associated with a particular low-rank approximation error
    method.",
    :approximator_error_recipe => "`error::ApproximatorErrorRecipe`, a fully initialized
    realization for a low rank approximation error method for a particular matrix.",
    :A => "`A::AbstractMatrix`, a target matrix for approximation.",
    :compressor_recipe => "`S::CompressorRecipe`, a fully initialized realization for a 
    compression method for a specific matrix or collection of matrices and vectors.",

)

approx_output_list = Dict{Symbol,String}(
    :approximator_recipe => "An `ApproximatorRecipe` object.",
    :approximator_error_recipe => "An `ApproximatorErrorRecipe` object.",
)

approx_method_description = Dict{Symbol,String}(
    :complete_approximator => "A function that generates an `ApproximatorRecipe` given 
    arguments.",

    :update_approximator => "A function that updates the `ApproximatorRecipe` in place
    given the arguments.",

    :rapproximate => "A function that computes a low-rank approximation of the matrix `A`
    using the information in the provided `Approximator` data structure.",
    :complete_approximator_error => "A function that generates an `ApproximatorErrorRecipe`
    given the arguments.",
    :compute_approximator_error => "A function that computes the approximation error of an
    `ApproximatorRecipe` for a matrix `A`.",
)

approx_error_list = Dict{Symbol,String}(
    :complete_approximator => "`ArgumentError` if no method for completing the \
    approximator exists for the given approximator type.",
    :rapproximate => "`ArgumentError` if no method for computing the approximation \
    exists for the given approximator type.",
    :complete_approximator_error => "`ArgumentError` if no method for completing the \
    approximator error exists for the given error type.",
    :compute_approximator_error => "`ArgumentError` if no method for computing the \
    approximator error exists for the given error type."
)

###################################
# Approximator Adjoint 
###################################
"""
    ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe 

A structure for the adjoint of an `ApproximatorRecipe`.

# Fields

  - `parent::ApproximatorRecipe`, the approximator that we compute the adjoint of.
"""
struct ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe
    parent::S
end

adjoint(A::ApproximatorRecipe) = ApproximatorAdjoint(A)
# Undo the transpose
adjoint(A::ApproximatorAdjoint{<:ApproximatorRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::ApproximatorRecipe) = ApproximatorAdjoint(A)
# Undo the transpose wrapper
transpose(A::ApproximatorAdjoint{<:ApproximatorRecipe}) = A.parent

###################################
# Size of Approximator 
###################################
function Base.size(S::ApproximatorRecipe)
    return S.n_rows, S.n_cols
end

function Base.size(S::ApproximatorRecipe, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.n_rows : S.n_cols
end

function Base.size(S::ApproximatorAdjoint)
    return S.parent.n_cols, S.parent.n_rows
end

function Base.size(S::ApproximatorAdjoint, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.parent.n_cols : S.parent.n_rows
end

###################################
# Complete Approximator Interface
###################################
"""
    complete_approximator(approximator::Approximator, A::AbstractMatrix)

$(approx_method_description[:complete_approximator])

# Arguments
- $(approx_arg_list[:approximator])
- $(approx_arg_list[:A]) 

# Returns
- $(approx_output_list[:approximator_recipe])

# Throws
- $(approx_error_list[:complete_approximator])
"""
function complete_approximator(approximator::Approximator, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No method `complete_approximator` exists for approximator of type\
            $(typeof(approximator)) and matrix of type $(typeof(A))."
        )
    )
end


###################################
# rapproximate Interface 
###################################
"""
    rapproximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)

$(approx_method_description[:rapproximate])

# Arguments
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

# Returns
- $(approx_output_list[:approximator_recipe])

# Throws
- $(approx_error_list[:rapproximate])
"""
function rapproximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No `rapproximate` method exists for approximator of type\
            $(typeof(approximator)) and matrix of type $(typeof(A))."
        )
    )
end

"""
    rapproximate(approximator::Approximator, A::AbstractMatrix)

$(approx_method_description[:rapproximate])

# Arguments
- $(approx_arg_list[:approximator])
- $(approx_arg_list[:A]) 

# Returns
- $(approx_output_list[:approximator_recipe])
"""
function rapproximate(approximator::Approximator, A::AbstractMatrix)
    approx_recipe = complete_approximator(approximator, A)
    rapproximate!(approx_recipe, A)
    return approx_recipe
end


###################################
# Complete Approximator Error Interface
###################################
"""
    complete_approximator_error(
        error::ApproximatorError, 
        approximator::ApproximatorRecipe, 
        A::AbstractMatrix
    )

$(approx_method_description[:complete_approximator_error])

# Arguments
- $(approx_arg_list[:approximator_error])
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

# Returns
- $(approx_output_list[:approximator_error_recipe])

# Throws
- $(approx_error_list[:complete_approximator_error])
"""
function complete_approximator_error(
    error::ApproximatorError, 
    approximator::ApproximatorRecipe, 
    A::AbstractMatrix
)
    return throw(
        ArgumentError(
            "No `complete_approximator_error! defined for error of type\
            $(typeof(error)), $(typeof(approximator)), and matrix of type $(typeof(A))."
        )
    )
end

###################################
# Compute Approximator Error Interface
###################################
"""
    compute_approximator_error!(
        error::ApproximatorErrorRecipe, 
        approximator::ApproximatorRecipe, 
        A::AbstractMatrix
    )

$(approx_method_description[:compute_approximator_error])

# Arguments
- $(approx_arg_list[:approximator_error_recipe])
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

# Returns
- Returns the `error::Float64` 

# Throws
- $(approx_error_list[:compute_approximator_error])
"""
function compute_approximator_error!(
    error::ApproximatorErrorRecipe, 
    approximator::ApproximatorRecipe, 
    A::AbstractMatrix
)
    return throw(
        ArgumentError(
            "No `complete_approximator_error! defined for error of type\
            $(typeof(error)), $(typeof(approximator)), and matrix of type $(typeof(A))."
        )
    )
end

# Implement a version of the compute error function that works without the recipe
"""
    compute_approximator_error(
        error::ApproximatorError, 
        approximator::ApproximatorRecipe, 
        A::AbstractMatrix
    )

$(approx_method_description[:compute_approximator_error])

# Arguments
- $(approx_arg_list[:approximator_error])
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

# Returns
- Returns the `error::Float64` 
"""
function compute_approximator_error(
    error::ApproximatorError, 
    approximator::ApproximatorRecipe, 
    A::AbstractMatrix
)
    error_recipe = complete_approximator_error(error, approximator, A)
    error_val = compute_approximator_error!(error_recipe, approximator, A)
    return error_val
end

########################################
# 5 Arg Compressor-Array Multiplications
########################################

# alpha*R*A + b*C -> C
function mul!(
    C::AbstractArray, 
    R::ApproximatorRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    return throw(
        ArgumentError(
            "No method `mul!` defined for ($(typeof(C)), $(typeof(R)), \
            $(typeof(A)), $(typeof(alpha)), $(typeof(beta)))."
        )
    )
end

# alpha*A*R + beta*C -> C
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::ApproximatorRecipe, 
    alpha::Number, 
    beta::Number
)
    return throw(
        ArgumentError(
            "No method `mul!` defined for ($(typeof(C)), $(typeof(A)), \
            $(typeof(R)), $(typeof(alpha)), $(typeof(beta)))."
        )
    )
end

# alpha * R'*A + beta*C -> C (equivalently, alpha * A' * R + beta + C' -> C')
function mul!(
    C::AbstractArray,
    R::ApproximatorAdjoint,
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    mul!(transpose(C), transpose(A), R.parent, alpha, beta)
    return nothing
end

# alpha * A*R' + beta*C -> C (equivalently, alpha * R * A' + beta*C' -> C')
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::ApproximatorAdjoint, 
    alpha::Number, 
    beta::Number
)
    mul!(transpose(C), R.parent, transpose(A), alpha, beta)
    return nothing
end

########################################
# 3 Arg Compressor-Array Multiplications
########################################
# R * A - > C
function mul!(C::AbstractArray, R::ApproximatorRecipe, A::AbstractArray)
    mul!(C, R, A, 1.0, 0.0)
    return nothing
end

# A * R - > C
function mul!(C::AbstractArray, A::AbstractArray, R::ApproximatorRecipe)
    mul!(C, A, R, 1.0, 0.0)
    return nothing
end

# R' * A - > C
function mul!(C::AbstractArray, R::ApproximatorAdjoint, A::AbstractArray)
    mul!(C, R, A, 1.0, 0.0)
    return nothing
end

# A * R' - > C
function mul!(C::AbstractArray, A::AbstractArray, R::ApproximatorAdjoint)
    mul!(C, A, R, 1.0, 0.0)
    return nothing
end

##################################################
# Binary Operator Approximator-Array Multiplications
##################################################
# R * A
function (*)(R::ApproximatorRecipe, A::AbstractArray)
    r_rows = size(R, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? fill!(similar(A, eltype(A), r_rows), zero(eltype(A))) :
                      fill!(similar(A, eltype(A), r_rows, a_cols), zero(eltype(A)))
    mul!(C, R, A)
    return C
end

# A * R
function (*)(A::AbstractArray, R::ApproximatorRecipe)
    r_cols = size(R, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? fill!(similar(A, eltype(A), r_cols), zero(eltype(A)))' :
                      fill!(similar(A, eltype(A), a_rows, r_cols), zero(eltype(A)))
    mul!(C, A, R)
    return C
end

# R' * A
function (*)(R::ApproximatorAdjoint, A::AbstractArray)
    r_rows = size(R, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? fill!(similar(A, eltype(A), r_rows), zero(eltype(A))) :
                      fill!(similar(A, eltype(A), r_rows, a_cols), zero(eltype(A)))
    mul!(C, R, A)
    return C
end

# A * R'
function (*)(A::AbstractArray, R::ApproximatorAdjoint)
    r_cols = size(R, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? fill!(similar(A, eltype(A), r_cols), zero(eltype(A)))' :
                      fill!(similar(A, eltype(A), a_rows, r_cols), zero(eltype(A)))
    mul!(C, A, R)
    return C
end

###########################################
# Include the Approximator files
############################################
include("Approximators/RangeApproximators/rangefinder.jl")
include("Approximators/RangeApproximators/randsvd.jl")
include("Approximators/RangeApproximators/helpers/power_its.jl")

############################################
# Include the selector files
############################################
include("Approximators/Selectors.jl")
