"""
    Compressor

An abstract supertype for structures that contain user-controlled parameters corresponding
to techniques that compress a matrix.
"""
abstract type Compressor end

"""
    CompressorRecipe

An abstract supertype for structures that contain both the user-controlled
parameters in the `Compressor` and the memory allocations necessary for applying the
compression technique to a particular set of matrices and vectors.
"""
abstract type CompressorRecipe end

"""
    Cardinality

An abstract type for types that specify whether a compressor will be applied from the
left or the right.
"""
abstract type Cardinality end

"""
    Left <: Cardinality

A struct indicating matrix multiplication from the left.
"""
struct Left <: Cardinality end

"""
    Right <: Cardinality

A struct indicating matrix multiplication from the right.
"""
struct Right <: Cardinality end

"""
    Undef <: Cardinality

A struct indicating matrix multiplication is undefined.
"""
struct Undef <: Cardinality end

###################################
# Docstring Components  
###################################
comp_arg_list = Dict{Symbol,String}(
    :compressor => "`compressor::Compressor`, a user-specified compression method.",
    :compressor_recipe => "`S::CompressorRecipe`, a fully 
    initialized realization for a compression method for a specific matrix or collection 
    of matrices and vectors.",
    :compressor_recipe_adjoint => "`S::CompressorAdjoint`, the representation of an adjoint
    of a compression operator.",
    :A => "`A::AbstractMatrix`, a target matrix for compression.",
    :C => "`C::AbstractMatrix`, a matrix where the output will be stored.",
    :b => "`b::AbstractVector`, a possible target vector for compression.",
    :x => "`x::AbstractVector`, a vector that ususally represents a current iterate 
    typically used in a solver.",
    :y => "`y::AbstractVector`, a vector.",
    :z => "`z::AbstractVector`, a vector.",
)

comp_output_list = Dict{Symbol,String}(:compressor_recipe => "A `CompressorRecipe` object.")

comp_method_description = Dict{Symbol,String}(
    :complete_compressor => "A function that generates a `CompressorRecipe` given the 
    arguments.",
    :update_compressor => "A function that updates the `CompressorRecipe` in place given 
    arguments.",
    :mul_check => "A function that checks the compatibility of arguments for multiplication",
)

comp_error_list = Dict{Symbol, String}(
    :complete_compressor => "`ArgumentError` if no method for completing the compressor exists 
    for the given arguments.",
    :update_compressor => "`ArgumentError` if no method exists for updating the compressor 
    exists."
)

###################################
# Compressor Adjoint  
###################################
"""
    CompressorAdjoint{S<:CompressorRecipe}

A structure for the adjoint of a compression recipe.

# Fields
- `parent::CompressorRecipe`, the CompressorRecipe the adjoint is being applied to.
"""
struct CompressorAdjoint{S<:CompressorRecipe}
    parent::S
end

adjoint(A::CompressorRecipe) = CompressorAdjoint(A)
# Undo the transpose
adjoint(A::CompressorAdjoint{<:CompressorRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::CompressorRecipe) = CompressorAdjoint(A)
# Undo the transpose wrapper
transpose(A::CompressorAdjoint{<:CompressorRecipe}) = A.parent



###################################
# Complete Compressor Interface  
###################################
"""
    complete_compressor(compressor::Compressor, x::AbstractVector)

$(comp_method_description[:complete_compressor])

# Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:x])

# Returns 
- $(comp_output_list[:compressor_recipe])

# Throws 
- $(comp_error_list[:complete_compressor])
"""
function complete_compressor(compressor::Compressor, x::AbstractVector)
    # Handle Vector input by reshaping to column matrix
    complete_compressor(compressor, reshape(x, :, 1))
end

"""
    complete_compressor(compressor::Compressor, A::AbstractMatrix)

$(comp_method_description[:complete_compressor])

# Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:A])

# Returns 
- $(comp_output_list[:compressor_recipe])

# Throws 
- $(comp_error_list[:complete_compressor])
"""
function complete_compressor(compressor::Compressor, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No `complete_compressor` method exists for compressor of type \
            $(typeof(compressor)) and matrix of type $(typeof(A))."
        )
    )
end

"""
    complete_compressor(compressor::Compressor, A::AbstractMatrix, b::AbstractVector)

$(comp_method_description[:complete_compressor])

# Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:A])
- $(comp_arg_list[:b])

# Returns
- $(comp_output_list[:compressor_recipe])

# Throws 
- $(comp_error_list[:complete_compressor])
"""
function complete_compressor(compressor::Compressor, A::AbstractMatrix, b::AbstractVector)
    # If this variant is not defined for a compressor call the one with input matrix A 
    return complete_compressor(compressor, A)
end

"""
    complete_compressor(
        compressor::Compressor, 
        x::AbstractVector
        A::AbstractMatrix, 
        b::AbstractVector, 
    )

$(comp_method_description[:complete_compressor])

# Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:x])
- $(comp_arg_list[:A])
- $(comp_arg_list[:b])

# Returns
- $(comp_output_list[:compressor_recipe])

# Throws 
- $(comp_error_list[:complete_compressor])
"""
function complete_compressor(
    compressor::Compressor, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    # If this variant is not defined for a compressor call the one with input matrix A 
    return complete_compressor(compressor, A, b)
end

###################################
# Update Compressor Interface 
###################################
"""
    update_compressor!(S::CompressorRecipe)

$(comp_method_description[:update_compressor])

# Arguments
- $(comp_arg_list[:compressor_recipe])

# Returns 
- `nothing`

# Throws 
- $(comp_error_list[:update_compressor])
"""
function update_compressor!(S::CompressorRecipe)
    return throw(
        ArgumentError(
            "No method `update_compressor` exists for compressor recipe of type \
            $(typeof(S))."
        )
    )
end

"""
    update_compressor!(S::CompressorRecipe, A::AbstractMatrix)

$(comp_method_description[:update_compressor])

# Arguments
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:A])

# Returns
- `nothing`

# Throws 
- $(comp_error_list[:update_compressor])
"""
function update_compressor!(S::CompressorRecipe, A::AbstractMatrix)
    update_compressor!(S)
    return nothing
end

"""
    update_compressor!(S::CompressorRecipe, A::AbstractMatrix, b::AbstractVector)

$(comp_method_description[:update_compressor])

# Arguments
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:A])
- $(comp_arg_list[:b])

# Returns 
- `nothing`

# Throws 
- $(comp_error_list[:update_compressor])
"""
function update_compressor!(S::CompressorRecipe, A::AbstractMatrix, b::AbstractVector)
    update_compressor!(S, A)
    return nothing
end

"""
    update_compressor!(
        S::CompressorRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector,
        x::AbstractMatrix
    )

$(comp_method_description[:update_compressor])

# Arguments
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:x])
- $(comp_arg_list[:A])
- $(comp_arg_list[:b])

# Returns
- `nothing`

# Throws 
- $(comp_error_list[:update_compressor])
"""
function update_compressor!(
    S::CompressorRecipe, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    update_compressor!(S, A, b)
    return nothing
end

###################################
# Size of Compressor 
###################################
function Base.size(S::CompressorRecipe)
    return S.n_rows, S.n_cols
end
function Base.size(S::CompressorRecipe, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.n_rows : S.n_cols
end
function Base.size(S::CompressorAdjoint)
    return S.parent.n_cols, S.parent.n_rows
end
function Base.size(S::CompressorAdjoint, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.parent.n_cols : S.parent.n_rows
end


########################################
# 5 Arg Compressor-Array Multiplications
########################################

# alpha*S*A + b*C -> C
function mul!(
    C::AbstractArray, 
    S::CompressorRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    return throw(
        ArgumentError(
            "No method `mul!` defined for ($(typeof(C)), $(typeof(S)), \
            $(typeof(A)), $(typeof(alpha)), $(typeof(beta)))."
        )
    )
end

# alpha*A*S + beta*C -> C
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::CompressorRecipe, 
    alpha::Number, 
    beta::Number
)
    return throw(
        ArgumentError(
            "No method `mul!` defined for ($(typeof(C)), $(typeof(A)), \
            $(typeof(S)), $(typeof(alpha)), $(typeof(beta)))."
        )
    )
end

# alpha * S'*A + beta*C -> C (equivalently, alpha * A' * S + beta + C' -> C')
function mul!(
    C::AbstractArray,
    S::CompressorAdjoint,
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    mul!(transpose(C), transpose(A), S.parent, alpha, beta)
    return nothing
end

# alpha * A*S' + beta*C -> C (equivalently, alpha * S * A' + beta*C' -> C')
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::CompressorAdjoint, 
    alpha::Number, 
    beta::Number
)
    mul!(transpose(C), S.parent, transpose(A), alpha, beta)
    return nothing
end
########################################
# 3 Arg Compressor-Array Multiplications
########################################
# S * A - > C
function mul!(C::AbstractArray, S::CompressorRecipe, A::AbstractArray)
    mul!(C, S, A, 1.0, 0.0)
    return nothing
end

# A * S -> C 
function mul!(C::AbstractArray, A::AbstractArray, S::CompressorRecipe)
    mul!(C, A, S, 1.0, 0.0)
    return nothing
end

# S' * A -> C; Equivalently: A' * S -> C'
function mul!(C::AbstractArray, S::CompressorAdjoint, A::AbstractArray)
    mul!(transpose(C), transpose(A), S.parent)
    return nothing
end

# A * S' -> C; Equivalently: S * A' -. C'
function mul!(C::AbstractArray, A::AbstractArray, S::CompressorAdjoint)
    mul!(transpose(C), S.parent, transpose(A))
    return nothing
end

##################################################
# Binary Operator Compressor-Array Multiplications
##################################################
# S * A
function (*)(S::CompressorRecipe, A::AbstractArray)
    s_rows = size(S, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? fill!(similar(A, eltype(A), s_rows), zero(eltype(A))) :
                      fill!(similar(A, eltype(A), s_rows, a_cols), zero(eltype(A)))
    mul!(C, S, A)
    return C
end

# A * S
function (*)(A::AbstractArray, S::CompressorRecipe)
    s_cols = size(S, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? fill!(similar(A, eltype(A), s_cols), zero(eltype(A)))' :
                      fill!(similar(A, eltype(A), a_rows, s_cols), zero(eltype(A)))
    mul!(C, A, S)
    return C
end

# S' * A
function (*)(S::CompressorAdjoint, A::AbstractArray)
    s_rows = size(S, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? fill!(similar(A, eltype(A), s_rows), zero(eltype(A))) :
                      fill!(similar(A, eltype(A), s_rows, a_cols), zero(eltype(A)))
    mul!(C, S, A)
    return C
end

# A * S'
function (*)(A::AbstractArray, S::CompressorAdjoint)
    s_cols = size(S, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? fill!(similar(A, eltype(A), s_cols), zero(eltype(A)))' :
                      fill!(similar(A, eltype(A), a_rows, s_cols), zero(eltype(A)))
    mul!(C, A, S)
    return C
end

###################################
# Include Compressor Files
###################################
include("Compressors/Distributions.jl")
include("Compressors/helpers/fwht.jl")
include("Compressors/count_sketch.jl")
include("Compressors/fjlt.jl")
include("Compressors/gaussian.jl") 
include("Compressors/identity.jl")
include("Compressors/sampling.jl")
include("Compressors/sparse_sign.jl")
include("Compressors/srht.jl")
