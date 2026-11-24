"""
    GeneralizedNystromCore

An abstract type for computing the core linking matrix in a generalized Nyström decomposition.
"""
abstract type GeneralizedNystromCore end

"""
    GeneralizedNystromCoreRecipe

An abstract type for the recipes containing the preallocated information required for
    computing the core linking matrix in a generalized Nyström decomposition.
"""
abstract type GeneralizedNystromCoreRecipe end

"""
    GeneralizedNystromCoreAdjoint{S<:GeneralizedNystromCoreRecipe} <: GeneralizedNystromCoreRecipe 

A structure for the adjoint of an `GeneralizedNystromCoreRecipe`.

# Fields

  - `Parent::GeneralizedNystromCoreRecipe`, the approximator that we are taking the adjoint.
"""
struct GeneralizedNystromCoreAdjoint{S<:GeneralizedNystromCoreRecipe} <: GeneralizedNystromCoreRecipe
    parent::S
end

adjoint(A::GeneralizedNystromCoreRecipe) = GeneralizedNystromCoreAdjoint(A)
# Undo the transpose
adjoint(A::GeneralizedNystromCoreAdjoint{<:GeneralizedNystromCoreRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::GeneralizedNystromCoreRecipe) = GeneralizedNystromCoreAdjoint(A)
# Undo the transpose wrapper
transpose(A::GeneralizedNystromCoreAdjoint{<:GeneralizedNystromCoreRecipe}) = A.parent

###################################
# Size of GeneralizedNystromCore 
###################################
function Base.size(S::GeneralizedNystromCoreRecipe)
    return S.n_rows, S.n_cols
end

function Base.size(S::GeneralizedNystromCoreRecipe, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.n_rows : S.n_cols
end

function Base.size(S::GeneralizedNystromCoreAdjoint)
    return S.parent.n_cols, S.parent.n_rows
end

function Base.size(S::GeneralizedNystromCoreAdjoint, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.parent.n_cols : S.parent.n_rows
end

# add the mul! functions for GeneralizedNystromCore matrices
function mul!(
    C::AbstractArray, 
    U::GeneralizedNystromCoreRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    return throw(
        ArgumentError(
            "No method `mul!` defined for ($(typeof(C)), $(typeof(U)), \
            $(typeof(A)), $(typeof(alpha)), $(typeof(beta)))."
        )
    )
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    U::GeneralizedNystromCoreRecipe, 
    alpha::Number, 
    beta::Number
)
    return throw(
        ArgumentError(
            "No method `mul!` defined for ($(typeof(C)), $(typeof(A)), \
            $(typeof(U)), $(typeof(alpha)), $(typeof(beta)))."
        )
    )
end

# C <- beta * C + alpha * U' * A
function mul!(  
    C::AbstractArray,
    U::GeneralizedNystromCoreAdjoint,
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    mul!(transpose(C), transpose(A), U.parent, alpha, beta)
    return nothing
end

# C <- beta * C + alpha * A * U'
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    U::GeneralizedNystromCoreAdjoint, 
    alpha::Number, 
    beta::Number
)
    mul!(transpose(C), U.parent, transpose(A), alpha, beta)
    return nothing
end

# C <- U * A
function mul!(C::AbstractArray, U::GeneralizedNystromCoreRecipe, A::AbstractArray)
    mul!(C, U, A, 1.0, 0.0)
    return nothing
end

# C <- A * U
function mul!(C::AbstractArray, A::AbstractArray, U::GeneralizedNystromCoreRecipe)
    mul!(C, A, U, 1.0, 0.0)
    return nothing
end

# C <- U' * A
function mul!(C::AbstractArray, U::GeneralizedNystromCoreAdjoint, A::AbstractArray)
    mul!(transpose(C), transpose(A), U.parent)
    return nothing
end

# C <- A * U'
function mul!(C::AbstractArray, A::AbstractArray, U::GeneralizedNystromCoreAdjoint)
    mul!(transpose(C), U.parent, transpose(A))
    return nothing
end

# U * A
function (*)(U::GeneralizedNystromCoreRecipe, A::AbstractArray)
    u_rows = size(U, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? zeros(eltype(A), u_rows) : zeros(eltype(A), u_rows, a_cols)
    mul!(C, U, A)
    return C
end

# A * U
function (*)(A::AbstractArray, U::GeneralizedNystromCoreRecipe)
    u_cols = size(U, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? zeros(eltype(A), u_cols)' : zeros(eltype(A), a_rows, u_cols)
    mul!(C, A, U)
    return C
end

# S' * A
function (*)(U::GeneralizedNystromCoreAdjoint, A::AbstractArray)
    u_rows = size(U, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? zeros(eltype(A), u_rows) : zeros(eltype(A), u_rows, a_cols)
    mul!(C, U, A)
    return C
end

# A * S'
function (*)(A::AbstractArray, U::GeneralizedNystromCoreAdjoint)
    u_cols = size(U, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? zeros(eltype(A), u_cols)' : zeros(eltype(A), a_rows, u_cols)
    mul!(C, A, U)
    return C
end
