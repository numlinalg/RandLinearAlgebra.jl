"""
    CURCore

An abstract type for computing the core linking matrix in a CUR decomposition.
"""
abstract type CURCore end

"""
    CURCoreRecipe

An abstract type for the recipes containing the preallocated information required for
    computing the core linking matrix in a CUR decomposition.
"""
abstract type CURCoreRecipe end

"""
    CURCoreAdjoint{S<:CURCoreRecipe} <: CURCoreRecipe 

A structure for the adjoint of an `CURCoreRecipe`.

# Fields

  - `Parent::CURCoreRecipe`, the approximator that we are taking the adjoint.
"""
struct CURCoreAdjoint{S<:CURCoreRecipe} <: CURCoreRecipe
    parent::S
end

adjoint(A::CURCoreRecipe) = CURCoreAdjoint(A)
# Undo the transpose
adjoint(A::CURCoreAdjoint{<:CURCoreRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::CURCoreRecipe) = CURCoreAdjoint(A)
# Undo the transpose wrapper
transpose(A::CURCoreAdjoint{<:CURCoreRecipe}) = A.parent

###################################
# Size of CURCore 
###################################
function Base.size(S::CURCoreRecipe)
    return S.n_rows, S.n_cols
end

function Base.size(S::CURCoreRecipe, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.n_rows : S.n_cols
end

function Base.size(S::CURCoreAdjoint)
    return S.parent.n_cols, S.parent.n_rows
end

function Base.size(S::CURCoreAdjoint, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.parent.n_cols : S.parent.n_rows
end

# add the mul! functions for CURCore matrices
function mul!(
    C::AbstractArray, 
    U::CURCoreRecipe, 
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
    U::CURCoreRecipe, 
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
    U::CURCoreAdjoint,
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
    U::CURCoreAdjoint, 
    alpha::Number, 
    beta::Number
)
    mul!(transpose(C), U.parent, transpose(A), alpha, beta)
    return nothing
end

# C <- U * A
function mul!(C::AbstractArray, U::CURCoreRecipe, A::AbstractArray)
    mul!(C, U, A, 1.0, 0.0)
    return nothing
end

# C <- A * U
function mul!(C::AbstractArray, A::AbstractArray, U::CURCoreRecipe)
    mul!(C, A, U, 1.0, 0.0)
    return nothing
end

# C <- U' * A
function mul!(C::AbstractArray, U::CURCoreAdjoint, A::AbstractArray)
    mul!(transpose(C), transpose(A), U.parent)
    return nothing
end

# C <- A * U'
function mul!(C::AbstractArray, A::AbstractArray, U::CURCoreAdjoint)
    mul!(transpose(C), U.parent, transpose(A))
    return nothing
end

# U * A
function (*)(U::CURCoreRecipe, A::AbstractArray)
    u_rows = size(U, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? zeros(eltype(A), u_rows) : zeros(eltype(A), u_rows, a_cols)
    mul!(C, U, A)
    return C
end

# A * U
function (*)(A::AbstractArray, U::CURCoreRecipe)
    u_cols = size(U, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? zeros(eltype(A), u_cols)' : zeros(eltype(A), a_rows, u_cols)
    mul!(C, A, U)
    return C
end

# S' * A
function (*)(U::CURCoreAdjoint, A::AbstractArray)
    u_rows = size(U, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? zeros(eltype(A), u_rows) : zeros(eltype(A), u_rows, a_cols)
    mul!(C, U, A)
    return C
end

# A * S'
function (*)(A::AbstractArray, U::CURCoreAdjoint)
    u_cols = size(U, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? zeros(eltype(A), u_cols)' : zeros(eltype(A), a_rows, u_cols)
    mul!(C, A, U)
    return C
end