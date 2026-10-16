"""
    Identity <: Compressor

An implementation of a compressor that returns the original matrix.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to
    be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.

# Constructor
    
    Identity(;cardinality=Left())

## Keywords
- `cardinality::Cardinality`, the direction the compression matrix is intended to
    be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.

## Returns
- A `Identity` object.
"""
mutable struct Identity <: Compressor 
    cardinality::Cardinality 
end

function Identity(; cardinality::Cardinality = Left())
    Identity(cardinality)
end

"""
    IdentityRecipe <: CompressorRecipe

The recipe containing all allocations and information for the `Identity` compressor.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to
    be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
"""
mutable struct IdentityRecipe{C} <: CompressorRecipe where C<:Cardinality
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
end

function complete_compressor(ingredients::Identity, A::AbstractMatrix)
    n_rows, n_cols = size(A)
    card = ingredients.cardinality
    if ingredients.cardinality == Left()
        return IdentityRecipe{typeof(card)}(Left(), n_rows, n_rows)
    elseif ingredients.cardinality == Right()
        return IdentityRecipe{typeof(card)}(Right(), n_cols, n_cols)
    end

end

function update_compressor!(S::IdentityRecipe{<:Left}, A::AbstractMatrix)
    S.n_rows = size(A,1)
    S.n_cols = size(A,1)
    return nothing
end

function update_compressor!(S::IdentityRecipe{<:Right}, A::AbstractMatrix)
    S.n_rows = size(A,2)
    S.n_cols = size(A,2)
    return nothing
end

function mul!(
    C::AbstractArray,
    S::IdentityRecipe,
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    # change the size of the compressor to align with the size of A
    # this is a special feature only for the identity compressor
    # this means setting the dimension of the compressor to the active dimension (rows)
    eff_dim = size(A, 1)
    S.n_rows = eff_dim
    S.n_cols = eff_dim
    left_mul_dimcheck(C, S, A)
    mul!(C, I, A, alpha, beta)
    return nothing
end

function mul!(
    C::AbstractArray,
    A::AbstractArray,
    S::IdentityRecipe,
    alpha::Number,
    beta::Number
)
    # change the size of the compressor to align with the size of A
    # this is a special feature only for the identity compressor
    # this means setting the dimension of the compressor to the active dimension (cols)
    eff_dim = size(A, 2)
    S.n_rows = eff_dim
    S.n_cols = eff_dim
    right_mul_dimcheck(C, A, S)
    mul!(C, A, I, alpha, beta)
    return nothing
end

# Because we want to return A we need to set the size of C to be that of A with identity
# compressor
# S * A
function (*)(S::IdentityRecipe, A::AbstractArray)
    C = fill!(similar(A), zero(eltype(A)))
    mul!(C, S, A)
    return C
end

# A * S
function (*)(A::AbstractArray, S::IdentityRecipe)
    C = fill!(similar(A), zero(eltype(A)))
    mul!(C, A, S)
    return C
end
