"""
    Gaussian <: Compressor

A specification for a Gaussian compressor.

# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress.

If we want to compress ``A`` from the left (i.e., we reduce the number of rows), then
we create a Gaussian sketch matrix, ``S``, with dimension ``s \\times m`` where
``s`` is the compression dimension that is supplied by the user.
Each entry of ``S`` is generated independently following N(0, 1/s), namely, a
Gaussian distribution with mean being zero and variance being 1 divided by the
compression dimension.

If ``A`` is compressed from the right, then we create a Gaussian sketch matrix, ``S``,
with dimension ``n \\times s``, where ``s`` is the compression dimension that
is supplied by the user.
Each entry of ``S`` is generated independently following N(0, 1/s), namely, a
Gaussian distribution with mean being zero and variance being 1 divided by the
compression dimension.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in 
    the mathematical description.
- `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    Gaussian(;cardinality=Left(), compression_dim=2, type=Float64)

## Arguments
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in 
    the mathemtical description. By default this is set to 2.
- `type::Type{<:Number}`, the type of elements in the compressor.

## Returns
- A `Gaussian` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive
"""
struct Gaussian <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    type::Type{<:Number}
    # Check on the compression dimension
    function Gaussian(cardinality, compression_dim, type)
        if compression_dim <= 0
            throw(ArgumentError("Field 'compression_dim' must be positive."))
        end

        return new(cardinality, compression_dim, type)
    end
end

function Gaussian(;
    cardinality::Cardinality=Left(),
    compression_dim::Int64=2, 
    type::Type{<:Number}=Float64,
)
    # Partially construct the Gaussian datatype
    return Gaussian(cardinality, compression_dim, type)
end

"""
    GaussianRecipe <: CompressorRecipe

The recipe containing all allocations and information for the Gaussian compressor.

# Fields
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. The
value is either `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `scale::Number`, the standard deviation of Gaussian distribution during the 
compression matrix generation.
- `op::M where M<:AbstractMatrix{<:Number}`, the Gaussian compression matrix.
"""
mutable struct GaussianRecipe{C<:Cardinality, M<:AbstractMatrix{<:Number}} <: CompressorRecipe
    cardinality::C
    compression_dim::Int64
    n_rows::Int64
    n_cols::Int64
    scale::Number
    op::M
end
Adapt.@adapt_structure GaussianRecipe

function GaussianRecipe(
    cardinality::Right,
    compression_dim::Int64,
    type::Type{<:Number},
    A::AbstractMatrix,
)
    n_rows = size(A, 2)
    n_cols = compression_dim
    # Generate entry values by N(0,1/d)
    scale = convert(type, 1 / sqrt(compression_dim))
    op = similar(A, type, n_rows, n_cols)
    randn!(op)
    lmul!(scale, op)
    return GaussianRecipe(cardinality, compression_dim, n_rows, n_cols, scale, op)
end

function GaussianRecipe(
    cardinality::Left,
    compression_dim::Int64,
    type::Type{<:Number},
    A::AbstractMatrix,
)
    n_rows = compression_dim
    n_cols = size(A, 1)
    # Generate entry values by N(0,1/d)
    scale = convert(type, 1 / sqrt(compression_dim))
    op = similar(A, type, n_rows, n_cols)
    randn!(op)
    lmul!(scale, op)
    return GaussianRecipe(cardinality, compression_dim, n_rows, n_cols, scale, op)
end

function complete_compressor(ingredients::Gaussian, A::AbstractMatrix)
    return GaussianRecipe(
        ingredients.cardinality,
        ingredients.compression_dim,
        ingredients.type,
        A,
    )
end

# Allocations in this function are entirely due to bitrand call
function update_compressor!(S::GaussianRecipe)
    # Inplace update of sketch_matrix
    randn!(S.op)
    lmul!(S.scale, S.op)
    return nothing
end

# Implement the matrix-Matrix Multiplication operators
# Begin with the left version
function mul!(
    C::AbstractArray,
    S::GaussianRecipe,
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    left_mul_dimcheck(C, S, A)
    # Built-in multiplication
    mul!(C, S.op, A, alpha, beta)
    return nothing
end

# Now implement the right versions
function mul!(
    C::AbstractArray,
    A::AbstractArray,
    S::GaussianRecipe,
    alpha::Number,
    beta::Number
)
    right_mul_dimcheck(C, A, S)
    # Built-in multiplication
    mul!(C, A, S.op, alpha, beta)
    return nothing
end
