"""
    CountSketch <: Compressor

An implementation of the count sketch compression method. See additional details in 
[woodruff2014sketching](@cite) Section 2.1, in which the CountSketch matrix is equivalently defined as sparse 
embedding matrix.


# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress. If we want to compress ``A`` from 
the left (i.e., we reduce the number of rows), then we construct a Count Sketch matrix ``S`` with 
dimension ``s \\times m``, where ``s`` is the user-specified compression dimension. Each column of 
``S`` is generated independently by the following steps:

1. Randomly select an integer between 1 and ``s`` to determine the row position of the nonzero entry.
2. Assign this entry a value of either +1 or -1, chosen uniformly at random.
3. Set all the other entries in the column to zero.

As a result, each column of S has exactly one nonzero element.

If ``A`` is compressed from the right, then we construct a Count Sketch matrix ``S`` with dimension 
``n \\times s``, where ``s`` is the user-specified compression dimension. Each row of ``S`` is 
generated independently using the following steps:

1. Randomly select an integer between 1 and ``s`` to determine the column position of the nonzero entry.
2. Assign this entry a value of either +1 or -1, chosen uniformly at random.
3. Set all other entries in the row to zero.

In this case, each row of S has exactly one nonzero entry. 
The compressed matrix is then formed by multiplying S A (for left compression) or A S (for right compression).

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    CountSketch(;carinality=Left(), compression_dim=2, type=Float64)

## Keywords
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
- `type::Type{<:Number}`, the type of the elements in the compressor. By default is set
    to Float64.

## Returns
- A `CountSketch` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive
- `ArgumentError` if `Undef()` is taken as the input for `cardinality`
"""
struct CountSketch <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    type::Type{<:Number}
    # Check on the compression dimension
    function CountSketch(cardinality, compression_dim, type)
        if compression_dim <= 0
            throw(ArgumentError("Field 'compression_dim' must be positive."))
        end

        if cardinality == Undef()
            throw(
                ArgumentError(
                    "`cardinality` must be specified as `Left()` or `Right()`.\
                    `Undef()` is not allowed in `CountSketch` structure."
                )
            )
        end

        return new(cardinality, compression_dim, type)
    end
end

function CountSketch(;
    cardinality::Cardinality=Left(),
    compression_dim::Int64=2,
    type::Type{<:Number}=Float64
)
    # Partially construct the count sketch datatype
    return CountSketch(cardinality, compression_dim, type)
end

"""
    CountSketchRecipe <: CompressorRecipe

The recipe containing all allocations and information for the CountSketch compressor.

# Fields
- `cardinality::C`, the cardinality of the compressor. The
    value is either `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `mat::S`, the compression matrix stored in a sparse form.
"""
mutable struct CountSketchRecipe{C<:Cardinality,S<:SparseMatrixCSC} <: CompressorRecipe
    cardinality::C
    compression_dim::Int64
    n_rows::Int64
    n_cols::Int64
    mat::S
end

function CountSketchRecipe(
    cardinality::Left,
    compression_dim::Int64,
    A::AbstractMatrix,
    type::Type{<:Number},
)
    # determine the initial size
    n_rows = compression_dim
    n_cols = size(A, 1)
    initial_size = n_cols
    # assign -1 or +1 in every row/column with probability 0.5
    signs = rand([type(-1.0), type(1.0)], initial_size)
    groups = rand(1:compression_dim, initial_size)
    ptr = collect(1:initial_size)
    mat = sparse(groups, ptr, signs, n_rows, n_cols)
    return CountSketchRecipe{Left,typeof(mat)}(cardinality, compression_dim, n_rows, n_cols, mat)
end

function CountSketchRecipe(
    cardinality::Right,
    compression_dim::Int64,
    A::AbstractMatrix,
    type::Type{<:Number},
)
    # determine the initial size
    n_rows = size(A, 2)
    n_cols = compression_dim
    initial_size = n_rows
    # assign -1 or +1 in every row/column with probability 0.5
    signs = rand([type(-1.0), type(1.0)], initial_size)
    groups = rand(1:compression_dim, initial_size)
    ptr = collect(1:initial_size)
    mat = sparse(groups, ptr, signs, n_cols, n_rows)
    return CountSketchRecipe{Right,typeof(mat)}(cardinality, compression_dim, n_rows, n_cols, mat)
end

function complete_compressor(ingredients::CountSketch, A::AbstractMatrix)
    return CountSketchRecipe(
        ingredients.cardinality,
        ingredients.compression_dim,
        A,
        ingredients.type,
    )
end

function update_compressor!(S::CountSketchRecipe)
    nzval = S.mat.nzval
    @inbounds for i in eachindex(nzval)
        nzval[i] = ifelse(rand(Bool), 1.0, -1.0)
    end
    rand!(S.mat.rowval, 1:S.compression_dim)
    return nothing
end

# Calculates S.mat * A and stores it in C 
function mul!(
    C::AbstractArray, 
    S::CountSketchRecipe{Left}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    left_mul_dimcheck(C, S, A)
    return mul!(C, S.mat, A, alpha, beta)
end

# Calculates A * S.mat and stores it in C 
function mul!(
    C::AbstractArray, 
    A::AbstractArray,
    S::CountSketchRecipe{Left},
    alpha::Number,
    beta::Number
)
    right_mul_dimcheck(C, A, S)
    mul!(C, A, S.mat, alpha, beta)
    return nothing
end

# Calculates S.mat' * A and stores it in C
function mul!(
    C::AbstractArray,
    S::CountSketchRecipe{Right},
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    left_mul_dimcheck(C, S, A)
    return mul!(C, S.mat', A, alpha, beta)
end

# Calculates A * S.mat' and stores it in C
function mul!(
    C::AbstractArray,
    A::AbstractArray,
    S::CountSketchRecipe{Right},
    alpha::Number,
    beta::Number
)
    right_mul_dimcheck(C, A, S)
    mul!(C, A, S.mat', alpha, beta)
    return nothing
end

function mul!(
    C::AbstractMatrix,
    S::CountSketchRecipe{Left},
    A::Transpose{T, <:SparseMatrixCSC},
    alpha::Number,
    beta::Number
) where T
    left_mul_dimcheck(C, S, A)

    # Optimized implementation for Transpose of SparseMatrixCSC
    # C = alpha * S.mat * A + beta * C
    # A = B' where B is SparseMatrixCSC

    B = parent(A)
    S_mat = S.mat

    if beta != 1
        if beta == 0
            fill!(C, 0)
        else
            rmul!(C, beta)
        end
    end

    S_rows = rowvals(S_mat)
    S_nz = nonzeros(S_mat)
    B_rows = rowvals(B)
    B_nz = nonzeros(B)

    # Iterate over columns of S_mat (which correspond to rows of A)
    @inbounds for j in 1:size(S_mat, 2)
        rng_S = nzrange(S_mat, j)
        for k_S in rng_S
            row_S = S_rows[k_S]
            val_S = S_nz[k_S]

            # Add val_S * (row j of A) to row row_S of C
            # Row j of A is Column j of B
            rng_B = nzrange(B, j)
            for k_B in rng_B
                col_C = B_rows[k_B] # row in B -> col in C
                val_B = B_nz[k_B]
                C[row_S, col_C] += alpha * val_S * val_B
            end
        end
    end
    return nothing
end
