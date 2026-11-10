"""
    Sampling <: Compressor

This method subsets the rows 
or columns of a matrix according to a user-supplied distribution. The size of the 
subset is also provided by the user.

# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress.

If we want to compress ``A`` from the left (i.e., we reduce the number of rows), then
we create an index set to contain all the indices of selected rows. The indices are 
chosen by sampling over all the rows with the user-specified distribution in the 
`distribution` field.

If ``A`` is compressed from the right (i.e., we reduce the number of columns), then
we create an index set to contain all the indices of selected columns. The indices 
are chosen by sampling over all the columns with the user-specified distribution 
in the `distribution` field.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension.
- `distribution::Distribution`, the distribution being used to assign probability weights
    on the indices.

# Constructor

    Sampling(;cardinality = Left(), compression_dim = 2, distribution)

## Arguments
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim::Int64`, the target compression dimension. By default this is set to 2.
- `distribution::Distribution`, the distribution being used to assign probability weights
    on the indices. By default this is set as `Uniform` distribution.

## Returns
- A `Sampling` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive
- `ArgumentError` if `Undef()` is taken as the input for `cardinality`
"""
struct Sampling <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    distribution::Distribution # Function that returns a probability vector over indices
    function Sampling(cardinality, compression_dim, distribution)
        # the compression dimension must be positive
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        end

        if cardinality == Undef()
            throw(ArgumentError("`cardinality` must be specified as `Left()` or `Right()`.\
            `Undef()` is not allowed in `Sampling` structure."))
        end

        return new(cardinality, compression_dim, distribution)
    end
end

function Sampling(;
    cardinality::Cardinality=Left(),
    compression_dim::Int64=2,
    distribution::Distribution=Uniform()
)
    # Partially construct the Sampling datatype
    return Sampling(cardinality, compression_dim, distribution)
end

"""
    SamplingRecipe{C<:Cardinality, D<:DistributionRecipe, V<:SubArray} <: CompressorRecipe

The recipe containing all allocations and information for the sampling compressor.

# Fields
- `cardinality::C`, the cardinality of the compressor. The
    value is either `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension.
- `n_rows::Int64`, number of rows of compression matrix.
- `n_cols::Int64`, number of columns of compression matrix.
- `distribution_recipe::D`, the user-specified distribution recipe.
- `idx::Vector{Int64}`, the index set that contains all the chosen indices.
- `idx_v::V`, the view of the `idx`.
"""
mutable struct SamplingRecipe{C<:Cardinality,D<:DistributionRecipe,V<:SubArray} <:
               CompressorRecipe
    cardinality::C
    compression_dim::Int64
    n_rows::Int64
    n_cols::Int64
    distribution_recipe::D
    idx::Vector{Int64}
    idx_v::V
end

# Computes the dimensions of the CompressorRecipe 
function get_dims(compression_dim::Int64, cardinality::Left, A::AbstractMatrix)
    n_rows = compression_dim
    n_cols = size(A, 1)
    return n_rows, n_cols
end

# Computes the dimensions of the CompressorRecipe 
function get_dims(compression_dim::Int64, cardinality::Right, A::AbstractMatrix)
    n_rows = size(A, 2)
    n_cols = compression_dim
    return n_rows, n_cols
end

function complete_compressor(sub_sampling::Sampling, A::AbstractMatrix)
    n_rows, n_cols = get_dims(sub_sampling.compression_dim, sub_sampling.cardinality, A)
    # Pull out the variables from ingredients
    compression_dim = sub_sampling.compression_dim
    sub_sampling.distribution.cardinality = sub_sampling.cardinality
    # Compute the weight for each index
    dist_recipe = complete_distribution(sub_sampling.distribution, A)
    idx = Vector{Int64}(undef, compression_dim)
    idx_v = view(idx,:)
    # Randomly generate samples from index set based on weights
    sample_distribution!(idx, dist_recipe)
    return SamplingRecipe{typeof(sub_sampling.cardinality),typeof(dist_recipe),typeof(idx_v)}(
        sub_sampling.cardinality,
        compression_dim,
        n_rows,
        n_cols,
        dist_recipe,
        idx,
        idx_v,
    )
end

function update_compressor!(S::SamplingRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    update_distribution!(S.distribution_recipe, x, A, b)
    # Randomly generate samples from index set based on weights
    sample_distribution!(S.idx_v, S.distribution_recipe)
end
    
# Matrix-matrix multiplication
# Begin with the left version
function mul!(
    C::AbstractArray, 
    S::SamplingRecipe{Left}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    left_mul_dimcheck(C, S, A)
    # if C and A are vectors
    if size(C, 2) == 1
       A_sub = view(A, S.idx_v)
       axpby!(alpha, A_sub, beta, C)
    #if C and A are matrices
    else
        A_sub = view(A, S.idx_v, :)
        mul!(C, A_sub, I, alpha, beta)
    end

    return nothing
end

function mul!(
    C::AbstractArray, 
    S::SamplingRecipe{Right}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    #left_mul_dimcheck(C, S, A)
    C .*= beta 
    if size(C, 2) == 1
       C_sub = view(C, S.idx_v)
       axpby!(alpha, A, 1, C_sub)
    else
        C_sub = view(C, S.idx_v, :)
        mul!(C_sub, A, I, alpha, 1)
    end

    return nothing
end

# right version
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SamplingRecipe{Right}, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    right_mul_dimcheck(C, A, S)
    A_sub = view(A, :, S.idx_v)
    mul!(C, A_sub, I, alpha, beta)

    return nothing
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SamplingRecipe{Left}, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    right_mul_dimcheck(C, A, S)
    C .*= beta
    C_sub = view(C, :, S.idx_v)
    mul!(C_sub, A, I, alpha, 1)

    return nothing
end

###############################################################################
# Binary Operator Compressor-Array Multiplications for sparse matrices/vectors
###############################################################################
# S * A
function (*)(S::SamplingRecipe, A::Union{SparseMatrixCSC, SparseVector})
    s_rows = size(S, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? spzeros(eltype(A), s_rows) : spzeros(eltype(A), s_rows, a_cols)
    mul!(C, S, A)
    return C
end

# A * S
function (*)(A::Union{SparseMatrixCSC, SparseVector}, S::SamplingRecipe)
    s_cols = size(S, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? spzeros(eltype(A), s_cols)' : spzeros(eltype(A), a_rows, s_cols)
    mul!(C, A, S)
    return C
end

# S' * A
function (*)(S::CompressorAdjoint{<:SamplingRecipe}, A::Union{SparseMatrixCSC, SparseVector})
    s_rows = size(S, 1)
    a_cols = size(A, 2)
    C = a_cols == 1 ? spzeros(eltype(A), s_rows) : spzeros(eltype(A), s_rows, a_cols)
    mul!(C, S, A)
    return C
end

# A * S'
function (*)(A::Union{SparseMatrixCSC, SparseVector}, S::CompressorAdjoint{<:SamplingRecipe})
    s_cols = size(S, 2)
    a_rows = size(A, 1)
    C = a_rows == 1 ? spzeros(eltype(A), s_cols)' : spzeros(eltype(A), a_rows, s_cols)
    mul!(C, A, S)
    return C
end

function mul!(
    C::AbstractMatrix,
    S::SamplingRecipe{Left},
    A::Transpose{T, <:SparseMatrixCSC},
    alpha::Number,
    beta::Number
) where T
    # Fast path for Transpose of Sparse
    # We want C = alpha * A[S.idx_v, :] + beta * C
    # A[rows, :] is (B[:, rows])' where B = parent(A)

    left_mul_dimcheck(C, S, A)

    B = parent(A)
    rows = S.idx_v

    # Handle beta
    if beta != 1
        if beta == 0
            fill!(C, 0)
        else
            rmul!(C, beta)
        end
    end

    # Add alpha * A_sub
    # A_sub rows are columns of B
    # C is dense (assumed)

    rv = rowvals(B)
    nz = nonzeros(B)

    @inbounds for (i, r) in enumerate(rows)
        rng = nzrange(B, r)
        for k in rng
            row = rv[k]
            val = nz[k]
            C[i, row] += alpha * val
        end
    end
    return nothing
end
