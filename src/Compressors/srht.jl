"""
    SRHT <: Compressor

An implementation of the Subsampled Randomized Hadamard Transform (SRHT) method. This 
technique applies a subsampling matrix, a Walsh-Hadamard transform, and a diagonal sign 
matrix to produce a sketch. See [tropp2011improved](@cite) for additional details.

# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress. If we want
to compress ``A`` from the left (i.e., we reduce the number of rows), then
we create a matrix, ``S``, with dimension ``s \\times m`` where
``s`` is the compression dimension that is supplied by the user.
Here ``S=KHD`` where 

- ``K`` is a matrix with  with dimension ``s \\times m``, where the rows are made up of a 
    random sample of the rows of a ``m \\times m`` identity matrix.
- ``H`` is a Hadamard matrix of dimension ``m \\times m``, which is implicitly applied 
    through the fast Walsh-Hadamard transform;
- ``D`` of is a diagonal matrix of dimension ``m \\times m`` with entries given by 
    independent Rademacher variables.

If we want to compress ``A`` from the right (i.e., we reduce the number of columns), then 
we would apply ``S=DHK`` from the right where the dimensions of the matrices are adjusted 
to reflect the number of columns in ``A``.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `block_size::Int64`, the number of vectors in the padding matrix.
- `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    SRHT(;
        cardinality = Left(),
        compression_dim::Int64=2,
        block_size::Int64=10,
        type::Type{N}=Float64
    ) where {N <: Number}

## Keywords
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description. By default this is set to 2.
- `block_size::Int64`, the number of vectors in the padding matrix.
- `type::Type{<:Number}`, the type of elements in the compressor.

## Returns
- A `SRHT` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive or if `block_size` is non-positive.
- `ArgumentError` if Cardinality is not either `Left` or `Right`.
"""
struct SRHT <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    block_size::Int64
    type::Type{<:Number}

    # perform checks on values of fields
    function SRHT(cardinality, compression_dim, block_size, type)
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        elseif block_size <= 0
            throw(ArgumentError("Field `block_size` must be positive."))
        elseif typeof(cardinality) == Undef
            throw(ArgumentError("Cardinality must be of type `Left` or `Right`."))
        end
            
        return new(cardinality, compression_dim, block_size, type)
    end

end

function SRHT(;
    cardinality::Cardinality = Left(),
    compression_dim::Int64=2,
    block_size::Int64=10,
    type::Type{N}=Float64
) where {N <: Number}
    return SRHT(cardinality, compression_dim, block_size, type)
end

"""
    SRHTRecipe{C<:Cardinality, M<:AbstractMatrix} <: CompressorRecipe

The recipe containing all allocations and information for the SRHT compressor.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to
    be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `scale::Float64`, the factor to ensure the isopmorphism of the sketch.
- `op::Vector{Int64}`, the vector of indices to be subsampled.
- `signs::BitVector`, the vector of signs where `0` indicates negative one and `1` indicates
    positive one. 
- `padding::AbstractMatrix`, the matrix containing the padding for the matrix being 
    sketched.

# Constructor
    SRHTRecipe(
        compression_dim::Int64,
        block_size::Int64,
        cardinality::Left,
        A::AbstractMatrix,
        type::Type{<:Number}
    )
   
## Keywords
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `block_size::Int64`, the number of vectors in the padding matrix.
- `cardinality::Left`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `A::AbstractMatrix`, the matrix being compressed.
- `type::Type{<:Number}`, the type of elements in the compressor.
"""
mutable struct SRHTRecipe{C<:Cardinality, M<:AbstractMatrix} <: CompressorRecipe
    cardinality::C
    n_rows::Int64
    n_cols::Int64
    scale::Float64
    op::Vector{Int64}
    signs::BitVector
    padding::M
end

function SRHTRecipe(
    compression_dim::Int64,
    block_size::Int64,
    cardinality::Left,
    A::AbstractMatrix,
    type::Type{<:Number}
)
    # For compressing from the left, the compressor's dimensions should be 
    # compression_dim by smallest power of 2 larger than size(A, 1)
    n_rows = compression_dim
    n_cols = size(A, 1)
    # Find nearest power 2 and allocate
    padded_size = Int64(2^(ceil(log(2, n_cols))))
    # Generate the padded matrix and signs which need padded size
    padded_matrix = zeros(type, padded_size, block_size)
    signs = bitrand(padded_size)
    idx = sample(1:padded_size, compression_dim, replace = false)
    scaling = type(sqrt(compression_dim) / sqrt(padded_size))
    return SRHTRecipe{typeof(cardinality), typeof(padded_matrix)}(
        cardinality,
        n_rows,
        n_cols,
        scaling,
        idx,
        signs,
        padded_matrix
    )
end

function SRHTRecipe(
    compression_dim::Int64,
    block_size::Int64,
    cardinality::Right,
    A::AbstractMatrix,
    type::Type{<:Number}
)
    # For compressing from the right, the compressor's dimensions should be 
    # compression_dim by smallest power of 2 larger than size(A, 2)
    n_cols = compression_dim
    n_rows = size(A, 2)
    # Find nearest power 2 and allocate
    padded_size = Int64(2^(ceil(log(2, n_rows))))
    # Generate the padded matrix and signs which need padded size
    padded_matrix = zeros(type, block_size, padded_size)
    signs = bitrand(padded_size)
    idx = sample(1:padded_size, compression_dim, replace = false)
    scaling = type(sqrt(compression_dim) / sqrt(padded_size))
    return SRHTRecipe{typeof(cardinality), typeof(padded_matrix)}(
        cardinality,
        n_rows,
        n_cols,
        scaling,
        idx,
        signs,
        padded_matrix
    )
end

function complete_compressor(ingredients::SRHT, A::AbstractMatrix)
    return SRHTRecipe(
        ingredients.compression_dim,
        ingredients.block_size,
        ingredients.cardinality,
        A,
        ingredients.type
    )
end

# Because the padded size depends on cardinality the update compressor functions 
# must also depend on cardinality
function update_compressor!(S::SRHTRecipe{Left, <:AbstractMatrix})
    padded_size = size(S.padding, 1)
    sample!(1:padded_size, S.op, replace = false)
    rand!(S.signs)
    return nothing
end

function update_compressor!(S::SRHTRecipe{Right, <:AbstractMatrix})
    padded_size = size(S.padding, 2)
    sample!(1:padded_size, S.op, replace = false)
    rand!(S.signs)
    return nothing
end

# Calculates S * A and stores it in C when S has Left cardinality
function mul!(
    C::AbstractArray, 
    S::SRHTRecipe{Left, <:AbstractMatrix},
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check dimensions
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    type = eltype(S.op)
    b_size = size(S.padding, 2)
    left_mul_dimcheck(C, S, A)

    # To be memory efficient we apply SRHT block-wise in column blocks
    last_block_size = rem(a_cols, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_cols, b_size)
    start_col = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_col = start_col + b_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, 1:a_rows, :)
        # ensure that padding matrix is set to zeros
        # if it becomes relevant; we should copyto!(pv, Av) first and then 
        # zero out the remaining entries of S.padding; this is a minor optimization.
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, :, i)
            fwht!(pv, S.signs, scaling = S.scale) 
        end

        # Apply the operator to the matrix
        Cv .= beta .* Cv .+ alpha .* S.padding[S.op, :]
        start_col = last_col + 1
    end
    
    # Handle the last block that is less than the block size, if it exists.
    if last_block_size > 0
        last_col = start_col + last_block_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, 1:a_rows, 1:last_block_size)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, :, i)
            fwht!(pv, S.signs, scaling = S.scale) 
        end
        
        # Perform accesses only up to the entries
        pv = view(S.padding, :, 1:last_block_size)
        # Apply the operator to the matrix
        Cv .= beta .* Cv .+ alpha .* pv[S.op, :]
    end

    return nothing 
end

# Calculates A*S and stores it in C  when S has left cardinality
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SRHTRecipe{Left, <:AbstractMatrix}, 
    alpha::Number, 
    beta::Number
)
    # Check dimensions
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    b_size = size(S.padding, 2)
    type = eltype(S.op)
    right_mul_dimcheck(C, A, S)

    # To be memory efficient we apply SRHT block-wise in column blocks
    last_block_size = rem(a_rows, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_rows, b_size)
    start_row = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_row = start_row + b_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, :)
        pv = view(S.padding, S.op, :)
        # Everything should be stored in the transpose of padding matrix because 
        # the padding matrix is column oriented
        fill!(S.padding, zero(type))
        pv' .= alpha .* Av 
        # Apply signs and fwht to the padding matrix along the columns of the padding matrix
        # this is equivalent to applying the hadamard transform to the rows of AS.op 
        # as desired
        for i in 1:b_size
            pv = view(S.padding, :, i)
            fwht!(pv, scaling = S.scale) 
        end

        # Because apply sign transform after hadmard is different than the reverse can't
        # using fwht with signs. Scale the rows of the 
        S.padding .*= ifelse.(S.signs, 1, -1) 
        pv = view(S.padding, 1:c_cols, :)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
        start_row = last_row + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_row = start_row + last_block_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, :)
        pv = view(S.padding, S.op, 1:last_block_size)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is structured with more rows than columns
        fill!(S.padding, zero(type))
        pv' .= alpha .* Av 
        # Apply signs and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, :, i)
            fwht!(pv, scaling = S.scale) 
        end
        
        # Because apply sign transform after hadamard is different than the reverse can't
        # using fwht with signs
        S.padding .*= ifelse.(S.signs, 1, -1) 
        pv = view(S.padding, 1:c_cols, 1:last_block_size)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
    end

    return nothing 
end

# Calculates A * S and stores it in C when S has Right cardinality
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SRHTRecipe{Right, <:AbstractMatrix}, 
    alpha::Number, 
    beta::Number
)
    # Check dimensions
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    type = eltype(S.op)
    b_size = size(S.padding, 1)
    right_mul_dimcheck(C, A, S)

    # To be memory efficient we apply SRHT block-wise in column blocks
    last_block_size = rem(a_rows, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_rows, b_size)
    start_row = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_row = start_row + b_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, :)
        pv = view(S.padding, :, 1:a_cols)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, i, :)
            fwht!(pv, S.signs, scaling = S.scale) 
        end

        # Apply the operator to the matrix
        Cv .= beta .* Cv .+ alpha .* S.padding[:, S.op]
        start_row = last_row + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_row = start_row + last_block_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, :)
        pv = view(S.padding, 1:last_block_size, 1:a_cols)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, i, :)
            fwht!(pv, S.signs, scaling = S.scale) 
        end

        pv = view(S.padding, 1:last_block_size, :)
        # Apply the operator to the matrix
        Cv .= beta .* Cv .+ alpha .* pv[:, S.op]
    end

    return nothing
end

# Calculates S * A and stores it in C when S has Right cardinality
function mul!(
    C::AbstractArray, 
    S::SRHTRecipe{Right, <:AbstractMatrix}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check dimensions
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    b_size = size(S.padding, 1)
    type = eltype(S.op)
    left_mul_dimcheck(C, S, A)

    # To be memory efficient we apply SRHT block-wise in column blocks
    last_block_size = rem(a_cols, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_cols, b_size)
    start_col = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_col = start_col + b_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, :, S.op)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is structured with more rows than columns
        fill!(S.padding, zero(type))
        pv' .= alpha .* Av 
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, i, :)
            fwht!(pv, scaling = S.scale) 
        end

        # Flip the signs
        S.padding' .*= ifelse.(S.signs, 1, -1)
        pv = view(S.padding, :, 1:c_rows)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
        start_col = last_col + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_col = start_col + last_block_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, 1:last_block_size, S.op)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is structured with more rows than columns
        fill!(S.padding, zero(type))
        pv' .= alpha .* Av 
        # Apply and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, i, :)
            fwht!(pv, scaling = S.scale)
        end
        
        # Apply signs to the padding matrix
        S.padding' .*= ifelse.(S.signs, 1, -1)
        # Because the padding 
        # Match padding matrix view to the output size
        pv = view(S.padding, 1:last_block_size, 1:c_rows)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
    end

    return nothing
end
