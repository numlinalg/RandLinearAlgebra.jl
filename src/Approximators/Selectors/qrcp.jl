"""
    QRCP <: Selector

A `Selector` that implements QR with column norm pivoting for selecting column indices from  
a matrix.

# Fields
- `compressor::Compressor`, the compression technique that will be applied to the matrix, 
    before selecting indices.

# Constructor
    QRCP(;compressor = Identity())

## Keywords
- `compressor::Compressor`, the compression technique that will applied to the matrix, 
    before selecting indices. Defaults the [Identity](@ref) compressor.

## Returns
- A `QRCP` object.
"""
mutable struct QRCP <: Selector
    compressor::Compressor
end

function QRCP(;compressor = Identity()) 
    QRCP(compressor)
end

"""
    QRCPRecipe <: SelectorRecipe

A `SelectorRecipe` that contains all the necessary preallocations for selecting column 
indices from a matrix using QR with column norm pivoting.

# Fields
- `compressor::Compressor`, the compression technique that will be applied to the matrix, 
    before selecting indices.
- `SA::AbstractMatrix`, a buffer matrix for storing the compressed matrix.
"""
mutable struct QRCPRecipe <: SelectorRecipe
    compressor::CompressorRecipe
    SA::AbstractMatrix
end

function complete_selector(ingredients::QRCP, A::AbstractMatrix)
    compressor = complete_compressor(ingredients.compressor, A)
    SA = similar(A, eltype(A), size(compressor, 1), size(A, 2))
    return QRCPRecipe(compressor, SA)
end

function update_selector!(selector::QRCPRecipe)
    update_compressor!(selector.compressor)
    return nothing
end

function select_indices!(
    idx::AbstractVector,
    selector::QRCPRecipe, 
    A::AbstractMatrix,
    n_idx::Int64, 
    start_idx::Int64
)
    if n_idx > size(A, 2)
        throw(
            DimensionMismatch( 
                "`n_idx` cannot be larger than the number of columns in `A`."
            )
        )
    end

    if start_idx + n_idx - 1 > size(idx, 1)
        throw(
            DimensionMismatch( 
                "`start_idx` + `n_idx` - 1 cannot be larger than the lenght of `idx`."
            )   
        )
    end

    if n_idx > selector.compressor.n_rows
        throw(
            DimensionMismatch( 
                "Must select fewer indices then the `compression_dim`."
            )   
        )

    end
    
    mul!(selector.SA, selector.compressor, A)
    p = qr!(selector.SA, ColumnNorm()).p
    # store newly selected indices at inputted index storage points 
    idx[start_idx:start_idx + n_idx - 1] = p[1:n_idx]
    return nothing
end
