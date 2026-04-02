"""
    LUPP <: Selector

A `Selector` that implements LU with partial pivoting for selecting column indices from a 
matrix.

# Fields
- `compressor::Compressor`, the compression technique that will be applied to the matrix, 
    before selecting indices.

# Constructor
    LUPP(;compressor = Identity())

## Keywords
- `compressor::Compressor`, the compression technique that will be applied to the matrix, 
    before selecting indices. Defaults to the `Identity` compressor.

## Returns
- A `LUPP` object.

!!! note "Implementation Note" 
    LU with partial pivoting is classically implemented to select rows of a matrix. Here we 
    apply LU with partial pivoting to the transpose of the inputted matrix to select 
    columns.
"""
mutable struct LUPP <: Selector
    compressor::Compressor
end
 
function LUPP(;compressor = Identity()) 
    return LUPP(compressor)
end

"""
    LUPPRecipe <: SelectorRecipe

A `SelectorRecipe` that contains all the necessary preallocations for selecting column 
indices from a matrix using LU with partial pivoting.


# Fields
- `compressor::CompressorRecipe`, the compression technique that will applied to the matrix, 
    before selecting indices.
- `SA::AbstractMatrix`, a buffer matrix for storing the sketched matrix.

!!! note "Implementation Note" 
    LU with partial pivoting is classically implemented to select rows of a matrix. Here we 
    apply LU with partial pivoting to the transpose of the inputted matrix to select 
    columns.
"""
mutable struct LUPPRecipe <: SelectorRecipe
    compressor::CompressorRecipe
    SA::AbstractMatrix
end

function complete_selector(ingredients::LUPP, A::AbstractMatrix)
    compressor = complete_compressor(ingredients.compressor, A)
    n_rows, n_cols = size(compressor)
    SA = similar(A, eltype(A), n_rows, n_cols)
    return LUPPRecipe(compressor, SA)
end

function update_selector!(selector::LUPPRecipe)
    update_compressor!(selector.compressor)
    return nothing
end

function select_indices!(
    idx::AbstractVector,
    selector::LUPPRecipe, 
    A::AbstractMatrix,
    n_idx::Int64, 
    start_idx::Int64
)
    # you cannot select more column indices than there are columns in the matrix
    if n_idx > size(A, 2)
        throw(
            DimensionMismatch( 
                "`n_idx` cannot be larger than the number of columns in `A`."
            )
        )
    end

    # start_idx + n_idx must be less than the length of the idx vector
    if start_idx + n_idx - 1 > size(idx, 1)
        throw(
            DimensionMismatch( 
                "`start_idx` + `n_idx` - 1 cannot be larger than the lenght of `idx`."
            )   
        )
    end

    # you cannot select more indices than the compression dimension because that is when 
    # LUPP will stop selecting new pivots because the LU factorization will have been formed
    if n_idx > selector.compressor.n_rows
        throw(
            DimensionMismatch( 
                "Must select fewer indices then the `compression_dim`."
            )   
        )

    end
    
    mul!(selector.SA, selector.compressor, A)
    # because LUPP selects rows and selectors select columns we need to pivot on A'
    p = lu!(selector.SA').p
    # store n_idx indices in the appropriate part of the idx
    idx[start_idx:start_idx + n_idx - 1] = p[1:n_idx]
    return nothing
end
