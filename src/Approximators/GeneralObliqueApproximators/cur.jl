###########################################
# Include the CURCore definition
##########################################
include("CURCore.jl")

"""
    CUR <: Approximator

A struct that implements the CUR decomposition, a technique for computing a 
    low-rank approximation to a matrix, ``A``. This technique selects column subsets, 
    ``C``, row subsets, ``R``, and a core matrix, ``U`` 
    (this can either be ``C^\\dagger A R^\\dagger`` or ``A[I,J]``, 
    where ``I`` is a set of row indices and ``J`` is a set of column indices) such that 
    ``
        A \\approx CUR.
    ``

# Mathematical Description
In general finding a ``I`` and ``J`` that minimize the approximation error is a
    NP-hard problem (you're not going to do it) [shitov2021column](@cite); 
    thus, we often aim to find sufficiently 
    good indices. The best known quality of a rank ``r`` CUR approximation to a  matrix
    ``\\tilde A_r`` is known to be 
    ``
        \\|A - \\tilde A_r \\|_F \\leq (r + 1) \\|A - A_r\\|_F,
    ``
    where ``A_r`` is the rank-``r`` truncated SVD [osinsky2025close](@cite).

In practice numerous randomized methods match the performance of this best possible 
    selection procedure. These approaches can be broken into sampling and pivoting 
    approaches. The sampling approaches, like leverage score sampling 
    [mahoney2009cur](@cite), typically have better theory, and involve 
    imputing a distribution over the rows/columns of a matrix
    and drawing from that distribution with replacement. Randomized pivoting procedures
    on the other hand tend to be more efficient and accurate in practice and involve 
    compressing the matrix then applying a pivoting procedure on the compressed form
    [dong2023simpler](@cite).

# Fields
- `rank::Int64`, the desired rank of approximation.
- `oversample::Int64`, the amount of extra rows to sample. By default this zero. Making it a
    positive integer can improve the approximation quality [park2025accuracy](@cite).
- `col_selector::Selector`, the technique used for selecting column indices from a matrix.
- `row_selector::Selector`, the technique used for selecting row indices from a matrix.
- `core::CURCore`, the method for computing the core matrix, `U`, in the CUR.
- `block_size::Int64`, number of vectors stored in a buffer matrix for multiplication.
# Constructor
    CUR(rank;
        oversample = 0,
        selector_cols = LUPP(),
        selector_rows = selector_cols,
        core = CrossApproximation(),
        block_size = 0
    )

## Keywords 
- `rank::Int64`, the desired rank of approximation.
- `oversample::Int64`, the amount of extra rows to sample. By default this zero. Making it a
    positive integer can improve the approximation quality [park2025accuracy](@cite).
- `col_selector::Selector`, the technique used for selecting column indices from a matrix.
- `row_selector::Selector`, the technique used for selecting row indices from a matrix.
- `core::CURCore`, the method for computing the core matrix, `U`, in the CUR.
- `block_size::Int64`, number of vectors stored in a buffer matrix for multiplication. If 
    zero is inputted it set this to be the number of columns in `A`.

## Returns
- A `CUR` object.

## Throws
- An `ArgumentError`, if `rank`, `oversample`, or `blocksize` are less than zero.

"""
mutable struct CUR <: Approximator
    rank::Int64
    oversample::Int64
    col_selector::Selector
    row_selector::Selector
    core::CURCore
    block_size::Int64
    function CUR(rank, oversample, col_selector, row_selector, core, block_size)
        if rank < 0
            return throw(ArgumentError("Field `rank` must be non-negative."))
        end

        if oversample < 0
            return throw(ArgumentError("Field `oversample` must be non-negative."))
        end

        if block_size < 0 
            return throw(ArgumentError("Field `block_size` must be non-negative."))
        end

        return new(rank, oversample, col_selector, row_selector, core, block_size)
    end

end


function CUR(;
    rank = 1,
    oversample = 0,
    selector_cols = LUPP(),
    selector_rows = selector_cols,
    core = CrossApproximation(),
    block_size = 0
)
    return CUR(rank, oversample, selector_cols, selector_rows, core, block_size)
end

"""
    CURRecipe{CURCoreRecipe} <: ApproximatorRecipe

A struct that contains the preallocated memory and completed `Selectors` to form a
    CUR approximation to the matrix ``A``.
    
# Fields
- `n_rows::Int64`, the row dimension of the approximation. 
- `n_cols::Int64`, the column dimension of the approximation. 
- `n_row_vecs::Int64`, the number of rows selected in the approximation. 
- `n_col_vecs::Int64`, the number of columns selected in the approximation.
- `col_selector::SelectorRecipe`, the method for selecting columns.
- `row_selector::SelectorRecipe`, the method for selecting rows.
- `row_idx::Vector{Int64}`, the selected row indices.
- `col_idx::Vector{Int64}`, the selected column indices.
- `C::AbstractMatrix`, the entries of `A` at the selected column indices.
- `U::CURCoreRecipe`, the core matrix linking the `C` and `R` matrices to approximate `A`.
- `R::AbstractMatrix`, the entries of `A` at the selected row indices.
- `buffer_row::AbstractArray`, a buffer matrix used to store the result of `R` 
    multiplied with an array.
- `buffer_core::AbstractArray`, a buffer matrix used to store the result of `U` 
    multiplied with an array.
"""
mutable struct CURRecipe{CR<:CURCoreRecipe} <: ApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
    n_row_vecs::Int64
    n_col_vecs::Int64
    col_selector::SelectorRecipe
    row_selector::SelectorRecipe
    row_idx::Vector{Int64}
    col_idx::Vector{Int64}
    C::AbstractMatrix
    U::CR
    R::AbstractMatrix
    buffer_row::AbstractArray
    buffer_core::AbstractArray
end


###############################################
# Include files for core implementations
###############################################
include("./CURCore/cross_approximation.jl")

function complete_approximator(ingredients::CUR, A::AbstractMatrix)
    n_col_vecs = ingredients.rank
    n_row_vecs = ingredients.rank + ingredients.oversample
    if n_row_vecs > size(A, 1)
        throw(DomainError("`rank + oversample` is greater than number of rows."))
    end

    if n_col_vecs > size(A, 2)
        throw(DomainError("`rank` is greater than number of columns."))
    end

    if ingredients.block_size == 0
        block_size = size(A, 2) 
    else
        block_size = ingredients.block_size
    end

    col_idx = zeros(Int64, n_col_vecs)
    row_idx = zeros(Int64, n_row_vecs)
    col_selector = complete_selector(ingredients.col_selector, A)
    row_selector = complete_selector(ingredients.row_selector, A)
    C = typeof(A)(undef, size(A, 1), n_col_vecs)
    R = typeof(A)(undef, n_row_vecs, size(A, 2))
    U = complete_core(ingredients.core, ingredients, A)
    return CURRecipe(
        size(A, 1),
        size(A, 2),
        n_row_vecs, 
        n_col_vecs, 
        col_selector,
        row_selector,
        row_idx, 
        col_idx, 
        C, 
        U, 
        R, 
        # because of oversampling
        typeof(A)(undef, n_row_vecs, block_size),
        typeof(A)(undef, n_col_vecs, block_size)
    )
end

# implementations of rapproximate can be found with the core matrix implementations
# This is done to ensure stability as different core matrix approaches require 
# different organizations of the CUR selection

# Implement the muls
function mul!(
    C::AbstractArray, 
    approx::CURRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # determine size of buffer we have and how much we need
    # okay just checking row buffer because both have the same number of columns
    buff_r_n = size(approx.buffer_row, 2)
    if typeof(A) <: AbstractVector
        m = size(A)
        n = 1
    else
        m, n = size(A)
    end

    n_its = div(n, buff_r_n)
    last_block = rem(n, buff_r_n)
    # if the buffer is bigger than necessary set block_size to be the 
    # number of columns in A
    block_size = min(buff_r_n, n)
    # Perform first iteration to scale C by beta
    br_v = view(approx.buffer_row, :, 1:block_size)
    bc_v = view(approx.buffer_core, :, 1:block_size)
    Cv = view(C, :, 1:block_size)
    mul!(br_v, approx.R, A[:, 1:block_size])
    mul!(bc_v, approx.U, br_v)
    mul!(Cv, approx.C, bc_v, alpha, beta) 

    start = block_size + 1
    # perform remaining necessary iterations
    for i = 2:n_its
        last = start + block_size - 1
        # if in the loop using all columns of buffer
        br_v = view(approx.buffer_row, :, 1:block_size)
        bc_v = view(approx.buffer_core, :, 1:block_size)
        Cv = view(C, :, start:last)
        mul!(br_v, approx.R, A[:, start:last])
        mul!(bc_v, approx.U, br_v)
        mul!(Cv, approx.C, bc_v, alpha, beta)
        start = last + 1
    end

    # complete the last block which could differ from block size
    if n_its > 0 && last_block > 0
        last = start + last_block - 1
        # perform last block
        br_v = view(approx.buffer_row, :, 1:last_block)
        bc_v = view(approx.buffer_core, :, 1:last_block)
        Cv = view(C, :, start:last)
        mul!(br_v, approx.R, A[:, start:last])
        mul!(bc_v, approx.U, br_v)
        mul!(Cv, approx.C, bc_v, alpha, beta)
    end

    return
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    approx::CURRecipe, 
    alpha::Number, 
    beta::Number
)
    # determine size of buffer we have and how much we need
    # okay just checking row buffer because both have the same number of columns
    buff_r_n = size(approx.buffer_row, 2)
    m, n = size(A)

    n_its = div(m, buff_r_n)
    last_block = rem(m, buff_r_n)
    # if the buffer is bigger than necessary set block_size to be the 
    # number of columns in A
    block_size = min(buff_r_n, m)
    # Perform first iteration to scale C by beta
    br_v = view(approx.buffer_row, :, 1:block_size)
    bc_v = view(approx.buffer_core, :, 1:block_size)
    Cv = view(C, 1:block_size, :)
    mul!(bc_v', A[1:block_size, :], approx.C)
    mul!(br_v', bc_v', approx.U)
    mul!(Cv, br_v', approx.R, alpha, beta) 

    start = block_size + 1
    # perform remaining necessary iterations
    for i = 2:n_its
        last = start + block_size - 1
        # if in the loop using all columns of buffer
        br_v = view(approx.buffer_row, :, :)
        bc_v = view(approx.buffer_core, :, :)
        Cv = view(C, start:last, :) 
        mul!(bc_v', A[start:last, :], approx.C)
        mul!(br_v', bc_v', approx.U)
        mul!(Cv, br_v', approx.R, alpha, beta) 
        start = last + 1
    end

    # complete the last block which could differ from block size
    if n_its > 0 && last_block > 0
        last = start + last_block - 1
        # perform last block
        br_v = view(approx.buffer_row, :, 1:last_block)
        bc_v = view(approx.buffer_core, :, 1:last_block)
        Cv = view(C, start:last, :)
        mul!(bc_v', A[start:last, :], approx.C)
        mul!(br_v', bc_v', approx.U)
        mul!(Cv, br_v', approx.R, alpha, beta) 
    end

    return
end

# functions for CUR must be placed here because they require the definition 
# of CUR to work

"""
    complete_core()

A function that generates a `CURCoreRecipe` given the arguments.

# Arguments
- `ingredients::CURCore`, a data structure containing the user-defined parameters 
    associated with a particular type of core matrix in a CUR decomposition.
- `cur::CUR`, a data structure containing the user-defined parameters for the CUR 
    approximation.
- `A::AbstractMatrix`, a target matrix for approximation.
"""
function complete_core(core::CURCore, cur::CUR, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No method `complete_core` exists for `CURCore` of type \
            $(typeof(core)), `CUR`, and matrix of type $(typeof(A))."
        )
    ) 
end

# this must be placed here because it requires the CURCoreRecipe to be defined first
"""
    update_core!()

A function that updates the `CURCoreRecipe` based on the parameters defined in the `CUR`
    structure.

# Arguments
- `core::CURCoreRecipe`, a data structure containing the preallocated data structures 
    necessary for computing the core matrix in a CUR decomposition.
- `cur::CURRecipe`, a data structure containing the user-defined parameters and 
    preallocated structures.
- `A::AbstractMatrix`, a target matrix to be approximated.
"""
function update_core!(core::CURCoreRecipe, cur::CURRecipe, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No method `update_core!` exists for `CURCore` of type \
            $(typeof(core)), `CURRecipe`, and matrix of type $(typeof(A))."
        )
    )
end
