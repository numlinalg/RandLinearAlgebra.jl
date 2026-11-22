###########################################
# Include the core definition
##########################################
include("generalized_nystrom_core.jl")

"""
    GeneralizedNystrom{CS,RS,C} <: Approximator

A struct that implements the generalized Nyström decomposition, a technique for computing a
    low-rank approximation to a matrix, ``A``. This technique forms a column operator applied
    to ``A`` to produce ``C``, a row operator applied to ``A`` to produce ``R``, and a core
    matrix, ``U``, such that
    ``
        A \\approx CUR.
    ``

The column and row operators may each be either a [`Selector`](@ref) (selecting actual
    column/row subsets, as in a CUR decomposition) or a [`Compressor`](@ref) (sketching the
    column/row space, as in the classical generalized Nyström method). The CUR decomposition
    is therefore the special case in which both operators are selectors; the type alias
    [`CUR`](@ref) refers to exactly this case.

# Type Parameters
- `CS<:Union{Selector,Compressor}`, the type of the column operator.
- `RS<:Union{Selector,Compressor}`, the type of the row operator.
- `C<:GeneralizedNystromCore`, the type of the core method.

# Mathematical Description
In the CUR case, ``C`` and ``R`` are subsets of the columns and rows of ``A`` and the core
    ``U`` is formed from their intersection. In general finding row indices ``I`` and column
    indices ``J`` that minimize the approximation error is an NP-hard problem
    [shitov2021column](@cite); thus, we often aim to find sufficiently good indices. The best
    known quality of a rank ``r`` CUR approximation to a matrix ``\\tilde A_r`` is known to be
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
- `col_selector::Union{Selector,Compressor}`, the operator used to form the columns, `C`.
- `row_selector::Union{Selector,Compressor}`, the operator used to form the rows, `R`.
- `core::GeneralizedNystromCore`, the method for computing the core matrix, `U`.
- `block_size::Int64`, number of vectors stored in a buffer matrix for multiplication.
# Constructor
    GeneralizedNystrom(rank;
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
- `selector_cols::Union{Selector,Compressor}`, the operator used to form the columns, `C`.
- `selector_rows::Union{Selector,Compressor}`, the operator used to form the rows, `R`.
- `core::GeneralizedNystromCore`, the method for computing the core matrix, `U`.
- `block_size::Int64`, number of vectors stored in a buffer matrix for multiplication. If
    zero is inputted it set this to be the number of columns in `A`.

## Returns
- A `GeneralizedNystrom` object.

## Throws
- An `ArgumentError`, if `rank`, `oversample`, or `blocksize` are less than zero.

"""
mutable struct GeneralizedNystrom{
    CS<:Union{Selector,Compressor},
    RS<:Union{Selector,Compressor},
    C<:GeneralizedNystromCore,
} <: Approximator
    rank::Int64
    oversample::Int64
    col_selector::CS
    row_selector::RS
    core::C
    block_size::Int64
    function GeneralizedNystrom(
        rank, oversample, col_selector::CS, row_selector::RS, core::C, block_size
    ) where {
        CS<:Union{Selector,Compressor},
        RS<:Union{Selector,Compressor},
        C<:GeneralizedNystromCore,
    }
        if rank < 0
            return throw(ArgumentError("Field `rank` must be non-negative."))
        end

        if oversample < 0
            return throw(ArgumentError("Field `oversample` must be non-negative."))
        end

        if block_size < 0
            return throw(ArgumentError("Field `block_size` must be non-negative."))
        end

        return new{CS,RS,C}(
            rank, oversample, col_selector, row_selector, core, block_size
        )
    end

end


function GeneralizedNystrom(;
    rank = 1,
    oversample = 0,
    selector_cols = LUPP(),
    selector_rows = selector_cols,
    core = CrossApproximation(),
    block_size = 0
)
    return GeneralizedNystrom(
        rank, oversample, selector_cols, selector_rows, core, block_size
    )
end

"""
    CUR

A type alias for a [`GeneralizedNystrom`](@ref) whose column and row operators are both
    [`Selector`](@ref)s, i.e. the classical CUR decomposition. Constructing a `CUR` forwards
    to the `GeneralizedNystrom` constructor, and an object is a `CUR` exactly when both of
    its operators are selectors.
"""
const CUR = GeneralizedNystrom{<:Selector,<:Selector}

CUR(; kwargs...) = GeneralizedNystrom(; kwargs...)
function CUR(rank, oversample, col_selector, row_selector, core, block_size)
    return GeneralizedNystrom(
        rank, oversample, col_selector, row_selector, core, block_size
    )
end

"""
    GeneralizedNystromRecipe{CR,CS,RS} <: ApproximatorRecipe

A struct that contains the preallocated memory and completed operators to form a
    generalized Nyström approximation to the matrix ``A``.

# Type Parameters
- `CR<:GeneralizedNystromCoreRecipe`, the type of the completed core method.
- `CS<:Union{SelectorRecipe,CompressorRecipe}`, the type of the completed column operator.
- `RS<:Union{SelectorRecipe,CompressorRecipe}`, the type of the completed row operator.

# Fields
- `n_rows::Int64`, the row dimension of the approximation.
- `n_cols::Int64`, the column dimension of the approximation.
- `n_row_vecs::Int64`, the number of rows selected in the approximation.
- `n_col_vecs::Int64`, the number of columns selected in the approximation.
- `col_selector::Union{SelectorRecipe,CompressorRecipe}`, the completed column operator.
- `row_selector::Union{SelectorRecipe,CompressorRecipe}`, the completed row operator.
- `row_idx::Vector{Int64}`, the selected row indices.
- `col_idx::Vector{Int64}`, the selected column indices.
- `C::AbstractMatrix`, the entries of `A` at the selected column indices.
- `U::GeneralizedNystromCoreRecipe`, the core matrix linking the `C` and `R` matrices to
    approximate `A`.
- `R::AbstractMatrix`, the entries of `A` at the selected row indices.
- `buffer_row::AbstractArray`, a buffer matrix used to store the result of `R`
    multiplied with an array.
- `buffer_core::AbstractArray`, a buffer matrix used to store the result of `U`
    multiplied with an array.
"""
mutable struct GeneralizedNystromRecipe{
    CR<:GeneralizedNystromCoreRecipe,
    CS<:Union{SelectorRecipe,CompressorRecipe},
    RS<:Union{SelectorRecipe,CompressorRecipe},
} <: ApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
    n_row_vecs::Int64
    n_col_vecs::Int64
    col_selector::CS
    row_selector::RS
    row_idx::Vector{Int64}
    col_idx::Vector{Int64}
    C::AbstractMatrix
    U::CR
    R::AbstractMatrix
    buffer_row::AbstractArray
    buffer_core::AbstractArray
end

"""
    CURRecipe

A type alias for a [`GeneralizedNystromRecipe`](@ref) whose column and row operators are both
    [`SelectorRecipe`](@ref)s, i.e. the completed form of a CUR decomposition.
"""
const CURRecipe = GeneralizedNystromRecipe{
    <:GeneralizedNystromCoreRecipe,<:SelectorRecipe,<:SelectorRecipe
}


###############################################
# Include files for core implementations
###############################################
include("./core/cross_approximation.jl")

# Complete a column/row operator, whether it is a selector or a compressor.
complete_operator(op::Selector, A::AbstractMatrix) = complete_selector(op, A)
complete_operator(op::Compressor, A::AbstractMatrix) = complete_compressor(op, A)

function complete_approximator(ingredients::GeneralizedNystrom, A::AbstractMatrix)
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
    col_selector = complete_operator(ingredients.col_selector, A)
    row_selector = complete_operator(ingredients.row_selector, A)
    C = typeof(A)(undef, size(A, 1), n_col_vecs)
    R = typeof(A)(undef, n_row_vecs, size(A, 2))
    U = complete_core(ingredients.core, ingredients, A)
    return GeneralizedNystromRecipe(
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
# different organizations of the generalized Nyström selection

###############################################
# Operator application helpers
###############################################
# Form the column factor `C` by applying the column operator to `A`. A selector selects and
# gathers columns; a compressor sketches the column space via a right multiplication.
function form_columns!(approx::GeneralizedNystromRecipe, A::AbstractMatrix)
    _form_columns!(approx.col_selector, approx, A)
    return nothing
end

function _form_columns!(col::SelectorRecipe, approx::GeneralizedNystromRecipe, A::AbstractMatrix)
    select_indices!(approx.col_idx, col, A, approx.n_col_vecs, 1)
    copyto!(approx.C, A[:, approx.col_idx])
    return nothing
end

function _form_columns!(col::CompressorRecipe, approx::GeneralizedNystromRecipe, A::AbstractMatrix)
    # C = A * X
    mul!(approx.C, A, col)
    return nothing
end

# Form the row factor `R` by applying the row operator to `A`, independently of the columns.
# A selector selects and gathers rows (pivoting on `A'`); a compressor sketches the row space
# via a left multiplication.
function form_rows!(approx::GeneralizedNystromRecipe, A::AbstractMatrix)
    _form_rows!(approx.row_selector, approx, A)
    return nothing
end

function _form_rows!(row::SelectorRecipe, approx::GeneralizedNystromRecipe, A::AbstractMatrix)
    select_indices!(approx.row_idx, row, A', approx.n_row_vecs, 1)
    copyto!(approx.R, A[approx.row_idx, :])
    return nothing
end

function _form_rows!(row::CompressorRecipe, approx::GeneralizedNystromRecipe, A::AbstractMatrix)
    # R = Y' * A
    mul!(approx.R, row, A)
    return nothing
end

# Implement the muls
function mul!(
    C::AbstractArray,
    approx::GeneralizedNystromRecipe,
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
    approx::GeneralizedNystromRecipe,
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

# functions for the core must be placed here because they require the definition
# of GeneralizedNystrom to work

"""
    complete_core()

A function that generates a `GeneralizedNystromCoreRecipe` given the arguments.

# Arguments
- `ingredients::GeneralizedNystromCore`, a data structure containing the user-defined
    parameters associated with a particular type of core matrix.
- `gn::GeneralizedNystrom`, a data structure containing the user-defined parameters for the
    generalized Nyström approximation.
- `A::AbstractMatrix`, a target matrix for approximation.
"""
function complete_core(core::GeneralizedNystromCore, gn::GeneralizedNystrom, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No method `complete_core` exists for `GeneralizedNystromCore` of type \
            $(typeof(core)), `GeneralizedNystrom`, and matrix of type $(typeof(A))."
        )
    )
end

# this must be placed here because it requires the GeneralizedNystromCoreRecipe to be defined
"""
    update_core!()

A function that updates the `GeneralizedNystromCoreRecipe` based on the parameters defined in
    the `GeneralizedNystrom` structure.

# Arguments
- `core::GeneralizedNystromCoreRecipe`, a data structure containing the preallocated data
    structures necessary for computing the core matrix.
- `gn::GeneralizedNystromRecipe`, a data structure containing the user-defined parameters and
    preallocated structures.
- `A::AbstractMatrix`, a target matrix to be approximated.
"""
function update_core!(
    core::GeneralizedNystromCoreRecipe, gn::GeneralizedNystromRecipe, A::AbstractMatrix
)
    return throw(
        ArgumentError(
            "No method `update_core!` exists for `GeneralizedNystromCore` of type \
            $(typeof(core)), `GeneralizedNystromRecipe`, and matrix of type $(typeof(A))."
        )
    )
end
