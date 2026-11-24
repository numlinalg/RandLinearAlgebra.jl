"""
    CrossApproximation <: GeneralizedNystromCore

A form of the core matrix in a generalized Nyström decomposition, such that for column indices, J, and row
indices, I, the matrix ``U`` is ``A[I,J]^\\dagger``.

# Fields
- None
"""
struct CrossApproximation <: GeneralizedNystromCore end

"""
    CrossApproximationRecipe <: GeneralizedNystromCoreRecipe

A structure featuring the preallocated arrays necessary for storing the cross-approximation
core matrix to a generalized Nyström decomposition.

The core matrix is ``U = A[I, J]^\\dagger``, the pseudoinverse of the intersection
submatrix. Rather than forming ``U`` explicitly, we store the factors of the thin QR
decomposition ``A[I, J] = QR`` and apply them (via a matrix multiply by `Q` and a
triangular solve with `R`) during multiplication, since ``U = R^{-1} Q'``.

# Fields
- `n_rows::Int64`, the number of rows in the core matrix.
- `n_cols::Int64`, the number of columns in the core matrix.
- `Q::AbstractMatrix`, the orthogonal factor of the QR decomposition of the intersection
    submatrix `A[I, J]`.
- `R::AbstractMatrix`, the upper triangular factor of the QR decomposition of the
    intersection submatrix `A[I, J]`.
- `buffer::AbstractMatrix`, a buffer matrix used to store the intermediate result of
    applying the `Q` factor during multiplication. It has `n_rows` rows and its number of
    columns sets the block size used to process wider inputs with views and a loop.
"""
mutable struct CrossApproximationRecipe <: GeneralizedNystromCoreRecipe
    n_rows::Int64
    n_cols::Int64
    Q::AbstractMatrix
    R::AbstractMatrix
    buffer::AbstractMatrix
end

function complete_core(core::CrossApproximation, gn::GeneralizedNystrom,  A::AbstractMatrix)
    # preallocate the Q and R factors of the intersection submatrix `A[I, J]`, which is
    # `(rank + oversample) x rank`. We use identity initializers because this form
    # preallocates all dense entries and works for sparse arrays.
    Q = typeof(A)(I, gn.rank + gn.oversample, gn.rank)
    R = typeof(A)(I, gn.rank, gn.rank)
    # preallocate a dense buffer with `rank` rows for applying the `Q` factor. Its column
    # count is the block size; inputs wider than this are processed in blocks via views.
    block_size = gn.block_size == 0 ? size(A, 2) : gn.block_size
    buffer = Matrix{eltype(A)}(undef, gn.rank, block_size)
    return CrossApproximationRecipe(
        gn.rank,
        gn.rank + gn.oversample,
        Q,
        R,
        buffer
    )
end

# The core matrix is `U = M^dagger = R^{-1} Q'`, where `M` is the row operator applied to the
# columns `C`. For selectors this is the intersection submatrix `A[I, J]` (the CUR case); for
# a compressor it is `Y'C` (the generalized Nyström case). `core_input` returns `M`.
core_input(row::SelectorRecipe, gn::GeneralizedNystromRecipe) = gn.Column[gn.row_idx, :]
core_input(row::CompressorRecipe, gn::GeneralizedNystromRecipe) = row * gn.Column

function update_core!(core::CrossApproximationRecipe, gn::GeneralizedNystromRecipe, A::AbstractMatrix)
    # Factor `M = Q R`. The core `U = M^dagger = R^{-1} Q'` is never formed explicitly;
    # instead we store the thin `Q` and the upper triangular `R` and apply them during
    # multiplication.
    F = qr!(core_input(gn.row_operator, gn))
    core.Q = Array(F.Q)
    core.R = Matrix(F.R)
    return nothing
end


# Implement the rapproximate function for the Cross Approximation.
#
# When both operators are selectors (the CUR case) the rows are selected *dependently*: the
# row selection routine is applied to the already-selected columns `C'` rather than to the
# whole matrix. This is necessary because of the example
# 3 0 0
# 0 2 0
# 1 0 0
# where the first column would be selected. If the rows are selected disregarding the
# selection of the first column the second row would be selected, which results in an
# intersection matrix of 0 causing overflows when computing the pseudo inverse. The core is
# then formed from the entries at the intersection of the index sets.
function rapproximate!(
    approx::GeneralizedNystromRecipe{
        CrossApproximationRecipe,<:SelectorRecipe,<:SelectorRecipe
    },
    A::AbstractMatrix
)
    # select column indices
    select_indices!(
        approx.col_idx,
        approx.column_operator,
        A,
        approx.n_col_vecs,
        1
    )

    # gather the columns to select rows dependently
    copyto!(approx.Column, A[:, approx.col_idx])

    # select row indices dependent on the columns selected (from `C'`, not `A'`)
    select_indices!(
        approx.row_idx,
        approx.row_operator,
        approx.Column',
        approx.n_row_vecs,
        1
    )

    # gather the rows entries
    copyto!(approx.Row, A[approx.row_idx, :])
    # Compute the core matrix
    update_core!(approx.U, approx, A)
    return nothing
end

# For all other cases (at least one operator is a compressor) the columns and rows are formed
# *independently*: each operator is applied to `A` on its own, with no dependence between the
# column and row choices.
function rapproximate!(
    approx::GeneralizedNystromRecipe{CrossApproximationRecipe}, A::AbstractMatrix
)
    # form the columns `C` and rows `R` by applying each operator to `A` independently
    form_columns!(approx, A)
    form_rows!(approx, A)
    # Compute the core matrix
    update_core!(approx.U, approx, A)
    return nothing
end

# C <- beta * C + alpha * U * A, where U = A[I, J]^dagger = R^{-1} Q'
function mul!(
    C::AbstractArray,
    core::CrossApproximationRecipe,
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    # dispatch through a function barrier so the abstractly typed fields are resolved to
    # their concrete types once, keeping the blocked loop allocation-free.
    _cross_left_mul!(C, core.Q, core.R, core.buffer, A, alpha, beta)
    return nothing
end

function _cross_left_mul!(C, Q, R, buffer, A, alpha, beta)
    Qt = Q'
    Rt = UpperTriangular(R)
    block_size = size(buffer, 2)
    n = size(A, 2)
    # process the columns of `A` in blocks no wider than the buffer
    start = 1
    while start <= n
        width = min(block_size, n - start + 1)
        cols = start:(start + width - 1)
        # buf = Q' * A[:, cols], then apply R^{-1} in place with a triangular solve
        buf = view(buffer, :, 1:width)
        mul!(buf, Qt, view(A, :, cols))
        ldiv!(Rt, buf)
        Cv = view(C, :, cols)
        # when beta is zero C is overwritten (and not read); this matches the BLAS
        # convention and avoids propagating uninitialized NaN/Inf values from C.
        if iszero(beta)
            Cv .= alpha .* buf
        else
            Cv .= beta .* Cv .+ alpha .* buf
        end

        start += width
    end

    return nothing
end

# C <- beta * C + alpha * A * U, where U = A[I, J]^dagger = R^{-1} Q'
function mul!(
    C::AbstractArray,
    A::AbstractArray,
    core::CrossApproximationRecipe,
    alpha::Number,
    beta::Number
)
    # dispatch through a function barrier so the abstractly typed fields are resolved to
    # their concrete types once, keeping the blocked loop allocation-free.
    _cross_right_mul!(C, core.Q, core.R, core.buffer, A, alpha, beta)
    return nothing
end

function _cross_right_mul!(C, Q, R, buffer, A, alpha, beta)
    Qt = Q'
    Rt = UpperTriangular(R)
    block_size = size(buffer, 2)
    n = size(A, 1)
    # process the rows of `A` in blocks no taller than the buffer. The buffer is used
    # transposed so each block has `width` rows and `rank` columns.
    start = 1
    while start <= n
        width = min(block_size, n - start + 1)
        rows = start:(start + width - 1)
        # buf = A[rows, :] * R^{-1} with a triangular solve, then multiply by Q'
        buf = transpose(view(buffer, :, 1:width))
        copyto!(buf, view(A, rows, :))
        rdiv!(buf, Rt)
        Cv = view(C, rows, :)
        mul!(Cv, buf, Qt, alpha, beta)
        start += width
    end

    return nothing
end
