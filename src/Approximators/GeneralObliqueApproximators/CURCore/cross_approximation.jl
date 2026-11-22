"""
    CrossApproximation <: CURCore

A form of the core matrix in CUR decomposition, such that for column indices, J, and row
indices, I, the matrix ``U`` is ``A[I,J]^\\dagger``.

# Fields
- None
"""
struct CrossApproximation <: CURCore end

"""
    CrossApproximationRecipe <: CURCore

A structure featuring the preallocated arrays necessary for storing the cross-approximation
core matrix to a CUR decomposition.

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
mutable struct CrossApproximationRecipe <: CURCoreRecipe
    n_rows::Int64
    n_cols::Int64
    Q::AbstractMatrix
    R::AbstractMatrix
    buffer::AbstractMatrix
end

function complete_core(core::CrossApproximation, cur::CUR,  A::AbstractMatrix)
    # preallocate the Q and R factors of the intersection submatrix `A[I, J]`, which is
    # `(rank + oversample) x rank`. We use identity initializers because this form
    # preallocates all dense entries and works for sparse arrays.
    Q = typeof(A)(I, cur.rank + cur.oversample, cur.rank)
    R = typeof(A)(I, cur.rank, cur.rank)
    # preallocate a dense buffer with `rank` rows for applying the `Q` factor. Its column
    # count is the block size; inputs wider than this are processed in blocks via views.
    block_size = cur.block_size == 0 ? size(A, 2) : cur.block_size
    buffer = Matrix{eltype(A)}(undef, cur.rank, block_size)
    return CrossApproximationRecipe(
        cur.rank,
        cur.rank + cur.oversample,
        Q,
        R,
        buffer
    )
end

function update_core!(core::CrossApproximationRecipe, cur::CURRecipe, A::AbstractMatrix)
    # Factor the intersection submatrix `A[I, J] = Q R`. The core matrix
    # `U = A[I, J]^dagger = R^{-1} Q'` is never formed explicitly; instead we store the
    # thin `Q` and the upper triangular `R` and apply them during multiplication.
    F = qr!(A[cur.row_idx, cur.col_idx])
    core.Q = Array(F.Q)
    core.R = Matrix(F.R)
    return nothing
end


# Implement the rapproximate function for the Cross Approximation 
# This function will 
# 1. apply a selection routine to the columns of A
# 2. on the selected columns only, apply the row selection routine to select rows. (This is 
# applied to only selected columns and not the whole matrix because of the example
# 3 0 0
# 0 2 0
# 1 0 0
# where the first column would be selected. If the rows are selected disregarding the 
# selection of the first column the second row would be selected, which results in an 
# intersection matrix of 0 causing overflows when computing the pseudo inverse.)
# 3. form the core matrix will be formed using the entries at the intersection 
# of the index sets.
function rapproximate!(approx::CURRecipe{CrossApproximationRecipe}, A::AbstractMatrix)
    # select column indices
    select_indices!(
        approx.col_idx,
        approx.col_selector,
        A,
        approx.n_col_vecs,
        1
    )
    
    # gather the columns to select rows dependently 
    copyto!(approx.C, A[:, approx.col_idx])

    # select row indices depent on the columns selected
    select_indices!(
        approx.row_idx,
        approx.row_selector,
        approx.C',
        approx.n_row_vecs,
        1
    )

    # gather the rows entries 
    copyto!(approx.R, A[approx.row_idx, :])
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
        @. Cv = beta * Cv + alpha * buf
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
