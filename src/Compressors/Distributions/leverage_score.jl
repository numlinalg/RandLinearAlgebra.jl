"""
    LeverageScore <: Distribution

Distribution where the probability of selecting a row (or column) is proportional
to its statistical leverage score, as defined by [drineas2012fast](@citet).

# Mathematical Description

The statistical leverage score of row ``i`` is ``\\ell_i = \\|Q_{i,:}\\|_2^2``, where
``Q`` is the thin ``n \\times d`` factor from the QR factorization of ``A``. The
sampling probability for row ``i`` is ``p_i = \\ell_i / \\sum_j \\ell_j``.

If compressing from the right, the leverage score of column ``j`` is
``\\ell_j = \\|Q_{j,:}\\|_2^2``, where ``Q`` is the ``d \\times d`` factor from the QR
factorization of ``A^\\top``.

If an approximate compressor is provided, leverage scores are estimated using the
randomized algorithm of [drineas2012fast](@citet): a sketch ``B = SA`` is formed,
the QR factorization of ``B`` yields ``R``, and the row norms of ``AR^{-1}``
approximate the exact leverage scores. Only `Left()` cardinality is supported in
approximate mode.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()`, `Right()`,
    or `Undef()`.
- `replace::Bool`, if `true`, sampling occurs with replacement; if `false`, sampling
    occurs without replacement.
- `compressor::Union{Nothing, Compressor}`, if `nothing`, exact leverage scores are
    computed via a thin QR factorization of `A`. If a `Compressor` with `Left()`
    cardinality is provided, approximate leverage scores are computed following
    [drineas2012fast](@citet).

# Constructor

    LeverageScore(; cardinality=Undef(), replace=false, compressor=nothing)

## Returns
- A `LeverageScore` object.
"""
mutable struct LeverageScore <: Distribution
    cardinality::Cardinality
    replace::Bool
    compressor::Union{Nothing, Compressor}
end

function LeverageScore(; cardinality = Undef(), replace = false, compressor = nothing)
    return LeverageScore(cardinality, replace, compressor)
end

"""
    LeverageScoreRecipe <: DistributionRecipe

The recipe containing all allocations and information for the leverage score distribution.

# Fields
- `cardinality::Cardinality`, the cardinality of the compressor. The value is either
    `Left()`, `Right()`, or `Undef()`.
- `replace::Bool`, an option to replace or not during the sampling process based on
    the given weights.
- `state_space::Vector{Int64}`, the row/column index set.
- `weights::ProbabilityWeights`, the leverage score of each element in the state space.
- `compressor_recipe::Union{Nothing, CompressorRecipe}`, the completed compressor for
    approximate leverage score computation, or `nothing` in exact mode.
"""
mutable struct LeverageScoreRecipe <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    state_space::Vector{Int64}
    weights::ProbabilityWeights
    compressor_recipe::Union{Nothing, CompressorRecipe}
end

"""
    complete_distribution(distribution::LeverageScore, A::AbstractMatrix)

A function that generates a `LeverageScoreRecipe` given the arguments. Computes
leverage scores of `A` either exactly via a thin QR factorization, or approximately
using the randomized algorithm of [drineas2012fast](@citet).

# Arguments
- `distribution::LeverageScore`, a user-specified leverage score distribution.
- `A::AbstractMatrix`, a coefficient matrix.

# Returns
- A `LeverageScoreRecipe` object.

# Throws
- `ArgumentError` if `distribution.cardinality` is `Undef()`.
- `ArgumentError` if approximate mode is requested with `Right()` cardinality.
- `ArgumentError` if the provided compressor does not have `Left()` cardinality.
- `ArgumentError` if the compressor's compression dimension is less than `size(A, 2)`.
"""
function complete_distribution(distribution::LeverageScore, A::AbstractMatrix)
    cardinality = distribution.cardinality
    compressor = distribution.compressor

    if cardinality == Undef()
        throw(
            ArgumentError(
                "`LeverageScore` cardinality must be `Left()` or `Right()`. \
                `Undef()` is not allowed in `complete_distribution`."
            ),
        )
    end

    if compressor !== nothing && cardinality == Right()
        throw(
            ArgumentError(
                "Approximate leverage scores with `Right()` cardinality are not \
                currently supported in `complete_distribution`."
            ),
        )
    end

    if compressor !== nothing && compressor.cardinality != Left()
        throw(
            ArgumentError(
                "The compressor provided to `LeverageScore` must have `Left()` \
                cardinality in `complete_distribution`."
            ),
        )
    end

    compressor_recipe = nothing

    if compressor === nothing
        if cardinality == Left()
            n_rows = size(A, 1)
            state_space = collect(1:n_rows)
            F = qr(A)
            # multiply by identity to extract thin Q (n×d) without materializing full n×n Q
            Q = F.Q * Matrix(I, n_rows, size(A, 2))
            weights = ProbabilityWeights(vec(sum(abs2, Q, dims = 2)))
        else
            n_cols = size(A, 2)
            state_space = collect(1:n_cols)
            # A' is d×n; its Q factor is already d×d (thin = full for fat matrices)
            Q = Matrix(qr(Matrix(A')).Q)
            weights = ProbabilityWeights(vec(sum(abs2, Q, dims = 2)))
        end
    else
        compressor_recipe = complete_compressor(compressor, A)
        if compressor_recipe.n_rows < size(A, 2)
            throw(
                ArgumentError(
                    "The compressor's compression dimension must be at least `size(A, 2)` \
                    for approximate leverage score computation."
                ),
            )
        end
        n_rows = size(A, 1)
        state_space = collect(1:n_rows)
        B = similar(A, compressor_recipe.n_rows, size(A, 2))
        mul!(B, compressor_recipe, A, 1, 0)
        R = UpperTriangular(qr(B).R)
        X = A / R  # X = A * R⁻¹; row norms² approximate exact leverage scores
        weights = ProbabilityWeights(vec(sum(abs2, X, dims = 2)))
    end

    return LeverageScoreRecipe(
        cardinality, distribution.replace, state_space, weights, compressor_recipe
    )
end

"""
    update_distribution!(ingredients::LeverageScoreRecipe, A::AbstractMatrix)

A function that updates the `LeverageScoreRecipe` in place to correspond with a
new matrix `A`. Recomputes leverage scores via QR factorization (exact mode) or
via the stored compressor recipe (approximate mode).

# Arguments
- `ingredients::LeverageScoreRecipe`, a fully initialized leverage score distribution.
- `A::AbstractMatrix`, the new coefficient matrix.

# Returns
- Modifies `ingredients` in place and returns `nothing`.

# Throws
- `ArgumentError` if `ingredients.cardinality` is `Undef()`.
- `ArgumentError` if matrix dimensions changed in approximate mode. Call
    `complete_distribution` again to reinitialize for a matrix of different size.
"""
function update_distribution!(ingredients::LeverageScoreRecipe, A::AbstractMatrix)
    if ingredients.cardinality == Undef()
        throw(
            ArgumentError(
                "`LeverageScore` cardinality must be `Left()` or `Right()`. \
                `Undef()` is not allowed in `update_distribution!`."
            ),
        )
    end

    if ingredients.compressor_recipe === nothing
        if ingredients.cardinality == Left()
            n_rows = size(A, 1)
            length(ingredients.state_space) != n_rows &&
                (ingredients.state_space = collect(1:n_rows))
            F = qr(A)
            Q = F.Q * Matrix(I, n_rows, size(A, 2))  # thin Q (n×d)
            ingredients.weights = ProbabilityWeights(vec(sum(abs2, Q, dims = 2)))
        else
            n_cols = size(A, 2)
            length(ingredients.state_space) != n_cols &&
                (ingredients.state_space = collect(1:n_cols))
            Q = Matrix(qr(Matrix(A')).Q)  # A' is d×n; Q is already d×d
            ingredients.weights = ProbabilityWeights(vec(sum(abs2, Q, dims = 2)))
        end
    else
        if length(ingredients.state_space) != size(A, 1)
            throw(
                ArgumentError(
                    "Matrix row dimension changed in approximate mode. Call \
                    `complete_distribution` again to reinitialize."
                ),
            )
        end
        update_compressor!(ingredients.compressor_recipe)
        B = similar(A, ingredients.compressor_recipe.n_rows, size(A, 2))
        mul!(B, ingredients.compressor_recipe, A, 1, 0)
        R = UpperTriangular(qr(B).R)
        X = A / R  # X = A * R⁻¹
        ingredients.weights = ProbabilityWeights(vec(sum(abs2, X, dims = 2)))
    end

    return nothing
end

"""
    sample_distribution!(x::AbstractVector, distribution::LeverageScoreRecipe)

A function that in place updates `x` with sampled indices following the leverage
score weights of `distribution`.

# Arguments
- `x::AbstractVector`, an abstract vector to store the sampled indices.
- `distribution::LeverageScoreRecipe`, a fully initialized leverage score distribution.

# Returns
- Modifies `x` in place and returns `nothing`.
"""
function sample_distribution!(x::AbstractVector, distribution::LeverageScoreRecipe)
    wsample!(
        distribution.state_space,
        distribution.weights,
        x,
        ordered = true,
        replace = distribution.replace,
    )
    return nothing
end
