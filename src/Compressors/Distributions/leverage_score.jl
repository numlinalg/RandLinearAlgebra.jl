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
randomized algorithm of [drineas2012fast](@citet): a sketch ``B = \\Pi_1 A`` is
formed via the supplied compressor, and the QR factorization of ``B`` yields ``R``.
A second, lightweight Gaussian sketch ``\\Pi_2 \\in \\mathbb{R}^{d \\times r_2}``
(sized automatically from ``n``) is then used to compute ``M = R^{-1}\\Pi_2`` and
``\\Omega = AM``, whose row norms approximate the exact leverage scores. Solving
for ``M`` before multiplying by ``A`` avoids ever materializing the ``n \\times d``
matrix ``AR^{-1}``, giving an ``O(nd \\log n)`` estimator rather than the
``O(nd^2)`` cost of the direct approach. Only `Left()` cardinality is supported in
approximate mode.

!!! note "Approximate Mode Accuracy"
    The relative-error guarantee in [drineas2012fast](@citet) (Theorem 2) holds with
    the sketch sizes the paper specifies, which carry large constants; sketches sized
    for the ``O(nd \\log n)`` asymptotic target (as used here) trade tight per-row
    accuracy for speed. In particular, rows whose true leverage score is small relative
    to the others are the hardest to estimate to tight relative error, since the
    estimator's noise floor dominates a small true value. Approximate mode is best
    suited to producing a sampling distribution (where the aggregate weighting matters
    more than any single row's exact value), not to recovering individual leverage
    scores precisely; use exact mode, or a much larger `compressor`, when precision
    matters.

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
        d = size(A, 2)
        state_space = collect(1:n_rows)
        B = similar(A, compressor_recipe.n_rows, d)
        mul!(B, compressor_recipe, A, 1, 0)
        R = UpperTriangular(qr(B).R)
        # Simpler O(nd²) estimator: materialize the full A*R⁻¹ and take its row
        # norms directly. Superseded below by the O(nd log n) two-sketch estimator
        # (Algorithm 1, steps 3-4 of drineas2012fast); kept here for reference.
        # X = A / R
        # weights = ProbabilityWeights(vec(sum(abs2, X, dims = 2)))

        # Second (Johnson-Lindenstrauss) sketch Π₂ ∈ R^{d × r2}. Solving
        # M = R⁻¹Π₂ first (a cheap d×d triangular solve) and only then forming
        # Ω = A*M avoids ever materializing the n×d matrix A*R⁻¹, which is what
        # gets the algorithm down to O(nd log n) instead of O(nd²).
        r2 = max(2, ceil(Int, 20 * log(n_rows)))
        Π2 = randn(d, r2) ./ sqrt(r2)
        M = R \ Π2
        Ω = A * M
        weights = ProbabilityWeights(vec(sum(abs2, Ω, dims = 2)))
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
        d = size(A, 2)
        B = similar(A, ingredients.compressor_recipe.n_rows, d)
        mul!(B, ingredients.compressor_recipe, A, 1, 0)
        R = UpperTriangular(qr(B).R)
        # X = A / R
        # ingredients.weights = ProbabilityWeights(vec(sum(abs2, X, dims = 2)))
        r2 = max(2, ceil(Int, 20 * log(size(A, 1))))
        Π2 = randn(d, r2) ./ sqrt(r2)
        M = R \ Π2
        Ω = A * M
        ingredients.weights = ProbabilityWeights(vec(sum(abs2, Ω, dims = 2)))
    end

    return nothing
end

"""
    sample_distribution!(indices::AbstractVector, distribution::LeverageScoreRecipe)

A function that in place updates `indices` with sampled indices following the leverage
score weights of `distribution`.

# Arguments
- `indices::AbstractVector`, an abstract vector to store the sampled indices.
- `distribution::LeverageScoreRecipe`, a fully initialized leverage score distribution.

# Returns
- Modifies `indices` in place and returns `nothing`.
"""
function sample_distribution!(indices::AbstractVector, distribution::LeverageScoreRecipe)
    wsample!(
        distribution.state_space,
        distribution.weights,
        indices,
        ordered = true,
        replace = distribution.replace,
    )
    return nothing
end
