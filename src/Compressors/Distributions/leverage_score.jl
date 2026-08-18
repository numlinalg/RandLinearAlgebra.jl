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
Leverage scores are then the row norms of ``AR^{-1}\\Pi_2``, where
``\\Pi_2 \\in \\mathbb{R}^{d \\times r_2}`` is a second sketch reducing the cost of
forming ``AR^{-1}`` from ``O(nd^2)`` to ``O(nd \\, r_2)``; this only helps when
``r_2 < d``. If `r2` is explicitly given, that value is used directly. Otherwise
`r2` defaults to the size of [drineas2012fast](@citet)'s Lemma 1 ``\\epsilon``-JLT
bound, ``r_2 = \\lceil(12\\ln n + 6\\ln 10)/\\epsilon^2\\rceil`` (their ``\\delta =
0.1``), using `epsilon` and ``n =`` `size(A, 1)`; when that default would not
satisfy ``r_2 < d``, the second sketch is skipped and leverage scores are the row
norms of ``AR^{-1}`` directly. Only `Left()` cardinality is supported in
approximate mode.

!!! note "Approximate Mode Accuracy"
    Inverting the sketch-based `R` is a biased estimator of `A'A`'s inverse (`R'R`
    is Wishart-distributed, and matrix inversion is convex, so the naive estimate
    is systematically too large by Jensen's inequality). This has an exact,
    closed-form correction, applied internally. It does not fix per-row variance,
    though: rows with small true leverage score are the hardest to pin down to
    tight relative error, since the estimator's noise floor dominates a small
    true value. Approximate mode is best suited to producing a sampling
    distribution (where aggregate weighting matters more than any single row's
    exact value); use exact mode, or a much larger `compressor`, when precision
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
- `r2::Union{Nothing, Int}`, only used when `compressor` is provided. If given
    explicitly, must satisfy `1 <= r2 < size(A, 2)`. If `nothing` (the default),
    `r2` is instead computed automatically from `epsilon` and `size(A, 1)`, as
    described above.
- `epsilon::Float64`, the target relative-error tolerance used to size the
    automatic `r2` default when `r2` is not given explicitly; ignored if `r2` is
    given explicitly, or if `compressor` is `nothing`. Must satisfy
    `0 < epsilon < 1`.

# Constructor

    LeverageScore(;
        cardinality=Undef(), replace=false, compressor=nothing, r2=nothing,
        epsilon=0.5,
    )

## Returns
- A `LeverageScore` object.
"""
mutable struct LeverageScore <: Distribution
    cardinality::Cardinality
    replace::Bool
    compressor::Union{Nothing, Compressor}
    r2::Union{Nothing, Int}
    epsilon::Float64
end

function LeverageScore(;
    cardinality = Undef(),
    replace = false,
    compressor = nothing,
    r2 = nothing,
    epsilon = 0.5,
)
    return LeverageScore(cardinality, replace, compressor, r2, epsilon)
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
- `r2::Union{Nothing, Int}`, the resolved second-sketch size: either the value the
    user supplied, the automatically computed default (see `LeverageScore`'s
    `epsilon` field), or `nothing` if no second sketch is used (leverage scores are
    then the row norms of ``AR^{-1}`` directly). Always `nothing` in exact mode.
"""
mutable struct LeverageScoreRecipe <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    state_space::Vector{Int64}
    weights::ProbabilityWeights
    compressor_recipe::Union{Nothing, CompressorRecipe}
    r2::Union{Nothing, Int}
end

"""
    complete_distribution(distribution::LeverageScore, A::AbstractMatrix)

Creates a `LeverageScoreRecipe` for the given distribution and matrix. Computes
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
- `ArgumentError` if the compressor's compression dimension is less than `size(A, 2) + 2`.
- `ArgumentError` if `distribution.r2` is given without a `compressor`, or does not
    satisfy `1 <= r2 < size(A, 2)`.
- `ArgumentError` if `distribution.epsilon` does not satisfy `0 < epsilon < 1`.
"""
function complete_distribution(distribution::LeverageScore, A::AbstractMatrix)
    cardinality = distribution.cardinality
    compressor = distribution.compressor
    r2 = distribution.r2
    epsilon = distribution.epsilon

    if !(0 < epsilon < 1)
        throw(ArgumentError("`LeverageScore`'s `epsilon` must satisfy `0 < epsilon < 1`."))
    end

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

    if r2 !== nothing && compressor === nothing
        throw(
            ArgumentError(
                "`LeverageScore`'s `r2` may only be set when `compressor` is also \
                provided."
            ),
        )
    end

    compressor_recipe = nothing

    if compressor === nothing
        if cardinality == Left()
            n_rows = size(A, 1)
            state_space = collect(1:n_rows)
            F = qr(A)
            # multiply by identity to extract thin Q (n×d) without materializing full Q
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
        n_cols = size(A, 2)
        r1 = compressor_recipe.n_rows  # the compressor's sketch size, not A's n_rows
        if r1 < n_cols + 2
            throw(
                ArgumentError(
                    "The compressor's compression dimension must be at least \
                    `size(A, 2) + 2` for approximate leverage score computation."
                ),
            )
        end
        n_rows = size(A, 1)
        if r2 !== nothing
            if !(1 <= r2 < n_cols)
                throw(ArgumentError("`r2` must satisfy `1 <= r2 < size(A, 2)`."))
            end
        else
            # Auto-size the second sketch via the ε-JLT bound of drineas2012fast's
            # Lemma 1 (δ=0.1): r2 = (12 ln n + 6 ln 10) / ε². This is what makes the
            # algorithm's headline O(nd log n) complexity the default; falls back to
            # the full AR⁻¹ path (no speedup, but still correct) when this bound
            # isn't actually smaller than d.
            auto_r2 = ceil(Int, (12 * log(n_rows) + 6 * log(10)) / epsilon^2)
            r2 = auto_r2 < n_cols ? auto_r2 : nothing
        end
        state_space = collect(1:n_rows)
        B = similar(A, r1, n_cols)
        mul!(B, compressor_recipe, A, 1, 0)
        R = UpperTriangular(qr(B).R)

        # R'R is Wishart(A'A/r1, r1)-distributed, so E[(R'R)⁻¹] = r1(A'A)⁻¹/(r1-d-1),
        # not (A'A)⁻¹ (textbook Wishart identity: inverting a noisy sketch of A'A
        # is biased). Exact fix: scale the raw weights by (r1-d-1)/r1.
        bias_correction = (r1 - n_cols - 1) / r1

        # Π₂ ∈ R^{d×r2}, if requested: a second sketch reducing AR⁻¹'s d columns
        # before taking row norms. Solving M = R⁻¹Π₂ then Ω = A*M avoids ever
        # forming the n×d matrix A*R⁻¹, giving O(nd*r2) instead of O(nd²).
        if r2 !== nothing
            Π2 = randn(n_cols, r2) ./ sqrt(r2)
            M = R \ Π2
            Ω = A * M
            weights =
                ProbabilityWeights(bias_correction .* vec(sum(abs2, Ω, dims = 2)))
        else
            X = A / R
            weights =
                ProbabilityWeights(bias_correction .* vec(sum(abs2, X, dims = 2)))
        end
    end

    return LeverageScoreRecipe(
        cardinality, distribution.replace, state_space, weights, compressor_recipe, r2
    )
end

"""
    update_distribution!(ingredients::LeverageScoreRecipe, A::AbstractMatrix)

Updates the leverage score distribution recipe with the current matrix.
Recomputes leverage scores via QR factorization (exact mode) or via the stored
compressor recipe (approximate mode).

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
        n_cols = size(A, 2)
        r1 = ingredients.compressor_recipe.n_rows  # sketch size, not A's n_rows
        r2 = ingredients.r2
        B = similar(A, r1, n_cols)
        mul!(B, ingredients.compressor_recipe, A, 1, 0)
        R = UpperTriangular(qr(B).R)
        # Exact Wishart bias correction; see the note in `complete_distribution`.
        bias_correction = (r1 - n_cols - 1) / r1
        if r2 !== nothing
            Π2 = randn(n_cols, r2) ./ sqrt(r2)
            M = R \ Π2
            Ω = A * M
            ingredients.weights =
                ProbabilityWeights(bias_correction .* vec(sum(abs2, Ω, dims = 2)))
        else
            X = A / R
            ingredients.weights =
                ProbabilityWeights(bias_correction .* vec(sum(abs2, X, dims = 2)))
        end
    end

    return nothing
end

"""
    sample_distribution!(indices::AbstractVector, distribution::LeverageScoreRecipe)

Samples indices according to the leverage score distribution.

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
