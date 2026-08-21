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
randomized algorithm of [drineas2012fast](@citet), following one of three tiers:

1. **Exact**: compute the leverage scores from the QR factorization of `A`
    directly (`compressor === nothing`).
2. **One sketch**: compress `A` with `S1` (`compressor`) to form
    ``B = S_1 A``, an ``r_1 \\times d`` matrix, where ``r_1`` is `S1`'s
    `compression_dim`; the QR factorization of ``B`` yields ``R``, and leverage
    scores are the row norms of ``AR^{-1}``.
3. **Two sketches**: as above, but also compress with a second compressor
    `S2` (`compressor2`), an ``d \\times r_2`` matrix, where ``r_2`` is `S2`'s
    `compression_dim`, and leverage scores are instead the row norms of
    ``AR^{-1}S_2``, reducing the cost of forming ``AR^{-1}`` from
    ``O(nd^2)`` to ``O(nd \\, r_2)``; this only helps when ``r_2 < d``.

Tier 3 is used whenever `compressor2` is given explicitly. If `compressor2` is
not given but `compressor` is, `S2` defaults to a `Gaussian` compressor with
`Right()` cardinality: sized by `r2` if given explicitly, or otherwise
auto-sized from [drineas2012fast](@citet)'s Lemma 1 ``\\epsilon``-JLT bound
(itself [achlioptas2003database](@citet)'s Theorem 1.1), ``r_2 =
\\lceil(12\\ln n + 6\\ln 10)/\\epsilon^2\\rceil`` (their ``\\delta = 0.1``), using
`epsilon` and ``n =`` `size(A, 1)`; when that automatic value would not satisfy
``r_2 < d``, tier 2 is used instead (no `S2`, no speedup, but still correct).
Only `Left()` cardinality is supported in approximate mode.

!!! note "Approximate Mode Accuracy"
    Any sketch-based `R` gives a biased estimate of `A'A`'s inverse: inverting a
    noisy sketch of `A'A` systematically overstates every leverage score (matrix
    inversion is convex, so this follows from Jensen's inequality). If `S1` is a
    `Gaussian` compressor, `R'R` is known to have a Wishart distribution, with an
    exact, closed-form bias correction term for that case (the standard
    inverse-Wishart mean identity, e.g. [muirhead1982aspects](@citet)), which is
    what this implementation applies internally. Other compressor types for `S1`
    may not share that distribution; the same correction term is still applied,
    but it may not fully resolve the bias for them. It also does not fix per-row
    variance regardless of compressor type: rows with small true leverage score
    are the hardest to pin down to tight relative error, since the estimator's
    noise floor dominates a small true value. Approximate mode is best suited to
    producing a sampling distribution (where aggregate weighting matters more
    than any single row's exact value); use exact mode, or a much larger `S1`,
    when precision matters.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()`, `Right()`,
    or `Undef()`.
- `replace::Bool`, if `true`, sampling occurs with replacement; if `false`, sampling
    occurs without replacement.
- `compressor::Union{Nothing, Compressor}`, `S1`. If `nothing`, exact leverage
    scores are computed via a thin QR factorization of `A`. If a `Compressor` with
    `Left()` cardinality is provided, approximate leverage scores are computed
    following [drineas2012fast](@citet), sketching `A` down to `S1`'s
    `compression_dim` (``r_1``) rows.
- `compressor2::Union{Nothing, Compressor}`, `S2`, only used when `compressor` is
    also provided. If a `Compressor` with `Right()` cardinality is given, tier 3
    (two sketches) is used, sized by `S2`'s own `compression_dim` (``r_2``). If
    `nothing` (the default), `S2` defaults to a `Gaussian` compressor sized by
    `r2` or, failing that, computed automatically from `epsilon`, as described
    above.
- `r2::Union{Nothing, Int}`, only used when `compressor` is provided and
    `compressor2` is not. If given explicitly, must satisfy
    `1 <= r2 < size(A, 2)`, and sizes a default `Gaussian` `S2`. If `nothing`
    (the default), `r2` is instead computed automatically from `epsilon` and
    `size(A, 1)`, as described above.
- `epsilon::Float64`, the target relative-error tolerance used to size the
    automatic `r2` default when neither `r2` nor `compressor2` is given
    explicitly; ignored otherwise, or if `compressor` is `nothing`. Must satisfy
    `0 < epsilon <= 0.5`, the validity range of the JL bound in
    [drineas2012fast](@citet)'s Lemma 1.

# Constructor

    LeverageScore(;
        cardinality = Undef(),
        replace = false,
        compressor = nothing,
        compressor2 = nothing,
        r2 = nothing,
        epsilon = 0.5,
    )

## Returns
- A `LeverageScore` object.
"""
mutable struct LeverageScore <: Distribution
    cardinality::Cardinality
    replace::Bool
    compressor::Union{Nothing, Compressor}
    compressor2::Union{Nothing, Compressor}
    r2::Union{Nothing, Int}
    epsilon::Float64
end

function LeverageScore(;
    cardinality = Undef(),
    replace = false,
    compressor = nothing,
    compressor2 = nothing,
    r2 = nothing,
    epsilon = 0.5,
)
    return LeverageScore(cardinality, replace, compressor, compressor2, r2, epsilon)
end

"""
    LeverageScoreRecipe <: DistributionRecipe

The recipe containing all allocations and information for the leverage score
distribution. Parametrized on the types of `S1` and `S2`'s completed compressor
recipes (each either `Nothing` or a `CompressorRecipe`), so that
`complete_distribution` returns a concrete type for each of the three tiers
described in `LeverageScore`'s docstring, and `update_distribution!` dispatches
to a dedicated, branch-free method per tier.

# Fields
- `cardinality::Cardinality`, the cardinality of the compressor. The value is either
    `Left()`, `Right()`, or `Undef()`.
- `replace::Bool`, an option to replace or not during the sampling process based on
    the given weights.
- `state_space::Vector{Int64}`, the row/column index set.
- `weights::ProbabilityWeights`, the leverage score of each element in the state space.
- `compressor_recipe::Union{Nothing, CompressorRecipe}`, the completed `S1`
    compressor for approximate leverage score computation, or `nothing` in exact
    mode.
- `compressor_recipe_2::Union{Nothing, CompressorRecipe}`, the completed `S2`
    compressor (explicit or defaulted, see `LeverageScore`), or `nothing` in
    exact mode or tier 2 (one sketch only).
"""
mutable struct LeverageScoreRecipe{
    C1<:Union{Nothing, CompressorRecipe}, C2<:Union{Nothing, CompressorRecipe}
} <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    state_space::Vector{Int64}
    weights::ProbabilityWeights
    compressor_recipe::C1
    compressor_recipe_2::C2
end

# Materializes any Right()-cardinality CompressorRecipe's own explicit r×s matrix
# by applying it to an r×r identity -- works generically for any compressor type,
# and stays cheap since it operates on the small (d-sized) dimension, not A's n.
function materialize_right_compressor(
    compressor_recipe::CompressorRecipe, n_rows::Int, n_cols::Int, ::Type{T}
) where {T}
    S = zeros(T, n_rows, n_cols)
    mul!(S, Matrix{T}(I, n_rows, n_rows), compressor_recipe, 1, 0)
    return S
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
- `ArgumentError` if a `compressor` is given but `distribution.cardinality` or
    `compressor.cardinality` is not `Left()`.
- `ArgumentError` if `S1`'s compression dimension is less than `size(A, 2) + 2`.
- `ArgumentError` if `distribution.compressor2` is given without `compressor`,
    does not have `Right()` cardinality, or is given alongside `r2`.
- `ArgumentError` if `distribution.r2` is given without a `compressor`, or does not
    satisfy `1 <= r2 < size(A, 2)`.
- `ArgumentError` if `distribution.epsilon` does not satisfy `0 < epsilon <= 0.5`.
"""
function complete_distribution(distribution::LeverageScore, A::AbstractMatrix)
    
    cardinality = distribution.cardinality
    compressor = distribution.compressor
    compressor2 = distribution.compressor2
    r2 = distribution.r2
    epsilon = distribution.epsilon

    if !(0 < epsilon <= 0.5)
        throw(
            ArgumentError("`LeverageScore`'s `epsilon` must satisfy `0 < epsilon <= 0.5`.")
        )
    end

    if cardinality == Undef()
        throw(
            ArgumentError(
                "`LeverageScore` cardinality must be `Left()` or `Right()`. \
                `Undef()` is not allowed in `complete_distribution`."
            ),
        )
    end

    if compressor !== nothing && (cardinality != Left() || compressor.cardinality != Left())
        throw(
            ArgumentError(
                "Approximate leverage scores require both `LeverageScore`'s \
                `cardinality` and `compressor`'s `cardinality` to be `Left()` in \
                `complete_distribution`."
            ),
        )
    end

    if compressor2 !== nothing && compressor === nothing
        throw(
            ArgumentError(
                "`LeverageScore`'s `compressor2` may only be set when `compressor` \
                is also provided."
            ),
        )
    end

    if compressor2 !== nothing && compressor2.cardinality != Right()
        throw(
            ArgumentError(
                "The `compressor2` provided to `LeverageScore` must have `Right()` \
                cardinality in `complete_distribution`."
            ),
        )
    end

    if compressor2 !== nothing && r2 !== nothing
        throw(
            ArgumentError(
                "`LeverageScore`'s `r2` and `compressor2` may not both be set; \
                size `compressor2` directly via its own `compression_dim` instead."
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
        return LeverageScoreRecipe(
            cardinality, distribution.replace, state_space, weights, nothing, nothing
        )
    end

    compressor_recipe = complete_compressor(compressor, A)
    n_cols = size(A, 2)
    n_rows = size(A, 1)
    r1 = compressor_recipe.n_rows  # S1's sketch size, not A's n_rows
    if r1 < n_cols + 2
        throw(
            ArgumentError(
                "The compressor's compression dimension must be at least \
                `size(A, 2) + 2` for approximate leverage score computation."
            ),
        )
    end
    state_space = collect(1:n_rows)

    compressor_recipe_2 = nothing
    if compressor2 !== nothing
        compressor_recipe_2 = complete_compressor(compressor2, A)
        if !(1 <= compressor_recipe_2.n_cols < n_cols)
            throw(
                ArgumentError(
                    "`compressor2`'s `compression_dim` must satisfy \
                    `1 <= compression_dim < size(A, 2)`."
                ),
            )
        end
    elseif r2 !== nothing
        if !(1 <= r2 < n_cols)
            throw(ArgumentError("`r2` must satisfy `1 <= r2 < size(A, 2)`."))
        end
        compressor_recipe_2 = complete_compressor(
            Gaussian(cardinality = Right(), compression_dim = r2), A
        )
    else
        # Auto-size S2 via the ε-JLT bound of drineas2012fast's Lemma 1 (δ=0.1):
        # r2 = (12 ln n + 6 ln 10) / ε². This is what makes the algorithm's
        # headline O(nd log n) complexity the default; falls back to tier 2
        # (no S2, no speedup, but still correct) when this bound isn't actually
        # smaller than d.
        auto_r2 = ceil(Int, (12 * log(n_rows) + 6 * log(10)) / epsilon^2)
        if auto_r2 < n_cols
            compressor_recipe_2 = complete_compressor(
                Gaussian(cardinality = Right(), compression_dim = auto_r2), A
            )
        end
    end

    B = similar(A, r1, n_cols)
    mul!(B, compressor_recipe, A, 1, 0)
    R = UpperTriangular(qr(B).R)

    # R'R is Wishart(A'A/r1, r1)-distributed when S1 is Gaussian, so
    # E[(R'R)⁻¹] = r1(A'A)⁻¹/(r1-d-1), not (A'A)⁻¹ (Wishart identity: inverting a
    # noisy sketch of A'A is biased). Exact fix for Gaussian S1: scale the raw
    # weights by (r1-d-1)/r1. Other S1 types may not share this distribution; the
    # same correction is still applied, but may not fully resolve the bias for
    # them (see the "Approximate Mode Accuracy" note above).
    bias_correction = (r1 - n_cols - 1) / r1

    if compressor_recipe_2 !== nothing
        r2_final = compressor_recipe_2.n_cols
        # S2 ∈ R^{d×r2}: S2's own explicit matrix. Solving M = R⁻¹S2 then
        # Ω = A*M avoids ever forming the n×d matrix A*R⁻¹, giving O(nd*r2)
        # instead of O(nd²).
        S2 = materialize_right_compressor(compressor_recipe_2, n_cols, r2_final, eltype(A))
        M = R \ S2
        Ω = A * M
        weights = ProbabilityWeights(bias_correction .* vec(sum(abs2, Ω, dims = 2)))
    else
        X = A / R
        weights = ProbabilityWeights(bias_correction .* vec(sum(abs2, X, dims = 2)))
    end

    return LeverageScoreRecipe(
        cardinality,
        distribution.replace,
        state_space,
        weights,
        compressor_recipe,
        compressor_recipe_2,
    )
end

function check_leverage_score_update(ingredients::LeverageScoreRecipe, ::AbstractMatrix)
    if ingredients.cardinality == Undef()
        throw(
            ArgumentError(
                "`LeverageScore` cardinality must be `Left()` or `Right()`. \
                `Undef()` is not allowed in `update_distribution!`."
            ),
        )
    end
    return nothing
end

"""
    update_distribution!(ingredients::LeverageScoreRecipe, A::AbstractMatrix)

Updates the leverage score distribution recipe with the current matrix.
Recomputes leverage scores via QR factorization (exact mode) or via the stored
compressor recipe(s) (approximate mode), dispatching on the recipe's tier (exact,
one sketch, or two sketches -- see `LeverageScore`'s docstring).

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
function update_distribution!(
    ingredients::LeverageScoreRecipe{Nothing, Nothing}, A::AbstractMatrix
)
    check_leverage_score_update(ingredients, A)
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
    return nothing
end

function update_distribution!(
    ingredients::LeverageScoreRecipe{<:CompressorRecipe, Nothing}, A::AbstractMatrix
)
    check_leverage_score_update(ingredients, A)
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
    B = similar(A, r1, n_cols)
    mul!(B, ingredients.compressor_recipe, A, 1, 0)
    R = UpperTriangular(qr(B).R)
    # Exact Wishart bias correction for Gaussian S1; see the note in
    # `complete_distribution`.
    bias_correction = (r1 - n_cols - 1) / r1
    X = A / R
    ingredients.weights = ProbabilityWeights(bias_correction .* vec(sum(abs2, X, dims = 2)))
    return nothing
end

function update_distribution!(
    ingredients::LeverageScoreRecipe{<:CompressorRecipe, <:CompressorRecipe},
    A::AbstractMatrix,
)
    check_leverage_score_update(ingredients, A)
    if length(ingredients.state_space) != size(A, 1)
        throw(
            ArgumentError(
                "Matrix row dimension changed in approximate mode. Call \
                `complete_distribution` again to reinitialize."
            ),
        )
    end

    update_compressor!(ingredients.compressor_recipe)
    update_compressor!(ingredients.compressor_recipe_2)
    n_cols = size(A, 2)
    r1 = ingredients.compressor_recipe.n_rows  # sketch size, not A's n_rows
    r2 = ingredients.compressor_recipe_2.n_cols
    B = similar(A, r1, n_cols)
    mul!(B, ingredients.compressor_recipe, A, 1, 0)
    R = UpperTriangular(qr(B).R)
    # Exact Wishart bias correction for Gaussian S1; see the note in
    # `complete_distribution`.
    bias_correction = (r1 - n_cols - 1) / r1
    S2 = materialize_right_compressor(
        ingredients.compressor_recipe_2, n_cols, r2, eltype(A)
    )
    M = R \ S2
    Ω = A * M
    ingredients.weights =
        ProbabilityWeights(bias_correction .* vec(sum(abs2, Ω, dims = 2)))
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
