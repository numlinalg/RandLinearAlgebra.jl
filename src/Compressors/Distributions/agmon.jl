"""
    Agmon <: Distribution

Agmon sampling distribution, implementing Agmon's maximal residual selection
method as proposed by [agmon1954relaxation](@citet). See [patel2023randomized](@citet)
(Supplement SM1.6) for a concise summary, and [haddock2020greed](@citet) for analysis
in the context of Sampling Kaczmarz--Motzkin.

# Mathematical Description

During sampling, the Agmon distribution scores indices of ``Ax = b``
(``A \\in \\mathbb{R}^{m\\times n}``, ``x \\in \\mathbb{R}^{n}``, ``b \\in \\mathbb{R}^{m}``)
by residual magnitude and selects the highest-scoring one(s):

- **`Left()` cardinality:** score row ``i`` by ``r_i = |A_{i,:}x - b_i|``.
- **`Right()` cardinality:** score column ``j`` by ``c_j = |A_{:,j}^T(Ax - b)|``.

The candidate set depends on ``β`` (subset size, ``1 ≤ β ≤ d``, where ``d`` is the
active dimension: ``d = m`` for `Left()`, ``d = n`` for `Right()`):
1. **``β = 1``**: Select one index uniformly at random.
2. **``β = d``**: Select greedily over the full index set (pure Agmon).
3. **``1 < β < d``**: Sample ``β`` indices at random, then select the index with the
    highest residual within the sampled subset.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()`, `Right()`,
    or `Undef()`. The default value is `Undef()`.
- `replace::Bool`, if `true`, then the sampling occurs with replacement; if `false`, 
    then the sampling occurs without replacement. The default value is `false`.
- `beta::Int`, the subset size for sampling (``1 ≤ β ≤ d``), where ``d`` is the active
    sampling dimension (``m`` for `Left()`, ``n`` for `Right()`). When ``β = 1``, this
    reduces to uniform random selection. When ``β = d``, this becomes pure greedy
    Agmon selection.
    The default value is 1.

# Constructor

    Agmon(;cardinality=Undef(), replace=false, beta=1)

## Keywords
- `cardinality::Cardinality`: the direction of index selection. Must be `Left()` or
    `Right()`. The default value is `Undef()`.
- `replace::Bool`: if `true`, sampling occurs with replacement. The default value is
    `false`.
- `beta::Int`: subset size for sampling (``1 ≤ β ≤ d``). The default value is `1`.

## Returns
- An `Agmon` object.

## Throws
- `ArgumentError` if `beta` < 1.
"""
mutable struct Agmon <: Distribution
    cardinality::Cardinality
    replace::Bool
    beta::Int  # Subset size for sampling (1 ≤ β ≤ d)
    function Agmon(cardinality::Cardinality, replace::Bool, beta::Int)
        if beta < 1
            throw(
                ArgumentError(
                    "`Agmon` beta must be >= 1, got beta=$beta"
                )
            )
        end
        new(cardinality, replace, beta)
    end
end

function Agmon(; cardinality::Cardinality = Undef(), replace::Bool = false, beta::Int = 1)
    return Agmon(cardinality, replace, beta)
end

"""
    AgmonRecipe <: DistributionRecipe

The recipe containing all allocations and information for the Agmon distribution.

# Fields
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. For Agmon,
    this should be `Left()` or `Right()`.
- `replace::Bool`, if `true`, then sampling is done with replacement; if `false`,
    then sampling is done without replacement.
- `beta::Int`, the subset size for sampling (``1 ≤ β ≤ d``), where ``d`` is the
    active sampling dimension (``m`` for `Left()`, ``n`` for `Right()`).
- `state_space::Vector{Int64}`, the active row/column index set.
- `sample_buffer::Vector{Int64}`, workspace to store the randomly sampled 
    subset of ``β`` indices.
- `A::AbstractMatrix`, reference to the coefficient matrix.
- `b::AbstractVector`, reference to the constant vector.
- `x::AbstractVector`, reference to the current solution iterate (updated each iteration).
- `r::Union{Nothing, AbstractVector}`, the residual vector ``Ax - b``. Starts as
    `nothing`; set by `update_distribution!`. If `nothing` when `sample_distribution!`
    is called, residuals are computed on-the-fly for the sampled candidates only.

!!! note "Determining Compression Subset"
    In general, the ``β`` candidates are selected uniformly with or without replacement
    depending on the value of `replace`.

    For `Left()`, the compression subset (whose size is determined by the argument
    passed to [`sample_distribution!`](@ref)) is selected by taking the top absolute
    residuals, stored in `r`, from the candidates.

    For `Right()`, the compression subset (whose size is determined by the argument
    passed to [`sample_distribution!`](@ref)) is selected by taking the top absolute
    residuals of the normal system. The residuals of the normal system are calculated
    by computing the coefficient matrix columns of the candidate subset and taking an
    inner product with the residuals, stored in `r`.

!!! note "Developer note"
    We intentionally keep `update_distribution!` deterministic and perform all
    randomization in `sample_distribution!`. This keeps update semantics clean,
    and matches user expectations in iterative usage, where repeated sampling
    calls should produce fresh samples each time.
"""
mutable struct AgmonRecipe <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    beta::Int
    state_space::Vector{Int64}
    sample_buffer::Vector{Int64}
    A::AbstractMatrix
    b::AbstractVector
    x::AbstractVector
    r::Union{Nothing, AbstractVector}
end

"""
    complete_distribution(
        distribution::Agmon,
        x::AbstractVector,
        A::AbstractMatrix,
        b::AbstractVector
    )

Creates an `AgmonRecipe` for the given Agmon distribution and linear system ``Ax = b``.

# Arguments
- `distribution::Agmon`: The Agmon distribution specification.
- `x::AbstractVector`: Current solution iterate of length ``n`` (columns of A).
- `A::AbstractMatrix`: Coefficient matrix.
- `b::AbstractVector`: Constant vector of length ``m`` (rows of A).

# Returns
- `AgmonRecipe`: A recipe containing all necessary allocations and references to A, b, x.

# Throws
- `ArgumentError` if cardinality is `Undef()`.
- `DimensionMismatch` if vector dimensions don't match the active cardinality layout.
- `ArgumentError` if β > number of rows (`Left()`) or columns (`Right()`) in A.
"""
function complete_distribution(
    distribution::Agmon,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector
)
    cardinality = distribution.cardinality

    n_rows = size(A, 1)
    n_cols = size(A, 2)

    sampling_dimension, expected_x_len, expected_b_len =
        if cardinality == Left()
            (n_rows, n_cols, n_rows)
        elseif cardinality == Right()
            (n_cols, n_cols, n_rows)
        elseif cardinality == Undef()
            throw(
                ArgumentError(
                    "`Agmon` cardinality must be specified as `Left()` or `Right()`.\
                    `Undef()` is not allowed in `complete_distribution`."
                )
            )
        end

    # Validate dimensions
    if length(b) != expected_b_len
        throw(
            DimensionMismatch(
                "Vector b has length $(length(b)), expected \
                $expected_b_len to match rows of A"
            )
        )
    end
    if length(x) != expected_x_len
        throw(
            DimensionMismatch(
                "Vector x has length $(length(x)), expected \
                $expected_x_len to match columns of A"
            )
        )
    end
    
    # Validate beta
    if distribution.beta > sampling_dimension
        dim_name = cardinality == Left() ? "rows" : "columns"
        throw(
            ArgumentError(
                "`Agmon` beta must be <= number of $dim_name \
                ($sampling_dimension), got beta=$(distribution.beta)"
            )
        )
    end
    
    # Initialize state space from active cardinality
    state_space = collect(1:sampling_dimension)

    # Allocate buffer for sampled indices
    sample_buffer = zeros(Int64, distribution.beta)

    return AgmonRecipe(cardinality, distribution.replace, distribution.beta,
        state_space, sample_buffer, A, b, x, nothing)
end

"""
    update_distribution!(
        ingredients::AgmonRecipe,
        x::AbstractVector,
        A::AbstractMatrix,
        b::AbstractVector;
        r::Union{Nothing, AbstractVector} = nothing
    )

Updates the Agmon distribution recipe with the current solution iterate x and
optionally the residual vector.

# Arguments
- `ingredients::AgmonRecipe`: The recipe to update.
- `x::AbstractVector`: Current solution iterate of length ``n`` (columns of A).
- `A::AbstractMatrix`: Coefficient matrix.
- `b::AbstractVector`: Constant vector of length ``m`` (rows of A).
- `r::Union{Nothing, AbstractVector}`: Optional residual vector. If `nothing`
    (default), recomputes ``Ax - b`` and stores it in the recipe. If provided,
    stores it as-is, allowing the caller to maintain `r` incrementally
    (e.g. ``r \\mathrel{{-}{=}} A S_k V_k``) to avoid a full recompute.

# Returns
- Modifies `ingredients` in place and returns nothing.

# Throws
- `DimensionMismatch` if vector dimensions don't match matrix dimensions.
- `ArgumentError` if β > number of rows (`Left()`) or columns (`Right()`) in A.
"""
function update_distribution!(
    ingredients::AgmonRecipe,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector;
    r::Union{Nothing, AbstractVector} = nothing
)
    n_rows = size(A, 1)
    n_cols = size(A, 2)

    sampling_dimension, expected_x_len, expected_b_len =
        if ingredients.cardinality == Left()
            (n_rows, n_cols, n_rows)
        elseif ingredients.cardinality == Right()
            (n_cols, n_cols, n_rows)
        elseif ingredients.cardinality == Undef()
            throw(
                ArgumentError(
                    "`Agmon` cardinality must be specified as `Left()` or `Right()`.\
                    `Undef()` is not allowed in `update_distribution!`."
                )
            )
        end

    # Validate dimensions
    if length(b) != expected_b_len
        throw(
            DimensionMismatch(
                "Vector b has length $(length(b)), expected \
                $expected_b_len to match rows of A"
            )
        )
    end
    if length(x) != expected_x_len
        throw(
            DimensionMismatch(
                "Vector x has length $(length(x)), expected \
                $expected_x_len to match columns of A"
            )
        )
    end
    
    # Update state space if active dimension changed
    if length(ingredients.state_space) != sampling_dimension
        ingredients.state_space = collect(1:sampling_dimension)
    end
    
    # Validate beta doesn't exceed active sampling dimension
    if ingredients.beta > sampling_dimension
        dim_name = ingredients.cardinality == Left() ? "rows" : "columns"
        throw(
            ArgumentError(
                "`Agmon` beta must be <= number of $dim_name \
                ($sampling_dimension), got beta=$(ingredients.beta)"
            )
        )
    end
    if length(ingredients.sample_buffer) != ingredients.beta
        ingredients.sample_buffer = zeros(Int64, ingredients.beta)
    end
    
    # Update current references
    ingredients.A = A
    ingredients.b = b
    ingredients.x = x

    # Store residual if provided; leave nothing to trigger on-the-fly in sample
    ingredients.r = r

    return nothing
end

"""
    sample_distribution!(indices::AbstractVector, distribution::AgmonRecipe)

Samples indices according to the Agmon distribution.

Residual strategy is controlled via `update_distribution!` and the stored
`distribution.r` field:

1. **On-the-fly** (`distribution.r === nothing`, default): computes residuals only
    for the β sampled candidates — O(β·n) for `Left()`. Best for small β.
2. **Stored exact** (call `update_distribution!` without `r`): residuals read from
    the stored ``r = Ax - b`` updated each iteration. O(β) per sample; O(mn) paid
    once in `update_distribution!`.
3. **Stored incremental** (call `update_distribution!` with `r`): residuals read from
    `r` maintained by the caller via ``r \\mathrel{{-}{=}} A S_k V_k``. Cheaper
    update cost; may accumulate drift over iterations.

!!! note "Design note"
    The optional `r` argument lives in `update_distribution!` to keep this function's
    contract simple: use `distribution.r` if available, fall back to on-the-fly if
    `nothing`. Discuss with advisor which strategy should be the default.

# Arguments
- `indices::AbstractVector`: Output vector to store selected index/indices.
- `distribution::AgmonRecipe`: The recipe containing sampling parameters.

# Returns
- Modifies `indices` in place with the selected index/indices and returns nothing.

## Notes
- Returns the top-`k` selected indices where `k = length(indices)`.
- Ties are broken deterministically by selecting smaller indices first.
- The selection within the sampled subset is deterministic.

# Throws
- `ArgumentError` if `length(indices) > β`.
- `ArgumentError` if cardinality is not `Left()` or `Right()`.

"""
function sample_distribution!(indices::AbstractVector, distribution::AgmonRecipe)
    active_dimension = length(distribution.state_space)
    k = length(indices)

    if k > distribution.beta
        throw(
            ArgumentError(
                "`Agmon` cannot output $k indices when beta=$(distribution.beta)."
            )
        )
    end

    if distribution.beta == 1
        # Pure random selection
        indices[1] = rand(distribution.state_space)
        return nothing
    end

    # Candidate set
    candidates = if distribution.beta >= active_dimension
        distribution.state_space
    else
        sample!(
            distribution.state_space,
            distribution.sample_buffer,
            replace = distribution.replace,
            ordered = false
        )
        distribution.sample_buffer
    end

    # Residuals: use stored r if available, otherwise compute on-the-fly
    residuals = if distribution.r !== nothing
        r = distribution.r
        if distribution.cardinality == Left()
            abs.(r[candidates])
        elseif distribution.cardinality == Right()
            abs.(distribution.A[:, candidates]' * r)
        else
            throw(ArgumentError("`Agmon` cardinality must be `Left()` or `Right()`."))
        end
    elseif distribution.cardinality == Left()
        abs.(distribution.A[candidates, :] * distribution.x - distribution.b[candidates])
    elseif distribution.cardinality == Right()
        r = distribution.A * distribution.x - distribution.b
        abs.(distribution.A[:, candidates]' * r)
    else
        throw(ArgumentError("`Agmon` cardinality must be `Left()` or `Right()`."))
    end

    # Top-k by residual (descending), tie-break by smaller index
    p = partialsortperm(
        eachindex(residuals),
        1:k;
        by = i -> (-residuals[i], candidates[i])
    )
    @inbounds for t in 1:k
        indices[t] = candidates[p[t]]
    end

    return nothing
end
