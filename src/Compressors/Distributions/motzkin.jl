"""
    Motzkin <: Distribution

Motzkin sampling distribution, also known as the greedy or maximum residual selection 
method, as proposed by [motzkin1954relaxation](@citet) and analyzed in the context of 
Sampling Kaczmarz-Motzkin by [haddock2020greed](@citet).

# Mathematical Description

During sampling, the Motzkin distribution selects an index from rows (`Left()`) or
columns (`Right()`) with cardinality-dependent dimensions.

- **`Left()` cardinality:** work on ``Ax = b`` with ``A \\in \\mathbb{R}^{m\\times n}``,
    ``x \\in \\mathbb{R}^{n}``, ``b \\in \\mathbb{R}^{m}``; score row ``i`` by
    ``r_i = |A_{i,:}x - b_i|``.
- **`Right()` cardinality:** use the transposed-system view with
    ``x \\in \\mathbb{R}^{m}``, ``b \\in \\mathbb{R}^{n}``; score column ``j`` by
    ``c_j = |A_{:,j}^T x - b_j|``.

The algorithm works as follows over the active index set:
1. **If β = 1 (Randomized Kaczmarz)**: Randomly select one index uniformly.
2. **If β = d (Pure Motzkin/Greedy)**: Select the top-`k` indices with maximum
    residuals.
3. **If 1 < β < d**: Randomly sample β distinct indices, then select the one
    with maximum residuals within the sampled subset.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()`, `Right()`,
    or `Undef()`. The default value is `Undef()`.
- `replace::Bool`, if `true`, then the sampling occurs with replacement; if `false`, 
    then the sampling occurs without replacement. The default value is `false`.
- `beta::Int`, the subset size for sampling (1 ≤ β ≤ d), where ``d`` is the active
    sampling dimension (`m` for `Left()`, `n` for `Right()`). When β = 1, this
    reduces to uniform random selection. When β = d, this becomes pure greedy
    Motzkin selection.
    The default value is 1.

# Constructor

    Motzkin(;cardinality=Undef(), replace=false, beta=1)

# Returns
- A `Motzkin` object.

# Throws
- `ArgumentError` if `beta` < 1.
"""
mutable struct Motzkin <: Distribution
    cardinality::Cardinality
    replace::Bool
    beta::Int  # Subset size for sampling (1 ≤ beta ≤ d)
end

function Motzkin(; cardinality = Undef(), replace = false, beta = 1)
    if beta < 1
        throw(
            ArgumentError(
                "`Motzkin` beta must be >= 1, got beta=$beta"
            )
        )
    end
    return Motzkin(cardinality, replace, beta)
end

"""
    MotzkinRecipe <: DistributionRecipe

The recipe containing all allocations and information for the Motzkin distribution.

# Fields
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. For Motzkin,
    this should be `Left()` or `Right()`.
- `replace::Bool`, an option to replace or not during the sampling process.
- `beta::Int`, the subset size for sampling (1 ≤ β ≤ d), where ``d`` is the
    active sampling dimension (`m` for `Left()`, `n` for `Right()`).
- `state_space::Vector{Int64}`, the active row/column index set.
- `sample_buffer::Vector{Int64}`, workspace to store the randomly sampled 
    subset of β indices.
- `A::AbstractMatrix`, reference to the coefficient matrix.
- `b::AbstractVector`, reference to the constant vector.
- `x::AbstractVector`, reference to the current solution iterate (updated each iteration).

!!! note "Implementation note"
    In `sample_distribution!`, when ``1 < β < d``, residuals are evaluated only on
    the β sampled indices (not over the full domain). For `Left()`, this means
    sampled rows; for `Right()`, sampled columns. This is efficient when β is much
    smaller than the active dimension.

!!! note "Developer note"
    We intentionally keep `update_distribution!` deterministic and perform all
    randomization in `sample_distribution!`. This keeps update semantics clean,
    and matches user expectations in iterative usage, where repeated sampling
    calls should produce fresh samples each time.
"""
mutable struct MotzkinRecipe <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    beta::Int
    state_space::Vector{Int64}
    sample_buffer::Vector{Int64}
    A::AbstractMatrix
    b::AbstractVector
    x::AbstractVector
end

"""
    complete_distribution(
        distribution::Motzkin,
        x::AbstractVector,
        A::AbstractMatrix,
        b::AbstractVector
    )

Creates a `MotzkinRecipe` for the given Motzkin distribution and linear system ``Ax = b``.

# Arguments
- `distribution::Motzkin`: The Motzkin distribution specification.
- `x::AbstractVector`: Current solution iterate (`n` for `Left()`, `m` for `Right()`).
- `A::AbstractMatrix`: Coefficient matrix.
- `b::AbstractVector`: Constant vector (`m` for `Left()`, `n` for `Right()`).

# Returns
- `MotzkinRecipe`: A recipe containing all necessary allocations and references to A, b, x.

# Throws
- `ArgumentError` if cardinality is `Undef()`.
- `DimensionMismatch` if vector dimensions don't match the active cardinality layout.
- `ArgumentError` if beta > number of rows (`Left()`) or columns (`Right()`) in A.
"""
function complete_distribution(
    distribution::Motzkin,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector
)
    cardinality = distribution.cardinality

    n_rows = size(A, 1)
    n_cols = size(A, 2)

    sampling_dimension, expected_x_len, expected_b_len = if cardinality == Left()
        (n_rows, n_cols, n_rows)
    elseif cardinality == Right()
        (n_cols, n_rows, n_cols)
    elseif cardinality == Undef()
        throw(
            ArgumentError(
                "`Motzkin` cardinality must be specified as `Left()` or `Right()`.\
                `Undef()` is not allowed in `complete_distribution`."
            )
        )
    end

    # Validate dimensions
    if length(b) != expected_b_len
        dim_name = cardinality == Left() ? "rows of A" : "columns of A"
        throw(
            DimensionMismatch(
                "Vector b has length $(length(b)), expected \
                $expected_b_len to match $dim_name"
            )
        )
    end
    if length(x) != expected_x_len
        dim_name = cardinality == Left() ? "columns of A" : "rows of A"
        throw(
            DimensionMismatch(
                "Vector x has length $(length(x)), expected \
                $expected_x_len to match $dim_name"
            )
        )
    end
    
    # Validate beta
    if distribution.beta > sampling_dimension
        dim_name = cardinality == Left() ? "rows" : "columns"
        throw(
            ArgumentError(
                "`Motzkin` beta must be <= number of $dim_name \
                ($sampling_dimension), got beta=$(distribution.beta)"
            )
        )
    end
    
    # Initialize state space from active cardinality
    state_space = collect(1:sampling_dimension)
    
    # Allocate buffer for sampled indices
    sample_buffer = zeros(Int64, distribution.beta)
    
    return MotzkinRecipe(cardinality, distribution.replace, distribution.beta,
        state_space, sample_buffer, A, b, x)
end

"""
    update_distribution!(
        ingredients::MotzkinRecipe,
        x::AbstractVector,
        A::AbstractMatrix,
        b::AbstractVector
    )

Updates the Motzkin distribution recipe with the current solution iterate x.

# Arguments
- `ingredients::MotzkinRecipe`: The recipe to update.
- `x::AbstractVector`: Current solution iterate (`n` for `Left()`, `m` for `Right()`).
- `A::AbstractMatrix`: Coefficient matrix.
- `b::AbstractVector`: Constant vector (`m` for `Left()`, `n` for `Right()`).

# Returns
- Modifies `ingredients` in place by updating the solution reference and returns nothing.

# Throws
- `DimensionMismatch` if vector dimensions don't match matrix dimensions.
- `ArgumentError` if beta > number of rows (`Left()`) or columns (`Right()`) in A.
"""
function update_distribution!(
    ingredients::MotzkinRecipe,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector
)
    n_rows = size(A, 1)
    n_cols = size(A, 2)

    sampling_dimension, expected_x_len, expected_b_len = if ingredients.cardinality == Left()
        (n_rows, n_cols, n_rows)
    elseif ingredients.cardinality == Right()
        (n_cols, n_rows, n_cols)
    elseif ingredients.cardinality == Undef()
        throw(
            ArgumentError(
                "`Motzkin` cardinality must be specified as `Left()` or `Right()`.\
                `Undef()` is not allowed in `update_distribution!`."
            )
        )
    end
    
    # Validate dimensions
    if length(b) != expected_b_len
        dim_name = ingredients.cardinality == Left() ? "rows of A" : "columns of A"
        throw(
            DimensionMismatch(
                "Vector b has length $(length(b)), expected \
                $expected_b_len to match $dim_name"
            )
        )
    end
    if length(x) != expected_x_len
        dim_name = ingredients.cardinality == Left() ? "columns of A" : "rows of A"
        throw(
            DimensionMismatch(
                "Vector x has length $(length(x)), expected \
                $expected_x_len to match $dim_name"
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
                "`Motzkin` beta must be <= number of $dim_name \
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
    
    return nothing
end

"""
    sample_distribution!(x::AbstractVector, distribution::MotzkinRecipe)

Samples indices according to the Motzkin distribution.

# Arguments
- `x::AbstractVector`: Output vector to store selected index/indices.
- `distribution::MotzkinRecipe`: The recipe containing residuals and
    sampling parameters.

# Returns
- Modifies `x` in place with the selected index/indices and returns nothing.

# Notes
- Returns the top-`k` selected indices where `k = length(x)`.
- Ties are broken deterministically by selecting smaller indices first.
- The selection within the sampled subset is deterministic.

# Throws
- `ArgumentError` if `length(x) > beta`.
- `ArgumentError` if cardinality is not `Left()` or `Right()`.
"""
function sample_distribution!(x::AbstractVector, distribution::MotzkinRecipe)
    active_dimension = length(distribution.state_space)
    k = length(x)

    if k > distribution.beta
        throw(
            ArgumentError(
                "`Motzkin` cannot output $k indices when beta=$(distribution.beta)."
            )
        )
    end

    if distribution.beta == 1
        # Pure random selection
        x[1] = rand(distribution.state_space)
        return nothing
    end

    # Candidate set
    candidates = if distribution.beta >= active_dimension
        # Pure greedy over full active domain
        distribution.state_space
    else
        # Sampling Kaczmarz-Motzkin: sample β indices first
        sample!(
            distribution.state_space,
            distribution.sample_buffer,
            replace = distribution.replace,
            ordered = false
        )
        distribution.sample_buffer
    end

    # Residuals on candidate set
    residuals = if distribution.cardinality == Left()
        abs.(distribution.A[candidates, :] * distribution.x - distribution.b[candidates])
    elseif distribution.cardinality == Right()
        # Transposed-system residuals for selected columns
        abs.(distribution.A[:, candidates]' * distribution.x - distribution.b[candidates])
    else
        throw(
            ArgumentError(
                "`Motzkin` cardinality must be `Left()` or `Right()`."
            )
        )
    end

    # Top-k by residual (descending), tie-break by smaller index
    p = partialsortperm(
        eachindex(residuals),
        1:k;
        by = i -> (-residuals[i], candidates[i])
    )
    @inbounds for t in 1:k
        x[t] = candidates[p[t]]
    end
    
    return nothing
end
