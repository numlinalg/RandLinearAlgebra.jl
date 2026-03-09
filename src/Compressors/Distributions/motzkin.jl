"""
    Motzkin <: Distribution

Motzkin sampling distribution, also known as the greedy or maximum residual selection 
method, as proposed by [motzkin1954relaxation](@citet) and analyzed in the context of 
Sampling Kaczmarz-Motzkin by [haddock2020greed](@citet).

# Mathematical Description

During the sampling, the Motzkin distribution selects rows based on their residuals 
for the linear system ``Ax = b``, where ``A`` is ``m \times n``. Given a current solution iterate ``x``, the residual 
for row ``i`` is ``r_i = |a_i^T x - b_i|``.

The algorithm works as follows:
1. **If β = 1 (Randomized Kaczmarz)**: Randomly select one row uniformly.
2. **If β = m (Pure Motzkin/Greedy)**: Select the row with maximum residual: ``i^* = \\arg\\max_i |a_i^T x - b_i|``.
3. **If 1 < β < m (Sampling Kaczmarz-Motzkin)**: Randomly sample β distinct rows, then 
   select the row with maximum residual among the sampled subset.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. For Motzkin sampling, only `Left()` is 
    supported (row selection). The default value is `Undef()`.
- `replace::Bool`, if `true`, then the sampling occurs with replacement; if `false`, 
    then the sampling occurs without replacement. The default value is `false`.
- `beta::Int`, the subset size for sampling (1 ≤ β ≤ m). When β = 1, this reduces to 
    uniform random selection. When β = m, this becomes pure greedy Motzkin selection.
    The default value is 1.

# Constructor

    Motzkin(;cardinality=Undef(), replace=false, beta=1)

## Returns
- A `Motzkin` object.
"""
mutable struct Motzkin <: Distribution
    cardinality::Cardinality
    replace::Bool
    beta::Int  # Subset size for sampling (1 ≤ beta ≤ m)
end

function Motzkin(; cardinality = Undef(), replace = false, beta = 1)
    if beta < 1
        throw(ArgumentError("`Motzkin` beta must be >= 1, got beta=$beta"))
    end
    return Motzkin(cardinality, replace, beta)
end

"""
    MotzkinRecipe <: DistributionRecipe

The recipe containing all allocations and information for the Motzkin distribution.

# Fields
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. For Motzkin,
    this should be `Left()` (row selection).
- `replace::Bool`, an option to replace or not during the sampling process.
- `beta::Int`, the subset size for sampling (1 ≤ β ≤ m).
- `state_space::Vector{Int64}`, the row index set {1, 2, ..., m}.
- `sample_buffer::Vector{Int64}`, workspace to store the randomly sampled subset of β indices.
- `A::AbstractMatrix`, reference to the coefficient matrix.
- `b::AbstractVector`, reference to the right-hand side vector.
- `x::AbstractVector`, reference to the current solution iterate (updated each iteration).

# Notes
- Following the paper's algorithm, residuals are computed only for the β sampled rows
    in `sample_distribution!`, not for all rows. This is efficient when β << m.
- `A` and `b` are stored as references (no copy), so they do not use extra memory.
- `x` is updated in `update_distribution!` at each iteration.
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
    complete_distribution(distribution::Motzkin, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)

Creates a `MotzkinRecipe` for the given Motzkin distribution and linear system Ax = b.

# Arguments
- `distribution::Motzkin`: The Motzkin distribution specification.
- `x::AbstractVector`: Current solution iterate (length n).
- `A::AbstractMatrix`: Coefficient matrix (size m × n).
- `b::AbstractVector`: Right-hand side vector (length m).

# Returns
- `MotzkinRecipe`: A recipe containing all necessary allocations and references to A, b, x.

# Throws
- `ArgumentError` if cardinality is not `Left()`.
- `ArgumentError` if cardinality is `Undef()`.
- `ArgumentError` if beta > number of rows in A.
"""
function complete_distribution(distribution::Motzkin, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    cardinality = distribution.cardinality
    
    # Motzkin only makes sense for row selection (solving Ax=b)
    if cardinality == Right()
        throw(ArgumentError("`Motzkin` distribution only supports `Left()` cardinality (row selection).\
        `Right()` cardinality is not supported."))
    elseif cardinality == Undef()
        throw(ArgumentError("`Motzkin` cardinality must be specified as `Left()`.\
        `Undef()` is not allowed in `complete_distribution`."))
    end
    
    n_rows = size(A, 1)
    n_cols = size(A, 2)
    
    # Validate dimensions
    if length(b) != n_rows
        throw(DimensionMismatch("Vector b has length $(length(b)), expected $n_rows to match rows of A"))
    end
    if length(x) != n_cols
        throw(DimensionMismatch("Vector x has length $(length(x)), expected $n_cols to match columns of A"))
    end
    
    # Validate beta
    if distribution.beta > n_rows
        throw(ArgumentError("`Motzkin` beta must be <= number of rows ($n_rows), got beta=$(distribution.beta)"))
    end
    
    # Initialize state space (all row indices)
    state_space = collect(1:n_rows)
    
    # Allocate buffer for sampled indices
    sample_buffer = zeros(Int64, distribution.beta)
    
    return MotzkinRecipe(cardinality, distribution.replace, distribution.beta,
        state_space, sample_buffer, A, b, x)
end

"""
    update_distribution!(ingredients::MotzkinRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)

Updates the Motzkin distribution recipe with the current solution iterate x.

# Arguments
- `ingredients::MotzkinRecipe`: The recipe to update.
- `x::AbstractVector`: Current solution iterate (length n).
- `A::AbstractMatrix`: Coefficient matrix (size m × n).
- `b::AbstractVector`: Right-hand side vector (length m).

# Outputs
- Modifies `ingredients` in place by updating the solution reference and returns nothing.

# Throws
- `DimensionMismatch` if vector dimensions don't match matrix dimensions.
- `ArgumentError` if beta > number of rows in A.
"""
function update_distribution!(ingredients::MotzkinRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    # TODO: Design decision for review - Currently, sampling S happens in sample_distribution!
    # to keep update_distribution! purely deterministic (consistent with L2Norm/Uniform).
    # Alternative: pre-draw S here after updating x, then only compute argmax in sample_distribution!
    # Both are functionally equivalent.
    
    n_rows = size(A, 1)
    n_cols = size(A, 2)
    
    # Validate dimensions
    if length(b) != n_rows
        throw(DimensionMismatch("Vector b has length $(length(b)), expected $n_rows to match rows of A"))
    end
    if length(x) != n_cols
        throw(DimensionMismatch("Vector x has length $(length(x)), expected $n_cols to match columns of A"))
    end
    
    # Update state space if matrix size changed
    if length(ingredients.state_space) != n_rows
        ingredients.state_space = collect(1:n_rows)
    end
    
    # Validate beta doesn't exceed n_rows
    if ingredients.beta > n_rows
        throw(ArgumentError("`Motzkin` beta must be <= number of rows ($n_rows), got beta=$(ingredients.beta)"))
    end
    if length(ingredients.sample_buffer) != ingredients.beta
        ingredients.sample_buffer = zeros(Int64, ingredients.beta)
    end
    
    # Update current solution reference
    ingredients.x = x
    
    return nothing
end

"""
    sample_distribution!(x::AbstractVector, distribution::MotzkinRecipe)

Samples row indices according to the Motzkin distribution.

# Arguments
- `x::AbstractVector`: Output vector to store selected row index/indices.
- `distribution::MotzkinRecipe`: The recipe containing residuals and \
    sampling parameters.

# Outputs
- Modifies `x` in place with the selected row index/indices and returns \
  nothing.

# Notes
- The output x typically has length 1 (single row selection per iteration).
- The selection within the sampled subset is deterministic (maximum residual).
"""
function sample_distribution!(x::AbstractVector, distribution::MotzkinRecipe)
    n_rows = length(distribution.state_space)
    
    if distribution.beta == 1
        # Pure random selection (Randomized Kaczmarz)
        x[1] = rand(distribution.state_space)
    elseif distribution.beta >= n_rows
        # Pure greedy (Motzkin method): find max residual over all rows
        # Alternative (allocates a length-m vector each call): 
        x[1] = argmax(abs.(distribution.A * distribution.x - distribution.b)) 
    else
        # Sampling Kaczmarz-Motzkin: sample β rows, compute their residuals, pick max
        sample!(distribution.state_space, distribution.sample_buffer,
        replace = false, ordered = false)

        r = abs.(distribution.A[distribution.sample_buffer, :] * distribution.x
                 - distribution.b[distribution.sample_buffer])
        x[1] = distribution.sample_buffer[argmax(r)]
    end
    
    return nothing
end
