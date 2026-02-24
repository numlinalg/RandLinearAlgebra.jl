"""
    L2Norm <: Distribution

Distribution where the probability of selecting a row (or column) is proportional 
     to its squared Euclidean norm, as proposed by [strohmer2009randomized](@citet).

# Mathematical Description

During the sampling, the distribution is defined on the domain of row/column 
     indices based on their norms. If it's compressing from the left, 
     the probability ``p_i`` of selecting 
     row ``i`` is: ``p_i = \\frac{\\|A_{i,:}\\|_2^2}{\\|A\\|_F^2}``;
     If it's compressing from the right, the probability ``p_j`` of selecting 
     column ``j`` is: ``p_j = \\frac{\\|A_{:,j}\\|_2^2}{\\|A\\|_F^2}``.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()` 
    or `Undef()`. The default value is `Undef()`. 
- `replace::Bool`, if `true`, then the sampling occurs with replacement; if `false`, 
    then the sampling occurs without replacement. The default value is `false`. 

# Constructor

    L2Norm(;cardinality=Undef(), replace = false)

## Returns
- A `L2Norm` object.
"""
mutable struct L2Norm <: Distribution
    cardinality::Cardinality
    replace::Bool
end

function L2Norm(; cardinality = Undef(), replace = false)
    return L2Norm(cardinality, replace)
end

"""
    L2NormRecipe <: DistributionRecipe

The recipe containing all allocations and information for the 
     Strohmer-Vershynin distribution.

# Fields
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. The
    value is either `Left()` or `Right()` or `Undef()`.
- `replace::Bool`, an option to replace or not during the sampling process based 
    on the given weights.
- `state_space::Vector{Int64}`, the row/column index set.
- `weights::ProbabilityWeights`, the probability of each element in the state space, 
    calculated as the squared Euclidean norms.
"""
mutable struct L2NormRecipe <: DistributionRecipe
    cardinality::Cardinality
    replace::Bool
    state_space::Vector{Int64}
    weights::ProbabilityWeights
end

function complete_distribution(distribution::L2Norm, A::AbstractMatrix)
    cardinality = distribution.cardinality
    if cardinality == Left()
        n_rows = size(A, 1)
        state_space = collect(1: n_rows)
        # Calculate row norms squared: sum(|x|^2) along dimension 2
        weights = ProbabilityWeights(vec(sum(abs2, A, dims=2)))
    elseif cardinality == Right()
        n_cols = size(A, 2)
        state_space = collect(1: n_cols)
        # Calculate column norms squared: sum(|x|^2) along dimension 1
        weights = ProbabilityWeights(vec(sum(abs2, A, dims=1)))
    elseif cardinality == Undef()
        throw(
            ArgumentError(
                "`L2Norm` cardinality must be specified as `Left()` or `Right()`.\
                `Undef()` is not allowed in `complete_distribution`."
            )
        )
    end

    return L2NormRecipe(cardinality, distribution.replace, state_space, weights)
end

function update_distribution!(ingredients::L2NormRecipe, A::AbstractMatrix)
    if ingredients.cardinality == Left()
        n_rows = size(A, 1)
        length(ingredients.state_space) != n_rows && (ingredients.state_space = collect(1: n_rows))
        # Update weights based on current matrix values
        ingredients.weights = ProbabilityWeights(vec(sum(abs2, A, dims=2)))
    elseif ingredients.cardinality == Right()
        n_cols = size(A, 2)
        length(ingredients.state_space) != n_cols && (ingredients.state_space = collect(1: n_cols))
        # Update weights based on current matrix values
        ingredients.weights = ProbabilityWeights(vec(sum(abs2, A, dims=1)))
    elseif ingredients.cardinality == Undef()
        throw(
            ArgumentError(
                "`L2Norm` cardinality must be specified as `Left()` or `Right()`.\
                `Undef()` is not allowed in `update_distribution!`."
            )
        )
    end

    return nothing
end

function sample_distribution!(x::AbstractVector, distribution::L2NormRecipe)
    wsample!(distribution.state_space, distribution.weights, x, ordered = true, replace = distribution.replace)
    return nothing
end