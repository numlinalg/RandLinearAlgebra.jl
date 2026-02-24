"""
    Distribution

An abstract supertype for structures specifying distribution for indices in sampling methods.
"""
abstract type Distribution end

"""
    DistributionRecipe

An abstract supertype for structures with pre-allocated memory for distribution function
    sampling methods.
"""
abstract type DistributionRecipe end

# Docstring Components
distribution_arg_list = Dict{Symbol,String}(
    :distribution => "`distribution::Distribution`, a user-specified distribution function for sampling.",
    :distribution_recipe => "`distribution::DistributionRecipe`, a fully initialized realization of distribution.",
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
    :x => "`x::AbstractVector`, an abstract vector to store the sampled indices.",
)

distribution_output_list = Dict{Symbol,String}(
    :distribution_recipe => "A `DistributionRecipe` object."
)

distribution_method_description = Dict{Symbol,String}(
    :complete_distribution => "A function that generates a `DistributionRecipe` given the 
    arguments.",
    :update_distribution! => "A function that updates the `DistributionRecipe` in place given 
    arguments.",
    :sample_distribution! => "A function that in place updates the `x` by given `DistributionRecipe` info.",
)
"""
    complete_distribution(distribution::Distribution, A::AbstractMatrix)

$(distribution_method_description[:complete_distribution])

# Arguments
- $(distribution_arg_list[:distribution])
- $(distribution_arg_list[:A]) 

# Outputs
- $(distribution_output_list[:distribution_recipe])
"""
function complete_distribution(distribution::Distribution, A::AbstractMatrix)
    return throw(ArgumentError("No `complete_distribution` method defined for a distribution of type \
    $(typeof(distribution)) and $(typeof(A))."))
end

function complete_distribution(distribution::Distribution, A::AbstractMatrix, b::AbstractVector)
    return complete_distribution(distribution, A)
end

function complete_distribution(distribution::Distribution, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    return complete_distribution(distribution, A, b)
end

"""
    update_distribution!(distribution::DistributionRecipe, A::AbstractMatrix)

$(distribution_method_description[:update_distribution!])

# Arguments
- $(distribution_arg_list[:distribution_recipe])
- $(distribution_arg_list[:A]) 

# Outputs
- Modifies the `DistributionRecipe` in place and returns nothing.
"""
function update_distribution!(distribution::DistributionRecipe, A::AbstractMatrix)
    return throw(ArgumentError("No `update_distribution!` method defined for a distribution of type \
    $(typeof(distribution)) and $(typeof(A))."))
end

function update_distribution!(distribution::DistributionRecipe, A::AbstractMatrix, b::AbstractVector)
    return update_distribution!(distribution, A)
end

function update_distribution!(distribution::DistributionRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    return update_distribution!(distribution, A, b)
end

"""
    sample_distribution!(x::AbstractVector, distribution::DistributionRecipe)

$(distribution_method_description[:sample_distribution!])

# Arguments
- $(distribution_arg_list[:x]) 
- $(distribution_arg_list[:distribution_recipe])

# Outputs
- Modifies the `x` in place by sampling that follows the weights and replacement given by 
'DistributionRecipe'.
"""
function sample_distribution!(x::AbstractVector, distribution::DistributionRecipe)
    return throw(ArgumentError("No `sample_distribution!` method defined for a distribution of type \
    $(typeof(distribution)) and $(typeof(x))."))
end

###########################################
# Include Distribution files
###########################################
include("Distributions/uniform.jl")
include("Distributions/strohmer_vershynin.jl")