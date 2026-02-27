###################################
# Abstract Types
###################################
"""
    Selector

An abstract type containing user controlled parameters for a technique that selects indices
from a matrix.
"""
abstract type Selector end

"""
    SelectorRecipe

An abstract type containing user controlled parameters and preallocated memory for a
technique that selects indices from a matrix.
"""
abstract type SelectorRecipe end

select_arg_list = Dict{Symbol,String}(
    :selector => "`selector::Selector`, a data structure containing the user-defined
    parameters associated with a particular selection method.",
    :selector_recipe => "`selector::SelectorRecipe`, a fully initialized realization
    for a selector method for a particular matrix.",
    :idx => "`idx::vector`, a vector where selected indices will be placed.",
    :n_idx => "`n_idx::Int64`, the number of indices to be selected.",
    :start_idx => "`start_idx::Int64`. the starting location in `idx` where the indices 
    will be placed.",
    :A => "`A::AbstractMatrix`, a target matrix for approximation.",
)
select_output_list = Dict{Symbol,String}(
    :selector_recipe => "A `SelectorRecipe` object.",
)
select_method_description = Dict{Symbol,String}(
    :update_selector => "A function that updates the `SelectorRecipe` in place given the
    arguments.",
    :complete_selector => "A function that generates a `SelectorRecipe` given
    arguments.",
    :select_indices => "A function that selects indices from a matrix `A` using a specific
    `SelectorRecipe`. It updates the vector `idx` in place with `n_idx` new indices starting
    at index `start_idx`."
)

select_error_list = Dict{Symbol,String}(
    :complete_selector => "`ArgumentError` if no method for completing the selector exists \
    for the given selector type.",
    :update_selector => "`ArgumentError` if no method for updating the selector exists for \
    the given selector type.",
    :select_indices => "`ArgumentError` if no method for selecting indices exists for the \
    given selector type."
)
##################################
# Complete Selector Interface
##################################
"""
    complete_selector(selector::Selector, A::AbstractMatrix)

$(select_method_description[:complete_selector])

# Arguments
- $(select_arg_list[:selector])
- $(select_arg_list[:A]) 

# Outputs
- $(select_output_list[:selector_recipe])

# Throws
- $(select_error_list[:complete_selector])
"""
function complete_selector(selector::Selector, A::AbstractMatrix)
    return throw(
        ArgumentError(
            "No method `complete_selector` exists for selector of type\
            $(typeof(selector)) and matrix of type $(typeof(A))."
        )
    )
end

##################################
# update_selector!
##################################
"""
    update_selector!(selector::SelectorRecipe)

$(select_method_description[:update_selector])

# Arguments
- $(select_arg_list[:selector_recipe])

# Outputs
- $(select_output_list[:selector_recipe])

# Throws
- $(select_error_list[:update_selector])
"""
function update_selector!(selector::SelectorRecipe)
    return throw(
        ArgumentError(
            "No method `update_selector!` exists for selector of type\
            $(typeof(selector))."
        )
    )
end

"""
    update_selector!(selector::SelectorRecipe, A::AbstractMatrix)

$(select_method_description[:update_selector])

# Arguments
- $(select_arg_list[:selector_recipe])
- $(select_arg_list[:A]) 

# Outputs
- $(select_output_list[:selector_recipe])
"""
function update_selector!(selector::SelectorRecipe, A::AbstractMatrix)
    return update_selector!(selector) 
end

####################################
# select_indices!
####################################
"""
    select_indices!(
        idx::AbstractVector,
        selector::SelectorRecipe, 
        A::AbstractMatrix,
        n_idx::Int64, 
        start_idx::Int64
    )

$(select_method_description[:select_indices])

# Arguments
- $(select_arg_list[:idx])
- $(select_arg_list[:selector_recipe])
- $(select_arg_list[:A]) 
- $(select_arg_list[:n_idx])
- $(select_arg_list[:start_idx])

# Outputs
-  Returns `nothing`

# Throws
- $(select_error_list[:select_indices])
"""
function select_indices!(
    idx::AbstractVector,
    selector::SelectorRecipe, 
    A::AbstractMatrix,
    n_idx::Int64, 
    start_idx::Int64
)
    return throw(
        ArgumentError(
            "No method `select_indices` exists for selector of type $(typeof(selector)),\
            matrix of type $(typeof(A)), idx of type $(typeof(idx)), n_idx of type \
            $(typeof(n_idx)), and start_idx of type $(typeof(start_idx))."
        )
    )
end

# Include the selector files
include("Selectors/lupp.jl")
include("Selectors/qrcp.jl")
