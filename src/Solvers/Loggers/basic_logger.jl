"""
    BasicLogger <: Logger

This is a mutable struct that contains the `max_it` parameter and stores the error metric 
    in a vector. Checks convergence of the solver based on the log information.

# Fields
 - `max_it::Int64`, The maximum number of iterations for the solver. If not specified by the
    user, it is set to 3 times the number of rows in the matrix.
 - `threshold_info::Union{Float64, Tuple}`, The parameters used for stopping the algorithm.
 - `collection_rate::Int64`, the rate that history is gathered. (Note: The last value is 
    always recorded.)
 - `stopping_criterion::Function`, function that evaluates the stopping criterion.
"""
struct BasicLogger <: Logger
    max_it::Int64
    collection_rate::Int64
    threshold_info::Union{Float64, Tuple}
    stopping_criterion::Function
    function BasicLogger(max_it, collection_rate, threshold_info, stopping_criterion)
        if max_it < 0 
            throw(ArgumentError("Field `max_it` must be positive or 0."))
        elseif collection_rate < 1
            throw(ArgumentError("Field `colection_rate` must be positive."))
        elseif collection_rate > max_it && max_it > 0
            throw(ArgumentError("Field `colection_rate` must be smaller than `max_it`."))
        end

        return new(max_it, collection_rate, threshold_info, stopping_criterion)
    end

end

BasicLogger(;
            max_it::Int64 = 0,
            collection_rate::Int64 = 1,
            threshold::Union{Float64, Tuple} = 0.0,
            stopping_criterion::Function = threshold_stop
           ) = BasicLogger(max_it, collection_rate, threshold, stopping_criterion)

"""
    BasicLoggerRecipe <: LoggerRecipe

This is a mutable struct that contains the `max_it` parameter and stores the error metric 
    in a vector. Checks convergence of the solver based on the log information.

# Fields
 - `max_it::Int64`, The maximum number of iterations for the solver.
 - `error::Float64`, The current error metric.
 - `threshold_info::Union{Float64, Tuple}`, The parameters used for stopping the algorithm.
 - `iteration::Int64`, the current iteration of the solver.
 - `record_location::Int64`, the location in the history vector of the most recent entry.
 - `collection_rate::Int64`, the rate that history is gathered. (Note: The last value is 
    always recorded.)
 - `converged::Bool`, A boolean indicating whether the stopping criterion is satisfied.
 - `StoppingCriterion::Function`, function that evaluates the stopping criterion.
 - `hist:AbstractVector`, vector that contains the history of the error metric.
"""
mutable struct BasicLoggerRecipe{F<:Function} <: LoggerRecipe
    max_it::Int64
    error::Float64
    threshold_info::Union{Float64, Tuple}
    iteration::Int64
    record_location::Int64
    collection_rate::Int64
    converged::Bool
    stopping_criterion::F
    hist::Vector{Float64}
end


function complete_logger(logger::BasicLogger)
    # By using ceil if we divide exactly we always have space to record last value, if it 
    # does not divide exactly we have one more than required and thus enough space to record
    # the last value
    max_collection = Int(ceil(logger.max_it / logger.collection_rate))
    # use one more than max it form collection
    hist = zeros(max_collection + 1)
    return BasicLoggerRecipe{typeof(logger.stopping_criterion)}(logger.max_it,
                                                                0.0,
                                                                logger.threshold_info,
                                                                1,
                                                                1,
                                                                logger.collection_rate,
                                                                false,
                                                                logger.stopping_criterion,
                                                                hist
                                                               )
end

function update_logger!(logger::BasicLoggerRecipe, error::Float64, iteration::Int64)
    logger.iteration = iteration
    logger.error = error
    # Always check max_it stopping criterion
    # Compute in this way to avoid bounds error from searching in the max_it + 1 location
    logger.converged = iteration < logger.max_it ? 
        logger.stopping_criterion(logger) : 
        true 
    
    # log according to collection rate or if we have converged 
    if logger.converged
        logger.hist[logger.record_location] = error
    elseif rem(iteration, logger.collection_rate) == 0
        logger.hist[logger.record_location] = error
        logger.record_location += 1
    end

    return nothing
end

function reset_logger!(logger::BasicLoggerRecipe)
    logger.error = 0.0
    logger.iteration = 1
    logger.record_location = 1
    logger.converged = false
    fill!(logger.hist, 0.0)
    return nothing
end

"""
    threshold_stop(log::BasicLoggerRecipe)

Function that takes an input threshold and stops when the most recent entry in the history
vector is less than the threshold.

# Arguments
 - `log::LoggerRecipe`, a structure containing the logger information

# Bool
 - Returns a Bool indicating if the stopping threshold is satisfied.
"""
function threshold_stop(log::LoggerRecipe)
    return log.error < log.threshold_info
end
