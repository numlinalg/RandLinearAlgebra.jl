"""
    fwht!(x::AbstractVector, signs::BitVector; scaling::Int64 = 1)  

Performs an in-place Fast Walsh Hadamard Transform on the vector x.  
`signs` allows the user to input a boolean vector that flips the signs of the entries 
of the vector `x` before applying the transform. `scaling` allows the user to scale the 
result of the transform. Choosing a scaling of 1/sqrt{size(x)} will result in the FWHT 
being an orthogonal transform.

# Arguments
- `x::AbstractVector`, the vector the FJLT will be applied to.
- `signs::Vector{Bool}`, whether the sign of each entry is positive or negative.

# Keywords
- `scaling::Number`, how much the vector is scaled, by default this is 1.


# Returns
- `nothing`
"""
function fwht!(x::AbstractVector, signs::BitVector; scaling = 1)
    oa = 0
    ln = size(x,1)
    ls = size(signs,1)
    if ls != ln
        throw(DimensionMismatch("Vector `x` and vector `signs` must be same length."))
    elseif rem(log(2, ln), 1) != 0 
        throw(DimensionMismatch("Size of vector must be power of 2."))
    end

    # size of separation between indices
    h = 1 
    #  set increment to be twice of h
    inc = h << 1
    # total iterations required ends up being this division minus one for the separate
    # loop at the beginning  
    total_its = Int64(log(2, ln)) - 1
    # In 1st pass scale and flip signs of entries, this does not need to be done 
    # in any other passes over the vector. To avoid unnecessary condition checking 
    # this portion of the loop has been separated out from the others.
    @inbounds for i in 1:inc:ln
        # Signs is vector of Bools
        z = x[i] * (signs[i] ? scaling : -scaling) 
        y = x[i+h] * (signs[i+h] ? scaling : -scaling)
        x[i] = z + y
        x[i+h] = z - y
        
    end

    @inbounds for k = 1:total_its
        # Double distance between next combintation
        h <<= 1
        # spacing between operations
        inc = h << 1
        for i in 1:inc:ln
            for j in i:(i+h-1)
                # Perform fwht 
                z = x[j]  
                y = x[j+h]
                x[j] = z + y
                x[j+h] = z - y
            end
        
        end

    end

    return nothing

end

# define a version that has no signs input
function fwht!(x::AbstractVector; scaling = 1)
    oa = 0
    ln = size(x,1)
    if rem(log(2, ln), 1) != 0 
        throw(DimensionMismatch("Size of vector must be power of 2."))
    end
    # size of separation between indicies
    h = 1 
    #  set increment to be twice of h
    inc = h << 1
    # total iterations required ends up being this division minus one for the separate
    # loop at the beginning  
    total_its = Int64(log(2, ln)) - 1
    # In 1st pass scale and flip signs of entries, this does not need to be done 
    # in any other passes over the vector. To avoid unnecessary condition checking 
    # this portion of the loop has been separated out fron the others.
    @inbounds for i in 1:inc:ln
        # Signs is vector of Bools
        z = x[i] * scaling 
        y = x[i+h] * scaling
        x[i] = z + y
        x[i+h] = z - y
        
    end

    @inbounds for k = 1:total_its
        # Double distance between next combintation
        h <<= 1
        # spacing between operations
        inc = h << 1
        for i in 1:inc:ln
            for j in i:(i+h-1)
                # Perform fwht 
                z = x[j]  
                y = x[j+h]
                x[j] = z + y
                x[j+h] = z - y
            end
        
        end

    end
    
    return nothing

end
