"""
    solve_back_euler_heat
    
Function that performs the implicit matrix inversion needed for backward Euler.

# Arguments
- `x::AbstractVector`, the vector storing the solution to the backward euler step 
- `b::AbstractVector`, the constant vector 
- `delta_t::Float64`, the distance of the timesteps 
- `delta_x::Float64`, the distance of the spatial grid points. 
- `mu::Float64`, heat conductivity coefficient

# Returns
- nothing
"""
function solve_back_euler_heat!(x::AbstractVector, b::AbstractVector, delta_t::Float64, delta_x::Float64, mu::Float64)
    # fill in x_new with the entries applying backward euler to x
    n_entries = size(x, 1)

    # compute on and off diagonal entries
    mat_scale = delta_t * mu / delta_x^2
    diagonal_entry = 1 + mat_scale * 2
    off_diagonal = -1 * delta_t * mu / delta_x^2
    # first pass of Thomas algorithm
    # To save memory we will save the upper diagonal information in the x vector and overwrite the b vector
    x[1] = off_diagonal / diagonal_entry 
    b[1] = b[1] / diagonal_entry
    for i in 2:n_entries - 1
        x[i] = off_diagonal / (diagonal_entry - off_diagonal * x[i - 1])
        b[i] = (b[i] - off_diagonal * b[i - 1]) / (diagonal_entry - off_diagonal * x[i - 1])
    end

    b[n_entries] = (b[n_entries] - off_diagonal * b[n_entries - 1]) / (diagonal_entry - off_diagonal * x[n_entries - 1]) 

    # now perform forward pass to solve the linear system
    x[n_entries] = b[n_entries]
    for i in n_entries-1:-1:1
        x[i] = b[i] - x[i] * x[i + 1]
    end

    return nothing

end

"""
    add_boundary!
    
Function that applies the boundary conditions to a vector.

# Arguments
- `u::AbstractVector`, the vector the boundary condition is applied to 
- `B::AbstractArray`, the boundary condition 
- `delta_t::Float64`, the width of the time step

# Returns
- nothing
"""
function add_boundary!(u::AbstractVector, B::AbstractArray, delta_t::Float64)
    u .+= delta_t * B

    return nothing
end

"""
    simulate_1d_heat

Function that performs a one-dimensional heat simulation.
- `initial_conditions::AbstractVector`, the initial conditions
- `B::AbstractArray`, the boundary conditions
- `ntimes::Int64`, the number of time steps
- `delta_t::Float64`, the timestep width
- `delta_x::Float64`, the spatial grid width
- `mu::Float64`, the heat conductivity coefficient

# Returns
- The trajectories from the heat simulation
"""
function simulate_1d_heat(initial_conditions::AbstractVector, B::AbstractArray, ntimes::Int64, delta_t::Float64, delta_x::Float64, mu::Float64)
    n = size(initial_conditions, 1)
    trajectories = zeros(n, ntimes)
    copyto!(view(trajectories, :, 1), initial_conditions)
    # from here use initial conditions vector as a buffer vector
    buffer = initial_conditions
    for i in 2:ntimes
        out = view(trajectories, :, i)
        # first incorporate the boundary conditions
        add_boundary!(buffer, B, delta_t)
        # now apply backward euler step
        solve_back_euler_heat!(out, buffer, delta_t, delta_x, mu) 
        # now copy the new state to the buffer for the next step
        copyto!(buffer, out)
    end

    return trajectories
end