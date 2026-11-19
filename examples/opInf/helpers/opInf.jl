"""
    vector_power(x::AbstractVector, p::Integer)

Function for creating a vector of all unique kronecker vector powers. 

# Fields
- `x::AbstractVector`, the vector being exponentiated
- `p::Integer`, the power the vector is being raised to
"""
function vector_power(x::AbstractVector, p::Integer)
    n = size(x, 1)
    # the vector power is the kroenecker product with repeat entries removed
    number_of_unique = binomial(n + p - 1, p)
    output = Array{eltype(x)}(undef, number_of_unique)

    # use fast implementation when power is less than 4
    o = 1
    if p == 2
        for i in 1:n
            for j in i:n
                output[o] = x[i] * x[j]
                o += 1
            end

        end

    elseif p == 3
        for i in 1:n
            for j in i:n
                for k in j:n
                    output[o] = x[i] * x[j] * x[k]
                    o += 1
                end

            end

        end
    
    elseif p == 4
        for i in 1:n
            for j in i:n
                for k in j:n
                    for l in k:n
                        output[o] = x[i] * x[j] * x[k] * x[l]
                        o += 1
                    end

                end

            end

        end
    
    # add general implementation where all combinations are computed first 
    else
        # make a vector of indices
        indices = ones(Int64, p)
        for i in 1:number_of_unique
            prod = 1
            for j in 1:p
                prod *= x[indices[j]]
            end

            output[o] = prod
            o += 1
            # increment the indices vector
            inc = true
            for k in p:-1:1
                # incremented if previous coordinate is n
                if inc 
                    if indices[k] == n
                        inc = true
                        # new min at previous coordinate will be incremented
                        # this loop is redundant and could be simplified in the future
                        l = k
                        while indices[l] == n && l >= 2
                            l -= 1
                        end

                        # set the coordinate index to correspond to the first non-n coordinate
                        indices[k] = indices[l] + 1
                    else
                        inc = false
                        indices[k] += 1
                    end

                end

            end

        end
        

    end

    return output
end

"""
    matrix_power

Function that applies the `vector_power` function to every vector in a matrix.

# Arguments
- `X::AbstractArray`, the matrix to be powered
- `p::Int64`, the power we wish to raise the matrix to

# Returns
- A matrix raised to the pth power in terms of unique kronecker products.
"""
function matrix_power(X::AbstractMatrix, p::Int64)
    function vec_power(x)
        return vector_power(x, p)
    end

    temp_list = vec_power.(eachcol(X))
    return reduce(hcat, temp_list)
end

"""
    create_regression_matrix

Function that creates the coefficient matrix to be used in the operator inference
least squares problem.

# Arguments
- `V::AbstractArray`, the matrix to project the trajectories into a lower dimensional subspace 
- `X::AbstractArray`, the matrix of trajectories
- `U::AbstractArray`, the base states for the trajectories 
- `degree::Integer`, the degree of the polynomial approximation we wish to make

# Returns
The matrix to be used in the operator fitting problem.
"""
function create_regression_matrix(
    V::AbstractArray, 
    X::AbstractArray, 
    U::AbstractArray, 
    degree::Integer
)
    # project to V subspace
    Xhat = V' * X
    m, n = size(Xhat)
    # determine size of outputted matrix for preallocation
    poly_sizes = zeros(Int64, degree + 1)
    for i = 1:degree + 1
        poly_sizes[i] = binomial(m + i - 1, i)
    end

    # compute number of rows required for matrix as 
    # total unique combinations plus one for the initial conditions
    n_cols = sum(poly_sizes) + size(U, 2)
    Reg_mat = Matrix{Float64}(undef, n, n_cols)
    start_idx = 1
    for i in 1:degree + 1
        term_idx = start_idx + poly_sizes[i] - 1
        Reg_mat[:, start_idx:term_idx] = matrix_power(Xhat, i)'
        start_idx = term_idx + 1
    end

    term_idx = start_idx + size(U, 2) - 1
    Reg_mat[:, start_idx:term_idx] .= U

    return Reg_mat
end

"""
    time_derivative_back_euler

Function that computes the time derivatives using the trajectories from a 
    numerical simulation.
    
# Arguments
- `X::AbstractArray`, the trajectories
- `delta_t::Float64`, the width of the timesteps
"""
function time_derivative_back_euler(X::AbstractArray, delta_t::Float64)
    dxdt = (X[:, 2:end] - X[:, 1:end-1]) / delta_t
    idx = 2:size(X,2)
    return dxdt, idx  
end

"""
    op_inf_problem
    
Function that computes the operator inference operators given trajectories and a projection subspace.

# Arguments
- `V::AbstractArray`, the matrix to project the trajectories into a lower dimensional subspace 
- `X::AbstractArray`, the matrix of trajectories
- `U::AbstractArray`, the base states for the trajectories 
- `degree::Integer`, the degree of the polynomial approximation we wish to make
- `ntimes::Number`, the number of timesteps in the simulation 
- `delta_t::Number`, the width of the timestep 
- `delta_x::Number`, the width of the spatial grid
- `solver::Function`, a function to solve the least squares problem

# Returns 
The trajectories corresponding to applying the operators for `ntimes` timesteps, 
    the time derivatives `R` and the coefficient matrix `Reg_mat`
"""
function op_inf_problem(
    V::AbstractArray, 
    X::AbstractArray, 
    U::AbstractArray, 
    degree::Integer, 
    ntimes::Number, 
    delta_t::Number, 
    delta_x::Number,
    solver::Function
)
    Xdot, idx = time_derivative_back_euler(X, delta_t)
    Reg_mat = create_regression_matrix(V, X[:, idx], U[idx], degree)
    R = Xdot' * V 
    sol = solver(Reg_mat, R)
    return sol, R, Reg_mat
end

"""
    least_squares_solve
    
Function that computes the soltution to a least squares problem.

# Arguments
- `Reg_mat::AbstractMatrix`, the coefficient matrix
-`R::AbstractArray`, the constant vector (matrix)

# Returns
The solution to the least squares problem
"""
function least_squares_solve(Reg_mat, R)
    return Reg_mat \ R
end 


"""
    GeneralBackwardEuler

Function that computes the trajectories using backward Euler.

# Arguments
- `A::AbstractMatrix`, the operator
- `B::AbstractArray`, the boundary operator
- `IC::AbstractArray`, the initial conditions
- `delta_t::Float64`, the width of the time steps
- `ntimes::Int64`, the number of time steps

# Returns
The trajectories corresponding to applying the operators for `ntimes` timesteps
"""
function GeneralBackwardEuler(
    A::AbstractMatrix, 
    B::AbstractArray, 
    IC::AbstractArray, 
    delta_t::Float64, 
    ntimes::Int64
)
    Out = zeros(size(A, 1), ntimes)
    buffer = zeros(size(A, 1))
    Out[:, 1] .= IC
    for i in 2:ntimes
        mv = view(Out, :, i)
        buffer .= Out[:, i - 1] + delta_t * B
        ldiv!(mv, qr(I - delta_t * A), buffer)
    end

    return Out
end