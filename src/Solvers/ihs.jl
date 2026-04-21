"""
    IHS <: Solver

An implementation of the Iterative Hessian Sketch solver for solving over determined 
least squares problems [pilanci2014iterative](@cite).
 
# Mathematical Description
Let ``A  \\in \\mathbb{R}^{m \\times n}, m \\gg n,`` and consider the least square problem 
``\\min_x \\|Ax - b \\|_2^2``. If we let ``S \\in \\mathbb{R}^{s \\times m}`` be a 
compression matrix, then Iterative Hessian Sketch iteratively finds a solution to this 
problem by repeatedly updating ``x_{k+1} = x_k + \\alpha u_k``where ``u_k`` is the solution 
to the convex optimization problem, 
``u_k \\in \\argmin_u \\{\\|S_k Au\\|_2^2 - \\langle A, b - Ax_k \\rangle \\}.`` This method 
has been to shown to converge geometrically at a rate ``\\rho \\in (0, 1/2]``. Typically the 
required compression dimension needs to be 4-8 times the size of n for the algorithm to 
perform successfully.

# Fields
- `alpha::Float64`, a step size parameter.
- `compressor::Compressor`, a technique for forming the compressed linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.

# Constructor
    function IHS(;
        compressor::Compressor = SparseSign(cardinality = Left()),
        log::Logger = BasicLogger(),
        error::SolverError = FullResidual(),
        alpha::Float64 = 1.0
    )
## Keywords
- `compressor::Compressor`, a technique for forming the compressed linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `alpha::Float64`, a step size parameter.

# Returns
- A `IHS` object.
"""
mutable struct IHS <: Solver
    alpha::Float64
    log::Logger
    compressor::Compressor
    error::SolverError
    function IHS(alpha, log, compressor, error)
        if typeof(compressor.cardinality) != Left
            @warn "Compressor has cardinality `Right` but IHS compresses from the `Left`."
        end 

        if alpha < 0
            @warn "Negative step size could lead to divergent iterates."
        end

        new(alpha, log, compressor, error)
    end

end

function IHS(;
    compressor::Compressor = SparseSign(cardinality = Left()),
    log::Logger = BasicLogger(),
    error::SolverError = FullResidual(),
    alpha::Float64 = 1.0
)
    return IHS(
        alpha, 
        log,
        compressor,
        error
    )
end

"""
    IHSRecipe{
        Type<:Number, 
        LR<:LoggerRecipe,
        CR<:CompressorRecipe,
        ER<:ErrorRecipe,
        M<:AbstractArray, 
        MV<:SubArray, 
        V<:AbstractVector
    } <: SolverRecipe

A mutable structure containing all information relevant to the Iterative Hessian Sketch 
solver. It is formed by calling the function `complete_solver` on a `IHS` solver, which 
includes all the user controlled parameters, the linear system `A`, and the constant 
vector `b`.

# Fields
- `log::LoggerRecipe`, a technique for logging the progress of the solver.
- `compressor::CompressorRecipe`, a technique for compressing the matrix ``A``.
- `error::SolverErrorRecipe`, a technique for estimating the progress of the solver.
- `alpha::Float64`, a step size parameter, by default is set to 1.
- `compressed_mat::AbstractMatrix`, a buffer for storing the compressed matrix.
- `mat_view::SubArray`, a container for storing a view of the compressed matrix buffer.
- `residual_vec::AbstractVector`, a vector that contains the residual of the linear system 
    ``Ax-b``.
- `gradient_vec::AbstractVector`, a vector that contains the gradient of the least squares 
    problem, ``A^\\top(b-Ax)``.
- `buffer_vec::AbstractVector`, a buffer vector for storing intermediate linear system solves.
- `solution_vec::AbstractVector`, a vector storing the current IHS solution.
- `R::UpperTriangular`, a container for storing the upper triangular portion of the R 
    factor from a QR factorization of `mat_view`. This is used to solve the IHS sub-problem.
"""
mutable struct IHSRecipe{
    Type<:Number, 
    LR<:LoggerRecipe,
    CR<:CompressorRecipe,
    ER<:SolverErrorRecipe,
    M<:AbstractArray, 
    MV<:SubArray, 
    V<:AbstractVector
} <: SolverRecipe
    log::LR
    compressor::CR
    error::ER
    alpha::Float64
    compressed_mat::M
    mat_view::MV
    residual_vec::V
    gradient_vec::V
    buffer_vec::V
    solution_vec::V
    R::UpperTriangular{Type, M}
end

function complete_solver(
    ingredients::IHS, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    compressor = complete_compressor(ingredients.compressor, x, A, b)
    logger = complete_logger(ingredients.log) 
    error = complete_error(ingredients.error, ingredients, A, b)
    sample_size::Int64 = compressor.n_rows
    rows_a, cols_a = size(A)
    # Check that required fields are in the types
    if !isdefined(error, :residual)
        throw(
            ArgumentError(
                "ErrorRecipe $(typeof(error)) does not contain the \
                field 'residual' and is not valid for an IHS solver."
            )
        )
    end

    if !isdefined(logger, :converged)
        throw(
            ArgumentError(
                "LoggerRecipe $(typeof(logger)) does not contain \
                the field 'converged' and is not valid for an IHS solver."
            )
        )
    end

    # Check that the sketch size is larger than the column dimension and return a warning
    # otherwise
    if cols_a > sample_size
        throw(
            ArgumentError(
                "Compression dimension not larger than column dimension this will lead to \
                singular QR decompositions, which cannot be inverted."
            )
        )
    end

    if rows_a < sample_size
        throw(
            ArgumentError(
                "Compression dimension larger row dimension."
            )
        )
    end

    if cols_a >= rows_a
        throw(
            ArgumentError(
                "Matrix must have more rows than columns."
            )
        )
    end
    compressed_mat = fill!(similar(A, eltype(A), sample_size, cols_a), zero(eltype(A)))
    res = fill!(similar(A, eltype(A), rows_a), zero(eltype(A)))
    grad = fill!(similar(A, eltype(A), cols_a), zero(eltype(A)))
    buffer_vec = fill!(similar(A, eltype(A), cols_a), zero(eltype(A)))
    solution_vec = x
    mat_view = view(compressed_mat, 1:sample_size, :)
    R = UpperTriangular(mat_view[1:cols_a, :])

    return IHSRecipe{
        eltype(compressed_mat),
        typeof(logger),
        typeof(compressor),
        typeof(error),
        typeof(compressed_mat),
        typeof(mat_view),
        typeof(buffer_vec)
    }(
        logger, 
        compressor, 
        error, 
        ingredients.alpha,
        compressed_mat, 
        mat_view, 
        res, 
        grad, 
        buffer_vec, 
        solution_vec, 
        R
    )
end

function rsolve!(solver::IHSRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    reset_logger!(solver.log)
    solver.solution_vec = x
    err = 0.0
    copyto!(solver.residual_vec, b)
    # compute the initial residual r = b - Ax
    mul!(solver.residual_vec, A, solver.solution_vec, -1.0, 1.0)
    for i in 1:solver.log.max_it
        # compute the gradient A'r
        mul!(solver.gradient_vec, A', solver.residual_vec)
        err = compute_error(solver.error, solver, A, b)
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return nothing
        end

        # generate a new compressor
        update_compressor!(solver.compressor, x, A, b)
        # Based on the size of the compressor update views of the matrix
        rows_s, cols_s = size(solver.compressor)
        solver.mat_view = view(solver.compressed_mat, 1:rows_s, :)
        # Compress the matrix
        mul!(solver.mat_view, solver.compressor, A)
        # Update the subsolver 
        # Note: qr (non-mutating) is used for GPU compatibility; qr! is not supported on GPU
        solver.R = UpperTriangular(qr(solver.mat_view).R)
        # Compute first R' solver R'R x = g
        ldiv!(solver.buffer_vec, solver.R', solver.gradient_vec)
        # Compute second R Solve Rx = (R')^(-1)g will be stored in gradient_vec
        ldiv!(solver.gradient_vec, solver.R, solver.buffer_vec)
        # update the solution
        # solver.solution_vec = solver.solution_vec + alpha * solver.gradient_vec
        axpy!(solver.alpha, solver.gradient_vec, solver.solution_vec)
        # compute the fast update of r = r - A * gradient_vec
        # note: in this case gradient vec stores the update
        mul!(solver.residual_vec, A, solver.gradient_vec, -1.0, 1.0)
    end

    return nothing

end
