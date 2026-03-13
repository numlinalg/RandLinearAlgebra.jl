"""
    ColumnProjection <: Solver

A specification of a column projection solver, which is a generalization of 
    (block) coordinate descent for least squares problems. 
    These solvers iteratively update a solution by projecting the solution 
    onto a compressed column space of the coefficient matrix. 

# Mathematical Description 

Let ``A`` be an ``m \\times n`` matrix and consider solving the linear least squares
problem
``
\\min_{x} \\Vert b - Ax \\Vert_2.
``
Column projection methods refine a current iterate `x` by the update 
    ``
x_{+} = x + Sv,
``
where ``S`` is a compression and ``v`` is the minimum two-norm solution 
to 
``
\\min_{w} \\Vert b - A(x + Sw) \\Vert_2.
``

Explicitly, the solution is 
``
    v = (S^\\top A^\\top A S)^\\dagger (AS)^\\top (b - Ax)
    = (S^\\top A^\\top)^\\dagger (b - A x),
``
which yields     
``
x_{+} = x + S (S^\\top A^\\top)^\\dagger (b - Ax).
``

Here, we allow for an additional relaxation parameter, ``\\alpha``, which 
results in the update 
``
x_{+} = x + \\alpha S (S^\\top A^\\top)^\\dagger (b - Ax).
``
    
When the compression ``S`` is a vector, the update simplifies to 
``
x_{+} = x + \\alpha S \\frac{ (AS)^\\top (b - Ax) }{\\Vert AS \\Vert^2}.
``
Letting ``r = b - Ax``, the residual ``r_{+} = b - Ax_{+}`` can be computed 
by  
``r_{+} = r - \\alpha (AS) v.``

# Fields
- `alpha::Float64`, the over-relaxation parameter. 
- `compressor::Compressor`, a technique for forming the compressed column space of the 
    linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
    compressed column space.

# Constructor

    ColumnProjection(;
        alpha::Float64 = 1.0,
        compressor::Compressor = SparseSign(cardinality=Right()), 
        error::SolverError = LSGradient(),
        log::Logger = BasicLogger(),
        sub_solver::SubSolver = QRSolver(), 
    )

## Keywords
- `alpha::Float64`, the over-relaxation parameter. By default this value is 1.
- `compressor::Compressor`, a technique for forming the compressed column space of the 
    linear system. By default it is a [`SparseSign`](@ref) compressor.
- `error::SolverError`, a method for estimating the progress of the solver. By default it is 
    the [`LSGradient`](@ref) error method.
- `log::Logger`, a technique for logging the progress of the solver. By default it is
    [`BasicLogger`](@ref).
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
    compressed rowspace. When the `compression_dim = 1` this is not used. For all other 
    cases, the default is [`QRSolver`](@ref).

## Returns 
- A `ColumnProjection` object.

!!! info
    The `alpha` parameter should be in ``(0,2)`` for convergence to be guaranteed. 
    This condition is not enforced in the constructor. There are instances where 
    setting `alpha=2` can lead to non-convergent cycles [motzkin1954relaxation](@cite).
"""
mutable struct ColumnProjection <: Solver 
    alpha::Float64
    compressor::Compressor
    error::SolverError
    log::Logger
    sub_solver::SubSolver
    function ColumnProjection(alpha, compressor, error, log, sub_solver) 
        if typeof(compressor.cardinality) != Right
            @warn "Compressor has cardinality `Left` but ColumnProjection\
            compresses from the `Right`. This may cause an inefficiency."
        end

        new(alpha, compressor, error, log, sub_solver)
    end

end


function ColumnProjection(;
    alpha::Float64 = 1.0,
    compressor::Compressor = SparseSign(cardinality=Right()), 
    error::SolverError = LSGradient(),
    log::Logger = BasicLogger(),
    sub_solver::SubSolver = QRSolver(), 
)
    return  ColumnProjection(
        alpha, 
        compressor, 
        error, 
        log, 
        sub_solver
    )
end

#------------------------------------------------------------------
"""
    ColumnProjectionRecipe{
        T<:Number, 
        V<:AbstractVector,
        M<:AbstractArray, 
        MV<:SubArray,
        C<:CompressorRecipe, 
        L<:LoggerRecipe,
        E<:SolverErrorRecipe, 
        B<:SubSolverRecipe
    } <: SolverRecipe

A mutable structure containing all information relevant to the `ColumnProjection` solver. It 
    is formed by calling the function [`complete_solver`](@ref) on a `ColumnProjection` 
    object.

# Fields
- `compressor::CompressorRecipe`, a technique for forming the compressed column space of the 
    linear system.
- `log::LoggerRecipe`, a technique for logging the progress of the solver.
- `error::SolverErrorRecipe`, a method for estimating the progress of the solver.
- `sub_solver::SubSolverRecipe`, a technique to perform the projection of the solution 
    onto the compressed column space.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
    affect convergence.
- `compressed_mat::AbstractMatrix`, a matrix container for storing the compressed matrix. 
    Will be set to be the largest possible block size.
- `solution_vec::AbstractVector`, a vector container for storing the current solution 
    to the linear system.
- `update_vec::AbstractVector`, a vector container for storing the update to the solution.
- `mat_view::SubArray`, a container for storing a view of the compressed matrix container. 
    Using views here allows for variable block sizes.
- `residual_vec::AbstractVector`, a vector container for storing the residual at each 
    iteration.
"""
mutable struct ColumnProjectionRecipe{
    T<:Number, 
    V<:AbstractVector,
    M<:AbstractArray, 
    MV<:SubArray,
    C<:CompressorRecipe, 
    L<:LoggerRecipe,
    E<:SolverErrorRecipe, 
    B<:SubSolverRecipe
   } <: SolverRecipe
    compressor::C
    log::L
    error::E
    sub_solver::B
    alpha::Float64
    compressed_mat::M
    solution_vec::V
    update_vec::V
    mat_view::MV
    residual_vec::V
end

#------------------------------------------------------------------
function complete_solver(
    solver::ColumnProjection, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)

    # Dimension checking will be performed in the complete_compressor
    compressor = complete_compressor(solver.compressor, A, b)
    logger = complete_logger(solver.log)
    error = complete_error(solver.error, solver, A, b) 
    
    # Check that required fields are in the types
    !isdefined(error, :gradient) && throw(
        ArgumentError(
            "ErrorRecipe $(typeof(error)) does not contain the \
            field 'gradient' and is not valid for a `ColumnProjection` solver."
        )
    )

    !isdefined(logger, :max_it) && throw(
        ArgumentError(
            "LoggerRecipe $(typeof(logger)) does not contain \
            the field `max_it` and is not valid for a `ColumnProjection` solver."
        )
    )

    !isdefined(logger, :converged) && throw(
        ArgumentError(
            "LoggerRecipe $(typeof(logger)) does not contain \
            the field `converged` and is not valid for a `ColumnProjection` solver."
        )
    )

    # Compute compression matrix dimension = rows of matrix A by compression_dim 
    rows_a = size(A, 1)
    sample_size = compressor.n_cols


    # Allocate the information in the buffer using the types of A and b
    compressed_mat = typeof(A)(undef, rows_a, sample_size) #Stores A*compressor
    residual_vec = typeof(b)(undef, rows_a) #Stores b - Ax 

    # Since sub_solver is applied to compressed matrices use here
    sub_solver = complete_sub_solver(solver.sub_solver, compressed_mat, residual_vec)

    # View of the compressed matrix 
    mat_view = view(compressed_mat, :, 1:sample_size) 

    # update_vec is the solution to the subproblem and is used as
    # x_+ = x + S * update_vec 
    update_vec = typeof(x)(undef, sample_size)
    
    return ColumnProjectionRecipe{
        eltype(A), 
        typeof(b), 
        typeof(A), 
        typeof(mat_view),
        typeof(compressor),
        typeof(logger),
        typeof(error),
        typeof(sub_solver)
    }(
        compressor, 
        logger, 
        error,
        sub_solver,
        solver.alpha,
        compressed_mat,
        x,
        update_vec,
        mat_view,
        residual_vec
    )
end

"""
    colproj_update!(solver::ColumnProjectionRecipe)

A function that performs the column projection update when the compression dimension 
is one. 
If ``a = AS`` is the resulting compression of the transpose of the coefficient matrix,
and ``r = b - Ax`` is the current residual,
this function computes
``x_{+} = x + \\alpha S \\frac{ a^\\top (b-Ax) }{\\Vert a \\Vert_2^2},``
and 
``r_{+} = r_{-} - \\alpha a \\frac{ a^\\top r}{\\Vert a \\Vert_2^2}.``

# Arguments
- `solver::ColumnProjectionRecipe`, the solver information required for performing the update.

# Returns
- returns `nothing`
"""
function colproj_update!(solver::ColumnProjectionRecipe)
    # one-dimensional subarray
    solver.update_vec[1] = dot(solver.mat_view, solver.residual_vec) / 
        dot(solver.mat_view, solver.mat_view)

    # x_+ = x + alpha * S * update_vec
    mul!(solver.solution_vec, solver.compressor, solver.update_vec, solver.alpha, 1.0)

    # r_+ = r_- - alpha * mat_view * update_vec 
    mul!(solver.residual_vec, solver.mat_view, solver.update_vec, -solver.alpha, 1.0)
    return nothing
end

"""
    colproj_update_block!(solver::ColumnProjectionRecipe)

A function that performs the column projection update when the compression dimension 
is greater than 1. If ``S`` is the compression matrix,  
the compressed matrix is ``\\tilde A = A S``, and the residual is ``r = b - A x``, 
this function computes 
``v =  (\\tilde A^\\top \\tilde A)^\\dagger \\tilde A^\\top r``
and stores it in `solver.update_vec`;
``x_+ = x + \\alpha S v;``
and stores it in `solver.solution_vec`; and
``r_+ = r - \\alpha \\tilde A v``
and stores it in `solver.residual_vec`.

# Arguments
- `solver::ColumnProjectionRecipe`, the solver information required for performing the update.

# Returns
- returns `nothing`
"""
function colproj_update_block!(solver::ColumnProjectionRecipe)
    # update the subsolver and solve for update vector
    update_sub_solver!(solver.sub_solver, solver.mat_view)
    ldiv!(solver.update_vec, solver.sub_solver, solver.residual_vec)

    # x_+ = x + alpha * S * update_vec
    mul!(solver.solution_vec, solver.compressor, solver.update_vec, solver.alpha, 1.0)

    # r_+ = r - alpha * (AS) * update_vec 
    mul!(solver.residual_vec, solver.mat_view, solver.update_vec, -solver.alpha, 1.0)
    return nothing
end

function rsolve!(
    solver::ColumnProjectionRecipe, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    # initialization
    reset_logger!(solver.log)
    solver.solution_vec = x
    copyto!(solver.residual_vec, b)

    # compute the residual b-Ax
    mul!(solver.residual_vec, A, solver.solution_vec, -1.0, 1.0)

    for i in 1:solver.log.max_it

        err = compute_error(solver.error, solver, A, b)

        # Update log adds value of err to log and checks stopping
        # We put in i-1 as this is a computation for the i-1 iteration
        update_logger!(solver.log, err, i-1) 
        if solver.log.converged
            return nothing
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.compressor, solver.solution_vec, A, b)

        # based on size of new compressor update views of matrix
        # this should not result in new allocations
        cols_s =  size(solver.compressor, 2)  
        solver.mat_view = view(solver.compressed_mat, :, 1:cols_s)

        # compress the matrix
        mul!(solver.mat_view, A, solver.compressor) 
        
        # Solve the undetermined sketched linear system and update the solution
        if cols_s == 1
            colproj_update!(solver)
        else
            colproj_update_block!(solver)
        end
    end

    # If the loop exits at the last iteration, we need to record the terminal 
    # error 
    err = compute_error(solver.error, solver, A, b)
    update_logger!(solver.log, err, solver.log.max_it)

    return nothing 
end
