"""
    Kaczmarz <: Solver

An implementation of a Kaczmarz solver. Specifically, it is a solver that iteratively
    updates an iterate by projecting the iterate onto (a subspace of) the row space of a
    consistent linear system.

# Mathematical Description
Let ``A`` be an ``m \\times n`` matrix and consider the consistent linear system ``Ax=b``. 
    We can view the solution to this linear system as lying at the intersection of the 
    row hyperplanes, 
    ``\\cap_{i \\in \\{1, \\ldots, m\\}}\\{u \\in \\mathbb{R}^{n} : A_{i \\cdot} u = b_i
    \\}``,
    where ``A_{i \\cdot}`` represents the ``i^\\text{th}`` row of ``A``. One way to find 
    this interesection is to iteratively project some abritrary point, ``x`` from one 
    hyperplane to the next, through 
    ``
    x_{+} = x + \\alpha \\frac{b_i -  A_{i\\cdot}^\\top x}{\\| A_{i\\cdot} \\|_2^2}
    ``
     Doing this with random permutation of ``i`` can lead to a geometric convergence 
    [strohmer2009randomized](@cite).
    Here ``\\alpha`` is viewed as an over-relaxation parameter and can improve convergence. 
    One can also generalize this procedure to blocks by considering the ``S`` being a 
    ``s \\times n`` random matrix. If we let ``\\tilde A = S A`` and ``\\tilde b = Sb`` 
    then we can perform block kaczmarz as described by [needell2014paved](@cite) with 
    ``
    x_{+} = x + \\alpha \\tilde A^\\top (\\tilde A \\tilde A^\\top)^\\dagger 
    (\\tilde b - \\tilde A x).
    ``
    While, `S` is often random, in reality, whether `S` is deterministic or random is 
    quite flexible see [patel2023randomized](@cite) for more details.
# Fields
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
    affect convergence.
- `compressor::Compressor`, a technique for forming the compressed rowspace of the linear 
    system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
    compressed rowspace.

# Constructor
    Kaczmarz(;
        compressor::Compressor = SparseSign(), 
        log::Logger = BasicLogger(),
        error::SolverError = FullResidual(),
        sub_solver::SubSolver = LQSolver(),
        alpha::Float64 = 1.0
    )
## Keywords
- `compressor::Compressor`, a technique for forming the compressed rowspace of the 
    linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
    compressed rowspace. When the `compression_dim = 1` this is not used.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
    affect convergence. By default this value is 1.

## Returns 
- A `Kaczmarz` object.

!!! info
    The `alpha` parameter should be in ``(0,2)``  for convergence to be guaranteed. This 
    condition is not enforced in the constructor. There are some instances where setting 
    `alpha = 2` can lead to non-convergent cycles [motzkin1954relaxation](@cite).
"""
mutable struct Kaczmarz <: Solver 
    alpha::Float64
    compressor::Compressor
    log::Logger
    error::SolverError
    sub_solver::SubSolver
    function Kaczmarz(alpha, compressor, log, error, sub_solver) 
        if typeof(compressor.cardinality) != Left
            @warn "Compressor has cardinality `Right` but kaczmarz\
            compresses  from the  `Left`."
        end

        new(alpha, compressor, log, error, sub_solver)
    end

end


function Kaczmarz(;
        compressor::Compressor = SparseSign(cardinality = Left()), 
        log::Logger = BasicLogger(),
        error::SolverError = FullResidual(),
        sub_solver::SubSolver = LQSolver(),
        alpha::Float64 = 1.0
    )
    # Intialize the datatype setting unkown elements to empty versions of correct datatype
    return Kaczmarz(
        alpha,
        compressor, 
        log, 
        error, 
        sub_solver
    )
end

"""
    KaczmarzRecipe{
        T<:Number, 
        V<:AbstractVector,
        M<:AbstractMatrix, 
        VV<:SubArray,
        MV<:SubArray,
        C<:CompressorRecipe, 
        L<:LoggerRecipe,
        E<:SolverErrorRecipe, 
        B<:SubSolverRecipe
    } <: SolverRecipe

A mutable structure containing all information relevant to the Kaczmarz solver. It is
    formed by calling the function `complete_solver` on `Kaczmarz` solver, which includes
    all the user controlled parameters, and the linear system matrix `A` and constant 
    vector `b`.

# Fields
- `compressor::CompressorRecipe`, a technique for forming the compressed rowspace of the 
    linear system.
- `log::LoggerRecipe`, a technique for logging the progress of the solver.
- `error::SolverErrorRecipe`, a method for estimating the progress of the solver.
- `sub_solver::SubSolverRecipe`, a technique to perform the projection of the solution onto
    the compressed rowspace.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
    affect convergence.
- `compressed_mat::AbstractMatrix`, a matrix container for storing the compressed matrix. 
    Will be set to be the largest possible block size.
- `compressed_vec::AbstractVector`, a vector container for storing the compressed constant
    vector. Will be set to be the largest possible block size.
- `solution_vec::AbstractVector`, a vector container for storing the solution to the linear
    system.
- `update_vec::AbstractVector`, a vector container for storing the update to the linear 
    system.
- `mat_view::SubArray`, a container for storing a view of compressed matrix container. 
    Using views here allows for variable block sizes.
- `vec_view::SubArray`, a container for storing a view of the compressed vector container.
    Using views here allows for variable block sizes.
"""
mutable struct KaczmarzRecipe{
    T<:Number, 
    V<:AbstractVector,
    M<:AbstractArray, 
    VV<:SubArray,
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
    compressed_vec::V
    solution_vec::Vector{T}
    update_vec::Vector{T}
    mat_view::MV
    vec_view::VV
end

function complete_solver(
    ingredients::Kaczmarz, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    # Dimension checking will be performed in the complete_compressor
    compressor = complete_compressor(ingredients.compressor, x, A, b)
    logger = complete_logger(ingredients.log)
    error = complete_error(ingredients.error, ingredients, A, b) 
    # Check that required fields are in the types
    if !isdefined(error, :residual)
        throw(
            ArgumentError(
                "ErrorRecipe $(typeof(error)) does not contain the \
                field 'residual' and is not valid for a kaczmarz solver."
            )
        )
    end

    if !isdefined(logger, :converged)
        throw(
            ArgumentError(
                "LoggerRecipe $(typeof(logger)) does not contain \
                the field 'converged' and is not valid for a kaczmarz solver."
            )
        )
    end

    # Assuming that max_it is defined in the logger
    alpha::Float64 = ingredients.alpha 
    # We assume the user is using compressors to only decrease dimension
    sample_size::Int64 = compressor.n_rows
    cols_a = size(A, 2)
    # because sampling recipe use subsets sparse matrices will be problematic
    if typeof(compressor) <: SamplingRecipe && typeof(A) <: SparseMatrixCSC
        compressed_mat = spzeros(eltype(A), sample_size, cols_a)
    else
        # Allocate the information in the buffer using the types of A
        compressed_mat = zeros(eltype(A), sample_size, cols_a)
    end
   
    # because sampling recipe use subsets sparse vectors will be problematic
    if typeof(compressor) <: SamplingRecipe && typeof(b) <: SparseVector
        compressed_vec = spzeros(eltype(b), sample_size)
    else
        # Allocate the information in the buffer using the type of b
        compressed_vec = zeros(eltype(b), sample_size)
    end

    # Since sub_solver is applied to compressed matrices use here
    sub_solver = complete_sub_solver(ingredients.sub_solver, compressed_mat, compressed_vec)
    mat_view = view(compressed_mat, 1:sample_size, :)
    vec_view = view(compressed_vec, 1:sample_size)
    solution_vec = x
    update_vec = zeros(eltype(x), cols_a)
    return KaczmarzRecipe{
        eltype(A), 
        typeof(b), 
        typeof(compressed_mat), 
        typeof(vec_view),
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
        alpha,
        compressed_mat,
        compressed_vec,
        solution_vec,
        update_vec,
        mat_view,
        vec_view
    )
end

"""
    kaczmarz_update!(solver::KaczmarzRecipe)

A function that performs the Kaczmarz update when the compression dimension is one. 
    If ``a`` is the resulting compression of the coefficient matrix, 
    and ``c`` is the resulting compression of the constant vector, 
    then we perform the update: ``x = x - \\alpha (a^\\top x -c) / \\|a\\|_2^2``. 

# Arguments
- `solver::KaczmarzRecipe`, the solver information required for performing the update.

# Returns
- returns `nothing`
"""
function kaczmarz_update!(solver::KaczmarzRecipe)
    # when the constant vector is a zero dimensional subArray we know that we should perform
    # the one dimension kaczmarz update

    # check that the row is non-zero
    if !iszero(solver.mat_view)
        # Compute the projection scaling (bi - dot(ai,x)) / ||ai||^2
        scaling = solver.alpha * (dotu(solver.mat_view, solver.solution_vec) 
            - solver.vec_view[1]) 
        scaling /= dot(solver.mat_view, solver.mat_view)
        # udpate the solution computes solution_vec = solution_vec - scaling * mat_view'
        axpby!(-scaling, solver.mat_view', 1.0, solver.solution_vec)
    end
    return nothing
end


"""
    kaczmarz_update_block!(solver::KaczmarzRecipe)

A function that performs the kaczmarz update when the compression dim is greater than 1.  
    In the block case where the compressed matrix ``\\tilde A``, and the compressed 
    contant vector ``\\tilde b``, we perform the updated: 
    ``x = x - \\alpha \\tilde A^\\top (\\tilde A \\tilde A^\\top)^\\dagger
    (\\tilde Ax-\\tilde b)``.

# Arguments
- `solver::KaczmarzRecipe`, the solver information required for performing the update.

# Returns
- returns `nothing`
"""
function kaczmarz_update_block!(solver::KaczmarzRecipe)
    # when the constant vector is a one dimensional subArray we know that we should perform
    # the one dimension kaczmarz update
    if !iszero(solver.mat_view)
        # sub-solver needs to designed for new compressed matrix
        update_sub_solver!(solver.sub_solver, solver.mat_view)
        # Compute the block residual 
        # (computes solver.vec_view - solver.mat_view * solver.solution_vec)
        mul!(solver.vec_view, solver.mat_view, solver.solution_vec, -1.0, 1.0)
        # use sub-solver to find update the solution (solves min ||tilde A - tilde b|| and 
        # stores in update_vec)
        ldiv!(solver.update_vec, solver.sub_solver, solver.vec_view)
        # computes solver.solution_vec = solver.solution_vec + alpha * solver.update_vec
        axpby!(solver.alpha, solver.update_vec, 1.0, solver.solution_vec)
    end

    return nothing
end

function rsolve!(
    solver::KaczmarzRecipe, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    reset_logger!(solver.log)
    # Check for dimension errors
    if size(x,1) != size(A, 2)
        throw(
            DimensionMismatch(
                "Dimension of `x`, $(size(x,1)) is different from number of columns in `A`,\
                $(size(A,2))."
            )
        )
    elseif size(b, 1) != size(A, 1)
        throw(
            DimensionMismatch(
                "Dimension of `b`, $(size(b,1)) is different from number of rows in `A`,\
                $(size(A,1))."
            )
        )

    end

    solver.solution_vec = x
    for i in 1:solver.log.max_it
        err = compute_error(solver.error, solver, A, b)
        # Update log adds value of err to log and checks stopping
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return nothing
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.compressor, x, A, b)
        # based on size of new compressor update views of matrix
        # this should not result in new allocations
        rows_s, cols_s =  size(solver.compressor)
        solver.mat_view = view(solver.compressed_mat, 1:rows_s, :)
        solver.vec_view = view(solver.compressed_vec, 1:rows_s)

        # compress the matrix and constant vector
        mul!(solver.mat_view, solver.compressor, A)
        mul!(solver.vec_view, solver.compressor, b)
        # Solve the undetermined sketched linear system and update the solution
        if size(solver.vec_view, 1) == 1
            kaczmarz_update!(solver)
        else
            kaczmarz_update_block!(solver)
        end

    end

    return nothing 
end
