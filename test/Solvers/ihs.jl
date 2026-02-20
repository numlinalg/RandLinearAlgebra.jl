module IHSTest
using Test, RandLinearAlgebra, LinearAlgebra
import RandLinearAlgebra: complete_compressor
import LinearAlgebra: mul!, norm
import Random: randn!, seed!
using ..FieldTest
using ..ApproxTol

# Define the test structures
##########################
# Compressors
##########################
mutable struct ITestCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
end

ITestCompressor() = ITestCompressor(Left(), 5)

mutable struct ITestCompressorRecipe <: CompressorRecipe 
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

function RandLinearAlgebra.complete_compressor(
    comp::ITestCompressor, 
    A::AbstractMatrix,
    b::AbstractVector 
)
    n_rows = comp.compression_dim
    n_cols = size(A, 1)
    # Make a gaussian compressor
    op = randn(n_rows, n_cols) ./ sqrt(n_cols)
    return ITestCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

function RandLinearAlgebra.update_compressor!(
    comp::ITestCompressorRecipe,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector
)
    randn!(comp.op)
    comp.op ./= sqrt(comp.n_cols)
end

# Define a mul function for the test compressor
function RandLinearAlgebra.mul!(
    C::AbstractArray,
    S::Main.IHSTest.ITestCompressorRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, S.op, A, alpha, beta)
end

##########################
# Error Method
##########################
mutable struct ITestError <: RandLinearAlgebra.SolverError
    g::Real
end

mutable struct ITestErrorRecipe <: RandLinearAlgebra.SolverErrorRecipe
    residual::Vector{Number}
end

ITestError() = ITestError(1.0) 

function RandLinearAlgebra.complete_error(
    error::ITestError, 
    solver::IHS,
    A::AbstractMatrix, 
    b::AbstractVector
)
    return ITestErrorRecipe(zeros(typeof(error.g), size(A, 1)))
end

function RandLinearAlgebra.compute_error(error::ITestErrorRecipe, solver, A, b)
    error.residual = A * solver.solution_vec - b
    return norm(error.residual)
end

##############################
# Residual-less Error Recipe
##############################
mutable struct ITestErrorNoRes <: RandLinearAlgebra.SolverError end

mutable struct ITestErrorRecipeNoRes <: RandLinearAlgebra.SolverErrorRecipe end

function RandLinearAlgebra.complete_error(
    error::ITestErrorNoRes, 
    solver::IHS,
    A::AbstractMatrix, 
    b::AbstractVector 
)
    return ITestErrorRecipeNoRes()
end

############################
# Loggers
############################
mutable struct ITestLog <: Logger
    max_it::Int64
    g::Number
end

ITestLog() = ITestLog(5, 1.0) 

ITestLog(max_it) = ITestLog(max_it, 1.0)

mutable struct ITestLogRecipe <: LoggerRecipe
    max_it::Int64
    hist::Vector{Real}
    thresh::Float64
    converged::Bool
    iteration::Int64
end

function RandLinearAlgebra.complete_logger(logger::ITestLog)
    return ITestLogRecipe(
        logger.max_it, 
        zeros(typeof(logger.g), logger.max_it), 
        logger.g, 
        false, 
        0
    )
end

function RandLinearAlgebra.update_logger!(logger::ITestLogRecipe, err::Real, i::Int64)
    logger.iteration = i
    logger.hist[i] = err
    logger.converged = err < logger.thresh ? true : false
end

function RandLinearAlgebra.reset_logger!(logger::ITestLogRecipe)
    fill!(logger.hist, 0.0)
end

##############################
# Converged-less Logger
##############################
mutable struct ITestLogNoCov <: Logger end

mutable struct ITestLogRecipeNoCov <: LoggerRecipe end

function RandLinearAlgebra.complete_logger(logger::ITestLogNoCov)
    return ITestLogRecipeNoCov()
end


@testset "IHS" begin
    seed!(12312)
    n_rows = 20
    n_cols = 2
    A = rand(n_rows, n_cols)
    xsol = rand(n_cols)
    b = A * xsol
    
    @testset "IHS" begin
        @test supertype(IHS) == Solver

        # test fieldnames and types
        @test fieldnames(IHS) == (:alpha, :log, :compressor, :error)
        @test fieldtypes(IHS) == (Float64, Logger, Compressor, SolverError)

        # test default constructor

        let solver = IHS()
            @test solver.alpha == 1.0
            @test typeof(solver.compressor) == SparseSign 
            @test typeof(solver.compressor.cardinality) == Left 
            @test typeof(solver.log) == BasicLogger
            @test typeof(solver.error) == FullResidual
        end

        # test constructor
        let solver = IHS(
            alpha = 2.0,
            compressor = ITestCompressor(),
            log = ITestLog(),
            error = ITestError(),
        )

            @test solver.alpha == 2.0
            @test typeof(solver.compressor) == ITestCompressor
            @test typeof(solver.log) == ITestLog
            @test typeof(solver.error) == ITestError
        end 
        
        # Test that warning gets returned with right compressor
        @test_logs (:warn,
            "Compressor has cardinality `Right` but IHS compresses from the `Left`."
        ) IHS(
            alpha = 2.0,
            compressor = ITestCompressor(Right(), 5),
            log = ITestLog(),
            error = ITestError(),
        )
        # Test that warning gets returned with  negative 
        @test_logs (:warn,
            "Negative step size could lead to divergent iterates."
        ) IHS(
            alpha = -2.0,
            compressor = ITestCompressor(Left(), 5),
            log = ITestLog(),
            error = ITestError(),
        )
    end
    
    @testset "IHSRecipe" begin
        @test supertype(IHSRecipe) == SolverRecipe

        # test fieldnames and types
        @test fieldnames(IHSRecipe) == (
            :log, 
            :compressor, 
            :error, 
            :alpha, 
            :compressed_mat, 
            :mat_view, 
            :residual_vec,
            :gradient_vec,
            :buffer_vec, 
            :solution_vec, 
            :R
        )
        @test fieldtypes(IHSRecipe) == (
            LoggerRecipe,
            CompressorRecipe,
            SolverErrorRecipe,
            Float64, 
            AbstractArray, 
            SubArray,
            AbstractVector, 
            AbstractVector, 
            AbstractVector, 
            AbstractVector, 
            UpperTriangular{Type, M} where {Type<:Number, M<:AbstractArray}
        )  
    end

    @testset "IHS: Complete Solver" begin
        # test error method with no residual error 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2 * size(A, 2),
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = ITestCompressor(Left(), comp_dim)
            log = ITestLog()
            err = ITestErrorNoRes()
            solver = IHS(
                log = log,
                compressor = comp,
                error = err,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "ErrorRecipe $(typeof(ITestErrorRecipeNoRes())) does not contain the \
                field 'residual' and is not valid for an IHS solver."
            ) complete_solver(solver, x, A, b)
        end

        # test logger method with no converged field 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2 * size(A, 2),
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = ITestCompressor(Left(), comp_dim)
            log = ITestLogNoCov()
            err = ITestError()
            solver = IHS(
                compressor = comp,
                log = log,
                error = err,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "LoggerRecipe $(typeof(ITestLogRecipeNoCov())) does not contain \
                the field 'converged' and is not valid for an IHS solver."
            ) complete_solver(solver, x, A, b)
        end

        # Test the error message about not large enough compression_dim 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 1,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = ITestCompressor(Left(), comp_dim)
            log = ITestLog()
            err = ITestError()
            solver = IHS(
                compressor = comp,
                log = log,
                error = err,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "Compression dimension not larger than column dimension this will lead to \
                singular QR decompositions, which cannot be inverted."
            ) complete_solver(solver, x, A, b)
        end

        # Test the error message about too large compression_dim 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = size(A, 1) + 1,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = ITestCompressor(Left(), comp_dim)
            log = ITestLog()
            err = ITestError()
            solver = IHS(
                compressor = comp,
                log = log,
                error = err,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "Compression dimension larger row dimension."
            ) complete_solver(solver, x, A, b)
        end

        # Test the error message about too large column dimension 
        let A = zeros(10, 10),
            xsol = xsol,
            b = b,
            comp_dim = 10,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = ITestCompressor(Left(), comp_dim)
            log = ITestLog()
            err = ITestError()
            solver = IHS(
                compressor = comp,
                log = log,
                error = err,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "Matrix must have more rows than columns."
            ) complete_solver(solver, x, A, b)
        end

        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 8 * size(A, 2),
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = ITestCompressor(Left(), comp_dim)
            log = ITestLog()
            err = ITestError()
            solver = IHS(
                compressor = comp,
                log = log,
                error = err,
                alpha = alpha
            )
            
            solver_rec = complete_solver(solver, x, A, b)

            # test types of the contents of the solver
            @test typeof(solver_rec) == IHSRecipe{
                Float64, 
                Main.IHSTest.ITestLogRecipe, 
                Main.IHSTest.ITestCompressorRecipe, 
                Main.IHSTest.ITestErrorRecipe, 
                Matrix{Float64}, 
                SubArray{
                    Float64, 
                    2, 
                    Matrix{Float64}, 
                    Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, 
                    false
                }, 
                Vector{Float64}, 
            }
            @test typeof(solver_rec.compressor) == ITestCompressorRecipe
            @test typeof(solver_rec.log) == ITestLogRecipe
            @test typeof(solver_rec.error) == ITestErrorRecipe
            @test typeof(solver_rec.alpha) == Float64
            @test typeof(solver_rec.compressed_mat) == Matrix{Float64}
            @test typeof(solver_rec.solution_vec) == Vector{Float64}
            @test typeof(solver_rec.buffer_vec) == Vector{Float64}
            @test typeof(solver_rec.gradient_vec) == Vector{Float64}
            @test typeof(solver_rec.residual_vec) == Vector{Float64}
            @test typeof(solver_rec.mat_view) <: SubArray
            @test typeof(solver_rec.R) <: UpperTriangular
            # Test sizes of vectors and matrices
            @test size(solver_rec.compressor) == (comp_dim, n_rows)
            @test size(solver_rec.compressed_mat) == (comp_dim, n_cols)
            @test size(solver_rec.buffer_vec) == (n_cols,)
            @test size(solver_rec.solution_vec) == (n_cols,)
            @test size(solver_rec.gradient_vec) == (n_cols,)
            @test size(solver_rec.residual_vec) == (n_rows,)
            
            # test values of entries
            solver_rec.alpha == alpha
            solver_rec.solution_vec == x
            solver_rec.buffer_vec == zeros(n_cols)
            solver_rec.gradient_vec == zeros(n_cols)
            solver_rec.residual_vec == zeros(n_rows)
        end

    end

    @testset "IHS: rsolve!" begin
        for type in [Float16, Float32, Float64, ComplexF32, ComplexF64]
            # test maxit stop
            let A = Array(qr(rand(type, n_rows, n_cols)).Q),
                xsol = ones(type, n_cols),
                b = A * xsol,
                # need to choose compression dim to be large enough
                comp_dim = 9 * size(A, 2),
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                x_st = deepcopy(x)
            
                comp = ITestCompressor(Left(), comp_dim)
                #check 10 iterations
                log = ITestLog(10, 0.0)
                err = ITestError()
                solver = IHS(
                    compressor = comp,
                    log = log,
                    error = err,
                    alpha = alpha
                )
            
                solver_rec = complete_solver(solver, x, A, b)
                
                result = rsolve!(solver_rec, x, A, b)
                #test that the residual decreases
                @test norm(A * x_st - b) > norm(A * x - b)
            end
    
            # using threshold stop 
            let A = Array(qr(rand(type, n_rows, n_cols)).Q),
                xsol = ones(type, n_cols),
                b = A * xsol,
                # need to choose compression dim to be large enough
                comp_dim = 9 * size(A, 2),
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                x_st = deepcopy(x)
            
                comp = ITestCompressor(Left(), comp_dim)
                #check 40 iterations we make a 1% improvement
                log = ITestLog(40, norm(b) * .99)
                err = ITestError()
                solver = IHS(
                    compressor = comp,
                    log = log,
                    error = err,
                    alpha = alpha
                )
            
                solver_rec = complete_solver(solver, x, A, b)
                
                result = rsolve!(solver_rec, x, A, b)
                #test that the error decreases
                @test norm(A * x_st - b) > norm(A * x - b)
                @test solver_rec.log.converged
                @test solver_rec.log.iteration < 40
            end

        end
    
    end

end

end
