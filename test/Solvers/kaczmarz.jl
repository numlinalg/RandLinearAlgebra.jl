module KaczmarzTest
using Test, RandLinearAlgebra, LinearAlgebra
import RandLinearAlgebra: complete_compressor, update_compressor!
import LinearAlgebra: mul!, norm
import Random: randn!
import SparseArrays: sprand, SparseMatrixCSC, SparseVector, spzeros
using ..FieldTest
using ..ApproxTol

# Define the test structures
##########################
# Compressors
##########################
mutable struct KTestCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
end

KTestCompressor() = KTestCompressor(Left(), 5)

mutable struct KTestCompressorRecipe <: CompressorRecipe 
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

function RandLinearAlgebra.complete_compressor(
    comp::KTestCompressor, 
    A::AbstractMatrix,
    b::AbstractVector 
)
    n_rows = comp.compression_dim
    n_cols = size(A, 1)
    # Make a gaussian compressor
    op = randn(n_rows, n_cols) ./ sqrt(n_cols)
    return KTestCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

function RandLinearAlgebra.update_compressor!(
    comp::KTestCompressorRecipe,
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
    S::Main.KaczmarzTest.KTestCompressorRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, S.op, A, alpha, beta)
end

##########################
# Error Method
##########################
mutable struct KTestError <: RandLinearAlgebra.SolverError
    g::Real
end

mutable struct KTestErrorRecipe <: RandLinearAlgebra.SolverErrorRecipe
    residual::Vector{Number}
end

KTestError() = KTestError(1.0) 

function RandLinearAlgebra.complete_error(
    error::KTestError, 
    solver::Kaczmarz,
    A::AbstractMatrix, 
    b::AbstractVector
)
    return KTestErrorRecipe(zeros(typeof(error.g), size(A, 1)))
end

function RandLinearAlgebra.compute_error(error::KTestErrorRecipe, solver, A, b)
    error.residual = A * solver.solution_vec - b
    return norm(error.residual)
end

##############################
# Residual-less Error Recipe
##############################
mutable struct KTestErrorNoRes <: RandLinearAlgebra.SolverError end

mutable struct KTestErrorRecipeNoRes <: RandLinearAlgebra.SolverErrorRecipe end

function RandLinearAlgebra.complete_error(
    error::KTestErrorNoRes, 
    solver::Kaczmarz,
    A::AbstractMatrix, 
    b::AbstractVector 
)
    return KTestErrorRecipeNoRes()
end

############################
# Loggers
############################
mutable struct KTestLog <: Logger
    max_it::Int64
    g::Number
end

KTestLog() = KTestLog(5, 1.0) 

KTestLog(max_it) = KTestLog(max_it, 1.0)

mutable struct KTestLogRecipe <: LoggerRecipe
    max_it::Int64
    hist::Vector{Real}
    thresh::Float64
    converged::Bool
end

function RandLinearAlgebra.complete_logger(logger::KTestLog)
    return KTestLogRecipe(
        logger.max_it, 
        zeros(typeof(logger.g), logger.max_it), 
        logger.g, 
        false
    )
end

function RandLinearAlgebra.update_logger!(logger::KTestLogRecipe, err::Real, i::Int64)
    logger.hist[i] = err
    logger.converged = err < logger.thresh ? true : false
end

function RandLinearAlgebra.reset_logger!(logger::KTestLogRecipe)
    fill!(logger.hist, 0.0)
end

##############################
# Converged-less Logger
##############################
mutable struct KTestLogNoCov <: Logger end

mutable struct KTestLogRecipeNoCov <: LoggerRecipe end

function RandLinearAlgebra.complete_logger(logger::KTestLogNoCov)
    return KTestLogRecipeNoCov()
end

##############################
# SubSolver
##############################
mutable struct KTestSubSolver <: SubSolver end

mutable struct KTestSubSolverRecipe <: SubSolverRecipe 
    A::AbstractMatrix
end

function RandLinearAlgebra.complete_sub_solver(
    solver::KTestSubSolver, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    return KTestSubSolverRecipe(A)
end

function RandLinearAlgebra.update_sub_solver!(solver::KTestSubSolverRecipe, A::AbstractMatrix)
    solver.A = A
end

function RandLinearAlgebra.ldiv!(
    x::AbstractVector, 
    S::Main.KaczmarzTest.KTestSubSolverRecipe, 
    b::AbstractVector, 
)
    ldiv!(x, qr(S.A')', b)
end

#####################################
# Testing the functions
#####################################
@testset "Kaczmarz" begin
    n_rows = 10
    n_cols = 3
    A = rand(n_rows, n_cols)
    xsol = rand(n_cols)
    b = A * xsol

    @testset "Kaczmarz Technique" begin
        @test supertype(Kaczmarz) == Solver

        # test fieldnames and types
        @test fieldnames(Kaczmarz) == (:alpha, :compressor, :log, :error, :sub_solver)
        @test fieldtypes(Kaczmarz) == (Float64, Compressor, Logger, SolverError, SubSolver)

        # test default constructor

        let solver = Kaczmarz()
            @test solver.alpha == 1.0
            @test typeof(solver.compressor) == SparseSign 
            @test typeof(solver.compressor.cardinality) == Left 
            @test typeof(solver.log) == BasicLogger
            @test typeof(solver.error) == FullResidual
            @test typeof(solver.sub_solver) == LQSolver 
        end

        # test constructor
        let solver = Kaczmarz(
            alpha = 2.0,
            compressor = KTestCompressor(),
            log = KTestLog(),
            error = KTestError(),
            sub_solver = KTestSubSolver()
        )

            @test solver.alpha == 2.0
            @test typeof(solver.compressor) == KTestCompressor
            @test typeof(solver.log) == KTestLog
            @test typeof(solver.error) == KTestError
            @test typeof(solver.sub_solver) == KTestSubSolver
        end 
        
        # Test that error gets returned with right compressor
        @test_logs (:warn, 
               "Compressor has cardinality `Right` but kaczmarz\
               compresses  from the  `Left`." 
        ) Kaczmarz(
            alpha = 2.0,
            compressor = KTestCompressor(Right(), 5),
            log = KTestLog(),
            error = KTestError(),
            sub_solver = KTestSubSolver()
        )
    end

    @testset "KaczmarzRecipe" begin
        @test supertype(KaczmarzRecipe) == SolverRecipe

        # test fieldnames and types
        @test fieldnames(KaczmarzRecipe) == (
            :compressor, 
            :log, 
            :error, 
            :sub_solver, 
            :alpha, 
            :compressed_mat, 
            :compressed_vec, 
            :solution_vec, 
            :update_vec, 
            :mat_view, 
            :vec_view
        )
        @test fieldtypes(KaczmarzRecipe) == (
            CompressorRecipe,
            LoggerRecipe,
            SolverErrorRecipe,
            SubSolverRecipe,
            Float64, 
            AbstractArray, 
            AbstractVector, 
            AbstractVector{T} where T<:Number,
            AbstractVector{T} where T<:Number,
            SubArray, 
            SubArray,
        )  
    end

    @testset "Kaczmarz: Complete Solver" begin
        # test error method with no residual error 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = KTestCompressor(Left(), comp_dim)
            log = KTestLog()
            err = KTestErrorNoRes()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "ErrorRecipe $(typeof(KTestErrorRecipeNoRes())) does not contain the \
                field 'residual' and is not valid for a kaczmarz solver."
            ) complete_solver(solver, x, A, b)
        end

        # test logger method with no converged field 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = KTestCompressor(Left(), comp_dim)
            log = KTestLogNoCov()
            err = KTestError()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
            
            @test_throws ArgumentError(
                "LoggerRecipe $(typeof(KTestLogRecipeNoCov())) does not contain \
                the field 'converged' and is not valid for a kaczmarz solver."
            ) complete_solver(solver, x, A, b)
        end

        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = KTestCompressor(Left(), comp_dim)
            log = KTestLog()
            err = KTestError()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
            
            solver_rec = complete_solver(solver, x, A, b)

            # test types of the contents of the solver
            @test typeof(solver_rec) == KaczmarzRecipe{
                Float64, 
                Vector{Float64}, 
                Matrix{Float64}, 
                SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, 
                SubArray{
                    Float64, 
                    2, 
                    Matrix{Float64}, 
                    Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, 
                    false
                   }, 
                Main.KaczmarzTest.KTestCompressorRecipe, 
                Main.KaczmarzTest.KTestLogRecipe, 
                Main.KaczmarzTest.KTestErrorRecipe, 
                Main.KaczmarzTest.KTestSubSolverRecipe
            }
            @test typeof(solver_rec.compressor) == KTestCompressorRecipe
            @test typeof(solver_rec.log) == KTestLogRecipe
            @test typeof(solver_rec.error) == KTestErrorRecipe
            @test typeof(solver_rec.sub_solver) == KTestSubSolverRecipe
            @test typeof(solver_rec.alpha) == Float64
            @test typeof(solver_rec.compressed_mat) == Matrix{Float64}
            @test typeof(solver_rec.compressed_vec) == Vector{Float64}
            @test typeof(solver_rec.solution_vec) == Vector{Float64}
            @test typeof(solver_rec.update_vec) == Vector{Float64}
            @test typeof(solver_rec.mat_view) <: SubArray
            @test typeof(solver_rec.vec_view) <: SubArray

            # Test sizes of vectors and matrices
            @test size(solver_rec.compressor) == (comp_dim, n_rows)
            @test size(solver_rec.compressed_mat) == (comp_dim, n_cols)
            @test size(solver_rec.compressed_vec) == (comp_dim,)
            @test size(solver_rec.update_vec) == (n_cols,)
            
            # test values of entries
            solver_rec.alpha == alpha
            solver_rec.solution_vec == x
            solver_rec.update_vec == zeros(n_cols)
        end
 
        # test with a sparse matrix with sampling compressor
        let A = sprand(n_rows, n_cols, .9),
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = Sampling(cardinality = Left(), compression_dim = comp_dim)
            log = KTestLog()
            err = KTestError()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
            
            solver_rec = complete_solver(solver, x, A, b)

            # test types of the contents of the solver
            @test typeof(solver_rec) == KaczmarzRecipe{
                Float64, 
                Vector{Float64}, 
                SparseMatrixCSC{Float64, Int64},
                SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, 
                SubArray{
                    Float64, 
                    2, 
                    SparseMatrixCSC{Float64, Int64}, 
                    Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, 
                    false
                },
                SamplingRecipe{Left}, 
                Main.KaczmarzTest.KTestLogRecipe, 
                Main.KaczmarzTest.KTestErrorRecipe, 
                Main.KaczmarzTest.KTestSubSolverRecipe
            }
            @test typeof(solver_rec.compressor) == SamplingRecipe{Left} 
            @test typeof(solver_rec.log) == KTestLogRecipe
            @test typeof(solver_rec.error) == KTestErrorRecipe
            @test typeof(solver_rec.sub_solver) == KTestSubSolverRecipe
            @test typeof(solver_rec.alpha) == Float64
            @test typeof(solver_rec.compressed_mat) == SparseMatrixCSC{Float64, Int64} 
            @test typeof(solver_rec.compressed_vec) == Vector{Float64}
            @test typeof(solver_rec.solution_vec) == Vector{Float64}
            @test typeof(solver_rec.update_vec) == Vector{Float64}
            @test typeof(solver_rec.mat_view) <: SubArray
            @test typeof(solver_rec.vec_view) <: SubArray

            # Test sizes of vectors and matrices
            @test size(solver_rec.compressor) == (comp_dim, n_rows)
            @test size(solver_rec.compressed_mat) == (comp_dim, n_cols)
            @test size(solver_rec.compressed_vec) == (comp_dim,)
            @test size(solver_rec.update_vec) == (n_cols,)
            
            # test values of entries
            solver_rec.alpha == alpha
            solver_rec.solution_vec == x
            solver_rec.update_vec == zeros(n_cols)
        end

        # Run with sparse vector and sampling matrix
        let A = A,
            xsol = xsol,
            b = sprand(n_rows, .3),
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)
            
            comp = Sampling(cardinality = Left(), compression_dim = comp_dim)
            log = KTestLog()
            err = KTestError()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
            
            solver_rec = complete_solver(solver, x, A, b)

            # test types of the contents of the solver
            @test typeof(solver_rec) == KaczmarzRecipe{
                Float64, 
                SparseVector{Float64, Int64}, 
                Matrix{Float64}, 
                SubArray{
                    Float64, 
                    1, 
                    SparseVector{Float64, Int64}, 
                    Tuple{UnitRange{Int64}}, 
                    false
                }, 
                SubArray{
                    Float64, 
                    2, 
                    Matrix{Float64}, 
                    Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, 
                    false
                }, 
                SamplingRecipe{Left}, 
                Main.KaczmarzTest.KTestLogRecipe, 
                Main.KaczmarzTest.KTestErrorRecipe, 
                Main.KaczmarzTest.KTestSubSolverRecipe
            }
            @test typeof(solver_rec.compressor) == SamplingRecipe{Left} 
            @test typeof(solver_rec.log) == KTestLogRecipe
            @test typeof(solver_rec.error) == KTestErrorRecipe
            @test typeof(solver_rec.sub_solver) == KTestSubSolverRecipe
            @test typeof(solver_rec.alpha) == Float64
            @test typeof(solver_rec.compressed_mat) == Matrix{Float64}
            @test typeof(solver_rec.compressed_vec) == SparseVector{Float64, Int64}
            @test typeof(solver_rec.solution_vec) == Vector{Float64}
            @test typeof(solver_rec.update_vec) == Vector{Float64}
            @test typeof(solver_rec.mat_view) <: SubArray
            @test typeof(solver_rec.vec_view) <: SubArray

            # Test sizes of vectors and matrices
            @test size(solver_rec.compressor) == (comp_dim, n_rows)
            @test size(solver_rec.compressed_mat) == (comp_dim, n_cols)
            @test size(solver_rec.compressed_vec) == (comp_dim,)
            @test size(solver_rec.update_vec) == (n_cols,)
            
            # test values of entries
            solver_rec.alpha == alpha
            solver_rec.solution_vec == x
            solver_rec.update_vec == zeros(n_cols)
        end


    end

    @testset "Kaczmarz: Kaczmarz Update" begin
        # Begin with a test of an update when the block size is 1
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let A = rand(type, n_rows, n_cols),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 1,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                
                comp = KTestCompressor(Left(), comp_dim)
                log = KTestLog()
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
                
                solver_rec = complete_solver(solver, x, A, b)
                
                # Sketch the matrix and vector
                sb = solver_rec.compressor * b
                sA = solver_rec.compressor * A 
                solver_rec.vec_view = view(sb, 1:1)
                solver_rec.mat_view = view(sA, 1:comp_dim, :)
                solver_rec.solution_vec = deepcopy(x) 
                
                # compute comparison update
                sc = (dot(conj(sA), x) - sb[1]) / dot(sA, sA) * alpha
                test_sol = x - sc * adjoint(sA)
    
                # compute the update
                RandLinearAlgebra.kaczmarz_update!(solver_rec)
                @test solver_rec.solution_vec ≈ test_sol
            end
            
            # Test when we have a sprase matrix and sampling compressor
            let A = sprand(type, n_rows, n_cols, .9),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 1,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                
                comp = Sampling(cardinality = Left(), compression_dim = comp_dim)
                log = KTestLog()
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
                
                solver_rec = complete_solver(solver, x, A, b)
                
                # Sketch the matrix and vector
                sb = solver_rec.compressor * b
                sA = solver_rec.compressor * A 
                solver_rec.vec_view = view(sb, 1:comp_dim)
                solver_rec.mat_view = view(sA, 1:comp_dim, :)
                solver_rec.solution_vec = deepcopy(x) 
    
                # compute comparison update
                sc = (dot(conj(sA), x) - sb[1]) / dot(sA, sA) * alpha
                test_sol = x - sc * adjoint(sA)
    
                # compute the update
                RandLinearAlgebra.kaczmarz_update!(solver_rec)
                @test solver_rec.solution_vec ≈ test_sol
            end
        
        end

    end

    @testset "Kaczmarz: Block Kaczmarz Update" begin
        # Begin with a test of an update when the block size is 2
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let A = rand(type, n_rows, n_cols),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 2,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                
                comp = KTestCompressor(Left(), comp_dim)
                log = KTestLog()
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
                
                solver_rec = complete_solver(solver, x, A, b)
                
                # Sketch the matrix and vector
                sb = solver_rec.compressor * b
                sA = solver_rec.compressor * A 
                solver_rec.vec_view = view(sb, 1:comp_dim)
                solver_rec.mat_view = view(sA, 1:comp_dim, :)
                solver_rec.solution_vec = deepcopy(x) 
    
                # compute comparison update
                test_sol =  x + lq(Array(sA)) \ (sb - sA * x)
    
                # compute the update
                RandLinearAlgebra.kaczmarz_update_block!(solver_rec)
                @test solver_rec.solution_vec ≈ test_sol
            end
            
            # test that this works with a sparse matrix and sampling compressor
            # let A = sprand(type, n_rows, n_cols, .9),
            #     xsol = ones(type, n_cols),
            #     b = A * xsol,
            #     comp_dim = 2,
            #     alpha = 1.0,
            #     n_rows = size(A, 1),
            #     n_cols = size(A, 2),
            #     x = zeros(type, n_cols)
                
            #     comp = Sampling(cardinality = Left(), compression_dim = comp_dim)
            #     log = KTestLog()
            #     err = KTestError()
            #     sub_solver = KTestSubSolver()
            #     solver = Kaczmarz(
            #         compressor = comp,
            #         log = log,
            #         error = err,
            #         sub_solver = sub_solver,
            #         alpha = alpha
            #     )
                
            #     solver_rec = complete_solver(solver, x, A, b)
                
            #     # Sketch the matrix and vector
            #     sb = solver_rec.compressor * b
            #     sA = solver_rec.compressor * A 
            #     solver_rec.vec_view = view(sb, 1:comp_dim)
            #     solver_rec.mat_view = view(sA, 1:comp_dim, :)
            #     solver_rec.solution_vec = deepcopy(x) 
    
            #     # compute comparison update
            #     test_sol =  x + lq(Array(sA)) \ (sb - sA * x)
    
            #     # compute the update
            #     RandLinearAlgebra.kaczmarz_update_block!(solver_rec)
            #     @test solver_rec.solution_vec ≈ test_sol
            # end
    
        end

    end

    @testset "Kaczmarz: rsolve!" begin
        # check the dimension errors
        # x wrong dimension
        let type = Float64,
            A = rand(type, n_rows, n_cols),
            xsol = ones(type, n_cols),
            b = A * xsol,
            comp_dim = 1,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(type, n_cols + 1)
            x_st = deepcopy(x)
        
            comp = KTestCompressor(Left(), comp_dim)
            # check after 10 iterations
            log = KTestLog(10, 0.0)
            err = KTestError()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
        
            solver_rec = complete_solver(solver, x, A, b)
            
            @test_throws DimensionMismatch rsolve!(solver_rec, x, A, b)
        end

        # b wrong dimension
        let type = Float64,
            A = rand(type, n_rows, n_cols),
            xsol = ones(type, n_cols),
            b = ones(n_rows + 1),
            comp_dim = 1,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(type, n_cols)
            x_st = deepcopy(x)
        
            comp = KTestCompressor(Left(), comp_dim)
            # check after 10 iterations
            log = KTestLog(10, 0.0)
            err = KTestError()
            sub_solver = KTestSubSolver()
            solver = Kaczmarz(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )
        
            solver_rec = complete_solver(solver, x, A, b)
            
            @test_throws DimensionMismatch rsolve!(solver_rec, x, A, b)
        end

        # test when the block size is one maxit stop
        for type in [Float16, Float32, Float64, ComplexF32, ComplexF64]
            let A = rand(type, n_rows, n_cols),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 1,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                x_st = deepcopy(x)
            
                comp = KTestCompressor(Left(), comp_dim)
                # check after 10 iterations
                log = KTestLog(10, 0.0)
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
            
                solver_rec = complete_solver(solver, x, A, b)
                
                result = rsolve!(solver_rec, x, A, b)
                # test that the error decreases
                @test norm(x_st - xsol) > norm(x - xsol)
            end
    
            # test when the block size is greater than 1 maxit stop
            let A = rand(type, n_rows, n_cols),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 2,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                x_st = deepcopy(x)
            
                comp = KTestCompressor(Left(), comp_dim)
                #check 10 iterations
                log = KTestLog(10, 0.0)
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
            
                solver_rec = complete_solver(solver, x, A, b)
                
                result = rsolve!(solver_rec, x, A, b)
                #test that the error decreases
                @test norm(x_st - xsol) > norm(x - xsol)
            end
    
            # test when the block size is one threshold stop 
            # orthogonalize Q to control the residual
            let A = Array(qr(rand(type, n_rows, n_cols)).Q),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 1,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                x_st = deepcopy(x)
            
                comp = KTestCompressor(Left(), comp_dim)
                # check after 20 iterations
                log = KTestLog(20, 0.5)
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
            
                solver_rec = complete_solver(solver, x, A, b)
                
                result = rsolve!(solver_rec, x, A, b)
                # test that the error decreases
                @test norm(x_st - xsol) > norm(x - xsol)
                # test that the solver actually converged
                @test solver_rec.log.converged
            end
    
            # test when the block size is greter than 1 using threshold stop 
            # Orthogonalize Q to control the residual
            let A = Array(qr(rand(type, n_rows, n_cols)).Q),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp_dim = 2,
                alpha = 1.0,
                n_rows = size(A, 1),
                n_cols = size(A, 2),
                x = zeros(type, n_cols)
                x_st = deepcopy(x)
            
                comp = KTestCompressor(Left(), comp_dim)
                #check 20 iterations
                log = KTestLog(20, 0.5)
                err = KTestError()
                sub_solver = KTestSubSolver()
                solver = Kaczmarz(
                    compressor = comp,
                    log = log,
                    error = err,
                    sub_solver = sub_solver,
                    alpha = alpha
                )
            
                solver_rec = complete_solver(solver, x, A, b)
                
                result = rsolve!(solver_rec, x, A, b)
                #test that the error decreases
                @test norm(x_st - xsol) > norm(x - xsol)
                # test that the solver actually converged
                @test solver_rec.log.converged
            end

        end
    
    end

end

end
