##########################
# ColumnProjection Constructor 
##########################
module column_projection_constructor 

using Test, RandLinearAlgebra

@testset "ColumnProjection Structure & Constructor" begin

    # Super type 
    @test supertype(ColumnProjection) == Solver

    # Test fieldnames and types
    @test fieldnames(ColumnProjection) == (:alpha, :compressor, :error, :log, :sub_solver)
    @test fieldtypes(ColumnProjection) == (Float64, Compressor, SolverError, Logger, SubSolver)

    # Test default constructor
    let solver = ColumnProjection()
        @test solver.alpha == 1.0
        @test typeof(solver.compressor) == SparseSign 
        @test typeof(solver.compressor.cardinality) == Right 
        @test typeof(solver.log) == BasicLogger
        @test typeof(solver.error) == LSGradient
        @test typeof(solver.sub_solver) == QRSolver 
    end

    ##########################
    # Define Test Structures 
    ##########################

    struct TestCompressor <: Compressor
        cardinality::Cardinality
    end
    struct TestError <: SolverError end
    struct TestLogger <: Logger end 
    struct TestSubSolver <: SubSolver end 
    mutable struct ColTestCompressor <: Compressor
        cardinality::Cardinality
        compression_dim::Int64
    end

    # Test Custom: Right Compressor 
    let alpha = 2.0,
        compressor = TestCompressor(Right()),
        error = TestError(),
        logger = TestLogger(),
        subsolver = TestSubSolver()

        solver = ColumnProjection(
            alpha = alpha,
            compressor = compressor,
            error = error,
            log = logger,
            sub_solver = subsolver
        )

        @test solver.alpha == alpha
        @test solver.compressor == compressor
        @test solver.log == logger
        @test solver.error == error
        @test solver.sub_solver == subsolver 
    end 

    # Test Custom: Left Compressor 
    let alpha = 2.0,
        compressor = TestCompressor(Left()),
        error = TestError(),
        logger = TestLogger(),
        subsolver = TestSubSolver()

        solver = @test_logs (
            :warn, 
            "Compressor has cardinality `Left` but ColumnProjection\
            compresses from the `Right`. This may cause an inefficiency."
        ) ColumnProjection(
            alpha = alpha,
            compressor = compressor,
            error = error,
            log = logger,
            sub_solver = subsolver
        )

        @test solver.alpha == alpha
        @test solver.compressor == compressor
        @test solver.log == logger
        @test solver.error == error
        @test solver.sub_solver == subsolver 
    end
end

end

####################################
# ColumnProjectionRecipe: Structure 
####################################
module column_projection_recipe_structure 

using Test, RandLinearAlgebra

@testset "ColumnProjectionRecipe Structure" begin

    # Super Type 
    @test supertype(ColumnProjectionRecipe) == SolverRecipe 

    # Test Field Names 
    @test fieldnames(ColumnProjectionRecipe) == (
        :compressor, :log, :error, :sub_solver, :alpha, :compressed_mat, 
        :solution_vec, :update_vec, :mat_view, :residual_vec
    )

    # Test Field Types 
    @test fieldtypes(ColumnProjectionRecipe) == (
        CompressorRecipe, LoggerRecipe, SolverErrorRecipe, SubSolverRecipe, 
        Float64, AbstractArray, AbstractVector, AbstractVector, SubArray,
        AbstractVector
    )
end 

end

###################################
# ColumnProjectionRecipe: Complete
###################################
module column_projection_recipe_constructor

using Test, RandLinearAlgebra

@testset "ColumnProjectionRecipe complete_solver" begin
    
    # SolverErrorRecipe lacks :gradient field 
    let A = randn(5, 10),
        b = randn(5),
        x = randn(10)

        struct TestSolverError <: SolverError end 
        struct TestSolverErrorRecipe <: SolverErrorRecipe end 
        RandLinearAlgebra.complete_error(
            SE::TestSolverError, 
            S::ColumnProjection, 
            A::Matrix, 
            b::Vector
        ) = TestSolverErrorRecipe()

        solver = ColumnProjection(error=TestSolverError()) # End of test parameters 

        @test_throws ArgumentError complete_solver(solver, x, A, b)
        @test_throws "gradient" complete_solver(solver, x, A, b)
    end

    # Logger lacks :max_it field 
    let A = randn(5, 10),
        b = randn(5),
        x = randn(10)

        struct TestLoggerI <: Logger end 
        struct TestLoggerRecipeI <: LoggerRecipe end 
        RandLinearAlgebra.complete_logger(TL::TestLoggerI) = TestLoggerRecipeI()

        solver = ColumnProjection(log=TestLoggerI()) # End of test parameters 

        @test_throws ArgumentError complete_solver(solver, x, A, b)
        @test_throws "max_it" complete_solver(solver, x, A, b)
    end

    # Logger lacks :converged field
    let A = randn(5, 10),
        b = randn(5),
        x = randn(10)

        struct TestLoggerII <: Logger end 
        struct TestLoggerRecipeII <: LoggerRecipe
            max_it::Int64
        end 
        RandLinearAlgebra.complete_logger(TL::TestLoggerII) = TestLoggerRecipeII(100)

        solver = ColumnProjection(log=TestLoggerII()) # End of test parameters

        @test_throws ArgumentError complete_solver(solver, x, A, b)
        @test_throws "converged" complete_solver(solver, x, A, b)

    end

    # Correct Execution 
    let m = 5,
        n = 10,
        A = randn(m, n),
        b = randn(m),
        x = randn(n),
        solver = ColumnProjection() # Default constructor

        CPR = complete_solver(solver, x, A, b)

        @test typeof(CPR) <: ColumnProjectionRecipe
        @test typeof(CPR.compressor) <: SparseSignRecipe
        @test typeof(CPR.log) <: BasicLoggerRecipe
        @test typeof(CPR.error) <: LSGradientRecipe
        @test typeof(CPR.sub_solver) <: QRSolverRecipe
        @test CPR.alpha == 1.0
        @test typeof(CPR.compressed_mat) == Matrix{Float64}
        @test size(CPR.compressed_mat) == (m, 2)
        @test typeof(CPR.solution_vec) == Vector{Float64}
        @test length(CPR.solution_vec) == n
        @test typeof(CPR.update_vec) == Vector{Float64}
        @test length(CPR.update_vec) == 2
        @test typeof(CPR.mat_view) <: SubArray
        @test size(CPR.mat_view) == (m, 2)
        @test typeof(CPR.residual_vec) == Vector{Float64}
        @test length(CPR.residual_vec) == m

    end

end

end

###################################
# Column Project Update 
###################################
module column_projection_update

using Test, RandLinearAlgebra, LinearAlgebra

@testset "ColumnProjection Update" begin

    # 1 Dimensional Case 
    let compression_dim = 1,
        A = randn(10, 5),
        b = randn(10),
        x = zeros(Float64, 5),
        solver = RandLinearAlgebra.complete_solver(
            ColumnProjection(
                compressor= Gaussian(cardinality=Right(), compression_dim=compression_dim)
            ), x, A, b
        )

        solver.residual_vec = b - A * x
        RandLinearAlgebra.mul!(solver.mat_view, A, solver.compressor)
        RandLinearAlgebra.colproj_update!(solver)

        tilde_a = A * solver.compressor 
        @test x ≈ solver.compressor * [dot(tilde_a, b) / norm(tilde_a)^2]

    end

    # Multi-dimensional case
    let compression_dim = 3,
        A = randn(10, 5),
        b = randn(10),
        x = zeros(Float64, 5),
        solver = RandLinearAlgebra.complete_solver(
            ColumnProjection(
                compressor= Gaussian(cardinality=Right(), compression_dim=compression_dim)
            ), x, A, b
        )

        solver.residual_vec = b - A * x # Should be just b
        RandLinearAlgebra.mul!(solver.mat_view, A, solver.compressor)
        RandLinearAlgebra.colproj_update_block!(solver)

        tilde_a = A * solver.compressor 
        @test x ≈ solver.compressor * (tilde_a \ b)

    end

end

end

###################################
# Column Project Solver  
###################################
module column_projection_solver 

using Test, RandLinearAlgebra, LinearAlgebra

@testset "ColumnProjection Solver: rsolve!" begin 

    # 1-Dim Compression: Base Case
    let compressor = Gaussian(cardinality=Right(), compression_dim=1),
        log=BasicLogger(max_it=1),
        ingredients = ColumnProjection(compressor=compressor,log=log),
        A = randn(5, 10),
        b = randn(5),
        x = zeros(10),
        solver = RandLinearAlgebra.complete_solver(ingredients, x, A, b)

        rsolve!(solver, x, A, b)

        @test norm(b - A*x) < norm(b) # Is the residual reduced?
    end

    # 1-Dim Compression: Induction and Conclusion
    let compressor = Gaussian(cardinality=Right(), compression_dim=1),
        log=BasicLogger(max_it=11),
        ingredients = ColumnProjection(compressor=compressor,log=log),
        A = randn(5, 10),
        b = randn(5),
        x = zeros(10),
        solver = RandLinearAlgebra.complete_solver(ingredients, x, A, b)

        # Induction 
        rsolve!(solver, x, A, b)

        # Conclusion 
        x_previous = deepcopy(x)
        ingredients = ColumnProjection(compressor=compressor, log=BasicLogger(max_it=1))
        solver = RandLinearAlgebra.complete_solver(ingredients, x, A, b)
        rsolve!(solver, x, A, b)

        @test norm(b - A*x) < norm(b - A*x_previous) # Is the residual reduced?
    end
    
    # Multidimensional Compression: Base Case 
    let ingredients = ColumnProjection(log=BasicLogger(max_it=1)),
        A = randn(5, 10),
        b = randn(5),
        x = zeros(10),
        solver = RandLinearAlgebra.complete_solver(ingredients, x, A, b)

        rsolve!(solver, x, A, b)

        @test norm(b - A*x) < norm(b) # Is the residual reduced? 
    end

    # Multidimensional Compression: Induction and Conclusion 
    let ingredients = ColumnProjection(log=BasicLogger(max_it=11)),
        A = randn(5, 10),
        b = randn(5),
        x = zeros(10),
        solver = RandLinearAlgebra.complete_solver(ingredients, x, A, b)

        # Induction 
        rsolve!(solver, x, A, b)

        # Conclusion
        x_previous = deepcopy(x)
        ingredients = ColumnProjection(log=BasicLogger(max_it=1))
        solver = RandLinearAlgebra.complete_solver(ingredients, x, A, b)
        rsolve!(solver, x, A, b)

        @test norm(b - A*x) < norm(b - A*x_previous) # Is the residual reduced?
    end

    # Exits via solver.log.converged == true
    let A = randn(5, 10),
        b = randn(5),
        x = zeros(10)

        # Parameter Data Structures 
        struct TestLoggerIII <: Logger end 
        mutable struct TestLoggerRecipeIII <: LoggerRecipe
            max_it::Int64 
            converged::Bool
        end
        RandLinearAlgebra.complete_logger(LG::TestLoggerIII) = TestLoggerRecipeIII(10, false)
        function RandLinearAlgebra.reset_logger!(LG::TestLoggerRecipeIII)
            LG.max_it = 10
            LG.converged = false
            return nothing 
        end
        function RandLinearAlgebra.update_logger!(LG::TestLoggerRecipeIII, err::Float64, i::Int64)
            if i > 1
                LG.converged = true
            end
            return nothing
        end

        ingredients = ColumnProjection(log=TestLoggerIII())
        solver = complete_solver(ingredients, x, A, b)

        rsolve!(solver, x, A, b)

        @test solver.log.converged

    end
end

end