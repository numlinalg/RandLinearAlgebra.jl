module SketchAndSolveTest
include("../test_helpers/preamble.jl")

@testset "SketchAndSolve" begin
    n_rows = 20
    n_cols = 4

    @testset "SketchAndSolve Technique" begin
        @test supertype(SketchAndSolve) == Solver

        # test fieldnames and types
        @test fieldnames(SketchAndSolve) == (:compressor, :error, :sub_solver)
        @test fieldtypes(SketchAndSolve) == (Compressor, SolverError, SubSolver)

        # test default constructor
        let solver = SketchAndSolve()
            @test typeof(solver.compressor) == SparseSign
            @test typeof(solver.compressor.cardinality) == Left
            @test typeof(solver.error) == FullResidual
            @test typeof(solver.sub_solver) == QRSolver
        end

        # test keyword constructor
        let solver = SketchAndSolve(
                compressor = Gaussian(cardinality = Left(), compression_dim = 10),
                error = FullResidual(),
                sub_solver = LQSolver()
            )
            @test typeof(solver.compressor) == Gaussian
            @test typeof(solver.sub_solver) == LQSolver
        end

        # test the Right-cardinality warning
        @test_logs (:warn,
            "Compressor has cardinality `Right` but sketch-and-solve\
            compresses  from the  `Left`."
        ) SketchAndSolve(
            compressor = Gaussian(cardinality = Right(), compression_dim = 10)
        )
    end

    @testset "SketchAndSolve: Complete Solver" begin
        # A completed SketchAndSolve is a KaczmarzRecipe configured for a single update
        let A = rand(n_rows, n_cols),
            b = rand(n_rows),
            x = zeros(n_cols),
            comp = Gaussian(cardinality = Left(), compression_dim = 10)

            solver = SketchAndSolve(compressor = comp, sub_solver = QRSolver())
            solver_rec = complete_solver(solver, x, A, b)

            @test solver_rec isa KaczmarzRecipe
            @test solver_rec.sub_solver isa QRSolverRecipe
            # max_it = 2 so the Kaczmarz loop performs exactly one update
            @test solver_rec.log.max_it == 2
            @test size(solver_rec.compressed_mat) == (10, n_cols)
        end
    end

    @testset "SketchAndSolve: rsolve!" begin
        # A consistent, overdetermined system solved with a full-rank tall sketch should
        # recover the solution exactly in a single sketch-and-solve step.
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let A = rand(type, n_rows, n_cols),
                xsol = ones(type, n_cols),
                b = A * xsol,
                comp = Gaussian(cardinality = Left(), compression_dim = 10),
                x = zeros(type, n_cols)

                solver = SketchAndSolve(compressor = comp, sub_solver = QRSolver())
                solver_rec = complete_solver(solver, x, A, b)
                rsolve!(solver_rec, x, A, b)

                @test x ≈ xsol
            end
        end
    end
end

end
