@testset "LSQR SubSolver Tests" begin
    using Test, RandLinearAlgebra, Random, LinearAlgebra, SparseArrays
    using ..FieldTest
    using ..ApproxTol

    @testset "LSQR SubSolver" begin
        @test supertype(LSQRSolver) == KrylovSubSolver
        @test supertype(KrylovSubSolver) == SubSolver
        @test fieldnames(LSQRSolver) == (:max_it, :atol, :btol)
        @test fieldtypes(LSQRSolver) == (Int64, Float64, Float64)

        # Test the default constructor
        let ss = LSQRSolver()
            @test typeof(ss) <: SubSolver
            @test ss.max_it == 0
            @test ss.atol == 1e-8
            @test ss.btol == 1e-8
        end

        # Test the keyword constructor
        let ss = LSQRSolver(; max_it = 5, atol = 1e-4, btol = 1e-6)
            @test ss.max_it == 5
            @test ss.atol == 1e-4
            @test ss.btol == 1e-6
        end

        # Test the argument errors
        @test_throws ArgumentError LSQRSolver(; max_it = -1)
        @test_throws ArgumentError LSQRSolver(; atol = -1.0)
        @test_throws ArgumentError LSQRSolver(; btol = -1.0)
    end

    @testset "LSQR SubSolver Recipe" begin
        @test supertype(LSQRSolverRecipe) == KrylovSubSolverRecipe
        @test supertype(KrylovSubSolverRecipe) == SubSolverRecipe
        @test fieldnames(LSQRSolverRecipe) ==
            (:A, :max_it, :atol, :btol, :u, :v, :w)
        @test fieldtypes(LSQRSolverRecipe) == (
            AbstractArray,
            Int64,
            Float64,
            Float64,
            AbstractVector,
            AbstractVector,
            AbstractVector
        )
    end

    @testset "LSQR SubSolver: Complete SubSolver" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = LSQRSolver(; max_it = 7, atol = 1e-4, btol = 1e-6),
                A = rand(type, 12, 3),
                ss_recipe = complete_sub_solver(ss, A)

                # test the attributes of the outputs of the complete function
                @test typeof(ss_recipe) ==
                    LSQRSolverRecipe{type, Vector{type}, Matrix{type}}
                @test ss_recipe.A == A
                @test ss_recipe.max_it == 7
                @test ss_recipe.atol == 1e-4
                @test ss_recipe.btol == 1e-6
                @test size(ss_recipe.u) == (12,)
                @test size(ss_recipe.v) == (3,)
                @test size(ss_recipe.w) == (3,)
            end

        end

    end

    @testset "LSQR SubSolver: Update SubSolver" begin
        # Same dimensions
        let A = rand(12, 3),
            B = rand(12, 3),
            ss = LSQRSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
            @test size(ss_recipe.u) == (12,)
            @test size(ss_recipe.v) == (3,)
            @test size(ss_recipe.w) == (3,)
        end

        # Different dimensions should resize the buffers
        let A = rand(12, 3),
            B = rand(10, 4),
            ss = LSQRSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
            @test size(ss_recipe.u) == (10,)
            @test size(ss_recipe.v) == (4,)
            @test size(ss_recipe.w) == (4,)
        end

    end

    @testset "LSQR SubSolver: ldiv!" begin
        Random.seed!(2131)
        # Overdetermined least squares problem
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
                A = rand(type, 12, 3),
                b = rand(type, 12),
                ss_recipe = complete_sub_solver(ss, A),
                x_sol = zeros(type, 3)

                x_true = qr(A) \ b
                ldiv!(x_sol, ss_recipe, b)
                @test x_sol ≈ x_true
            end

        end

        # Underdetermined system should return the minimum norm solution
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
                A = rand(type, 3, 12),
                b = rand(type, 3),
                ss_recipe = complete_sub_solver(ss, A),
                x_sol = zeros(type, 12)

                x_true = A' * ((A * A') \ b)
                ldiv!(x_sol, ss_recipe, b)
                @test x_sol ≈ x_true
            end

        end

        # Sparse matrices
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
            A = sprand(30, 5, 0.5) + sparse(1.0I, 30, 5),
            b = rand(30),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(5)

            x_true = Matrix(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # Solving after updating the matrix to one with new dimensions
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
            A = rand(12, 3),
            B = rand(10, 4),
            b = rand(10),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(4)

            update_sub_solver!(ss_recipe, B)
            x_true = qr(B) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # A zero constant vector should return the zero solution
        let ss = LSQRSolver(),
            A = rand(12, 3),
            b = zeros(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(3)

            ldiv!(x_sol, ss_recipe, b)
            @test x_sol == zeros(3)
        end

        # Hitting the maximum iteration should produce an inexact solution
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 1),
            A = rand(12, 6),
            b = rand(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(6)

            x_true = qr(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test !(x_sol ≈ x_true)
        end

    end

end
