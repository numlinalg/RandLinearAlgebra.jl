@testset "LQ SubSolver Tests" begin
    using Test, RandLinearAlgebra, Random, LinearAlgebra
    using ..FieldTest
    using ..ApproxTol
    
    @testset "LQ SubSolver" begin
        @test supertype(LQSolver) == SubSolver
        @test fieldnames(LQSolver) == ()
        @test fieldtypes(LQSolver) == ()
        # Checks that the compressor produces a SubSolver
        ss = LQSolver()
        @test typeof(ss) <: SubSolver
    end

    @testset "LQ SubSolver Recipe" begin
        @test supertype(LQSolverRecipe) == SubSolverRecipe
        @test fieldnames(LQSolverRecipe) == (:A, )
        @test fieldtypes(LQSolverRecipe) == (AbstractArray,)
    end
    
    @testset "LQ SubSolver: Complete SubSolver" begin
        let ss = LQSolver()
            # lq only works for floats and complex nums greater than 16 so we only test them 
            for type in [Float32, Float64, ComplexF32, ComplexF64]
                A = rand(type, 3, 10)
                b = rand(type, 3)
                ss_recipe = complete_sub_solver(ss, A)
                # test the attributes of the outputs of the complete function
                @test typeof(ss_recipe) == LQSolverRecipe{Matrix{type}}
                @test ss_recipe.A == A
            end

        end

    end

    @testset "LQ SubSolver: Update SubSolver" begin
        let A = rand(3, 10),
            B = rand(10, 3),
            ss = LQSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
        end

    end

    @testset "LQ SubSolver: ldiv!" begin
        # begin by testing the matrix case
        let ss = LQSolver()
            # lq only works for floats and complex nums greater than 16 so we only test them 
            for type in [Float32, Float64, ComplexF32, ComplexF64]
                A = rand(type, 3, 10)
                b = rand(type, 3)
                ss_recipe = complete_sub_solver(ss, A)
                x_sol = rand(type, 10)
                x_true = lq(A) \ b
                ldiv!(x_sol, ss_recipe, b)
                @test x_sol ≈ x_true
            end

        end

    end

end
