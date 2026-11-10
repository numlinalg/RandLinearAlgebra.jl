module LQ_subsolver
include("../../test_helpers/preamble.jl")

@testset "LQ SubSolver Tests" begin
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
        @test fieldnames(LQSolverRecipe) == (:A, :T, :work)
        @test fieldtypes(LQSolverRecipe) == (AbstractMatrix, AbstractMatrix, AbstractVector)
    end
    
    @testset "LQ SubSolver: Complete SubSolver" begin
        let ss = LQSolver()
            # lq only works for floats and complex nums greater than 16 so we only test them
            for type in [Float32, Float64, ComplexF32, ComplexF64]
                A = rand(type, 3, 10)
                b = rand(type, 3)
                ss_recipe = complete_sub_solver(ss, A)
                # test the attributes of the outputs of the complete function
                @test typeof(ss_recipe) <: LQSolverRecipe{Matrix{type}}
                @test typeof(ss_recipe.T) == Matrix{type}
                @test all(isconcretetype, fieldtypes(typeof(ss_recipe)))
                @test ss_recipe.A == Matrix(A')
                @test size(ss_recipe.T) == (3, 3)
            end

        end

    end

    @testset "LQ SubSolver: Update SubSolver" begin
        let A = rand(3, 10),
            B = rand(3, 10),
            ss = LQSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == Matrix(B')
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

    # LAPACK's geqrt3! supports only the BLAS element types, so other types use the
    # generic fallback
    @testset "LQ SubSolver: ldiv! for non-BLAS element types" begin
        for type in [BigFloat, Float16]
            A = rand(type, 3, 10)
            b = rand(type, 3)
            ss_recipe = complete_sub_solver(LQSolver(), A)
            @test all(isconcretetype, fieldtypes(typeof(ss_recipe)))
            x_sol = zeros(type, 10)
            ldiv!(x_sol, ss_recipe, b)
            @test isapprox(x_sol, A \ b; rtol=sqrt(eps(real(type))))
        end
    end

    @testset "LQ SubSolver: allocations" begin
        A = rand(3, 10)
        b = rand(3)
        x = zeros(10)
        ss_recipe = complete_sub_solver(LQSolver(), A)
        update_sub_solver!(ss_recipe, A)
        ldiv!(x, ss_recipe, b)
        update_sub_solver!(ss_recipe, A)
        @test (@allocated update_sub_solver!(ss_recipe, A)) == 0
        # only the small views passed to the LAPACK wrappers allocate
        @test (@allocated ldiv!(x, ss_recipe, b)) <= 128
    end

end

end
