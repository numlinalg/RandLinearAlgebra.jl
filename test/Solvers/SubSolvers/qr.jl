@testset "QR SubSolver Tests" begin
    using Test, RandLinearAlgebra, Random, LinearAlgebra
    using ..FieldTest
    using ..ApproxTol

    @testset "QR SubSolver" begin
        @test supertype(QRSolver) == SubSolver
        @test fieldnames(QRSolver) == ()
        @test fieldtypes(QRSolver) == ()
        # Checks that the QRSolver produces a SubSolver
        ss = QRSolver()
        @test typeof(ss) <: SubSolver
    end

    @testset "QR SubSolver Recipe" begin
        @test supertype(QRSolverRecipe) == SubSolverRecipe
        @test fieldnames(QRSolverRecipe) == (:A, )
        @test fieldtypes(QRSolverRecipe) == (AbstractArray,)
    end

    @testset "QR SubSolver: Complete SubSolver" begin
        # qr only works for floats and complex nums greater than 16 so we only test them 
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = QRSolver(),
                A = rand(type, 12, 3),
                b = rand(type, 12),
                ss_recipe = complete_sub_solver(ss, A)

                # test the attributes of the outputs of the complete function
                @test typeof(ss_recipe) == QRSolverRecipe{Matrix{type}}
                @test ss_recipe.A == A
            end

        end

    end

    @testset "QR SubSolver: Update SubSolver" begin
        let A = rand(12, 3),
            B = rand(10, 3),
            ss = QRSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
        end

    end

    @testset "QR SubSolver: ldiv!" begin
        # qr only works for floats and complex nums greater than 16 so we only test them  ????
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = QRSolver(),
                A = rand(type, 12, 3),
                b = rand(type, 12),
                ss_recipe = complete_sub_solver(ss, A),
                x_sol = rand(type, 3)

                x_true = qr(A) \ b
                ldiv!(x_sol, ss_recipe, b)
                @test x_sol ≈ x_true
            end

        end

    end

end