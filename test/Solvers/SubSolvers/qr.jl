module QR_subsolver
include("../../test_helpers/preamble.jl")

@testset "QR SubSolver Tests" begin
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
        @test fieldnames(QRSolverRecipe) == (:A, :T, :work)
        @test fieldtypes(QRSolverRecipe) == (AbstractMatrix, AbstractMatrix, AbstractVector)
    end

    @testset "QR SubSolver: Complete SubSolver" begin
        # qr only works for floats and complex nums greater than 16 so we only test them
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = QRSolver(),
                A = rand(type, 12, 3),
                b = rand(type, 12),
                ss_recipe = complete_sub_solver(ss, A)

                # test the attributes of the outputs of the complete function
                @test typeof(ss_recipe) <: QRSolverRecipe{Matrix{type}}
                @test typeof(ss_recipe.T) == Matrix{type}
                @test all(isconcretetype, fieldtypes(typeof(ss_recipe)))
                @test ss_recipe.A == A
                @test size(ss_recipe.T) == (3, 3)
            end

        end

    end

    @testset "QR SubSolver: Update SubSolver" begin
        let A = rand(12, 3),
            B = rand(12, 3),
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

    # LAPACK's geqrt3! supports only the BLAS element types, so other types use the
    # generic fallback
    @testset "QR SubSolver: ldiv! for non-BLAS element types" begin
        for type in [BigFloat, Float16]
            A = rand(type, 10, 3)
            b = rand(type, 10)
            ss_recipe = complete_sub_solver(QRSolver(), A)
            @test all(isconcretetype, fieldtypes(typeof(ss_recipe)))
            x_sol = zeros(type, 3)
            ldiv!(x_sol, ss_recipe, b)
            @test isapprox(x_sol, A \ b; rtol=sqrt(eps(real(type))))
        end
    end

    @testset "QR SubSolver: allocations" begin
        A = rand(10, 3)
        b = rand(10)
        x = zeros(3)
        ss_recipe = complete_sub_solver(QRSolver(), A)
        update_sub_solver!(ss_recipe, A)
        ldiv!(x, ss_recipe, b)
        update_sub_solver!(ss_recipe, A)
        @test (@allocated update_sub_solver!(ss_recipe, A)) == 0
        # only the small views passed to the LAPACK wrappers allocate
        @test (@allocated ldiv!(x, ss_recipe, b)) <= 128
    end

end

end
