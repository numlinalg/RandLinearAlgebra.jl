module Identity_compressor
using Test, RandLinearAlgebra
import Base.*
import LinearAlgebra:mul!
import Random:seed!
using ..FieldTest
using ..ApproxTol

seed!(21321)
@testset "Identity" begin
    @testset "Identity: Compressor" begin
        # Verify supertype
        @test supertype(Identity) == Compressor

        # Verify Field names
        @test fieldnames(Identity) == (:cardinality,)
        @test fieldtypes(Identity) == (Cardinality,)

        # Verify that the constructor works
        @test typeof(Identity()) == Identity 
    end

    @testset "Identity: CompressorRecipe" begin
        @test_compressor IdentityRecipe

        @test fieldnames(IdentityRecipe) == (:cardinality, :n_rows, :n_cols)
        @test fieldtypes(IdentityRecipe) == (Cardinality, Int64, Int64)
    end

    @testset "Identity: Complete Compressor" begin
        # test with left cardinality
        let n_rows = 10,
            n_cols = 3,
            A = ones(n_rows, n_cols)

            compressor_recipe = complete_compressor(Identity(), A)

            # test values and types
            @test compressor_recipe.cardinality == Left()
            @test compressor_recipe.n_rows == n_rows
            @test compressor_recipe.n_cols == n_rows
        end

        # test with right cardinality
        let n_rows = 10,
            n_cols = 3,
            A = ones(n_rows, n_cols)

            compressor_recipe = complete_compressor(Identity(cardinality = Right()), A)

            # test values and types
            @test compressor_recipe.cardinality == Right()
            @test compressor_recipe.n_rows == n_cols 
            @test compressor_recipe.n_cols == n_cols 
        end

    end
    
    @testset "Identity: Update Compressor" begin
        # try updating a left compressor
        let n_rows = 10,
            n_cols = 10,
            A = ones(n_rows, n_cols),
            B = ones(n_rows - 1, n_cols)

            compressor_recipe = complete_compressor(Identity(), A)
            update_compressor!(compressor_recipe, B)
            # test values and types
            @test compressor_recipe.cardinality == Left()
            @test compressor_recipe.n_rows == n_rows - 1
            @test compressor_recipe.n_cols == n_rows - 1
        end
        
        # try updating a right compressor
        let n_rows = 10,
            n_cols = 10,
            A = ones(n_rows, n_cols),
            B = ones(n_rows, n_cols - 1)

            compressor_recipe = complete_compressor(Identity(cardinality = Right()), A)
            update_compressor!(compressor_recipe, B)
            # test values and types
            @test compressor_recipe.cardinality == Right()
            @test compressor_recipe.n_rows == n_cols - 1
            @test compressor_recipe.n_cols == n_cols - 1
        end

    end

    @testset "Identity: Left multiplication" begin
        let n_rows = 10,
            n_cols = 3,
            c_dim = 6,
            A = rand(n_rows, n_cols),
            C = rand(n_rows, n_cols),
            x = rand(n_cols),
            y = rand(n_cols)

            S = complete_compressor(Identity(), A)
            # Test matrix multiplication from the left
            # See if multiplying by S or S' always returns the matrix/vector it is being
            # applied to
            @test S * A ≈ A
            @test S' * A ≈ A
            @test S * x ≈ x
            @test S' * x ≈ x
            mul!(C, S, A, 2.0, 0.0)
            @test C ≈ 2.0 * A
            mul!(C', A', S, 2.0, 0.0)
            @test C ≈ 2.0 * A
            mul!(y, S, x, 2.0, 0.0)
            @test y ≈ 2.0 * x
            mul!(x, S', y, 2.0, 0.0)
            @test x ≈ 2.0 * y
        end

    end

    @testset "Identity: Right multiplication" begin
        let n_rows = 10,
            n_cols = 3,
            c_dim = 6,
            A = rand(n_rows, n_cols),
            C = rand(n_rows, n_cols),
            x = rand(n_cols),
            y = rand(n_cols)

            S = complete_compressor(Identity(), A)
            # Test matrix multiplication from the left
            # See if multiplying by S or S' always returns the matrix/vector it is being
            # applied to
            @test A * S ≈ A
            @test A * S' ≈ A
            @test x' * S ≈ x'
            @test x' * S' ≈ x'
            mul!(C, A, S, 2.0, 0.0)
            @test C ≈ 2.0 * A
            mul!(C', S, A', 2.0, 0.0)
            @test C ≈ 2.0 * A
            mul!(y', x', S, 2.0, 0.0)
            @test y ≈ 2.0 * x
            mul!(x', y', S', 2.0, 0.0)
            @test x ≈ 2.0 * y
        end

    end

end

end
