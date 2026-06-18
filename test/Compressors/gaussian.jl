module Gaussian_compressor
using Test, RandLinearAlgebra
import Base.*
import LinearAlgebra: mul!, lmul!
import Random: randn!, seed!, rand
using ..FieldTest
using ..ApproxTol

seed!(21321)
@testset "Gaussian" begin
    @testset "Gaussian: Compressor" begin
        # Verify Supertype
        @test supertype(Gaussian) == Compressor

        # Verify fields and types
        @test fieldnames(Gaussian) == (:cardinality, :compression_dim, :type)
        @test fieldtypes(Gaussian) == (Cardinality, Int64, Type{<:Number})

        let cardinality = Left(), compression_dim = 0, type = Float64
            @test_throws ArgumentError(
                "Field 'compression_dim' must be positive."
            ) Gaussian(
                cardinality, compression_dim, type
            )
        end

        let cardinality = Left(), compression_dim = -7, type = Float64
            @test_throws ArgumentError(
                "Field 'compression_dim' must be positive."
            ) Gaussian(
                cardinality, compression_dim, type
            )
        end

        # Verify external constructor and type 
        for Card in [Left, Right]
            compressor = Gaussian(; cardinality=Card())
            typeof(compressor.cardinality) == Card
        end

        for type in [Float16, Float32, Float64]
            compressor = Gaussian(; cardinality=Right(), type=type)
            @test compressor.type == type
        end

    end

    @testset "Gaussian: CompressorRecipe" begin
        @test_compressor GaussianRecipe
        @test fieldnames(GaussianRecipe) ==
            (:cardinality, :compression_dim, :n_rows, :n_cols, :scale, :op)
        @test fieldtypes(GaussianRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Int64,
            Number,
            AbstractMatrix{<:Number},
        )
    end

    @testset "Gaussian: Complete Compressor" begin
        # Test with left compressor
        let card = Left(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                Gaussian(; cardinality=card, compression_dim=c_dim, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows
            sc = convert(type, 1 / sqrt(c_dim))
            @test compressor_recipe.scale == sc
            @test typeof(compressor_recipe.op) == Matrix{type}
        end

        let card = Left(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = GaussianRecipe(
                card, c_dim, type, A
            )


            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows
            sc = convert(type, 1 / sqrt(c_dim))
            @test compressor_recipe.scale == sc
            @test typeof(compressor_recipe.op) == Matrix{type}
        end

        let card = Right(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = GaussianRecipe(
                card, c_dim, type, A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            sc = convert(type, 1 / sqrt(c_dim))
            @test compressor_recipe.scale == sc
            @test typeof(compressor_recipe.op) == Matrix{type}
        end

        # Test with right compressor
        let card = Right(),
            n_rows = 2,
            n_cols = 6,
            c_dim = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                Gaussian(; cardinality=card, compression_dim=c_dim, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            sc = convert(type, 1 / sqrt(c_dim))
            @test compressor_recipe.scale == sc
            @test typeof(compressor_recipe.op) == Matrix{type}
        end

    end

    @testset "Gaussian: Update Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                Gaussian(; cardinality=card, compression_dim=c_dim, type=type), A
            ) 

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows
            sc = convert(type, 1 / sqrt(c_dim))
            @test compressor_recipe.scale == sc
            @test typeof(compressor_recipe.op) == Matrix{type}
            # Test that the matrix has changed
            @test compressor_recipe.op != oldmat
        end

        # test with right compressor
        let card = Right(),
            n_rows = 2,
            n_cols = 6,
            c_dim = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                Gaussian(; cardinality=card, compression_dim=c_dim, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            sc = convert(type, 1 / sqrt(c_dim))
            @test compressor_recipe.scale == sc
            @test typeof(compressor_recipe.op) == Matrix{type}
            # Test that the matrix has changed
            @test compressor_recipe.op != oldmat
        end        

    end

    # Test multimplcations with left compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "Gaussian: Left Multiplication" begin
        let n_rows = 10,
            n_cols = 3,
            c_dim = 6,
            type = Float16,
            A = rand(n_rows, n_cols),
            B = rand(c_dim, n_cols),
            C1 = rand(c_dim, n_cols),
            C2 = rand(n_rows, c_dim),
            C3 = rand(n_rows, n_cols),
            x = rand(n_rows),
            y = rand(c_dim),
            S_info = Gaussian(cardinality = Left(), compression_dim = c_dim, type = type),
            S = complete_compressor(S_info, A)

            # Form a vector corresponding to the columns to generate the sketch matrix
            S_test = deepcopy(S.op)
            # Test matrix multiplication from the left
            @test S * A ≈ S_test * A
            # Using transpose will test matrix multiplication from the right
            @test S' * B ≈ S_test' * B
            # Test matrix vector multiplication from the left
            @test S * x ≈ S_test * x
            # Using transpose will test vec multiplication from the right
            @test S' * y ≈ S_test' * y
            S' * y

            # Test the scalar addition portion of the multiplications
            mul!(C1, S, A, 2.0, 0.0)
            @test C1 ≈ 2.0 * S_test * A
            mul!(C3', B', S, 2.0, 0.0)
            @test C3' ≈ 2.0 * B' * S_test 
            mul!(y, S, x, 2.0, 0.0)
            @test y ≈ 2.0 * S_test * x 
            mul!(x, S', y, 2.0, 0.0)
            @test x ≈ 2.0 * S_test' * y 
        end

    end

    # Test multimplcations with right compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "Gaussian: Right Multiplication" begin
        let n_rows = 3,
            n_cols = 10,
            c_dim = 6,
            type = Float16,
            A = rand(n_rows, n_cols),
            B = rand(n_cols, c_dim),
            C1 = rand(n_rows, c_dim),
            C3 = rand(c_dim, c_dim),
            x = rand(c_dim),
            y = rand(n_cols),
            S_info = Gaussian(cardinality = Right(), compression_dim = c_dim, type = type),
            S = complete_compressor(S_info, A)

            # Form a vector corresponding to the columns to generate the sketch matrix
            S_test = deepcopy(S.op)
            # Test matrix multiplication from the right
            @test A * S ≈ A * S_test
            # Test transpose from right
            @test B * S' ≈ B * S_test'
            # Test matrix multiplication from the left
            @test S * B' ≈ S_test * B'
            # Using transpose from left
            @test S' * A' ≈ S_test' * A'
            # Test matrix vector multiplication from the left
            @test S * x ≈ S_test * x
            # Test multiplication from the right using transpose
            @test S' * y ≈ S_test' * y

            # Test the scalar addition portion of the multiplications
            mul!(C1, A, S, 2.0, 0.0)
            @test C1 ≈ 2.0 * A * S_test
            mul!(C3, S', B, 2.0, 0.0)
            @test C3' ≈ 2.0 * B' * S_test 
            mul!(y, S, x, 2.0, 0.0)
            @test y ≈ 2.0 * S_test * x 
            mul!(x, S', y, 2.0, 0.0)
            @test x ≈ 2.0 * S_test' * y 
        end

    end

end

end
