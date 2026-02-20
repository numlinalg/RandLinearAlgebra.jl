module CountSketch_compressor
using Test, RandLinearAlgebra, Random
import SparseArrays: sparse, SparseMatrixCSC, sprandn
import LinearAlgebra: mul!, lmul!
import Random: randn!, seed!, rand
using ..FieldTest

@testset "CountSketch" begin
    @testset "CountSketch: Compressor" begin
        # Verify Supertype
        @test supertype(CountSketch) == Compressor

        # Verify fields and types
        @test fieldnames(CountSketch) == (:cardinality, :compression_dim, :type)
        @test fieldtypes(CountSketch) == (Cardinality, Int64, Type{<:Number})

        let cardinality = Left(), compression_dim = 0, type = Float64
            @test_throws ArgumentError(
                "Field 'compression_dim' must be positive."
            ) CountSketch(
                cardinality, compression_dim, type
            )
        end

        let cardinality = Left(), compression_dim = -7, type = Float64
            @test_throws ArgumentError(
                "Field 'compression_dim' must be positive."
            ) CountSketch(
                cardinality, compression_dim, type
            )
        end

        let cardinality = Undef(), compression_dim = 2, type = Float64
            @test_throws ArgumentError(
                "`cardinality` must be specified as `Left()` or `Right()`.\
                    `Undef()` is not allowed in `CountSketch` structure."
            ) CountSketch(
                cardinality, compression_dim, type
            )
        end

        # Verify external constructor and type 
        for Card in [Left, Right]
            compressor = CountSketch(; cardinality=Card())
            @test typeof(compressor.cardinality) == Card
        end

        for type in [Bool, Int16, Int32, Int64, Float16, Float32, Float64]
            compressor = CountSketch(; cardinality=Left(), type=type)
            @test compressor.type == type
        end

        for type in [Bool, Int16, Int32, Int64, Float16, Float32, Float64]
            compressor = CountSketch(; cardinality=Right(), type=type)
            @test compressor.type == type
        end

    end

    @testset "CountSketch: CompressorRecipe" begin
        @test_compressor CountSketchRecipe
        @test fieldnames(CountSketchRecipe) ==
            (:cardinality, :compression_dim, :n_rows, :n_cols, :mat)
        @test fieldtypes(CountSketchRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Int64,
            SparseMatrixCSC,
        )
    end
    
    @testset "CountSketch: Complete Compressor" begin
        # Test with left compressor
        let card = Left(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            A = ones(n_rows, n_cols),
            type = Float32
            
            compressor_recipe = complete_compressor(
                CountSketch(; cardinality=card, compression_dim=c_dim, type=type), A
            )
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows
            @test typeof(compressor_recipe.mat) == SparseMatrixCSC{type,Int64}
            # Test whether each column just contains one non-zero entry
            @test all(col -> count(!iszero, col) == 1, eachcol(Matrix(compressor_recipe.mat)))
        end

        let card = Left(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            A = ones(n_rows, n_cols),
            type = Float32
            
            compressor_recipe = CountSketchRecipe(
                card, c_dim, A, type
            )
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows
            @test typeof(compressor_recipe.mat) == SparseMatrixCSC{type,Int64}
            # Test whether each column just contains one non-zero entry
            @test all(col -> count(!iszero, col) == 1, eachcol(Matrix(compressor_recipe.mat)))
        end

        let card = Right(),
            n_rows = 2,
            n_cols = 6,
            c_dim = 3,
            A = ones(n_rows, n_cols),
            type = Int32

            compressor_recipe = CountSketchRecipe(
                card, c_dim, A, type
            )
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            @test typeof(compressor_recipe.mat) == SparseMatrixCSC{type,Int64}
            # Test whether each row just contains one non-zero entry
            @test all(row -> count(!iszero, row) == 1, eachrow(Matrix((compressor_recipe.mat)')))
        end

        # Test with right compressor
        let card = Right(),
            n_rows = 2,
            n_cols = 6,
            c_dim = 3,
            A = ones(n_rows, n_cols),
            type = Int32

            compressor_recipe = complete_compressor(
                CountSketch(; cardinality=card, compression_dim=c_dim, type=type), A
            )
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            @test typeof(compressor_recipe.mat) == SparseMatrixCSC{type,Int64}
            # Test whether each row just contains one non-zero entry
            @test all(row -> count(!iszero, row) == 1, eachrow(Matrix((compressor_recipe.mat)')))
        end

    end

    @testset "CountSketch: Update Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 4,
            n_cols = 2,
            c_dim = 3,
            A = ones(n_rows, n_cols),
            type = Float16

            compressor_recipe = complete_compressor(
                CountSketch(; cardinality=card, compression_dim=c_dim, type=type), A
            ) 
            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.mat)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows
            @test typeof(compressor_recipe.mat) == SparseMatrixCSC{type,Int64}
            # Test that the matrix has changed
            @test compressor_recipe.mat != oldmat
            # Test whether each column just contains one non-zero entry
            @test all(col -> count(!iszero, col) == 1, eachcol(Matrix(compressor_recipe.mat)))
        end

        # test with right compressor
        let card = Right(),
            n_rows = 2,
            n_cols = 6,
            c_dim = 3,
            type = Float16,
            A = ones(n_rows, n_cols)

            compressor_recipe = complete_compressor(
                CountSketch(; cardinality=card, compression_dim=c_dim, type=type), A
            )
            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.mat)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            @test typeof(compressor_recipe.mat) == SparseMatrixCSC{type,Int64}
            # Test that the matrix has changed
            @test compressor_recipe.mat != oldmat
            # Test whether each row just contains one non-zero entry
            @test all(row -> count(!iszero, row) == 1, eachrow(Matrix((compressor_recipe.mat)')))
        end        

    end

    @testset "Count Sketch: Left Cardinality" begin
        let n_rows = 20,
            n_cols = 3,
            nnz = 8,
            c_dim = 10,
            A = rand(n_rows, n_cols),
            B = rand(c_dim, n_cols),
            C1 = rand(c_dim, n_cols),
            C2 = rand(n_rows, n_cols),
            x = rand(n_rows),
            y = rand(c_dim)

            # copies are for comparing with the "true version"
            C1c = deepcopy(C1)
            C2c = deepcopy(C2)
            yc = deepcopy(y)
            # Start by testing left sketching multiplications
            S_info = CountSketch(; compression_dim=c_dim)
            S = complete_compressor(S_info, A)
            sparse_S = Matrix(deepcopy(S.mat))
            # Using * will test three element mul and * multiplication
            # Test matrix multiplication from the left
            @test S * A ≈ sparse_S * A
            # test transpose multiplication from the left 
            @test S' * B ≈ sparse_S' * B
            # Test multiplication from the right
            @test B' * S ≈ B' * sparse_S
            # Test transpose multiplication from the right
            @test A' * S' ≈ A' * sparse_S'
            # Test matrix vector multiplication from the left
            @test S * x ≈ sparse_S * x
            # Test multiplication from the right using transpose
            @test S' * y ≈ sparse_S' * y

            # Test the summing of the five argument mul
            mul!(C1, S, A, 2.0, 2.0)
            @test C1 ≈ 2.0 * sparse_S * A + 2.0 * C1c
            mul!(C2', B', S, 2.0, 2.0)
            @test C2' ≈ 2.0 * B' * sparse_S + 2.0 * C2c'
            mul!(y, S, x, 2.0, 2.0)
            @test y ≈ 2.0 * sparse_S * x + 2.0 * yc
            # make copy of x here because we have overwritten it above 
            xc = deepcopy(x)
            mul!(x, S', y, 2.0, 2.0)
            @test x ≈ 2.0 * sparse_S' * y + 2.0 * xc
        end

    end

    @testset "Count Sketch: Left Cardinality Sparse Transpose" begin
        let n_rows = 7,
            n_cols = 5,
            c_dim = 4,
            alpha = 1.4,
            beta = 0.5

            B = sprandn(n_cols, n_rows, 0.5)
            A = transpose(B)

            S_info = CountSketch(; compression_dim=c_dim)
            S = complete_compressor(S_info, A)
            sparse_S = Matrix(deepcopy(S.mat))

            C = rand(c_dim, n_cols)
            C0 = deepcopy(C)
            A_dense = Matrix(A)

            mul!(C, S, A, alpha, beta)
            @test C ≈ alpha * (sparse_S * A_dense) + beta * C0

            # Cover the beta == 0 fill! branch in the specialized method
            C = rand(c_dim, n_cols)
            mul!(C, S, A, alpha, 0.0)
            @test C ≈ alpha * (sparse_S * A_dense)
        end
    end

    @testset "Count Sketch: Right Cardinality" begin
        let n = 20,
            nnz = 8,
            c_dim = 10,
            A = rand(n, c_dim),
            B = rand(n, n),
            C1 = rand(c_dim, c_dim),
            C2 = rand(n, c_dim),
            x = rand(c_dim),
            y = rand(n)
            
            C1c = deepcopy(C1)
            C2c = deepcopy(C2)
            yc = deepcopy(y)
            S_info = CountSketch(; cardinality=Right(), compression_dim=c_dim)
            S = complete_compressor(S_info, B)
            sparse_S = Matrix(deepcopy(S.mat'))

            # Using * will test three element mul and * multiplication
            # Test matrix multiplication from the left
            @test S' * A ≈ sparse_S' * A
            # test transpose multiplication from the left 
            @test B * S ≈ B * sparse_S
            # Test multiplication from the right
            @test B' * S ≈ B' * sparse_S
            # Test transpose multiplication from the right
            @test A * S' ≈ A * sparse_S'
            # Test matrix vector multiplication from the left
            @test S * x ≈ sparse_S * x
            # Test multiplying the matrix to a vector from the right
            @test y' * S ≈ y' * sparse_S
            # Test multiplication from the right using transpose
            @test S' * y ≈ sparse_S' * y

            # Test the scalar addition portion of the multiplications
            # The unscaled versions are tested in the start multipliction
            mul!(C1, S', A, 2.0, 2.0)
            @test C1 ≈ 2.0 * sparse_S' * A + 2.0 * C1c
            mul!(C2, B, S, 2.0, 2.0)
            @test C2 ≈ 2.0 * B * sparse_S + 2.0 * C2c
            mul!(y, S, x, 2.0, 2.0)
            @test y ≈ 2.0 * sparse_S * x + 2.0 * yc
            # make copy of x here it was overwritten in previous mul 
            xc = deepcopy(x)
            mul!(x, S', y, 2.0, 2.0)
            @test x ≈ 2.0 * sparse_S' * y + 2.0 * xc
        end

    end

end

end
