module sparse_sign
using Test, RandLinearAlgebra, Random
import SparseArrays: sparse, SparseMatrixCSC, sprand
import LinearAlgebra: mul!, Adjoint
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "Sparse Sign" begin
    @testset "Sparse Sign: Compressor" begin
        # Verify Supertype
        @test supertype(SparseSign) == Compressor

        # Verify fields and types
        @test fieldnames(SparseSign) == (:cardinality, :compression_dim, :nnz, :type)
        @test fieldtypes(SparseSign) == (Cardinality, Int64, Int64, Type{<:Number})

        # Verify the Internal Constructor
        let cardinality = Left(), compression_dim = 0, nnz = 8, type = Float64
            @test_throws ArgumentError(
                "Field `compression_dim` must be positive."
            ) SparseSign(
                cardinality, compression_dim, nnz, type
            )
        end

        let cardinality = Left(), compression_dim = 1, nnz = 8, type = Float64
            @test_throws ArgumentError(
                "Number of non-zero indices, $nnz, must be less than or equal to \
                compression dimension, $compression_dim."
            ) SparseSign(cardinality, compression_dim, nnz, type)
        end

        let cardinality = Left(), compression_dim = 1, nnz = 0, type = Float64
            @test_throws ArgumentError("Field `nnz` must be positive.") SparseSign(
                cardinality, compression_dim, nnz, type
            )
        end

        # Verify external constructor and type 
        for Card in [Left, Right]
            compressor = SparseSign(; cardinality=Card())
            typeof(compressor.cardinality) == Card
        end

        for type in [Bool, Int16, Int32, Int64, Float16, Float32, Float64]
            compressor = SparseSign(; cardinality=Right(), type=type)
            @test compressor.type == type
        end

    end

    @testset "Verify field modification" begin
        compressor = SparseSign(cardinality=Left(), compression_dim=5, nnz=3, type=Float64)

        # Test compression_dim
        @test_throws ArgumentError compressor.compression_dim = 0  
        @test_throws ArgumentError compressor.compression_dim = -1 
        @test_throws ArgumentError compressor.compression_dim = 2  
        @test_throws TypeError compressor.compression_dim = 5.5

        # Test nnz
        @test_throws ArgumentError compressor.nnz = 0  
        @test_throws ArgumentError compressor.nnz = -1  
        @test_throws ArgumentError compressor.nnz = 6  
        @test_throws TypeError compressor.nnz = 2.5
        
        # Test correct assignments
        compressor.compression_dim = 10 
        @test compressor.compression_dim == 10
        compressor.nnz = 8 
        @test compressor.nnz == 8
        
        # Test no checking assignments
        compressor.cardinality = Right()
        @test typeof(compressor.cardinality) == Right
        compressor.type = Float32 
        @test compressor.type == Float32
    end

    @testset "Sparse Sign: CompressorRecipe" begin
        @test_compressor SparseSignRecipe
        @test fieldnames(SparseSignRecipe) ==
            (:cardinality, :n_rows, :n_cols, :nnz, :scale, :op)
        @test fieldtypes(SparseSignRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Int64,
            Vector{<:Number},
            Union{
                Adjoint{T,SparseMatrixCSC{T,I}},SparseMatrixCSC
            } where {T<:Number,I<:Integer},
        )

        # Verify the internal constructor
        let card = Left,
            n_rows = 10,
            n_cols = 20,
            nnz = 3,
            scale = [-3.0, 3.0],
            op = sprand(10, 20, 0.5)

            compressor_recipe = SparseSignRecipe(card(), n_rows, n_cols, nnz, scale, op)
            @test typeof(compressor_recipe.cardinality) == card
            @test compressor_recipe.n_rows == n_rows
            @test compressor_recipe.n_cols == n_cols
            @test compressor_recipe.nnz == nnz
            @test compressor_recipe.scale == scale
            @test typeof(compressor_recipe.op) == SparseMatrixCSC{Float64,Int64}
        end

        let card = Right,
            n_rows = 10,
            n_cols = 20,
            nnz = 3,
            scale = [-3.0, 3.0],
            op = sprand(10, 20, 0.5)'

            compressor_recipe = SparseSignRecipe(card(), n_rows, n_cols, nnz, scale, op)
            @test typeof(compressor_recipe.cardinality) == card
            @test compressor_recipe.n_rows == n_rows
            @test compressor_recipe.n_cols == n_cols
            @test compressor_recipe.nnz == nnz
            @test compressor_recipe.scale == scale
            @test typeof(compressor_recipe.op) ==
                Adjoint{Float64,SparseMatrixCSC{Float64,Int64}}
        end

    end

    @testset "Sparse Sign: Complete Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            nnz = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                SparseSign(; cardinality=card, compression_dim=c_dim, nnz=nnz, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_cols
            @test compressor_recipe.nnz == nnz
            sc = type(1 / sqrt(nnz))
            @test compressor_recipe.scale == [-sc, sc]
            @test typeof(compressor_recipe.op) == SparseMatrixCSC{type,Int64}
            # Test the number of nonzeros per row is correct
            correct_nnz = [sum(compressor_recipe.op[:, i] .!= 0) != nnz for i in 1:n_rows]
            @test sum(correct_nnz) == 0
        end

        # test with Left SparseSignConstructor 
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            nnz = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = SparseSignRecipe(card, c_dim, A, nnz, type)

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_cols
            @test compressor_recipe.nnz == nnz
            sc = type(1 / sqrt(nnz))
            @test compressor_recipe.scale == [-sc, sc]
            @test typeof(compressor_recipe.op) == SparseMatrixCSC{type,Int64}
            # Test the number of nonzeros per column is correct
            correct_nnz = [sum(compressor_recipe.op[:, i] .!= 0) != nnz for i in 1:n_rows]
            @test sum(correct_nnz) == 0
        end

        # test with right compressor
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            nnz = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                SparseSign(; cardinality=card, compression_dim=c_dim, nnz=nnz, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.nnz == nnz
            sc = type(1 / sqrt(nnz))
            @test compressor_recipe.scale == [-sc, sc]
            @test typeof(compressor_recipe.op) == Adjoint{type,SparseMatrixCSC{type,Int64}}

            # Test the number of nonzeros per row is correct
            correct_nnz = [sum(compressor_recipe.op[i, :] .!= 0) != nnz for i in 1:n_cols]
            @test sum(correct_nnz) == 0
        end

        # test with Right SparseSignConstructor 
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            nnz = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = SparseSignRecipe(card, c_dim, A, nnz, type)

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.nnz == nnz
            sc = type(1 / sqrt(nnz))
            @test compressor_recipe.scale == [-sc, sc]
            @test typeof(compressor_recipe.op) == Adjoint{type,SparseMatrixCSC{type,Int64}}

            # Test the number of nonzeros per row is correct
            correct_nnz = [sum(compressor_recipe.op[i, :] .!= 0) != nnz for i in 1:n_rows]
            @test sum(correct_nnz) == 0
        end

    end

    @testset "Sparse Sign: Update Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            nnz = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                SparseSign(; cardinality=card, compression_dim=c_dim, nnz=nnz, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_cols
            @test compressor_recipe.nnz == nnz
            sc = type(1 / sqrt(nnz))
            @test compressor_recipe.scale == [-sc, sc]
            @test typeof(compressor_recipe.op) == SparseMatrixCSC{type,Int64}
            # Test that the matrix has changed
            @test compressor_recipe.op != oldmat
            # Test the number of nonzeros per row is correct
            correct_nnz = [sum(compressor_recipe.op[:, i] .!= 0) != nnz for i in 1:n_rows]
            @test sum(correct_nnz) == 0
        end

        # test with right compressor
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            nnz = 3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                SparseSign(; cardinality=card, compression_dim=c_dim, nnz=nnz, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.nnz == nnz
            sc = type(1 / sqrt(nnz))
            @test compressor_recipe.scale == [-sc, sc]
            @test typeof(compressor_recipe.op) == Adjoint{type,SparseMatrixCSC{type,Int64}}
            # Test that the matrix has changed
            @test compressor_recipe.op != oldmat
            # Test the number of nonzeros per row is correct
            correct_nnz = [sum(compressor_recipe.op[i, :] .!= 0) != nnz for i in 1:n_cols]
            @test sum(correct_nnz) == 0
        end

    end

    # Test multimplcations with left compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "Sparse Sign: Left Multiplication" begin
        let n_rows = 20,
            n_cols = 3,
            nnz = 8,
            c_dim = 10,
            A = rand(n_rows, n_cols),
            B = rand(c_dim, n_cols),
            C1 = rand(c_dim, n_cols),
            C2 = rand(n_rows, n_cols),
            x = rand(n_rows),
            y = rand(c_dim),

            # copies are for comparing with the "true version"
            C1c = deepcopy(C1)

            C2c = deepcopy(C2)
            yc = deepcopy(y)
            # Start by testing left sketching multiplications
            S_info = SparseSign(; compression_dim=c_dim)
            S = complete_compressor(S_info, A)
            # Form a vector corresponding to the columns to be generate the sparse mat
            nnz_cols = reduce(vcat, [repeat(i:i, S.nnz) for i in 1:n_rows])
            # Form the compressor to form the actual compressor
            sparse_S = sparse(S.op.rowval, nnz_cols, S.op.nzval)

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

    @testset "Sparse Sign: Left Multiplication Sparse Transpose" begin
        let n_rows = 6,
            n_cols = 4,
            nnz = 3,
            c_dim = 5,
            alpha = 1.25,
            beta = 0.4

            B = sprand(n_cols, n_rows, 0.6)
            A = transpose(B)

            S_info = SparseSign(; compression_dim=c_dim, nnz=nnz)
            S = complete_compressor(S_info, A)
            sparse_S = Matrix(S.op)

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

    @testset "Sparse Sign: Left Multiplication Sparse Transpose (Adjoint op)" begin
        let m = 7,      # number of columns of A = number of rows of B
            n = 6,      # number of rows of A = number of columns of B
            c_dim = 5,
            alpha = 1.1

            # A is a transpose of sparse B (m x n) -> A is (n x m)
            B = sprand(m, n, 0.5)
            A = transpose(B)
            A_dense = Matrix(A)

            # Construct a Left recipe whose op is stored as an Adjoint (c_dim x n)
            P = sprand(n, c_dim, 0.5) # (n x c_dim) so P' is (c_dim x n)
            op = adjoint(P)
            S = SparseSignRecipe(Left(), c_dim, n, 1, [-1.0, 1.0], op)

            C = rand(c_dim, m)
            mul!(C, S, A, alpha, 0.0)
            @test C ≈ alpha * (Matrix(S.op) * A_dense)
        end
    end

    # Test multimplcations with right compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "Sparse Sign: Right Multiplication" begin
        let n = 20,
            nnz = 8,
            c_dim = 10,
            A = rand(n, c_dim),
            B = rand(n, n),
            C1 = rand(c_dim, c_dim),
            C2 = rand(n, c_dim),
            x = rand(c_dim),
            y = rand(n),

            C1c = deepcopy(C1)

            C2c = deepcopy(C2)
            yc = deepcopy(y)
            S_info = SparseSign(; cardinality=Right(), compression_dim=c_dim)
            S = complete_compressor(S_info, B)
            # Form a vector corresponding to the columns to be generate the sparse mat
            nnz_cols = reduce(vcat, [repeat(i:i, S.nnz) for i in 1:n])
            # Form the compressor to form the actual compressor
            sparse_S = sparse(S.op.parent.rowval, nnz_cols, S.op.parent.nzval)'

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
