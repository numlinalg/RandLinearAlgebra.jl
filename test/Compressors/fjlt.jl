module fjlt 
using Test, RandLinearAlgebra, Random
import SparseArrays: sparse, SparseMatrixCSC, sprand
import LinearAlgebra: mul!, Adjoint, Diagonal
import Hadamard: hadamard
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "FJLT" begin
    @testset "FJLT: Compressor" begin
        # Verify Supertype
        @test supertype(FJLT) == Compressor

        # Verify fields and types
        @test fieldnames(FJLT) == (
            :cardinality, 
            :compression_dim, 
            :block_size, 
            :sparsity, 
            :type
        )
        @test fieldtypes(FJLT) == (Cardinality, Int64, Int64, Float64, Type{<:Number})

        # Verify the Internal Constructor
        let cardinality = Left(), comp_dim = 0, bs = 2, sparsity = 0.0, type = Float64
            @test_throws ArgumentError(
                "Field `compression_dim` must be positive."
            ) FJLT(
                cardinality, comp_dim, bs, sparsity, type
            )
        end

        let cardinality = Left(), comp_dim = 1, bs = 0, sparsity = 0.0, type = Float64
            @test_throws ArgumentError(
                "Field `block_size` must be positive."
            ) FJLT(
                cardinality, comp_dim, bs, sparsity, type
            )
        end

        let cardinality = Left(), comp_dim = 1, bs = 2, sparsity = 1.1, type = Float64
            @test_throws ArgumentError(
                "Field `sparsity` must be between 0 and 1."
            ) FJLT(cardinality, comp_dim, bs, sparsity, type)
        end

        let cardinality = Undef(), comp_dim = 1, bs = 2, sparsity = 0.1, type = Float64
            @test_throws ArgumentError(
                "Cardinality must be of type `Left` or `Right`."
            ) FJLT(cardinality, comp_dim, bs, sparsity, type)
        end

        # Verify external constructor and type 
        for Card in [Left, Right]
            compressor = FJLT(; cardinality=Card())
            typeof(compressor.cardinality) == Card
        end

        for type in [Bool, Int16, Int32, Int64, Float16, Float32, Float64]
            compressor = FJLT(cardinality=Right(), type=type)
            @test compressor.type == type
        end

    end

    @testset "FJLT: CompressorRecipe" begin
        @test_compressor FJLTRecipe
        # test supertype
        @test supertype(FJLTRecipe) == CompressorRecipe
        @test fieldnames(FJLTRecipe) ==
            (:cardinality, :n_rows, :n_cols, :sparsity, :scale, :op, :signs, :padding)
        @test fieldtypes(FJLTRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Float64,
            Float64,
            SparseMatrixCSC,
            BitVector,
            AbstractMatrix
        )

        # test the constructors
        let card = Left(),
            n_rows = 8,
            n_cols = 3,
            sparsity = .3,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Left()
            @test recipe.n_rows == comp_dim
            @test recipe.n_cols == n_rows
            @test recipe.sparsity == sparsity
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (comp_dim, padded_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (padded_dim, block_size) 
            @test eltype(recipe.padding) == type
        end

        # test the with sparsity of 0 and non power of 2 row 
        let card = Left(),
            n_rows = 6,
            n_cols = 3,
            sparsity = 0.0,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Left()
            @test recipe.n_rows == comp_dim
            @test recipe.n_cols == n_rows
            @test recipe.sparsity == .25 * log(size(A,1))^2 / n_rows 
            sparsity = recipe.sparsity
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (comp_dim, padded_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (padded_dim, block_size) 
            @test eltype(recipe.padding) == type
        end

        # test the with sparsity of 0 and non power of 2 row 
        let card = Right(),
            n_rows = 3,
            n_cols = 6,
            sparsity = 0.0,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Right()
            @test recipe.n_rows == n_cols
            @test recipe.n_cols == comp_dim 
            @test recipe.sparsity == .25 * log(size(A, 2))^2 / n_cols
            sparsity = recipe.sparsity
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (padded_dim, comp_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (block_size, padded_dim) 
            @test eltype(recipe.padding) == type
        end

    end

    @testset "FJLT: Complete Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows 
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.scale == type(1 / (sqrt(4) * sqrt(.3) * sqrt(16)))
            @test typeof(compressor_recipe.op) <: SparseMatrixCSC
            @test typeof(compressor_recipe.signs) == BitVector
            @test typeof(compressor_recipe.padding) == Matrix{type}
        end

        # test the with sparsity of 0 and non power of 2 row 
        let card = Right(),
            n_rows = 3,
            n_cols = 6,
            sparsity = 0.3,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Right()
            @test recipe.n_rows == n_cols
            @test recipe.n_cols == comp_dim 
            @test recipe.sparsity == sparsity 
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (padded_dim, comp_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (block_size, padded_dim) 
            @test eltype(recipe.padding) == type
        end

        # test with right compressor
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols 
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.scale == type(1 / (sqrt(4) * sqrt(.3) * sqrt(16)))
            @test typeof(compressor_recipe.op) <: SparseMatrixCSC
            @test typeof(compressor_recipe.signs) == BitVector
            @test typeof(compressor_recipe.padding) == Matrix{type}
        end

    end

    @testset "FJLT: Update Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            oldsigns = deepcopy(compressor_recipe.signs)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.n_cols == 10 
            @test compressor_recipe.op != oldmat
            @test compressor_recipe.signs != oldsigns
        end

        # test with right compressor
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            oldsigns = deepcopy(compressor_recipe.signs)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == 10 
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.op != oldmat
            @test compressor_recipe.signs != oldsigns
        end

    end

    # Test multimplcations with left compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "FJLT: Left Multiplication" begin
        # Do left multiplicatin from the left side
        # test left multiplication from the left with power of 2
        let n_rows = 8,
            pad_dim = n_rows,
            n_cols = 6,
            comp_dim = 2,
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                    block_size = 2,
                    compression_dim = comp_dim
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(comp_dim, n_cols)
            SAc = deepcopy(SA)
            Sy = rand(comp_dim)
            Syc = deepcopy(Sy)

            mul!(SA, S, A, 2.0, 1.0)
            @test SA ≈ SAc + S.op * (H * (ifelse.(S.signs, 1, -1) .* A) * 2 * S.scale)
            mul!(Sy, S, A[:, 1], 2.0, 1.0)
            @test Syc ≈ Syc + S.op * (H * (ifelse.(S.signs, 1, -1) .* A[:,1]) * 2 * S.scale)
        end

        # test left multiplication from the left with nonpower of 2
        let n_rows = 10,
            pad_dim = 16,
            n_cols = 6,
            comp_dim = 2,
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                    block_size = 2,
                    compression_dim = comp_dim
                ), 
                A
            )
            H = hadamard(pad_dim)
            pad_mat = zeros(pad_dim, n_cols)
            pad_vec = zeros(pad_dim)
            SA = rand(comp_dim, n_cols)
            SAc = deepcopy(SA)
            Sy = rand(comp_dim)
            Syc = deepcopy(Sy)

            mul!(SA, S, A, 2.0, 1.0)
            pad_mat[1:n_rows, :] = A
            pad_res = S.op * (H * (ifelse.(S.signs, 1, -1) .* pad_mat) * 2 * S.scale)
            @test SA ≈ SAc + pad_res[1:comp_dim, :]
            pad_vec[1:n_rows] = A[:, 1]
            pad_res = S.op * (H * (ifelse.(S.signs, 1, -1) .* pad_vec) * 2 * S.scale)
            mul!(Sy, S, A[:, 1], 2.0, 1.0)
            @test Sy ≈ Syc + pad_res[1:comp_dim]
        end

        # Do left multiplication back from right side
        # test left multiplication from the left with power of 2
        let n_rows = 8,
            pad_dim = n_rows,
            n_cols = 2,
            comp_dim = 2,
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                    block_size = 2,
                    compression_dim = comp_dim
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(n_rows, n_rows)
            SAc = deepcopy(SA)
            Sy = rand(n_rows)
            Syc = deepcopy(Sy)

            mul!(SA, A, S, 2.0, 1.0)
            testm = SAc + (A * S.op * H) * Diagonal(ifelse.(S.signs, 1, -1)) * 2 * S.scale
            @test SA ≈ testm
            mul!(Sy', A[1, :]', S, 2.0, 1.0)
            testv = Syc' + A[1, :]' * S.op * H * Diagonal(ifelse.(S.signs, 1, -1)) * 2 * S.scale
            @test Sy ≈ testv'
        end

        # test left multiplication from the left with nonpower of 2
        let n_rows = 10,
            pad_dim = 16,
            n_cols = 2,
            comp_dim = 2
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                    block_size = 2,
                    compression_dim = comp_dim
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(n_rows, size(S, 2))
            SAc = deepcopy(SA)
            Sy = rand(size(S, 2))
            Syc = deepcopy(Sy)
            
            # Test matrix multiplication
            mul!(SA, A, S, 2.0, 1.0)
            pad_res =  A * S.op * H * Diagonal(ifelse.(S.signs, 1, -1)) * S.scale
            @test SA ≈ SAc + 2 * pad_res[:, 1:size(A,1)]
            # Test vector multiplication
            mul!(Sy', A[1, :]', S, 2.0, 1.0)
            pad_res = A[1, :]' * S.op * H * Diagonal(ifelse.(S.signs, 1, -1)) * S.scale 
            @test Sy ≈ Syc + 2 * pad_res[1:size(A, 1)]
        end
    
    end

    # Test multimplcations with right compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "FJLT: Right Multiplication" begin
        # Do right multiplicatin from the left side
        # test right multiplication from the left with power of 2
        let n_rows = 2,
            n_cols = 8,
            comp_dim = 2,
            pad_dim = n_cols,
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                        cardinality = Right(),
                        compression_dim = comp_dim,
                        sparsity = .5,
                        block_size = 2,
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(n_cols, n_cols)
            SAc = deepcopy(SA)
            Sy = rand(n_cols)
            Syc = deepcopy(Sy)

            mul!(SA, S, A, 2.0, 1.0)
            @test SA ≈ SAc + Diagonal(ifelse.(S.signs, 1, -1)) * H * S.op * A * S.scale * 2
            mul!(Sy, S, A[:, 1], 2.0, 1.0)
            @test Sy ≈ Syc + (ifelse.(S.signs, 1, -1) .* H * S.op * A[:,1]) * 2 * S.scale
        end

        # test right multiplication from the left with nonpower of 2
        let n_rows = 2;
            n_cols = 10;
            comp_dim = 2;
            pad_dim = 16;
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                        cardinality = Right(),
                        compression_dim = comp_dim,
                        sparsity = .5,
                        block_size = 2
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(n_cols, n_cols)
            SAc = deepcopy(SA)
            Sy = rand(n_cols)
            Syc = deepcopy(Sy)

            mul!(SA, S, A, 2.0, 1.0)
            pad_res = ifelse.(S.signs, 1, -1) .* H * S.op * A * 2 * S.scale
            @test SA ≈ SAc + pad_res[1:size(S, 1), :]
            pad_res = ifelse.(S.signs, 1, -1) .* H * S.op * A[:, 1] * 2 * S.scale
            mul!(Sy, S, A[:, 1], 2.0, 1.0)
            @test Sy ≈ Syc + pad_res[1:size(S,1)]
        end

        # Do right multiplication back from right side
          # test right multiplication from the left with power of 2
          let n_cols = 8,
            n_rows = 2,
            comp_dim = 2,
            pad_dim = n_cols,
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                        cardinality = Right(),
                        compression_dim = comp_dim,
                        sparsity = .5,
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(n_rows, comp_dim)
            SAc = deepcopy(SA)
            Sy = rand(comp_dim)
            Syc = deepcopy(Sy)

            mul!(SA, A, S, 2.0, 1.0)
            testm = SAc + A * Diagonal(ifelse.(S.signs, 1, -1)) * H * S.op * 2 * S.scale
            @test SA ≈ testm
            mul!(Sy', A[1, :]', S, 2.0, 1.0)
            testv = A[1, :]' * Diagonal(ifelse.(S.signs, 1, -1)) * H * S.op * 2 * S.scale
            @test Sy ≈ Syc + testv'
        end

        # test right multiplication from the left with nonpower of 2
        let n_cols = 10,
            n_rows = 2,
            comp_dim = 2,
            pad_dim = 16,
            A = rand(n_rows, n_cols)
            S = complete_compressor(
                FJLT(
                        cardinality = Right(),
                        compression_dim = comp_dim,
                        sparsity = .5,
                        block_size = 2,
                ), 
                A
            )
            H = hadamard(pad_dim)
            SA = rand(n_rows, size(S, 2))
            SAc = deepcopy(SA)
            Sy = rand(size(S, 2))
            Syc = deepcopy(Sy)
            pad_mat = zeros(n_rows, pad_dim)
            pad_vec = zeros(pad_dim)
            
            # Test matrix multiplication
            mul!(SA, A, S, 2.0, 1.0)
            pad_mat[:, 1:n_cols] = A
            pad_res =  pad_mat * Diagonal(ifelse.(S.signs, 1, -1)) * H * S.op * S.scale
            @test SA ≈ SAc + 2 * pad_res[:, 1:size(A,1)]
            # Test vector multiplication
            mul!(Sy', A[1, :]', S, 2.0, 1.0)
            pad_vec[1:n_cols] =  A[1, :]
            pad_res = pad_vec' * Diagonal(ifelse.(S.signs, 1, -1)) * H * S.op * S.scale
            @test Sy ≈ Syc + 2 * pad_res[1:size(A, 1)]
        end

    end

end

end
