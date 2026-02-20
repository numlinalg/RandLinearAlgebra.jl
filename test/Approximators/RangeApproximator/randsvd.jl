module  RandomizedSVD
using Test, RandLinearAlgebra, LinearAlgebra
import RandLinearAlgebra: complete_compressor
import LinearAlgebra: mul!, svd, Diagonal
using ..FieldTest
using ..ApproxTol

mutable struct TestCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
end

TestCompressor() = TestCompressor(Right(), 5)

mutable struct TestCompressorRecipe <: CompressorRecipe 
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

function RandLinearAlgebra.complete_compressor(comp::TestCompressor, A::AbstractMatrix)
    n_cols = comp.compression_dim
    n_rows = size(A, 2)
    # Make a gaussian compressor
    op = randn(n_rows, n_cols) ./ sqrt(n_cols)
    return TestCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

# Define a mul function for the test compressor
function RandLinearAlgebra.mul!(
    C::AbstractMatrix, 
    A::AbstractMatrix, 
    S::TestCompressorRecipe, 
    alpha::Float64, 
    beta::Float64
)
    mul!(C, A, S.op, alpha, beta)
end

@testset "RandSVD" begin 
    @testset "RandSVD" begin
        supertype(RandSVD) == Approximator

        # test the fieldnames and types
        fieldnames(RandSVD) == (:compressor, :power_its, :orthogonalize, :block_size)
        fieldtypes(RandSVD) == (Compressor, Int64, Bool, Int64)
        
        # test errors
        let compressor = TestCompressor(),
            power_its = -1,
            orthogonalize = false,
            block_size = 0

            @test_throws ArgumentError(
                "Field `power_its` must be non-negative."
            ) RandSVD(compressor, power_its, orthogonalize, block_size)
        end

        let compressor = TestCompressor(),
            power_its = 0,
            orthogonalize = false,
            block_size = -1

            @test_throws ArgumentError(
                "Field `block_size` must be non-negative."
            ) RandSVD(compressor, power_its, orthogonalize, block_size)
        end

        # Test constructor
        let compressor = TestCompressor(),
            power_its = 2,
            orthogonalize = false,
            block_size = 2
            rf = RandSVD(
                compressor = compressor, 
                power_its = power_its, 
                orthogonalize = orthogonalize,
                block_size = block_size
            ) 

            @test typeof(rf.compressor) == TestCompressor
            @test rf.power_its == 2
            @test rf.orthogonalize == false
            @test rf.block_size == 2
        end
    
    end

    @testset "RandSVD Recipe" begin
        @test_range_approximator RandSVDRecipe
        supertype(RandSVDRecipe) == ApproximatorRecipe

        # test the fieldnames and types
        @test fieldnames(RandSVDRecipe) == (
            :n_rows, :n_cols, :compressor, :power_its, :orthogonalize, :U, :S, :V, :buffer
        )
        @test fieldtypes(RandSVDRecipe) == (
            Int64, 
            Int64, 
            CompressorRecipe, 
            Int64, 
            Bool, 
            AbstractArray, 
            AbstractVector, 
            AbstractArray,
            AbstractArray
        )
    end
    
    @testset "RandSVD: Complete Approximator" begin
        # Test when correct cardinality, right, is specified with blocksize 0
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            power_its = 2,
            ortho = true,
            block_size = 0,
            cardinality = Right(),
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = complete_approximator(approx, A)
            @test approx_rec.compressor.cardinality == Right()

            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho 
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == n_cols
            @test size(approx_rec.buffer) == (compression_dim, n_cols)
        end
        
        # Test when correct cardinality, right, is specified with blocksize 0
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            power_its = 2,
            block_size = 0,
            ortho = true,
            block_size = 3,
            cardinality = Right(),
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = complete_approximator(approx, A)
            @test approx_rec.compressor.cardinality == Right()

            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho 
            @test approx_rec.n_rows == n_rows
            @test approx_rec.n_cols == n_cols
            @test size(approx_rec.buffer) == (compression_dim, block_size)
        end

        # Test when correct cardinality, left, is specified with blocksize 0
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            power_its = 2,
            ortho = true,
            block_size = 0,
            cardinality = Left(),
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = complete_approximator(approx, A)
            @test approx_rec.compressor.cardinality == Left()

            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho 
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == n_cols
            @test size(approx_rec.buffer) == (compression_dim, n_cols)
        end
        
        # Test when correct cardinality, left, is specified with blocksize 0
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            power_its = 2,
            block_size = 0,
            ortho = true,
            block_size = 3,
            cardinality = Left(),
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = complete_approximator(approx, A)
            @test approx_rec.compressor.cardinality == Left()

            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho 
            @test approx_rec.n_rows == n_rows
            @test approx_rec.n_cols == n_cols
            @test size(approx_rec.buffer) == (compression_dim, block_size)
        end
        
    end

    @testset "RandSVD Recipe: rapproximate" begin
        # By testing the rapproximate function we also test rapproximate!
        # with power iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            power_its = 2,
            ortho = false,
            block_size = 0,
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
            # bound from theorem 9.1 of Halko Martinsson and Tropp is satisfied
            _,S,_ = svd(A)
            norm(S[1:5] - approx_rec.S) < norm(S[6:end])^2 + 
                norm(Diagonal(S[6:end]) * 
                approx_rec.compressor.op[6:end, :] * 
                pinv(approx_rec.compressor.op[1:5, :]))^2
        end


        # Test that compression_dim == n_rows gives a matrix that spans the range of A
        # with power iterations, this way the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            power_its = 2,
            block_size = 0,
            ortho = false,
            cardinality = Right(),
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            @test approx_rec.compressor.cardinality == Right()
            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec.U * (approx_rec.U' * A)) < ATOL
            @test norm(A - A * approx_rec.V * approx_rec.V') < ATOL
        end

        # By testing the rapproximate function we also test rapproximate!
        # with subspace iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            power_its = 2, 
            block_size = 0,
            ortho = true,
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
             # test bound from theorem 9.1 of Halko Martinsson and Tropp is satisfied
             _,S,_ = svd(A)
             norm(S[1:5] - approx_rec.S) < norm(S[6:end])^2 + 
                 norm(Diagonal(S[6:end]) * 
                 approx_rec.compressor.op[6:end, :] * 
                 pinv(approx_rec.compressor.op[1:5, :]))^2
        end

        # Test that compression_dim == n_rows gives a matrix that spans the range of A 
        # with subspace iterations, so that the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            cardinality = Right(),
            power_its = 2,
            block_size = 0,
            ortho = true,
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            @test approx_rec.compressor.cardinality == Right()
            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho 
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec.U * (approx_rec.U' * A)) < ATOL
            @test norm(A - A * approx_rec.V * approx_rec.V') < ATOL
        end
        
        # Perform same sets of test from the left
        # By testing the rapproximate function we also test rapproximate!
        # with power iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Left(),
            power_its = 2,
            block_size = 0,
            ortho = false,
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
            # bound from theorem 9.1 of Halko Martinsson and Tropp is satisfied
            _,S,_ = svd(A)
            norm(S[1:5] - approx_rec.S) < norm(S[6:end])^2 + 
                norm(Diagonal(S[6:end]) * 
                approx_rec.compressor.op[6:end, :] * 
                pinv(approx_rec.compressor.op[1:5, :]))^2
        end


        # Test that compression_dim == n_rows gives a matrix that spans the range of A
        # with power iterations, this way the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            power_its = 2,
            block_size = 0,
            ortho = false,
            cardinality = Left(),
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            @test_logs (:warn, "Compressor with cardinality `Left` \
                being applied from `Right`.") approx_rec = rapproximate(approx, A)
            approx_rec = rapproximate(approx, A)
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            @test approx_rec.compressor.cardinality == Left()
            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec.U * (approx_rec.U' * A)) < ATOL
            @test norm(A - A * approx_rec.V * approx_rec.V') < ATOL
        end

        # By testing the rapproximate function we also test rapproximate!
        # with subspace iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Left(),
            power_its = 2, 
            block_size = 0,
            ortho = true,
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            @test_logs (:warn, "Compressor with cardinality `Left` \
                being applied from `Right`.") approx_rec = rapproximate(approx, A)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
             # test bound from theorem 9.1 of Halko Martinsson and Tropp is satisfied
             _,S,_ = svd(A)
             norm(S[1:5] - approx_rec.S) < norm(S[6:end])^2 + 
                 norm(Diagonal(S[6:end]) * 
                 approx_rec.compressor.op[6:end, :] * 
                 pinv(approx_rec.compressor.op[1:5, :]))^2
        end

        # Test that compression_dim == n_rows gives a matrix that spans the range of A 
        # with subspace iterations, so that the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            cardinality = Left(),
            power_its = 2,
            block_size = 0,
            ortho = true,
            A = rand(n_rows, n_cols),
            compressor = TestCompressor(cardinality, compression_dim)

            approx = RandSVD(compressor, power_its, ortho, block_size)
            @test_logs (:warn, "Compressor with cardinality `Left` \
                being applied from `Right`.") approx_rec = rapproximate(approx, A)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            @test approx_rec.compressor.cardinality == Left()
            @test approx_rec.power_its == power_its 
            @test approx_rec.orthogonalize == ortho 
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # check that the off diagonals are nearly zero 
            @test opnorm(gram_matrix) ≈ 1
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec.U * (approx_rec.U' * A)) < ATOL
            @test norm(A - A * approx_rec.V * approx_rec.V') < ATOL
        end

    end

    @testset "RandSVD Recipe: mul!" begin
        n_rows = 10
        n_cols = 10
        compression_dim = 10
        block_size = 0 
        power_its = 2
        orthogonalize = true
        cardinality = Right()
        A = rand(n_rows, n_cols)
        C = rand(n_rows, n_cols)
        v = rand(n_cols)
        b = rand(n_cols)
        compressor = TestCompressor(cardinality, compression_dim)
        approx = RandSVD(compressor, power_its, orthogonalize, block_size)
        approx_rec = rapproximate(approx, A)
        # Check that the size function works
        size(approx_rec) == (approx_rec.n_rows, approx_rec.n_cols)
        size(approx_rec') == (approx_rec.n_cols, approx_rec.n_rows)
        # also test that the tranpose is the same as the parent
        ApproximatorAdjoint(approx_rec) == transpose(approx_rec)
        approx_rec = transpose(transpose(approx_rec))
        # test multiplication from the left
        let approx_rec = approx_rec, 
            A = A, 
            C = C,
            Cc = deepcopy(C) 

            mul!(C, approx_rec, A, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * approx_rec.U * (approx_rec.S .* approx_rec.V') * A
        end

        # test multiplication from the right
        let approx_rec = approx_rec, 
            A = A, 
            C = C,
            Cc = deepcopy(C) 

            mul!(C, A, approx_rec, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * A * approx_rec.U * (approx_rec.S .* approx_rec.V')
        end

        # Test the vector multiplication
        # from the right
        let approx_rec = approx_rec, 
            v = v, 
            b = b,
            bc = deepcopy(b) 

            mul!(b, approx_rec, v, 2.0, 1.0)
            @test b ≈ bc + 2.0 * approx_rec.U * (approx_rec.S .* approx_rec.V') * v
        end

        # Test the vector multiplication
        # from the left 
        let approx_rec = approx_rec, 
            v = v, 
            b = b,
            bc = deepcopy(b) 

            mul!(b', v', approx_rec, 2.0, 1.0)
            @test b ≈ (bc' + 2.0  * v' * approx_rec.U * (approx_rec.S .* approx_rec.V'))'
        end

        # run the same tests with a different block_size parameter
        let block_size = 3,
            approx = RandSVD(compressor, power_its, orthogonalize, block_size),
            A = A, 
            C = C,
            Cc = deepcopy(C) 
            
            approx_rec = rapproximate(approx, A)
            mul!(C, approx_rec, A, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * approx_rec.U * (approx_rec.S .* approx_rec.V') * A
        end

        # test multiplication from the right
        let block_size = 3,
            approx = RandSVD(compressor, power_its, orthogonalize, block_size),
            A = A, 
            C = C,
            Cc = deepcopy(C) 
            
            approx_rec = rapproximate(approx, A)
            mul!(C, A, approx_rec, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * A * approx_rec.U * (approx_rec.S .* approx_rec.V')
        end

        # Test the vector multiplication
        # from the right
        let block_size = 3,
            approx = RandSVD(compressor, power_its, orthogonalize, block_size),
            v = v, 
            b = b,
            bc = deepcopy(b) 

            approx_rec = rapproximate(approx, A)
            mul!(b, approx_rec, v, 2.0, 1.0)
            @test b ≈ bc + 2.0 * approx_rec.U * (approx_rec.S .* approx_rec.V') * v
        end

        # Test the vector multiplication
        # from the left 
        let block_size = 3,
            approx = RandSVD(compressor, power_its, orthogonalize, block_size),
            v = v, 
            b = b,
            bc = deepcopy(b) 

            approx_rec = rapproximate(approx, A)
            mul!(b', v', approx_rec, 2.0, 1.0)
            @test b ≈ (bc' + 2.0  * v' * approx_rec.U * (approx_rec.S .* approx_rec.V'))'
        end
    end


end

end
