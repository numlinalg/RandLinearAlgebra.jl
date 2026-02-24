module  RangeApproximator
using Test, RandLinearAlgebra, LinearAlgebra
import RandLinearAlgebra: complete_compressor
import LinearAlgebra: mul!
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
    S::Main.RangeApproximator.TestCompressorRecipe, 
    alpha::Float64, 
    beta::Float64
)
    mul!(C, A, S.op, alpha, beta)
end

@testset "Randomized RangeFinder" begin 
    @testset "Randomized RangeFinder" begin
        supertype(RangeFinder) == Approximator

        # test the fieldnames and types
        fieldnames(RangeFinder) == (:compressor, :power_its, :orthogonalize)
        fieldtypes(RangeFinder) == (Compressor, Int64, Bool)
        
        # test errors
        let compressor = TestCompressor(),
            power_its = -1,
            orthogonalize = false

            @test_throws ArgumentError(
                "Field `power_its` must be non-negative."
            ) RangeFinder(compressor, power_its, orthogonalize)
        end

        # Test constructor
        let compressor = TestCompressor(),
            power_its = 2,
            orthogonalize = false,
            rf = RangeFinder(compressor, power_its, orthogonalize) 

            @test typeof(rf.compressor) == TestCompressor
            @test rf.power_its == 2
            @test rf.orthogonalize == false
        end
    
    end

    @testset "Randomized RangeFinder Recipe" begin
        @test_range_approximator RangeFinderRecipe
        supertype(RangeFinderRecipe) == ApproximatorRecipe

        # test the fieldnames and types
        @test fieldnames(RangeFinderRecipe) == (
            :n_rows, :n_cols, :compressor, :power_its, :orthogonalize, :range
        )
        @test fieldtypes(RangeFinderRecipe) == (
            Int64, Int64, CompressorRecipe, Int64, Bool, AbstractMatrix
        )
    end
    
    @testset "Randomized RangeFinder: Complete Approximator" begin
        # Test when correct cardinality, right, is specified
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RangeFinder(compressor, 2, true)
            approx_rec = complete_approximator(approx, A)
            approx_rec.compressor.cardinality == Right()

            @test approx_rec.power_its == 2
            @test approx_rec.orthogonalize == true
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 5
        end
        
        # Test when wrong cardinality, left, is specified
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Left(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RangeFinder(compressor, 2, false)
            @test_logs (:warn,
                "Compressor with cardinality `Left` being applied from `Right`."
            ) complete_approximator(approx, A)
            approx_rec = complete_approximator(approx, A)
            approx_rec.compressor.cardinality == Left()
            
            @test approx_rec.power_its == 2
            @test approx_rec.orthogonalize == false 
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 5
        end
        
    end

    @testset "Randomized RangeFinder Recipe: rapproximate" begin
        # By testing the rapproximate function we also test rapproximate!
        # with power iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RangeFinder(compressor, 2, false)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.range' * approx_rec.range
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

        end

        # Test that compression_dim == n_rows gives a matrix that spans the range of A
        # with power iterations, this way the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RangeFinder(compressor, 2, false)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            approx_rec.compressor.cardinality == Right()
            @test approx_rec.power_its == 2
            @test approx_rec.orthogonalize == false
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.range' * approx_rec.range
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec * (approx_rec' * A)) < ATOL
        end

        # By testing the rapproximate function we also test rapproximate!
        # with orthogonalized power iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RangeFinder(compressor, 2, true)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.range' * approx_rec.range
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

        end

        # Test that compression_dim == n_rows gives a matrix that spans the range of A 
        # with orthogonalized power iterations, so that the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RangeFinder(compressor, 2, true)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            approx_rec.compressor.cardinality == Right()
            @test approx_rec.power_its == 2
            @test approx_rec.orthogonalize == true
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.range' * approx_rec.range
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec * (approx_rec' * A)) < ATOL
        end

    end

    @testset "Randomized RangeFinder Recipe: mul!" begin
        n_rows = 10
        n_cols = 10
        compression_dim = 10 
        cardinality = Right()
        A = rand(n_rows, n_cols)
        C = rand(n_rows, n_cols)
        v = rand(n_cols)
        b = rand(n_cols)
        compressor = TestCompressor(cardinality, compression_dim)
        approx = RangeFinder(compressor, 2, true)
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
            @test C ≈ Cc + 2.0 * approx_rec.range * A
        end

        # test multiplication from the right
        let approx_rec = approx_rec, 
            A = A, 
            C = C,
            Cc = deepcopy(C) 

            mul!(C, A, approx_rec, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * A * approx_rec.range
        end

        # Test the vector multiplication
        # from the right
        let approx_rec = approx_rec, 
            v = v, 
            b = b,
            bc = deepcopy(b) 

            mul!(b, approx_rec, v, 2.0, 1.0)
            @test b ≈ bc + 2.0 * approx_rec.range * v
        end

        # Test the vector multiplication
        # from the left 
        let approx_rec = approx_rec, 
            v = v, 
            b = b,
            bc = deepcopy(b) 

            mul!(b', v', approx_rec, 2.0, 1.0)
            @test b ≈ (bc' + 2.0  * v' * approx_rec.range)'
        end

    end


end

end
