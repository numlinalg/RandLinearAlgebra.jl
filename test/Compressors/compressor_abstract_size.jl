
module compressor_abstract_size
using Test, RandLinearAlgebra
import LinearAlgebra: mul!
using ..FieldTest
using ..ApproxTol

#####################
# Testing Parameters
##################### 
mutable struct TestCompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
end
s = 3
n_rows = 4
n_cols = 5
S = TestCompressorRecipe(s, n_rows)


@testset "Compressor Size" begin
    # Size of Compressor 
    let S = deepcopy(S), s=s, n_rows=n_rows

        # Get size 
        m, n = size(S)
        @test m == s
        @test n == n_rows

        # Get individual sizes 
        @test n_rows == size(S, 2)
        @test s == size(S, 1)

        # Fail individual sizes for incorrect dims 
        @test_throws DomainError size(S, 0)
        @test_throws DomainError size(S, 3)
    end

    # Size of Adjoint 
    let S = deepcopy(S)', s=s, n_rows=n_rows

        # Get Size 
        @test size(S) == (n_rows, s)

        # Get individual sizes 
        @test size(S, 1) == n_rows
        @test size(S, 2) == s

        # Fail individual sizes for incorrect dims 
        @test_throws DomainError size(S, 0)
        @test_throws DomainError size(S, 3)
    end
end
end
