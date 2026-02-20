module compressor_abstract_adjoint
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
s_rows = 3
s_cols = 4

@testset "Compressor Recipe Adjoints" begin
    # Adjoint 
    let compress = TestCompressorRecipe(s_rows, s_cols)
        compress_adjoint = adjoint(compress) 
        @test typeof(compress_adjoint) == CompressorAdjoint{TestCompressorRecipe}
    end

    # Adjoint of Adjoint 
    let compress = TestCompressorRecipe(s_rows, s_cols),
        compress_adjoint = adjoint(compress)

        compress_adjoint_adjoint = adjoint(compress_adjoint)
        @test compress_adjoint_adjoint == compress 
        @test typeof(compress_adjoint_adjoint) == TestCompressorRecipe
    end

    # Transpose 
    let compress = TestCompressorRecipe(s_rows, s_cols)
        compress_transpose = transpose(compress) 
        @test typeof(compress_transpose) == CompressorAdjoint{TestCompressorRecipe}
    end

    # Transpose of Transpose
    let compress = TestCompressorRecipe(s_rows, s_cols),
        compress_transpose = transpose(compress)

        compressor_transpose_transpose = transpose(compress_transpose)
        @test compressor_transpose_transpose == compress
        @test typeof(compressor_transpose_transpose) == TestCompressorRecipe
    end
end

end
