module compressor_abstract_adjoint
include("../test_helpers/preamble.jl")
using .MockCompressors

#####################
# Testing Parameters
##################### 
s_rows = 3
s_cols = 4

@testset "Compressor Recipe Adjoints" begin
    # Adjoint 
    let compress = MockCompressorRecipe(s_rows, s_cols)
        compress_adjoint = adjoint(compress) 
        @test typeof(compress_adjoint) == CompressorAdjoint{MockCompressorRecipe}
    end

    # Adjoint of Adjoint 
    let compress = MockCompressorRecipe(s_rows, s_cols),
        compress_adjoint = adjoint(compress)

        compress_adjoint_adjoint = adjoint(compress_adjoint)
        @test compress_adjoint_adjoint == compress 
        @test typeof(compress_adjoint_adjoint) == MockCompressorRecipe
    end

    # Transpose 
    let compress = MockCompressorRecipe(s_rows, s_cols)
        compress_transpose = transpose(compress) 
        @test typeof(compress_transpose) == CompressorAdjoint{MockCompressorRecipe}
    end

    # Transpose of Transpose
    let compress = MockCompressorRecipe(s_rows, s_cols),
        compress_transpose = transpose(compress)

        compressor_transpose_transpose = transpose(compress_transpose)
        @test compressor_transpose_transpose == compress
        @test typeof(compressor_transpose_transpose) == MockCompressorRecipe
    end
end

end
