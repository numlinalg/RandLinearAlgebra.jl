module compressor_abstract_multiplication_errors
include("../test_helpers/preamble.jl")
using .MockCompressors

#####################
# Testing Parameters
#####################
s_rows = 3
s_cols = 4

@testset "Compressor Recipe Multiplication Errors" begin 
    # Test Set Parameters 
    x = randn(2)
    y = randn(2)
    A = randn(2, 2)
    C = randn(2, 2) 

    # Five argument muls 
    @test_throws ArgumentError mul!(C, MockCompressorRecipe(s_rows, s_cols), A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, MockCompressorRecipe(s_rows, s_cols), 1.0, 1.0)
    @test_throws ArgumentError mul!(x, MockCompressorRecipe(s_rows, s_cols), y, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, MockCompressorRecipe(s_rows, s_cols), 1.0, 1.0)

    # Three argument muls 
    @test_throws ArgumentError mul!(C, MockCompressorRecipe(s_rows, s_cols), A)
    @test_throws ArgumentError mul!(C, A, MockCompressorRecipe(s_rows, s_cols))
    @test_throws ArgumentError mul!(x, MockCompressorRecipe(s_rows, s_cols), y)
    @test_throws ArgumentError mul!(x, y, MockCompressorRecipe(s_rows, s_cols))

    # Binary muls 
    @test_throws ArgumentError MockCompressorRecipe(s_rows, s_cols)*A 
    @test_throws ArgumentError A*MockCompressorRecipe(s_rows, s_cols) 
    @test_throws ArgumentError MockCompressorRecipe(s_rows, s_cols)*y
    @test_throws ArgumentError y*MockCompressorRecipe(s_rows, s_cols)

    # Five argument muls for adjoint 
    @test_throws ArgumentError mul!(C, MockCompressorRecipe(s_rows, s_cols)', A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, MockCompressorRecipe(s_rows, s_cols)', 1.0, 1.0)
    @test_throws ArgumentError mul!(y, MockCompressorRecipe(s_rows, s_cols)', x, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, MockCompressorRecipe(s_rows, s_cols)', 1.0, 1.0)

    # Three arguments muls for adjoint 
    @test_throws ArgumentError mul!(C, MockCompressorRecipe(s_rows, s_cols)', A)
    @test_throws ArgumentError mul!(C, A, MockCompressorRecipe(s_rows, s_cols)')
    @test_throws ArgumentError mul!(y, MockCompressorRecipe(s_rows, s_cols)', x)
    @test_throws ArgumentError mul!(y, x, MockCompressorRecipe(s_rows, s_cols)')

    # Binary muls 
    @test_throws ArgumentError MockCompressorRecipe(s_rows, s_cols)'*A 
    @test_throws ArgumentError A*MockCompressorRecipe(s_rows, s_cols)' 
    @test_throws ArgumentError MockCompressorRecipe(s_rows, s_cols)'*y
    @test_throws ArgumentError y*MockCompressorRecipe(s_rows, s_cols)' 
end

end
