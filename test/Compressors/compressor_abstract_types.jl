module compressor_abstract_types
using Test, RandLinearAlgebra
import LinearAlgebra: mul!
using ..FieldTest
using ..MockTypes

@testset "Compressor Abstract Types" begin
    @test isdefined(Main, :Compressor)
    @test isdefined(Main, :CompressorRecipe)
    @test isdefined(Main, :CompressorAdjoint)
    @test isdefined(Main, :Cardinality)
    @test isdefined(Main, :Left)
    @test isdefined(Main, :Right)
    @test isdefined(Main, :Undef)
end

@testset "Compressor Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_compressor(TestCompressor(), A)
    @test_throws ArgumentError complete_compressor(TestCompressor(), b)
    @test_throws ArgumentError complete_compressor(TestCompressor(), A, b)
    @test_throws ArgumentError complete_compressor(TestCompressor(), x, A, b)
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(0, 0))
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(0, 0), A)
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(0, 0), A, b)
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(0, 0), x, A, b)
end

end
