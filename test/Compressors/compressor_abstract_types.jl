module compressor_abstract_types
include("../test_helpers/preamble.jl")
using .MockCompressors

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

    @test_throws ArgumentError complete_compressor(MockCompressor(), A)
    @test_throws ArgumentError complete_compressor(MockCompressor(), b)
    @test_throws ArgumentError complete_compressor(MockCompressor(), A, b)
    @test_throws ArgumentError complete_compressor(MockCompressor(), x, A, b)
    @test_throws ArgumentError update_compressor!(MockCompressorRecipe(2, 2))
    @test_throws ArgumentError update_compressor!(MockCompressorRecipe(2, 2), A)
    @test_throws ArgumentError update_compressor!(MockCompressorRecipe(2, 2), A, b)
    @test_throws ArgumentError update_compressor!(MockCompressorRecipe(2, 2), x, A, b)
end

end
