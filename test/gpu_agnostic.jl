module gpu_agnostic

using Test
using RandLinearAlgebra
using LinearAlgebra
using Random

include("test_helpers/mock_array.jl")

@testset "Array-agnostic allocations (GPU compatibility)" begin

    m, n, s = 20, 10, 4
    A_cpu  = randn(m, n)
    A_mock = MockArray(A_cpu)
    b_cpu  = randn(m)
    b_mock = MockArray(b_cpu)
    x_cpu  = zeros(n)
    x_mock = MockArray(x_cpu)

    @testset "GaussianRecipe respects array type" begin
        recipe = complete_compressor(Gaussian(compression_dim=s), A_mock)
        @test recipe.op isa MockArray
        @test size(recipe.op) == (s, m)
    end

    @testset "Gaussian binary * respects array type" begin
        result = Gaussian(compression_dim=s) |> g -> complete_compressor(g, A_mock) |> r -> r * A_mock
        @test result isa MockArray
    end

    @testset "KaczmarzRecipe respects array type" begin
        solver = Kaczmarz(compressor=Gaussian(compression_dim=s))
        recipe = complete_solver(solver, x_mock, A_mock, b_mock)
        @test recipe.compressed_mat isa MockArray
        @test recipe.compressed_vec isa MockArray
        @test recipe.update_vec isa MockArray
    end

    @testset "FullResidualRecipe respects array type" begin
        recipe = complete_error(FullResidual(), Kaczmarz(), A_mock, b_mock)
        @test recipe.residual isa MockArray
    end

    @testset "CompressedResidualRecipe respects array type" begin
        recipe = complete_error(CompressedResidual(), Kaczmarz(), A_mock, b_mock)
        @test recipe.residual isa MockArray
    end

    @testset "LSGradientRecipe respects array type" begin
        recipe = complete_error(LSGradient(), Kaczmarz(), A_mock, b_mock)
        @test recipe.gradient isa MockArray
    end

    @testset "RandSVDRecipe respects array type" begin
        approx = RandSVD(compressor=Gaussian(compression_dim=s), power_its=0)
        recipe = complete_approximator(approx, A_mock)
        @test recipe.buffer isa MockArray
    end

    @testset "RangeFinderRecipe respects array type" begin
        approx = RangeFinder(compressor=Gaussian(compression_dim=s), power_its=0)
        recipe = complete_approximator(approx, A_mock)
        # range field is a placeholder until rapproximate! is called; check buffer type
        @test recipe.compressor.op isa MockArray
    end

end

end