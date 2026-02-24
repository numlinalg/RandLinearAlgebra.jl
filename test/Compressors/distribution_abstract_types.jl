module distribution_abstract_types
using Test, RandLinearAlgebra
struct TestDistribution <: Distribution end
struct TestDistributionRecipe <: DistributionRecipe end

@testset "Distribution Abstract Types" begin
    @test isdefined(Main, :Distribution)
    @test isdefined(Main, :DistributionRecipe)
end

@testset "Distribution Argument Errors" begin
    A = ones(2, 2)
    b = ones(2)
    x = ones(2)

    @test_throws ArgumentError complete_distribution(TestDistribution(), A)
    @test_throws ArgumentError complete_distribution(TestDistribution(), A, b)
    @test_throws ArgumentError complete_distribution(TestDistribution(), x, A, b)
    @test_throws ArgumentError update_distribution!(TestDistributionRecipe(), A)
    @test_throws ArgumentError update_distribution!(TestDistributionRecipe(), A, b)
    @test_throws ArgumentError update_distribution!(TestDistributionRecipe(), x, A, b)
    @test_throws ArgumentError sample_distribution!(x, TestDistributionRecipe())
end

end