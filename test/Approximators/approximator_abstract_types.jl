module approximators_abstract_types
using Test, RandLinearAlgebra
import RandLinearAlgebra: complete_approximator


#############################
# Initial Testing Parameters
#############################
struct TestApproximator <: Approximator end
struct TestApproximatorRecipe <: ApproximatorRecipe
    code::Int64
end

########################################
# Tests for Approximator Abstract Types
########################################
@testset "Approximator Abstract Types" begin
    @test isdefined(Main, :Approximator)
    @test isdefined(Main, :ApproximatorRecipe)
    @test isdefined(Main, :ApproximatorAdjoint)
end

@testset "Approximator Completion Errors" begin
    A = ones(2,2)
        
    @test_throws ArgumentError complete_approximator(TestApproximator(), A)
end

#############################
# Update Testing Parameters 
#############################
complete_approximator(approx::TestApproximator, A::AbstractMatrix) = 
    TestApproximatorRecipe(1)

@testset "Approximator Completion" begin
    A =ones(2, 2)

    approx = complete_approximator(TestApproximator(), A) 
    @test approx isa TestApproximatorRecipe
    @test approx.code == 1
end

end
