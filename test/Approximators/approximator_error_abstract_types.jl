module approximator_error_abstract_types
using Test, RandLinearAlgebra
import RandLinearAlgebra: complete_approximator_error, compute_approximator_error!

#############################
# Initial Testing Parameters
#############################
struct TestApproximator <: Approximator end
struct TestApproximatorRecipe <: ApproximatorRecipe end 
struct TestApproximatorError <: ApproximatorError end
mutable struct TestApproximatorErrorRecipe <: ApproximatorErrorRecipe
    code::Int64
end


@testset "ApproximatorError Abstract Types" begin
    @test isdefined(Main, :ApproximatorError)
    @test isdefined(Main, :ApproximatorErrorRecipe)
end

@testset "ApproximatorErrorRecipe Completion Errors" begin 
    # Parameters 
    error = TestApproximatorError()
    approx = TestApproximatorRecipe() 
    A = ones(2, 2) 

    # Test 
    @test_throws ArgumentError complete_approximator_error(error, approx, A)
end

@testset "ApproximatorError Compute Errors" begin 
    # Parameters 
    error = TestApproximatorErrorRecipe(1)
    approx = TestApproximatorRecipe()
    A = ones(2, 2)

    # Test 
    @test_throws ArgumentError compute_approximator_error!(error, approx, A)
    @test error.code == 1
end

#############################
# Updated Testing Parameters
#############################
complete_approximator_error(
    error::TestApproximatorError, 
    approx::TestApproximatorRecipe, 
    A::AbstractMatrix
) = TestApproximatorErrorRecipe(1)
compute_approximator_error!(
    error::TestApproximatorErrorRecipe, 
    approx::TestApproximatorRecipe, 
    A::AbstractMatrix
) = begin
    error.code = 2
    return 3 
end
@testset "ApproximatorError Compute" begin
    # Check if completition is done correctly  
    let error = TestApproximatorError(),
        approx = TestApproximatorRecipe(),
        A = ones(2, 2)

        error_recipe = complete_approximator_error(error, approx, A)
        @test error_recipe isa TestApproximatorErrorRecipe
        @test error_recipe.code == 1
    end 

    # Check if compute is done correctly 
    let error = TestApproximatorErrorRecipe(1),
        approx = TestApproximatorRecipe(),
        A = ones(2, 2)

        result = compute_approximator_error!(error, approx, A)
        @test result == 3
        @test error.code == 2
    end

    # Now test the compute_approximator_error function
    let error = TestApproximatorError(), 
        approx = TestApproximatorRecipe(),
        A = ones(2, 2)

        # This should return the error code 3
        result = compute_approximator_error(error, approx, A)
        @test result == 3
    end
end
end 