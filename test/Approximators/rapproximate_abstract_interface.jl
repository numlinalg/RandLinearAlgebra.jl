module rapproximate_abstract_interface
using Test, RandLinearAlgebra 
import RandLinearAlgebra: complete_approximator, rapproximate!

#############################
# Initial Testing Parameters
#############################
struct TestApproximator <: Approximator end
mutable struct TestApproximatorRecipe <: ApproximatorRecipe
    code::Int64
end
complete_approximator(approx::TestApproximator, A::AbstractMatrix) = 
    TestApproximatorRecipe(1)

########################################
# Tests for rapproximator Errors 
########################################
@testset "rapproximate Errors" begin 

    let A = ones(2, 2), 
        approx = TestApproximatorRecipe(1)

        @test_throws ArgumentError rapproximate!(approx, A)
    end

    let A = ones(2, 2),
        approx = TestApproximator()

        # Error thrown because rapproximate! not defined for Recipe 
        @test_throws ArgumentError rapproximate(approx, A)
    end
end

########################################
# Tests for rapproximator 
########################################
# Updated Testing Parameters 
function rapproximate!(approx::TestApproximatorRecipe, A::AbstractMatrix)
    approx.code = 2
    return nothing
end
    

@testset "rapproximate Interface" begin

    let A=ones(2, 2),
        approx = TestApproximatorRecipe(1)

        @test isnothing(rapproximate!(approx, A))
        @test approx.code == 2
    end

    let A=ones(2, 2),
        approx = TestApproximator()

        
        approx_recipe = rapproximate(approx, A)
        @test approx_recipe isa TestApproximatorRecipe
        @test approx_recipe.code == 2
    end

end
end 