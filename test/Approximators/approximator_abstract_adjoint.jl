module approximators_abstract_adjoint
using Test, RandLinearAlgebra
import LinearAlgebra: mul!

#####################
# Testing Parameters
##################### 
mutable struct TestApproximatorRecipe <: ApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
end
s_rows = 3
s_cols = 4

@testset "Approximator Recipe Adjoints" begin
    # Adjoint 
    let approx = TestApproximatorRecipe(s_rows, s_cols)

        approx_adjoint = adjoint(approx) 
        @test typeof(approx_adjoint) == ApproximatorAdjoint{TestApproximatorRecipe}
        @test size(approx_adjoint) == (s_cols, s_rows)
    end

    # Adjoint of Adjoint 
    let approx = TestApproximatorRecipe(s_rows, s_cols),
        approx_adjoint = adjoint(approx)

        approx_adjoint_adjoint = adjoint(approx_adjoint)
        @test approx_adjoint_adjoint == approx 
        @test typeof(approx_adjoint_adjoint) == TestApproximatorRecipe
    end

    # Transpose 
    let approx = TestApproximatorRecipe(s_rows, s_cols)

        approx_transpose = transpose(approx) 
        @test typeof(approx_transpose) == ApproximatorAdjoint{TestApproximatorRecipe}
        @test size(approx_transpose) == (s_cols, s_rows)
    end 

    # Transpose of Transpose 
    let approx = TestApproximatorRecipe(s_rows, s_cols),
        approx_transpose = transpose(approx)

        approx_transpose_transpose = transpose(approx_transpose)
        @test approx_transpose_transpose == approx 
        @test typeof(approx_transpose_transpose) == TestApproximatorRecipe
    end
end #End Testset 

end #End Module 