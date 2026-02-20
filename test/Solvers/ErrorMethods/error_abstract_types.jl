module solver_error_abstract_types
using Test, RandLinearAlgebra
import Random: seed!
using ..FieldTest
using ..ApproxTol
seed!(1232)

struct TestSolver <: Solver end
struct TestSolverRecipe <: SolverRecipe end
struct TestSolverError <: SolverError end
struct TestSolverErrorRecipe <: SolverErrorRecipe end
@testset "Solver Error Abstract Types" begin
    @test isdefined(Main, :SolverError)
    @test isdefined(Main, :SolverErrorRecipe)
end

# Test SolverError argment error
@testset "SolverError Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_error(TestSolverError(), TestSolver(), A, b)
    @test_throws ArgumentError compute_error(
        TestSolverErrorRecipe(), TestSolverRecipe(), x, A, b
    )
end

end
