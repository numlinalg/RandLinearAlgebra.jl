module solver_abstract_types
using Test, RandLinearAlgebra
import RandLinearAlgebra: complete_solver, rsolve!

###################################
# Initial Testing Parameters   
###################################
struct TestSolver <: Solver end
mutable struct TestSolverRecipe <: SolverRecipe 
    code::Int64
end

@testset "Solver Abstract Types" begin
    @test isdefined(Main, :Solver)
    @test isdefined(Main, :SolverRecipe)
end

@testset "Solver Interfaces Errors" begin
    A = ones(2, 2)
    b = ones(2)
    x = ones(2)

    @test_throws ArgumentError complete_solver(TestSolver(), x, A, b)
    @test_throws ArgumentError rsolve!(TestSolverRecipe(1), x, A, b)
    @test_throws ArgumentError rsolve!(TestSolver(), x, A, b)
end

###################################
# Updated Testing Parameters   
###################################
complete_solver(
    solver::TestSolver, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
) = TestSolverRecipe(1)
rsolve!(
    solver::TestSolverRecipe,
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
) = begin
    solver.code = 2
    return nothing
end

@testset "Solver Interfaces" begin

    # Check complete_solver works as expected 
    let A = ones(2, 2),
        b = ones(2),
        x = ones(2),
        solver = TestSolver()

        solver_recipe = complete_solver(solver, x, A, b)
        @test solver_recipe isa TestSolverRecipe
        @test solver_recipe.code == 1
    end
    
    # Check rsolve! with solver recipe works as expected 
    let A = ones(2, 2),
        b = ones(2),
        x = ones(2),
        solver_recipe = TestSolverRecipe(1)

        rsolve!(solver_recipe, x, A, b)
        @test solver_recipe.code == 2
    end

    #Check rsolve! with solver ingredients works as expected 
    let A = ones(2, 2),
        b = ones(2),
        x = ones(2),
        solver = TestSolver()

        x, solver_recipe = rsolve!(solver, x, A, b)
        @test solver_recipe.code == 2
    end
end

end
