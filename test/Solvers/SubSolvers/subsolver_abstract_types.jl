module sub_solver_abstract_types
include("../../test_helpers/preamble.jl")
struct TestSubSolver <: SubSolver end
struct TestSubSolverRecipe <: SubSolverRecipe end

@testset "Sub-Solver Abstract Types" begin
    @test isdefined(Main, :SubSolver)
    @test isdefined(Main, :SubSolverRecipe)
end

@testset "Sub-Solver LAPACK Helpers" begin
    @test RandLinearAlgebra._qt_char(Float64) == 'T'
    @test RandLinearAlgebra._qt_char(ComplexF64) == 'C'
end

@testset "Sub-Solver Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_sub_solver(TestSubSolver(), A)
    @test_throws ArgumentError complete_sub_solver(TestSubSolver(), A, b)
    @test_throws ArgumentError update_sub_solver!(TestSubSolverRecipe(), A)
    @test_throws ArgumentError ldiv!(x, TestSubSolverRecipe(), b)
end
end
