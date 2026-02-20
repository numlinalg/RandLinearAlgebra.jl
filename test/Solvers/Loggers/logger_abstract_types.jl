module logger_abstract_types
using Test, RandLinearAlgebra
using ..FieldTest
using ..ApproxTol
struct TestLogger <: Logger end
struct TestLoggerRecipe <: LoggerRecipe end

@testset "Logger Abstract Types" begin
    @test isdefined(Main, :Logger)
    @test isdefined(Main, :LoggerRecipe)
end

@testset "Logger Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)

    @test_throws ArgumentError complete_logger(TestLogger())
    @test_throws ArgumentError update_logger!(TestLoggerRecipe(), 1.0, 1)
    @test_throws ArgumentError reset_logger!(TestLoggerRecipe())
end
end
