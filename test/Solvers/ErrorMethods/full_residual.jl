module residual_error 
using Test, RandLinearAlgebra, Random
import LinearAlgebra: mul!, norm
using ..FieldTest
using ..ApproxTol
Random.seed!(1232)

mutable struct TestSolver <: Solver end

mutable struct TestSolverRecipe <: SolverRecipe
    solution_vec::AbstractVector
end

@testset "Full Residual" begin
    @testset "Full Residual: SolverError" begin
        # Verify Supertype
        @test supertype(FullResidual) == SolverError 

        # Verify fieldnames and types
        @test fieldnames(FullResidual) == ()
        @test fieldtypes(FullResidual) == ()
        # Verify the internal constructor

    end

    @testset "Full Residual: SolverErrorRecipe" begin
        # Verify Supertype
        @test supertype(FullResidualRecipe) ==  SolverErrorRecipe

        # Verify fieldnames and types
        @test fieldnames(FullResidualRecipe) == (:residual,)
        @test fieldtypes(FullResidualRecipe) == (AbstractVector,)
    end

    @testset "Residual: Complete error" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64] 
            let n_rows = 4,
                n_cols = 3,
                A = rand(type, n_rows, n_cols),
                b = rand(type, n_rows),
                x = rand(type, n_cols),
                solver_rec = TestSolverRecipe(x)
                error_rec = complete_error(FullResidual(), TestSolver(), A, b)

                # Test the type
                @test typeof(error_rec) == FullResidualRecipe{typeof(b)}
                # Test type of residual vector
                @test eltype(error_rec.residual) == type
                # Test residual vector to be all zeros
                @test error_rec.residual == zeros(type, n_rows)
            end

        end

    end

    @testset "Residual: Compute Error" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64] 
            let n_rows = 4,
                n_cols = 3,
                A = rand(type, n_rows, n_cols),
                b = rand(type, n_rows),
                x = rand(type, n_cols),
                solver_rec = TestSolverRecipe(x),
                solver = TestSolver(),

                error_rec = complete_error(FullResidual(), TestSolver(), A, b)

                # compute the error value
                err_val = compute_error(error_rec, solver_rec, A, b)
                # compute the residual
                res = b - A * x
                # compute norm squared of residual
                @test norm(res) ≈ err_val
            end

        end

    end

end

end
