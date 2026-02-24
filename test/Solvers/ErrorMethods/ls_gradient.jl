module ls_gradient_error 

using Test, RandLinearAlgebra, Random
import LinearAlgebra: mul!, norm
using ..FieldTest
using ..ApproxTol
Random.seed!(1232)

mutable struct TestSolver <: Solver end

mutable struct TestSolverRecipe <: SolverRecipe
    residual_vec::AbstractVector
end

@testset "LS Gradient" begin

    @testset "LS Gradient: SolverError" begin
        # Verify Supertype
        @test supertype(LSGradient) == SolverError 

        # Verify fieldnames and types
        @test fieldnames(LSGradient) == ()
        @test fieldtypes(LSGradient) == ()
        # Verify the internal constructor

    end

    @testset "LS Gradient: SolverErrorRecipe" begin
        # Verify Supertype
        @test supertype(LSGradientRecipe) ==  SolverErrorRecipe

        # Verify fieldnames and types
        @test fieldnames(LSGradientRecipe) == (:gradient,)
        @test fieldtypes(LSGradientRecipe) == (AbstractVector,)
    end

    @testset "Residual: Complete error" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64] 
            let n_rows = 4,
                n_cols = 6,
                A = rand(type, n_rows, n_cols),
                b = rand(type, n_rows)

                error_rec = complete_error(LSGradient(), TestSolver(), A, b)

                # Test the type
                @test typeof(error_rec) == LSGradientRecipe{typeof(b)}
                
                # Test type of residual vector
                @test eltype(error_rec.gradient) == type
                
                # Test residual vector to be all zeros
                @test error_rec.gradient == zeros(type, n_cols)
            end

        end

    end

    @testset "Residual: Compute Error" begin
        for type in [Float32, Float64] 
            let n_rows = 4,
                n_cols = 6,
                A = rand(type, n_rows, n_cols),
                b = rand(type, n_rows),
                solver = TestSolver(),
                x = rand(type, n_cols),
                r = b-A*x,
                solver_rec = TestSolverRecipe(r),
                error_rec = complete_error(LSGradient(), TestSolver(), A, b)

                # compute the error value
                err_val = compute_error(error_rec, solver_rec, A, b)

                # Verify gradient calculation 
                @test error_rec.gradient ≈ -A'*(b-A*x)

                # compute norm squared of residual
                @test norm(-A'*(b-A*x)) ≈ err_val
            end
        end
    end

end

end
