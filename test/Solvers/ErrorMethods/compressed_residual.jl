module compressed_residual_error 
using Test, RandLinearAlgebra, Random
import LinearAlgebra: mul!, norm
using ..FieldTest
using ..ApproxTol
Random.seed!(1232)

###############################
# Test Solver Structures
###############################
mutable struct TestSolver <: Solver end

mutable struct TestSolverRecipe <: SolverRecipe
    compressor::AbstractMatrix
    vec_view::SubArray
    mat_view::SubArray
    solution_vec::AbstractVector
end


@testset "Compressed Residual" begin
    @testset "Compressed Residual: SolverError" begin
        # Verify Supertype
        @test supertype(CompressedResidual) == SolverError 

        # Verify fieldnames and types
        @test fieldnames(CompressedResidual) == ()
        @test fieldtypes(CompressedResidual) == ()
        # Verify the internal constructor

    end

    @testset "Compressed Residual: SolverErrorRecipe" begin
        # Verify Supertype
        @test supertype(CompressedResidualRecipe) ==  SolverErrorRecipe

        # Verify fieldnames and types
        @test fieldnames(CompressedResidualRecipe) == (:residual, :residual_view)
        @test fieldtypes(CompressedResidualRecipe) == (AbstractVector, SubArray)
    end

    @testset "Compressed Residual: Complete error" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64] 
            let n_rows = 4,
                n_cols = 3,
                comp_dim = 2,
                S = ones(type, n_rows, n_cols),
                A = ones(type, n_rows, n_cols),
                b = ones(type, n_rows),
                x = ones(type, n_cols),
                error_rec = complete_error(CompressedResidual(), TestSolver(), A, b)

                # Test the type
                @test typeof(error_rec) == CompressedResidualRecipe{
                    Vector{type}, 
                    SubArray{type, 1, Vector{type}, Tuple{UnitRange{Int64}}, true}
                }
                # Test type of residual vector
                @test eltype(error_rec.residual) == type
                # Test residual vector to be all zeros
                @test error_rec.residual == zeros(type, n_rows)
            end

        end

    end

    @testset "Compressed Residual: Compute Error" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64] 
            let n_rows = 4,
                n_cols = 3,
                comp_dim = 2,
                A = ones(type, n_rows, n_cols),
                S = ones(type, comp_dim, n_rows),
                b = ones(type, n_rows),
                x = ones(type, n_cols),
                Sb = S * b,
                SA = S * A
                solver_rec = TestSolverRecipe(
                    S,
                    view(Sb, 1:comp_dim),
                    view(SA, 1:comp_dim, 1:n_cols), 
                    x 
                )

                error_rec = complete_error(
                    CompressedResidual(), 
                    TestSolver(), 
                    A, 
                    b
                )

                # compute the error value
                err_val = compute_error(error_rec, solver_rec, A, b)
                # compute the residual
                res = Sb - SA * x
                # compute norm squared of residual
                @test norm(res) ≈ err_val
            end

        end

    end

end

end
