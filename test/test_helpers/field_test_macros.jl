module FieldTest
using RandLinearAlgebra, Test

LoggerFields = Dict(
    :error => Real,
    :threshold_info => Union{Float64,Tuple},
    :converged => Bool,
    :hist => Vector{Float64},
    :stopping_criterion => Function,
)

"""
    @test_logger(type)

Macro for implementing Logger sub-routines. It checks that every LoggerRecipe includes the
fields `error::Real`, `threshold_info::Union{Float64, Tuple}`, `converged::Bool`,
`hist::Vector{Float64}`, and `stopping_criterion::Function` to ensure a common interface.
"""
macro test_logger(type)
    expr = quote
        @testset verbose = true "Logger: $(string($(esc(type))))" begin
            # Test the super type
            @test supertype($(esc(type))) == LoggerRecipe

            # Test the field names and types
            for (fname, ftype) in LoggerFields
                @test fname in fieldnames($(esc(type)))
                @test fieldtype($(esc(type)), fname) <: ftype
            end
        end
    end

    return expr
end

CompressorFields = Dict(:n_rows => Int64, :n_cols => Int64)

"""
    @test_comprtessor(type)

Macro for implementing Compressor routines. It checks that every `CompressorRecipe`
includes the fields `n_rows::Int64` and `n_cols::Int64` to
ensure a common interface.
"""
macro test_compressor(type)
    expr = quote
        @testset verbose = true "Compressor: $(string($(esc(type))))" begin
            # Test the super type
            @test supertype($(esc(type))) == CompressorRecipe

            # Test the field names and types
            for (fname, ftype) in CompressorFields
                @test fname in fieldnames($(esc(type)))
                @test fieldtype($(esc(type)), fname) <: ftype
            end
        end
    end

    return expr
end

ProjectionSolverFields = Dict(
    :mat_view => SubArray,
    :solution_vec => AbstractVector,
    :update_vec => AbstractVector,
    :compressed_mat => AbstractMatrix,
    :sub_solver => SubSolverRecipe,
    :error => SolverErrorRecipe,
    :S => CompressorRecipe,
    :log => LoggerRecipe,
)

"""
    @test_projection_solver(type)

Macro for implementing projection solver routines, such as Kaczmarz and coordinate descent.
It checks that every SolverRecipe  includes the fields
`mat_view::SubArray`, `solution_vec::AbstractVector`, `update_vec::AbstractVector`,
`compressed_mat::AbstractMatrix`, `sub_solver::SubSolverRecipe`, `error::SolverErrorRecipe`,
`S::CompressorRecipe`, and `log::LoggerRecipe` to ensure a common interface.
"""
macro test_projection_solver(type)
    expr = quote
        @testset verbose = true "Projection Solver: $(string($(esc(type))))" begin
            # Test the super type
            @test supertype($(esc(type))) == SolverRecipe

            # Test the field names and types
            for (fname, ftype) in ProjectionSolverFields
                @test fname in fieldnames($(esc(type)))
                @test fieldtype($(esc(type)), fname) <: ftype
            end
        end
    end

    return expr
end

RangeApproximatorFields = Dict(
    :n_cols => Int64,
    :n_rows => Int64,
    :power_its => Int64,
    :orthogonalize => Bool,
    :compressor => CompressorRecipe,
)

"""
    @test_range_approximator(type)

Macro for implementing Range Approximator routines, such as the Randomized Range Finder and
the Randomized SVD. It checks that every ApproximatorRecipe  includes the fields
`range::AbstractMatrix`, `power_its::Int64`, `rand_subspace::Bool`,
`compressor::CompressorRecipe`, `n_rows::Int64`, and `n_cols::Int64` 
to ensure a common interface.
"""
macro test_range_approximator(type)
    expr = quote
        @testset verbose = true "Range Approximator: $(string($(esc(type))))" begin
            # Test the super type
            @test supertype($(esc(type))) <: ApproximatorRecipe

            # Test the field names and types
            for (fname, ftype) in RangeApproximatorFields
                @test fname in fieldnames($(esc(type)))
                @test fieldtype($(esc(type)), fname) <: ftype
            end
        end
    end

    return expr
end

export @test_projection_solver, @test_compressor, @test_logger, @test_range_approximator
end
