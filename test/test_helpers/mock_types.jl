module MockTypes

using RandLinearAlgebra, LinearAlgebra
import RandLinearAlgebra: complete_compressor, update_compressor!
import LinearAlgebra: mul!

# ------------------------------------------------------------------
# Minimal stubs used in abstract-type interface tests
# ------------------------------------------------------------------

struct TestCompressor <: Compressor end

struct TestApproximator <: Approximator end

struct TestSolver <: Solver end

# ------------------------------------------------------------------
# TestCompressorRecipe — superset of fields needed by the compressor
# abstract tests (n_rows, n_cols for most; status for the update test).
#
# mul! and update_compressor! are intentionally NOT defined here:
# the abstract-type error tests verify those functions throw before
# any concrete implementation is provided.
# ------------------------------------------------------------------

mutable struct TestCompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
    status::Bool
end

function TestCompressorRecipe(n_rows::Int64, n_cols::Int64)
    return TestCompressorRecipe(n_rows, n_cols, false)
end

# ------------------------------------------------------------------
# TestApproximatorRecipe — superset of fields needed by the approximator
# abstract tests (n_rows, n_cols for adjoint/multiplication tests;
# code for interface/error tests).
#
# mul!, rapproximate!, and complete_approximator are NOT defined here
# for the same reason as above.
# ------------------------------------------------------------------

mutable struct TestApproximatorRecipe <: ApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
    code::Int64
end

function TestApproximatorRecipe(code::Int64)
    return TestApproximatorRecipe(0, 0, code)
end

function TestApproximatorRecipe(n_rows::Int64, n_cols::Int64)
    return TestApproximatorRecipe(n_rows, n_cols, 0)
end

# ------------------------------------------------------------------
# TestSolverRecipe — holds a code field used in solver interface tests.
# complete_solver and rsolve! are NOT defined here.
# ------------------------------------------------------------------

mutable struct TestSolverRecipe <: SolverRecipe
    code::Int64
end

# ------------------------------------------------------------------
# TestFullCompressor / TestFullCompressorRecipe
#
# A unified compressor mock that supports both Left (solver) and
# Right (approximator) cardinalities via multiple dispatch.
# All methods are defined here because the tests that use this type
# do not have throw-first error phases.
# ------------------------------------------------------------------

mutable struct TestFullCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
end

function TestFullCompressor(cardinality::Cardinality)
    return TestFullCompressor(cardinality, 5)
end

mutable struct TestFullCompressorRecipe <: CompressorRecipe
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

# 2-argument: approximator use (Right cardinality, compresses columns)
function complete_compressor(comp::TestFullCompressor, A::AbstractMatrix)
    n_cols = comp.compression_dim
    n_rows = size(A, 2)
    op = randn(n_rows, n_cols) ./ sqrt(n_cols)
    return TestFullCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

# 3-argument: solver use (Left cardinality, compresses rows)
function complete_compressor(
    comp::TestFullCompressor,
    A::AbstractMatrix,
    _b::AbstractVector,
)
    n = size(A, 1)
    op = randn(comp.compression_dim, n) ./ sqrt(n)
    return TestFullCompressorRecipe(comp.cardinality, comp.compression_dim, n, op)
end

# Left multiplication: C = S * A
function mul!(
    C::AbstractArray,
    S::TestFullCompressorRecipe,
    A::AbstractArray,
    alpha::Number,
    beta::Number,
)
    return mul!(C, S.op, A, alpha, beta)
end

# Right multiplication: C = A * S
function mul!(
    C::AbstractArray,
    A::AbstractArray,
    S::TestFullCompressorRecipe,
    alpha::Float64,
    beta::Float64,
)
    return mul!(C, A, S.op, alpha, beta)
end

function update_compressor!(
    comp::TestFullCompressorRecipe,
    _x::AbstractVector,
    A::AbstractMatrix,
    _b::AbstractVector,
)
    n = size(A, 1)
    comp.op = randn(comp.n_rows, n) ./ sqrt(n)
    return nothing
end

# ------------------------------------------------------------------
# TestFullSolverRecipe — a unified solver-recipe mock holding the
# union of fields read by the ErrorMethods `compute_error` routines:
# `compressor` (CompressedResidual), `vec_view`/`mat_view`/`solution_vec`
# (CompressedResidual, FullResidual), and `residual_vec` (LSGradient).
#
# The all-optional keyword constructor lets each error test populate
# only the fields it exercises. `vec_view`/`mat_view` are typed loosely
# (AbstractVector/AbstractMatrix) so the defaults can be plain empty
# arrays; the tests still pass real `view(...)` objects.
# ------------------------------------------------------------------

mutable struct TestFullSolverRecipe <: SolverRecipe
    compressor::AbstractMatrix
    vec_view::AbstractVector
    mat_view::AbstractMatrix
    solution_vec::AbstractVector
    residual_vec::AbstractVector
end

function TestFullSolverRecipe(;
    compressor::AbstractMatrix = zeros(0, 0),
    vec_view::AbstractVector = zeros(0),
    mat_view::AbstractMatrix = zeros(0, 0),
    solution_vec::AbstractVector = zeros(0),
    residual_vec::AbstractVector = zeros(0),
)
    return TestFullSolverRecipe(compressor, vec_view, mat_view, solution_vec, residual_vec)
end

export TestCompressor,
    TestCompressorRecipe,
    TestApproximator,
    TestApproximatorRecipe,
    TestSolver,
    TestSolverRecipe,
    TestFullCompressor,
    TestFullCompressorRecipe,
    TestFullSolverRecipe

end
