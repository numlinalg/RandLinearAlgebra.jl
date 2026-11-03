"""
    MockCompressors

Shared mock `Compressor` and `CompressorRecipe` types for the compressor tests.

The types are grouped into tiers by behavior. The **inert** tier holds fields only and
defines no methods, so the abstract fallbacks in `RandLinearAlgebra` throw
`ArgumentError`. Working tiers define just enough interface to
test one layer.

!!! note
    This file is included from `preamble.jl`, which is included within individual 
    modules fore each test file, so each test module gets its own copy of
    `MockCompressors`. Methods defined on a mock in one test file therefore never leak into
    another test file.
"""
module MockCompressors
using RandLinearAlgebra
import LinearAlgebra: mul!

export MockCompressor, MockCompressorRecipe
export MockMul5CompressorRecipe, MockMul3CompressorRecipe

"""
    MockCompressor <: Compressor

Inert `Compressor` with no fields and no methods. Calling `complete_compressor` on it
reaches the abstract fallback and throws an `ArgumentError`.
"""
struct MockCompressor <: Compressor end

"""
    MockCompressorRecipe <: CompressorRecipe

Inert `CompressorRecipe` that holds only its dimensions and defines no methods. Use it to
test functionality that depends only on `n_rows` and `n_cols` (`size`, `adjoint`, dimension
checks), and to check that `update_compressor!` and `mul!` fallbacks throw an
`ArgumentError`.

# Fields
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
"""
mutable struct MockCompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
end

"""
    MockMul5CompressorRecipe <: CompressorRecipe

Working-tier `CompressorRecipe` that defines only the two five-argument `mul!` methods. The
library's three-argument, adjoint and binary `*` methods all reduce to these, so they can be
tested through this type. A left multiplication fills the output with `-1` and a right
multiplication fills it with `-2`.

# Fields
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
"""
mutable struct MockMul5CompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
end

"""
    mul!(C, S::MockMul5CompressorRecipe, A, alpha, beta)

Left five-argument multiplication. Fills `C` with `-1`.
"""
function mul!(
    C::AbstractArray,
    S::MockMul5CompressorRecipe,
    A::AbstractArray,
    alpha::Number,
    beta::Number,
)
    fill!(C, -1)
end

"""
    mul!(C, A, S::MockMul5CompressorRecipe, alpha, beta)

Right five-argument multiplication. Fills `C` with `-2`.
"""
function mul!(
    C::AbstractArray,
    A::AbstractArray,
    S::MockMul5CompressorRecipe,
    alpha::Number,
    beta::Number,
)
    fill!(C, -2)
end

"""
    MockMul3CompressorRecipe <: CompressorRecipe

Working-tier `CompressorRecipe` that defines only the two three-argument `mul!` methods,
which are more specific than the library's three-argument fallbacks. The five-argument
methods are never reached. A left multiplication fills the output with `1` and a right
multiplication fills it with `2`.

# Fields
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
"""
mutable struct MockMul3CompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
end

"""
    mul!(C, S::MockMul3CompressorRecipe, A)

Left three-argument multiplication. Fills `C` with `1`.
"""
function mul!(C::AbstractArray, S::MockMul3CompressorRecipe, A::AbstractArray)
    fill!(C, 1)
end

"""
    mul!(C, A, S::MockMul3CompressorRecipe)

Right three-argument multiplication. Fills `C` with `2`.
"""
function mul!(C::AbstractArray, A::AbstractArray, S::MockMul3CompressorRecipe)
    fill!(C, 2)
end

end
