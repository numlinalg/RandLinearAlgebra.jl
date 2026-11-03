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

export MockCompressor, MockCompressorRecipe

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

end
