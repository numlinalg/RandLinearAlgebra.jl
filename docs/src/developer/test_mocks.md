# Test Mocks

Many tests need a small stand-in for a `Compressor` or `CompressorRecipe`. Instead of each
file defining its own, shared mock types live in `test/test_helpers/mocks/`.

!!! note
    Only the compressor mocks exist so far. Other families (approximators, solvers and so on)
    still use types defined locally in each test file.

## Where mocks live and how they load

- Each family has one file, `mock_<family>.jl`, that defines one module, for example
  `MockCompressors` in `mock_compressors.jl`. The module exports `Mock*` names.
- `preamble.jl` includes every mock file, so each test module has its own copy.
- A test opts in with `using .MockCompressors`.

Because a mock is copied into each test module, a method defined on a mock in one file never
affects another file.

## Tiers

A family offers types by behavior. A test uses the lowest tier that exercises what it checks.

- **Inert:** fields only, no methods. Calls fall through to the abstract fallbacks in
  `RandLinearAlgebra`, which throw `ArgumentError`. Use inert types to test the fallbacks and
  anything that depends only on the fields, such as `size`.
- **Working:** just enough methods to test one layer. For example, a type that defines only
  `mul!` with five arguments lets the three-argument, adjoint and binary `*` methods be
  tested through it.

If one test file needs two tiers, split the file instead of defining extra methods partway
through it.

## Naming

- `Mock<Thing>` is the inert tier.
- `Mock<Tier><Thing>` is a working tier, for example `MockMul5CompressorRecipe`.

## `MockCompressors`

All recipe types are mutable, with `n_rows::Int64` and `n_cols::Int64`.

| Type | Tier | Extra fields | Methods defined | Used by |
|:-----|:-----|:-------------|:----------------|:--------|
| `MockCompressor` | inert | none | none | `compressor_abstract_types` |
| `MockCompressorRecipe` | inert | none | none | `compressor_abstract_{types,size,adjoint,dimchecks,multiplication_errors}` |
| `MockMul5CompressorRecipe` | 5-argument `mul!` | none | left `mul!` fills `-1`, right fills `-2` | `compressor_abstract_multiplication_5arg` |
| `MockMul3CompressorRecipe` | 3-argument `mul!` | none | left `mul!` fills `1`, right fills `2` | `compressor_abstract_multiplication_3arg` |
| `MockUpdateCompressorRecipe` | update | `status::Bool` | one-argument `update_compressor!` sets `status` and returns `true` | `compressor_abstract_update` |

`MockUpdateCompressorRecipe` has no default for `status`. A test passes it explicitly, for
example `MockUpdateCompressorRecipe(3, 4, false)`, and checks it before and after the update.

## Adding a mock

1. Add the type to the family's file, or create `mock_<family>.jl` with a new module and
   include it from `preamble.jl`.
2. Give every type and method a docstring that states its tier and what it does.
3. Add a row to the table above.
4. Render the docstrings. In `docs/make.jl`, `include` the new mock file and add its module to
   `modules = [...]`. Then add a `@docs` block for it in the API section below. Documenter
   reports any docstring in a listed module that no page includes.
5. Replace the local copies in the test files with `using .Mock<Family>`. Test checks should
   not change.

Run the full suite afterwards, since the preamble change affects every test file.

## API

### Types

```@docs
MockCompressors
MockCompressors.MockCompressor
MockCompressors.MockCompressorRecipe
MockCompressors.MockMul5CompressorRecipe
MockCompressors.MockMul3CompressorRecipe
MockCompressors.MockUpdateCompressorRecipe
```

### Methods

```@docs
MockCompressors.mul!(::AbstractArray, ::MockCompressors.MockMul5CompressorRecipe, ::AbstractArray, ::Number, ::Number)
MockCompressors.mul!(::AbstractArray, ::AbstractArray, ::MockCompressors.MockMul5CompressorRecipe, ::Number, ::Number)
MockCompressors.mul!(::AbstractArray, ::MockCompressors.MockMul3CompressorRecipe, ::AbstractArray)
MockCompressors.mul!(::AbstractArray, ::AbstractArray, ::MockCompressors.MockMul3CompressorRecipe)
MockCompressors.update_compressor!(::MockCompressors.MockUpdateCompressorRecipe)
```
