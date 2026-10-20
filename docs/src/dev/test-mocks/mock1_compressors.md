# Mock 1: `MockCompressors` (dimension-only recipe)

Part of the [test mock consolidation](overview.md). Risk: low.

## Current status

- **State:** Not started. The plan is approved; no mock definition, preamble change or
  migration has been written.
- **Next step:** Increment 1 stub. Write `MockCompressor` and `MockCompressorRecipe` with
  docstrings in `test/test_helpers/mocks/compressors.jl`, add the preamble include, and
  outline the migration of `size`, `adjoint`, `dimchecks` and `types`. Then stop for review.
- **Last updated:** 2026-10-20

## Goal

Share the `TestCompressorRecipe` that six `Compressors/compressor_abstract_*.jl` files each
define locally. Every copy holds only `n_rows` and `n_cols` (plus `status` in `update`) and
differs only in which methods the file adds. This mock is also the first use of tiers.

## Types

| Type | Tier | Fields | Methods defined | Replaces |
|:-----|:-----|:-------|:----------------|:---------|
| `MockCompressor` | inert | none | none | `TestCompressor` in `types` |
| `MockCompressorRecipe` | inert | mutable `n_rows`, `n_cols` | none | `types`, `adjoint`, `size`, `dimchecks`, error section of `multiplication` |
| `MockMul5CompressorRecipe` | 5-arg `mul!` | same | left 5-arg `mul!` fills `-1`, right fills `-2` | 5-arg section of `multiplication` |
| `MockMul3CompressorRecipe` | 3-arg `mul!` | same | left 3-arg `mul!` fills `1`, right fills `2` | 3-arg section of `multiplication` |
| `MockUpdateCompressorRecipe` | update | adds `status::Bool` | 1-arg `update_compressor!` sets `status` | `update` |

## Files

| File | Purpose | Change | Est. lines |
|:-----|:--------|:-------|:-----------|
| `test/test_helpers/mocks/compressors.jl` | `MockCompressors` module | New | ~85 |
| `test/test_helpers/preamble.jl` | Shared preamble | Add `include("mocks/compressors.jl")` | +1 |
| `Compressors/compressor_abstract_size.jl` | Size tests | Drop local struct, `using .MockCompressors` (pilot) | -6 |
| `Compressors/compressor_abstract_adjoint.jl` | Adjoint tests | Same | -6 |
| `Compressors/compressor_abstract_dimchecks.jl` | Dimension checks | Same; the local `TestApproximatorRecipe` stays until Mock 2 | -5 |
| `Compressors/compressor_abstract_types.jl` | Fallback errors | Use `MockCompressor`, `MockCompressorRecipe` | ~0 |
| `Compressors/compressor_abstract_multiplication.jl` | Mixed-tier mul tests | Split into the three files below, then delete | n/a |
| `Compressors/compressor_abstract_multiplication_errors.jl` | Fallback `ArgumentError`s | New, from lines 1-57 | ~60 |
| `Compressors/compressor_abstract_multiplication_5arg.jl` | 5-arg dispatch and binary `*` | New, from lines 59-195 | ~140 |
| `Compressors/compressor_abstract_multiplication_3arg.jl` | 3-arg dispatch and binary `*` | New, from lines 197-289 | ~95 |
| `Compressors/compressor_abstract_update.jl` | Update dispatch | Use `MockUpdateCompressorRecipe` | -12 |

## Functions (in `compressors.jl`)

| Definition | Reason | Est. lines |
|:-----------|:-------|:-----------|
| Module header, imports, exports | Scope for the mock family | 8 |
| `MockCompressor` | Inert technique stub | 6 |
| `MockCompressorRecipe` | Inert tier, shared by 5 test files | 12 |
| `MockMul5CompressorRecipe` + two `mul!` | Layer 2 of the mul dispatch ladder | 22 |
| `MockMul3CompressorRecipe` + two `mul!` | Layer 3 of the mul dispatch ladder | 22 |
| `MockUpdateCompressorRecipe` + `update_compressor!` | Update dispatch | 18 |

## Increments (each a commit point)

1. Module, preamble include, inert tier. Migrate `size` (pilot), `adjoint`, `dimchecks`,
   `types`.
2. `Mul5` and `Mul3` tiers. Split `multiplication`.
3. `Update` tier. Migrate `update`.

Open point for increment 2: confirm `Mul3` needs only the 3-arg methods. The 3-arg section
appears not to call the 5-arg `mul!`, but this is unverified.

## Status checklist

- [ ] Increment 1: inert tier + `size`, `adjoint`, `dimchecks`, `types`
  - [ ] stub + migration outline approved
  - [ ] migrations approved
- [ ] Increment 2: `Mul5`, `Mul3` tiers + split `multiplication`
  - [ ] stub + migration outline approved
  - [ ] migrations approved
- [ ] Increment 3: `Update` tier + `update`
  - [ ] stub + migration outline approved
  - [ ] migrations approved
