# Mock 1: `MockCompressors` (dimension-only recipe)

Part of the [test mock consolidation](overview.md). Risk: low.

## Current status

- **State:** Increment 1 done and committed (`b74a3c76`, full suite passed). Increment 2
  migrations approved (the three new files pass; full suite not run): `Mul5`/`Mul3` `mul!` bodies filled, and
  `multiplication` split into `_errors`, `_5arg` and `_3arg` (original deleted). The
  `@test` count is unchanged (55).
- **Next step:** After approval, plan and stub increment 3 (`Update` tier, migrate `update`).
- **Last updated:** 2026-11-03

!!! warning "Before opening a PR"
    Delete `docs/src/dev/` and remove the `Development` entry from `pages` in
    `docs/make.jl`. These plan docs must not be merged.

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

## Increment 1 migration outline

Each file keeps its own module and `include("../test_helpers/preamble.jl")`, adds
`using .MockCompressors`, and replaces `TestCompressorRecipe` with `MockCompressorRecipe`.
No test checks change.

- `compressor_abstract_size.jl` (pilot): delete the local struct (lines 8-11). Construct
  `S = MockCompressorRecipe(s, n_rows)`.
- `compressor_abstract_adjoint.jl`: delete the local struct (lines 7-10). Rename the type in
  the four constructors and the four `typeof` checks, e.g.
  `CompressorAdjoint{MockCompressorRecipe}`.
- `compressor_abstract_dimchecks.jl`: delete the local `TestCompressorRecipe` (lines 7-10)
  only. `TestApproximatorRecipe` stays until Mock 2. Construct
  `S = MockCompressorRecipe(s, n_rows)`.
- `compressor_abstract_types.jl`: delete both local empty structs (lines 3-4). Replace
  `TestCompressor()` with `MockCompressor()`. The recipe now has fields, so replace each
  `TestCompressorRecipe()` with `MockCompressorRecipe(2, 2)`, matching the sizes of `A`.

## Increment 2 migration outline

Resolved: `Mul3` needs only the 3-arg methods. In `src/Compressors.jl` the 3-arg adjoint
methods and binary `*` call the 3-arg `mul!`, and the mock's 3-arg methods are more specific
than the library's 3-arg fallback, which is the only path to the 5-arg method.

Split `compressor_abstract_multiplication.jl`, then delete it. Each new file keeps its own
module (named after the file), includes the preamble, adds `using .MockCompressors`, and
defines `s_rows = 3` and `s_cols = 4`. The local struct and the zero-argument constructor are
dropped; tests construct `Mock*Recipe(s_rows, s_cols)` directly.

- `..._multiplication_errors.jl` (old lines 1-57): `MockCompressorRecipe(s_rows, s_cols)` in
  place of each `TestCompressorRecipe()`. Checks are unchanged.
- `..._multiplication_5arg.jl` (old lines 59-195): `MockMul5CompressorRecipe`. Drop the two
  `mul!` definitions, which now live in the mock, and keep `c_rows`, `c_cols`, `a_rows`,
  `a_cols`. The testset still checks the 3-arg and binary paths, which reduce to the 5-arg
  methods.
- `..._multiplication_3arg.jl` (old lines 197-289): `MockMul3CompressorRecipe`. Same
  parameters. Drop the two `mul!` definitions.

Because the 5-arg file no longer defines 3-arg methods later in the same module, the file
split removes the dependence on top-level execution order.

## Status checklist

- [x] Increment 1: inert tier + `size`, `adjoint`, `dimchecks`, `types`
  - [x] stub + migration outline approved
  - [x] migrations approved (full suite passed; committed `b74a3c76`)
- [x] Increment 2: `Mul5`, `Mul3` tiers + split `multiplication`
  - [x] stub + migration outline approved
  - [x] migrations approved (the three new multiplication files pass; full suite not run)
- [ ] Increment 3: `Update` tier + `update`
  - [ ] stub + migration outline approved
  - [ ] migrations approved
- [ ] Before PR: delete `docs/src/dev/` and the `Development` entry in `docs/make.jl`
