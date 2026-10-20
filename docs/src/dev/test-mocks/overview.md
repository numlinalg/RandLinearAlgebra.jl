# Test Mock Consolidation: Overview

## Goal

Replace the per-file `Test*` stub types in `test/` with shared `Mock*` types, one mock
family at a time, without changing what any test checks.

## Current status

- **State:** Planned. No code written yet; these plan docs are the only changes on the
  branch `vp/test-mocks`.
- **Next step:** Stub Mock 1, increment 1 (the inert tier in
  `test/test_helpers/mocks/compressors.jl` plus the preamble include), then stop for review.
  See [Mock 1](mock1_compressors.md).
- **Last updated:** 2026-10-20

Update this section, and the one in the active mock's file, at every gate.

## Design (applies to every mock)

- **Location:** `test/test_helpers/mocks/<family>.jl`. Each file defines one module (e.g.
  `MockCompressors`) that exports `Mock*` names and imports whatever it extends
  (`LinearAlgebra.mul!`, `RandLinearAlgebra` interface functions).
- **Loading:** `preamble.jl` `include`s every mock file, as it does `FieldTest` and
  `ApproxTol`. Each test module therefore gets its own copy, so methods defined for a mock
  never leak between test files. A test opts in with `using .MockCompressors`.
- **Tiers:** a family offers types by behavior. The **inert** tier has fields only and no
  methods, so abstract fallbacks throw `ArgumentError`. **Working** tiers define just enough
  interface to test one layer (e.g. `mul!`). A test uses the lowest tier that exercises what
  it checks. If one file needs two tiers, the file is split rather than extending a mock
  mid-file.
- **Naming:** `Mock<Thing>` for the inert tier, `Mock<Tier><Thing>` for working tiers. A
  `Test*` type stays local only where its shape is unique to one file (decided per mock).
- **Gates:** mock definitions are their own implementation, so the stub gate covers the mock
  definitions with docstrings plus a per-file migration outline, and the implement gate
  applies the migrations. No test logic changes. The plan is revised after every mock.
- **Plan files:** this overview holds the roadmap and overall status. Each mock (or group of
  provisional mocks) has its own file with detailed status.

## Roadmap

Rows are in execution order.

| Item | Scope | Risk | Plan |
|:-----|:------|:-----|:-----|
| Mock 1 | `CompressorRecipe` with `n_rows`, `n_cols` (tiered) | Low | [Mock 1](mock1_compressors.md) (detailed) |
| Mock 2 | `ApproximatorRecipe` (tiered) | Low | [Mocks 2-8](mocks_2_to_8.md) (provisional) |
| Mock 3 | Right-sketch compressor with `op` | Medium | [Mocks 2-8](mocks_2_to_8.md) (provisional) |
| Mock 4 | Left-sketch compressor with `op` | Medium-high | [Mocks 2-8](mocks_2_to_8.md) (provisional) |
| Mock 5 | `SolverError` and recipe | High | [Mocks 2-8](mocks_2_to_8.md) (provisional) |
| Mock 6 | `Logger` and recipe | High | [Mocks 2-8](mocks_2_to_8.md) (provisional) |
| Mock 7 | `SubSolver` and recipe | Medium | [Mocks 2-8](mocks_2_to_8.md) (provisional) |
| Mock 8 | Remaining inert stubs | Low | [Mocks 2-8](mocks_2_to_8.md) (provisional, may be dropped) |

## Status checklist

- [ ] Mock 1
- [ ] Mock 2
- [ ] Mock 3
- [ ] Mock 4
- [ ] Mock 5
- [ ] Mock 6
- [ ] Mock 7
- [ ] Mock 8

## Out of scope / potential issues

- Open PR #185 edits test files that Mocks 3-7 will also touch (`randsvd.jl`,
  `rangefinder.jl`, `kaczmarz.jl`, `ihs.jl`, `col_projection.jl`,
  `subsolver_abstract_types.jl`). Mocks 1 and 2 touch none of its files. Check its merge
  status before starting Mock 3.
- Before merging, remove the `Development` entry in `docs/make.jl` and these docs, as was
  done for `test-preamble`.

## Decisions log

- Mock modules are included from the preamble (user decision). This replaces the earlier
  idea of including each mock file from the individual test.
- Names use `Mock*` (user decision).
- Conflicting requirements are handled with tiers (inert vs working) and by splitting files,
  not by mutating a shared mock inside a file (user decision).
- Mock 1 includes `types` and `update`, which the first proposal had left out, because the
  tier design covers them naturally.
- Plans are written per mock, with Mocks 2-8 provisional (user decision).
- Plan docs are split: overview, one file per detailed mock, one file for the provisional
  mocks (user decision).
- Restoring missing `@test` assertions and fixing type instabilities are handled by an open
  PR, not by this plan (user decision).
