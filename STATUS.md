# RandLinearAlgebra.jl — Status

_Last updated: 2026-11-03_

## Current focus

Test infrastructure cleanup, done in small steps.

## Test consolidation

- **Done:** shared test preamble (`test/test_helpers/preamble.jl`), merged in
  [#194](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/194).
- **Mock 1 (compressors) done, in review:**
  [#197](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/197) adds `MockCompressors`
  (`test/test_helpers/mocks/mock_compressors.jl`, included from `preamble.jl`) and migrates
  the `size`, `adjoint`, `dimchecks`, `types` and `update` compressor tests. It splits
  `compressor_abstract_multiplication.jl` into `_errors`, `_5arg` and `_3arg`. The full suite
  and the docs build passed; no reviewers requested yet. The same PR adds a Developer Guide
  (`docs/src/developer/`): testing overview, preamble and helpers, test mocks (with the mock
  docstrings), and writing tests.
- **Not started:** mocks for approximators and solvers (Mocks 2-8). Those tests still define
  local `Test*` types. An earlier attempt was abandoned and its branch deleted.
- **Branches:** `vp/test-mocks` is the long-lived branch that holds the plan docs
  (`docs/src/dev/test-mocks/`: `overview.md`, `mock1_compressors.md`, `mocks_2_to_8.md`). It
  stays until all mocks are done. Each step gets a short-lived branch, such as
  `vp/test-mocks-compressor` for #197, with `docs/src/dev/` and the `Development` entry in
  `docs/make.jl` removed before the PR.
  - If `vp/test-mocks` later merges `main`, git will delete the plan docs there once #197
    lands; restore them from history.
- **Workflow:**
  - Mocks live in `test/test_helpers/mocks/mock_<family>.jl`, one module per family, included
    from `preamble.jl`; tests opt in with `using`.
  - Types come in tiers: an **inert** tier (fields only, so fallbacks throw) and **working**
    tiers (just enough methods for one layer, e.g. `mul!`). A file needing two tiers is split.
  - Each increment has two review gates: (1) mock definitions with docstrings plus a
    migration outline, (2) the migrations. No test logic changes.
  - Adding a mock family also needs an `include` and a `modules` entry in `docs/make.jl`
    and a `@docs` block in `docs/src/developer/test_mocks.md`.
  - Roadmap: Mock 2 approximator recipe → Mocks 3-4 right/left-sketch compressors → Mocks 5-7
    solver error, logger, sub-solver → Mock 8 remaining inert stubs (may be dropped). Mocks
    2-8 are provisional; no detailed plan yet.
- **Known constraint:** the empty stub types used to check that errors are thrown before any
  methods exist (`*_abstract_types.jl`) can't be shared safely until those tests are
  restructured.

## Related open work

- [#185](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/185): type stability and
  allocation fixes in Recipe structs (author: dmaldona). **Changes requested** by vp314 on
  2026-10-27. The branch is behind `main` and needs a merge (not a rebase); a docs build issue
  was also raised. No update since 2026-10-27. It touches test files used by Mocks 3-7;
  check its status before starting Mock 3.
- Other open PRs: #196 (LSQR sub-solver), #195 (LUPP distribution), #193 (`compression_dim`),
  #192 (streaming CountSketch), #187 (move Selectors), #186 (LeverageScore distribution), #170 (GPU support).

## Next steps

| # | Step | Suggested due |
|:--|:-----|:--------------|
| 1 | Request review on #197 and address comments | 2026-11-10 |
| 2 | Follow up on #185 once the author responds to the review | 2026-11-10 |
| 3 | After #197 merges, check `vp/test-mocks` against `main` (plan docs, make.jl) | 2026-11-17 |
| 4 | Write detailed plan for Mock 2 (`MockApproximators`); revisit Mocks 3-8 | 2026-11-19 |
| 5 | Plan how to restructure the throw-first stub tests so their stubs can be shared | 2026-11-19 |
| 6 | Extend mock consolidation to Approximators and Solvers tests (Mocks 2-7) | 2026-12-03 |
