# RandLinearAlgebra.jl — Status

_Last updated: 2026-11-03_

## Current focus

Test infrastructure cleanup, done in small steps.

## Test consolidation

- **Done:** shared test preamble (`test/test_helpers/preamble.jl`), merged in
  [#194](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/194).
- **Planned, not started:** consolidating mock types into shared `Mock*` types. Mocks are
  still defined in each test file; for example, `TestCompressorRecipe` is defined locally in
  six `test/Compressors/compressor_abstract_*.jl` files. An earlier attempt was abandoned and
  its branch deleted.
- **Planned workflow:** the plan lives on `vp/test-mocks` in `docs/src/dev/test-mocks/`
  (`overview.md`, `mock1_compressors.md`, `mocks_2_to_8.md`; commit `a19246c8`, 2026-10-20).
  - Mocks live in `test/test_helpers/mocks/<family>.jl`, one module per family (e.g.
    `MockCompressors`), included from `preamble.jl`; tests opt in with `using`.
  - Types come in tiers: an **inert** tier (fields only, so fallbacks throw) and **working**
    tiers (just enough methods for one layer, e.g. `mul!`). A file needing two tiers is split.
  - Each increment has two review gates: (1) mock definitions with docstrings plus a
    migration outline, (2) the migrations. No test logic changes.
  - Roadmap: Mock 1 compressor recipe (detailed plan) → Mock 2 approximator recipe →
    Mocks 3-4 right/left-sketch compressors → Mocks 5-7 solver error, logger, sub-solver →
    Mock 8 remaining inert stubs (may be dropped). Mocks 2-8 are provisional.
  - Mock 1 increments: (1) inert tier + migrate `size`, `adjoint`, `dimchecks`, `types`;
    (2) `Mul5`/`Mul3` tiers + split `multiplication` into errors/5-arg/3-arg files;
    (3) `Update` tier + migrate `update`.
  - Before merging, remove the `Development` entry in `docs/make.jl` and the plan docs.
- **Known constraint:** the empty stub types used to check that errors are thrown before any
  methods exist (`*_abstract_types.jl`) can't be shared safely until those tests are
  restructured.

## Related open work

- [#185](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/185): type stability and
  allocation fixes in Recipe structs (author: dmaldona). **Changes requested** by vp314 on
  2026-10-27. The branch is behind `main` and needs a merge (not a rebase); a docs build issue
  was also raised. No update since 2026-10-27. It touches test files used by Mocks 3-7;
  check its status before starting Mock 3.
- Other open PRs: #196 (LSQR sub-solver), #195 (LUPP distribution), #193 (`compression_dim`), #192 (streaming
  CountSketch), #187 (move Selectors), #186 (LeverageScore distribution), #170 (GPU support).

## Next steps

| # | Step | Suggested due |
|:--|:-----|:--------------|
| 1 | Mock 1, increment 1 stub: `MockCompressor`/`MockCompressorRecipe` + preamble include + migration outline, then review | 2026-11-05 |
| 2 | Mock 1, increments 1-3 migrations (first PR: shared `TestCompressorRecipe` replacement) | 2026-11-12 |
| 3 | Follow up on #185 once the author responds to the review | 2026-11-10 |
| 4 | Write detailed plan for Mock 2 (`MockApproximators`); revisit Mocks 3-8 | 2026-11-19 |
| 5 | Plan how to restructure the throw-first stub tests so their stubs can be shared | 2026-11-19 |
| 6 | Extend mock consolidation to Approximators and Solvers tests (Mocks 2-7) | 2026-12-03 |
