# RandLinearAlgebra.jl — Status

_Last updated: 2026-10-29_

## Current focus

Test infrastructure cleanup, done in small steps.

## Test consolidation

- **Done:** shared test preamble (`test/test_helpers/preamble.jl`), merged in
  [#194](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/194).
- **Not started:** consolidating mock types. An earlier attempt was abandoned and its
  branch deleted. Mocks are still defined in each test file; for example, an identical
  `TestCompressorRecipe` is defined in four `test/Compressors/compressor_abstract_*.jl` files.
- **Planned workflow:** a non-merging planning branch, `vp/test-refactoring-plan` (not yet
  created), holds the plan docs. Each step goes to `main` in its own PR. The detailed plan
  will live on that branch.
- **Known constraint:** the empty stub types used to check that errors are thrown before any
  methods exist (`*_abstract_types.jl`) can't be shared safely until those tests are
  restructured.

## Related open work

- [#185](https://github.com/numlinalg/RandLinearAlgebra.jl/pull/185): type stability and
  allocation fixes in Recipe structs (author: dmaldona). **Changes requested** by vp314 on
  2026-10-27. The branch is behind `main` and needs a merge (not a rebase); a docs build issue
  was also raised.
- Other open PRs: #195 (LUPP distribution), #193 (`compression_dim`), #192 (streaming
  CountSketch), #187 (move Selectors), #186 (LeverageScore distribution), #170 (GPU support).

## Next steps

| # | Step | Suggested due |
|:--|:-----|:--------------|
| 1 | Create `vp/test-refactoring-plan` and write the incremental plan, starting with Compressors | 2026-11-05 |
| 2 | First PR: share the duplicated `TestCompressorRecipe` across compressor tests | 2026-11-12 |
| 3 | Follow up on #185 once the author responds to the review | 2026-11-10 |
| 4 | Plan how to restructure the throw-first stub tests so their stubs can be shared | 2026-11-19 |
| 5 | Extend mock consolidation to Approximators and Solvers tests | 2026-12-03 |
