# Mocks 2-8 (provisional)

Part of the [test mock consolidation](overview.md). Each mock gets a detailed plan, in its
own file, once the previous mock is done. Expect changes for Mocks 4-7 in particular.

## Current status

- **State:** Not started. Only the provisional notes below exist.
- **Next step:** None until [Mock 1](mock1_compressors.md) is done. Then write the detailed
  plan for Mock 2 in its own file and revisit the notes for Mocks 3-8.
- **Last updated:** 2026-10-20

## Mock 2: `MockApproximators`

Mirrors [Mock 1](mock1_compressors.md). Shapes in use: `(n_rows, n_cols)` in
`approximator_abstract_adjoint.jl`, `approximator_abstract_multiplication.jl` and
`compressor_abstract_dimchecks.jl`; `(code::Int64)` in `approximator_abstract_types.jl` and
`rapproximate_abstract_interface.jl`; empty in `approximator_error_abstract_types.jl`. The
`code` variants may reduce to the sized type.

## Mock 3: working right-sketch compressor

Adds `MockRightCompressor` and its recipe to `MockCompressors`. It has `cardinality`, `op`,
`complete_compressor(comp, A)` and a right `mul!`. `rangefinder.jl` and `randsvd.jl` hold
identical copies. `range_finder_helpers.jl` uses an immutable variant without `cardinality`
and a `TestRangeApproximatorRecipe`, so its fit is uncertain. `rangefinder.jl` and
`range_finder_helpers.jl` qualify the mock in `mul!` signatures as
`Main.<Module>.TestCompressorRecipe`, which disappears here.

## Mock 4: working left-sketch compressor

The `KTest`/`ITest` copies differ from Mock 3 by cardinality, the `(A, b)` signature of
`complete_compressor`, `update_compressor!` and a left `mul!`. Decide whether Mocks 3 and 4
share one parameterized type. The 900- and 500-line solver files assert on
`Main.KaczmarzTest.KTest*` names in `fieldtypes` checks, which must be rewritten.

## Mock 5: solver errors

`complete_error` is currently defined for `::Kaczmarz` in one copy and `::IHS` in the other.
Options are dispatching on `::Solver`, or per-solver extension methods kept in each test
file. The `NoRes` variants and the `col_projection` validation stubs are inert-like.

## Mock 6: loggers

The IHS logger adds `iteration`. The `col_projection` loggers (`I`, `II`, `III`) each omit a
different field to trigger a specific validation error, so they may become a tier ladder or
stay local.

## Mock 7: sub-solver

`KTestSubSolver` has `ldiv!`, `complete_sub_solver` and `update_sub_solver!`. The other
sub-solver stubs are inert. Moves with Mocks 5 and 6.

## Mock 8: remaining inert stubs

`*_abstract_types.jl` for Solver, SolverError, Logger, SubSolver, Selector, Distribution and
ApproximatorError, plus the `ErrorMethods` solver recipes. Migrate a stub only if the same
shape appears in two or more files. The `ErrorMethods` solver recipes differ by design and
will likely stay local. This mock may be dropped.

## Status checklist

- [ ] Mock 2: detailed plan to be written
- [ ] Mock 3: detailed plan to be written
- [ ] Mock 4: detailed plan to be written
- [ ] Mock 5: detailed plan to be written
- [ ] Mock 6: detailed plan to be written
- [ ] Mock 7: detailed plan to be written
- [ ] Mock 8: detailed plan to be written
