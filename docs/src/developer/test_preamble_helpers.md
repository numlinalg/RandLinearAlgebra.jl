# Test Preamble and Helpers

Shared test code lives in `test/test_helpers/`. This page describes the preamble and the two
helper modules. Mock types are covered in [Test Mocks](test_mocks.md).

## The preamble

`test/test_helpers/preamble.jl` is included inside each test module, as shown in
[Testing Overview](testing_overview.md). It does three things:

1. Loads the packages every test uses: `Test`, `RandLinearAlgebra`, `Random`,
   `LinearAlgebra`, `SparseArrays` and `StatsBase`.
2. Imports `LinearAlgebra.mul!`, so a test can add `mul!` methods for its own types.
3. Includes the helper modules and mocks, then brings `FieldTest` and `ApproxTol` into scope.

Because the preamble is included rather than loaded as a package, each test module gets its
own copy of every helper module and mock. Nothing is shared between test modules at runtime.

!!! info
    Do not add `using Test`, `using Random` or `using LinearAlgebra` to a test file. The
    preamble already provides them, so `seed!` and `mul!` are available without an import.

## `FieldTest`

`FieldTest` (`field_test_macros.jl`) provides macros that check that a recipe type has the
fields and supertype its category requires. They keep the interface of all implementations
of one category consistent.

| Macro | Checks that the type |
|:------|:---------------------|
| `@test_compressor(T)` | is a `CompressorRecipe` with `n_rows::Int64` and `n_cols::Int64` |
| `@test_range_approximator(T)` | is an `ApproximatorRecipe` with the range-approximator fields |
| `@test_projection_solver(T)` | is a `SolverRecipe` with the projection-solver fields |
| `@test_logger(T)` | is a `LoggerRecipe` with the logger fields |

Use a macro inside a test set, passing the recipe type:

```julia
@testset "Gaussian fields" begin
    @test_compressor GaussianRecipe
end
```

Each macro opens its own `verbose` test set named after the type.

## `ApproxTol`

`ApproxTol` (`approx_tol.jl`) exports `ATOL = 1e-10`, a single absolute tolerance for
floating-point comparisons. Use it instead of a literal so the tolerance can be changed in one
place:

```julia
@test isapprox(x, y; atol = ATOL)
```
