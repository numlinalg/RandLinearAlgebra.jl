# Testing Overview

This page describes how the test suite is organized and how to run it. See also
[Test Preamble and Helpers](test_preamble_helpers.md), [Test Mocks](test_mocks.md) and
[Writing Tests](writing_tests.md).

## Layout

The directories under `test/` mirror `src/`:

```
test/
├── runtests.jl          # Entry point
├── test_helpers/        # Shared preamble, helper modules and mocks
├── Compressors/         # Compressor, Distribution and helper tests
├── Approximators/       # Approximator, Selector and ErrorMethods tests
└── Solvers/             # Solver, Logger, ErrorMethods and SubSolver tests
```

Files named `<thing>_abstract_*.jl` test the abstract interface of a technique, such as
fallback methods and argument errors. The remaining files test one implementation each, for
example `gaussian.jl`.

## How tests are discovered

`runtests.jl` holds a list of directories. For each directory it includes every file ending
in `.jl`, except `runtests.jl` itself. A new test file is therefore picked up automatically
as long as it is in a listed directory. A new directory must be added to the list.

!!! note
    The listing is not recursive. A subdirectory such as `Compressors/helpers/` is only
    tested if it appears in the list.

## One module per test file

Every test file defines its own `module`, named after the file, and includes the shared
preamble inside it:

```julia
module compressor_abstract_size
include("../test_helpers/preamble.jl")

@testset "Compressor Size" begin
    # tests
end

end
```

Each module has its own namespace, so a type or method defined in one file never leaks into
another. This isolation is what lets test files define stand-in (mock) types freely.

## Running tests

Run the whole suite from the repository root:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Run one file while developing:

```bash
julia --project -e 'include("test/Compressors/compressor_abstract_size.jl")'
```

Run the whole suite before opening a pull request, since a change to a shared helper or mock
can affect many files.
