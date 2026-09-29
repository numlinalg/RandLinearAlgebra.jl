# Test Preamble Consolidation

## Goal

Replace the near-identical `using`/`import` block at the top of each test module with a
single shared `test/test_helpers/preamble.jl` that each module `include`s.

## Design

`preamble.jl` is included *inside* each test module and contains:

```julia
using Test, RandLinearAlgebra, Random, LinearAlgebra, SparseArrays, StatsBase
import LinearAlgebra: mul!
include("field_test_macros.jl")
include("approx_tol.jl")
using .FieldTest
using .ApproxTol
```

`field_test_macros.jl` and `approx_tol.jl` are no longer included in `runtests.jl` as this 
would be redundant.

Rarely used packages (e.g. `Hadamard`), `import Base.*`, `import RandLinearAlgebra: ...`, 
`import Random: seed!`, RNG seeds, and test-specific structs remain within each file.

## Files

| File | Purpose | Planned change |
|:-----|:--------|:---------------|
| `test/test_helpers/preamble.jl` | Shared module preamble | New |
| `test/Compressors/**` | Compressor and distribution tests | Replace preamble with `include`, drop redundant imports |
| `test/Approximators/**` | Approximator and selector tests | Same |
| `test/Solvers/**` | Solver, error, logger, subsolver tests | Same; two files already self-include helpers |
| `test/runtests.jl` | Test driver | Remove redundant helper includes (last increment) |

No functions are added. 

## Status

- [x] 1. `preamble.jl` + pilots (`gaussian.jl`, `rangefinder.jl`, `kaczmarz.jl`)
- [x] 2. Remaining Compressors files
- [ ] 3. Remaining Approximators files — converted, awaiting review
- [ ] 4. Remaining Solvers files and `runtests.jl` cleanup

## Out of scope / potential issues

- Per-file RNG seeds could be centralized later.
- `julia --project=docs/ docs/make.jl` currently fails on four unresolved `@ref` links
  (`RangeFinder` in `api/approximators.md`, `Gaussian` in `manual/compression.md`,
  `SparseSign` in `api/compressors.md`, `Identity` in `api/selectors.md`). These are in
  files untouched by this work; with `warnonly = [:cross_references]` the build succeeds.

## Decisions log

- Preamble includes `FieldTest`/`ApproxTol` itself so its full scope is visible in one file.
- Broad `using` for Test, RandLinearAlgebra, Random, LinearAlgebra, SparseArrays, StatsBase;
  single-use packages stay local.
