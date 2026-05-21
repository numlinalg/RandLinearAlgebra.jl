# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RandLinearAlgebra.jl is a Julia package implementing randomized linear algebra algorithms for:
- Low-rank matrix approximations (Randomized SVD, Range Finder)
- Solving linear systems (Kaczmarz, Column Projection, IHS solvers)
- Matrix compression techniques (Gaussian, FJLT, SRHT, CountSketch, SparseSign, Sampling)

**Documentation:** https://numlinalg.github.io/RandLinearAlgebra.jl/dev

## Build and Test Commands

```bash
# Run all tests (standard Julia Pkg approach)
julia --project -e 'using Pkg; Pkg.test()'

# Build documentation locally
julia --project=docs/ docs/make.jl
```

## Code Style

Uses [BlueStyle](https://github.com/invenia/BlueStyle) formatting. Configuration in `.JuliaFormatter.toml`:
- `style = "blue"`
- `format_markdown = true`
- `format_docstrings = false`

## Architecture

The library uses a **Recipe Pattern** with two abstraction levels:

1. **Technique Types** (user-facing): Hold algorithm parameters only
   - `Compressor`, `Solver`, `Approximator`, `Logger`, `SolverError`, `SubSolver`

2. **Recipe Types** (internal): Include pre-allocated memory for computation
   - `CompressorRecipe`, `SolverRecipe`, `ApproximatorRecipe`, etc.

**Workflow:** `complete_*()` initializes a Recipe from a Technique → main functions (`rsolve!`, `rapproximate!`, `mul!`) execute

### Key Components

**Compressors** (`src/Compressors/`): 7 compression methods - CountSketch, FJLT, Gaussian, Identity, Sampling, SparseSign, SRHT. Support left/right multiplication via `Cardinality` types.

**Solvers** (`src/Solvers/`): 3 iterative solvers - Kaczmarz (row projection), ColumnProjection (column subspace), IHS (Iterative Hessian Sketching). Supporting infrastructure includes Loggers, ErrorMethods, and SubSolvers.

**Approximators** (`src/Approximators/`): RangeFinder and RandSVD for low-rank approximation. Selectors (LUPP, QRCP) for pivot selection.

### Module Structure

```
src/RandLinearAlgebra.jl  # Main module, exports
src/Compressors.jl        # Compressor abstractions
src/Solvers.jl            # Solver abstractions
src/Approximators.jl      # Approximator abstractions
src/*/                    # Implementations in subdirectories
```

## Dependencies

Runtime: LinearAlgebra, Random, SparseArrays, StatsBase (all in Project.toml)

## PR Guidelines

See `.github/pull_request_template.md` for checklist. Key requirements:
- Detailed docstrings for exported functions
- Unit tests with sufficient coverage
- BlueStyle code formatting
- Local doc compilation to verify changes

## Docstring Conventions

### Section Headers
Use single `#` for top-level docstring sections:
- `# Arguments` - function parameters
- `# Returns` - return values (not "Outputs")
- `# Throws` - exceptions that may be raised
- `# Fields` - struct fields
- `# Constructor` - constructor documentation
- `# Mathematical Description` - algorithm explanation

### Nested Sections
Use `##` for subsections within constructors:
- `## Keywords` - keyword arguments
- `## Returns` - constructor return value

### Reusable Docstring Components
Use Dict interpolation for consistent argument/output descriptions:
```julia
arg_list = Dict{Symbol,String}(
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
    :b => "`b::AbstractVector`, a constant vector.",
)

"""
    my_function(A, b)

# Arguments
- $(arg_list[:A])
- $(arg_list[:b])
"""
```

### Admonitions
Use standard Documenter.jl admonition syntax:
```julia
!!! note
    Important information here.

!!! info
    Additional context here.
```

### Citations
Use Documenter.jl citation syntax: `[author_year](@cite)`
