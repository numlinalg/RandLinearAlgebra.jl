# A First Streaming Architecture

Randomized linear algebra can reduce the amount of data retained in memory, but randomness
alone does not make an implementation out of core. The implementation must also control how
the original matrix moves from storage into memory.

This page documents a deliberately small first step in RandLinearAlgebra.jl. It is intended
both as usable code and as a teaching example for future development.

## The example problem

Consider the least-squares problem

```math
\min_x \|Ax-b\|_2,
```

where ``A`` is too large to retain in memory. A CountSketch ``S`` gives the smaller problem

```math
\min_x \|SAx-Sb\|_2.
```

For every input row ``i``, CountSketch chooses a bucket ``h(i)`` and sign ``\sigma(i)`` and
accumulates

```math
(SA)_{h(i),:} \mathrel{+}= \sigma(i) A_{i,:}, \qquad
(Sb)_{h(i)} \mathrel{+}= \sigma(i)b_i.
```

Each row can be discarded immediately after this update. RandLinearAlgebra.jl derives the
bucket and sign from `(seed, i)`, rather than storing a sparse ``S`` with data proportional to
the number of rows.

## Why a row-source interface?

`AbstractMatrix` describes mathematical indexing, but it does not describe data movement.
Scalar indexing might access RAM, trigger a disk read, download a remote chunk, or recompute
an entry. The small [`AbstractRowSource`](@ref) interface instead asks a backend to implement
one explicit operation:

```julia
readrows!(destination, source, rows)
```

The streaming algorithm calls it with bounded, consecutive ranges. This makes the access
contract visible and testable. [`MatrixRowSource`](@ref) is an in-memory reference backend:

```julia
using RandLinearAlgebra

A = randn(10_000, 20)
b = randn(10_000)
source = MatrixRowSource(A)

x, stats = sketched_least_squares(
    source,
    b;
    sketch_size=100,
    block_size=256,
    seed=123,
)

@show stats.passes          # 1
@show stats.max_block_rows  # 256
```

Wrapping `A` does not save memory. It lets the same algorithm and tests exercise the protocol
before disk and remote backends are added.

## Memory model

For an ``m \times n`` matrix, sketch size ``s``, and block size ``B``, the implementation
stores approximately:

```math
O(Bn + sn + s + m)
```

values: one matrix block, ``SA``, ``Sb``, and the currently in-memory vector ``b``. It does not
store the ``mn`` entries of ``A`` through the source interface and does not materialize ``S``.

The `StreamingSketchStats` result makes passes and maximum read size observable. Tests use a
generated source that throws on oversized or nonsequential reads. Consequently, a small CI
test can verify the access discipline without committing or uploading a large matrix.

## What this prototype does not solve

This is intentionally not a complete out-of-core framework:

- The right-hand side `b` remains in memory.
- The sketch ``SA`` has size ``s \times n`` and must fit in memory.
- The direct sketched solution is an approximation, not an iterative refinement.
- There is no asynchronous prefetching or overlap of I/O and computation.
- There is no file format or disk backend yet.
- Only sequential row-block access is represented. Some algorithms instead need random rows,
  columns, `A*x`, `A'*x`, or multiple passes.

These limitations are useful design boundaries. Future algorithms should state their required
capabilities rather than accepting a broad type and discovering expensive access at runtime.

## Natural next steps

1. Add a block-vector source so that `b` can also remain out of memory.
2. Add a disk backend and benchmark physical bytes read, not only requested rows.
3. Add operator capabilities for `mul!(y, A, x)` and `mul!(y, A', x)`.
4. Use a sketch as a preconditioner for an iterative least-squares solver.
5. Explore single-pass randomized SVD variants; the current randomized SVD requires products
   with both `A` and `A'` and retains dimension-sized factors.
6. Add cancellation, prefetching, and double buffering only after measurements show that I/O
   latency is the bottleneck.

## Related ecosystem and systems intuition

Julia's standard-library `Mmap` module can expose local files as arrays, but memory mapping
does not by itself guarantee an efficient access order. The file layout and the algorithm's
block direction must agree. DiskArrays.jl provides a broader vocabulary for chunked arrays;
HDF5.jl and Zarr.jl provide chunked storage formats; LinearMaps.jl and LinearOperators.jl
represent matrix-free products. Adapters to these packages can be optional rather than core
dependencies.

LLM inference systems offer useful systems lessons: explicit memory budgets, memory-mapped
weights, prefetching, fixed workspaces, and CPU/GPU offloading. Their numerical access pattern
is different, however. Transformer weights are usually consumed layer by layer, while a
linear-algebra method may request `A' * x`, sampled rows, or repeated full passes. The reusable
lesson is to model storage and transfer explicitly, not to copy an LLM runtime architecture.

The long-term goal is therefore not a single "out-of-core matrix" type. It is a set of small,
measurable access capabilities from which randomized algorithms can declare what they need.

## API

```@docs
AbstractRowSource
MatrixRowSource
readrows!
StreamingSketchStats
streaming_count_sketch
sketched_least_squares
```
