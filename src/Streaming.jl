"""
    AbstractRowSource{T}

An interface for a matrix whose rows can be read in bounded blocks.

A subtype must implement `size(source)` and
`readrows!(destination, source, rows)`. Unlike `AbstractMatrix`, this interface does not
promise inexpensive scalar or random access. Algorithms using it can therefore make their
data-transfer pattern explicit.
"""
abstract type AbstractRowSource{T} end

Base.eltype(::Type{<:AbstractRowSource{T}}) where {T} = T
Base.eltype(::AbstractRowSource{T}) where {T} = T

"""
    readrows!(destination, source, rows)

Copy the consecutive `rows` of `source` into `destination`.

The destination must have `length(rows)` rows and `size(source, 2)` columns. Implementations
may read from memory, a file, a remote store, or generate the rows on demand.
"""
function readrows!(destination, source::AbstractRowSource, rows::UnitRange{Int})
    return throw(MethodError(readrows!, (destination, source, rows)))
end

"""
    MatrixRowSource(A)

Adapt an `AbstractMatrix` to the [`AbstractRowSource`](@ref) interface. This is the reference
backend for correctness and small examples; wrapping a matrix does not move it out of memory.
"""
struct MatrixRowSource{T,M<:AbstractMatrix{T}} <: AbstractRowSource{T}
    matrix::M
end

Base.size(source::MatrixRowSource) = size(source.matrix)
Base.size(source::MatrixRowSource, dimension::Integer) = size(source.matrix, dimension)

function readrows!(
    destination::AbstractMatrix,
    source::MatrixRowSource,
    rows::UnitRange{Int},
)
    expected = (length(rows), size(source, 2))
    size(destination) == expected || throw(
        DimensionMismatch(
            "destination has size $(size(destination)); expected $expected for rows $rows",
        ),
    )
    copyto!(destination, view(source.matrix, rows, :))
    return destination
end

"""
    StreamingSketchStats

Observable data-access costs for a streaming sketch.

`max_block_rows` bounds the number of source rows resident in the input buffer. It does not
include the in-memory `SA` and `Sb` sketch, whose combined size is
`sketch_size * (size(source, 2) + 1)`.
"""
struct StreamingSketchStats
    passes::Int
    blocks_read::Int
    rows_read::Int
    max_block_rows::Int
end

# SplitMix64 gives each row a reproducible bucket and sign without storing O(m) random values.
function _streaming_hash(seed::UInt64, row::Int)
    value = seed + UInt64(row) + 0x9e3779b97f4a7c15
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

"""
    streaming_count_sketch(source, b; sketch_size, block_size=1024, seed=0)

Construct `SA` and `Sb` in one row-wise pass over the least-squares problem `Ax ≈ b`.

The CountSketch row and sign for input row `i` are generated from `(seed, i)`, so the method
stores no sketch operator proportional to the number of source rows. Its principal working
storage is an input block of size `block_size * size(source, 2)` and the output sketch of size
`sketch_size * size(source, 2)`.

This initial interface keeps `b` in memory. A future block-vector source can remove that
restriction without changing the matrix source protocol.
"""
function streaming_count_sketch(
    source::AbstractRowSource,
    b::AbstractVector;
    sketch_size::Int,
    block_size::Int=1024,
    seed::Integer=0,
)
    rows, columns = size(source)
    length(b) == rows || throw(DimensionMismatch("b has length $(length(b)); expected $rows"))
    rows > 0 || throw(ArgumentError("source must contain at least one row"))
    sketch_size > 0 || throw(ArgumentError("sketch_size must be positive"))
    block_size > 0 || throw(ArgumentError("block_size must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    seed <= typemax(UInt64) || throw(ArgumentError("seed must fit in a UInt64"))

    output_type = promote_type(Float32, float(eltype(source)), float(eltype(b)))
    sketched_matrix = zeros(output_type, sketch_size, columns)
    sketched_vector = zeros(output_type, sketch_size)
    input_buffer = Matrix{eltype(source)}(undef, min(block_size, rows), columns)

    blocks_read = 0
    max_block_rows = 0
    first_row = 1
    while first_row <= rows
        last_row = min(first_row + block_size - 1, rows)
        source_rows = first_row:last_row
        block_rows = length(source_rows)
        block = view(input_buffer, 1:block_rows, :)
        readrows!(block, source, source_rows)

        @inbounds for local_row in 1:block_rows
            global_row = first_row + local_row - 1
            code = _streaming_hash(UInt64(seed), global_row)
            bucket = Int(mod(code, UInt64(sketch_size))) + 1
            sign = isodd(code >> 63) ? -one(output_type) : one(output_type)
            for column in 1:columns
                sketched_matrix[bucket, column] += sign * block[local_row, column]
            end
            sketched_vector[bucket] += sign * b[global_row]
        end

        blocks_read += 1
        max_block_rows = max(max_block_rows, block_rows)
        first_row = last_row + 1
    end

    stats = StreamingSketchStats(1, blocks_read, rows, max_block_rows)
    return sketched_matrix, sketched_vector, stats
end

function streaming_count_sketch(A::AbstractMatrix, b::AbstractVector; kwargs...)
    return streaming_count_sketch(MatrixRowSource(A), b; kwargs...)
end

"""
    sketched_least_squares(source, b; sketch_size, block_size=1024, seed=0)

Compute a pedagogical one-pass approximation to `argmin_x ||Ax-b||₂` by solving the smaller
problem `argmin_x ||SAx-Sb||₂` produced by [`streaming_count_sketch`](@ref).

This routine demonstrates bounded matrix reads; it is not yet a high-accuracy iterative
solver. In particular, it keeps `b`, `SA`, and `Sb` in memory and does not refine the solution
against the original data.
"""
function sketched_least_squares(source, b::AbstractVector; kwargs...)
    sketched_matrix, sketched_vector, stats =
        streaming_count_sketch(source, b; kwargs...)
    return sketched_matrix \ sketched_vector, stats
end
