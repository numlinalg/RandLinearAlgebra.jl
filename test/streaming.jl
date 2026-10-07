using LinearAlgebra
using Random

mutable struct GuardedGeneratedSource{T,F} <: AbstractRowSource{T}
    dimensions::Tuple{Int,Int}
    entry::F
    maximum_rows_per_read::Int
    next_row::Int
    blocks_read::Int
end

Base.size(source::GuardedGeneratedSource) = source.dimensions
Base.size(source::GuardedGeneratedSource, dimension::Integer) = source.dimensions[dimension]

function RandLinearAlgebra.readrows!(
    destination::AbstractMatrix,
    source::GuardedGeneratedSource,
    rows::UnitRange{Int},
)
    length(rows) <= source.maximum_rows_per_read || error("requested an oversized block")
    first(rows) == source.next_row || error("rows were not read in sequential order")
    size(destination) == (length(rows), size(source, 2)) || error("wrong destination size")

    for column in axes(destination, 2), (local_row, global_row) in enumerate(rows)
        destination[local_row, column] = source.entry(global_row, column)
    end
    source.next_row = last(rows) + 1
    source.blocks_read += 1
    return destination
end

@testset "Streaming row sources" begin
    A = reshape(collect(1.0:24.0), 6, 4)
    source = MatrixRowSource(A)
    buffer = zeros(3, 4)
    readrows!(buffer, source, 2:4)
    @test buffer == A[2:4, :]
    @test size(source) == size(A)
    @test eltype(source) == Float64
    @test_throws DimensionMismatch readrows!(zeros(2, 4), source, 2:4)
end

@testset "Streaming CountSketch" begin
    rng = Xoshiro(42)
    A = randn(rng, 240, 8)
    x_exact = randn(rng, 8)
    b = A * x_exact

    SA, Sb, stats = streaming_count_sketch(
        MatrixRowSource(A),
        b;
        sketch_size=64,
        block_size=17,
        seed=11,
    )
    SA_again, Sb_again, _ = streaming_count_sketch(
        A,
        b;
        sketch_size=64,
        block_size=31,
        seed=11,
    )

    @test SA == SA_again
    @test Sb == Sb_again
    @test stats == StreamingSketchStats(1, cld(size(A, 1), 17), size(A, 1), 17)

    S = zeros(64, size(A, 1))
    for row in axes(A, 1)
        code = RandLinearAlgebra._streaming_hash(UInt64(11), row)
        bucket = Int(mod(code, UInt64(size(S, 1)))) + 1
        S[bucket, row] = isodd(code >> 63) ? -1.0 : 1.0
    end
    @test SA == S * A
    @test Sb == S * b

    x, solve_stats = sketched_least_squares(
        MatrixRowSource(A),
        b;
        sketch_size=64,
        block_size=17,
        seed=11,
    )
    @test x ≈ x_exact rtol = 1.0e-10
    @test solve_stats == stats
end

@testset "Bounded sequential access without a materialized matrix" begin
    rows, columns = 10_003, 5
    block_size = 128
    coefficients = collect(1.0:columns)
    entry(row, column) = sin(row + column) + (row == column ? 2.0 : 0.0)
    source = GuardedGeneratedSource{Float64,typeof(entry)}(
        (rows, columns),
        entry,
        block_size,
        1,
        0,
    )
    b = [sum(entry(row, column) * coefficients[column] for column in 1:columns) for row in 1:rows]

    x, stats = sketched_least_squares(
        source,
        b;
        sketch_size=80,
        block_size=block_size,
        seed=7,
    )

    @test x ≈ coefficients rtol = 1.0e-9
    @test source.next_row == rows + 1
    @test source.blocks_read == cld(rows, block_size)
    @test stats.passes == 1
    @test stats.max_block_rows == block_size
end

@testset "Streaming input validation" begin
    source = MatrixRowSource(ones(4, 2))
    @test_throws DimensionMismatch streaming_count_sketch(source, ones(3); sketch_size=2)
    @test_throws ArgumentError streaming_count_sketch(source, ones(4); sketch_size=0)
    @test_throws ArgumentError streaming_count_sketch(source, ones(4); sketch_size=2, block_size=0)
    @test_throws ArgumentError streaming_count_sketch(source, ones(4); sketch_size=2, seed=-1)
    @test_throws ArgumentError streaming_count_sketch(
        source,
        ones(4);
        sketch_size=2,
        seed=big(typemax(UInt64)) + 1,
    )
    @test_throws ArgumentError streaming_count_sketch(
        MatrixRowSource(zeros(0, 2)),
        zeros(0);
        sketch_size=2,
    )

    SA, Sb, _ = streaming_count_sketch(
        fill(Float16(10), 100, 2),
        fill(Float16(10), 100);
        sketch_size=1,
    )
    @test eltype(SA) == Float32
    @test eltype(Sb) == Float32
    @test all(isfinite, SA)
    @test all(isfinite, Sb)
end
