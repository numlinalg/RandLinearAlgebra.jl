module recipe_allocations
include("test_helpers/preamble.jl")

# Each hot-loop operation is called once to compile it, then measured.
#
# Not covered here:
# - `update_compressor!` for `SRHT` allocates inside `StatsBase.sample!` without
#   replacement, and for `FJLT` it intentionally draws a new sparse matrix.
# - `update_sub_solver!` and `ldiv!` for the QR/LQ sub-solvers are covered in their own
#   test files.
@testset "Recipe Allocations" begin
    Random.seed!(5)
    A = rand(200, 100)

    @testset "mul!: $name" for (name, technique) in (
        ("CountSketch", card -> CountSketch(compression_dim=10, cardinality=card)),
        ("Gaussian", card -> Gaussian(compression_dim=10, cardinality=card)),
        ("SparseSign", card -> SparseSign(compression_dim=10, cardinality=card)),
        ("SRHT", card -> SRHT(compression_dim=10, block_size=16, cardinality=card)),
        ("FJLT", card -> FJLT(compression_dim=10, block_size=16, cardinality=card)),
        (
            "FixedPatternFJLT",
            card -> FixedPatternFJLT(compression_dim=10, block_size=16, cardinality=card),
        ),
    )
        # S * A
        S = complete_compressor(technique(Left()), A)
        C = zeros(size(S, 1), size(A, 2))
        mul!(C, S, A, 1.0, 0.0)
        @test (@allocated mul!(C, S, A, 1.0, 0.0)) == 0

        # A * S
        S = complete_compressor(technique(Right()), A)
        C = zeros(size(A, 1), size(S, 2))
        mul!(C, A, S, 1.0, 0.0)
        @test (@allocated mul!(C, A, S, 1.0, 0.0)) == 0
    end

    @testset "update_compressor!: $name" for (name, technique) in (
        ("CountSketch", card -> CountSketch(compression_dim=10, cardinality=card)),
        ("Gaussian", card -> Gaussian(compression_dim=10, cardinality=card)),
        ("SparseSign", card -> SparseSign(compression_dim=10, cardinality=card)),
        (
            "FixedPatternFJLT",
            card -> FixedPatternFJLT(compression_dim=10, block_size=16, cardinality=card),
        ),
    )
        for card in (Left(), Right())
            S = complete_compressor(technique(card), A)
            update_compressor!(S)
            @test (@allocated update_compressor!(S)) == 0
        end
    end
end

end
