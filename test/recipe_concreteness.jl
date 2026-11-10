module recipe_concreteness
include("test_helpers/preamble.jl")

# The field-type tests in each recipe's test file check the `UnionAll` recipe type, which
# cannot detect abstract fields. These tests check one completed recipe of each type
# instead, so that a field regressing to an abstract type is caught.
#
# Allowed exceptions:
# - `AgmonRecipe.r::Union{Nothing,AbstractVector}` is intentionally abstract, since the
#   residual is supplied by the caller (see issue #191).
# - `UniformRecipe` and `L2NormRecipe` have not been parameterized yet and are not
#   included.
@testset "Recipe Concreteness" begin
    Random.seed!(21)
    m, n = 40, 10
    A = rand(m, n)
    b = rand(m)
    x = zeros(n)

    @testset "Compressors: $name" for (name, technique) in (
        ("CountSketch", card -> CountSketch(compression_dim=5, cardinality=card)),
        ("FJLT", card -> FJLT(compression_dim=5, cardinality=card)),
        ("FixedPatternFJLT", card -> FixedPatternFJLT(compression_dim=5, cardinality=card)),
        ("Gaussian", card -> Gaussian(compression_dim=5, cardinality=card)),
        ("Identity", card -> Identity(cardinality=card)),
        ("Sampling", card -> Sampling(compression_dim=5, cardinality=card)),
        ("SparseSign", card -> SparseSign(compression_dim=5, cardinality=card)),
        ("SRHT", card -> SRHT(compression_dim=5, cardinality=card)),
    )
        for card in (Left(), Right())
            @test nonconcrete_fields(complete_compressor(technique(card), A)) == Symbol[]
        end
    end

    @testset "Distributions: Agmon" begin
        recipe = complete_distribution(Agmon(cardinality=Left()), x, A, b)
        @test nonconcrete_fields(recipe; exclude=(:r,)) == Symbol[]
    end

    @testset "Approximators" begin
        right = Gaussian(compression_dim=5, cardinality=Right())
        @test nonconcrete_fields(complete_approximator(RandSVD(compressor=right), A)) ==
            Symbol[]
        @test nonconcrete_fields(complete_approximator(RangeFinder(compressor=right), A)) ==
            Symbol[]
    end

    @testset "Selectors" begin
        @test nonconcrete_fields(complete_selector(LUPP(), A)) == Symbol[]
        @test nonconcrete_fields(complete_selector(QRCP(), A)) == Symbol[]
    end

    @testset "Loggers" begin
        @test nonconcrete_fields(complete_logger(BasicLogger(max_it=10))) == Symbol[]
    end

    @testset "Sub-solvers" begin
        @test nonconcrete_fields(complete_sub_solver(QRSolver(), A)) == Symbol[]
        @test nonconcrete_fields(complete_sub_solver(LQSolver(), Matrix(A'))) == Symbol[]
    end

    @testset "Solver errors: $name" for (name, err) in (
        ("FullResidual", FullResidual()),
        ("LSGradient", LSGradient()),
        ("CompressedResidual", CompressedResidual()),
    )
        @test nonconcrete_fields(complete_error(err, Kaczmarz(), A, b)) == Symbol[]
    end

    @testset "Solvers: $name" for (name, solver) in (
        ("Kaczmarz", Kaczmarz()),
        ("ColumnProjection", ColumnProjection()),
        ("IHS", IHS(compressor=Gaussian(compression_dim=20))),
    )
        @test nonconcrete_fields(complete_solver(solver, x, A, b)) == Symbol[]
    end
end

end
