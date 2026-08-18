module leverage_score_distribution
using Test, RandLinearAlgebra
using LinearAlgebra: pinv, diag
using StatsBase: ProbabilityWeights
using Random: seed!

seed!(4471)

# Ground-truth leverage scores computed independently of the src implementation,
# via the projection-matrix formula ℓ_i = (A(A'A)^{-1}A')_ii (and its A' analogue
# for columns), rather than by re-deriving the same QR call the source code uses.
left_leverage(A) = diag(A * pinv(A' * A) * A')
right_leverage(A) = diag(A' * pinv(A * A') * A)

@testset "LeverageScore" begin
    @testset "LeverageScore: Distribution" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(LeverageScore) == Distribution
        @test fieldnames(LeverageScore) ==
              (:cardinality, :replace, :compressor, :r2, :epsilon)
        @test fieldtypes(LeverageScore) == (
            Cardinality, Bool, Union{Nothing, Compressor}, Union{Nothing, Int}, Float64
        )

        # Default constructor
        let
            m = LeverageScore()
            @test m.cardinality == Undef()
            @test m.replace == false
            @test m.compressor === nothing
            @test m.r2 === nothing
            @test m.epsilon == 0.5
        end

        # Custom constructor
        let
            m2 = LeverageScore(cardinality = Left(), replace = true)
            @test m2.cardinality == Left()
            @test m2.replace == true
            @test m2.compressor === nothing
        end

        let
            m3 = LeverageScore(cardinality = Right(), replace = true)
            @test m3.cardinality == Right()
            @test m3.replace == true
        end

        # Constructor with a compressor supplied for approximate mode
        let
            comp = Gaussian(cardinality = Left(), compression_dim = 10)
            m4 = LeverageScore(cardinality = Left(), compressor = comp)
            @test m4.compressor === comp
        end

        # Custom epsilon
        let
            m5 = LeverageScore(cardinality = Left(), epsilon = 0.2)
            @test m5.epsilon == 0.2
        end
    end

    @testset "LeverageScore: DistributionRecipe" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(LeverageScoreRecipe) == DistributionRecipe
        @test fieldnames(LeverageScoreRecipe) ==
              (:cardinality, :replace, :state_space, :weights, :compressor_recipe, :r2)
        @test fieldtypes(LeverageScoreRecipe)[1:4] ==
              (Cardinality, Bool, Vector{Int64}, ProbabilityWeights)
    end

    @testset "LeverageScore: Complete Distribution - Exact Mode" begin
        # Left(), tall A: weights match the independent ground-truth formula and
        # sum to the rank of A
        let A = randn(20, 5),
            m = LeverageScore(cardinality = Left())

            mr = complete_distribution(m, A)
            @test mr.cardinality == Left()
            @test mr.replace == false
            @test mr.compressor_recipe === nothing
            @test mr.state_space == collect(1:20)
            @test Vector(mr.weights) ≈ left_leverage(A) atol = 1e-8
            @test sum(mr.weights) ≈ 5 atol = 1e-8
        end

        # Right(), fat A: weights match the independent ground-truth formula and
        # sum to the rank of A
        let A = randn(5, 20),
            m = LeverageScore(cardinality = Right())

            mr = complete_distribution(m, A)
            @test mr.cardinality == Right()
            @test mr.compressor_recipe === nothing
            @test mr.state_space == collect(1:20)
            @test Vector(mr.weights) ≈ right_leverage(A) atol = 1e-8
            @test sum(mr.weights) ≈ 5 atol = 1e-8
        end

        # Right(), tall A: column leverage scores of a tall matrix are trivially 1
        # for every column (Drineas et al. 2012, Section 1.1)
        let A = randn(20, 5),
            m = LeverageScore(cardinality = Right())

            mr = complete_distribution(m, A)
            @test Vector(mr.weights) ≈ ones(5) atol = 1e-8
        end

        # replace field is preserved into the recipe
        let A = randn(10, 3),
            m = LeverageScore(cardinality = Left(), replace = true)

            mr = complete_distribution(m, A)
            @test mr.replace == true
        end

        # Undef() cardinality throws
        let A = randn(10, 3),
            m = LeverageScore(cardinality = Undef())

            @test_throws ArgumentError complete_distribution(m, A)
        end
    end

    @testset "LeverageScore: Complete Distribution - Approximate Mode" begin
        # Approximate mode with Right() distribution cardinality is unsupported
        let A = randn(20, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 15),
            m = LeverageScore(cardinality = Right(), compressor = comp)

            @test_throws ArgumentError complete_distribution(m, A)
        end

        # Compressor must itself have Left() cardinality
        let A = randn(20, 5),
            comp = Gaussian(cardinality = Right(), compression_dim = 15),
            m = LeverageScore(cardinality = Left(), compressor = comp)

            @test_throws ArgumentError complete_distribution(m, A)
        end

        # Compressor's compression dimension must be at least size(A, 2) + 2
        # (the Wishart bias correction below is undefined otherwise)
        let A = randn(20, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 3),
            m = LeverageScore(cardinality = Left(), compressor = comp)

            @test_throws ArgumentError complete_distribution(m, A)
        end

        # Wishart bias correction: averaged over repeated draws (lower variance
        # than a single one), weights should be close to unbiased around d
        let d = 20,
            r1 = 40,
            trials = 60,
            sums = Float64[]

            for _ in 1:trials
                A = randn(300, d)
                comp = Gaussian(cardinality = Left(), compression_dim = r1)
                m = LeverageScore(cardinality = Left(), compressor = comp)
                mr = complete_distribution(m, A)
                push!(sums, sum(mr.weights))
            end
            @test sum(sums) / trials ≈ d rtol = 0.15
        end

        # r2 without a compressor is meaningless
        let A = randn(20, 5),
            m = LeverageScore(cardinality = Left(), r2 = 3)

            @test_throws ArgumentError complete_distribution(m, A)
        end

        # r2 must satisfy 1 <= r2 < size(A, 2)
        let A = randn(20, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 15),
            m = LeverageScore(cardinality = Left(), compressor = comp, r2 = 5)

            @test_throws ArgumentError complete_distribution(m, A)
        end

        # r2 fast path (Π2), requested explicitly: confirm the correction still
        # holds and r2 is carried into the recipe
        let d = 20,
            r1 = 40,
            r2_val = 6,
            trials = 60,
            sums = Float64[]

            for _ in 1:trials
                A = randn(300, d)
                comp = Gaussian(cardinality = Left(), compression_dim = r1)
                m = LeverageScore(cardinality = Left(), compressor = comp, r2 = r2_val)
                mr = complete_distribution(m, A)
                @test mr.r2 == r2_val
                push!(sums, sum(mr.weights))
            end
            @test sum(sums) / trials ≈ d rtol = 0.15
        end

        # r2's automatic default (drineas2012fast's Lemma 1 JL bound) activates
        # when it is actually smaller than d, and its value matches the formula;
        # epsilon is pushed toward 1 so the bound is small enough to beat a
        # moderately large d, matching the regime this default is meant for
        let n_rows = 500,
            n_cols = 200,
            r1 = 400,
            eps = 0.9,
            A = randn(n_rows, n_cols),
            comp = Gaussian(cardinality = Left(), compression_dim = r1),
            m = LeverageScore(cardinality = Left(), compressor = comp, epsilon = eps)

            expected_r2 = ceil(Int, (12 * log(n_rows) + 6 * log(10)) / eps^2)
            @test expected_r2 < n_cols  # sanity: test only meaningful if it helps

            mr = complete_distribution(m, A)
            @test mr.r2 == expected_r2
            w = Vector(mr.weights)
            @test all(w .> 0)
            @test sum(w) ≈ n_cols rtol = 0.5
        end

        # ...and falls back to the full AR⁻¹ path (r2 === nothing) when the JL
        # bound would not be smaller than d -- the common case for the small
        # matrices and default epsilon used throughout the rest of this file
        let A = randn(200, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 100),
            m = LeverageScore(cardinality = Left(), compressor = comp)

            mr = complete_distribution(m, A)
            @test mr.r2 === nothing
        end

        # epsilon must lie in (0, 1)
        let A = randn(20, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 15),
            m = LeverageScore(cardinality = Left(), compressor = comp, epsilon = 1.0)

            @test_throws ArgumentError complete_distribution(m, A)
        end

        # Weights should approximate the exact leverage scores. Theorem 2 in
        # drineas2012fast is an 80%-probability guarantee PER ROW, not across all
        # rows at once, so we check that most rows (not every row) land within
        # tolerance, rather than a worst-case-over-all-rows bound that would be
        # flaky by construction (see the "Approximate Mode Accuracy" note).
        let A = randn(200, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 100),
            m = LeverageScore(cardinality = Left(), compressor = comp),
            m_exact = LeverageScore(cardinality = Left())

            mr = complete_distribution(m, A)
            mr_exact = complete_distribution(m_exact, A)
            w = Vector(mr.weights)
            w_exact = Vector(mr_exact.weights)

            @test mr.cardinality == Left()
            @test mr.compressor_recipe !== nothing
            @test length(mr.state_space) == 200
            @test all(w .> 0)
            @test sum(w) ≈ 5 rtol = 0.35
            @test sum(abs.(w .- w_exact) ./ w_exact .< 0.5) / length(w) >= 0.7
        end
    end

    @testset "LeverageScore: Update Distribution" begin
        # Exact Left(): update recomputes weights and resizes state_space for a
        # matrix with a different number of rows
        let A1 = randn(10, 3),
            A2 = randn(15, 3),
            m = LeverageScore(cardinality = Left()),
            mr = complete_distribution(m, A1)

            @test length(mr.state_space) == 10

            update_distribution!(mr, A2)
            @test mr.state_space == collect(1:15)
            @test Vector(mr.weights) ≈ left_leverage(A2) atol = 1e-8
        end

        # Exact Left(): update with a same-size matrix still recomputes weights
        let A1 = randn(10, 3),
            A2 = randn(10, 3),
            m = LeverageScore(cardinality = Left()),
            mr = complete_distribution(m, A1)

            update_distribution!(mr, A2)
            @test mr.state_space == collect(1:10)
            @test Vector(mr.weights) ≈ left_leverage(A2) atol = 1e-8
        end

        # Exact Right(): update recomputes weights and resizes state_space
        let A1 = randn(3, 10),
            A2 = randn(3, 15),
            m = LeverageScore(cardinality = Right()),
            mr = complete_distribution(m, A1)

            @test length(mr.state_space) == 10

            update_distribution!(mr, A2)
            @test mr.state_space == collect(1:15)
            @test Vector(mr.weights) ≈ right_leverage(A2) atol = 1e-8
        end

        # Undef() cardinality guard
        let A = randn(10, 3),
            m = LeverageScore(cardinality = Left()),
            mr = complete_distribution(m, A)

            mr.cardinality = Undef()
            @test_throws ArgumentError update_distribution!(mr, A)
        end

        # Approximate mode: update recomputes weights via the stored compressor
        # recipe and still approximates the exact leverage scores (see the
        # fraction-based rationale in the "Complete Distribution" test above)
        let A1 = randn(200, 5),
            A2 = randn(200, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 100),
            m = LeverageScore(cardinality = Left(), compressor = comp),
            mr = complete_distribution(m, A1)

            update_distribution!(mr, A2)
            mr_exact = complete_distribution(LeverageScore(cardinality = Left()), A2)
            w = Vector(mr.weights)
            w_exact = Vector(mr_exact.weights)

            @test sum(w) ≈ 5 rtol = 0.35
            @test sum(abs.(w .- w_exact) ./ w_exact .< 0.5) / length(w) >= 0.7
        end

        # Approximate mode: changing the row dimension throws, since the stored
        # compressor recipe was sized for the original matrix
        let A1 = randn(200, 5),
            A2 = randn(150, 5),
            comp = Gaussian(cardinality = Left(), compression_dim = 100),
            m = LeverageScore(cardinality = Left(), compressor = comp),
            mr = complete_distribution(m, A1)

            @test_throws ArgumentError update_distribution!(mr, A2)
        end

        # Approximate mode update, r2 fast path: same averaged-sum verification
        # as the "Complete Distribution" case above
        let d = 20,
            r1 = 40,
            r2_val = 6,
            trials = 60,
            sums = Float64[]

            comp = Gaussian(cardinality = Left(), compression_dim = r1)
            m = LeverageScore(cardinality = Left(), compressor = comp, r2 = r2_val)
            mr = complete_distribution(m, randn(300, d))
            for _ in 1:trials
                update_distribution!(mr, randn(300, d))
                push!(sums, sum(mr.weights))
            end
            @test sum(sums) / trials ≈ d rtol = 0.15
        end
    end

    @testset "LeverageScore: Sample Distribution" begin
        # Left(), single sample within range
        let A = randn(20, 4),
            m = LeverageScore(cardinality = Left()),
            mr = complete_distribution(m, A)

            out = zeros(Int, 1)
            sample_distribution!(out, mr)
            @test 1 <= out[1] <= 20
        end

        # Right(), single sample within range
        let A = randn(4, 20),
            m = LeverageScore(cardinality = Right()),
            mr = complete_distribution(m, A)

            out = zeros(Int, 1)
            sample_distribution!(out, mr)
            @test 1 <= out[1] <= 20
        end

        # replace = true: repeated sampling remains valid
        let A = randn(10, 3),
            m = LeverageScore(cardinality = Left(), replace = true),
            mr = complete_distribution(m, A)

            out = zeros(Int, 5)
            for _ in 1:20
                sample_distribution!(out, mr)
                @test all(1 .<= out .<= 10)
            end
        end

        # replace = false: multiple output indices are distinct and within range
        let A = randn(15, 4),
            m = LeverageScore(cardinality = Left(), replace = false),
            mr = complete_distribution(m, A)

            out = zeros(Int, 5)
            sample_distribution!(out, mr)
            @test length(unique(out)) == 5
            @test all(1 .<= out .<= 15)
        end

        # Sampling frequency should reflect the leverage-score weights: row 1 of
        # this matrix has leverage score 1.0 (probability 1/2) while rows 2-4 each
        # have leverage score 1/3 (probability 1/6), so row 1 should dominate draws
        let A = [100.0 0.0; 0.0 1.0; 0.0 1.0; 0.0 1.0],
            m = LeverageScore(cardinality = Left(), replace = true),
            mr = complete_distribution(m, A)

            out = zeros(Int, 1)
            counts = zeros(Int, 4)
            for _ in 1:500
                sample_distribution!(out, mr)
                counts[out[1]] += 1
            end
            @test counts[1] > 2 * maximum(counts[2:end])
        end
    end
end

end # module
