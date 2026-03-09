module motzkin_distribution
using Test, RandLinearAlgebra
using LinearAlgebra: dot

@testset "Motzkin" begin
    @testset "Motzkin: Distribution" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(Motzkin) == Distribution
        @test fieldnames(Motzkin) == (:cardinality, :replace, :beta)
        @test fieldtypes(Motzkin) == (Cardinality, Bool, Int)

        # Default constructor
        let 
            m = Motzkin()
            @test m.cardinality == Undef()
            @test m.replace == false
            @test m.beta == 1
        end

        # Custom constructor
        let 
            m2 = Motzkin(cardinality = Left(), replace = true, beta = 10)
            @test m2.cardinality == Left()
            @test m2.replace == true
            @test m2.beta == 10
        end

        # Test beta validation in constructor
        @test_throws ArgumentError Motzkin(beta = 0)
        @test_throws ArgumentError Motzkin(beta = -5)
        
        # Test cardinality validation in constructor
        @test_throws ArgumentError Motzkin(cardinality = Right())
    end

    @testset "Motzkin: DistributionRecipe" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(MotzkinRecipe) == DistributionRecipe
        @test fieldnames(MotzkinRecipe) == (:cardinality, :replace, :beta, :state_space, 
                                             :sample_buffer, :A, :b, :x)
        @test fieldtypes(MotzkinRecipe)[1:5] == (Cardinality, Bool, Int, Vector{Int64}, Vector{Int64})
    end

    @testset "Motzkin: Complete Distribution" begin
        # Test with valid Left cardinality
        let A = [1.0 0.0; 0.0 2.0; 1.0 1.0], 
            b = [1.0, 2.0, 1.0],
            x = [0.5, 0.5],
            m = Motzkin(cardinality = Left(), beta = 2)
            
            mr = complete_distribution(m, x, A, b)
            @test mr.cardinality == Left()
            @test mr.beta == 2
            @test mr.replace == false
            @test length(mr.state_space) == 3
            @test mr.state_space == [1, 2, 3]
            @test length(mr.sample_buffer) == 2
            @test mr.A === A  # Reference, not copy
            @test mr.b === b
            @test mr.x === x
        end

        # Test with Undef cardinality (should throw)
        let A = randn(5, 3), 
            b = randn(5),
            x = randn(3),
            m = Motzkin(cardinality = Undef(), beta = 2)

            @test_throws ArgumentError complete_distribution(m, x, A, b)
        end

        # Test beta > n_rows validation
        let A = randn(5, 3),
            b = randn(5),
            x = randn(3),
            m = Motzkin(cardinality = Left(), beta = 10)

            @test_throws ArgumentError complete_distribution(m, x, A, b)
        end

        # Test dimension mismatch: b length
        let A = randn(5, 3),
            b = randn(4),  # Wrong length
            x = randn(3),
            m = Motzkin(cardinality = Left(), beta = 2)

            @test_throws DimensionMismatch complete_distribution(m, x, A, b)
        end

        # Test dimension mismatch: x length
        let A = randn(5, 3),
            b = randn(5),
            x = randn(4),  # Wrong length
            m = Motzkin(cardinality = Left(), beta = 2)

            @test_throws DimensionMismatch complete_distribution(m, x, A, b)
        end
    end

    @testset "Motzkin: Update Distribution" begin
        # Test updating with new x
        let A = randn(5, 3),
            b = randn(5),
            x1 = randn(3),
            x2 = randn(3),
            m = Motzkin(cardinality = Left(), beta = 2),
            mr = complete_distribution(m, x1, A, b)
            
            @test mr.x === x1
            
            update_distribution!(mr, x2, A, b)
            @test mr.x === x2
        end

        # Test dimension change handling (state_space should update)
        let A = randn(5, 3),
            b = randn(5),
            x = randn(3),
            A2 = randn(7, 3),
            b2 = randn(7),
            m = Motzkin(cardinality = Left(), beta = 2),
            mr = complete_distribution(m, x, A, b)
            
            @test length(mr.state_space) == 5
            
            update_distribution!(mr, x, A2, b2)
            @test length(mr.state_space) == 7
            @test mr.state_space == collect(1:7)
        end

        # Test sample_buffer resizing when beta doesn't change
        let A = randn(5, 3),
            b = randn(5),
            x = randn(3),
            A2 = randn(7, 3),
            b2 = randn(7),
            m = Motzkin(cardinality = Left(), beta = 2),
            mr = complete_distribution(m, x, A, b)
            
            @test length(mr.sample_buffer) == 2
            
            update_distribution!(mr, x, A2, b2)
            @test length(mr.sample_buffer) == 2
        end

        # Test beta validation in update
        let A = randn(5, 3),
            b = randn(5),
            x = randn(3),
            A2 = randn(3, 3),  # Only 3 rows now
            b2 = randn(3),
            m = Motzkin(cardinality = Left(), beta = 5),  # beta > new n_rows
            mr = complete_distribution(m, x, A, b)
            
            @test_throws ArgumentError update_distribution!(mr, x, A2, b2)
        end

        # Test dimension mismatch: b length
        let A = randn(5, 3),
            b = randn(5),
            x = randn(3),
            b2 = randn(4),  # Wrong length
            m = Motzkin(cardinality = Left(), beta = 2),
            mr = complete_distribution(m, x, A, b)

            @test_throws DimensionMismatch update_distribution!(mr, x, A, b2)
        end

        # Test dimension mismatch: x length
        let A = randn(5, 3),
            b = randn(5),
            x = randn(3),
            x2 = randn(4),  # Wrong length
            m = Motzkin(cardinality = Left(), beta = 2),
            mr = complete_distribution(m, x, A, b)

            @test_throws DimensionMismatch update_distribution!(mr, x2, A, b)
        end
    end

    @testset "Motzkin: Sample Distribution - Beta Cases" begin
        # Test beta = 1 (pure random)
        let A = randn(10, 3),
            b = randn(10),
            x = randn(3),
            m = Motzkin(cardinality = Left(), beta = 1),
            mr = complete_distribution(m, x, A, b)
            
            update_distribution!(mr, x, A, b)
            out = zeros(Int, 1)
            sample_distribution!(out, mr)
            
            @test 1 <= out[1] <= 10
        end

        # Test beta >= n_rows (pure greedy) - should pick max residual
        let A = [1.0 0.0; 0.0 1.0; 0.0 0.0],
            b = [1.0, 5.0, 0.1],  # Row 2 has largest residual for x=0
            x = [0.0, 0.0],
            m = Motzkin(cardinality = Left(), beta = 3),
            mr = complete_distribution(m, x, A, b)
            
            update_distribution!(mr, x, A, b)
            out = zeros(Int, 1)
            sample_distribution!(out, mr)
            
            # Residuals: |0-1|=1, |0-5|=5, |0-0.1|=0.1 → should pick row 2
            @test out[1] == 2
        end

        # Test 1 < beta < n_rows (SKM) - should pick from sampled subset
        let A = randn(100, 5),
            b = randn(100),
            x = randn(5),
            m = Motzkin(cardinality = Left(), beta = 10),
            mr = complete_distribution(m, x, A, b)
            
            update_distribution!(mr, x, A, b)
            out = zeros(Int, 1)
            sample_distribution!(out, mr)
            
            # Should return a valid row index
            @test 1 <= out[1] <= 100
        end
    end

    @testset "Motzkin: Sample Distribution - Correctness" begin
        # Verify greedy selection picks correct max residual row
        let A = [2.0 1.0; 1.0 3.0; 0.5 0.5; 1.0 1.0],
            b = [1.0, 2.0, 3.0, 0.5],
            x = [0.5, 0.5],
            m = Motzkin(cardinality = Left(), beta = 4),  # beta >= n_rows
            mr = complete_distribution(m, x, A, b)
            
            update_distribution!(mr, x, A, b)
            
            # Compute residuals manually
            residuals = [abs(dot(A[i, :], x) - b[i]) for i in 1:4]
            expected_row = argmax(residuals)
            
            out = zeros(Int, 1)
            sample_distribution!(out, mr)
            
            @test out[1] == expected_row
        end

        # Verify update changes which row is selected
        let A = [1.0 0.0; 0.0 1.0],
            b = [2.0, 3.0],
            x1 = [0.0, 0.0],  # Residuals: [2, 3] → row 2
            x2 = [10.0, 0.0], # Residuals: [8, 3] → row 1
            m = Motzkin(cardinality = Left(), beta = 2),
            mr = complete_distribution(m, x1, A, b)
            
            out = zeros(Int, 1)
            
            # First update and sample
            update_distribution!(mr, x1, A, b)
            sample_distribution!(out, mr)
            @test out[1] == 2
            
            # Second update with different x and sample
            update_distribution!(mr, x2, A, b)
            sample_distribution!(out, mr)
            @test out[1] == 1
        end
    end

    @testset "Motzkin: Sample Distribution - Output Validity" begin
        # Test that output is always within valid range
        let A = randn(50, 10),
            b = randn(50),
            x = randn(10),
            m = Motzkin(cardinality = Left(), beta = 5),
            mr = complete_distribution(m, x, A, b)
            
            out = zeros(Int, 1)
            
            # Run multiple samples
            for _ in 1:100
                update_distribution!(mr, x, A, b)
                sample_distribution!(out, mr)
                @test 1 <= out[1] <= 50
            end
        end

        # Test that beta=1 produces varied samples (not stuck on one row)
        let A = randn(20, 5),
            b = randn(20),
            x = randn(5),
            m = Motzkin(cardinality = Left(), beta = 1),
            mr = complete_distribution(m, x, A, b)
            
            out = zeros(Int, 1)
            samples = Int[]
            
            # Collect 50 samples
            for _ in 1:50
                update_distribution!(mr, x, A, b)
                sample_distribution!(out, mr)
                push!(samples, out[1])
            end
            
            # Should see multiple different rows (very unlikely to hit same row 50 times)
            @test length(unique(samples)) > 1
        end
    end
end

end # module
