module uniform_distribution
using Test, RandLinearAlgebra
using StatsBase: ProbabilityWeights

@testset "Uniform" begin
    @testset "Uniform: Distribution" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(Uniform) == Distribution
        @test fieldnames(Uniform) == (:cardinality, :replace)
        @test fieldtypes(Uniform) == (Cardinality, Bool)

        # Default
        let u = Uniform()
            @test u.cardinality == Undef()
            @test u.replace == false
        end

        # check other constructor
        let u2 = Uniform(cardinality = Left(), replace = true)
            @test u2.cardinality == Left()
            @test u2.replace == true
        end

        let u3 = Uniform(cardinality = Right(), replace = true)
            @test u3.cardinality == Right()
            @test u3.replace == true
        end

    end

    @testset "Uniform: DistributionRecipe" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(UniformRecipe) == DistributionRecipe
        @test fieldnames(UniformRecipe) == (:cardinality, :replace, :state_space, :weights)
        @test fieldtypes(UniformRecipe) == (Cardinality, Bool, Vector{Int64}, ProbabilityWeights)
    end

    @testset "Uniform: Complete Distribution" begin
        # Test with left compressor
        let A = randn(5, 3), 
            u = Uniform(cardinality = Left()),
            ur = complete_distribution(u, A)

            @test ur.cardinality == Left()
            @test length(ur.state_space) == 5
            @test ur.weights == ProbabilityWeights(ones(5))
        end
        
        # Test with right compressor
        let A = randn(3, 5), 
            u = Uniform(cardinality = Right()),
            ur = complete_distribution(u, A)

            @test ur.cardinality == Right()
            @test length(ur.state_space) == 5
            @test ur.weights == ProbabilityWeights(ones(5))
        end

        # Test with undef compressor
        let A = randn(3, 5), 
            u = Uniform(cardinality = Undef())

            @test_throws ArgumentError complete_distribution(complete_distribution(u, A))
        end

    end

    @testset "Uniform: Update Distribution" begin
        # Test if the result changes after updating
        let A = randn(3, 5),
            A2 = randn(4, 6),
            u = Uniform(cardinality = Right()),
            ur = complete_distribution(u, A)
            
            update_distribution!(ur, A2)
            @test ur.cardinality == Right()
            @test ur.state_space == collect(1:6)
            @test ur.weights == ProbabilityWeights(ones(6))
        end

        let A = randn(5, 3),
            A2 = randn(6, 4),
            u = Uniform(cardinality = Left()),
            ur = complete_distribution(u, A)
            
            update_distribution!(ur, A2)
            @test ur.cardinality == Left()
            @test ur.state_space == collect(1:6)
            @test ur.weights == ProbabilityWeights(ones(6))
        end

        let A = randn(5, 3),
            A2 = randn(6, 4),
            card = Undef(),
            replace = false,
            state_space = collect(1:5),
            weights = ProbabilityWeights(ones(5))
            ur = UniformRecipe(card, replace, state_space, weights)
            
            @test_throws ArgumentError update_distribution!(ur, A2)
        end       

    end

    @testset "Uniform: Sample Distribution" begin
        # Test if the sample_distribution! is returning the correct prob weights
        let A = randn(3, 5),
            x = randn(4),
            u = Uniform(cardinality = Right()),
            ur = complete_distribution(u, A)
            
            sample_distribution!(x, ur)
            @test ur.cardinality == Right()
            @test all(s -> 1 <= s <= 5, x)
        end

    end

end

end


