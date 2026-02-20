module Sampling_compressor
using Test, RandLinearAlgebra, Random
using StatsBase: ProbabilityWeights, sample
import LinearAlgebra: mul!, Adjoint
import SparseArrays: sprandn
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "Sampling" begin
    @testset "Sampling: Compressor" begin
        # Verify Supertype
        @test supertype(Sampling) == Compressor

        # Verify fields and types
        @test fieldnames(Sampling) == (:cardinality, :compression_dim, :distribution)
        @test fieldtypes(Sampling) == (Cardinality, Int64, Distribution)

        # Verify the Internal Constructor
        let cardinality = Left(), compression_dim = 0, distribution = Uniform()
            @test_throws ArgumentError(
                "Field `compression_dim` must be positive."
            ) Sampling(
                cardinality, compression_dim, distribution
            )
        end

        let cardinality = Undef(), compression_dim = 2, distribution = Uniform()
            @test_throws ArgumentError(
                "`cardinality` must be specified as `Left()` or `Right()`.\
                `Undef()` is not allowed in `Sampling` structure."
            ) Sampling(
                cardinality, compression_dim, distribution
            )
        end

        # Default values
        let sc_default = Sampling()
            @test sc_default.cardinality == Left()
            @test sc_default.compression_dim == 2
            @test sc_default.distribution isa Uniform
            # Check defaults of the default Uniform distribution instance
            @test sc_default.distribution.cardinality == Undef()
            @test sc_default.distribution.replace == false
        end

        # Specific cardinality
        let card_val = Right(),
            dist_instance = Uniform(cardinality=card_val), 
            compressor = Sampling(; cardinality=card_val, distribution=dist_instance)

            @test compressor.cardinality == card_val
            @test compressor.distribution.cardinality == card_val
        end

        let card_val = Left(),
            dist_instance = Uniform(cardinality=card_val),
            compressor = Sampling(; cardinality=card_val, distribution=dist_instance)

            typeof(compressor.cardinality) == Cardinality
            @test compressor.cardinality == card_val
            @test compressor.distribution.cardinality == card_val
        end

        # Test with specific compression_dim
        let comp_dim_val = 15,
            compressor = Sampling(; compression_dim=comp_dim_val)
            
            @test compressor.compression_dim == comp_dim_val
        end

        # Test with specific distribution instance
        let my_dist = Uniform(cardinality=Right(), replace=true),
            compressor = Sampling(; distribution=my_dist)

            @test compressor.distribution === my_dist
            @test compressor.distribution.cardinality == Right()
            @test compressor.distribution.replace == true
            # Default cardinality and compression_dim should be used
            @test compressor.cardinality == Left()
            @test compressor.compression_dim == 2
        end

        # Test combination of specific arguments
        let final_card = Right(), final_cdim = 22,
            final_dist = Uniform(cardinality=final_card, replace=false),
            compressor = Sampling(; cardinality=final_card, compression_dim=final_cdim, distribution=final_dist)

            @test compressor.cardinality == final_card
            @test compressor.compression_dim == final_cdim
            @test compressor.distribution === final_dist
        end
    end

    @testset "Sampling: CompressorRecipe" begin
        @test_compressor SamplingRecipe
        @test fieldnames(SamplingRecipe) == (
            :cardinality,
            :compression_dim,
            :n_rows,
            :n_cols,
            :distribution_recipe,
            :idx,
            :idx_v
        )
        @test fieldtypes(SamplingRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Int64,
            DistributionRecipe,
            Vector{Int64},
            SubArray
        )

        let card_type = Left, 
            comp_dim = 5,
            actual_n_rows = comp_dim,
            actual_n_cols = 20,           
            replace_sampling = false

            # Setup for distribution_recipe 
            dist_state_space = collect(1:actual_n_cols)
            dist_weights = ProbabilityWeights(ones(actual_n_cols) ./ actual_n_cols)
            dist_recipe_val = UniformRecipe(card_type(), replace_sampling, dist_state_space, dist_weights)

            # Setup for idx and idx_v
            idx_val = sort(sample(dist_state_space, comp_dim, replace=replace_sampling))
            idx_v_val = view(idx_val, :)

            # Construct SamplingRecipe
            recipe = SamplingRecipe{card_type}(card_type(), comp_dim, actual_n_rows, actual_n_cols,
                                         dist_recipe_val, idx_val, idx_v_val)

            @test typeof(recipe) == SamplingRecipe{card_type}
            @test recipe.cardinality == card_type() 
            @test isa(recipe.cardinality, card_type)  
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == actual_n_rows
            @test recipe.n_cols == actual_n_cols
            @test recipe.distribution_recipe == dist_recipe_val
            @test isa(recipe.distribution_recipe, UniformRecipe)
            @test recipe.idx == idx_val
            @test recipe.idx_v == idx_v_val
            @test eltype(recipe.idx_v) == eltype(idx_val)
            @test parent(recipe.idx_v) === recipe.idx 
        end

        let card_type = Right, 
            comp_dim = 6,
            actual_n_rows = 25,       
            actual_n_cols = comp_dim, 
            replace_sampling = true

            # Setup for distribution_recipe 
            dist_state_space = collect(1:actual_n_rows) 
            dist_weights = ProbabilityWeights(ones(actual_n_rows) ./ actual_n_rows)
            dist_recipe_val = UniformRecipe(card_type(), replace_sampling, dist_state_space, dist_weights)

            # Setup for idx and idx_v
            idx_val = sort(sample(dist_state_space, comp_dim, replace=replace_sampling))
            idx_v_val = view(idx_val, :)

            # Construct SamplingRecipe
            recipe = SamplingRecipe{card_type}(card_type(), comp_dim, actual_n_rows, actual_n_cols,
                                         dist_recipe_val, idx_val, idx_v_val)

            @test typeof(recipe) == SamplingRecipe{card_type}
            @test recipe.cardinality == card_type()
            @test isa(recipe.cardinality, card_type)
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == actual_n_rows
            @test recipe.n_cols == actual_n_cols
            @test recipe.distribution_recipe == dist_recipe_val
            @test isa(recipe.distribution_recipe, UniformRecipe)
            @test recipe.idx == idx_val
            @test recipe.idx_v == idx_v_val
            @test eltype(recipe.idx_v) == eltype(idx_val)
            @test parent(recipe.idx_v) === recipe.idx
        end

    end

    @testset "Sampling: complete_compressor" begin
        let card_instance = Left(),
            a_matrix_rows = 10, 
            a_matrix_cols = 8,  
            comp_dim = 4,  
            replace_sampling = false 

            A = randn(a_matrix_rows, a_matrix_cols)

            # Create the distribution instance. Its cardinality will be updated by complete_compressor
            dist_instance = Uniform(cardinality=Undef(), replace=replace_sampling)
            
            sub_comp_settings = Sampling(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)
            
            # First, check it's not already set to card_instance
            if dist_instance.cardinality != card_instance
                @test dist_instance.cardinality != card_instance 
            end

            recipe = complete_compressor(sub_comp_settings, A)

            # Sampling.distribution.cardinality should be updated
            @test dist_instance.cardinality == card_instance
            
            # Test the recipe values and types. 
            # For Left, recipe.n_rows = compression dim, recipe.n_cols = A's rows
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == comp_dim       
            @test recipe.n_cols == a_matrix_rows  
            
            @test recipe.distribution_recipe isa UniformRecipe
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_rows)
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_rows) 

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64

            # Check indices inside rows of A
            @test all(x -> 1 <= x <= a_matrix_rows, recipe.idx) 

            # Test no replacement
            @test length(unique(recipe.idx)) == comp_dim
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end

        let card_instance = Right(),
            a_matrix_rows = 12, 
            a_matrix_cols = 15, 
            comp_dim = 5,         
            replace_sampling = true

            # Define A inside the let block
            A = randn(a_matrix_rows, a_matrix_cols)
            
            # Create the distribution instance
            dist_instance = Uniform(cardinality=Undef(), replace=replace_sampling)
            
            sub_comp_settings = Sampling(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)

            # The distribution's cardinality should not be the same
            if dist_instance.cardinality != card_instance
                @test dist_instance.cardinality != card_instance
            end
            
            recipe = complete_compressor(sub_comp_settings, A)

            # Verify the update
            @test dist_instance.cardinality == card_instance

            # Test the recipe values and types
            # For Right, recipe.n_rows =  A's columns, recipe.n_cols = compression dim
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == a_matrix_cols  
            @test recipe.n_cols == comp_dim        
            
            @test recipe.distribution_recipe isa UniformRecipe
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_cols)
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_cols) 

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64

            # Check indices inside cols of A
            @test all(x -> 1 <= x <= a_matrix_cols, recipe.idx)
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end
    end

    @testset "Sampling: update_compressor!" begin
        let card_instance = Left(),
            a_matrix_rows = 100, 
            a_matrix_cols = 15,  
            comp_dim = 10,          
            replace_sampling = false 

            A = randn(a_matrix_rows, a_matrix_cols)
            x_dummy = randn(a_matrix_cols) 
            b_dummy = randn(a_matrix_rows) 
            
            dist_instance = Uniform(cardinality=card_instance, replace=replace_sampling)
            sub_comp_settings = Sampling(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)
            recipe = complete_compressor(sub_comp_settings, A)
            
            # Store the old indices to verify they change
            old_idx = deepcopy(recipe.idx)
            # Store old distribution recipe state if it's supposed to change and be testable
            # For Uniform, update_distribution! re-initializes based on A

            update_compressor!(recipe, x_dummy, A, b_dummy)

            # After updates, index should be changed
            @test recipe.idx != old_idx
            
            # Test invariant properties of the recipe after update
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == comp_dim
            @test recipe.n_cols == a_matrix_rows
            
            @test recipe.distribution_recipe isa UniformRecipe 
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance 
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_rows) 
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_rows)

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64
            @test all(val -> 1 <= val <= a_matrix_rows, recipe.idx)
            if !replace_sampling
                @test length(unique(recipe.idx)) == comp_dim
            end
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end

        let card_instance = Right(),
            a_matrix_rows = 15,
            a_matrix_cols = 100,
            comp_dim = 10,
            replace_sampling = false

            A = randn(a_matrix_rows, a_matrix_cols)
            x_dummy = randn(a_matrix_cols)
            b_dummy = randn(a_matrix_rows)

            dist_instance = Uniform(cardinality=card_instance, replace=replace_sampling)
            sub_comp_settings = Sampling(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)
            recipe = complete_compressor(sub_comp_settings, A)

            old_idx = deepcopy(recipe.idx)

            update_compressor!(recipe, x_dummy, A, b_dummy)

            # After updates, index should be changed
            @test recipe.idx != old_idx
    
            # Test unchanged values
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == a_matrix_cols
            @test recipe.n_cols == comp_dim

            @test recipe.distribution_recipe isa UniformRecipe
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_cols)
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_cols) 

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64
            @test all(val -> 1 <= val <= a_matrix_cols, recipe.idx)
            if !replace_sampling
                @test length(unique(recipe.idx)) == comp_dim
            end
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end
    end


    @testset "Sampling: Multiplication (mul!)" begin
        @testset "Left Cardinality" begin
            let a_matrix_rows = 20,
                a_matrix_cols = 12,
                comp_dim = 7,
                alpha = 2.5,
                beta = 1.5

                # Setup matrices and vectors 
                A = randn(a_matrix_rows, a_matrix_cols)
                B = randn(comp_dim, a_matrix_cols)
                C1 = randn(comp_dim, a_matrix_cols)
                C2 = randn(a_matrix_rows, a_matrix_cols)
                x = randn(a_matrix_rows)
                y = randn(comp_dim)

                # Keep copies for 5-argument mul! verification
                C1c = deepcopy(C1)
                C2c = deepcopy(C2)
                yc = deepcopy(y)
                xc = deepcopy(x)

                # Setup the Sampling compressor recipe
                S_info = Sampling(
                    cardinality=Left(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Left(), replace=false)
                )
                S = complete_compressor(S_info, A)

                # Calculate all ground truth results
                SA_exact = A[S.idx, :]
                StB_exact = zeros(a_matrix_rows, a_matrix_cols); for i in 1:comp_dim; StB_exact[S.idx[i], :] = B[i, :]; end
                Sx_exact = x[S.idx]
                Sty_exact = zeros(a_matrix_rows); for i in 1:comp_dim; Sty_exact[S.idx[i]] = y[i]; end

                # Test '*' operations by comparing to ground truths
                @test S * A ≈ SA_exact
                @test S' * B ≈ StB_exact
                @test A' * S' ≈ SA_exact'
                @test B' * S ≈ StB_exact'
                @test S * x ≈ Sx_exact
                @test x' * S' ≈ Sx_exact'
                @test S' * y ≈ Sty_exact
                @test y' * S ≈ Sty_exact'

                # Test the 5-argument mul!
                mul!(C1, S, A, alpha, beta)
                @test C1 ≈ alpha * SA_exact + beta * C1c

                mul!(C2, S', B, alpha, beta)
                @test C2 ≈ alpha * StB_exact + beta * C2c
                
                mul!(y, S, xc, alpha, beta)
                @test y ≈ alpha * Sx_exact + beta * yc

                mul!(x, S', yc, alpha, beta)
                @test x ≈ alpha * Sty_exact + beta * xc
            end
        end

        # Test multiplications with right compressors
        @testset "Right Cardinality" begin
            let n = 20,         
                comp_dim = 10,     
                alpha = 2.0,
                beta = 2.0

                # Setup matrices and vectors with dimensions
                A = randn(n, comp_dim)
                B = randn(n, n)
                # C1 is for S'*A, C2 is for B*S
                C1 = randn(n, n)
                C2 = randn(n, comp_dim)
                x = randn(comp_dim)
                y = randn(n)

                # Keep copies for 5-argument mul! verification
                C1c = deepcopy(C1)
                C2c = deepcopy(C2)
                yc = deepcopy(y)
                xc = deepcopy(x)

                # Setup the Sampling compressor recipe. It's created from B, an n x n matrix.
                # The operator S will have conceptual dimensions (n x comp_dim).
                S_info = Sampling(
                    cardinality=Right(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Right(), replace=false)
                )
                S = complete_compressor(S_info, B)

                # Calculate all ground truth results based on direct indexing/operations
                StA_exact = A[S.idx, :]
                BS_exact = B[:, S.idx]
                BtS_exact = B'[:, S.idx]
                ASt_exact = zeros(n, n); for i in 1:comp_dim; ASt_exact[:, S.idx[i]] = A[:, i]; end
                Sx_exact = zeros(n); for i in 1:comp_dim; Sx_exact[S.idx[i]] = x[i]; end
                Sty_exact = y[S.idx]

                # Test '*' operations by comparing to ground truths
                @test S' * A ≈ StA_exact
                @test A' * S ≈ StA_exact'
                @test B * S ≈ BS_exact
                @test S' * B' ≈ BS_exact'
                @test B' * S ≈ BtS_exact
                @test S' * B ≈ BtS_exact'
                @test A * S' ≈ ASt_exact
                @test S * A' ≈ ASt_exact'
                @test S * x ≈ Sx_exact
                @test x' * S' ≈ Sx_exact'
                @test y' * S ≈ Sty_exact'
                @test S' * y ≈ Sty_exact

                # Test the 5-argument mul!
                mul!(C1, A, S', alpha, beta)
                @test C1 ≈ alpha * ASt_exact + beta * C1c

                mul!(C2, B, S, alpha, beta)
                @test C2 ≈ alpha * BS_exact + beta * C2c

                mul!(y, S, xc, alpha, beta)
                @test y ≈ alpha * Sx_exact + beta * yc

                mul!(x, S', yc, alpha, beta)
                @test x ≈ alpha * Sty_exact + beta * xc
            end

        end
        
        @testset "Left Cardinality Sparse" begin
            let a_matrix_rows = 20,
                a_matrix_cols = 12,
                comp_dim = 7,
                alpha = 2.5,
                beta = 1.5

                # Setup matrices and vectors 
                A = sprandn(a_matrix_rows, a_matrix_cols, .8)
                B = sprandn(comp_dim, a_matrix_cols, .8)
                C1 = sprandn(comp_dim, a_matrix_cols, .8)
                C2 = sprandn(a_matrix_rows, a_matrix_cols, .8)
                x = sprandn(a_matrix_rows, .8)
                y = sprandn(comp_dim, .8)

                # Keep copies for 5-argument mul! verification
                C1c = deepcopy(C1)
                C2c = deepcopy(C2)
                yc = deepcopy(y)
                xc = deepcopy(x)

                # Setup the Sampling compressor recipe
                S_info = Sampling(
                    cardinality=Left(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Left(), replace=false)
                )
                S = complete_compressor(S_info, A)

                # Calculate all ground truth results
                SA_exact = A[S.idx, :]
                StB_exact = zeros(a_matrix_rows, a_matrix_cols); for i in 1:comp_dim; StB_exact[S.idx[i], :] = B[i, :]; end
                Sx_exact = x[S.idx]
                Sty_exact = zeros(a_matrix_rows); for i in 1:comp_dim; Sty_exact[S.idx[i]] = y[i]; end

                # Test '*' operations by comparing to ground truths
                @test S * A ≈ SA_exact
                @test S' * B ≈ StB_exact
                @test A' * S' ≈ SA_exact'
                @test B' * S ≈ StB_exact'
                @test S * x ≈ Sx_exact
                @test x' * S' ≈ Sx_exact'
                @test S' * y ≈ Sty_exact
                @test y' * S ≈ Sty_exact'

                # Test the 5-argument mul!
                mul!(C1, S, A, alpha, beta)
                @test C1 ≈ alpha * SA_exact + beta * C1c

                mul!(C2, S', B, alpha, beta)
                @test C2 ≈ alpha * StB_exact + beta * C2c
                
                mul!(y, S, xc, alpha, beta)
                @test y ≈ alpha * Sx_exact + beta * yc

                mul!(x, S', yc, alpha, beta)
                @test x ≈ alpha * Sty_exact + beta * xc
            end
        end

        @testset "Left Cardinality Sparse Transpose" begin
            let a_matrix_rows = 8,
                a_matrix_cols = 6,
                comp_dim = 3,
                alpha = 1.7,
                beta = 0.6

                # A is a Transpose of a sparse matrix
                B = sprandn(a_matrix_cols, a_matrix_rows, 0.5)
                A = transpose(B)

                C = randn(comp_dim, a_matrix_cols)
                C0 = deepcopy(C)

                S_info = Sampling(
                    cardinality=Left(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Left(), replace=false)
                )
                S = complete_compressor(S_info, A)

                A_dense = Matrix(A)
                SA_exact = A_dense[S.idx, :]

                mul!(C, S, A, alpha, beta)
                @test C ≈ alpha * SA_exact + beta * C0

                # Cover the beta == 0 fill! branch in the specialized method
                C = randn(comp_dim, a_matrix_cols)
                mul!(C, S, A, alpha, 0.0)
                @test C ≈ alpha * SA_exact
            end
        end

        # Test multiplications with right compressors
        @testset "Right Cardinality" begin
            let n = 20,         
                comp_dim = 10,     
                alpha = 2.0,
                beta = 2.0

                # Setup matrices and vectors with dimensions
                A = sprandn(n, comp_dim, .8)
                B = sprandn(n, n, .8)
                # C1 is for S'*A, C2 is for B*S
                C1 = sprandn(n, n, .8)
                C2 = sprandn(n, comp_dim, .8)
                x = sprandn(comp_dim, .8)
                y = sprandn(n, .8)

                # Keep copies for 5-argument mul! verification
                C1c = deepcopy(C1)
                C2c = deepcopy(C2)
                yc = deepcopy(y)
                xc = deepcopy(x)

                # Setup the Sampling compressor recipe. It's created from B, an n x n matrix.
                # The operator S will have conceptual dimensions (n x comp_dim).
                S_info = Sampling(
                    cardinality=Right(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Right(), replace=false)
                )
                S = complete_compressor(S_info, B)

                # Calculate all ground truth results based on direct indexing/operations
                StA_exact = A[S.idx, :]
                BS_exact = B[:, S.idx]
                BtS_exact = B'[:, S.idx]
                ASt_exact = zeros(n, n); for i in 1:comp_dim; ASt_exact[:, S.idx[i]] = A[:, i]; end
                Sx_exact = zeros(n); for i in 1:comp_dim; Sx_exact[S.idx[i]] = x[i]; end
                Sty_exact = y[S.idx]

                # Test '*' operations by comparing to ground truths
                @test S' * A ≈ StA_exact
                @test A' * S ≈ StA_exact'
                @test B * S ≈ BS_exact
                @test S' * B' ≈ BS_exact'
                @test B' * S ≈ BtS_exact
                @test S' * B ≈ BtS_exact'
                @test A * S' ≈ ASt_exact
                @test S * A' ≈ ASt_exact'
                @test S * x ≈ Sx_exact
                @test x' * S' ≈ Sx_exact'
                @test y' * S ≈ Sty_exact'
                @test S' * y ≈ Sty_exact

                # Test the 5-argument mul!
                mul!(C1, A, S', alpha, beta)
                @test C1 ≈ alpha * ASt_exact + beta * C1c

                mul!(C2, B, S, alpha, beta)
                @test C2 ≈ alpha * BS_exact + beta * C2c

                mul!(y, S, xc, alpha, beta)
                @test y ≈ alpha * Sx_exact + beta * yc

                mul!(x, S', yc, alpha, beta)
                @test x ≈ alpha * Sty_exact + beta * xc
            end
            
        end
    end

end

end
