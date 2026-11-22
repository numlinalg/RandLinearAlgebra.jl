module CrossApproximation_tests
using Test, RandLinearAlgebra, LinearAlgebra, SparseArrays, Random
import RandLinearAlgebra: complete_selector, select_indices!
import RandLinearAlgebra: complete_core, update_core!
# Write test selector recipe 
mutable struct TestSelector <: Selector end 
mutable struct TestSelectorRecipe <: SelectorRecipe end

function complete_selector(T::TestSelector, A::AbstractMatrix)
    return TestSelectorRecipe()
end

function select_indices!(
    idx::Vector{Int64}, 
    recipe::TestSelectorRecipe,
    A::AbstractMatrix,
    n::Int64,
    offset::Int64
)
    idx[1:n] = 1:n
end


@testset "CrossApproximation" begin
    @testset "CrossApproximation" begin
        @test supertype(CrossApproximation) == GeneralizedNystromCore

        # test the fieldnames and types
        @test fieldnames(CrossApproximation) == ()
        @test fieldtypes(CrossApproximation) == ()

        # test constructor
        core = CrossApproximation()

        @test typeof(core) == CrossApproximation

    end

    @testset "CrossApproximation Recipe" begin
        supertype(CrossApproximationRecipe) == GeneralizedNystromCoreRecipe

        # test the fieldnames and types
        @test fieldnames(CrossApproximationRecipe) == (
            :n_rows, :n_cols, :Q, :R, :buffer
        )
        @test fieldtypes(CrossApproximationRecipe) == (
            Int64, Int64, AbstractMatrix, AbstractMatrix, AbstractMatrix
        )
    end

    @testset "CrossApproximation: Complete Core" begin
        # test with matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(core, approx, A)
            @test size(recipe) == (rank, rank + oversample)
            # check that the Q and R factors are preallocated with the correct sizes
            @test size(recipe.Q) == (rank + oversample, rank)
            @test size(recipe.R) == (rank, rank)
        end

        # test with sparse matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)
            approx = CUR(
                rank = rank,
                oversample = oversample,
                selector_cols = TestSelector()
            )
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(core, approx, A)
            @test size(recipe) == (rank, rank + oversample)
            # check that the Q and R factors are preallocated with the correct sizes
            @test size(recipe.Q) == (rank + oversample, rank)
            @test size(recipe.R) == (rank, rank)
        end

    end
    
    @testset "CrossApproximation: Update Core" begin
        # test with sparse matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)
            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(core, approx, A)
            approx_recipe = complete_approximator(approx, A)
            # change indices selected to be first n
            approx_recipe.col_idx = 1:approx_recipe.n_col_vecs
            approx_recipe.row_idx = 1:approx_recipe.n_row_vecs
            # update the core recipe
            update_core!(recipe, approx_recipe, A)
            # check that a QR decomposition is stored of with correct sizes
            @test size(recipe) == (rank, rank + oversample)
        end

        # test with matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()
            # test the complete core function
            recipe = complete_core(core, approx, A)
            approx_recipe = complete_approximator(approx, A)
            # change indices selected to be first n
            approx_recipe.col_idx = 1:approx_recipe.n_col_vecs
            approx_recipe.row_idx = 1:approx_recipe.n_row_vecs
            # update the core recipe
            update_core!(recipe, approx_recipe, A)
            # check that a QR decomposition is stored of with correct sizes
            @test size(recipe) == (rank, rank + oversample)
        end

    end

    @testset "CrossApproximation: rapproximate" begin
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            
            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector(),
                core = CrossApproximation() 
            ) 
            approx_recipe = complete_approximator(approx, A)
            rapproximate!(approx_recipe, A)
            col_idx = approx_recipe.n_col_vecs
            row_idx = approx_recipe.n_row_vecs
            # test that the cur approximation corresponds to the first 2 columns and 
            # first 3 rows
            @test approx_recipe.col_idx == 1:col_idx
            @test approx_recipe.row_idx == 1:row_idx
            @test approx_recipe.C == A[:, 1:col_idx]
            @test approx_recipe.R == A[1:row_idx, :]
            @test typeof(approx_recipe.U) == CrossApproximationRecipe
            @test size(approx_recipe.U) == (col_idx, row_idx)
        end

        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)

            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector(), 
                core = CrossApproximation()
            ) 
            approx_recipe = complete_approximator(approx, A)
            rapproximate!(approx_recipe, A)
            col_idx = approx_recipe.n_col_vecs
            row_idx = approx_recipe.n_row_vecs
            # test that the cur approximation corresponds to the first 2 columns and 
            # first 3 rows
            @test approx_recipe.col_idx == 1:col_idx
            @test approx_recipe.row_idx == 1:row_idx
            @test approx_recipe.C == A[:, 1:col_idx]
            @test approx_recipe.R == A[1:row_idx, :]
            @test typeof(approx_recipe.U) == CrossApproximationRecipe
            @test size(approx_recipe.U) == (col_idx, row_idx)
        end

    end

    @testset "CrossApproximation: mul!" begin
        # test multiplying the core matrix from the left
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols),
            B = ones(rank, rank + 1),
            C = ones(rank + oversample, rank + 1),
            x = ones(rank),
            y = ones(rank + oversample),
            alpha = 0.2,
            beta = 1.0

            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector(),
                core = CrossApproximation()
            ) 
            recipe = complete_approximator(approx, A)
            rapproximate!(recipe, A) 
            # test for matrices
            Btest = deepcopy(B)
            mul!(B, recipe.U, C, alpha, beta)
            @test norm(B - ( beta * Btest + alpha * pinv(A[1:3, 1:2]) * C)) < 1e-10
            # test for vectors
            xtest = deepcopy(x)
            mul!(x, recipe.U, y, alpha, beta)
            @test norm(x - ( beta * xtest + alpha * pinv(A[1:3, 1:2]) * y)) < 1e-10
            # test for adjoints from the left
            Ctest = deepcopy(C)
            mul!(C, recipe.U', B, alpha, beta)
            @test norm(C - ( beta * Ctest + alpha * pinv(A[1:3, 1:2])' * B)) < 1e-10
            #test for adjoint with vector
            ytest = deepcopy(y)
            mul!(y, recipe.U', x, alpha, beta)
            @test norm(y - ( beta * ytest + alpha * pinv(A[1:3, 1:2])' * x)) < 1e-10
        end

        # test multiplying the core matrix from the right
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols),
            B = ones(rank + oversample, rank),
            C = ones(rank, rank + oversample),
            D = ones(rank + oversample, rank + oversample),
            E = ones(rank, rank),
            x = ones(rank + oversample),
            y = ones(rank),
            alpha = 0.2,
            beta = 1.0

            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector(),
                core = CrossApproximation()
            ) 
            recipe = complete_approximator(approx, A)
            rapproximate!(recipe, A) 
            # test for matrices
            Dtest = deepcopy(D)
            mul!(D, B, recipe.U, alpha, beta)
            @test norm(D - ( beta * Dtest + alpha * B * pinv(A[1:3, 1:2]))) < 1e-10
            # test for vectors
            xtest = deepcopy(x)
            mul!(x', y', recipe.U, alpha, beta)
            @test norm(x' - ( beta * xtest' + alpha * y' * pinv(A[1:3, 1:2]))) < 1e-10
            # test for adjoints from the left
            Etest = deepcopy(E)
            mul!(E, C, recipe.U', alpha, beta)
            @test norm(E - ( beta * Etest + alpha *  C * pinv(A[1:3, 1:2])')) < 1e-10
            #test for adjoint with vector
            ytest = deepcopy(y)
            mul!(y', x', recipe.U', alpha, beta)
            @test norm(y' - ( beta * ytest' + alpha * x' * pinv(A[1:3, 1:2])')) < 1e-10
        end

        # same tests with sparse matrices
        # test multiplying the core matrix from the left
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9),
            B = ones(rank, rank + 1),
            C = ones(rank + oversample, rank + 1),
            x = ones(rank),
            y = ones(rank + oversample),
            alpha = 0.2,
            beta = 1.0

            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector(),
                core = CrossApproximation()
            ) 
            recipe = complete_approximator(approx, A)
            rapproximate!(recipe, A) 
            # test for matrices
            Btest = deepcopy(B)
            mul!(B, recipe.U, C, alpha, beta)
            @test norm(B - ( beta * Btest + alpha * pinv(Array(A[1:3, 1:2])) * C)) < 1e-10
            # test for vectors
            xtest = deepcopy(x)
            mul!(x, recipe.U, y, alpha, beta)
            @test norm(x - ( beta * xtest + alpha * pinv(Array(A[1:3, 1:2])) * y)) < 1e-10
            # test for adjoints from the left
            Ctest = deepcopy(C)
            mul!(C, recipe.U', B, alpha, beta)
            @test norm(C - ( beta * Ctest + alpha * pinv(Array(A[1:3, 1:2]))' * B)) < 1e-10
            #test for adjoint with vector
            ytest = deepcopy(y)
            mul!(y, recipe.U', x, alpha, beta)
            @test norm(y - ( beta * ytest + alpha * pinv(Array(A[1:3, 1:2]))' * x)) < 1e-10
        end

        # test multiplying the core matrix from the right
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9),
            B = ones(rank + oversample, rank),
            C = ones(rank, rank + oversample),
            D = ones(rank + oversample, rank + oversample),
            E = ones(rank, rank),
            x = ones(rank + oversample),
            y = ones(rank),
            alpha = 0.2,
            beta = 1.0

            approx = CUR(
                rank = rank, 
                oversample = oversample, 
                selector_cols = TestSelector(),
                core = CrossApproximation()
            ) 
            recipe = complete_approximator(approx, A)
            rapproximate!(recipe, A) 
            # test for matrices
            Dtest = deepcopy(D)
            mul!(D, B, recipe.U, alpha, beta)
            @test norm(D - ( beta * Dtest + alpha * B * pinv(Array(A[1:3, 1:2])))) < 1e-10
            # test for vectors
            xtest = deepcopy(x)
            mul!(x', y', recipe.U, alpha, beta)
            @test norm(x' - ( beta * xtest' + alpha * y' * pinv(Array(A[1:3, 1:2])))) < 1e-10
            # test for adjoints from the left
            Etest = deepcopy(E)
            mul!(E, C, recipe.U', alpha, beta)
            @test norm(E - ( beta * Etest + alpha *  C * pinv(Array(A[1:3, 1:2]))')) < 1e-10
            #test for adjoint with vector
            ytest = deepcopy(y)
            mul!(y', x', recipe.U', alpha, beta)
            @test norm(y' - ( beta * ytest' + alpha * x' * pinv(Array(A[1:3, 1:2]))')) < 1e-10
        end

    end

    # The two rapproximate! paths: dependent selection for two selectors (CUR), and
    # independent selection/compression for every other operator combination.
    @testset "CrossApproximation: rapproximate dispatch" begin
        Random.seed!(0xC0FFEE)
        let m = 30, n = 20, rankv = 4, oversample = 2,
            A = randn(m, rankv) * randn(rankv, n),
            Id = Matrix{Float64}(I, n, n)

            # both compressors -> classical generalized Nystrom (independent path)
            gn = GeneralizedNystrom(
                rank = rankv,
                oversample = oversample,
                selector_cols = Gaussian(cardinality = Right(), compression_dim = rankv),
                selector_rows = Gaussian(
                    cardinality = Left(), compression_dim = rankv + oversample
                ),
            )
            @test !(gn isa CUR)
            @test norm(A - rapproximate(gn, A) * Id) / norm(A) < 1e-8

            # mixed: compressor columns, selector rows (independent path)
            mixed = GeneralizedNystrom(
                rank = rankv,
                oversample = oversample,
                selector_cols = Gaussian(cardinality = Right(), compression_dim = rankv),
                selector_rows = TestSelector(),
            )
            @test !(mixed isa CUR)
            @test norm(A - rapproximate(mixed, A) * Id) / norm(A) < 1e-8

            # both selectors -> dependent CUR path
            cur = CUR(rank = rankv, oversample = oversample, selector_cols = TestSelector())
            @test cur isa CUR
            @test norm(A - rapproximate(cur, A) * Id) / norm(A) < 1e-8
        end
    end

    # A compressor operator must align with the approximation dimensions.
    @testset "CrossApproximation: operator dimension validation" begin
        let m = 30, n = 20, rankv = 4, oversample = 2,
            A = rand(m, n),
            good_col = Gaussian(cardinality = Right(), compression_dim = rankv),
            good_row = Gaussian(cardinality = Left(), compression_dim = rankv + oversample)

            # wrong column compression dimension
            @test_throws DimensionMismatch complete_approximator(
                GeneralizedNystrom(
                    rank = rankv, oversample = oversample,
                    selector_cols = Gaussian(cardinality = Right(), compression_dim = rankv + 1),
                    selector_rows = good_row,
                ),
                A,
            )
            # wrong column cardinality (a `Left` compressor used for columns)
            @test_throws DimensionMismatch complete_approximator(
                GeneralizedNystrom(
                    rank = rankv, oversample = oversample,
                    selector_cols = Gaussian(cardinality = Left(), compression_dim = rankv),
                    selector_rows = good_row,
                ),
                A,
            )
            # wrong row compression dimension
            @test_throws DimensionMismatch complete_approximator(
                GeneralizedNystrom(
                    rank = rankv, oversample = oversample,
                    selector_cols = good_col,
                    selector_rows = Gaussian(cardinality = Left(), compression_dim = rankv),
                ),
                A,
            )
            # a correctly sized pair completes without error
            @test complete_approximator(
                GeneralizedNystrom(
                    rank = rankv, oversample = oversample,
                    selector_cols = good_col, selector_rows = good_row,
                ),
                A,
            ) isa GeneralizedNystromRecipe
        end
    end

end

end
