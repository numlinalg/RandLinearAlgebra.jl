module cur_test 
using Test, RandLinearAlgebra, LinearAlgebra, SparseArrays
import Base.*
import Base: size
import LinearAlgebra: mul!
import RandLinearAlgebra: CURCoreAdjoint, complete_selector, complete_core
import RandLinearAlgebra: rapproximate!, update_core! 

# Write test selector recipe 
mutable struct TestSelector <: Selector end 
mutable struct TestSelectorRecipe <: SelectorRecipe end

function complete_selector(TestSelector::TestSelector, A::AbstractMatrix)
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

# Test core structures
mutable struct TestCore <: CURCore end
mutable struct TestCoreRecipe <: CURCoreRecipe
    A::AbstractMatrix
end

function complete_core(core::TestCore, cur::CUR, A::AbstractMatrix)
    return TestCoreRecipe(rand(2,2))
end

function update_core!(core::TestCoreRecipe, cur::CURRecipe, A::AbstractMatrix)
    core.A = pinv(Array(A[cur.row_idx, cur.col_idx]))
    return nothing
end


# define the size functions for the test core
function size(S::TestCoreRecipe)
    return size(S.A)
end

function size(S::TestCoreRecipe, dim::Int64)
    return size(S.A, dim)
end

function size(S::CURCoreAdjoint{TestCoreRecipe})
    return size(S.parent.A')
end

function size(S::CURCoreAdjoint{TestCoreRecipe}, dim::Int64)
    return size(S.parent.A', dim)
end

function mul!(
    C::AbstractArray, 
    S::TestCoreRecipe, 
    B::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, S.A, B, alpha, beta)
end

function mul!(
    C::AbstractArray, 
    B::AbstractArray, 
    S::TestCoreRecipe, 
    alpha::Number, 
    beta::Number
)
    mul!(C, B, S.A, alpha, beta)
end

function rapproximate!(approx::CURRecipe{TestCoreRecipe}, A::AbstractMatrix)
    # select column indices
    select_indices!(approx.col_idx, approx.col_selector, A, approx.n_col_vecs, 0)
    # select row indices
    select_indices!(approx.row_idx, approx.row_selector, A, approx.n_row_vecs, 0) 
    approx.C .= A[:, approx.col_idx]
    approx.R .= A[approx.row_idx, :]
    # update the core matrix
    update_core!(approx.U, approx, A)
    return
end

@testset "CUR" begin
    @testset "CUR" begin
        supertype(CUR) == Approximator

        # test the fieldnames and types
        fieldnames(CUR) == (
            :rank, 
            :oversample, 
            :col_selector, 
            :row_selector, 
            :core, 
            :block_size
        )
        fieldtypes(CUR) == (
            Int64,
            Int64,
            Selector,
            Selector,
            CURCore,
            Int64
        )

        # test errors
        let block_size = 1,
            rank = -1,
            oversample = 1

            @test_throws  ArgumentError(
                "Field `rank` must be non-negative."
            ) CUR(
                rank = rank, 
                block_size = block_size, 
                oversample = oversample
            )
        end
        
        let block_size = 1,
            rank = 1,
            oversample = -1

            @test_throws ArgumentError(
                "Field `oversample` must be non-negative."
            ) CUR(
                rank = rank, 
                block_size = block_size, 
                oversample = oversample
            ) 
        end

        let block_size = -1,
            rank = 1,
            oversample = 1

            @test_throws  ArgumentError(
                "Field `block_size` must be non-negative."
            ) CUR(
                rank = rank, 
                block_size = block_size, 
                oversample = oversample
            )
        end

        # Test Constructor
        let selector = TestSelector(),
            block_size = 1,
            rank = 1,
            oversample = 1, 
            core = TestCore()

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = TestSelector()
            )

            @test typeof(cur.core) == TestCore
            @test typeof(cur.col_selector) == TestSelector
            @test typeof(cur.row_selector) == TestSelector
            @test cur.oversample == oversample
            @test cur.block_size === block_size
            @test cur.rank == rank
        end

    end 

    @testset "CURRecipe" begin
        @test supertype(CURRecipe) == ApproximatorRecipe

        # test the fieldnames and types
        @test fieldnames(CURRecipe) == (
            :n_rows,
            :n_cols,
            :n_row_vecs,
            :n_col_vecs,
            :col_selector,
            :row_selector,
            :row_idx,
            :col_idx,
            :C,
            :U,
            :R,
            :buffer_row,
            :buffer_core
        )
        @test fieldtypes(CURRecipe) == (
            Int64,
            Int64,
            Int64,
            Int64,
            SelectorRecipe,
            SelectorRecipe,
            Vector{Int64},
            Vector{Int64},
            AbstractMatrix,
            CURCoreRecipe,
            AbstractMatrix,
            AbstractArray,
            AbstractArray
        )
    end 

    @testset "CUR: Complete Approximator" begin
        # test the basic functionality of the complete function
        let n_rows = 4,
            n_cols = 5,
            A = rand(n_rows, n_cols),
            selector = TestSelector(),
            block_size = 1,
            rank = 1,
            oversample = 1, 
            core = TestCore()

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = TestSelector()
            ) 

            recipe = complete_approximator(cur, A)
            typeof(recipe) == CURRecipe{TestCoreRecipe}
            # test that the size function works correctly for recipe
            @test recipe.n_row_vecs == rank + oversample
            @test recipe.n_col_vecs == rank
            @test size(recipe) == (n_rows, n_cols)
            @test size(recipe') == (n_cols, n_rows)
            # test that the selectors are the right type
            @test typeof(recipe.col_selector) == TestSelectorRecipe
            @test typeof(recipe.row_selector) == TestSelectorRecipe
            # test that the index vectors are correct type and size
            @test typeof(recipe.row_idx) == Vector{Int64}
            @test size(recipe.row_idx) == (rank + oversample, )
            @test typeof(recipe.col_idx) == Vector{Int64}
            @test size(recipe.col_idx) == (rank, )
            # test the matrices are allocated with the correct 
            # types and size
            @test typeof(recipe.C) == typeof(A)
            @test size(recipe.C) == (n_rows, rank)
            @test typeof(recipe.R) == typeof(A)
            @test size(recipe.R) == (rank + oversample, n_cols)
            @test typeof(recipe.U) == TestCoreRecipe
            @test size(recipe.U) == (2, 2)
            # test that the buffer matrices are the correct size and type
            @test typeof(recipe.buffer_row) == typeof(A)
            @test size(recipe.buffer_row) == (rank + oversample, block_size)
            @test typeof(recipe.buffer_core) == typeof(A)
            @test size(recipe.buffer_core) == (rank, block_size)
        end

        # perform same tests with a sparse matrix
        let n_rows = 4,
            n_cols = 5,
            A = sprand(n_rows, n_cols, .9),
            selector = TestSelector(),
            block_size = 1,
            rank = 1,
            oversample = 1, 
            core = TestCore()

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = TestSelector()
            ) 

            recipe = complete_approximator(cur, A)
            # test that the size function works correctly for recipe
            @test recipe.n_row_vecs == rank + oversample
            @test recipe.n_col_vecs == rank
            @test size(recipe) == (n_rows, n_cols)
            @test size(recipe') == (n_cols, n_rows)
            # test that the selectors are the right type
            @test typeof(recipe.col_selector) == TestSelectorRecipe
            @test typeof(recipe.row_selector) == TestSelectorRecipe
            # test that the index vectors are correct type and size
            @test typeof(recipe.row_idx) == Vector{Int64}
            @test size(recipe.row_idx) == (rank + oversample, )
            @test typeof(recipe.col_idx) == Vector{Int64}
            @test size(recipe.col_idx) == (rank, )
            # test the matrices are allocated with the correct 
            # types and size
            @test typeof(recipe.C) == typeof(A)
            @test size(recipe.C) == (n_rows, rank)
            @test typeof(recipe.R) == typeof(A)
            @test size(recipe.R) == (rank + oversample, n_cols)
            @test typeof(recipe.U) == TestCoreRecipe
            @test size(recipe.U) == (2, 2)
            # test that the buffer matrices are the correct size and type
            @test typeof(recipe.buffer_row) == typeof(A)
            @test size(recipe.buffer_row) == (rank + oversample, block_size)
            @test typeof(recipe.buffer_core) == typeof(A)
            @test size(recipe.buffer_core) == (rank, block_size)
        end

        # test that blocksize of 0 works correctly
        let n_rows = 4,
            n_cols = 5,
            A = rand(n_rows, n_cols),
            selector = TestSelector(),
            block_size = 0,
            rank = 1,
            oversample = 1, 
            core = TestCore()

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = TestSelector()
            ) 

            recipe = complete_approximator(cur, A)
            @test typeof(recipe.buffer_row) == typeof(A)
            @test size(recipe.buffer_row) == (rank + oversample, size(A, 2))
            @test typeof(recipe.buffer_core) == typeof(A)
            @test size(recipe.buffer_core) == (rank, size(A, 2))
        end

        # test when rank is larger than number of columns
        let n_rows = 4,
            n_cols = 2,
            A = rand(n_rows, n_cols),
            selector = TestSelector(),
            block_size = 0,
            rank = 3,
            oversample = 1, 
            core = TestCore()

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = TestSelector()
            ) 

            @test_throws DomainError(
                "`rank` is greater than number of columns."
            ) complete_approximator(cur, A)
        end

        # test when rank + oversample is larger than number of rows
        let n_rows = 4,
            n_cols = 5,
            A = rand(n_rows, n_cols),
            selector = TestSelector(),
            block_size = 0,
            rank = 2,
            oversample = 4, 
            core = TestCore()

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = TestSelector()
            ) 

            @test_throws DomainError(
                "`rank + oversample` is greater than number of rows."
            ) complete_approximator(cur, A)
        end

    end

    @testset "CUR: mul!" begin
        # begin with left multiplication with a matrix and vector and default 
        # default block size
        let n_rows = 10,
            n_cols = 12,
            rankv = 2,
            oversample = 1,
            alpha = 0.2,
            beta = 0.1,
            A = rand(n_rows, n_cols),
            block_size = 0,
            col_selector = TestSelector(),
            core = TestCore(),
            B = rand(n_cols, 10),
            Btest = deepcopy(B),
            C = rand(n_rows, 10),
            Ctest = deepcopy(C),
            x = rand(n_cols),
            xtest = deepcopy(x),
            y = rand(n_rows),
            ytest = deepcopy(y)

            cur = CUR(
                rank = rankv,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = col_selector 
            ) 
            recipe = rapproximate(cur, A)

            # test left mat multiplication
            mul!(C, recipe, B, alpha, beta)
            @test norm(C - (Ctest * beta + alpha * recipe.C * recipe.U * recipe.R * B)) < 1e-10
            # reset the C for later test
            C = deepcopy(Ctest)
            #test left vec multiplication
            mul!(y, recipe, x, alpha, beta)
            @test norm(y - (ytest * beta + alpha * recipe.C * recipe.U * recipe.R * x)) < 1e-10
            # reset the y for later test
            y = deepcopy(ytest)
            # now test with the adjoint of the recipe
            mul!(B, recipe', C, alpha, beta)
            @test norm(B - (Btest * beta + alpha * recipe.R' * (recipe.U' * (recipe.C' * C)))) < 1e-10
            # now test for the vector case
            mul!(x, recipe', y, alpha, beta)
            @test norm(x - (xtest * beta + alpha * recipe.R' * (recipe.U' * (recipe.C' * y)))) < 1e-10
        end
            

        # begin with left multiplication with a matrix and vector and default 
        # small block size
        let n_rows = 10,
            n_cols = 12,
            rankv = 2,
            oversample = 1,
            alpha = 0.2,
            beta = 0.1,
            A = rand(n_rows, n_cols),
            block_size = 3,
            col_selector = TestSelector(),
            core = TestCore(),
            B = rand(n_cols, 10),
            Btest = deepcopy(B),
            C = rand(n_rows, 10),
            Ctest = deepcopy(C),
            x = rand(n_cols),
            xtest = deepcopy(x),
            y = rand(n_rows),
            ytest = deepcopy(y)

            cur = CUR(
                rank = rankv,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = col_selector 
            ) 
            recipe = rapproximate(cur, A)

            # test left mat multiplication
            mul!(C, recipe, B, alpha, beta)
            @test norm(C - (Ctest * beta + alpha * recipe.C * recipe.U * recipe.R * B)) < 1e-10
            # reset the C for later test
            C = deepcopy(Ctest)
            #test left vec multiplication
            mul!(y, recipe, x, alpha, beta)
            @test norm(y - (ytest * beta + alpha * recipe.C * recipe.U * recipe.R * x)) < 1e-10
            # reset the y for later test
            y = deepcopy(ytest)
            # now test with the adjoint of the recipe
            mul!(B, recipe', C, alpha, beta)
            @test norm(B - (Btest * beta + alpha * recipe.R' * (recipe.U' * (recipe.C' * C)))) < 1e-10
            # now test for the vector case
            mul!(x, recipe', y, alpha, beta)
            @test norm(x - (xtest * beta + alpha * recipe.R' * (recipe.U' * (recipe.C' * y)))) < 1e-10
        end

        # begin with right multiplication with a matrix and vector and default 
        # default block size
        let n_rows = 10,
            n_cols = 12,
            rankv = 2,
            oversample = 1,
            alpha = 0.2,
            beta = 0.1,
            A = rand(n_rows, n_cols),
            block_size = 0,
            col_selector = TestSelector(),
            core = TestCore(),
            B = rand(10, n_rows),
            Btest = deepcopy(B),
            C = rand(10, n_cols),
            Ctest = deepcopy(C),
            x = rand(n_cols),
            xtest = deepcopy(x),
            y = rand(n_rows),
            ytest = deepcopy(y)

            cur = CUR(
                rank = rankv,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = col_selector 
            ) 
            recipe = rapproximate(cur, A)

            # test left mat multiplication
            mul!(C, B, recipe, alpha, beta)
            @test norm(C - (Ctest * beta + alpha * B * recipe.C * recipe.U * recipe.R)) < 1e-10
            # reset the C for later test
            C = deepcopy(Ctest)
            #test left vec multiplication
            mul!(x', y', recipe, alpha, beta)
            @test norm(x' - (xtest' * beta + alpha * (y' * (recipe.C * recipe.U) * recipe.R))) < 1e-10
            # reset the y for later test
            y = deepcopy(ytest)
            # now test with the adjoint of the recipe
            mul!(B, C, recipe', alpha, beta)
            @test norm(B - (Btest * beta + alpha * C * recipe.R' * (recipe.U' * (recipe.C')))) < 1e-10
            # now test for the vector case
            mul!(y', x', recipe', alpha, beta)
            @test norm(y' - (ytest' * beta + alpha * x' * recipe.R' * (recipe.U' * (recipe.C')))) < 1e-10
        end

        # begin with right multiplication with a matrix and vector and default 
        # small block size
        let n_rows = 10,
            n_cols = 12,
            rankv = 2,
            oversample = 1,
            alpha = 0.2,
            beta = 0.1,
            A = rand(n_rows, n_cols),
            block_size = 3,
            col_selector = TestSelector(),
            core = TestCore(),
            B = rand(10, n_rows),
            Btest = deepcopy(B),
            C = rand(10, n_cols),
            Ctest = deepcopy(C),
            x = rand(n_cols),
            xtest = deepcopy(x),
            y = rand(n_rows),
            ytest = deepcopy(y)

            cur = CUR(
                rank = rankv,
                oversample = oversample,
                core = core,
                block_size = block_size,
                selector_cols = col_selector 
            ) 
            recipe = rapproximate(cur, A)

            # test left mat multiplication
            mul!(C, B, recipe, alpha, beta)
            @test norm(C - (Ctest * beta + alpha * B * recipe.C * recipe.U * recipe.R)) < 1e-10
            # reset the C for later test
            C = deepcopy(Ctest)
            #test left vec multiplication
            mul!(x', y', recipe, alpha, beta)
            @test norm(x' - (xtest' * beta + alpha * (y' * (recipe.C * recipe.U) * recipe.R))) < 1e-10
            # reset the y for later test
            y = deepcopy(ytest)
            # now test with the adjoint of the recipe
            mul!(B, C, recipe', alpha, beta)
            @test norm(B - (Btest * beta + alpha * C * recipe.R' * (recipe.U' * (recipe.C')))) < 1e-10
            # now test for the vector case
            mul!(y', x', recipe', alpha, beta)
            @test norm(y' - (ytest' * beta + alpha * x' * recipe.R' * (recipe.U' * (recipe.C')))) < 1e-10
        end

    end

end

end
