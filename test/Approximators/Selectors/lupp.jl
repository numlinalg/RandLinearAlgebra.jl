module LUPP_tests
using Test
using ..FieldTest
using ..ApproxTol
using RandLinearAlgebra
import LinearAlgebra: mul!

@testset "LUPP Tests" begin
    @testset "LUPP" begin
        @test supertype(LUPP) == Selector
        
        # test fieldnames and types
        @test fieldnames(LUPP) == (:compressor,)
        @test fieldtypes(LUPP) == (Compressor,)
        
        # Test Constructor
        let 
            sel = LUPP()
            @test typeof(sel) == LUPP 
            @test typeof(sel.compressor) == Identity
        end

        let 
            sel = LUPP(compressor = SparseSign())
            @test typeof(sel) == LUPP 
            @test typeof(sel.compressor) == SparseSign
        end

    end

    @testset "LUPPRecipe" begin
        @test supertype(LUPPRecipe) == SelectorRecipe
        # test fieldnames and types
        @test fieldnames(LUPPRecipe) == (:compressor, :SA)
        @test fieldtypes(LUPPRecipe) == (CompressorRecipe, AbstractMatrix)
    end

    @testset "LUPP: Complete Selector" begin
        # test with identity compressor
        let n_rows = 2,
            n_cols = 2,
            A = zeros(n_rows, n_cols),
            sel = LUPP()

            sel_rec = complete_selector(sel, A)
            @test typeof(sel_rec) == LUPPRecipe
            @test typeof(sel_rec.compressor) == IdentityRecipe{Left}
            @test typeof(sel_rec.SA) <: AbstractMatrix
            @test size(sel_rec.SA) == (n_rows, n_cols)
        end

        # test with gaussian compressor
        let n_rows = 3,
            n_cols = 3,
            A = zeros(n_rows, n_cols),
            comp_dim = 2,
            sel = LUPP(compressor = Gaussian(compression_dim = comp_dim))

            sel_rec = complete_selector(sel, A)
            @test typeof(sel_rec) == LUPPRecipe
            @test typeof(sel_rec.compressor) <: GaussianRecipe{Left}
            @test typeof(sel_rec.SA) <: AbstractMatrix
            @test size(sel_rec.SA) == (comp_dim, n_cols)
        end

    end

    @testset "LUPP: Update Selector" begin
        let n_rows = 2,
            n_cols = 2,
            A = zeros(n_rows, n_cols),
            sel_rec = complete_selector(LUPP(compressor = Gaussian()), A)
            G_old = deepcopy(sel_rec.compressor.op)
            update_selector!(sel_rec)
            @test typeof(sel_rec) == LUPPRecipe 
            # check that Gaussian Matrix actually changes
            @test sel_rec.compressor.op != G_old
        end

    end

    @testset "LUPP: Select Indices" begin
        A = [0.0 10.0 0.0;
             0.0 0.0 20.0;
             30.0 0.0 0.0]
        # test the error checking
        # start with checking n_idx not being larger than number of columns
        let A = A,
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 4

            @test_throws DimensionMismatch select_indices!(
                idx, 
                complete_selector(LUPP(), A), 
                A, 
                n_idx, 
                start_idx
            ) 
        end
        
        # check that n_idx will not go over index vector 
        let A = A,
            idx = zeros(Int64, 3),
            start_idx = 3,
            n_idx = 2

            @test_throws DimensionMismatch select_indices!(
                idx, 
                complete_selector(LUPP(), A), 
                A, 
                n_idx, 
                start_idx
            ) 
        end

        # check that n_idx is not larger than the compression_dim
        let A = A,
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 3

            @test_throws DimensionMismatch select_indices!(
                idx, 
                complete_selector(
                    LUPP(compressor = Gaussian(compression_dim = 2)), 
                    A
                ),
                A, 
                n_idx, 
                start_idx
            ) 
        end

        # Test with identity compressor
        # notice that in this selection it has nothing to do with column norm
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 2,
            n_idx = 1,
            sel_rec = complete_selector(LUPP(), A)

            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx[2] == 2
        end

        # test selecting three indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 3,
            sel_rec = complete_selector(LUPP(), A)

            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx == [2; 3; 1]
        end

        # test with gaussian compressor
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 2,
            n_idx = 1,
            sel_rec = complete_selector(LUPP(compressor = Gaussian()), A)
            
            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx[2] == 1
        end

        # test selecting two indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 2,
            sel_rec = complete_selector(LUPP(compressor = Gaussian()), A)

            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx == [1; 3; 0]
        end
    end

end

end
