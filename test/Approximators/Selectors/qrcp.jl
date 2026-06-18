module QRCP_tests
using Test
using RandLinearAlgebra
import LinearAlgebra: mul!

@testset "QRCP Tests" begin
    @testset "QRCP" begin
        @test supertype(QRCP) == Selector
        
        # test fieldnames and types
        @test fieldnames(QRCP) == (:compressor,)
        @test fieldtypes(QRCP) == (Compressor,)
        
        # Test Constructor
        let 
            sel = QRCP()
            @test typeof(sel) == QRCP 
            @test typeof(sel.compressor) == Identity
        end

        let 
            sel = QRCP(compressor = SparseSign())
            @test typeof(sel) == QRCP 
            @test typeof(sel.compressor) == SparseSign
        end

    end

    @testset "QRCPRecipe" begin
        @test supertype(QRCPRecipe) == SelectorRecipe
        # test fieldnames and types
        @test fieldnames(QRCPRecipe) == (:compressor, :SA)
        @test fieldtypes(QRCPRecipe) == (CompressorRecipe, AbstractMatrix)
    end

    @testset "QRCP: Complete Selector" begin
        # test with identity compressor
        let n_rows = 2,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            sel = QRCP()

            sel_rec = complete_selector(sel, A)
            @test typeof(sel_rec) == QRCPRecipe
            @test typeof(sel_rec.compressor) == IdentityRecipe{Left}
            @test typeof(sel_rec.SA) <: AbstractMatrix
            @test size(sel_rec.SA) == (n_rows, n_cols)
        end

        # test with gaussian compressor
        let n_rows = 3,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            comp_dim = 2,
            sel = QRCP(compressor=Gaussian(compression_dim = comp_dim))

            sel_rec = complete_selector(sel, A)
            @test typeof(sel_rec) == QRCPRecipe
            @test typeof(sel_rec.compressor) <: GaussianRecipe{Left}
            @test typeof(sel_rec.SA) <: AbstractMatrix
            @test size(sel_rec.SA) == (comp_dim, n_cols)
        end

    end

    @testset "QRCP: Update Selector" begin
        let n_rows = 2,
            n_cols = 3,
            A = zeros(n_rows, n_cols),
            sel_rec = complete_selector(QRCP(compressor = Gaussian()), A)
            update_selector!(sel_rec)
            @test typeof(sel_rec) == QRCPRecipe 
        end

    end

    @testset "QRCP: Select Indices" begin
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
                complete_selector(QRCP(), A), 
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
                complete_selector(QRCP(), A), 
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
                    QRCP(compressor = Gaussian(compression_dim = 2)), 
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
            sel_rec = complete_selector(QRCP(), A)

            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx[2] == 1
        end

        # test selecting three indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 3,
            sel_rec = complete_selector(QRCP(), A)

            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx == [1; 3; 2]
        end

        # test with gaussian compressor
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 2,
            n_idx = 1,
            sel_rec = complete_selector(QRCP(compressor = Gaussian()), A)
            
            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx[2] == 1
        end

        # test selecting two indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 2,
            sel_rec = complete_selector(QRCP(compressor = Gaussian()), A)

            select_indices!(idx, sel_rec, A, n_idx, start_idx)
            @test idx == [1; 3; 0]
        end

    end

end

end
