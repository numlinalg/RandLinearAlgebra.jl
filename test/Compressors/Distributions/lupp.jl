module LUPP_tests
using Test
using ..FieldTest
using ..ApproxTol
using RandLinearAlgebra
import LinearAlgebra: mul!

@testset "LUPP Tests" begin
    @testset "LUPP" begin
        @test supertype(LUPP) == Distribution
        
        # test fieldnames and types
        @test fieldnames(LUPP) == (:cardinality, :compressor, :oversample, :compression_dim)
        @test fieldtypes(LUPP) == (Cardinality, Compressor, Int64, Int64)
        
        # Test Constructor
        let 
            sel = LUPP()
            @test typeof(sel) == LUPP 
            @test typeof(sel.compressor) == Identity
            @test typeof(sel.cardinality) == Undef
            @test sel.oversample == 0
            @test sel.compression_dim == 0
        end

        let 
            sel = LUPP(compressor = SparseSign())
            @test typeof(sel) == LUPP 
            @test typeof(sel.compressor) == SparseSign
            @test typeof(sel.cardinality) == Undef
            @test sel.oversample == 0
            @test sel.compression_dim == 0
        end

    end

    @testset "LUPPRecipe" begin
        @test supertype(LUPPRecipe) == DistributionRecipe
        # test fieldnames and types
        @test fieldnames(LUPPRecipe) == (:compressor, :permuted_indices, :SA)
        @test fieldtypes(LUPPRecipe) == (CompressorRecipe, Vector{Integer}, AbstractMatrix)
    end

    @testset "LUPP: Complete Distribution" begin
        # test error with undefined cardinality
        let n_rows = 2,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            sel = LUPP()

            @test_throws ArgumentError("`LUPP` cardinality must be specified as `Left()` \
            or `Right()`. `Undef()` is not allowed in `complete_distribution`.") complete_distribution(sel, A)
        end

        # test with identity compressor
        let n_rows = 2,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            sel = LUPP(cardinality = Left())

            sel_rec = complete_distribution(sel, A)
            @test typeof(sel_rec) == LUPPRecipe{Left}
            @test typeof(sel_rec.compressor) == IdentityRecipe{Right}
            @test typeof(sel_rec.SA) <: AbstractMatrix
            @test size(sel_rec.SA) == (n_rows, n_cols)
        end

        # test 0 compression_dim error
        let n_rows = 2,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            sel = LUPP(compressor = Gaussian(), cardinality = Left())
            @test_throws ArgumentError("`LUPP` `compression_dim` must be specified as >0") complete_distribution(sel, A)
        end

        # test undef card and 0 comp dim
        let n_rows = 2,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            sel = LUPP(compressor = Gaussian())
            @test_throws ArgumentError("Neither `cardinality` nor `compression_dim` specified with \
            non-default arguments. If you are not using LUPP as a `distribution` in a \
            `Sampler` please specify a positive `compression_dim` and a `Left()` or \
             `Right()` `cardinality`.") complete_distribution(sel, A)
        end 

        # test with gaussian compressor
        let n_rows = 2,
            n_cols = 4,
            A = zeros(n_rows, n_cols),
            comp_dim = 2,
            os = 1,
            sel = LUPP(
                cardinality = Right(),
                compressor = Gaussian(), 
                compression_dim = comp_dim, 
                oversample = os
            )
            sel_rec = complete_distribution(sel, A)
            @test typeof(sel_rec) == LUPPRecipe{Right}
            @test typeof(sel_rec.compressor) == GaussianRecipe{Left} 
            @test typeof(sel_rec.SA) <: AbstractMatrix
            @test size(sel_rec.SA) == (comp_dim + os, n_cols)
        end

    end

    @testset "LUPP: Update Distribution" begin
        # test with Left cardinality
        let n_rows = 4,
            n_cols = 3,
            A = [0.0 10.0 0.0;
                0.0 0.0 200.0;
                3000.0 0.0 0.0],
            comp_dim = 2,
            os = 1,
            sel = LUPP(
                compressor = Gaussian(), 
                cardinality = Left(),
                compression_dim = comp_dim, 
                oversample = os
            )
            sel_rec = complete_distribution(sel, A)
            G_old = deepcopy(sel_rec.compressor.op)
            indices_old = deepcopy(sel_rec.permuted_indices)
            update_distribution!(sel_rec, A)
            @test typeof(sel_rec) == LUPPRecipe{Left} 
            # check that Gaussian Matrix actually changes
            @test sel_rec.compressor.op != G_old
            @test sel_rec.permuted_indices != indices_old
        end

        # test with Right cardinality
        let n_rows = 3,
            n_cols = 4,
            comp_dim = 2,
            os = 1,
            sel = LUPP(
                compressor = Gaussian(), 
                cardinality = Right(),
                compression_dim = comp_dim, 
                oversample = os
            ),
            A = [0.0 10.0 0.0;
            0.0 0.0 200.0;
            3000.0 0.0 0.0],
            sel_rec = complete_distribution(sel, A)
            G_old = deepcopy(sel_rec.compressor.op)
            indices_old = deepcopy(sel_rec.permuted_indices)
            update_distribution!(sel_rec, A)
            @test typeof(sel_rec) == LUPPRecipe{Right}
            # check that Gaussian Matrix actually changes
            @test sel_rec.compressor.op != G_old
            @test sel_rec.permuted_indices != indices_old
        end

    end

    @testset "LUPP: Sample Distribution " begin
        A = [0.0 10.0 0.0;
             0.0 0.0 200.0;
             3000.0 0.0 0.0]
        # test the error checking
        # start with checking n_idx not being larger than number of columns
        let A = deepcopy(A),
            idx = zeros(Int64, 4)

            @test_throws DimensionMismatch sample_distribution!(
                idx, 
                complete_distribution(LUPP(cardinality = Right()), A)
            ) 
        end
        
        # Test with identity compressor
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 1),
            sel_rec = complete_distribution(LUPP(cardinality = Right()), A)
            update_distribution!(sel_rec, A)
            sample_distribution!(idx, sel_rec)
            @test idx[1] == 2
        end

        # test selecting three indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            sel_rec = complete_distribution(LUPP(cardinality = Right()), A)
            update_distribution!(sel_rec, A)
            sample_distribution!(idx, sel_rec)
            @test idx == [2; 3; 1]
        end

        # test with gaussian compressor
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 1)

            sel_rec = complete_distribution(
                LUPP(
                    compressor = Gaussian(), 
                    cardinality = Left(),
                    compression_dim = 1
                ), 
                    A
            )
            update_distribution!(sel_rec, A)
            sample_distribution!(idx, sel_rec)
            @test idx[1] == 3
        end

        # test selecting two indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3);
            sel_rec = complete_distribution(
                LUPP(
                    compressor = Gaussian(), 
                    cardinality = Left(),
                    compression_dim = 3
                ), 
                A
            )
            update_distribution!(sel_rec, A)
            sample_distribution!(idx, sel_rec)
            @test idx == [3; 2; 1]
        end
    end

end

end