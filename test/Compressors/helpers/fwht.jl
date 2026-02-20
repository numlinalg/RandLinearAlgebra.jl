module fwht_test
using Test, RandLinearAlgebra, Random
import Hadamard: hadamard
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "FWHT" begin
    # begin by testing the errors
    # first test that an error for different scaling vectors is thrown
    let ln = 4,
        x = rand(ln),
        signs = bitrand(ln + 1)
        @test_throws DimensionMismatch RandLinearAlgebra.fwht!(x, signs)
    end

    # test not power of 2 for two argument implementation
    let ln = 3,
        x = rand(ln),
        signs = bitrand(ln)
        @test_throws DimensionMismatch RandLinearAlgebra.fwht!(x, signs)
    end
    
    # test not power of 2 for one argument implementation
    let ln = 3,
        x = rand(ln)
        @test_throws DimensionMismatch RandLinearAlgebra.fwht!(x)
    end
    
    # Now run tests for the hadamrd transform. Run at 8 and 16 because of possible division
    # issues first for the one argument implementation
    for ln in [8, 16]
        # Generate a hadamard matrix of the specified size
        H = hadamard(ln)
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            x = rand(type, ln)
            xc = deepcopy(x)
            RandLinearAlgebra.fwht!(x)
            @test x ≈ H * xc
            # test that it is also a self inverse this tests the scaling as well
            RandLinearAlgebra.fwht!(x, scaling = type(1/ln))
            @test x ≈ xc
        end

    end

    # Perform same tests for two argument version
    for ln in [8, 16]
        # Generate a hadamard matrix of the specified size
        H = hadamard(ln)
        signs = bitrand(ln)
        all_true = similar(signs)
        signs .= true
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            x = rand(type, ln)
            xc = deepcopy(x)
            xc2 = deepcopy(x)
            xc3 = deepcopy(x)
            RandLinearAlgebra.fwht!(x, signs)
            @test x ≈ H * (ifelse.(signs, 1 , -1) .* xc)
            # test with scaling  
            RandLinearAlgebra.fwht!(xc, signs, scaling = type(1/ln))
            @test xc ≈ H * (ifelse.(signs, 1 , -1) .* xc2) .* type(1/ln)
            # test signs self inverse when signs are all 1
            RandLinearAlgebra.fwht!(xc2, all_true) 
            RandLinearAlgebra.fwht!(xc2, all_true, scaling = type(1/ln)) 
            @test xc2 ≈ xc3
        end

    end

end

end
