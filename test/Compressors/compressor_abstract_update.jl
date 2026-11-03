module compressor_abstract_update 
include("../test_helpers/preamble.jl")
using .MockCompressors

@testset "Compressor Recipe Update" begin

    # Check Single Argument Update 
    let compressor = MockUpdateCompressorRecipe(3, 4, false)
        # Check initial status 
        @test compressor.status == false

        # Update compressor 
        val = update_compressor!(compressor)

        # Check updated status 
        @test compressor.status == true
        @test val == true 
    end

    # Check Two Argument Update 
    let compressor = MockUpdateCompressorRecipe(3, 4, false),
        A = randn(5, 6)

        # Check initial status 
        @test compressor.status == false

        # Update compressor 
        val = update_compressor!(compressor, A)

        # Check updated status 
        @test compressor.status == true
        @test isnothing(val)
    end

    # Check Three Argument Update 
    let compressor = MockUpdateCompressorRecipe(3, 4, false),
        A = randn(5, 6),
        b = randn(7)

        # Check initial status 
        @test compressor.status == false

        # Update compressor 
        val = update_compressor!(compressor, A, b)

        # Check updated status 
        @test compressor.status == true
        @test isnothing(val)
    end

    # Check Four Argument Update 
    let compressor = MockUpdateCompressorRecipe(3, 4, false),
        A = randn(5, 6),
        b = randn(7),
        x = randn(8)

        # Check initial status 
        @test compressor.status == false

        # Update compressor 
        val = update_compressor!(compressor, x, A, b)

        # Check updated status 
        @test compressor.status == true
        @test isnothing(val)
    end

end
end

