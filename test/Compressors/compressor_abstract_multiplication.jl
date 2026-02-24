module compressor_abstract_multiplication
using Test, RandLinearAlgebra
import LinearAlgebra: mul!
using ..FieldTest
using ..ApproxTol

#####################
# Testing Parameters
##################### 
mutable struct TestCompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
end
s_rows = 3
s_cols = 4
TestCompressorRecipe() = TestCompressorRecipe(s_rows, s_cols)

@testset "Compressor Recipe Multiplication Errors" begin 
    # Test Set Parameters 
    x = randn(2)
    y = randn(2)
    A = randn(2, 2)
    C = randn(2, 2) 

    # Five argument muls 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe(), A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe(), 1.0, 1.0)
    @test_throws ArgumentError mul!(x, TestCompressorRecipe(), y, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe(), 1.0, 1.0)

    # Three argument muls 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe(), A)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe())
    @test_throws ArgumentError mul!(x, TestCompressorRecipe(), y)
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe())

    # Binary muls 
    @test_throws ArgumentError TestCompressorRecipe()*A 
    @test_throws ArgumentError A*TestCompressorRecipe() 
    @test_throws ArgumentError TestCompressorRecipe()*y
    @test_throws ArgumentError y*TestCompressorRecipe()

    # Five argument muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe()', A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe()', 1.0, 1.0)
    @test_throws ArgumentError mul!(y, TestCompressorRecipe()', x, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe()', 1.0, 1.0)

    # Three arguments muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe()', A)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe()')
    @test_throws ArgumentError mul!(y, TestCompressorRecipe()', x)
    @test_throws ArgumentError mul!(y, x, TestCompressorRecipe()')

    # Binary muls 
    @test_throws ArgumentError TestCompressorRecipe()'*A 
    @test_throws ArgumentError A*TestCompressorRecipe()' 
    @test_throws ArgumentError TestCompressorRecipe()'*y
    @test_throws ArgumentError y*TestCompressorRecipe()' 
end

#################################
# Additional Testing Parameters 
#################################

# Left 5 Arg mul: S*A, C[i,j] =-1
mul!(C::AbstractArray, S::TestCompressorRecipe, A::AbstractArray, alpha::Number, 
    β::Number) = fill!(C, -1)

# Right mul: A*S, C[i,j] = -2
mul!(C::AbstractArray, A::AbstractArray, S::TestCompressorRecipe, alpha::Number, 
    β::Number) = fill!(C, -2)

c_rows = 2
c_cols = 3
a_rows = 5
a_cols = 8

@testset "Compressor Recipe Abstract 5-Arg Multiplication" begin 
    for C in [zeros(c_rows), zeros(c_rows, c_cols)], 
        A in [zeros(a_rows), zeros(a_rows, a_cols)]
        
        S = TestCompressorRecipe()

        ##########################
        # 5 Arg mul
        ########################## 

        # S * A -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, S, A, 1.0, 0.0)
            for c in C 
                @test c == -1 #Left mul sets everything to -1
            end
        end

        # A * S -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, A, S, 1.0, 0.0)
            for c in C 
                @test c == -2 #Right mul sets everything to -2
            end
        end

        # A * S' -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, A, S', 1.0, 0.0) # Should call mul!(C', S, A', 1.0, 0.0)
            for c in C
                @test c == -1 #Left mul should be -1
            end
        end

        # S' * A -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, S', A, 1.0, 0.0) # Should call mul!(C', A', S, 1.0, 0.0)
            for c in C 
                @test c == -2 #Right mul should be -2
            end
        end


        ##########################
        # 3 Arg mul
        ########################## 

        # S * A -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, S, A)
            for c in C 
                @test c == -1 #Left mul sets everything to -1
            end
        end

        # A * S -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, A, S)
            for c in C 
                @test c == -2 #Right mul sets everything to -2
            end
        end

        # A * S' -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, A, S') # Should call mul!(C', S, A')
            for c in C
                @test c == -1 #Left mul should be -1
            end
        end

        # S' * A -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, S', A) # Should call mul!(C', A', S)
            for c in C 
                @test c == -2 #Right mul should be -2
            end
        end

        ##########################
        # Binary *
        ########################## 
        # S * A
        let S=S, A=A
            D = S*A
            for c in D
                @test c == -1 #Left mul sets everything to -1
            end
            @test size(D) == (size(A, 2) == 1 ? (s_rows,) : (s_rows, a_cols))
        end

        # A * S
        let S=S, A=A
            D = A*S
            for c in D 
                @test c == -2 #Right mul sets everything to -2
            end
            @test size(D) == (a_rows, s_cols)
        end

        # A * S' -> C
        let S=S, A=A
            D = A*S' 
            for c in D
                @test c == -1 #Left mul should be -1
            end
            @test size(D) ==  (a_rows, s_rows)
        end

        # S' * A -> C
        let S=S, A=A
            D = S'*A
            for c in D 
                @test c == -2 #Right mul should be -2
            end
            @test size(D) == (size(A,2)==1 ? (s_cols,) : (s_cols, a_cols))
        end 

    end
end

#################################
# Additional Testing Parameters 
#################################

# Left 3 Arg mul: S*A, C[i,j] = 1
mul!(C::AbstractArray, S::TestCompressorRecipe, A::AbstractArray) = fill!(C, 1)

# Right 3 Arg mul: A*S, C[i,j] = 2
mul!(C::AbstractArray, A::AbstractArray, S::TestCompressorRecipe) = fill!(C, 2)

@testset "Compressor Recipe Abstract 3-Arg Multiplication" begin 
    for C in [zeros(c_rows), zeros(c_rows, c_cols)], 
        A in [zeros(a_rows), zeros(a_rows, a_cols)]
        
        S = TestCompressorRecipe()

        ##########################
        # 3 Arg mul
        ########################## 

        # S * A -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, S, A)
            for c in C 
                @test c == 1 #Left mul sets everything to 1
            end
        end

        # A * S -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, A, S)
            for c in C 
                @test c == 2 #Right mul sets everything to 2
            end
        end

        # A * S' -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, A, S') # Should call mul!(C', S, A')
            for c in C
                @test c == 1 #Left mul should be 1
            end
        end

        # S' * A -> C
        let C=deepcopy(C), S=S, A=A
            mul!(C, S', A) # Should call mul!(C', A', S)
            for c in C 
                @test c == 2 #Right mul should be 2
            end
        end

        ##########################
        # Binary *
        ########################## 
        # S * A
        let S=S, A=A
            D = S*A
            for c in D
                @test c == 1 #Left mul sets everything to 1
            end
            @test size(D) == (size(A, 2) == 1 ? (s_rows,) : (s_rows, a_cols))
        end

        # A * S
        let S=S, A=A
            D = A*S
            for c in D 
                @test c == 2 #Right mul sets everything to 2
            end
            @test size(D) == (a_rows, s_cols)
        end

        # A * S' -> C
        let S=S, A=A
            D = A*S' 
            for c in D
                @test c == 1 #Left mul should be 1
            end
            @test size(D) ==  (a_rows, s_rows)
        end

        # S' * A -> C
        let S=S, A=A
            D = S'*A
            for c in D 
                @test c == 2 #Right mul should be 2
            end
            @test size(D) == (size(A,2)==1 ? (s_cols,) : (s_cols, a_cols))
        end 

    end
end

end
