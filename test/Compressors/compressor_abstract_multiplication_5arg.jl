module compressor_abstract_multiplication_5arg
include("../test_helpers/preamble.jl")
using .MockCompressors

#####################
# Testing Parameters
#####################
s_rows = 3
s_cols = 4
c_rows = 2
c_cols = 3
a_rows = 5
a_cols = 8

@testset "Compressor Recipe Abstract 5-Arg Multiplication" begin 
    for C in [zeros(c_rows), zeros(c_rows, c_cols)], 
        A in [zeros(a_rows), zeros(a_rows, a_cols)]
        
        S = MockMul5CompressorRecipe(s_rows, s_cols)

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

end
