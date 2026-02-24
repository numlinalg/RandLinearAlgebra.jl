module CURCore_test 
using Test, RandLinearAlgebra, LinearAlgebra
import LinearAlgebra: mul!
import RandLinearAlgebra: complete_core, update_core!

struct TestCURCore <: CURCore end 
struct TestCURCoreRecipe <: CURCoreRecipe 
    n_rows::Int64
    n_cols::Int64
    code::Int64
end 

TestCoreRecipe() = TestCURCoreRecipe(2, 2, 1)
struct TestSelectorRecipe <: SelectorRecipe end

@testset "CURCore: Abstract Types" begin
    @test isdefined(Main, :CURCore)
    @test isdefined(Main, :CURCoreRecipe)
    @test isdefined(Main, :CURCoreAdjoint)
end

@testset "CURCore: Completion Errors" begin
    cur = CUR(rank = 1)
    A = ones(2, 2)
    core = TestCURCore()
    
    @test_throws ArgumentError complete_core(core, cur, A)
end

@testset "CURCore: size" begin
    cur = CURRecipe{TestCURCoreRecipe}(
        2, 
        2,
        2,
        2,
        TestSelectorRecipe(),
        TestSelectorRecipe(),
        ones(Int64, 2),
        ones(Int64, 2),
        zeros(2, 2), 
        TestCURCoreRecipe(1, 2, 1),
        zeros(2, 2),
        zeros(1), 
        zeros(1)
    )

    # test the size functions 
    @test size(cur.U) == (1, 2)
    @test size(cur.U') == (2, 1)
    @test size(cur.U, 1) == 1 
    @test size(cur.U', 1) == 2
    @test size(cur.U, 2) == 2 
    @test size(cur.U', 2) == 1
    @test_throws DomainError size(cur.U, 3)
    @test_throws DomainError size(cur.U', 3)
    @test_throws DomainError size(cur.U, 0)
    @test_throws DomainError size(cur.U', 0)
end

# check that the adjoint functions work correctly
@testset "CURCore: adjoint" begin
    core = TestCURCoreRecipe(1, 2, 1)
    core_t = transpose(core)
    core_a = adjoint(core)
    @test typeof(core_t) <: CURCoreAdjoint
    @test typeof(core_a) <: CURCoreAdjoint
    @test transpose(core_t) == core
    @test adjoint(core_a)  == core
end

@testset "CURCore: Update Errors" begin
    cur = CURRecipe{TestCURCoreRecipe}(
        2, 
        2,
        2,
        2,
        TestSelectorRecipe(),
        TestSelectorRecipe(),
        ones(Int64, 2),
        ones(Int64, 2),
        zeros(2, 2), 
        TestCURCoreRecipe(1, 2, 1),
        zeros(2, 2),
        zeros(1), 
        zeros(1)
    )


    A = ones(2, 2)
    @test_throws ArgumentError update_core!(cur.U, cur, A)
end

@testset "CURCore Multiplication Errors" begin 
    # Test Set Parameters 
    x = randn(2)
    y = randn(2)
    A = randn(2, 2)
    C = randn(2, 2) 

    # Five argument muls 
    @test_throws ArgumentError mul!(C, TestCoreRecipe(), A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCoreRecipe(), 1.0, 1.0)
    @test_throws ArgumentError mul!(x, TestCoreRecipe(), y, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, TestCoreRecipe(), 1.0, 1.0)

    # Three argument muls 
    @test_throws ArgumentError mul!(C, TestCoreRecipe(), A)
    @test_throws ArgumentError mul!(C, A, TestCoreRecipe())
    @test_throws ArgumentError mul!(x, TestCoreRecipe(), y)
    @test_throws ArgumentError mul!(x, y, TestCoreRecipe())

    # Binary muls 
    @test_throws ArgumentError TestCoreRecipe() * A 
    @test_throws ArgumentError A * TestCoreRecipe() 
    @test_throws ArgumentError TestCoreRecipe() * y
    @test_throws ArgumentError y * TestCoreRecipe()

    # Five argument muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCoreRecipe()', A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCoreRecipe()', 1.0, 1.0)
    @test_throws ArgumentError mul!(y, TestCoreRecipe()', x, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, TestCoreRecipe()', 1.0, 1.0)

    # Three arguments muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCoreRecipe()', A)
    @test_throws ArgumentError mul!(C, A, TestCoreRecipe()')
    @test_throws ArgumentError mul!(y, TestCoreRecipe()', x)
    @test_throws ArgumentError mul!(y, x, TestCoreRecipe()')

    # Binary muls 
    @test_throws ArgumentError TestCoreRecipe()' * A 
    @test_throws ArgumentError A * TestCoreRecipe()' 
    @test_throws ArgumentError TestCoreRecipe()' * y
    @test_throws ArgumentError y * TestCoreRecipe()' 
end

# Left 5 Arg mul: S*A, C[i,j] =-1
mul!(C::AbstractArray, S::TestCURCoreRecipe, A::AbstractArray, alpha::Number, 
    β::Number) = fill!(C, -1)

# Right mul: A*S, C[i,j] = -2
mul!(C::AbstractArray, A::AbstractArray, S::TestCURCoreRecipe, alpha::Number, 
    β::Number) = fill!(C, -2)

@testset "Compressor Recipe Abstract 5-Arg Multiplication" begin 
    c_rows = 2
    c_cols = 3
    a_rows = 5
    a_cols = 8
    for C in [zeros(c_rows), zeros(c_rows, c_cols)], 
        A in [zeros(a_rows), zeros(a_rows, a_cols)]
        
        S = TestCoreRecipe()
        s_rows, s_cols = size(S)
        ##########################
        # 5 Arg mul
        ########################## 

        # S * A -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, S, A, 1.0, 0.0)
            for c in C 
                @test c == -1 #Left mul sets everything to -1
            end

        end

        # A * S -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, A, S, 1.0, 0.0)
            for c in C 
                @test c == -2 #Right mul sets everything to -2
            end

        end

        # A * S' -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, A, S', 1.0, 0.0) # Should call mul!(C', S, A', 1.0, 0.0)
            for c in C
                @test c == -1 #Left mul should be -1
            end

        end

        # S' * A -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, S', A, 1.0, 0.0) # Should call mul!(C', A', S, 1.0, 0.0)
            for c in C 
                @test c == -2 #Right mul should be -2
            end

        end


        ##########################
        # 3 Arg mul
        ########################## 

        # S * A -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, S, A)
            for c in C 
                @test c == -1 #Left mul sets everything to -1
            end

        end

        # A * S -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, A, S)
            for c in C 
                @test c == -2 #Right mul sets everything to -2
            end

        end

        # A * S' -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, A, S') # Should call mul!(C', S, A')
            for c in C
                @test c == -1 #Left mul should be -1
            end

        end

        # S' * A -> C
        let C = deepcopy(C), S = S, A = A
            mul!(C, S', A) # Should call mul!(C', A', S)
            for c in C 
                @test c == -2 #Right mul should be -2
            end
            
        end

        ##########################
        # Binary *
        ########################## 
        # S * A
        let S = S, A = A
            D = S*A
            for c in D
                @test c == -1 #Left mul sets everything to -1
            end

            @test size(D) == (size(A, 2) == 1 ? (s_rows,) : (s_rows, a_cols))
        end

        # A * S
        let S = S, A = A
            D = A * S
            for c in D 
                @test c == -2 #Right mul sets everything to -2
            end

            @test size(D) == (a_rows, s_cols)
        end

        # A * S' -> C
        let S = S, A = A 
            D = A * S' 
            for c in D
                @test c == -1 #Left mul should be -1
            end

            @test size(D) ==  (a_rows, s_rows)
        end

        # S' * A -> C
        let S = S, A = A
            D = S' * A
            for c in D 
                @test c == -2 #Right mul should be -2
            end

            @test size(D) == (size(A,2)==1 ? (s_cols,) : (s_cols, a_cols))
        end 

    end


end

end
