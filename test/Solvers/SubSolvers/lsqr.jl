using LinearAlgebra

# A preconditioner that reports a size and can be applied from the left but provides no
# `adjoint`. `complete_sub_solver` should reject it, since the bidiagonalization applies
# the adjoint of the preconditioner.
struct LSQRNoAdjointPrecond
    n::Int
end
Base.size(p::LSQRNoAdjointPrecond) = (p.n, p.n)
LinearAlgebra.ldiv!(y::AbstractVector, ::LSQRNoAdjointPrecond, x::AbstractVector) =
    copyto!(y, x)

# A dimension-agnostic scaled-identity preconditioner. It implements no `size` method, so
# it validates against a coefficient matrix with any number of rows. This lets the
# preconditioner buffers be resized when the matrix dimensions change rather than the
# preconditioner being rejected.
struct LSQRScaledIdentity
    c::Float64
end
LinearAlgebra.ldiv!(y::AbstractVector, p::LSQRScaledIdentity, x::AbstractVector) =
    (y .= x ./ p.c)
Base.adjoint(p::LSQRScaledIdentity) = p

@testset "LSQR SubSolver Tests" begin
    using Test, RandLinearAlgebra, Random, LinearAlgebra, SparseArrays

    @testset "LSQR SubSolver" begin
        @test supertype(LSQRSolver{NoPreconditioner}) == KrylovSubSolver
        @test supertype(KrylovSubSolver) == SubSolver
        @test fieldnames(LSQRSolver) == (:max_it, :atol, :btol, :zero_init, :precond)
        @test fieldtypes(LSQRSolver) == (Int64, Float64, Float64, Bool, Any)

        # Test the default constructor
        let ss = LSQRSolver()
            @test typeof(ss) <: SubSolver
            @test ss.max_it == 0
            @test ss.atol == 1e-8
            @test ss.btol == 1e-8
            @test ss.zero_init
            @test ss.precond == NoPreconditioner()
        end

        # Test the keyword constructor
        let precond = Diagonal(collect(1.0:12.0)),
            ss = LSQRSolver(;
                max_it = 5,
                atol = 1e-4,
                btol = 1e-6,
                zero_init = false,
                precond = precond
            )

            @test ss.max_it == 5
            @test ss.atol == 1e-4
            @test ss.btol == 1e-6
            @test !ss.zero_init
            @test ss.precond === precond
        end

        # Test the argument errors
        @test_throws ArgumentError LSQRSolver(; max_it = -1)
        @test_throws ArgumentError LSQRSolver(; atol = -1.0)
        @test_throws ArgumentError LSQRSolver(; btol = -1.0)
    end

    @testset "LSQR SubSolver Recipe" begin
        @test supertype(LSQRSolverRecipe) == KrylovSubSolverRecipe
        @test supertype(KrylovSubSolverRecipe) == SubSolverRecipe
        @test fieldnames(LSQRSolverRecipe) ==
            (:A, :max_it, :atol, :btol, :zero_init, :u, :v, :w, :precond, :t, :s)
        @test fieldtypes(LSQRSolverRecipe) == (
            AbstractArray,
            Int64,
            Float64,
            Float64,
            Bool,
            AbstractVector,
            AbstractVector,
            AbstractVector,
            Any,
            AbstractVector,
            AbstractVector
        )
    end

    @testset "LSQR SubSolver: Complete SubSolver" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = LSQRSolver(; max_it = 7, atol = 1e-4, btol = 1e-6),
                A = rand(type, 12, 3),
                ss_recipe = complete_sub_solver(ss, A)

                # test the attributes of the outputs of the complete function
                @test typeof(ss_recipe) == LSQRSolverRecipe{
                    type,
                    Vector{type},
                    Matrix{type},
                    NoPreconditioner
                }
                @test ss_recipe.A == A
                @test ss_recipe.max_it == 7
                @test ss_recipe.atol == 1e-4
                @test ss_recipe.btol == 1e-6
                @test ss_recipe.zero_init
                @test ss_recipe.precond == NoPreconditioner()
                @test size(ss_recipe.u) == (12,)
                @test size(ss_recipe.v) == (3,)
                @test size(ss_recipe.w) == (3,)
                # the preconditioner buffers are unallocated when none is applied
                @test size(ss_recipe.t) == (0,)
                @test size(ss_recipe.s) == (0,)
            end

        end

        # A preconditioner should be stored and should allocate the work buffers
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let precond = Diagonal(ones(type, 12)),
                ss = LSQRSolver(; zero_init = false, precond = precond),
                A = rand(type, 12, 3),
                ss_recipe = complete_sub_solver(ss, A)

                @test typeof(ss_recipe) == LSQRSolverRecipe{
                    type,
                    Vector{type},
                    Matrix{type},
                    typeof(precond)
                }
                @test ss_recipe.precond === precond
                @test !ss_recipe.zero_init
                @test size(ss_recipe.t) == (12,)
                @test size(ss_recipe.s) == (12,)
            end

        end

        # The preconditioner must be square and conformable with the matrix
        let A = rand(12, 3)
            @test_throws ArgumentError complete_sub_solver(
                LSQRSolver(; precond = Diagonal(ones(5))), A
            )
            @test_throws ArgumentError complete_sub_solver(
                LSQRSolver(; precond = UpperTriangular(rand(3, 3) + 3I)), A
            )
            # An unwrapped matrix only matches the generic `ldiv!` fallback of
            # `LinearAlgebra` and cannot be applied to a vector
            @test_throws ArgumentError complete_sub_solver(
                LSQRSolver(; precond = rand(12, 12)), A
            )
            # An object with no `ldiv!` at all
            @test_throws ArgumentError complete_sub_solver(
                LSQRSolver(; precond = "not a preconditioner"), A
            )
            # A preconditioner that can be applied from the left but has no `adjoint`,
            # which the bidiagonalization requires
            @test_throws ArgumentError complete_sub_solver(
                LSQRSolver(; precond = LSQRNoAdjointPrecond(12)), A
            )
        end

    end

    @testset "LSQR SubSolver: Update SubSolver" begin
        # Same dimensions
        let A = rand(12, 3),
            B = rand(12, 3),
            ss = LSQRSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
            @test size(ss_recipe.u) == (12,)
            @test size(ss_recipe.v) == (3,)
            @test size(ss_recipe.w) == (3,)
        end

        # Different dimensions should resize the buffers
        let A = rand(12, 3),
            B = rand(10, 4),
            ss = LSQRSolver(),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
            @test size(ss_recipe.u) == (10,)
            @test size(ss_recipe.v) == (4,)
            @test size(ss_recipe.w) == (4,)
            @test size(ss_recipe.t) == (0,)
            @test size(ss_recipe.s) == (0,)
        end

        # The preconditioner buffers should follow the row dimension of the matrix
        let A = rand(12, 3),
            B = rand(12, 4),
            ss = LSQRSolver(; precond = Diagonal(collect(1.0:12.0))),
            ss_recipe = complete_sub_solver(ss, A)

            update_sub_solver!(ss_recipe, B)
            @test size(ss_recipe.t) == (12,)
            @test size(ss_recipe.s) == (12,)
        end

        # A matrix with a new number of rows invalidates the preconditioner
        let A = rand(12, 3),
            B = rand(10, 4),
            ss = LSQRSolver(; precond = Diagonal(collect(1.0:12.0))),
            ss_recipe = complete_sub_solver(ss, A)

            @test_throws ArgumentError update_sub_solver!(ss_recipe, B)
        end

        # A matrix with a new number of rows should resize the preconditioner buffers when
        # the preconditioner remains valid at the new dimension
        let A = rand(12, 3),
            B = rand(10, 4),
            b = rand(10),
            ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100,
                precond = LSQRScaledIdentity(2.0)),
            ss_recipe = complete_sub_solver(ss, A)

            @test size(ss_recipe.t) == (12,)
            @test size(ss_recipe.s) == (12,)
            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
            @test size(ss_recipe.u) == (10,)
            @test size(ss_recipe.t) == (10,)
            @test size(ss_recipe.s) == (10,)
            # The resized recipe should still solve the new system
            x_sol = zeros(4)
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ qr(B) \ b
        end

    end

    @testset "LSQR SubSolver: ldiv!" begin
        Random.seed!(2131)
        # Overdetermined least squares problem
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
                A = rand(type, 12, 3),
                b = rand(type, 12),
                ss_recipe = complete_sub_solver(ss, A),
                x_sol = zeros(type, 3)

                x_true = qr(A) \ b
                ldiv!(x_sol, ss_recipe, b)
                @test x_sol ≈ x_true
            end

        end

        # Underdetermined system should return the minimum norm solution
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
                A = rand(type, 3, 12),
                b = rand(type, 3),
                ss_recipe = complete_sub_solver(ss, A),
                x_sol = zeros(type, 12)

                x_true = A' * ((A * A') \ b)
                ldiv!(x_sol, ss_recipe, b)
                @test x_sol ≈ x_true
            end

        end

        # Sparse matrices
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
            A = sprand(30, 5, 0.5) + sparse(1.0I, 30, 5),
            b = rand(30),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(5)

            x_true = Matrix(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # Solving after updating the matrix to one with new dimensions
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
            A = rand(12, 3),
            B = rand(10, 4),
            b = rand(10),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(4)

            update_sub_solver!(ss_recipe, B)
            x_true = qr(B) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # A zero constant vector should return the zero solution
        let ss = LSQRSolver(),
            A = rand(12, 3),
            b = zeros(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(3)

            ldiv!(x_sol, ss_recipe, b)
            @test x_sol == zeros(3)
        end

        # Hitting the maximum iteration should produce an inexact solution
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 1),
            A = rand(12, 6),
            b = rand(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(6)

            x_true = qr(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test !(x_sol ≈ x_true)
        end

    end

    @testset "LSQR SubSolver: Initial Solution" begin
        Random.seed!(1013)
        # With `zero_init` the contents of the solution vector should be discarded
        let ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100),
            A = rand(12, 3),
            b = rand(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = 1e3 .* rand(3)

            x_true = qr(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # Without `zero_init` the solution vector is the initial iterate, and starting
        # from zero should reproduce the same solution
        let ss = LSQRSolver(;
                atol = 1e-12,
                btol = 1e-12,
                max_it = 100,
                zero_init = false
            ),
            A = rand(12, 3),
            b = rand(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(3)

            x_true = qr(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # Starting from a nearby point should reach the same solution
        let ss = LSQRSolver(;
                atol = 1e-12,
                btol = 1e-12,
                max_it = 100,
                zero_init = false
            ),
            A = rand(12, 3),
            b = rand(12),
            ss_recipe = complete_sub_solver(ss, A),
            x_true = qr(A) \ b,
            x_sol = x_true .+ 1e-6

            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # Starting from a nearby point should need fewer iterations than starting from
        # zero, which confirms the initial residual is formed from the initial iterate
        let A = rand(12, 3),
            b = rand(12),
            x_true = qr(A) \ b

            let warm = complete_sub_solver(
                    LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 1,
                        zero_init = false),
                    A
                ),
                cold = complete_sub_solver(
                    LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 1), A
                ),
                x_warm = x_true .+ 1e-6,
                x_cold = zeros(3)

                ldiv!(x_warm, warm, b)
                ldiv!(x_cold, cold, b)
                @test norm(x_warm - x_true) < norm(x_cold - x_true)
            end

        end

    end

    @testset "LSQR SubSolver: Left Preconditioning" begin
        Random.seed!(4057)
        # A consistent square system should be solved for any nonsingular preconditioner
        for type in [Float32, Float64, ComplexF32, ComplexF64]
            tol = type <: Union{Float32, ComplexF32} ? 1e-5 : 1e-11
            A = rand(type, 8, 8) + 8I
            x_true = rand(type, 8)
            b = A * x_true
            for precond in [
                Diagonal(rand(real(type), 8) .+ one(real(type))),
                UpperTriangular(rand(type, 8, 8) + 8I),
                lu(rand(type, 8, 8) + 8I)
            ]
                let ss = LSQRSolver(;
                        atol = tol,
                        btol = tol,
                        max_it = 200,
                        precond = precond
                    ),
                    ss_recipe = complete_sub_solver(ss, A),
                    x_sol = zeros(type, 8)

                    ldiv!(x_sol, ss_recipe, b)
                    @test isapprox(x_sol, x_true; atol = sqrt(tol))
                end

            end

        end

        # Applying the preconditioner implicitly should match running the unpreconditioned
        # method on the explicitly preconditioned system
        for type in [Float64, ComplexF64]
            let A = rand(type, 10, 10) + 10I,
                b = rand(type, 10),
                precond = lu(rand(type, 10, 10) + 10I),
                implicit = complete_sub_solver(
                    LSQRSolver(; atol = 0.0, btol = 0.0, max_it = 4,
                        precond = precond),
                    A
                ),
                explicit = complete_sub_solver(
                    LSQRSolver(; atol = 0.0, btol = 0.0, max_it = 4), precond \ A
                ),
                x_implicit = zeros(type, 10),
                x_explicit = zeros(type, 10)

                ldiv!(x_implicit, implicit, b)
                ldiv!(x_explicit, explicit, precond \ b)
                @test x_implicit ≈ x_explicit
            end

        end

        # A unitary preconditioner leaves the least squares solution unchanged
        let A = rand(20, 4),
            b = rand(20),
            precond = lu(Matrix(qr(randn(20, 20)).Q)),
            ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100,
                precond = precond),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(4)

            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ qr(A) \ b
        end

        # An underdetermined consistent system should still give the minimum norm solution
        let A = rand(3, 12),
            b = rand(3),
            precond = Diagonal(rand(3) .+ 1.0),
            ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100,
                precond = precond),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(12)

            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ A' * ((A * A') \ b)
        end

        # Preconditioning should compose with an initial solution
        let A = rand(8, 8) + 8I,
            x_true = rand(8),
            b = A * x_true,
            precond = Diagonal(rand(8) .+ 1.0),
            ss = LSQRSolver(; atol = 1e-12, btol = 1e-12, max_it = 100,
                zero_init = false, precond = precond),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = x_true .+ 1e-6

            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # A zero constant vector should return the zero solution
        let A = rand(12, 3),
            ss = LSQRSolver(; precond = Diagonal(rand(12) .+ 1.0)),
            ss_recipe = complete_sub_solver(ss, A),
            x_sol = zeros(3)

            ldiv!(x_sol, ss_recipe, zeros(12))
            @test x_sol == zeros(3)
        end

        # A row scaling preconditioner should speed up convergence on a badly scaled
        # system
        let n = 40,
            A = Diagonal(10.0 .^ range(-5, 5, length = n)) * (randn(n, n) + 3I),
            x_true = randn(n),
            b = A * x_true,
            precond = Diagonal([norm(view(A, i, :)) for i in 1:n])

            let preconditioned = complete_sub_solver(
                    LSQRSolver(; atol = 0.0, btol = 0.0, max_it = n,
                        precond = precond),
                    A
                ),
                unpreconditioned = complete_sub_solver(
                    LSQRSolver(; atol = 0.0, btol = 0.0, max_it = n), A
                ),
                x_prec = zeros(n),
                x_none = zeros(n)

                ldiv!(x_prec, preconditioned, b)
                ldiv!(x_none, unpreconditioned, b)
                @test norm(x_prec - x_true) < norm(x_none - x_true)
            end

        end

    end

end
