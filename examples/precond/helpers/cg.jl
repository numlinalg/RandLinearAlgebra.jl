using LinearAlgebra
using RandLinearAlgebra

"""
    Preconditioner

An abstract type for preconditioners to the pcg function.
"""
abstract type Preconditioner end

"""
    IdentityPreconditioner

A preconditioner that applies the identity matrix to a vector.
"""
mutable struct IdentityPreconditioner <: Preconditioner end

function ldiv!(x::AbstractVector, L::IdentityPreconditioner, y::AbstractVector)
    x .= y
end

"""
    pcg(
        A, 
        b; 
        x = zeros(size(b, 1)), 
        L = IdentityPreconditioner(), 
        maxit = size(b, 1), 
        threshold = 1e-12
    )

Function that implements preconditioned CG.

# Arguments
- `A::AbstractMatrix`, the matrix in linear system (should be symmetric)
- `b::AbstractVector`, the constant vector in the linear system 
- `x::AbstractVector`, the initialized solution vector
- `L::Preconditioner`, the preconditioner to be applied from the left
- `maxit::Integer`, the maximum number of iterations that CG will be run
- `threshold::Float64`, the stopping threshold for the residual of the soltion

# Returns
A solution vector, `x`, and the number of iterations before exiting.
"""
function pcg(
    A::AbstractMatrix, 
    b::AbstractVector; 
    x::AbstractVector = zeros(size(b, 1)), 
    L::Preconditioner = IdentityPreconditioner(), 
    maxit::Int64 = size(b, 1), 
    threshold::Float64 = 1e-12
)
    # preallocate vectors neeeded for solver
    Ap = zeros(size(b, 1))
    r_new = zeros(size(b, 1))
    p = zeros(size(b, 1))
    z = zeros(size(b, 1))
    z_new = zeros(size(b, 1))
    norm_A = norm(A)
    r = b - A * x
    # apply preconditioner
    ldiv!(z, L, r)
    p .= z
    for i in 1:maxit
        # compute A * p
        mul!(Ap, A, p)
        alpha = dot(r, z) / dot(Ap, p)
        x .+= alpha * p
        r_new .= r - alpha * Ap
        if norm(r_new) / norm_A < threshold 
            return x, i
        end
        
        ldiv!(z_new, L, r_new)
        beta = dot(r_new, z_new) / dot(r, z)
        p .= z_new + beta * p
        r .= r_new
        z .= z_new
    end

    return x, maxit

end