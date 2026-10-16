"""
    MockArray{T,N} <: AbstractArray{T,N}

A thin wrapper around a standard `Array` that overrides `similar` to return
another `MockArray`. Used in tests to verify that allocations inside
`complete_*` and binary operators respect the caller's array type — a
prerequisite for GPU compatibility.
"""
struct MockArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

MockArray(data::Array{T,N}) where {T,N} = MockArray{T,N}(data)

Base.size(A::MockArray) = size(A.data)
Base.getindex(A::MockArray, i::Int...) = getindex(A.data, i...)
Base.setindex!(A::MockArray, v, i::Int...) = (setindex!(A.data, v, i...); A)
Base.similar(A::MockArray, ::Type{T}, dims::Dims) where {T} =
    MockArray(Array{T}(undef, dims))

import LinearAlgebra: mul!, norm, qr

function LinearAlgebra.mul!(
    C::MockArray,
    A::MockArray,
    B::MockArray,
    α::Number,
    β::Number,
)
    mul!(C.data, A.data, B.data, α, β)
    return C
end

function LinearAlgebra.mul!(
    C::MockArray,
    A::MockArray,
    B::MockArray,
)
    mul!(C.data, A.data, B.data)
    return C
end

# Adjoint / Transpose forwarding
function LinearAlgebra.mul!(
    C::MockArray,
    A::LinearAlgebra.Adjoint{<:Any,<:MockArray},
    B::MockArray,
    α::Number,
    β::Number,
)
    mul!(C.data, adjoint(A.parent.data), B.data, α, β)
    return C
end

function LinearAlgebra.mul!(
    C::MockArray,
    A::MockArray,
    B::LinearAlgebra.Adjoint{<:Any,<:MockArray},
    α::Number,
    β::Number,
)
    mul!(C.data, A.data, adjoint(B.parent.data), α, β)
    return C
end

LinearAlgebra.norm(A::MockArray) = norm(A.data)

# Allow qr to work on MockArray by delegating to the underlying data
LinearAlgebra.qr(A::MockArray) = qr(A.data)

# randn! support
import Random: randn!
Random.randn!(A::MockArray) = (randn!(A.data); A)

# lmul! support
import LinearAlgebra: lmul!
LinearAlgebra.lmul!(α::Number, A::MockArray) = (lmul!(α, A.data); A)

# copyto! support
Base.copyto!(dst::MockArray, src::MockArray) = (copyto!(dst.data, src.data); dst)
Base.copyto!(dst::MockArray, src::AbstractArray) = (copyto!(dst.data, src); dst)
