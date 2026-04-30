"""
    attention

Function that computes the attention given three weight matrices.

# Arguments
- `X::AbstractArray`, a data matrix
- `Wq::AbstractArray`, a weight matrix
- `Wv::AbstractArray`, a weight matrix
- `Wk::AbstractArray`, a weight matrix
"""
function attention(X::AbstractArray, Wq::AbstractArray, Wk::AbstractArray, Wv::AbstractArray)
    Q = X * Wq
    K = X * Wk
    V = X * Wv
    d_k = size(K, 2)
    scores = Q * K' / sqrt(Float32(d_k))
    # softmax over each row
    scores = scores .- maximum(scores, dims=2)
    weights = exp.(scores) ./ sum(exp.(scores), dims=2)
    return weights * V
end
