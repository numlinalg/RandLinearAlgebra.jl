"""
    RangeFinder

A struct that implements the Randomized Range Finder technique which uses compression from 
    the right to form an low-dimensional orthogonal matrix ``Q`` that approximates the 
    range of ``A``. See [halko2011finding](@cite) for additional details.

# Mathematical Description
Suppose we have a matrix ``A \\in \\mathbb{R}^{m \\times n}`` of which we wish to form a low 
    rank approximation that approximately captures the range of ``A``. Specifically, we wish
    to find an Orthogonal matrix ``Q`` such that ``QQ^\\top A \\approx A``. 

A simple way to find such a matrix is to choose a ``k`` representing the number of 
    vectors we wish to have in the subspace. Then we can generate a compression matrix 
    ``S\\in\\mathbb{R}^{n \\times k}`` and compute ``Q = \\text{qr}(AS)``. 
    With high probability we will have ``\\|A - QQ^\\top A\\|_F \\leq
    \\sqrt{k+1} (\\sum_{i=k+1}^{\\min{(m,n)}}\\sigma_{i})^{1/2}``, 
    where ``\\sigma_{k+1}`` is the ``k+1^\\text{th}`` singular value 
    of A (see Theorem 10.5 of [halko2011finding](@cite)). This bound is often conservative 
    as long as the singular values of ``A`` decay quickly.  
    
When the singular values decay slowly, we can improve the quality of the approximation using the 
    power iteration, which applies ``A`` and ``A^\\top``, ``q`` times 
    and take the qr factorization of ``(AA^\\top)^q AS``. Using these power iterations increases the 
    relative gap between the singular values leading to  better Rangefinder performance. 

Performing power iterations in floating points can destroy all information 
    related to the smallest singular values of ``A`` 
    (see Remark 4.3 in [halko2011finding](@cite)). We can preserve this information by 
    orthogonalizing inbetween the products of ``AS`` with ``A`` or ``A^\\top`` 
    in the power iteration. These steps are known as the orthogonalized power 
    iteration (see Algorithm 4.4 of [halko2011finding](@cite)).  
    Orthogonalized power iterations progress according to the following steps:

1. ``\\tilde{A}_1 = AS``  
2. ``Q_1,R_1 = \\textbf{qr}(\\tilde{A}_1)``  
3. ``\\tilde{A}_2 = A^\\top Q_1``  
4. ``Q_2,R_2 = \\textbf{qr}(\\tilde{A}_2)``  
5. ``\\tilde{A}_1 = A Q_2``  
6. ``Q_1, R_1 = \\textbf{qr}(\\tilde{A}_1)``  
7. Repeat Steps 3 through 6 for the desired number of power iterations 
   set ``Q = Q_1``. 

# Fields
- `compressor::Compressor`, the technique that will compress the matrix from the right.
- `power_its::Int64`, the number of power iterations that should be performed.
- `orthogonalize::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization.

# Constructor
    
    RangeFinder(;
        compressor = SparseSign(), 
        orthogonalize = false, 
        power_its = 0
    )

## Keywords    
- `compressor::Compressor`, the technique that will compress the matrix from the right.
- `power_its::Int64`, the number of power iterations that should be performed. Default is
    zero.
- `orthogonalize::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization. Default is false.

## Returns
- A `RangeFinder` object.

# Throws
- `ArgumentError` if `power_its` is negative.
"""
mutable struct RangeFinder <: RangeApproximator
    compressor::Compressor
    power_its::Int64
    orthogonalize::Bool
    function RangeFinder(compressor, power_its, orthogonalize)
        if power_its < 0
            return throw(ArgumentError("Field `power_its` must be non-negative."))
        end
        
        return new(compressor, power_its, orthogonalize)
    end

end

RangeFinder(;
    compressor::Compressor = SparseSign(cardinality = Right()),
    orthogonalize::Bool = false,
    power_its::Int64 = 0
) = RangeFinder(compressor, power_its, orthogonalize)

"""
    RangeFinderRecipe

A struct that contains the preallocated memory and completed compressor to form a
    RangeFinder approximation to the matrix ``A``.

# Fields
- `compressor::CompressorRecipe`, the compressor to be applied from the right to ``A``.
- `power_its::Int64`, the number of power iterations that should be performed.
- `orthogonalize::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization.
- `range::AbstractMatrix`, the orthogonal matrix that approximates the range of ``A``.
"""
mutable struct RangeFinderRecipe <: RangeApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
    compressor::CompressorRecipe
    power_its::Int64
    orthogonalize::Bool
    range::AbstractMatrix
end

function complete_approximator(approx::RangeFinder, A::AbstractMatrix)
    type = eltype(A)
    # You need to make sure you orient the compressor in the correct direction
    if typeof(approx.compressor.cardinality) <: Left
        @warn "Compressor with cardinality `Left` being applied from `Right`."
    end

    compress = complete_compressor(approx.compressor, A)
    # Determine the dimensions of the range approximator
    a_rows = size(A, 1)
    c_cols = size(compress, 2)
    approx_recipe = RangeFinderRecipe(
        a_rows,
        c_cols,
        compress, 
        approx.power_its,
        approx.orthogonalize, 
        Matrix{type}(undef, 2, 2)
    )
end

function rapproximate!(approx::RangeFinderRecipe, A::AbstractMatrix)
    # User may wish to choose to use a different power iteration
    if approx.orthogonalize 
        approx.range = rand_ortho_it(A, approx)
    else
        approx.range = rand_power_it(A, approx)
    end

    return nothing
end

function rapproximate(approx::RangeFinder, A::AbstractMatrix)
    approx_recipe = complete_approximator(approx, A)
    rapproximate!(approx_recipe, A)
    return  approx_recipe
end

function mul!(
    C::AbstractArray, 
    R::RangeFinderRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, R.range, A, alpha, beta)
end


function mul!(
    C::AbstractArray, 
    R::ApproximatorAdjoint{RangeFinderRecipe}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, R.parent.range', A, alpha, beta)
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::RangeFinderRecipe, 
    alpha::Number, 
    beta::Number
)
    mul!(C, A, R.range, alpha, beta)
end


function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::ApproximatorAdjoint{RangeFinderRecipe}, 
    alpha::Number, 
    beta::Number
)
    mul!(C, A, R.parent.range', alpha, beta)
end
