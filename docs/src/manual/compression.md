# Compressing a Matrix/Vector
Most Randomized Linear Algebra routines start by forming a low-dimensional representation of 
a matrix/vector. These representations are typically formed by multiplying a 
random matrix with a large matrix/vector. 
For instance, if we have ``x \in \mathbb{R}^{10,000}`` we can apply a matrix, 
``S \in \mathbb{R}^{20 \times 10,000}`` made up of 
independent and identically distributed ``\textbf{Normal}(0, 1/\sqrt{20})`` to obtain
``y = Sx \in \mathbb{R}^{20}`` where
``(1-\epsilon)\|x\| \leq \|y\| \leq (1+\epsilon) \|x\|`` with high probability. 

Of course, many other techniques beyond a Gaussian matrix can be used to generate 
``S`` and they vary both in terms of their approximation accuracy and computational 
efficiency. In papers, Randomized Linear Algebraists often refer to techniques
for generating ``S`` as either sampling (random subset of identity) or sketching 
(general random matrix) techniques. For simplicity RandLinearAlgebra.jl refers to both types 
of techniques under the general family of Compressors. The choice of the terminology 
`Compressors` also allows us to incorporate deterministic approaches to compressing 
matrices/vectors. 

In RandLinearAlgebra.jl, using a compression technique requires two main steps. The first,
uses the `complete_compressor` function to generate a CompressorRecipe. The second 
step uses the `mul!` or `*` functions to apply your `CompressorRecipe` to your matrix/
vector object. If you ever want to form a new realization of your compressor, you can 
use the `update_compressor!` function to update the random entries in your 
`CompressorRecipe`. 

## Using complete_compressor
In its simplest form, `complete_compressor` has two arguments, the first is `compressor`,
where you specify the compressor technique that you wish to use. All compressors
have two fields that the user can specify; although some have some additional technique  
specific fields (see [Compressors API](@ref) for more details). The first field 
is `compression_dim`, where you specify the dimension that your `CompressorRecipe` will map 
a matrix/vector into after being applied to the matrix/vector. The second field is 
`Cardinality`, which can either be `Left()` or `Right()`. `Cardinality` allows you to 
specify how you intend to multiply your compressor to a matrix/vector. In the matrix case, 
if `S` is your `CompressorRecipe` and `A` is your matrix, a `Left()` cardinality 
corresponds to multiplying `SA` and a `Right()` corresponds to multiplying `AS`. Once 
you have specified these two fields in your `Compressor` object, the second input in to the 
`complete_compressor` function will be the matrix/vector you wish to compress. 

## Multiplying your CompressorRecipe
Once you have your `CompressorRecipe` you can multiply it to any matrix/vector 
just as you would any matrix object, using either the `mul!` or `*` functions. You can also 
take transposes of the `CompressorRecipe` just as you would any other matrix object. 
Just like in `LinearAlgebra`, the `mul!` function should be used when you have preallocated 
an output array and the `*` function should be used when you have not. 
!!! note "Effect of Cardinality on Multiplication"
    Provided the dimensions align you can provide a compressor to the left or 
    right of a matrix or vector regardless of the `Cardinality`. However, if the 
    direction applied does not align with the `Cardinality`, this could be a slower 
    operation.

## Updating update_compressor!
Because most compression techniques are randomized, it is likely that once you have a 
realization of `CompressorRecipe` that you would want another realization of that same 
recipe. This can be done using the `update_compressor!` function. This function in 
its simplest form has only one argument although for some compressors it could have more
arguments (see [Compressors API](@ref) for more details). The one argument is the 
`CompressorRecipe`.

## Compressing a Matrix Example
Knowing that compressors allow us to reduce one of the dimensions of a matrix, the next 
important question is how do we do this in RandLinearAlgebra.jl? In the following example 
we show how to do exactly this using a [Gaussian](@ref) compressor. In this 
example we will generate a `GaussianRecipe`, `S`, with `compression_dim` 10 and 
`cardinality` `Left()`, then we will apply `S` and its transpose, `S'`, to a matrix, `A`, 
with a 100 rows and 100 columns using `*`. 
Then we will generate a new realization of the recipe using 
`update_compressor!` and use the `mul!` to apply this new compressor to `A` from 
the left.

```julia 
using RandLinearAlgebra
using LinearAlgebra

A = rand(100, 100)

# Generate the CompressorRecipe with compression_dim 10 and cardinality Left()
S = complete_compressor(
    Gaussian(
        compression_dim = 10,
        cardinality = Left()
    ),
    A
)

# Multiply this compressor and its transpose to the left and right of A, respectively
C = S * A * S'

# Generate a new realization of S
update_compressor!(S)

# use mul! to multiply S from the left
# the output matrix will have the number of rows in S and the number of columns in A
C = zeros(size(S, 1), size(A, 2))

mul!(C, S, A)
```

## Sampling Compressors and Distributions
A special sub-type of compressors are known as `Sampling` compressors. These compressors
are unique in that they compress the matrix by selecting rows or columns according to a 
specific distribution. For example, we could compress a matrix by sampling 10 rows uniformly
at random in what is known unsurprisingly as a uniform sampling approach. RandLinearAlgebra.jl
allows you to use the Sampling techniques with a wide range of distributions (see 
[Distributions](@ref) for more details). You can specify the distribution you want to use
with the `distribution` argument in the sampling structure. As an example, if we want to 
perform uniform sampling from a matrix we can form our SamplingRecipe by writing:
```julia
using RandLinearAlgebra
A = rand(100, 100)

# Generate Compression recipe using uniform sampling distribution
S = complete_compressor(
    Sampling(
        compression_dim = 10,
        cardinality = Left(),
        distribution = Uniform()
    ),
    A
)
```

## Summary of Compressors
We now know that anytime we want to reduce one of the dimensions of a matrix or vector,
we need to form a `CompressorRecipe`. To form the `CompressorRecipe`, we need to call
`complete_compressor` with a `Compressor` data structure, which specifies the technique
we want to use to compress a matrix/vector, and a matrix/vector that we wish to compress.
Once we have this recipe, we can generate a new realization of it by calling 
`update_compressor!` and can apply it to a matrix or vector using `mul!` or `*`. For more
information on specific compressors see [Compressors API](@ref).
