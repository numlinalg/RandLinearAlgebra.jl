# Multiplying by a Compressor

Let $A$ be a matrix that we wish to compress for some subsequent calculation
(e.g., approximating the column space or solving a least squares problem). 

```@example CompressorTutorial; continued=true
A = randn(1000, 500)
```

We can generate any number of compressors (see [Compressors API](@ref)).
We can then multiply $A$ from the left or right by the compressor. 
We provide three examples below:

- The [`SparseSign`](@ref) compressor [martinsson2020randomized; Section 9.2](@cite).
- The [`Gaussian`](@ref) compressor.
- A [`Uniform`](@ref) [`Sampling`](@ref) of the columns of $A$.

```@example CompressorTutorial; continued=true
using RandLinearAlgebra

I1 = SparseSign(
    cardinality=Left(), #Apply the compressor to the rows of A               
    compression_dim=20, #Reduce the number of rows of A to 20
    nnz=8,                      
    type=Float64                  
)

I2 = Gaussian(
    cardinality=Right(), #Apply the compressor to the columns of A
    compression_dim=10,  #Reduce the number of columns of A to 10
    type=Float64
)

I3 = Sampling(
    cardinality=Right(), #Apply the compressor to the columns of A
    compression_dim=15,  #Reduce the number of columns of A to 15
    distribution=Uniform()
)
```

The previous code block specifies the ingredients for constructing the three 
compressors. The next code block constructs the compressors using 
[`complete_compressor`](@ref).

```@example CompressorTutorial; continued=true
C1 = complete_compressor(I1, A)
C2 = complete_compressor(I2, A)
C3 = complete_compressor(I3, A)
```

We can now apply the compressors to $A$. The first compressor reduces the 
number of rows of $A$ from $1000$ to $20$.

```@example CompressorTutorial
CA1 = C1 * A
size(CA1)
```

The second compressor reduces the number of columns of $A$ from $500$ to $10$.
```@example CompressorTutorial
CA2 = A * C2
size(CA2)
```

The final compressor reduces the number of columns of $A$ from $500$ to $15$.
```@example CompressorTutorial
CA3 = A * C3
size(CA3)
```