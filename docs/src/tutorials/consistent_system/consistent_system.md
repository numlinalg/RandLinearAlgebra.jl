# Solving a Consistent Linear System

This guide demonstrates how to use `RandLinearAlgebra.jl` package to solve a 
**consistent linear system**. That is, how to approximately find $x$
that satisfies

$$Ax = b,$$

where $A$ is a matrix; and $b$ is a vector in the column space of $A$.

Define $A$ and define $b$ to be in the column space of $A$.

```@example ConsistentExample; continued=true
num_rows, num_cols = 100, 5
A = randn(Float64, num_rows, num_cols)
b = A*randn(Float64, num_cols)
```


`RandLinearAlgebra.jl` can find an approximate solution using the generalized 
[`Kaczmarz`](@ref) method [patel2023randomized](@cite).

```@example ConsistentExample; continued=true
using RandLinearAlgebra
solver = Kaczmarz(log = BasicLogger(max_it = 300))
solution = zeros(Float64, num_cols) # Initial guess of zeros
rsolve!(solver, solution, A, b)
```
`rsolve!` updates `solution` with the approximate solution.