# Overview of Tutorials

Below are examples of using different routines available in `RandLinearAlgebra.jl`. 

- [Multiplying by a Compressor](@ref) demonstrates how to compress a matrix by
    multiplication with a compressor.

- [Solving a Consistent Linear System](@ref) demonstrates how to approximately find
    $x$ that solves $Ax = b$ when $b$ is in the column space of $A$ using the default
    solver.
    - [Configuring a Basic Logger for the Generalized Kaczmarz Solver](@ref) demonstrates 
        how to customize the [`BasicLogger`](@ref) argument for a `Kaczmarz` solver.

- [Solving a Least Squares Problem](@ref) demonstrates how to approximately solve 
    $\min_{x}\|Ax - b\|_2^2$.




