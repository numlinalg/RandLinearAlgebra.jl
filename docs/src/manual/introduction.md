# A library for exploring Randomized Linear Algebra
If you are here, you probably know that Linear Algebra is foundational to data science,
scientific computing, and AI. You also probably know Linear Algebra routines dominate the 
computational cost of many of the algorithms in these fields. Thus, improving the 
scalability of these algorithms requires more scalable Linear Algebra techniques.

Randomized Linear Algebra is an exciting new approach to classical linear algebra problems 
that offers strong improvements in scalability. In general, Randomized Linear Algebra 
techniques achieve this improved scalability by forming a compressed representation of a 
matrix and performing operations on that compressed form.
In some circumstances operating on this compressed form
can offer profound speed-ups as can seen in the following example 
where a technique known as the RandomizedSVD (see [halko2011finding](@cite)) 
is used to compute a rank-20 approximation to ``3000 \times 3000`` matrix 
``A`` in place of a truncated SVD. Compared to computing the SVD and 
truncating it, the RandomizedSVD is 100 times faster and just as accurate 
as the truncated SVD.

```julia
using RandLinearAlgebra
using LinearAlgebra

# Generate a rank-20 matrix
A = randn(3000, 20) * randn(20, 3000);

@time U,S,V = svd(A);
#    4.566639 seconds (13 allocations: 412.354 MiB, 0.92% gc time)

# Form the RandomizedSVD data structure
technique = RandSVD(
    compressor = Gaussian(compression_dim = 22,  cardinality = Right()) 
)

# Take the RandomizedSVD of A
@time rec = rapproximate(technique, A);
#    0.050950 seconds (39 allocations: 5.069 MiB)

# Take the norm of the difference between the RandomizedSVD and TrunctatedSVD at rank 22
norm(rec.U * Diagonal(rec.S) * rec.V' - U[:,1:22] * Diagonal(S[1:22]) * (V[:, 1:22])')
# 6.914995919005829e-11
```

Randomized Linear Algebra is a fast growing field with a myriad of new methods being 
proposed every year. Because randomized linear algebra methods are based around similar 
sub-routines many of the innovations could offer improvements to established techniques.
Unfortunately, most implementations of these techniques are static making testing the 
effectiveness of innovations on previous techniques challenging. RandLinearAlgebra.jl aims to 
make incorporating new innovations into established randomized linear algebra techniques
easy.

In particular, RandLinearAlgebra.jl leverages a modular design to allow you 
to easily test Randomized Linear Algebra routines under a wide-range of parameter choices.  
RandLinearAlgebra.jl provides routines for two core Linear Algebra tasks: finding a solution to
a linear system via ``Ax=b`` or ``\min_x \|Ax - b\|`` and forming a low rank 
approximation to a matrix, ``\hat A`` where ``\hat A \approx A``. The solution to a linear
system appears everywhere: Optimization, Tomography, Statistics, Scientific Computing, 
Machine Learning, etc. The low-rank approximation problem has only become more relevant in 
recent years owing to the drastic increase in matrix sizes. It has been widely used in 
Statistics via PCA, but also has become increasingly more relevant.

This manual will walk you through the use of the RandLinearAlgebra.jl library. The remainder 
of this section will be focused on providing an overview of the common design elements in 
the library, and information about how to get started using the library.

## Overview of the Library
You can think of using RandLinearAlgebra.jl as being a producer on 
[*Chopped*](https://www.youtube.com/watch?v=7sm8VrnuOFc&list=PLpfv1AIjenVOxDKUfPuOtjsoF7OWIZVK-&index=9), 
the long running Food Network cooking competition. 
For those unfamiliar, the show takes place in three rounds, where one of four contestants 
get eliminated at the end each round. In each round,
the producers provide the contestants with a fix set of ingredients 
[(often rather unconventional ones at that)](https://www.mashed.com/1244321/most-bizarre-mystery-basket-ingredients-ever-seen-chopped/)
and a general category of food (e.g. appetizer, entree, or dessert) that the contestants
then have to use in a recipe that they prepare for a panel of judges.

RandLinearAlgebra.jl gives you as the user the fun job of deciding which ingredients 
(techniques) you want to use to solve your linear system, compress your matrix/vector, 
or form an approximation of your matrix. Then, once you specify that information, you 
can start the clock by calling `rsolve` or `rapproximate`, with information about your 
matrix/linear system and watch RandLinearAlgebra.jl do the rest. Behind the scenes 
it calls all the  `complete_[technique]` functions that will generate recipe data 
structures that have all the necessary preparations (preallocations) for handling your 
proposed task. Then once the preparations are done, RandLinearAlgebra follows its designed 
recipes to cook-up a solution to your problem. 

With this analogy of how RandLinearAlgebra.jl works, the next two sections provide an overview
over the two key data structures in RandLinearAlgebra.jl, the **technique** structures (your 
ingredients) and the **recipe** structures (what RandLinearAlgebra.jl creates to perform your
task).

### The Technique Types (The Ingredients)
With an understanding of the basic structures in the library, one may wonder, what 
types of techniques are there? First, there are the techniques for solving the linear 
system, `Solvers`, and techniques for forming a low-rank approximation to a matrix, 
`Approximators`. Both `Solvers` and `Approximators` achieve speedups by working on 
compressed forms (often known as sketched or sampled) of the linear system or matrix, 
techniques that compress the linear system are known as `Compressors`. 
Aside from these global techniques, there are also techniques that are specific to 
`Solvers`, which include: 

1. `SubSolvers`, techniques that solve the inner (compressed) linear system.
2. `Loggers`, techniques that log information and determine whether a stopping criterion has
    been met.
3. `SolverError`, a technique that computes the error of a current iterate of a solver.  

Similarly, `Approximators` have their own specific techniques, which include:

1. `ApproximatorError`, a technique that computes the error of an `Approximator`.

With all these technique structures, you may be wondering, what functions
can I call on these structures? Well, the answer is not many. As is 
summarized in the following table.  

| Technique         | Parent Technique   | Function Calls                        |  
| :----------       | :----------------- | :----------------------------------   |  
|`Approximator`     | None               | `complete_approximator`,`rapproximate`|
|`Compressor`       | None               | `complete_compressor`                 |  
|`Solver`           | None               | `complete_solver`, `rsolve`           |
|`ApproximatorError`| `Approximator`     | `complete_approximator_error`         |
|`Logger`           | `Solver`           | `complete_logger`                     |
|`SolverError`      | `Solver`           | `complete_solver_error`               |
|`SubSolver`        | `Solver`           | `complete_sub_solver`                 |


From the above table we can see that all you are able to do (unless you are using 
an `Approximator` or a `Solver`) is complete the technique. The reason being that all the 
technique structures contain only information about algorithmic parameters that require no 
information about the linear system. The recipes on the other hand have all the information 
required to execute a technique including the required pre-allocated memory. We determine the 
preallocations for the Recipes by merging the parameter information of the technique 
structures with the matrix and linear system information via the `complete_[technique]` 
functions, which is the only function that you can call when you have a technique structure. 
There is a special exception for `rsolve` and `rapproximate` because they implicitly call 
all the necessary completes to form the appropriate recipe. The bottom line is that do 
anything useful you will need a recipe.

### The Recipe Types 
Every technique can be transformed into a recipe. As has been stated before, what makes the 
recipes different is that they contain all the required memory allocations. These 
allocations can only be determined from once the matrix is known. As a user, 
all you have to know is that as soon as you have a recipe you can do a lot. As can be seen 
in the following table.

| Technique Recipe        | Parent Recipe    | Function Calls                   |  
|:----------------------  |:-----------------| :--------------------------------|
|`ApproximatorRecipe`     | None             | `mul!`, `rapproximate!`          |
|`CompressorRecipe`       | None             | `mul!`,`update_compressor!`      |
|`SolverRecipe`           | None             | `rsolve!`                        |
|`ApproximatorErrorRecipe`| `Approximator`   | `compute_approximator_error`     |
|`LoggerRecipe`           | `Solver`         | `reset_logger!`, `update_logger!`|
|`SolverErrorRecipe`      | `Solver`         | `compute_error`                  |
|`SubSolverRecipe`        | `Solver`         | `update_sub_solver!`,`ldiv!`     |

Instead of providing 
a different function for each method associated with these tasks, RandLinearAlgebra.jl 
leverages the multiple-dispatch functionality of Julia to allow all linear systems and 
least squares problems to be solved calling the function 
`rsolve(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)` 
and all matrices to be approximated by calling the function 
`rapproximate(approximator::Approximator, A::AbstractMatrix)`. Under this design, changing 
the routine for solving your linear system or approximate your matrix is as 
simple as changing the`solver` or `approximator` arguments. 

## Installing RandLinearAlgebra
Currently, RandLinearAlgebra.jl is not registered in Julia's official package registry. There 
are two main ways of installing RandLinearAlgebra.jl. The preferred way of doing it is through
the local registry. You can install this approach by writing in the REPL:
```julia
] registry add https://github.com/numlinalg/NumLingAlg
add RandLinearAlgebra
```

It can also be installed by writing in the REPL:
```julia
] add https://github.com/numlinalg/RandLinearAlgebra.jl.git
```
It can also be cloned into a local directory and installed by:
1. `cd` into the local project directory 
2. Call `git clone https://github.com/numlinalg/RandLinearAlgebra.jl.git`
3. Run Julia
4. Call `using Pkg`
5. Call `Pkg.activate(RandLinearAlgebra.jl)`
6. Call `Pkg.instantiate()`

For more information see 
[Using someone else's project](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project).

## Using RandLinearAlgebra.jl
For this example let's assume that we have a vector that we wish to compress
using one the RandLinearAlgebra.jl `SparseSign` compressor. To do this: 

1. Load RandLinearAlgebra.jl and generate your vector
2. Define the `SparseSign` technique. This requires us to specify a `cardinality`,
    the direction we intend to apply the compressor from, and a `compression_dim`, 
    the number of entries we want in the compressed vector. In this instance we 
    want the cardinality to be `Left()` and the `compression_dim = 20`.
3. Use the `complete_compressor` function to generate the `SparseSignRecipe`
4. Apply the compressor to the vector using the multiplication function
```julia
# Step 1: load RandLinearAlgebra.jl and generate vector
using RandLinearAlgebra
using LinearAlgebra
# Specify the size of the vector
n = 10000
x = rand(n)

# Step 2: Define Sparse Sign Compressor
comp = SparseSign(compression_dim = 20, cardinality = Left())

# Step 4: Define Sparse Sign Compressor Recipe
S = complete_compressor(comp, x)

# Step 4: Apply the compressor to the vector using the multiplication function
Sx = S * x

norm(Sx)

norm(x)
```
