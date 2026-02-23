# Solving Linear Systems

Given a linear system consisting of a matrix ``A \in \mathbb{R}^{m \times n}`` and
constant vector ``b \in \mathbb{R}^m`` in the column space of ``A``, we aim to 
find ``x^*`` such that ``Ax^* = b``.

`RandLinearAlgebra` allows you to find an approximation to a solution of the linear 
system by calling the function `rsolve!(solver, x, A, b)`, 
where `A` is the coefficient matrix, 
`b` is the constant vector, 
and `x` is the initialization of the solution vector that will
be replaced with solution produced by the solver. 
The `solver` variable of type 
`Solver` allows you to specify the algorithmic parameters specific to the solver 
you wish to use to approximate a solution to the system. 


# Generalized Kaczmarz Method
Kaczmarz is one randomized approach for solving consistent linear systems with a nice 
geometric interpretation. We can understand this geometric interpretation by first 
recognizing that when a linear system has solution, ``x^*``,  
``A[i, :] x^* = b[i]`` for every row of the linear system. Geometrically this means that
``x^*`` lies at the intersection of all row hyperplanes. Kaczmarz is able to converge to 
this solution by projecting orthogonally from one hyperplane to the next. 
Performing such 
<!-- TODO: Review - should this be "projects" or "projections"? @gisslab -->
projects ensures that every update gets closer to the solution, as can be seen in the 
following figure.

```@raw html
<img src="../images/kaczmarz.svg" width =400 height = 300/> 
```

In the following example, we run a Kaczmarz solver for 30 iterations. 
To do this appropriately, we set the `max_it` field in the `BasicLogger` structure to be the 
`log` field in the `Kaczmarz` solver.

```julia
using RandLinearAlgebra
using LinearAlgebra

# Generate the linear system
A = rand(10, 10);
x_sol = rand(10);
b = A * x_sol;

# create initalization for solution
x = zeros(10);

# Create the Kaczmarz Solver
solver = Kaczmarz(
    log = BasicLogger(max_it = 30)
);

# solve the system
rsolve!(solver, x, A, b)

# Check error of solution 
norm(x - x_sol)
```

Often for Kaczmarz, we can improve the rate of convergence by projecting onto blocks of
rows.
In `RandLinearAlgebra` we can do this by changing the `compression_dim` of the 
compressor. 
Below, we set the `compression_dim = 5`. We also may wish to 
stop when the residual falls below `1e-1`, which we do by specifying the `threshold` field
in the `Logger` structure. After setting up these systems, we run our solver and plot the
history of the `FullResidual` by plotting the `hist` field of `log` field of the returned 
`SolverRecipe`.
```julia
using RandLinearAlgebra
using LinearAlgebra
using Plots

# Generate the linear system
A = rand(10, 10);
x_sol = rand(10);
b = A * x_sol;

# create initalization for solution
x = zeros(10);

# Create the Kaczmarz Solver
solver = Kaczmarz(
    log = BasicLogger(
        max_it = 100,
        threshold = 1e-1
         
    ),
    compressor = SparseSign(compression_dim = 5)
);

# solve the system
x, kaczmarz_recipe = rsolve!(solver, x, A, b)

# Check error of solution 
norm(x - x_sol)

# Plot the residual history
plot(
    kaczmarz_recipe.log.hist[kaczmarz_recipe.log.hist .> 0.0], 
    yscale = :log10, 
    xlab = "Iteration", 
    ylab = "Norm of Residual"
)
```

# `Solver` Concepts

`Solver`s tend to have the following fields.
1. `Compressor` specifies how we will compress the system.
2. `SubSolver` allows you to specify how to solve the compressed system.
3. `ErrorMethod` allows you to specify the technique that you wish to use to 
    determine a solver's progress.
4. `Logger` performs two key tasks: (1) it stores the information from 
    an error method in a `hist` field and (2) it determines whether an inputted stopping 
    criterion is satisfied.
See the [Solvers API](@ref) for more details.
