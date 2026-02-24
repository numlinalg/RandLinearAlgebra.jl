# Configuring a Basic Logger for the Generalized Kaczmarz Solver

The generalized [`Kaczmarz` solver](@ref Kaczmarz) solver [patel2023randomized](@cite)
is specified by five quantities:
- A [`Compressor`](@ref)
- A [`Logger`](@ref)
- A [`SolverError`](@ref)
- A [`SubSolver`](@ref)
- An over-relaxation parameter in $(0,2]$

Here, we demonstrate how to configure the [`BasicLogger`](@ref).
The [`BasicLogger`](@ref) has three arguments:
- the maximum number of iterations (`max_it`) to run the generalized Kaczmarz solver;
- a `threshold` for the error that terminates the solver when the error drops below 
    this value;
- and the interval of iterates at which to log the error. 

Below is an example for specifying the [`BasicLogger`](@ref).

```@setup ConfiguringLogger
using RandLinearAlgebra
```

```@example ConfiguringLogger; continued=true 
logger = BasicLogger(
    max_it = 500,           #Maximum of 500 iterations
    threshold = 1e-6,       #Error threshold of 1e-6
    collection_rate = 5     #Records error at every fifth iterate
)
```