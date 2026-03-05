# Approximators 
```@contents
Pages = ["approximators.md"]
```

## Abstract Types
```@docs
Approximator

ApproximatorRecipe

ApproximatorAdjoint

RandLinearAlgebra.RangeApproximator

RangeApproximatorRecipe

CURCore

CURCoreRecipe

CURCoreAdjoint

```

## Range Approximator Structures
```@docs
RandSVD

RandSVDRecipe

RangeFinder

RangeFinderRecipe

```

## General Oblique Approximators
```@docs
CUR

CURRecipe
```
### CURCore Structures
```@docs
CrossApproximation

CrossApproximationRecipe
```

## Exported Functions
```@docs
complete_approximator

rapproximate

rapproximate!
```

## Internal Functions
```@docs
RandLinearAlgebra.rand_power_it

RandLinearAlgebra.rand_ortho_it

RandLinearAlgebra.complete_core

RandLinearAlgebra.update_core!

```
