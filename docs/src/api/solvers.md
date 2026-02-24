# Solvers API 
```@contents
Pages = ["solvers.md"]
```

## Abstract Types
```@docs
Solver

SolverRecipe
```

## Solver Structures
```@docs
ColumnProjection

ColumnProjectionRecipe

IHS

IHSRecipe

Kaczmarz

KaczmarzRecipe
```

## Exported Functions
```@docs
complete_solver

rsolve!
```

## Internal Functions
```@docs
RandLinearAlgebra.colproj_update!

RandLinearAlgebra.colproj_update_block!

RandLinearAlgebra.kaczmarz_update!

RandLinearAlgebra.kaczmarz_update_block!

RandLinearAlgebra.dotu
```
