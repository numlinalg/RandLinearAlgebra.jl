# Compressors API 
```@contents
Pages = ["compressors.md"]
```

## Abstract Types
```@docs
Compressor

CompressorRecipe

CompressorAdjoint

Cardinality

Left

Right

Undef
```

## Compressor Structures
```@docs
CountSketch

CountSketchRecipe

FJLT

FJLTRecipe

Gaussian

GaussianRecipe

Identity

IdentityRecipe

Sampling

SamplingRecipe

SparseSign

SparseSignRecipe

SRHT 

SRHTRecipe
```

## Exported  Functions
```@docs
complete_compressor

update_compressor!
```

## Internal Functions
```@docs
RandLinearAlgebra.left_mul_dimcheck

RandLinearAlgebra.right_mul_dimcheck

RandLinearAlgebra.sparse_idx_update!

RandLinearAlgebra.fwht!
```
