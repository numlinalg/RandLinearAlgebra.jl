module RandLinearAlgebra
import Base.:*
import Base: transpose, adjoint, size, setproperty!
import LinearAlgebra: Adjoint, axpby!, axpy!, ColumnNorm, dot, I, ldiv!
import LinearAlgebra: lmul!, lq!, lq, LQ, lu!, mul!, norm, qr!, rmul!, svd
import LinearAlgebra: Transpose, UpperTriangular
import StatsBase: ProbabilityWeights, sample, sample!, wsample!
import Random: bitrand, rand!, randn!
import SparseArrays: rowvals, nonzeros, nzrange, SparseMatrixCSC, SparseVector, SPQR
import SparseArrays: sparse, sprandn, spzeros


# Include the files corresponding to the top-level techniques
include("Compressors.jl")
include("Solvers.jl")
include("Approximators.jl")
include("helpers/mul_dim_checks.jl")

# Export Approximator types and functions
export Approximator, ApproximatorRecipe, ApproximatorAdjoint
export rapproximate, rapproximate!, complete_approximator
export RangeApproximator, RangeApproximatorRecipe
export RangeFinder, RangeFinderRecipe
export RandSVD, RandSVDRecipe
export CUR, CURRecipe
export CURCore, CURCoreRecipe, CURCoreAdjoint
export CrossApproximation, CrossApproximationRecipe

# Export Compressor types and functions
export Compressor, CompressorRecipe, CompressorAdjoint
export Cardinality, Left, Right, Undef
export complete_compressor, update_compressor!
export CountSketch, CountSketchRecipe
export FJLT, FJLTRecipe
export Gaussian, GaussianRecipe
export Identity, IdentityRecipe
export Sampling, SamplingRecipe
export SparseSign, SparseSignRecipe
export SRHT, SRHTRecipe

# Export Distribution types and functions
export Distribution, DistributionRecipe
export complete_distribution, update_distribution!, sample_distribution!
export Uniform, UniformRecipe, L2Norm, L2NormRecipe

# Export Solver types and functions
export Solver, SolverRecipe
export Kaczmarz, KaczmarzRecipe
export ColumnProjection, ColumnProjectionRecipe
export complete_solver, update_solver!, rsolve!
export IHS, IHSRecipe

# Export Logger types and functions
export Logger, LoggerRecipe
export BasicLogger, BasicLoggerRecipe
export complete_logger, update_logger!, reset_logger!
export threshold_stop

# Export SubSolver types and functions
export SubSolver, SubSolverRecipe, ldiv!
export complete_sub_solver, update_sub_solver!
export LQSolver, LQSolverRecipe
export QRSolver, QRSolverRecipe

# Export SolverError types and functions
export SolverError, SolverErrorRecipe
export complete_error, compute_error
export FullResidual, FullResidualRecipe, CompressedResidual, CompressedResidualRecipe
export LSGradient, LSGradientRecipe

# Export ApproximatorError types and functions
export ApproximatorError, ApproximatorErrorRecipe
export complete_approximator_error, compute_approximator_error, compute_approximator_error!

# Export Selector types and functions
export Selector, SelectorRecipe
export LUPP, LUPPRecipe, QRCP, QRCPRecipe
export complete_selector, update_selector!, select_indices!


end #module
