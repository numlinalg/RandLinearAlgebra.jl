# Shared preamble for test modules. Include inside each test module, e.g.
#   module my_test
#   include("../test_helpers/preamble.jl")
# so that everything below is evaluated in that module's scope.
using Test, RandLinearAlgebra, Random, LinearAlgebra, SparseArrays, StatsBase
import LinearAlgebra: mul!

# Local copies of the test helper modules, resolved relative to this file
include("field_test_macros.jl")
include("approx_tol.jl")
using .FieldTest
using .ApproxTol
