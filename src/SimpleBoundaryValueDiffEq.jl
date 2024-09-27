module SimpleBoundaryValueDiffEq

using Reexport
import DiffEqBase: solve
@reexport using DiffEqBase
using FiniteDiff
using OrdinaryDiffEqTsit5
using SimpleNonlinearSolve

abstract type SimpleBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractSimpleMIRK <: SimpleBoundaryValueDiffEqAlgorithm end
abstract type AbstractSimpleShooting <: SimpleBoundaryValueDiffEqAlgorithm end

include("utils.jl")
include("mirk.jl")
include("single_shooting.jl")

end
