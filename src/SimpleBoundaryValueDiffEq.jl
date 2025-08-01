module SimpleBoundaryValueDiffEq

using Reexport
import DiffEqBase: solve
using DiffEqBase
@reexport using SciMLBase
using FiniteDiff
using OrdinaryDiffEqTsit5
using SimpleNonlinearSolve
using KernelAbstractions, CUDA, DiffEqGPU

abstract type SimpleBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractSimpleMIRK <: SimpleBoundaryValueDiffEqAlgorithm end
abstract type AbstractSimpleShooting <: SimpleBoundaryValueDiffEqAlgorithm end

include("utils.jl")
include("mirk.jl")
include("single_shooting.jl")

end
