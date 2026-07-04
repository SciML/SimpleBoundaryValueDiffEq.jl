module SimpleBoundaryValueDiffEq

using Reexport: Reexport, @reexport
@reexport using SciMLBase
using SciMLBase: SciMLBase, BVProblem, NonlinearFunction, NonlinearProblem, ODEProblem,
    isinplace, solve!
using DiffEqBase: DiffEqBase, solve
using FiniteDiff: FiniteDiff
using OrdinaryDiffEqTsit5: OrdinaryDiffEqTsit5, Tsit5
using SimpleNonlinearSolve: SimpleNonlinearSolve, SimpleNewtonRaphson
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload

abstract type SimpleBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractSimpleMIRK <: SimpleBoundaryValueDiffEqAlgorithm end
abstract type AbstractSimpleShooting <: SimpleBoundaryValueDiffEqAlgorithm end

include("utils.jl")
include("mirk.jl")
include("single_shooting.jl")
include("precompilation.jl")

end
