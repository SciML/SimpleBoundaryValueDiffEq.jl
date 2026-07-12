module SimpleBoundaryValueDiffEq

using Reexport: @reexport
import CommonSolve: solve!
import DiffEqBase
import DiffEqBase: solve
import FiniteDiff
import SciMLBase
@reexport using SciMLBase
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: AbstractBVPAlgorithm, BVProblem, NonlinearFunction, NonlinearProblem,
    ODEProblem, ODESolution, StandardBVProblem, TwoPointBVProblem,
    build_solution, isinplace
using SimpleNonlinearSolve: SimpleNewtonRaphson

abstract type SimpleBoundaryValueDiffEqAlgorithm <: AbstractBVPAlgorithm end
abstract type AbstractSimpleMIRK <: SimpleBoundaryValueDiffEqAlgorithm end
abstract type AbstractSimpleShooting <: SimpleBoundaryValueDiffEqAlgorithm end

include("utils.jl")
include("mirk.jl")
include("single_shooting.jl")
include("precompilation.jl")

end
