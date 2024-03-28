module SimpleBoundaryValueDiffEq

using Reexport
import DiffEqBase: solve
@reexport using DiffEqBase
using FiniteDiff
using SimpleNonlinearSolve

abstract type SimpleBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractSimpleMIRK <: SimpleBoundaryValueDiffEqAlgorithm end

include("mirk.jl")

function solve(prob::BVProblem, alg::SimpleBoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

end
