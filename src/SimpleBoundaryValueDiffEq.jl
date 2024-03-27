module SimpleBoundaryValueDiffEq

using Reexport
import DiffEqBase: solve
@reexport using DiffEqBase
using FiniteDiff
using SimpleNonlinearSolve

abstract type AbstractSimpleBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractODEAlgorithm end

include("mirk.jl")

function solve(prob::BVProblem, alg::AbstractSimpleBoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

end
