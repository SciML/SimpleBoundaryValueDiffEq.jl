module SimpleBoundaryValueDiffEq

import CommonSolve: solve!
import DiffEqBase
import FiniteDiff
import SciMLBase
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: AbstractBVPAlgorithm, BVProblem, NonlinearFunction, NonlinearProblem,
    ODEProblem, ODESolution, ReturnCode, StandardBVProblem, TwoPointBVProblem,
    build_solution, init, isinplace, remake, solve, successful_retcode
using SimpleNonlinearSolve: SimpleNewtonRaphson

export AbstractSimpleMIRK, SimpleMIRK4, SimpleMIRK5, SimpleMIRK6, SimpleShooting
export alg_order, alg_stage, constructSimpleMIRK
export BVProblem, ODESolution, ReturnCode, TwoPointBVProblem, remake, solve,
    successful_retcode

"""
    AbstractSimpleMIRK <: AbstractBVPAlgorithm

Developer interface for mono-implicit Runge-Kutta algorithms handled by
SimpleBoundaryValueDiffEq.jl.

# Interface

To implement a new method, define a concrete subtype and `import`
[`alg_order`](@ref), [`alg_stage`](@ref), and [`constructSimpleMIRK`](@ref) before extending
them for that subtype. The generic `solve(prob::BVProblem, alg::AbstractSimpleMIRK; dt)` method
then constructs and solves the collocation nonlinear problem. Implementations must provide a
nonlinear solver as the `nlsolve` field or otherwise support `getproperty(alg, :nlsolve)`.

# Required Methods

  - `alg_order(alg) -> Integer`: formal convergence order of `alg`.
  - `alg_stage(alg) -> Integer`: number of stage derivatives used by `alg`.
  - `constructSimpleMIRK(alg) -> (c, v, b, x)`: tableau data whose first
    `alg_stage(alg)` entries and rows describe the stage derivatives. The coefficients must
    be compatible with arithmetic on the state and time element types accepted by the BVP.

# Usage

See the developer API reference for a minimal subtype and the generic contract test in this
package for an executable example. This is a developer interface; end users should select one
of the documented `SimpleMIRK*` algorithms instead of subtyping it.
"""
abstract type AbstractSimpleMIRK <: AbstractBVPAlgorithm end

include("utils.jl")
include("mirk.jl")
include("single_shooting.jl")
include("precompilation.jl")

end
