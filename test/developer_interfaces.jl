using SimpleBoundaryValueDiffEq: AbstractSimpleMIRK
import SimpleBoundaryValueDiffEq: alg_order, alg_stage, constructSimpleMIRK
using SciMLBase: BVProblem, solve, successful_retcode
using SimpleNonlinearSolve: SimpleNewtonRaphson
using Test

struct MockMIRK{N} <: AbstractSimpleMIRK
    nlsolve::N
end

MockMIRK() = MockMIRK(SimpleNewtonRaphson())

alg_order(::MockMIRK) = 4
alg_stage(::MockMIRK) = 3

function constructSimpleMIRK(::MockMIRK)
    c = [0, 1, 1 // 2, 3 // 4]
    v = [0, 1, 1 // 2, 27 // 32]
    b = [1 // 6, 1 // 6, 2 // 3, 0]
    x = [
        0 0 0 0
        0 0 0 0
        1 // 8 -1 // 8 0 0
        3 // 64 -9 // 64 0 0
    ]
    return c, v, b, x
end

@testset "AbstractSimpleMIRK developer interface" begin
    f(u, p, t) = zero(u)
    bc(y, p, t) = [y[1][1] - 1]
    prob = BVProblem(f, bc, [0.0], (0.0, 1.0))

    sol = solve(prob, MockMIRK(); dt = 0.25)

    @test alg_order(MockMIRK()) == 4
    @test alg_stage(MockMIRK()) == 3
    @test successful_retcode(sol)
    @test sol.u[end][1] ≈ 1.0
end
