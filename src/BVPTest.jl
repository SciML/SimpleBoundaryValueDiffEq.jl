using DiffEqBase, DifferentialEquations, NonlinearSolve, Plots

include("SimpleBoundaryValueDiffEq.jl")

function test_f(u, p, t)
    [u[2], -0.1 * u[1]]
end

function test_bc(u, p, t)
    [u[1][1] - 1, u[end][1]]
end

prob = BVProblem(test_f, test_bc, [0.0, 0.0], (0.0, 1.0))
final_sol = SimpleBoundaryValueDiffEq.single_shooting(prob, 0.0)

plot(final_sol, linewidth = 2, title = "Solution of the ODE",
    xaxis = "Time (t)", yaxis = "u(t)", label = "u(t)")
