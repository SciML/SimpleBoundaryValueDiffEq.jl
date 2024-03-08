module SimpleBoundaryValueDiffEq

using DiffEqBase, DifferentialEquations, NonlinearSolve, Plots

# Only for out-of-place BVP, in place needs to be converted to out-of-place
function single_shooting(prob, p)
    function ivp_deviation(u0, p)
        ivp = ODEProblem(prob.f, u0, prob.tspan, 0)
        ivp_sol = solve(ivp)
        prob.f.bc(ivp_sol, 0, 0)
    end

    root_finding_prob = NonlinearProblem(ivp_deviation, prob.u0, 0)
    best_u0 = solve(root_finding_prob)
    ivp_with_init_condition = ODEProblem(prob.f, best_u0, prob.tspan)
    solve(ivp_with_init_condition)
end

end