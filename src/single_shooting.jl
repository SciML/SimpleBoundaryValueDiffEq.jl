struct SimpleShooting{N, O} <: AbstractSimpleShooting
    nlsolve::N
    ode_alg::O
end
SimpleShooting(; nlsolve = SimpleNewtonRaphson(), ode_alg = Tsit5()) = SimpleShooting(nlsolve, ode_alg)

export SimpleShooting

function DiffEqBase.solve(prob::BVProblem, alg::SimpleShooting; abstol=1e-6, reltol=1e-6, odesolve_kwargs = (;), nlsolve_kwargs = (;))
    u0 = prob.u0
    pt = prob.problem_type

    iip = isinplace(prob)

    internal_prob = ODEProblem{iip}(prob.f, u0, prob.tspan, prob.p)
    ode_cache = SciMLBase.__init(internal_prob, alg.ode_alg)

    loss = if iip
        function (resid, u, p)
        SciMLBase.reinit!(ode_cache, u)
        odesol = solve!(ode_cache)
        eval_bc_residual!(resid, odesol, odesol.t, prob, pt)
        return resid
        end
    else
        function (u, p)
        SciMLBase.reinit!(ode_cache, u)
        odesol = solve!(ode_cache)
        return eval_bc_residual(odesol, odesol.t, prob, pt)
        end
    end

    jac = if iip
        (J, u, p) -> FiniteDiff.finite_difference_jacobian!(J, (resid, y) -> loss(resid, y, p), u)
    else
        (u, p) -> FiniteDiff.finite_difference_jacobian(y -> loss(y, p), u)
    end

    nlfun = NonlinearFunction(loss, jac = jac)
    nlprob = NonlinearProblem(nlfun, u0)
    nlsol = solve(nlprob, alg.nlsolve, abstol = abstol, reltol = reltol, nlsolve_kwargs...)

    internal_prob_final = ODEProblem{iip}(
        prob.f, nlsol.u, prob.tspan, prob.p)
    odesol = SciMLBase.__solve(internal_prob_final, alg.ode_alg, abstol = abstol, reltol = reltol, odesolve_kwargs...)

    return DiffEqBase.build_solution(prob, alg, odesol.t, odesol.u, retcode = odesol.retcode, resid = nlsol.resid)
end
