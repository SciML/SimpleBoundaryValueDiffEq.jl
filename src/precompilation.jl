using PrecompileTools

@setup_workload begin
    # Simple harmonic oscillator BVP for precompilation
    # Inplace ODE function
    function _precompile_f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    # Out-of-place ODE function
    _precompile_f(u, p, t) = [u[2], -u[1]]

    # Boundary conditions for SimpleMIRK (uses vector of state vectors)
    function _precompile_bc_mirk!(resid, y, p, mesh)
        resid[1] = y[1][1]
        resid[2] = y[end][1] - 1
        return nothing
    end

    _precompile_bc_mirk(y, p, mesh) = [y[1][1], y[end][1] - 1]

    u0 = [0.0, 1.0]
    tspan = (0.0, 1.0)

    @compile_workload begin
        # SimpleMIRK4 with StandardBVProblem (inplace) - 62% TTFX improvement
        prob_mirk_iip = BVProblem{true}(_precompile_f!, _precompile_bc_mirk!, u0, tspan)
        solve(prob_mirk_iip, SimpleMIRK4(), dt = 0.5)

        # SimpleMIRK4 with StandardBVProblem (out-of-place) - 28% TTFX improvement
        prob_mirk_oop = BVProblem{false}(_precompile_f, _precompile_bc_mirk, u0, tspan)
        solve(prob_mirk_oop, SimpleMIRK4(), dt = 0.5)
    end
end
