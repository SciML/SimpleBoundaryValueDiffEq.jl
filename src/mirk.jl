struct SimpleMIRK4{N} <: AbstractSimpleMIRK
    nlsolve::N
end
SimpleMIRK4(; nlsolve = SimpleNewtonRaphson()) = SimpleMIRK4(nlsolve)

struct SimpleMIRK5{N} <: AbstractSimpleMIRK
    nlsolve::N
end
SimpleMIRK5(; nlsolve = SimpleNewtonRaphson()) = SimpleMIRK5(nlsolve)

struct SimpleMIRK6{N} <: AbstractSimpleMIRK
    nlsolve::N
end
SimpleMIRK6(; nlsolve = SimpleNewtonRaphson()) = SimpleMIRK6(nlsolve)

export SimpleMIRK4
export SimpleMIRK5
export SimpleMIRK6

alg_order(alg::SimpleMIRK4) = 4
alg_stage(alg::SimpleMIRK4) = 3

alg_order(alg::SimpleMIRK5) = 5
alg_stage(alg::SimpleMIRK5) = 4

alg_order(alg::SimpleMIRK6) = 6
alg_stage(alg::SimpleMIRK6) = 5

function DiffEqBase.solve(prob::BVProblem, alg::AbstractSimpleMIRK; dt = 0.0, kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    N = Int(cld(prob.tspan[2] - prob.tspan[1], dt))
    mesh = collect(range(prob.tspan[1], prob.tspan[2], length = N + 1))
    iip = SciMLBase.isinplace(prob)
    pt = prob.problem_type

    stage = alg_stage(alg)
    M, u0, guess = __extract_details(prob, N)
    resid = [similar(u0) for _ in 1:(N + 1)]
    y = [similar(u0) for _ in 1:(N + 1)]
    discrete_stages = [similar(u0) for _ in 1:stage]

    c, v, b, x = constructSimpleMIRK(alg)

    loss = if iip
        function (res, u, p)
            y_ = recursive_unflatten!(y, u)
            Φ!(resid, y_, mesh, discrete_stages, c, v, b, x, prob, dt)
            eval_bc_residual!(resid[end], y_, mesh, prob, pt)
            recursive_flatten!(res, resid)
            return res
        end
    else
        function (u, p)
            y_ = recursive_unflatten!(y, u)
            resid_co = Φ(y_, mesh, discrete_stages, c, v, b, x, prob, dt)
            resid_bc = eval_bc_residual(y_, mesh, prob, pt)
            return vcat(mapreduce(vec, vcat, resid_co), resid_bc)
        end
    end

    jac = if iip
        (J, u, p) -> FiniteDiff.finite_difference_jacobian!(
            J, (res, y) -> loss(res, y, p), u)
    else
        (u, p) -> FiniteDiff.finite_difference_jacobian(y -> loss(y, p), u)
    end

    nlfun = NonlinearFunction(loss, jac = jac)
    nlprob = NonlinearProblem(nlfun, reduce(vcat, guess))
    nlsol = solve(nlprob, alg.nlsolve)
    u = recursive_unflatten!(y, nlsol.u)

    return DiffEqBase.build_solution(prob, alg, mesh, u)
end

@inline function __extract_details(prob::BVProblem, N::Integer)
    if isa(prob.u0[1], AbstractArray)
        u0 = prob.u0[1]
        guess = prob.u0
    else
        u0 = prob.u0
        guess = [u0 for i in 1:(N + 1)]
    end
    M = length(u0)

    return M, u0, guess
end

function Φ!(residual, y, mesh, discrete_stages, c, v, b, x, prob, dt)
    for i in 1:(length(mesh) - 1)
        for r in eachindex(discrete_stages)
            x_temp = mesh[i] + c[r] * dt
            y_temp = (1 - v[r]) * y[i] + v[r] * y[i + 1]
            if r > 1
                y_temp += dt * sum(j -> x[r, j] * discrete_stages[j], 1:(r - 1))
            end
            prob.f(discrete_stages[r], y_temp, prob.p, x_temp)
        end
        residual[i] = y[i + 1] - y[i] -
                      dt * sum(j -> b[j] * discrete_stages[j], 1:length(discrete_stages))
    end
end

function Φ(y, mesh, discrete_stages, c, v, b, x, prob, dt)
    residual = [similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    for i in 1:(length(mesh) - 1)
        for r in eachindex(discrete_stages)
            x_temp = mesh[i] + c[r] * dt
            y_temp = (1 - v[r]) * y[i] + v[r] * y[i + 1]
            if r > 1
                y_temp += dt * sum(j -> x[r, j] * discrete_stages[j], 1:(r - 1))
            end
            tmp = prob.f(y_temp, prob.p, x_temp)
            copyto!(discrete_stages[r], tmp)
        end
        residual[i] = y[i + 1] - y[i] -
                      dt * sum(j -> b[j] * discrete_stages[j], 1:length(discrete_stages))
    end
    return residual
end

function constructSimpleMIRK(alg::SimpleMIRK4)
    c = [0, 1, 1 // 2, 3 // 4]
    v = [0, 1, 1 // 2, 27 // 32]
    b = [1 // 6, 1 // 6, 2 // 3, 0]
    x = [0 0 0 0
         0 0 0 0
         1//8 -1//8 0 0
         3//64 -9//64 0 0]

    return c, v, b, x
end

function constructSimpleMIRK(alg::SimpleMIRK5)
    c = [0, 1, 3 // 4, 3 // 10]
    v = [0, 1, 27 // 32, 837 // 1250]
    b = [5 // 54, 1 // 14, 32 // 81, 250 // 567]
    x = [0 0 0 0
         0 0 0 0
         3//64 -9//64 0 0
         21//1000 63//5000 -252//625 0]

    return c, v, b, x
end

function constructSimpleMIRK(alg::SimpleMIRK6)
    c = [0, 1, 1 // 4, 3 // 4, 1 // 2]
    v = [0, 1, 5 // 32, 27 // 32, 1 // 2]
    b = [7 // 90, 7 // 90, 16 // 45, 16 // 45, 2 // 15, 0, 0, 0, 0]
    x = [0 0 0 0 0
         0 0 0 0 0
         9//64 -3//64 0 0 0
         3//64 -9//64 0 0 0
         -5//24 5//24 2//3 -2//3 0]

    return c, v, b, x
end
