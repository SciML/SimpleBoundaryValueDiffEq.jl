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

mutable struct SimpleMIRKCache{iip, T}
    prob::Any
    alg::Any
    mesh::Any
    N::Any
    M::Any
    p::Any
    pt::Any

    order::Any
    stage::Any

    c::Any
    v::Any
    b::Any
    x::Any

    y::Any
    y0::Any
    residual::Any
    discrete_stages::Any
    guess::Any

    dt::Any
    kwargs::Any
end

Base.eltype(::SimpleMIRKCache{iip, T}) where {iip, T} = T

function SciMLBase.__init(prob::BVProblem, alg::AbstractSimpleMIRK; dt = 0.0, kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    N = Int(cld(prob.tspan[2] - prob.tspan[1], dt))
    mesh = collect(range(prob.tspan[1], prob.tspan[2], length = N + 1))
    iip = SciMLBase.isinplace(prob)
    pt = prob.problem_type

    order = alg_order(alg)
    stage = alg_stage(alg)
    T, M, u0, guess = __extract_details(prob, N)
    y = [similar(u0) for i in 1:(N + 1)]
    y0 = [similar(u0) for i in 1:(N + 1)]
    residual = [similar(u0) for i in 1:(N + 1)]
    discrete_stages = [similar(u0) for i in 1:stage]

    c, v, b, x = constructSimpleMIRK(alg)
    return SimpleMIRKCache{iip, T}(prob, alg, mesh, N, M, prob.p, pt,
        order, stage,
        c, v, b, x,
        y, y0, residual, discrete_stages, guess,
        dt, kwargs)
end

function SciMLBase.solve!(cache::SimpleMIRKCache{iip, T}) where {iip, T}
    reorder! = function (resid)
        # reorder the Jacobian matrix such that it is banded
        tmp_last = resid[end]
        for i in (length(resid) - 1):-1:1
            resid[i + 1] = resid[i]
        end
        resid[1], resid[end] = resid[end], tmp_last
    end
    function loss(y, p)
        resid = copy(y)
        unflatten_vector!(cache.y, y)
        resid!(cache)
        eval_bc_residual!(cache, cache.pt)
        flatten_vector!(resid, cache.residual)
        reorder!(resid)
        return resid
    end

    jac(u, p) = FiniteDiff.finite_difference_jacobian(y -> loss(y, p), u)

    ig = zeros(T, cache.M * (cache.N + 1))
    flatten_vector!(ig, cache.guess)

    nlfun = NonlinearFunction(loss, jac = jac)
    nlprob = NonlinearProblem(nlfun, ig, cache.p)
    nlsol = solve(nlprob, cache.alg.nlsolve)
    sol = [similar(cache.prob.u0) for i in 1:(cache.N + 1)]
    unflatten_vector!(sol, nlsol.u)

    return DiffEqBase.build_solution(cache.prob, cache.alg, cache.mesh, sol)
end

function __extract_details(prob::BVProblem, N::Integer)
    if isa(prob.u0[1], AbstractArray)
        u0 = prob.u0[1]
        guess = prob.u0
    else
        u0 = prob.u0
        guess = [u0 for i in 1:(N + 1)]
    end
    M = length(u0)
    T = eltype(u0)

    return T, M, u0, guess
end

function eval_bc_residual!(
        cache::SimpleMIRKCache{iip, T}, pt::SciMLBase.StandardBVProblem) where {iip, T}
    cache.prob.f.bc(cache.residual[end], cache.y, cache.p, cache.mesh)
end

function eval_bc_residual!(
        cache::SimpleMIRKCache{iip, T}, pt::SciMLBase.TwoPointBVProblem) where {iip, T}
    length_a = length(cache.prob.f.bcresid_prototype[1])
    length_b = length(cache.prob.f.bcresid_prototype[2])
    @views cache.prob.f.bc[1](cache.residual[end][1:length_a], cache.y[1], cache.p)
    @views cache.prob.f.bc[2](
        cache.residual[end][(length_a + 1):(length_a + length_b)], cache.y[end], cache.p)
end

function resid!(cache::SimpleMIRKCache{iip, T}) where {iip, T}
    for i in 1:(cache.N)
        for r in 1:(cache.stage)
            x_temp = cache.mesh[i] + cache.c[r] * cache.dt
            y_temp = (1 - cache.v[r]) * cache.y[i] + cache.v[r] * cache.y[i + 1]
            if r > 1
                y_temp += cache.dt *
                          sum(j -> cache.x[r, j] * cache.discrete_stages[j], 1:(r - 1))
            end
            if iip
                cache.prob.f(cache.discrete_stages[r], y_temp, cache.p, x_temp)
            else
                tmp = cache.prob.f(y_temp, cache.p, x_temp)
                copyto!(cache.discrete_stages[r], tmp)
            end
        end
        cache.residual[i] = cache.y[i + 1] - cache.y[i] -
                            cache.dt *
                            sum(j -> cache.b[j] * cache.discrete_stages[j], 1:(cache.stage))
    end
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

function flatten_vector!(dest::T1,
        src::Vector{T2}) where {T1 <: AbstractArray, T2 <: AbstractArray}
    M = length(src[1])
    for i in eachindex(src)
        dest[(((i - 1) * M) + 1):(i * M)] = src[i]
    end
end

function unflatten_vector!(dest::Vector{T1},
        src::T2) where {T1 <: AbstractArray, T2 <: AbstractArray}
    M = length(dest[1])
    for i in 1:length(dest)
        copyto!(dest[i], src[((M * (i - 1)) + 1):(M * i)])
    end
end
