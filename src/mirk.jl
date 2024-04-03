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
    pt = prob.problem_type

    stage = alg_stage(alg)
    M, u0, guess = __extract_details(prob, N)
    resid = [similar(u0) for _ in 1:(N + 1)]
    discrete_stages = [similar(u0) for _ in 1:stage]

    c, v, b, x = constructSimpleMIRK(alg)

    function loss(u, p)
        u = unflatten_vector(u, M, N)
        resid!(resid, u, mesh, discrete_stages, c, v, b, x, prob, dt)
        eval_bc_residual!(resid, u, mesh, prob, pt)
        res = flatten_vector(resid)
        return res
    end

    jac(u, p) = FiniteDiff.finite_difference_jacobian(y -> loss(y, p), u)

    nlfun = NonlinearFunction(loss, jac = jac)
    nlprob = NonlinearProblem(nlfun, flatten_vector(guess))
    nlsol = solve(nlprob, alg.nlsolve)
    u = unflatten_vector(nlsol.u, M, N)

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

function eval_bc_residual!(residual, y, mesh, prob, pt::SciMLBase.StandardBVProblem)
    prob.f.bc(residual[end], y, prob.p, mesh)
end

function eval_bc_residual!(residual, y, _, prob, pt::SciMLBase.TwoPointBVProblem)
    length_a = length(prob.f.bcresid_prototype[1])
    length_b = length(prob.f.bcresid_prototype[2])
    @views first(prob.f.bc)(residual[end][1:length_a], y[1], prob.p)
    @views last(prob.f.bc)(
        residual[end][(length_a + 1):(length_a + length_b)], y[end], prob.p)
end

function resid!(residual, y, mesh, discrete_stages, c, v, b, x, prob, dt)
    iip = SciMLBase.isinplace(prob)
    for i in 1:(length(mesh) - 1)
        for r in eachindex(discrete_stages)
            x_temp = mesh[i] + c[r] * dt
            y_temp = (1 - v[r]) * y[i] + v[r] * y[i + 1]
            if r > 1
                y_temp += dt * sum(j -> x[r, j] * discrete_stages[j], 1:(r - 1))
            end
            if iip
                prob.f(discrete_stages[r], y_temp, prob.p, x_temp)
            else
                tmp = prob.f(y_temp, prob.p, x_temp)
                copyto!(discrete_stages[r], tmp)
            end
        end
        residual[i] = y[i + 1] - y[i] -
                      dt * sum(j -> b[j] * discrete_stages[j], 1:length(discrete_stages))
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

@inline function flatten_vector(src::Vector{T}) where {T <: AbstractArray}
    M = length(src[1])
    new_src = zeros(M * length(src))
    for i in eachindex(src)
        new_src[(((i - 1) * M) + 1):(i * M)] = src[i]
    end
    return new_src
end

@inline function unflatten_vector(src::AbstractArray{T}, M::P, N::P) where {P <: Integer, T}
    dest = [zeros(M) for i in 1:(N + 1)]
    for i in 1:(N + 1)
        copyto!(dest[i], src[((M * (i - 1)) + 1):(M * i)])
    end
    return dest
end
