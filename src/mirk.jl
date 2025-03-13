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

    return DiffEqBase.build_solution(prob, alg, mesh, u; retcode = nlsol.retcode)
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
    size_u = size(y[1])[1]
    residual = [similar(yᵢ) for yᵢ in y[1:(end - 1)]]

    y_flat = recursive_flatten!(zeros(sum(length, y)), y)
    stages_flat = recursive_flatten!(zeros(sum(length, discrete_stages)), discrete_stages)

    # right now this backend is forced till we make a solver similar to
    # EnsembleGPUKernel which takes the backend as a parameter
    len_mesh = length(mesh)
    gpu_mesh = CuArray(mesh)
    gpu_residual = CuArray(zeros(Float32, size_u * len_mesh))
    k = reskernel!(CUDA.CUDABackend())
    
    gpu_c, gpu_v, gpu_b, gpu_x = (CuArray(Float32.(n)) for n in [c,v,b,x])
    y_flat = CuArray(y_flat)
    stages_flat = CuArray(stages_flat)

    prob = DiffEqGPU.make_prob_compatible(prob)
    prob = cu(prob)

    k(gpu_residual, y_flat, gpu_mesh, len_mesh, stages_flat,
        length(discrete_stages), gpu_c, gpu_v, gpu_b, gpu_x, prob, dt, size_u, ndrange=len_mesh)

    recursive_unflatten!(residual, gpu_residual)
    return residual
end

@kernel function reskernel!(residual, y, mesh, len_mesh, stages_flat, n_stage, c, v, b, x, prob, dt, size_u)
    i = @index(Global)
    if i <= (len_mesh - 1)
        for r in 1:n_stage
            @inbounds x_temp = mesh[i] + c[r] * dt
            y_temp = []

            # construct y_temp for this mesh iteration
            for z = 0:(size_u - 1)
                @inbounds push!(y_temp, (1 - v[r]) * y[i + z] + v[r] * y[i + size_u + z])

                # add summation of x_rj * K_j to y_temp 
                if r > 1
                    summation_xk = 0
                    for j = 1:(r-1)
                        @inbounds summation_xk += x[r,j] * stages_flat[(j - 1)*size_u + z + 1]
                    end
                    @inbounds y_temp[z + 1] += dt * summation_xk
                end
            end

            # get prob.f and replace the stage with its result
            temp_stage = prob.f(y_temp, prob.p, x_temp)
            for z = 1:size_u
                @inbounds stages_flat[(r-1)*size_u + z] = temp_stage[z]
            end

        end

        # finally, Φᵢ = yᵢ₊₁ - yᵢ - hᵢ∑bᵣKᵣ
        for j = 1:size_u
            sum_bstages = 0
            for r = 1:n_stage
                @inbounds sum_bstages += b[r] * stages_flat[j+(r-1)*size_u]
            end
            @inbounds residual[(i-1)*size_u + j] = y[(i-1)*size_u + j + size_u] - y[(i-1)*size_u + j]
                                        - dt * sum_bstages
        end
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
