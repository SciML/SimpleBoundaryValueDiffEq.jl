using SimpleBoundaryValueDiffEq, LinearAlgebra, Test

tspan = (0.0, 100.0)
u0 = [0.0, 1.0]
# Inplace
function f1!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end

function bc1!(resid, sol, p, t)
    t₀, t₁ = first(t), last(t)
    resid[1] = sol(t₀)[1]
    resid[2] = sol(t₁)[1] - 1
    return nothing
end

bvp1 = BVProblem(f1!, bc1!, u0, tspan)

# Out of Place
f1(u, p, t) = [u[2], -u[1]]

function bc1(sol, p, t)
    t₀, t₁ = first(t), last(t)
    return [sol(t₀)[1], sol(t₁)[1] - 1]
end

bvp2 = BVProblem(f1, bc1, u0, tspan)

# Inplace
bc2a!(resid, ua, p) = (resid[1] = ua[1])
bc2b!(resid, ub, p) = (resid[1] = ub[1] - 1)

bvp3 = TwoPointBVProblem(f1!, (bc2a!, bc2b!), u0, tspan;
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1)))

# Out of Place
bc2a(ua, p) = [ua[1]]
bc2b(ub, p) = [ub[1] - 1]

bvp4 = TwoPointBVProblem(f1, (bc2a, bc2b), u0, tspan)

for prob in (bvp1, bvp2, bvp3, bvp4)
    sol = solve(prob, SimpleShooting(), abstol = 1e-8, reltol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid, Inf) < 1e-8
end