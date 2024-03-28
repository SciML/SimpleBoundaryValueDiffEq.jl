using SimpleBoundaryValueDiffEq, Test, BVProblemLibrary
using LinearAlgebra

for i=1:18
    # 15 no analytical solution, 13 and 17 test problem is not right.
    if (i==15) || (i==13) || (i==17)
        continue
    end
    @testset "Test MIRK on linear BVP No.$(i)" begin
        prob = eval(Symbol("prob_bvp_linear_$(i)"))
        sol1 = solve(prob, SimpleMIRK4(), dt=0.01)
        sol2 = solve(prob, SimpleMIRK5(), dt=0.01)
        sol3 = solve(prob, SimpleMIRK6(), dt=0.01)

        @test norm(sol1.u .- sol1.u_analytic) < 1e-4
        @test norm(sol2.u .- sol2.u_analytic) < 1e-4
        @test norm(sol3.u .- sol3.u_analytic) < 1e-4
    end
end
