using SimpleBoundaryValueDiffEq, Test, BVProblemLibrary, DiffEqDevTools

dts1 = 1 .// 2 .^ (7:-1:4)
dts2 = 1 .// 2 .^ (5:-1:3)
testTol = 0.2
for i in 1:5
    @testset "Test MIRK on linear BVP No.$(i)" begin
        sim1 = test_convergence(dts1, eval(Symbol("prob_bvp_linear_$(i)")), SimpleMIRK4())
        @test sim1.𝒪est[:l2]≈4 atol=testTol

        sim2 = test_convergence(dts1, eval(Symbol("prob_bvp_linear_$(i)")), SimpleMIRK5())
        @test sim2.𝒪est[:l2]≈5 atol=testTol

        sim3 = test_convergence(dts2, eval(Symbol("prob_bvp_linear_$(i)")), SimpleMIRK6())
        @test sim3.𝒪est[:l2]≈6 atol=testTol
    end
end
