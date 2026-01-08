using SimpleBoundaryValueDiffEq, Test, JET

@testset "Code linting (JET.jl)" begin
    JET.test_package(SimpleBoundaryValueDiffEq; target_modules = (SimpleBoundaryValueDiffEq,))
end
