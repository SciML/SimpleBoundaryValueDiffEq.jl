using .SimpleBoundaryValueDiffEq
using Test
using Aqua
using JET

@testset "SimpleBoundaryValueDiffEq.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SimpleBoundaryValueDiffEq)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SimpleBoundaryValueDiffEq; target_defined_modules = true)
    end
    # Write your tests here.
end
