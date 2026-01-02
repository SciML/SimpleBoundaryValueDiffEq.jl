using SimpleBoundaryValueDiffEq
using Test
using Aqua
using BVProblemLibrary
using JET
using SafeTestsets

@testset "SimpleBoundaryValueDiffEq.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        include("aqua_tests.jl")
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SimpleBoundaryValueDiffEq; target_modules = (SimpleBoundaryValueDiffEq,))
    end
    @testset "Test MIRK methods convergence" begin
        include("mirk_tests.jl")
    end

    @testset "Test Shooting methods convergence" begin
        include("shooting_tests.jl")
    end
end
