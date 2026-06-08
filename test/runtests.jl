using SimpleBoundaryValueDiffEq
using Test
using Aqua
using BVProblemLibrary
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@testset "SimpleBoundaryValueDiffEq.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @testset "Code quality (Aqua.jl)" begin
            include("aqua_tests.jl")
        end
        @testset "Test MIRK methods convergence" begin
            include("mirk_tests.jl")
        end

        @testset "Test Shooting methods convergence" begin
            include("shooting_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "NoPre" && isempty(VERSION.prerelease)
        import Pkg
        Pkg.activate("nopre")
        Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @testset "Code linting (JET.jl)" begin
            include("nopre/jet_tests.jl")
        end
    end
end
