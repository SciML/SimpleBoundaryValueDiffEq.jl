using SimpleBoundaryValueDiffEq
using Test
using Aqua
using BVProblemLibrary
using SafeTestsets
using SciMLTesting

const GROUP = current_group()

run_tests(;
    core = () -> begin
        @testset "Code quality (Aqua.jl)" begin
            include(joinpath(@__DIR__, "aqua_tests.jl"))
        end
        @testset "Test MIRK methods convergence" begin
            include(joinpath(@__DIR__, "mirk_tests.jl"))
        end
        @testset "Test Shooting methods convergence" begin
            include(joinpath(@__DIR__, "shooting_tests.jl"))
        end
    end,
    qa = () -> begin
        activate_group_env(joinpath(@__DIR__, "qa"))
        include(joinpath(@__DIR__, "qa", "qa.jl"))
    end,
    groups = Dict(
        # `env` is left off the group entry so this group also runs under "All",
        # matching the original `GROUP == "All" || GROUP == "NoPre" && ...` branch.
        # The original prerelease guard applied only to the bare-NoPre path, so it
        # is preserved verbatim inside the thunk: under "All" the block always runs,
        # under "NoPre" only when the running Julia is not a prerelease. The nopre
        # sub-env is activated by the thunk itself (only when the block actually runs).
        "NoPre" => () -> begin
            if GROUP == "All" || GROUP == "NoPre" && isempty(VERSION.prerelease)
                activate_group_env(joinpath(@__DIR__, "nopre"))
                @testset "Code linting (JET.jl)" begin
                    include(joinpath(@__DIR__, "nopre", "jet_tests.jl"))
                end
            end
        end,
    ),
)
