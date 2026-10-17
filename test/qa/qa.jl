using SciMLTesting, SimpleBoundaryValueDiffEq, Test

const REEXPORTS = (
    :BVProblem, :ODESolution, :ReturnCode, :TwoPointBVProblem, :remake, :solve,
    :successful_retcode,
)

run_qa(SimpleBoundaryValueDiffEq; reexports_allow = REEXPORTS)

@testset "Reexport surface" begin
    @testset "$name" for name in REEXPORTS
        @test name in names(SimpleBoundaryValueDiffEq)
        @test isdefined(@__MODULE__, name)
    end
end
