@testset "Aqua" begin
    using SimpleBoundaryValueDiffEq, Aqua

    Aqua.find_persistent_tasks_deps(SimpleBoundaryValueDiffEq)
    Aqua.test_ambiguities(SimpleBoundaryValueDiffEq; recursive = false)
    Aqua.test_deps_compat(SimpleBoundaryValueDiffEq)
    Aqua.test_project_extras(SimpleBoundaryValueDiffEq)
    Aqua.test_stale_deps(SimpleBoundaryValueDiffEq; ignore = [:TimerOutputs])
    Aqua.test_unbound_args(SimpleBoundaryValueDiffEq)
    Aqua.test_undefined_exports(SimpleBoundaryValueDiffEq)
end
