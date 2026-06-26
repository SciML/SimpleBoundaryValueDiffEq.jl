using SciMLTesting, SimpleBoundaryValueDiffEq, JET, Test

run_qa(
    SimpleBoundaryValueDiffEq;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # SciMLBase internals (still non-public; SimpleBoundaryValueDiffEq
                # subtypes / dispatches on / calls them as part of the SciML BVP interface).
                :AbstractBVPAlgorithm, :StandardBVProblem, :__init, :__solve,
                # FiniteDiff internals (non-public Jacobian kernels).
                :finite_difference_jacobian, Symbol("finite_difference_jacobian!"),
            ),
        ),
    ),
    # Heavy `using DiffEqBase/SciMLBase/SimpleNonlinearSolve/...` style: many implicit
    # imports; making them all explicit is a risky mass refactor tracked in
    # https://github.com/SciML/SimpleBoundaryValueDiffEq.jl/issues/55.
    ei_broken = (:no_implicit_imports,),
)
