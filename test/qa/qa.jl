using SciMLTesting, SimpleBoundaryValueDiffEq, JET, Test

run_qa(
    SimpleBoundaryValueDiffEq;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # SciMLBase BVP-interface internals (still non-public in SciMLBase 3.30.0):
                # the BVP-algorithm supertype this package subtypes, and the standard
                # BVP problem-type marker it dispatches on. No public owner to migrate to.
                :AbstractBVPAlgorithm, :StandardBVProblem,
                # FiniteDiff Jacobian kernels (FiniteDiff declares none of its API public).
                :finite_difference_jacobian, Symbol("finite_difference_jacobian!"),
            ),
        ),
    ),
)
