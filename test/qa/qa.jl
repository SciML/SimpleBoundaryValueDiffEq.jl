using SciMLTesting, SimpleBoundaryValueDiffEq, JET

run_qa(
    SimpleBoundaryValueDiffEq;
    explicit_imports = true,
)
