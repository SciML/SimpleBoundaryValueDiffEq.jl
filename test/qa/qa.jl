using SciMLTesting, SimpleBoundaryValueDiffEq, JET

function binding_owner(pkg::Module, name::Symbol)
    value = getfield(pkg, name)
    return value isa Module ? value : parentmodule(value)
end

# Only dependency-owned reexports are skipped from rendered-docs coverage.
const DEPENDENCY_REEXPORTS = Tuple(
    name for name in public_api_names(SimpleBoundaryValueDiffEq)
        if isdefined(SimpleBoundaryValueDiffEq, name) &&
        binding_owner(SimpleBoundaryValueDiffEq, name) !== SimpleBoundaryValueDiffEq
)

run_qa(
    SimpleBoundaryValueDiffEq;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true, rendered_ignore = DEPENDENCY_REEXPORTS),
)
