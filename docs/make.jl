using Documenter
using SimpleBoundaryValueDiffEq

DocMeta.setdocmeta!(
    SimpleBoundaryValueDiffEq,
    :DocTestSetup,
    :(using SimpleBoundaryValueDiffEq);
    recursive = true,
)

makedocs(;
    modules = [SimpleBoundaryValueDiffEq],
    sitename = "SimpleBoundaryValueDiffEq.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/SimpleBoundaryValueDiffEq/stable/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Reexported interface" => "reexports.md",
        "Developer API" => "developer_api.md",
    ],
    checkdocs = :exports,
)
