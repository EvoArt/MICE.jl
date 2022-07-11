using MICE
using Documenter

DocMeta.setdocmeta!(MICE, :DocTestSetup, :(using MICE); recursive=true)

makedocs(;
    modules=[MICE],
    authors="Arthur Newbury",
    repo="https://github.com/EvoArt/MICE.jl/blob/{commit}{path}#{line}",
    sitename="MICE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/MICE.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/EvoArt/MICE.jl",
)
