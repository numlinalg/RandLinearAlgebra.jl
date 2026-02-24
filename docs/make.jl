using Documenter
using DocumenterCitations
using RandLinearAlgebra

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)
makedocs(
    sitename = "RandLinearAlgebra",
    format = Documenter.HTML(
    collapselevel=1,
    ),
    plugins=[bib],
    modules = [RandLinearAlgebra],
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Introduction" => "manual/introduction.md", 
            "Compression" => "manual/compression.md",
            "Low-Rank Approximation" => "manual/low_rank_approximators.md",
            "Linear Systems" => "manual/linear_systems.md"
        ],
        "Tutorials" => [
            "tutorials/tutorials_overview.md",
            "Compressors" => [
                "tutorials/compressors/compressor_example.md",
            ],
            "Consistent Linear System" => [
                "tutorials/consistent_system/consistent_system.md",
                "tutorials/consistent_system/configuring_kaczmarz.md",
            ],
            "Least Squares" => [
                "tutorials/least_squares/least_squares.md",
                #"tutorials/least_squares/least_squares_configure.md",
            ],
        ],
        "API Reference" => [
            "Compressors" => [
                "Compressors API" => "api/compressors.md",
                "Distributions API" => "api/distributions.md",
            ],
            "Solvers" => [
                "Solvers API" => "api/solvers.md",
                "SubSolvers API" => "api/sub_solvers.md",
                "SolverErrors API" => "api/solver_errors.md",
                "Loggers API" => "api/loggers.md",
            ],
            "Approximators" => [
                "Approximators API" => "api/approximators.md",
                "Selectors API" => "api/selectors.md",
                "ApproximatorErrors API" => "api/approximator_errors.md",
            ],
        ],
        "References" => "references.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/numlinalg/RandLinearAlgebra.jl",
    devbranch = "main", # master's newest commit will become dev
    push_preview = true # pull requests to the master will become available
)
