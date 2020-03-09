using Documenter, ParticleDetection

makedocs(
    modules = [ParticleDetection],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Francesco Alemanno",
    sitename = "ParticleDetection.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/francescoalemanno/ParticleDetection.jl.git",
    push_preview = true
)
