
__precompile__()

module DaChoppa
    import MathProgBase
    using JuMP

    include("solver.jl")
    include("algorithm.jl")
end
