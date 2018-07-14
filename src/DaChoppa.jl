
__precompile__()

module DaChoppa
    import MathProgBase
    using JuMP
    using ConicNonlinearBridge

    include("solver.jl")
    include("algorithm.jl")
end
