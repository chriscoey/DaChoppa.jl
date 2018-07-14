
export DaChoppaSolver

# Dummy solver
type UnsetSolver <: MathProgBase.AbstractMathProgSolver end

# DaChoppa solver
type DaChoppaSolver <: MathProgBase.AbstractMathProgSolver
    log_level::Int              # Verbosity flag
    timeout::Float64            # Time limit (in seconds)
    rel_gap::Float64            # Relative optimality gap
    zero_tol::Float64           # Zero tolerance for cut coefficients
    feas_tol::Float64           # Absolute feasibility tolerance
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver
end

function DaChoppaSolver(;
    log_level = 1,
    timeout = Inf,
    rel_gap = 1e-5,
    zero_tol = 1e-10,
    feas_tol = 1e-6,
    mip_solver = UnsetSolver(),
    )

    if mip_solver == UnsetSolver()
        error("No MIP solver specified (set mip_solver)\n")
    end

    DaChoppaSolver(log_level, timeout, rel_gap, zero_tol, feas_tol, deepcopy(mip_solver))
end

# Create DaChoppa conic model
MathProgBase.ConicModel(s::DaChoppaSolver) = MathProgBase.ConicModel(ConicNonlinearBridge.ConicNLPWrapper(nlp_solver=s))

# Create DaChoppa LinearQuadratic model
MathProgBase.LinearQuadraticModel(s::DaChoppaSolver) = MathProgBase.NonlinearToLPQPBridge(MathProgBase.NonlinearModel(s))

# Create DaChoppa nonlinear model
MathProgBase.NonlinearModel(s::DaChoppaSolver) = DaChoppaNonlinearModel(s.log_level, s.timeout, s.rel_gap, s.zero_tol, s.feas_tol, s.mip_solver)
