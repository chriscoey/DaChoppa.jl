
#=========================================================
 Nonlinear model object
=========================================================#

type DaChoppaNonlinearModel <: MathProgBase.AbstractNonlinearModel
    log_level::Int              # Verbosity flag
    timeout::Float64            # Time limit (in seconds)
    rel_gap::Float64            # Relative optimality gap
    zero_tol::Float64           # Zero tolerance for cut coefficients
    feas_tol::Float64           # Absolute feasibility tolerance
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver

    sense::Symbol
    numvar::Int
    vartypes::Vector{Symbol}
    warmstart::Vector{Float64}
    numconstr::Int
    geqnlcons::Vector{Int}
    leqnlcons::Vector{Int}
    conlbs::Vector{Float64}
    conubs::Vector{Float64}
    conjac::Vector{Vector{Int}}
    d

    status::Symbol
    oam::Model
    x::Vector{JuMP.Variable}
    incumbent::Vector{Float64}
    objval::Float64
    objbound::Float64
    objgap::Float64
    totaltime::Float64

    function DaChoppaNonlinearModel(log_level, timeout, rel_gap, zero_tol, feas_tol, mip_solver)
        m = new()

        m.log_level = log_level
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.zero_tol = zero_tol
        m.feas_tol = feas_tol
        m.mip_solver = mip_solver

        m.status = :NotLoaded
        m.incumbent = Float64[]
        m.objval = Inf
        m.objbound = -Inf
        m.objgap = Inf
        m.totaltime = 0.0

        return m
    end
end


#=========================================================
 MathProgBase functions
=========================================================#

MathProgBase.numvar(m::DaChoppaNonlinearModel) = m.numvar
MathProgBase.numconstr(m::DaChoppaNonlinearModel) = m.numconstr
MathProgBase.setwarmstart!(m::DaChoppaNonlinearModel, x) = (m.incumbent = x) # TODO set values on mip model, maybe add cuts at point
MathProgBase.setvartype!(m::DaChoppaNonlinearModel, v::Vector{Symbol}) = (m.vartypes = v)

function MathProgBase.loadproblem!(m::DaChoppaNonlinearModel, numvar, numconstr, l, u, lb, ub, sense, d)
    m.numvar = numvar
    m.numconstr = numconstr
    m.sense = sense
    m.incumbent = fill(NaN, numvar)

    MathProgBase.initialize(d, [:Jac])
    m.d = d
    xzero = zeros(m.numvar)

    # Initialize JuMP model for MIP OA
    m.oam = oam = JuMP.Model(solver=m.mip_solver)
    m.x = x = @variable(oam, [j in 1:numvar], lowerbound=l[j], upperbound=u[j])

    # Set objective
    if !MathProgBase.isobjlinear(d)
        # TODO allow quadratic obj, check isobjquadratic and use obj_expr
        error("Objective function must be linear")
    end
    # TODO no constant in obj
    @assert MathProgBase.eval_f(d, xzero) == 0.0
    c = zeros(m.numvar)
    MathProgBase.eval_grad_f(d, c, xzero)
    if sense == :Max
        c .*= -1
    end
    @objective(oam, Min, dot(c, x))

    # Add constraints
    isconlin = fill(false, numconstr)
    numlin = 0
    contolin = fill(-1, numconstr)
    leqnlcons = Int[]
    geqnlcons = Int[]
    for i in 1:numconstr
        if MathProgBase.isconstrlinear(d, i)
            isconlin[i] = true
            numlin += 1
            contolin[i] = numlin
        elseif isfinite(lb[i]) && isfinite(ub[i])
            error("Nonlinear equality/two-sided constraints not accepted")
        elseif isfinite(lb[i])
            push!(geqnlcons, i)
        else
            @assert isfinite(ub[i])
            push!(leqnlcons, i)
        end
    end
    m.geqnlcons = geqnlcons
    m.leqnlcons = leqnlcons
    m.conlbs = lb
    m.conubs = ub

    (jac_I, jac_J) = MathProgBase.jac_structure(d)
    jac_V = zeros(length(jac_I))
    MathProgBase.eval_jac_g(d, jac_V, xzero)
    A_lin_I = Int[]
    A_lin_J = Int[]
    A_lin_V = Float64[]
    conjac = [Int[] for i in 1:numconstr]
    for k in 1:length(jac_I)
        i = jac_I[k]
        if isconlin[i]
            push!(A_lin_I, contolin[i])
            push!(A_lin_J, jac_J[k])
            push!(A_lin_V, jac_V[k])
        else
            push!(conjac[i], k)
        end
    end
    m.conjac = conjac
    A_lin = sparse(A_lin_I, A_lin_J, A_lin_V, numlin, numvar)
    convals = zeros(numconstr)
    MathProgBase.eval_g(d, convals, xzero)
    b_lin = convals[isconlin] - A_lin*xzero
    lb_lin = lb[isconlin]
    ub_lin = ub[isconlin]

    @constraint(oam, lb_lin .<= A_lin*x + b_lin .<= ub_lin)

    m.status = :Loaded
end

function MathProgBase.optimize!(m::DaChoppaNonlinearModel)
    totaltime = time()
    m.status = :Unsolved

    for j in 1:m.numvar
        setcategory(m.x[j], m.vartypes[j])
    end

    conoa = zeros(m.numconstr)
    (jac_I, jac_J) = MathProgBase.jac_structure(m.d)
    jac_V = zeros(length(jac_I))
    num_cuts = 0
    num_cbs = 0

    function callback_lazy(cb)
        num_cbs += 1
        xoa = getvalue(m.x)
        MathProgBase.eval_g(m.d, conoa, xoa)
        MathProgBase.eval_jac_g(m.d, jac_V, xoa)
        # @show xoa
        # @show conoa
        # @show jac_V

        # Add gradient cuts on nonlinear constraints
        for i in m.geqnlcons
            if m.conlbs[i] - conoa[i] > m.feas_tol
                # @show m.conlbs[i] - conoa[i]
                # @show sum(jac_V[k]*(m.x[jac_J[k]] - xoa[jac_J[k]]) for k in m.conjac[i])
                @lazyconstraint(cb, m.conlbs[i] <= conoa[i] + sum(jac_V[k]*(m.x[jac_J[k]] - xoa[jac_J[k]]) for k in m.conjac[i]))
                num_cuts += 1
            end
        end
        for i in m.leqnlcons
            if conoa[i] - m.conubs[i] > m.feas_tol
                # @show conoa[i] - m.conubs[i]
                # @show sum(jac_V[k]*(m.x[jac_J[k]] - xoa[jac_J[k]]) for k in m.conjac[i])
                @lazyconstraint(cb, conoa[i] + sum(jac_V[k]*(m.x[jac_J[k]] - xoa[jac_J[k]]) for k in m.conjac[i]) <= m.conubs[i])
                num_cuts += 1
            end
        end
    end
    addlazycallback(m.oam, callback_lazy)

    # Start MIP solver
    if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
        MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(0.1, m.timeout - (time() - totaltime)))
        setsolver(m.oam, m.mip_solver)
    end
    if m.log_level > 0
        println("\nDaChoppa solver starting...")
    end
    mip_time = time()
    status_mip = solve(m.oam, suppress_warnings=true)
    mip_time = time() - mip_time

    if status_mip in (:Optimal, :Suboptimal)
        m.incumbent = getvalue(m.x)
        m.objval = getobjectivevalue(m.oam)
        m.objbound = MathProgBase.getobjbound(m.oam)
        @assert isfinite(m.objval) && isfinite(m.objbound)
        m.objgap = (m.objval - m.objbound)/(abs(m.objval) + 1e-5)
        m.status = status_mip
    elseif status_mip == :UserLimit
        m.objval = getobjectivevalue(m.oam)
        m.objbound = MathProgBase.getobjbound(m.oam)
        if isfinite(m.objval)
            m.incumbent = getvalue(m.x)
            if isfinite(m.objbound)
                m.objgap = (m.objval - m.objbound)/(abs(m.objval) + 1e-5)
            end
        end
        m.status = status_mip
    elseif status_mip == :Infeasible
        m.status = status_mip
    elseif status_mip in (:Unbounded, :InfeasibleOrUnbounded)
        warn("MIP solver status was $status_mip (try using initial outer approximation cuts)")
        m.status = :UnboundedRelaxation
    else
        warn("MIP solver status was $status_mip")
        m.status = :MIPFailure
    end

    m.totaltime = time() - totaltime
    if m.log_level > 0
        println("\nDaChoppa solver finished...")
        @printf "Status           %13s\n" m.status
        if isfinite(m.objgap)
            @printf "Objective value  %13.5f\n" (m.sense == :Min) ? m.objval : -m.objval
            @printf "Objective bound  %13.5f\n" (m.sense == :Min) ? m.objbound : -m.objbound
            @printf "Objective gap    %13.5f\n" (m.sense == :Min) ? m.objgap : -m.objgap
        end
        @printf "Total time       %13.5f s\n" m.totaltime
        @printf "MIP time         %13.5f s\n" mip_time
        @printf "Cuts added       %13d\n" num_cuts
        @printf "Callbacks        %13d\n" num_cbs
        println()
    end
end

MathProgBase.status(m::DaChoppaNonlinearModel) = m.status
MathProgBase.getobjval(m::DaChoppaNonlinearModel) = (m.sense == :Min) ? m.objval : -m.objval
MathProgBase.getobjbound(m::DaChoppaNonlinearModel) = (m.sense == :Min) ? m.objbound : -m.objbound
MathProgBase.getobjgap(m::DaChoppaNonlinearModel) = (m.sense == :Min) ? m.objgap : -m.objgap
MathProgBase.getsolution(m::DaChoppaNonlinearModel) = m.incumbent
MathProgBase.getsolvetime(m::DaChoppaNonlinearModel) = m.totaltime
MathProgBase.getnodecount(m::DaChoppaNonlinearModel) = MathProgBase.getnodecount(m.oam)
