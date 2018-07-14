#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 DaChoppa solver unit tests
=========================================================#

using JuMP
import ConicBenchmarkUtilities
using DaChoppa
using Base.Test

include("nlptest.jl")
include("conictest.jl")

# test absolute tolerance and DaChoppa printing level
TOL = 1e-3
ll = 2
redirect = true

# use JuMP list of available solvers
include(Pkg.dir("JuMP", "test", "solvers.jl"))

# MIP solvers
tol_int = 1e-9
tol_feas = 1e-7
tol_gap = 0.0

mip_solvers = Dict{String,MathProgBase.AbstractMathProgSolver}()
if glp
    mip_solvers["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=GLPK.MSG_OFF, tol_int=tol_int, tol_bnd=tol_feas, mip_gap=tol_gap)
end
if cpx
    mip_solvers["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas, CPX_PARAM_EPGAP=tol_gap)
end
if grb
    mip_solvers["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas, MIPGap=tol_gap)
end

# print solvers
println("\nMILP solvers:")
for (i, sname) in enumerate(keys(mip_solvers))
    @printf "%2d  %s\n" i sname
end

# run tests
@testset "MILP solver - $mipname" for (mipname, mip) in mip_solvers
    @testset "NLP models" begin
        println("\nNLP models")
        run_qp(mip, ll, redirect)
        run_nlp(mip, ll, redirect)
    end
    @testset "Exp+SOC models" begin
        println("\nExp+SOC models")
        run_soc(mip, ll, redirect)
        run_expsoc(mip, ll, redirect)
    end
    println()
    flush(STDOUT)
    flush(STDERR)
end
println()
