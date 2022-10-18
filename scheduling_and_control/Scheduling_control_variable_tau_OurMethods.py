from __future__ import division
import sys
sys.path.insert(0, '/home/dadapy/GeneralBenders/')
from functions.d_bd_functions import run_function_dbd
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_gdpopt,solve_with_minlp
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
from Scheduling_control_variable_tau_model_reduced import scheduling_and_control,problem_logic_scheduling
from Scheduling_control_variable_tau_model import scheduling_and_control as scheduling_and_control_GDP 



if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    #Solver declaration
    minlp_solver='dicopt'
    nlp_solver='conopt4'
    mip_solver='cplex'
    gdp_solver='LBB'
    if minlp_solver=='dicopt':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','$offecho \n']}
    else:
        sub_options={}

    # #Solve with LD-SDA
    # model_fun =scheduling_and_control
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[2,2,2,2,2,2],ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 1000,timelimit = 3600,gams_output = False,tee= False,global_tee = True,rel_tol = 1e-3)


    #Solve with pyomo.GDP
    kwargs={}
    model_fun=scheduling_and_control_GDP 
    m=model_fun(**kwargs)
    m_solved = solve_with_gdpopt(m, mip=mip_solver,minlp=minlp_solver,nlp=nlp_solver,minlp_options=sub_options, timelimit=1000,strategy=gdp_solver, mip_output=False, nlp_output=False,rel_tol=0,tee=True)