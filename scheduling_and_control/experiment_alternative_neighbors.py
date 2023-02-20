from __future__ import division
import sys
# sys.path.insert(0, '/home/dadapy/GeneralBenders/')
# sys.path.append('C:/Users/TEMP/Desktop/GeneralBenders/') #for LRLAB5
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_aprox
from functions.dsda_functions import get_external_information,external_ref,external_ref_neighborhood,solve_subproblem,generate_initialization,initialize_model,solve_with_gdpopt,solve_with_minlp
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,solve_subproblem_aprox,generate_initialization,initialize_model,solve_with_dsda,solve_with_dsda_aprox,sequential_iterative_1,sequential_iterative_2,neighborhood_k_eq_l_natural,neighborhood_k_eq_m_natural,neighborhood_k_eq_l_natural_modified,solve_with_dsda_aprox_tau_only
import logging
# from Scheduling_control_variable_tau_model_reduced import scheduling_and_control,problem_logic_scheduling
# from Scheduling_control_variable_tau_model import scheduling_and_control as scheduling_and_control_GDP 
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N as scheduling_and_control_GDP_complete
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx as scheduling_and_control_GDP_complete_approx
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx_only_tau as scheduling_and_control_GDP_complete_approx_tau_only
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx_sequential
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_solvegdp_simpler
from Scheduling_control_variable_tau_model import problem_logic_scheduling, problem_logic_scheduling_tau_only,problem_logic_scheduling_dummy
from Scheduling_control_variable_tau_model import scheduling_only_gdp_N_solvegdp_simpler,scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    nlp_solver='conopt4'
    minlp_solver='dicopt'
    # # SHORT SCHEDULING
    # ext_vars=[4, 4, 6, 6, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2] #Best solution known from sequential iterative, short scheduling obj=-1148
    # ext_vars=[3, 5, 5, 6, 2, 5, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3] #Solution fron infeasible initialization, obj=-1085
    ext_vars=[4, 4, 5, 5, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2] #Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
    # ext_vars=[1, 1, 1, 1, 1, 1, 3, 3, 2, 4, 4, 3, 4, 4, 4, 5] #Scheduling only. Remember to activate scheduling only in solution of subproblem
    sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 20000 \n','infeasder 1','$offecho \n']}
    # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
    start=time.time()
    model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    logic_fun=problem_logic_scheduling
    kwargs={}
    m=model_fun(**kwargs)
    end=time.time()
    # print('model generation time=',str(end-start))
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    start=time.time()
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    end=time.time()
    # print('get info from model time=',str(end-start))
    start=time.time()
    m_fixed = external_ref_neighborhood(m=m,x=ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
    end=time.time()
    # print('ext_Ref_required time=',str(end-start))
        # Transformation step
    start=time.time()
    m =solve_with_minlp(m_fixed,transformation='hull',minlp=minlp_solver,minlp_options=sub_options,timelimit=3600000,gams_output=True,tee=True,rel_tol=0) 
    end=time.time()
    # print('solve subproblem time=',str(end-start))