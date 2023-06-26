from __future__ import division
from pickle import TRUE

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min_ref_2
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_minlp,sequential_iterative_2_case2,sequential_non_iterative_2_case2,sequential_non_iterative_2
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_all,neighborhood_k_eq_l_natural,neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
from functions.d_bd_functions import run_function_dbd_aprox
import logging
from case_study_2_model import case_2_scheduling_control_gdp_var_proc_time,case_2_scheduling_control_gdp_var_proc_time_simplified,problem_logic_scheduling,problem_logic_scheduling_complete,case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential,case_2_scheduling_control_gdp_var_proc_time_min_proc_time,case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation, case_2_scheduling_control_gdp_var_proc_time_min_proc_time_with_distillation,case_2_scheduling_only_lower_bound_tau  
import os
import matplotlib.pyplot as plt
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_GBD,problem_logic_scheduling as problem_logic_scheduling_case1
import numpy as np
from math import fabs
if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


####CASE STUDY 1###############################

    print('******CASE STUDY 1************')


# ###############################################################################
# #########--------------base case ------------------############################
# ###############################################################################
# ###############################################################################

    initialization=[1, 1, 1, 1, 1, 1]
  
    mip_solver='CPLEX'
    minlp_solver='DICOPT'
    nlp_solver='conopt4'
    transform='bigm'


    if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='OCTERACT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    kwargs={}


    print('\n-------GENERALIZED BENDERS TEST-------------------------------------')
    kwargs2=kwargs.copy()

    logic_fun=problem_logic_scheduling_case1
    model_fun=scheduling_and_control_gdp_N_GBD
    m=model_fun(**kwargs2)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)


    ## RUN THIS TO SOLVE

    
    ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_generalized_benders_Test')


    # Sol_found=[]
    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    # print('EXT_VARS_FOUND',Sol_found)
    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJ_FOUND=TPC1+TPC2+TPC3+TMC-SALES

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJECTIVE:',str(OBJ_FOUND))



