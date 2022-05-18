from __future__ import division

import sys
sys.path.insert(0, '/home/dadapy/GeneralBenders/')
from functions.d_bd_functions import run_function_dbd
from scheduling_cost_min_model_maravelias import problem_logic_scheduling,build_scheduling
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model
import pyomo.environ as pe
import time
import logging
import pickle

if __name__ == "__main__":
    #Do not show warnings
    #logging.getLogger('pyomo').setLevel(logging.ERROR)

    kwargs={}
    model_fun =build_scheduling
    logic_fun=problem_logic_scheduling
    m=model_fun(**kwargs)
    ext_ref = {m.Z: m.N} #reformulation sets and variables
    initialization=[1,1,1,1,1,1,1,1] 
    infinity_val=1e+6
    nlp_solver='cplex'
    sub_options={'add_options':[
        'GAMS_MODEL.optfile = 1;'
        '\n'
        '$onecho > cplex.opt \n'
        'intsollim 1  \n'
        '$offecho \n']}
    neigh=neighborhood_k_eq_2(len(initialization))
    maxiter=1000

    [important_info,important_info_preprocessing,D,x_actual]=run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options)
    a_file = open("data_maravelis_d_bd.pkl", "wb")
    pickle.dump(important_info, a_file)
    pickle.dump(important_info_preprocessing, a_file)
    pickle.dump(D, a_file)
    pickle.dump(x_actual, a_file)
    a_file.close()