from __future__ import division

import sys
sys.path.insert(0, '/home/dadapy/GeneralBenders/')
from functions.d_bd_functions import run_function_dbd
from scheduling_cost_min_model_maravelias_rho_variable import problem_logic_scheduling,build_scheduling
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import pyomo.environ as pe
import time
import logging
import pickle
import random
import time

#Random seed for master problem initializations (if required)
random.seed(30)

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    kwargs={}
    model_fun =build_scheduling
    logic_fun=problem_logic_scheduling
    m=model_fun(**kwargs)
    ext_ref = {m.Z: m.N} #reformulation sets and variables
    #initialization=[15,11,21,71,1,42,3,10] #optimal large
    # initialization=[2,1,1,2,1,2,1,2] #optimal small
    #initialization=[15,8,23,71,2,45,2,10] 
    initialization=[41,1,1,73,6,49,21,21]
    #initialization=[15,1,1,29,1,31,1,9] 
    infinity_val=1e+6
    nlp_solver='dicopt'
    sub_options={'add_options':[
        'GAMS_MODEL.optfile = 1;'
        '\n'
        '$onecho > dicopt.opt \n'
        '*optcr 1\n'
        '*optca 1000000000\n'
        '$offecho \n']}
    neigh=neighborhood_k_eq_2(len(initialization))
    maxiter=1000
    ## DBD
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    # [important_info,important_info_preprocessing,D,x_actual]=run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    # print('obj= ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
    
     #DSDA
    m_init_fixed = external_ref(m=m,x=initialization,extra_logic_function=logic_fun,dict_extvar=ext_ref,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver='baron', timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)   
    start=time.time()
    D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,{},initialization,ext_ref,logic_fun,k = '2',provide_starting_initialization= True,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 1000,timelimit = 3600,gams_output = False,tee= False,global_tee = True,rel_tol = 1e-3)
    end=time.time()
    print('Objective='+str(pe.value(D_SDAsol.obj))+', best='+str(routeDSDA[-1]))
    print('cputime= '+str(end-start))




    # a_file = open("data_maravelis_d_bd_rho_variable.pkl", "wb")
    # pickle.dump([important_info,important_info_preprocessing,D,x_actual], a_file)
    # a_file.close()

    # 