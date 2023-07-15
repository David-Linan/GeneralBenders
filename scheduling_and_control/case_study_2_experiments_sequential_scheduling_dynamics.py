from __future__ import division
from pickle import TRUE

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min_ref_2
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,solve_subproblem_aprox_fix_all_scheduling,generate_initialization,initialize_model,solve_with_minlp,sequential_iterative_2_case2,sequential_non_iterative_2_case2,sequential_non_iterative_2
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_all,neighborhood_k_eq_l_natural,neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda,sequential_iterative_2
from functions.d_bd_functions import run_function_dbd_aprox
import logging
from case_study_2_model import case_2_scheduling_control_gdp_var_proc_time,case_2_scheduling_control_gdp_var_proc_time_simplified,problem_logic_scheduling,problem_logic_scheduling_complete,case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential,case_2_scheduling_control_gdp_var_proc_time_min_proc_time,case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation, case_2_scheduling_control_gdp_var_proc_time_min_proc_time_with_distillation,case_2_scheduling_only_lower_bound_tau  
import os
import matplotlib.pyplot as plt
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_solvegdp_simpler,scheduling_and_control_gdp_N_approx_sequential_naive,problem_logic_scheduling as problem_logic_scheduling_case1
from Scheduling_control_variable_tau_model import scheduling_only_gdp_N_solvegdp_simpler,scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx_sequential
from Scheduling_control_variable_tau_model import  problem_logic_scheduling_tau_only,problem_logic_scheduling_dummy
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N as scheduling_and_control_GDP_complete
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx as scheduling_and_control_GDP_complete_approx
import numpy as np
from math import fabs
if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


####CASE STUDY 1###############################

    print('******CASE STUDY 1: CHU AND YOU, SHORT SCHEDULING HORIZON (14 h)************')


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
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
        print('DICOPT options:',sub_options)
    elif minlp_solver=='OCTERACT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    kwargs={}


# ###############################################################################
# #########--------------sequential naive-------------###########################
# ###############################################################################
# ###############################################################################

    # print('\n-------SEQUENTIAL NAIVE-------------------------------------')
    # kwargs2=kwargs.copy()
    # kwargs2['sequential']=True

    # logic_fun=problem_logic_scheduling_case1
    # model_fun=scheduling_and_control_gdp_N_approx_sequential_naive
    # m=model_fun(**kwargs2)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # initialization_test=[]
    # for k in upper_bounds.keys():
    #     initialization_test.append(upper_bounds[k]) 
    # print('upper bound ext var related to proc time',initialization_test)
    # ## RUN THIS TO SOLVE
    # m=sequential_non_iterative_2(logic_fun,initialization_test,model_fun,kwargs2,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = False, global_tee= True,rel_tol = 0)
    # #SAVE SOLUTION
    # save=generate_initialization(m=m,model_name='case_1_scheduling_and_dynamics_solution')
    # ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_scheduling_and_dynamics_solution')
    # # m.varTime.pprint()

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

###############################################################################
#########--------------sequential iterative-------------#######################
###########--------+ DSDA ---------------------------##########################
###############################################################################
    print('\n-------SEQUENTIAL ITERATIVE-------------------------------------')
    # STEP 1: SCHEDULING ONLY WITH VARIABLE PROCESSSING TIMES
    # kwargs3=kwargs.copy()
    # kwargs3['x_initial']=[1,1,1,1,1,1]
    # model_fun=scheduling_only_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # m = solve_with_minlp(m,transformation=transform,minlp=mip_solver,minlp_options=sub_options,timelimit=3600000,gams_output=False,tee=True,rel_tol=0)
    # save=generate_initialization(m=m,model_name='case_1_scheduling_only_solution')
    # # SCHEDULING INITIALIZATION
    # Sol_found=[]
    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)

    # STEP 2: SEQUENTIAL STRATEGY TO IDENTIFY FEASIBILITY
    # model_fun =scheduling_and_control_gdp_N_approx_sequential
    # logic_fun=problem_logic_scheduling_dummy
    # m=model_fun()
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,Sol_found=sequential_iterative_2(logic_fun,Sol_found,model_fun,kwargs,ext_ref,rate_tau=1,provide_starting_initialization = False,subproblem_solver=nlp_solver,iter_timelimit = 1000000,subproblem_solver_options=sub_options,gams_output = False,tee = False,global_tee = True,rel_tol = 0)
       

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


    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # logic_fun=problem_logic_scheduling_case1
    # m_partial=model_fun(**kwargs)
    # def _obj_scheduling(m_partial):
    #     return ( m_partial.TCP1+m_partial.TCP2+m_partial.TMC-m_partial.SALES  )/100
    # m_partial.obj_scheduling = pe.Objective(rule=_obj_scheduling, sense=pe.minimize)  
    
    # def _obj_dummy(m_partial):
    #     return 1
    # m_partial.obj_dummy = pe.Objective(rule=_obj_dummy, sense=pe.minimize)  

    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    # m_partial = external_ref(m=m_partial,x=Sol_found,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)

     
    # m_partial=solve_subproblem_aprox_fix_all_scheduling(m_partial,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = True,rel_tol = 0)   
    # save=generate_initialization(m=m_partial,model_name='case_1_scheduling_and_dynamics_solution_seq_iterative')


    # Sol_found_seq_naive=Sol_found
    # print('\n-------DSDA-------------------------------------')

    ### NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # # STEP 3: DSDA
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # logic_fun=problem_logic_scheduling_case1
    # m=model_fun(**kwargs)   
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,Sol_found_seq_naive,ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))
    # save=generate_initialization(m=m,model_name='case_1_scheduling_and_dynamics_solution_DSDA_naive')

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


# ###############################################################################
# #########--------------dicopt ----------------#################################
# ###############################################################################
# ###############################################################################

    print('\n-------DICOPT-------------------------------------')
    init_name='case_1_scheduling_and_dynamics_solution_seq_iterative'

    model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    m=model_fun(**kwargs)
    if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
        # NOTE: we have to modify options slighlty to guarantee that DICOPT starts from user provided initialization!!!!!!! If we remove this, DICOPT WILL NEVER FIND A FEASIBLE SOLUTION!!!
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','relaxed 0 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}


    m=initialize_model(m=m,from_feasible=True,feasible_model=init_name) 
    start=time.time()
    m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    end=time.time()    
    solname='case_1_minlp_'+minlp_solver+'_from_'+init_name
    save=generate_initialization(m=m,model_name=solname)

    if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
        m.dicopt_status='Infeasible'
    else:
        m.dicopt_status='Optimal'

    if m.dicopt_status=='Optimal':
        Sol_founddicopt=[]
        for I in m.I_reactions:
            for J in m.J_reactors:
                if m.I_i_j_prod[I,J]==1:
                    for K in m.ordered_set[I,J]:
                        if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                            Sol_founddicopt.append(K-m.minTau[I,J]+1)
        for I_J in m.I_J:
            Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


        print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    else:
        print('DICOPT infeasible','cputime DICOPT=',str(end-start))

    TPC1=pe.value(m.TCP1)
    TPC2=pe.value(m.TCP2)
    TPC3=pe.value(m.TCP3)
    TMC=pe.value(m.TMC)
    SALES=pe.value(m.SALES)
    OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJ:',str(OBJVAL))

# NOTE: RESULTS ABOVE ARE FOR FIRST ARTICLE. RESULTS BELOW WOULD BE FOR SECOND ARTICLE
# NOTE: results related to benders decomposition are obtained by running benders decomposition file V2.

###############################################################################
#########--------------LD-BD METHODOLOGIES--------------#######################
###########------------FROM FEASIBLE      -----------##########################
##############################################################################

    # print('\n-------MULTICUT LDBD NAIVE-------------------------------------')
    # # #NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # initialization=Sol_found_seq_naive 
    # infinity_val=1e+4
    # maxiter=1000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # logic_fun=problem_logic_scheduling_case1
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
    # save=generate_initialization(m=m,model_name='case_1_scheduling_and_dynamics_solution_DBD_multicut_naive')
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

    # print('\n-------MULTICUT LDBD PRUNING RIGUROUS SUBPROBLEMS-------------------------------------')
    # # # #NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # Sol_found=Sol_found_seq_naive # from sequential iterative
    # feas_model='case_1_scheduling_and_dynamics_solution' # from sequential iterative
    # initialization=Sol_found
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # neigh=neighborhood_k_eq_2(len(Sol_found))
    # logic_fun=problem_logic_scheduling_case1
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
    # kwargs['prunning']=True
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()                                                                                                                                                              ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=False,with_distillation=False,provide_starting_initialization=True,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_1_scheduling_and_dynamics_solution_DBD_pruning_rigurous_subpr_'+minlp_solver+'_'+neighdef
    # save=generate_initialization(m=m,model_name=solname) 
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
    # kwargs['prunning']=False

    # print('\n-------MULTICUT LDBD PRUNING APROX SUBPROBLEMS-------------------------------------')
    # # #NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # Sol_found_seq_naive=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # Sol_found=Sol_found_seq_naive # from sequential iterative
    # feas_model='case_1_scheduling_and_dynamics_solution' # from sequential iterative
    # kwargs['prunning']=True
    # initialization=Sol_found
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # neigh=neighborhood_k_eq_2(len(Sol_found))
    # logic_fun=problem_logic_scheduling_case1
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau

    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()                                                                                                                                                              ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=False,with_distillation=False,provide_starting_initialization=True,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_1_scheduling_and_dynamics_solution_DBD_pruning_aprox_subpr_'+minlp_solver+'_'+neighdef+'test'
    # save=generate_initialization(m=m,model_name=solname) 
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
    # kwargs['prunning']=False
    # print(sub_options)

###############################################################################
#########--------------LD-BD METHODOLOGIES--------------#######################
###########------------FROM INFEASIBLE  vs DICOPT    -----------###############
###############################################################################

    # print('\n-------MULTICUT LDBD PRUNING APROX SUBPROBLEMS FROM INFEASIBLE-------------------------------------')

    # feas_model='' 
    # kwargs['prunning']=True

    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'

    # logic_fun=problem_logic_scheduling_case1
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau

    # m_init=model_fun(**kwargs)

    # m_init=initialize_model(m_init,from_feasible=True,feasible_model='case_1_scheduling_only_solution')
    # Sol_found=[]
    # for I in m_init.I_reactions:
    #     for J in m_init.J_reactors:
    #         if m_init.I_i_j_prod[I,J]==1:
    #             for K in m_init.ordered_set[I,J]:
    #                 if round(pe.value(m_init.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m_init.minTau[I,J]+1)
    # for I_J in m_init.I_J:
    #     Sol_found.append(1+round(pe.value(m_init.Nref[I_J])))
    # print('EXT_VARS_INFEASIBLE_INIT',Sol_found)


    # neigh=neighborhood_k_eq_2(len(Sol_found))
    # initialization=Sol_found

    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()                                                                                                                                                              ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=False,with_distillation=False,provide_starting_initialization=False,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_1_scheduling_and_dynamics_solution_DBD_pruning_aprox_subpr_from_infeasible_'+minlp_solver+'_'+neighdef
    # save=generate_initialization(m=m,model_name=solname) 
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
    # kwargs['prunning']=False

    # print('\n-------DICOPT FROM INFEASIBLE-------------------------------------')
    # model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
    #     # NOTE: we cannot use relaxed 0 here to guaranteee feasibility of first NLP, which in this case is RMINLP
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_scheduling_only_solution') 

    # start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()    
    # solname='case_1_minlp_from_infeasible_'+minlp_solver
    # save=generate_initialization(m=m,model_name=solname)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     m.dicopt_status='Infeasible'
    # else:
    #     m.dicopt_status='Optimal'

    # if m.dicopt_status=='Optimal':
    #     Sol_founddicopt=[]
    #     for I in m.I_reactions:
    #         for J in m.J_reactors:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_founddicopt.append(K-m.minTau[I,J]+1)
    #     for I_J in m.I_J:
    #         Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    # else:
    #     print('DICOPT infeasible','cputime DICOPT=',str(end-start))

    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJ:',str(OBJVAL))


####CASE STUDY 1.1###############################

    print('******CASE STUDY 1: CHU AND YOU, LONG SCHEDULING HORIZON (28 h)************')

# ###############################################################################
# #########--------------base case ------------------############################
# ###############################################################################
# ###############################################################################

    # initialization=[1, 1, 1, 1, 1, 1]
  
    # mip_solver='CPLEX'
    # minlp_solver='DICOPT'
    # nlp_solver='conopt4'
    # transform='bigm'


    # if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    #     print('DICOPT options:',sub_options)
    # elif minlp_solver=='OCTERACT':
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    # kwargs={'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    # rel_tol=0.05


# ###############################################################################
# #########--------------sequential naive-------------###########################
# ###############################################################################
# ###############################################################################

    # print('\n-------SEQUENTIAL NAIVE-------------------------------------')
    # kwargs2=kwargs.copy()
    # kwargs2['sequential']=True

    # logic_fun=problem_logic_scheduling_case1
    # model_fun=scheduling_and_control_gdp_N_approx_sequential_naive
    # m=model_fun(**kwargs2)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # initialization_test=[]
    # for k in upper_bounds.keys():
    #     initialization_test.append(upper_bounds[k]) 
    # print('upper bound ext var related to proc time',initialization_test)
    # ## RUN THIS TO SOLVE
    # m=sequential_non_iterative_2(logic_fun,initialization_test,model_fun,kwargs2,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = False, global_tee= True,rel_tol = rel_tol)
    # #SAVE SOLUTION
    # save=generate_initialization(m=m,model_name='case_1_28h_scheduling_and_dynamics_solution')
    # ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_28h_scheduling_and_dynamics_solution')
    # # m.varTime.pprint()

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

###############################################################################
#########--------------sequential iterative-------------#######################
###########--------+ DSDA ---------------------------##########################
###############################################################################
    print('\n-------SEQUENTIAL ITERATIVE-------------------------------------')
    # # STEP 1: SCHEDULING ONLY WITH VARIABLE PROCESSSING TIMES
    # kwargs3=kwargs.copy()
    # kwargs3['x_initial']=[1,1,1,1,1,1]
    # model_fun=scheduling_only_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # m = solve_with_minlp(m,transformation=transform,minlp=mip_solver,minlp_options=sub_options,timelimit=3600000,gams_output=False,tee=True,rel_tol=rel_tol)
    # save=generate_initialization(m=m,model_name='case_1_28h_scheduling_only_solution')
    # # SCHEDULING INITIALIZATION
    # Sol_found=[]
    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)

    # # STEP 2: SEQUENTIAL STRATEGY TO IDENTIFY FEASIBILITY
    # model_fun =scheduling_and_control_gdp_N_approx_sequential
    # logic_fun=problem_logic_scheduling_dummy
    # m=model_fun()
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,Sol_found=sequential_iterative_2(logic_fun,Sol_found,model_fun,kwargs,ext_ref,rate_tau=1,provide_starting_initialization = False,subproblem_solver=nlp_solver,iter_timelimit = 1000000,subproblem_solver_options=sub_options,gams_output = False,tee = False,global_tee = True,rel_tol = rel_tol)
    # save=generate_initialization(m=m,model_name='case_1_28h_scheduling_and_dynamics_solution_seq_iterative')

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


    # Sol_found_seq_naive=Sol_found
    print('\n-------DSDA-------------------------------------')

    # ## NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=
    # # STEP 3: DSDA
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # logic_fun=problem_logic_scheduling_case1
    # m=model_fun(**kwargs)   
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,Sol_found_seq_naive,ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 100000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))
    # save=generate_initialization(m=m,model_name='case_1_28h_scheduling_and_dynamics_solution_DSDA_naive')

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


# ###############################################################################
# #########--------------dicopt ----------------#################################
# ###############################################################################
# ###############################################################################

    # print('\n-------DICOPT-------------------------------------')
    # model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
    #     # NOTE: we have to modify options slighlty to guarantee that DICOPT starts from user provided initialization!!!!!!! If we remove this, DICOPT WILL NEVER FIND A FEASIBLE SOLUTION!!!
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','relaxed 0 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_28h_scheduling_and_dynamics_solution') 

    # start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=100000,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()    
    # solname='case_1_28h_minlp_'+minlp_solver
    # save=generate_initialization(m=m,model_name=solname)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     m.dicopt_status='Infeasible'
    # else:
    #     m.dicopt_status='Optimal'

    # if m.dicopt_status=='Optimal':
    #     Sol_founddicopt=[]
    #     for I in m.I_reactions:
    #         for J in m.J_reactors:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_founddicopt.append(K-m.minTau[I,J]+1)
    #     for I_J in m.I_J:
    #         Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    # else:
    #     print('DICOPT infeasible','cputime DICOPT=',str(end-start))

    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJ:',str(OBJVAL))

###############################################################################
#########--------------LD-BD METHODOLOGIES--------------#######################
###########------------FROM FEASIBLE      -----------##########################
###############################################################################

    print('\n-------MULTICUT LDBD NAIVE-------------------------------------')
    # # #NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=
    # # Sol_found_seq_naive=
    # initialization=Sol_found_seq_naive 
    # infinity_val=1e+4
    # maxiter=1000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # logic_fun=problem_logic_scheduling_case1
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
    # save=generate_initialization(m=m,model_name='case_1_28h_scheduling_and_dynamics_solution_DBD_multicut_naive')
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

    print('\n-------MULTICUT LDBD PRUNING RIGUROUS SUBPROBLEMS-------------------------------------')
    # # #NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=
    # # Sol_found_seq_naive=
    # Sol_found=Sol_found_seq_naive # from sequential iterative
    # feas_model='case_1_28h_scheduling_and_dynamics_solution' # from sequential iterative
    # initialization=Sol_found
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # neigh=neighborhood_k_eq_2(len(Sol_found))
    # logic_fun=problem_logic_scheduling_case1
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
    # kwargs['prunning']=True
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()                                                                                                                                                              ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=False,with_distillation=False,provide_starting_initialization=True,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_1_28h_scheduling_and_dynamics_solution_DBD_pruning_rigurous_subpr_'+minlp_solver+'_'+neighdef
    # save=generate_initialization(m=m,model_name=solname) 
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
    # kwargs['prunning']=False

    # print('\n-------MULTICUT LDBD PRUNING APROX SUBPROBLEMS-------------------------------------')
    # # #NOTE: IN CASE I DO NOT WANT TO RUN PREVIOUS CODE: Sol_found_seq_naive=
    # Sol_found_seq_naive=[4, 4, 5, 5, 3, 3, 4, 3, 3, 5, 5, 5, 4, 6, 5, 7] #NOTE: this is sequential naive!!!!!--still have to run seq iterative!!!!
    # Sol_found=Sol_found_seq_naive # from sequential iterative
    # feas_model='case_1_28h_scheduling_and_dynamics_solution' # from sequential iterative
    # kwargs['prunning']=True
    # initialization=Sol_found
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # neigh=neighborhood_k_eq_2(len(Sol_found))
    # logic_fun=problem_logic_scheduling_case1
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau

    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()                                                                                                                                                              ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=rel_tol,new_case=False,with_distillation=False,provide_starting_initialization=True,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_1_28h_scheduling_and_dynamics_solution_DBD_pruning_aprox_subpr_'+minlp_solver+'_'+neighdef+'test'
    # save=generate_initialization(m=m,model_name=solname) 
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
    # kwargs['prunning']=False

###############################################################################
#########--------------LD-BD METHODOLOGIES--------------#######################
###########------------FROM INFEASIBLE  vs DICOPT    -----------###############
###############################################################################

    # print('\n-------MULTICUT LDBD PRUNING APROX SUBPROBLEMS FROM INFEASIBLE-------------------------------------')

    # feas_model='' 
    # kwargs['prunning']=True

    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'

    # logic_fun=problem_logic_scheduling_case1
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau

    # m_init=model_fun(**kwargs)

    # m_init=initialize_model(m_init,from_feasible=True,feasible_model='case_1_28h_scheduling_only_solution')
    # Sol_found=[]
    # for I in m_init.I_reactions:
    #     for J in m_init.J_reactors:
    #         if m_init.I_i_j_prod[I,J]==1:
    #             for K in m_init.ordered_set[I,J]:
    #                 if round(pe.value(m_init.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m_init.minTau[I,J]+1)
    # for I_J in m_init.I_J:
    #     Sol_found.append(1+round(pe.value(m_init.Nref[I_J])))
    # print('EXT_VARS_INFEASIBLE_INIT',Sol_found)


    # neigh=neighborhood_k_eq_2(len(Sol_found))
    # initialization=Sol_found

    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()                                                                                                                                                              ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=rel_tol,new_case=False,with_distillation=False,provide_starting_initialization=False,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_1_28h_scheduling_and_dynamics_solution_DBD_pruning_aprox_subpr_from_infeasible_'+minlp_solver+'_'+neighdef
    # save=generate_initialization(m=m,model_name=solname) 
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
    # kwargs['prunning']=False

    # print('\n-------DICOPT FROM INFEASIBLE-------------------------------------')
    # model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
    #     # NOTE: we cannot use relaxed 0 here to guaranteee feasibility of first NLP, which in this case is RMINLP
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_28h_scheduling_only_solution') 

    # start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=100000,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()    
    # solname='case_1_28h_minlp_from_infeasible_'+minlp_solver
    # save=generate_initialization(m=m,model_name=solname)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     m.dicopt_status='Infeasible'
    # else:
    #     m.dicopt_status='Optimal'

    # if m.dicopt_status=='Optimal':
    #     Sol_founddicopt=[]
    #     for I in m.I_reactions:
    #         for J in m.J_reactors:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_founddicopt.append(K-m.minTau[I,J]+1)
    #     for I_J in m.I_J:
    #         Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    # else:
    #     print('DICOPT infeasible','cputime DICOPT=',str(end-start))

    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJ:',str(OBJVAL))


####CASE STUDY 2###############################

    print('******CASE STUDY 2: SEMIBATCH, without distillation column************')

# ###############################################################################
# #########--------------base case ------------------############################
# ###############################################################################
# ###############################################################################

    obj_Selected='profit_max'



    initialization=[1, 1, 1, 1, 1, 1, 1, 1]
  
    mip_solver='CPLEX'
    minlp_solver='DICOPT'
    nlp_solver='conopt4'
    transform='bigm'
    #tried 5 and no improvement. With 15 DICOT is unable, and now DSDA can solve the problem.
    last_disc=15
    last_time_h=5

    if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
        print('DICOPT options:',sub_options)
    elif minlp_solver=='OCTERACT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME,'sequential':False}




# ###############################################################################
# #########--------------sequential naive-------------###########################
# ###############################################################################
# ###############################################################################
    # initialization_test=[1, 6, 6, 1, 1, 1, 1, 1]
    # print('\n-------SEQUENTIAL NAIVE-------------------------------------')
    # kwargs2=kwargs.copy()
    # kwargs2['sequential']=True

    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_min_proc_time
    # m=model_fun(**kwargs2)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)



    # # ## RUN THIS TO SOLVE
    # # m=sequential_non_iterative_2_case2(logic_fun,initialization_test,model_fun,kwargs2,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = True, global_tee= True,rel_tol = 0)
    # # ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_and_dynamics_solution')
    # # NOTE: This print is only for the case where B is variable in scheduling
    # # print('Minimum product composition that becomes infeasible',str(pe.value(m.CC['T2','U3',5][m.N['T2','U3',5].last()])))


    # Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    # print(Sol_found)
    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJ_VAL=TPC1+TPC2+TPC3+TMC-SALES

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJECTIVE: ',str(OBJ_VAL))
# ###############################################################################
# #########--------------sequential ------------------###########################
# ###############################################################################
# ###############################################################################
    # print('\n-------SEQUENTIAL-------------------------------------')
    # kwargs['sequential']=True

    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # m,_=sequential_iterative_2_case2(logic_fun,initialization,model_fun,kwargs,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = False, global_tee= True,rel_tol = 0)
    # save=generate_initialization(m=m,model_name='case_2_sequential')
    # Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)
    # # for I_J in m.I_J:
    # #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))

# ###############################################################################
# #########--------------dicopt ----------------#################################
# ###############################################################################
# ###############################################################################
    # print('\n-------DICOPT-------------------------------------')
    # kwargs['sequential']=False
    # # kwargs['x_initial']=Sol_found
    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential
    # if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
    #     # NOTE: we have to modify options slighlty to guarantee that DICOPT starts from user provided initialization!!!!!!! If we remove this, DICOPT WILL NEVER FIND A FEASIBLE SOLUTION!!!
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 3 \n','relaxed 0 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # m=model_fun(**kwargs)
    # feas_mol_name='case_2_sequential'
    # # m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_and_dynamics_solution')
    # m=initialize_model(m,from_feasible=True,feasible_model=feas_mol_name) 
    # m.pprint()
    # start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()    
    # solname='case_2_opt_'+minlp_solver
    # save=generate_initialization(m=m,model_name=solname)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     m.dicopt_status='Infeasible'
    # else:
    #     m.dicopt_status='Optimal'

    # if m.dicopt_status=='Optimal':
    #     Sol_founddicopt=[]
    #     for I in m.I:
    #         for J in m.J:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_founddicopt.append(K-m.minTau[I,J]+1)
    #     # for I_J in m.I_J:
    #     #     Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    # else:
    #     print('DICOPT infeasible')

    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJ:',str(OBJVAL))
###############################################################################
#########--------------dsda ------------------#################################
###############################################################################
###############################################################################
    # print('\n-------DSDA-------------------------------------')
    # kwargs['sequential']=False
    # kwargs['x_initial']=Sol_found
    # initialization=Sol_found
    # infinity_val=1e+2
    # maxiter=10000
    # neighdef='2'
    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential
    # start=time.time()
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,initialization,ext_ref,logic_fun,k = neighdef,provide_starting_initialization= True,feasible_model='case_2_sequential',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 1000,timelimit = 86400,gams_output = False,tee= False,global_tee = True,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=False)
    # end=time.time()
    # m=D_SDAsol
    # solname='case_2_dsda_'+minlp_solver+'_'+neighdef+'_all_neigh_Verified'
    # save=generate_initialization(m=m,model_name=solname) 
    

    # if m.dsda_status=='optimal':
    #     print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  
    #     TPC1=pe.value(D_SDAsol.TCP1)
    #     TPC2=pe.value(D_SDAsol.TCP2)
    #     TPC3=pe.value(D_SDAsol.TCP3)
    #     TMC=pe.value(D_SDAsol.TMC)
    #     SALES=pe.value(D_SDAsol.SALES)
    #     OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    #     print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    #     print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    #     print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    #     print('TMC: Total material cost: ',str(TMC))
    #     print('SALES: Revenue form selling products: ',str(SALES))
    #     print('OBJ:',str(OBJVAL))

###############################################################################
#########--------------dbd-approx_sol_subproblems ------------------###########
###############################################################################
###############################################################################

    # print('\n-------DBD-approx solution of subproblems-------------------------------------')
    # Sol_found=[1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 5, 4, 1, 2, 1, 2] # from sequential iterative
    # feas_model='case_2_sequential' # from sequential iterative
    # kwargs['sequential']=True
    # kwargs['x_initial']=Sol_found
    # initialization=Sol_found
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # neigh=neighborhood_k_eq_2(len(Sol_found))



    # logic_fun=problem_logic_scheduling_complete
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential


    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # start=time.time()
    # #TODO: IF I WANTED TO PERFORM A FAIR COMPARISON BETWEEN THIS ONE AND THE PREVIOUS DSDA RUN, I SHOULD IMPOSE A TIME LIMIT IN THE SOLUTION OF SUBPROBLEMS!!!!1                                                                                                                                                                ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=True,with_distillation=False,provide_starting_initialization=True,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_2_dbd_aprox_subproblems_'+minlp_solver+'_'+neighdef+'_all_neigh_Verified'
    # save=generate_initialization(m=m,model_name=solname) 
    # new_Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     new_Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     new_Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    # print(new_Sol_found)
    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES)) 
 
# #######-------plots------------------------
#     for I in m.I_dynamics:
#         for J in m.J_dynamics:
#             for T in m.T:
#                 if pe.value(m.X[I,J,T])==1: 
#                     case=(I,J,T)
#                     t=[]
#                     CA=[]
#                     CB=[]
#                     CC=[]
#                     Tr=[]
#                     Tj=[]
#                     Fhot=[]
#                     Fcold=[]
#                     u_input=[]
#                     for N in m.N[case]:
#                         t.append(N*m.varTime[case].value)
#                         Tr.append(m.TRvar[case][N].value)
#                         Tj.append(m.TJvar[case][N].value)
#                         Fhot.append(m.Fhot[case][N].value)
#                         Fcold.append(m.Fcold[case][N].value)
#                         CA.append( m.CA[case][N].value)
#                         CB.append( m.CB[case][N].value)
#                         CC.append( m.CC[case][N].value)
#                         u_input.append(m.u_input[case][N].value)
                        
                        
#                     plt.plot(t, CA,label='CA',color='red')
#                     plt.plot(t, CB,label='CB',color='green')
#                     plt.plot(t, CC,label='CC',color='blue')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('$Concentration [kmol/m^{3}]$')
#                     title=case[0]+' in '+case[1]+' Concentration'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.legend()
#                     plt.show()
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()

#                     plt.plot(t,Tr,label='T_reactor',color='red')
#                     plt.plot(t,Tj,label='T_jacket',color='blue')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('Temperature [K]')
#                     title=case[0]+' in '+case[1]+' Temperature'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.legend()
#                     plt.show()
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()
                    
#                     plt.plot(t, Fhot,label='F_hot',color='red')
#                     plt.plot(t,Fcold,label='F_cold',color='blue')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('Flow rate $[m^{3}/h]$')
#                     title=case[0]+' in '+case[1]+' Flow rate'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.legend()
#                     plt.show()    
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()

#                     plt.plot(t, u_input,color='red')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('Flow rate of B $[m^{3}/h]$')
#                     title=case[0]+' in '+case[1]+' Flow rate'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.show()    
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()
#     # plot of states
#     for k in m.K:
#         t_pro=[]
#         state=[]
#         for t in m.T:
#             t_pro.append(m.t_p[t])
#             state.append(pe.value(m.S[k,t]))

#         plt.plot(t_pro, state,color='red')
#         plt.xlabel('Time [h]')
#         plt.ylabel('State level $[m^{3}]$')
#         title='state '+k
#         plt.title(title)
#         plt.show()
#         # plt.savefig("figures/"+title+".svg") 
#         plt.clf()
#         plt.cla()
#         plt.close()

#     #--------------------------------- Gantt plot--------------------------------------------
#     fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
#     # Setting Y-axis limits
#     gnt.set_ylim(8, 52) #TODO: change depending case study
    
#     # Setting X-axis limits
#     gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
#     # Setting labels for x-axis and y-axis
#     gnt.set_xlabel('Time [h]')
#     gnt.set_ylabel('Units')
    
#     # Setting ticks on y-axis
#     gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
#     # Labelling tickes of y-axis
#     gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
    
    
#     # Setting graph attribute
#     gnt.grid(False)
    
#     # Declaring bars in schedule
#     height=9
#     already_used=[]
#     for j in m.J:

#         if j=='U1':
#             lower_y_position=40    
#         elif j=='U2':
#             lower_y_position=30    
#         elif j=='U3':
#             lower_y_position=20
#         elif j=='U4':
#             lower_y_position=10
#         for i in m.I:
#             if i=='T1':
#                 bar_color='tab:red'
#             elif i=='T2':
#                 bar_color='tab:green'    
#             elif i=='T3':
#                 bar_color='tab:blue'    
#             elif i=='T4':
#                 bar_color='tab:orange' 
#             elif i=='T5':
#                 bar_color='tab:olive'
#             for t in m.T:
#                 try:
#                     if i in m.I_dynamics and j in m.J_dynamics:
#                         if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
#                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
#                             already_used.append(i)
#                         elif pe.value(m.X[i,j,t])==1:
#                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
#                     else:
#                         if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
#                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
#                             already_used.append(i)
#                         elif pe.value(m.X[i,j,t])==1:
#                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        

#                 except:
#                     pass 
#     gnt.tick_params(axis='both', which='major', labelsize=15)
#     gnt.tick_params(axis='both', which='minor', labelsize=15) 
#     gnt.yaxis.label.set_size(15)
#     gnt.xaxis.label.set_size(15)
#     plt.legend()
#     plt.show()
#     # plt.savefig("figures/gantt_minlp.svg")   
#     plt.clf()
#     plt.cla()
#     plt.close()


####CASE STUDY 2 WITH DISTILLATION DYNAMICS ###############################

    print('******CASE STUDY 2 WITH DISTILLATION DYNAMICS************')

# ###############################################################################
# #########--------------base case ------------------############################
# ###############################################################################
# ###############################################################################

    # obj_Selected='profit_max'



    # initialization=[1, 1, 1, 1, 1, 1, 1, 1]
  
    # mip_solver='CPLEX'
    # minlp_solver='DICOPT'
    # nlp_solver='conopt4'
    # transform='bigm'
    # #tried 5 and no improvement. With 15 DICOT is unable, and now DSDA can solve the problem.
    # last_disc=15
    # last_time_h=5

    # if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    #     print('DICOPT options:',sub_options)
    # elif minlp_solver=='OCTERACT':
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    # LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):0.1}
    # UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):3}
    # kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME,'sequential':False}


    # model_witn_distillation_dynamics=True

# ###############################################################################
# #########--------------sequential naive-------------###########################
# ###############################################################################
# ###############################################################################
    # initialization_test=[1, 6, 6, 1, 1, 1, 1, 9] 
    # print('\n-------SEQUENTIAL NAIVE-------------------------------------')
    # kwargs2=kwargs.copy()
    # kwargs2['sequential']=True

    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_min_proc_time_with_distillation
    # m=model_fun(**kwargs2)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)



    # ## RUN THIS TO SOLVE
    # m=sequential_non_iterative_2_case2(logic_fun,initialization_test,model_fun,kwargs2,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = True, global_tee= True,rel_tol = 0, with_distillation=model_witn_distillation_dynamics)
    # ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_and_dynamics_solution_with_distillation')

    # Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    # print(Sol_found)
    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJECTIVE=TPC1+TPC2+TPC3+TMC-SALES

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJECTIVE: ',OBJECTIVE)

# ###############################################################################
# #########--------------sequential ------------------###########################
# ###############################################################################
# ###############################################################################
    # print('\n-------SEQUENTIAL-------------------------------------')
    # kwargs['sequential']=True

    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,_=sequential_iterative_2_case2(logic_fun,initialization,model_fun,kwargs,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,gams_output=False,tee = True, global_tee= True,rel_tol = 0,dynamic_dist_model=True)
    # save=generate_initialization(m=m,model_name='case_2_sequential_with_distillation')
    # # save=generate_initialization(m=m,model_name='case_2_sequential_with_distillation_discrete_processing_time')
    # Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m.minTau[I,J]+1)
    # # for I_J in m.I_J:
    # #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))

# ###############################################################################
# #########--------------dicopt ----------------#################################
# ###############################################################################
# ###############################################################################
    # print('\n-------DICOPT-------------------------------------')
    # kwargs['sequential']=False
    # kwargs['x_initial']=[1, 2, 3, 1, 1, 1, 1, 7, 2, 2, 4, 4, 1, 2, 1, 2] # from sequential iterative
    # logic_fun=problem_logic_scheduling_complete
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
    # m=model_fun(**kwargs)
    # m=initialize_model(m,from_feasible=True,feasible_model='case_2_sequential_with_distillation') 
    # start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()    
    # solname='case_2_with_distillation_opt_'+minlp_solver
    # save=generate_initialization(m=m,model_name=solname)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     m.dicopt_status='Infeasible'
    # else:
    #     m.dicopt_status='Optimal'

    # if m.dicopt_status=='Optimal':
    #     Sol_founddicopt=[]
    #     for I in m.I:
    #         for J in m.J:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_founddicopt.append(K-m.minTau[I,J]+1)
    #     for I_J in m.I_J:
    #         Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    # else:
    #     print('DICOPT infeasible')

    #     TPC1=pe.value(m.TCP1)
    #     TPC2=pe.value(m.TCP2)
    #     TPC3=pe.value(m.TCP3)
    #     TMC=pe.value(m.TMC)
    #     SALES=pe.value(m.SALES)
    #     OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    #     print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    #     print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    #     print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    #     print('TMC: Total material cost: ',str(TMC))
    #     print('SALES: Revenue form selling products: ',str(SALES))
    #     print('OBJ:',str(OBJVAL))
###############################################################################
#########--------------dbd-approx_sol_subproblems ------------------###########
###############################################################################
###############################################################################

    # print('\n-------DBD-approx solution of subproblems-------------------------------------')
    # Sol_found=[1, 2, 3, 1, 1, 1, 1, 7, 2, 2, 4, 4, 1, 2, 1, 2] # from sequential iterative
    # feas_model='case_2_sequential_with_distillation' # from sequential iterative
    # kwargs['sequential']=True
    # kwargs['x_initial']=Sol_found
    # initialization=Sol_found
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # neigh=neighborhood_k_eq_2(len(Sol_found))



    # logic_fun=problem_logic_scheduling_complete
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation


    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

    # for j in (jj for jj in neigh if np.all(np.array(Sol_found)+np.array(neigh[jj])>=np.array([lower_bounds[k] for k in lower_bounds.keys()]))  and np.all(np.array(Sol_found)+np.array(neigh[jj])<=np.array([upper_bounds[k] for k in lower_bounds.keys()]))): 
    #     print(np.array(Sol_found)+np.array(neigh[j]))
    # print(np.array([lower_bounds[k] for k in lower_bounds.keys()])>=np.array([lower_bounds[k] for k in upper_bounds.keys()]))

    # start=time.time()
    #                                                                                                                                                                 ## TODO: this second model function is a version of the model with only scheduling constraints. Work on this!!!!!
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=True,with_distillation=model_witn_distillation_dynamics,provide_starting_initialization=True,feasible_model=feas_model)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_2_dbd_with_distillation_aprox_subproblems_'+minlp_solver+'_'+neighdef+'_all_neigh_Verified'
    # save=generate_initialization(m=m,model_name=solname) 
    # new_Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     new_Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     new_Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    # print(new_Sol_found)
    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))  

###############################################################################
#########--------------dbd-approx_sol_subproblems from infeasible ----#########
###############################################################################
###############################################################################

    # print('\n-------DBD-approx solution of subproblems from infeasible-------------------------------------')



    # logic_fun=problem_logic_scheduling_complete
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
    # model_fun_scheduling=case_2_scheduling_only_lower_bound_tau
    # infinity_val=1e+4
    # maxiter=10000
    # neighdef='2'
    # kwargs['sequential']=True


    # m_scheduling_only=model_fun_scheduling(**kwargs)
    # sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','$offecho \n']} 
    # m_scheduling_only = solve_with_minlp(m_scheduling_only,transformation='bigm',minlp='cplex',minlp_options=sub_options_cplex_Feas,timelimit=360000000,gams_output=False,tee=True,rel_tol=0)
    # Sol_found=[]
    # for I in m_scheduling_only.I:
    #     for J in m_scheduling_only.J:
    #         if m_scheduling_only.I_i_j_prod[I,J]==1:
    #             for K in m_scheduling_only.ordered_set[I,J]:
    #                 if round(pe.value(m_scheduling_only.YR_disjunct[I,J][K].indicator_var))==1:
    #                     Sol_found.append(K-m_scheduling_only.minTau[I,J]+1)
    # for I_J in m_scheduling_only.I_J:
    #     Sol_found.append(1+round(pe.value(m_scheduling_only.Nref[I_J])))

    # print('Initialization=',Sol_found)



    # initialization=Sol_found
    # neigh=neighborhood_k_eq_2(len(Sol_found))




    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)


    # start=time.time()

    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=True,with_distillation=model_witn_distillation_dynamics,provide_starting_initialization=False)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # end=time.time()

    # solname='case_2_dbd_with_distillation_aprox_subproblems_'+minlp_solver+'_'+neighdef+'_all_neigh_Verified_from_infeasible'
    # save=generate_initialization(m=m,model_name=solname) 
    # new_Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     new_Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     new_Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    # print(new_Sol_found)
    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES)) 
# #######-------plots------------------------
#     for I in m.I_dynamics:
#         for J in m.J_dynamics:
#             for T in m.T:
#                 if pe.value(m.X[I,J,T])==1: 
#                     case=(I,J,T)
#                     t=[]
#                     CA=[]
#                     CB=[]
#                     CC=[]
#                     Tr=[]
#                     Tj=[]
#                     Fhot=[]
#                     Fcold=[]
#                     u_input=[]
#                     for N in m.N[case]:
#                         t.append(N*m.varTime[case].value)
#                         Tr.append(m.TRvar[case][N].value)
#                         Tj.append(m.TJvar[case][N].value)
#                         Fhot.append(m.Fhot[case][N].value)
#                         Fcold.append(m.Fcold[case][N].value)
#                         CA.append( m.CA[case][N].value)
#                         CB.append( m.CB[case][N].value)
#                         CC.append( m.CC[case][N].value)
#                         u_input.append(m.u_input[case][N].value)
                        
                        
#                     plt.plot(t, CA,label='CA',color='red')
#                     plt.plot(t, CB,label='CB',color='green')
#                     plt.plot(t, CC,label='CC',color='blue')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('$Concentration [kmol/m^{3}]$')
#                     title=case[0]+' in '+case[1]+' Concentration'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.legend()
#                     plt.show()
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()

#                     plt.plot(t,Tr,label='T_reactor',color='red')
#                     plt.plot(t,Tj,label='T_jacket',color='blue')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('Temperature [K]')
#                     title=case[0]+' in '+case[1]+' Temperature'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.legend()
#                     plt.show()
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()
                    
#                     plt.plot(t, Fhot,label='F_hot',color='red')
#                     plt.plot(t,Fcold,label='F_cold',color='blue')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('Flow rate $[m^{3}/h]$')
#                     title=case[0]+' in '+case[1]+' Flow rate'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.legend()
#                     plt.show()    
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()

#                     plt.plot(t, u_input,color='red')
#                     plt.xlabel('Time [h]')
#                     plt.ylabel('Flow rate of B $[m^{3}/h]$')
#                     title=case[0]+' in '+case[1]+' Flow rate'
#                     plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
#                     plt.show()    
#                     # plt.savefig("figures/"+title+".svg") 
#                     plt.clf()
#                     plt.cla()
#                     plt.close()
#     # plot of states
#     for k in m.K:
#         t_pro=[]
#         state=[]
#         for t in m.T:
#             t_pro.append(m.t_p[t])
#             state.append(pe.value(m.S[k,t]))

#         plt.plot(t_pro, state,color='red')
#         plt.xlabel('Time [h]')
#         plt.ylabel('State level $[m^{3}]$')
#         title='state '+k
#         plt.title(title)
#         plt.show()
#         # plt.savefig("figures/"+title+".svg") 
#         plt.clf()
#         plt.cla()
#         plt.close()

#     #--------------------------------- Gantt plot--------------------------------------------
#     fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
#     # Setting Y-axis limits
#     gnt.set_ylim(8, 52) #TODO: change depending case study
    
#     # Setting X-axis limits
#     gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
#     # Setting labels for x-axis and y-axis
#     gnt.set_xlabel('Time [h]')
#     gnt.set_ylabel('Units')
    
#     # Setting ticks on y-axis
#     gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
#     # Labelling tickes of y-axis
#     gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
    
    
#     # Setting graph attribute
#     gnt.grid(False)
    
#     # Declaring bars in schedule
#     height=9
#     already_used=[]
#     for j in m.J:

#         if j=='U1':
#             lower_y_position=40    
#         elif j=='U2':
#             lower_y_position=30    
#         elif j=='U3':
#             lower_y_position=20
#         elif j=='U4':
#             lower_y_position=10
#         for i in m.I:
#             if i=='T1':
#                 bar_color='tab:red'
#             elif i=='T2':
#                 bar_color='tab:green'    
#             elif i=='T3':
#                 bar_color='tab:blue'    
#             elif i=='T4':
#                 bar_color='tab:orange' 
#             elif i=='T5':
#                 bar_color='tab:olive'
#             for t in m.T:
#                 try:
#                     if i in m.I_dynamics and j in m.J_dynamics:
#                         if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
#                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
#                             already_used.append(i)
#                         elif pe.value(m.X[i,j,t])==1:
#                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
#                     else:
#                         if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
#                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
#                             already_used.append(i)
#                         elif pe.value(m.X[i,j,t])==1:
#                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        

#                 except:
#                     pass 
#     gnt.tick_params(axis='both', which='major', labelsize=15)
#     gnt.tick_params(axis='both', which='minor', labelsize=15) 
#     gnt.yaxis.label.set_size(15)
#     gnt.xaxis.label.set_size(15)
#     plt.legend()
#     plt.show()
#     # plt.savefig("figures/gantt_minlp.svg")   
#     plt.clf()
#     plt.cla()
#     plt.close()

