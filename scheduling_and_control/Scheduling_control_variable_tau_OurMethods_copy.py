from __future__ import division
import sys
# sys.path.insert(0, '/home/dadapy/GeneralBenders/')
# sys.path.append('C:/Users/TEMP/Desktop/GeneralBenders/') #for LRLAB5
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_aprox
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_gdpopt,solve_with_minlp
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
from Scheduling_control_variable_tau_model import scheduling_only_gdp_N_solvegdp_simpler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    #Solver declaration
    minlp_solver='dicopt'
    nlp_solver='conopt4'
    mip_solver='cplex'
    gdp_solver='GLOA'
    if minlp_solver=='dicopt':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option optcr=0;\n','option optca=0;\n','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 2000 \n','$offecho \n']}
    else:
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n']}



    ##### ----------ONLY PROCESSING TIMES AS EXTERNAL VARIABLES------------------------
    ##### -----------------------------------------------------------------------------

    # #Solve with LD-SDA. Only processing times as external vars
    # model_fun =scheduling_and_control_GDP
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[1,1,1,1,1,1],ext_ref,logic_fun,k = 'Infinity',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = True,tee= True,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    # #Solve with LD-BD. Only processing times as external vars
    # initialization=[1,1,1,1,1,1]
    # infinity_val=1e+8
    # maxiter=1000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_GDP
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dbd.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())



    #Solve with pyomo.GDP. Only processing times as external vars
    # kwargs={}
    # model_fun=scheduling_and_control_GDP 
    # m=model_fun(**kwargs)
    # m = solve_with_gdpopt(m, mip=mip_solver,minlp=minlp_solver,nlp=nlp_solver,minlp_options=sub_options, timelimit=1000,strategy=gdp_solver, mip_output=False, nlp_output=False,rel_tol=0,tee=True)

    #Solve with MINLP. Only processing times as external vars
    # kwargs={}
    # model_fun=scheduling_and_control_GDP
    # m=model_fun(**kwargs)
    # m = solve_with_minlp(m, transformation='hull', minlp='dicopt', minlp_options=sub_options,gams_output=False,tee=True,rel_tol=0)

    ##### ----------BOTH PROCESSING TIMES AND BATCHING VARIABLES AS EXTERNAL VARIABLES-
    ##### -----------------------------------------------------------------------------

    # # Solve with LD-SDA_COMPLETE GDP
    # model_fun =scheduling_and_control_GDP_complete#scheduling_and_control_gdp_N_solvegdp_simpler ## or i can use scheduling_and_control_GDP_complete and problem_logic_scheduling_dummy and add the complementary model alternatively
    # logic_fun=problem_logic_scheduling_dummy#problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2],ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    # SOLVE WITH LD-BD***************************************
    initialization=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2] 
    infinity_val=1e+4
    maxiter=1000
    neigh=neighborhood_k_eq_2(len(initialization))
    model_fun =scheduling_and_control_GDP_complete
    logic_fun=problem_logic_scheduling_dummy
    kwargs={}
    m=model_fun(**kwargs)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    print('Objective value: ',str(pe.value(m.obj)))
    print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    textbuffer = io.StringIO()
    for v in m.component_objects(pe.Var, descend_into=True):
        v.pprint(textbuffer)
        textbuffer.write('\n')
    textbuffer.write('\n Objective: \n') 
    textbuffer.write(str(pe.value(m.obj)))    
    with open('Results_variable_tau_dbd_complete.txt', 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())


    #############Solve with pyomo.GDP COMPLETE GDP
    # kwargs={'x_initial':[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]}
    # model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # m = solve_with_gdpopt(m, mip=mip_solver,minlp=minlp_solver,nlp=nlp_solver,minlp_options=sub_options, timelimit=360000,strategy=gdp_solver, mip_output=True, nlp_output=True,minlp_output=True,rel_tol=0,tee=True)

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_gdp_complete.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    #Solve with MINLP
    # kwargs={'x_initial':[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]}
    # model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # m = solve_with_minlp(m,transformation='hull',minlp=minlp_solver,minlp_options=sub_options,timelimit=3600000,gams_output=False,tee=True,rel_tol=0)

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_MINLP_complete.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())   
# ####--------Objective function summary---------------------------------
#     TPC1=sum(sum(sum(  m.fixed_cost[I,J]*pe.value(m.X[I,J,T]) for J in m.J)for I in m.I)for T in m.T)
#     TPC2=sum(sum(sum( m.variable_cost[I,J]*pe.value(m.B[I,J,T]) for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
#     TPC3=sum(sum(sum(pe.value(m.X[I,J,T])*(m.hot_cost*pe.value(m.Integral_hot[I,J][m.N[I,J].last()])   +  m.cold_cost*pe.value(m.Integral_cold[I,J][m.N[I,J].last()])  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors)
#     TMC=sum( m.raw_cost[K]*(m.S0[K]-pe.value(m.S[K,m.lastT])) for K in m.K_inputs)
#     SALES=sum( m.revenue[K]*pe.value(m.S[K,m.lastT])  for K in m.K_products)
#     OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)/100
#     print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
#     print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
#     print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
#     print('TMC: Total material cost: ',str(TMC))
#     print('SALES: Revenue form selling products: ',str(SALES))
#     print('OBJ:',str(OBJVAL))
#     print('----')
#     print('TCP1 gams:',str(pe.value(m.TCP1)))
#     print('TCP2 gams:',str(pe.value(m.TCP2)))
#     print('TCP3 gams:',str(pe.value(m.TCP3)))
#     print('TMC gams:',str(pe.value(m.TMC)))
#     print('SALES gams:',str(pe.value(m.SALES)))

    ##### ----------TESTS APPROXIMATED SOLUTION OF SUBPROBLEMS-------------------------
    ##### -----------------------------------------------------------------------------
    # start=time.time()
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # end=time.time()
    # print('model generation time=',str(end-start))
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # start=time.time()
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    # end=time.time()
    # print('get info from model time=',str(end-start))
    # start=time.time()
    # m_fixed = external_ref(m=m,x=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2],extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
    # end=time.time()
    # print('ext_Ref_required time=',str(end-start))
    # start=time.time()
    # m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=False,rel_tol=0)
    # end=time.time()
    # print('solve subproblem time=',str(end-start))



    #Solve with enhanced LD-SDA_COMPLETE GDP. Approximated solution of subproblems with pruning depending on parameter aproximate_solution in solve_subproblem_aprox
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda_aprox(model_fun,kwargs,[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2],ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete_scheduling_only.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    # Enhanced DBD WITH APPROXIMATE AND OPTIMAL SOLUTION OF SUBPROBLEMS
    # initialization=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]
    # # initialization=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # # initialization=[1,1,1,1,1,1,3,3,2,4,4,3,4,4,4,5]
    # # initialization=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #TODO: to run this I hae to activate fbbt in dsda_functions (it was deactivated)
    # infinity_val=1e+4 #TODO: DBD FROM FEASIBLE WORKED VERY WELL WITH 1E+4. I HAVE TO USE DIFFFERENT INFINITY VALUES DEPENDING ON STAGE 1 2 OR 3. I have scaled objective in phase 2
    # maxiter=10000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dbd_complete_optimal_sol_multicut_at_1_from_infeasible.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


## ----------------------------from nominal schedule------------------------------------------------
    # Enhanced DSDA
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda_aprox(model_fun,kwargs,[3,3,5,5,2,2,2,2,2,3,3,3,2,3,3,3],ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete_enhanced_solution_from_nominal.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    #Naive DSDA
    # Solve with LD-SDA_COMPLETE GDP
    # model_fun =scheduling_and_control_GDP_complete
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[3,3,5,5,2,2,2,2,2,3,3,3,2,3,3,3],ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete_from_nominal.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())   






#----------------------------Sequential iterative methodology-----------------------------------

    ## STEP 1: SCHEDULING ONLY WITH VARIABLE PROCESSSING TIMES
    # kwargs={'x_initial':[1,1,1,1,1,1]}
    # model_fun=scheduling_only_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # m = solve_with_minlp(m,transformation='hull',minlp=mip_solver,minlp_options=sub_options,timelimit=3600000,gams_output=False,tee=True,rel_tol=0)

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('STEP1_Results_variable_tau_MIP_complete.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())  


    ## STEP 2: SEQUENTIAL STRATEGY
    # model_fun =scheduling_and_control_gdp_N_approx_sequential
    # logic_fun=problem_logic_scheduling_dummy
    # m=model_fun()
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m=sequential_iterative_1(logic_fun,[1,1,1,1,1,1],model_fun,ext_ref,rate_tau=1,provide_starting_initialization = False,subproblem_solver=nlp_solver,iter_timelimit = 1000000,subproblem_solver_options=sub_options,gams_output = False,tee = False,global_tee = True,rel_tol = 0)
    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_sequential_Strategy.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    ## STEP 2: SEQUENTIAL STRATEGY IMPROVED
    # model_fun =scheduling_and_control_gdp_N_approx_sequential
    # logic_fun=problem_logic_scheduling_dummy
    # m=model_fun()
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m=sequential_iterative_2(logic_fun,[1,1,1,1,1,1],model_fun,ext_ref,rate_tau=1,provide_starting_initialization = False,subproblem_solver=nlp_solver,iter_timelimit = 1000000,subproblem_solver_options=sub_options,gams_output = False,tee = False,global_tee = True,rel_tol = 0)
    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_sequential_Strategy_improved.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

#-------------------from sequential iterative methodology-------------------------------------
    # aa=neighborhood_k_eq_l_natural_modified(16)
    # print(aa)
    # Enhanced DSDA
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda_aprox(model_fun,kwargs,[5,5,5,5,5,5,2,2,1,2,2,2,2,2,2,2],ext_ref,logic_fun,k = 'L_natural_modified',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete_approx_solution_from_iterative_L_natural_modified.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    # Naive DSDA
    # model_fun =scheduling_and_control_GDP_complete
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[5,5,5,5,5,5,2,2,1,2,2,2,2,2,2,2],ext_ref,logic_fun,k = 'M_natural',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete_from_iterative_m_natural.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())   


#-------------------from sequential iterative methodology tau only-------------------------------------
    # aa=neighborhood_k_eq_l_natural_modified(16)
    # print(aa)
    # Enhanced DSDA
    # model_fun =scheduling_and_control_GDP_complete_approx_tau_only
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda_aprox_tau_only(model_fun,kwargs,[5,5,5,5,5,5],ext_ref,logic_fun,k = 'Infinity',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_tau_only_approx_solution_from_iterative_k_infty.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    # Naive DSDA
    # model_fun =scheduling_and_control_GDP_complete
    # logic_fun=problem_logic_scheduling
    # kwargs={}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[5,5,5,5,5,5,2,2,1,2,2,2,2,2,2,2],ext_ref,logic_fun,k = 'M_natural',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = 0)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_dsda_complete_from_iterative_m_natural.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())   

#######-------plots------------------------
    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         case=(I,J)
    #         t=[]
    #         c1=[]
    #         c2=[]
    #         c3=[]
    #         Tr=[]
    #         Tj=[]
    #         Fhot=[]
    #         Fcold=[]
    #         for N in m.N[case]:
    #             t.append(N)
    #             Tr.append(m.TRvar[case][N].value)
    #             Tj.append(m.TJvar[case][N].value)
    #             Fhot.append(m.Fhot[case][N].value)
    #             Fcold.append(m.Fcold[case][N].value)
    #             c1.append( m.Cvar[case][N,list(m.Q_balance[I])[0]].value)
    #             c2.append( m.Cvar[case][N,list(m.Q_balance[I])[1]].value)
    #             c3.append( m.Cvar[case][N,list(m.Q_balance[I])[2]].value)
                
                
    #         plt.plot(t, c1,label=list(m.Q_balance[I])[0],color='red')
    #         plt.plot(t, c2,label=list(m.Q_balance[I])[1],color='green')
    #         plt.plot(t, c3,label=list(m.Q_balance[I])[2],color='blue')
    #         plt.xlabel('Time [-]')
    #         plt.ylabel('$Concentration [kmol/m^{3}]$')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()
            
    #         plt.plot(t,Tr,label='T_reactor',color='red')
    #         plt.plot(t,Tj,label='T_jacket',color='blue')
    #         plt.xlabel('Time [-]')
    #         plt.ylabel('Temperature [K]')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()
            
    #         plt.plot(t, Fhot,label='F_hot',color='red')
    #         plt.plot(t,Fcold,label='F_cold',color='blue')
    #         plt.xlabel('Time [-]')
    #         plt.ylabel('Flow rate $[m^{3}/h]$')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()    
            
    # # plot of states
    # for k in m.K:
    #     t_pro=[]
    #     state=[]
    #     for t in m.T:
    #         t_pro.append(m.t_p[t])
    #         state.append(pe.value(m.S[k,t]))

    #     plt.plot(t_pro, state,color='red')
    #     plt.xlabel('Time [h]')
    #     plt.ylabel('State level $[/m^{3}]$')
    #     plt.title('state:'+k)
    #     plt.show()

    #--------------------------------- Gantt plot--------------------------------------------
    fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 62)
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Time [h]')
    gnt.set_ylabel('Units')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45, 55])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['Pack', 'Sep', 'R_small', 'R_large','Mix'])
    
    
    # Setting graph attribute
    gnt.grid(False)
    
    # Declaring bars in schedule
    height=9
    already_used=[]
    for j in m.J:
        if j=='Mix':
            lower_y_position=50
        elif j=='R_large':
            lower_y_position=40    
        elif j=='R_small':
            lower_y_position=30    
        elif j=='Sep':
            lower_y_position=20
        elif j=='Pack':
            lower_y_position=10
        for i in m.I:
            if i=='Mix':
                bar_color='tab:red'
            elif i=='R1':
                bar_color='tab:green'    
            elif i=='R2':
                bar_color='tab:blue'    
            elif i=='R3':
                bar_color='tab:orange' 
            elif i=='Sep':
                bar_color='tab:olive'
            elif i=='Pack1':
                bar_color='tab:purple'                
            elif i=='Pack2':
                bar_color='teal'
            for t in m.T:
                try:
                    if i in m.I_reactions and j in m.J_reactors:
                        if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif pe.value(m.X[i,j,t])==1:
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                                
                    else:
                        if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif pe.value(m.X[i,j,t])==1:
                            gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        
                except:
                    pass 
    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    plt.show()
    # plt.savefig("gantt_minlp.png")
    # plt.savefig("gantt_minlp.svg")   

