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
from Scheduling_control_variable_tau_model import scheduling_only_gdp_N_solvegdp_simpler,scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    nlp_solver='conopt4'

    # # SHORT SCHEDULING
    # # ext_vars=[4, 4, 6, 6, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2] #Best solution known from sequential iterative, short scheduling obj=-1148
    # # ext_vars=[3, 5, 5, 6, 2, 5, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3] #Solution fron infeasible initialization, obj=-1085
    # ext_vars=[4, 4, 5, 5, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2] #Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
    # # ext_vars=[1, 1, 1, 1, 1, 1, 3, 3, 2, 4, 4, 3, 4, 4, 4, 5] #Scheduling only. Remember to activate scheduling only in solution of subproblem
    # sub_options={}
    # # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
    # start=time.time()
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={} #,'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    # m=model_fun(**kwargs)
    # end=time.time()
    # # print('model generation time=',str(end-start))
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # start=time.time()
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    # end=time.time()
    # # print('get info from model time=',str(end-start))
    # start=time.time()
    # m_fixed = external_ref(m=m,x=ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
    # end=time.time()
    # # print('ext_Ref_required time=',str(end-start))
    # start=time.time()
    # m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=False,rel_tol=0)
    # end=time.time()
    # # print('solve subproblem time=',str(end-start))


# #     # LONG SCHEDULING (30)
#     # ext_vars= #Best solution known from sequential iterative, short scheduling obj=-1148
#     # ext_vars= #Solution fron infeasible initialization, obj=-1085
#     ext_vars=[4, 4, 5, 5, 3, 3, 4, 2, 5, 5, 6, 7, 2, 7, 6, 8] #Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
#     # ext_vars= #Scheduling only. Remember to activate scheduling only in solution of subproblem
#     sub_options={}
#     # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
#     start=time.time()
#     model_fun =scheduling_and_control_GDP_complete_approx
#     logic_fun=problem_logic_scheduling_dummy
#     kwargs={'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3} #,'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
#     m=model_fun(**kwargs)
#     end=time.time()
#     # print('model generation time=',str(end-start))
#     ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
#     ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
#     start=time.time()
#     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
#     end=time.time()
#     # print('get info from model time=',str(end-start))
#     start=time.time()
#     m_fixed = external_ref(m=m,x=ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
#     end=time.time()
#     # print('ext_Ref_required time=',str(end-start))
    
#     # m.cuts.add(m.SALES<=100000000)
#     start=time.time()
#     m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=False,rel_tol=0.1)
#     end=time.time()
#     # print('solve subproblem time=',str(end-start))

#     # MEDIUM SCHEDULING 28
    # # ext_vars= #Best solution known from sequential iterative, short scheduling obj=
    # # ext_vars= #Solution fron infeasible initialization, obj=
    # ext_vars=[4, 4, 5, 5, 3, 3, 4, 3, 3, 5, 5, 5, 4, 6, 5, 7] #Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
    # # ext_vars= #Scheduling only. Remember to activate scheduling only in solution of subproblem
    # sub_options={}
    # # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
    # start=time.time()
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2} #,'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    # m=model_fun(**kwargs)
    # end=time.time()
    # # print('model generation time=',str(end-start))
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # start=time.time()
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    # end=time.time()
    # # print('get info from model time=',str(end-start))
    # start=time.time()
    # m_fixed = external_ref(m=m,x=ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
    # end=time.time()
    # # print('ext_Ref_required time=',str(end-start))
    
    # # m.cuts.add(m.SALES<=100000000)
    # start=time.time()
    # m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=False,rel_tol=0)
    # end=time.time()
    # # print('solve subproblem time=',str(end-start))



# LOOP ANALYSIS LONG SCHEDULING
# max_sales=[13360,
# 12925,
# 12490,
# 12055,
# 11620,
# 11185,
# 10750,
# 10315,
# 9880,
# 9445,
# 9010,
# 8575,
# 8140,
# 7705,
# 7270,
# 6835,
# 6400]

# OBJECTIVE_list=[0 for i in range(len(max_sales))]
# TPC1_list=[0 for i in range(len(max_sales))]
# TPC2_list=[0 for i in range(len(max_sales))]
# TPC3_list=[0 for i in range(len(max_sales))]
# TMC_list=[0 for i in range(len(max_sales))]
# TOTAL_COST_list=[0 for i in range(len(max_sales))]
# SALES_list=[0 for i in range(len(max_sales))]
# PRODUCT1_list=[0 for i in range(len(max_sales))]
# PRODUCT2_list=[0 for i in range(len(max_sales))]
# for i in range(len(max_sales)):

#     ext_vars=[4, 4, 5, 5, 3, 3, 4, 2, 5, 5, 6, 7, 2, 7, 6, 8] #Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
#     # ext_vars= #Scheduling only. Remember to activate scheduling only in solution of subproblem
#     sub_options={}
#     # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
#     start=time.time()
#     model_fun =scheduling_and_control_GDP_complete_approx
#     logic_fun=problem_logic_scheduling_dummy
#     kwargs={'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3} #,'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
#     m=model_fun(**kwargs)
#     end=time.time()
#     # print('model generation time=',str(end-start))
#     ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
#     ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
#     start=time.time()
#     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
#     end=time.time()
#     # print('get info from model time=',str(end-start))
#     start=time.time()
#     m_fixed = external_ref(m=m,x=ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
#     end=time.time()
#     # print('ext_Ref_required time=',str(end-start))
    
#     # m.cuts.add(m.SALES<=max_sales[i])
#     # m.revenue.pprint()
#     start=time.time()
#     m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=True,rel_tol=0.05,max_sales=max_sales[i])
#     end=time.time()
#     # print('solve subproblem time=',str(end-start))
    
#     TPC1_list[i]=sum(sum(sum(  m.fixed_cost[I,J]*pe.value(m.X[I,J,T]) for J in m.J)for I in m.I)for T in m.T)
#     TPC2_list[i]=sum(sum(sum( m.variable_cost[I,J]*pe.value(m.B[I,J,T]) for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
#     TPC3_list[i]=sum(sum(sum(pe.value(m.X[I,J,T])*(m.hot_cost*pe.value(m.Integral_hot[I,J][m.N[I,J].last()])   +  m.cold_cost*pe.value(m.Integral_cold[I,J][m.N[I,J].last()])  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors)
#     TMC_list[i]=sum( m.raw_cost[K]*(m.S0[K]-pe.value(m.S[K,m.lastT])) for K in m.K_inputs)
#     TOTAL_COST_list[i]=TPC1_list[i]+TPC2_list[i]+TPC3_list[i]+TMC_list[i]
#     SALES_list[i]=sum( m.revenue[K]*pe.value(m.S[K,m.lastT])  for K in m.K_products)
#     OBJECTIVE_list[i]=(TPC1_list[i]+TPC2_list[i]+TPC3_list[i]+TMC_list[i]-SALES_list[i])
#     PRODUCT1_list[i]=pe.value(m.S['P1',60])
#     PRODUCT2_list[i]=pe.value(m.S['P2',60])

# print('OBJECTIVE ',OBJECTIVE_list)
# print('TPC1 ',TPC1_list)
# print('TPC2 ',TPC2_list)
# print('TPC3',TPC3_list)
# print('TMC',TMC_list)
# print('SALES',SALES_list)
# print('TOTAL COST',TOTAL_COST_list)
# print('P1',PRODUCT1_list)
# print('P2',PRODUCT2_list)
# plt.plot(max_sales, TPC2_list,label='TPC steady state',color='green')
# plt.plot(max_sales, TPC3_list,label='TPC dynamics',color='orange')
# plt.plot(max_sales, TMC_list,label='TMC',color='red')
# plt.plot(max_sales, SALES_list,label='SALES',color='blue')
# plt.plot(max_sales, OBJECTIVE_list,label='OBJECTIVE',color='black')
# plt.xlabel('SALES [$]')
# plt.ylabel('Objective function terms [$]')
# plt.legend()
# plt.show()

# plt.plot(max_sales, PRODUCT1_list,label='P1',color='green')
# plt.plot(max_sales, PRODUCT2_list,label='P2',color='orange')

# plt.ylabel('Amount produced [$m^{3}$]')
# plt.xlabel('SALES [$]')
# plt.legend()
# plt.show()

# plt.plot(max_sales, TOTAL_COST_list,label='TOTAL COST',color='green')
# plt.plot(max_sales, SALES_list,label='SALES',color='blue')
# plt.plot(max_sales, OBJECTIVE_list,label='OBJECTIVE',color='black')
# plt.xlabel('SALES [$]')
# plt.ylabel('Objective function terms [$]')
# plt.legend()
# plt.show()


# LOOP ANALYSIS SHORT SCHEDULING
# max_sales=[4000,	
# 3990.27777777778,	
# 3980.55555555555,	
# 3970.83333333333,	
# 3961.11111111111,	
# 3951.38888888889,	
# 3941.66666666666,	
# 3931.94444444444,	
# 3922.22222222222,	
# 3912.5,	
# 3902.77777777777,	
# 3893.05555555555,	
# 3883.33333333333,	
# 3873.61111111111,	
# 3863.88888888888,	
# 3854.16666666666,	
# 3844.4444444444444444444444444,
# 3737.5,
# 3606.25,
# 3475,
# 3343.75,
# 3212.5,
# 3081.25,
# 2950,
# 2818.75,
# 2687.5,
# 2556.25,
# 2425,
# 2293.75,
# 2162.5,
# 2031.25,
# 1900]

# OBJECTIVE_list=[0 for i in range(len(max_sales))]
# TPC1_list=[0 for i in range(len(max_sales))]
# TPC2_list=[0 for i in range(len(max_sales))]
# TPC3_list=[0 for i in range(len(max_sales))]
# TMC_list=[0 for i in range(len(max_sales))]
# TOTAL_COST_list=[0 for i in range(len(max_sales))]
# SALES_list=[0 for i in range(len(max_sales))]
# PRODUCT1_list=[0 for i in range(len(max_sales))]
# PRODUCT2_list=[0 for i in range(len(max_sales))]
# for i in range(len(max_sales)):

#     ext_vars=[4, 4, 5, 5, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2]#Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
#     # ext_vars= #Scheduling only. Remember to activate scheduling only in solution of subproblem
#     sub_options={}
#     # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
#     start=time.time()
#     model_fun =scheduling_and_control_GDP_complete_approx
#     logic_fun=problem_logic_scheduling_dummy
#     kwargs={} #,'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
#     m=model_fun(**kwargs)
#     end=time.time()
#     # print('model generation time=',str(end-start))
#     ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
#     ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
#     start=time.time()
#     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
#     end=time.time()
#     # print('get info from model time=',str(end-start))
#     start=time.time()
#     m_fixed = external_ref(m=m,x=ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
#     end=time.time()
#     # print('ext_Ref_required time=',str(end-start))
    
#     # m.cuts.add(m.SALES<=max_sales[i])
#     # m.revenue.pprint()
#     start=time.time()
#     m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=True,rel_tol=0.05,max_sales=max_sales[i])
#     end=time.time()
#     # print('solve subproblem time=',str(end-start))
    
#     TPC1_list[i]=sum(sum(sum(  m.fixed_cost[I,J]*pe.value(m.X[I,J,T]) for J in m.J)for I in m.I)for T in m.T)
#     TPC2_list[i]=sum(sum(sum( m.variable_cost[I,J]*pe.value(m.B[I,J,T]) for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
#     TPC3_list[i]=sum(sum(sum(pe.value(m.X[I,J,T])*(m.hot_cost*pe.value(m.Integral_hot[I,J][m.N[I,J].last()])   +  m.cold_cost*pe.value(m.Integral_cold[I,J][m.N[I,J].last()])  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors)
#     TMC_list[i]=sum( m.raw_cost[K]*(m.S0[K]-pe.value(m.S[K,m.lastT])) for K in m.K_inputs)
#     TOTAL_COST_list[i]=TPC1_list[i]+TPC2_list[i]+TPC3_list[i]+TMC_list[i]
#     SALES_list[i]=sum( m.revenue[K]*pe.value(m.S[K,m.lastT])  for K in m.K_products)
#     OBJECTIVE_list[i]=(TPC1_list[i]+TPC2_list[i]+TPC3_list[i]+TMC_list[i]-SALES_list[i])
#     PRODUCT1_list[i]=pe.value(m.S['P1',28])
#     PRODUCT2_list[i]=pe.value(m.S['P2',28])

# print('OBJECTIVE ',OBJECTIVE_list)
# print('TPC1 ',TPC1_list)
# print('TPC2 ',TPC2_list)
# print('TPC3',TPC3_list)
# print('TMC',TMC_list)
# print('SALES',SALES_list)
# print('TOTAL COST',TOTAL_COST_list)
# print('P1',PRODUCT1_list)
# print('P2',PRODUCT2_list)
# plt.plot(max_sales, TPC2_list,label='TPC steady state',color='green')
# plt.plot(max_sales, TPC3_list,label='TPC dynamics',color='orange')
# plt.plot(max_sales, TMC_list,label='TMC',color='red')
# plt.plot(max_sales, SALES_list,label='SALES',color='blue')
# plt.plot(max_sales, OBJECTIVE_list,label='OBJECTIVE',color='black')
# plt.xlabel('SALES  [$]')
# plt.ylabel('Objective function terms [$]')
# plt.legend()
# plt.show()

# plt.plot(max_sales, PRODUCT1_list,label='P1',color='green')
# plt.plot(max_sales, PRODUCT2_list,label='P2',color='orange')

# plt.ylabel('Amount produced [$m^{3}$]')
# plt.xlabel('SALES  [$]')
# plt.legend()
# plt.show()

# plt.plot(max_sales, TOTAL_COST_list,label='TOTAL COST',color='green')
# plt.plot(max_sales, SALES_list,label='SALES',color='blue')
# plt.plot(max_sales, OBJECTIVE_list,label='OBJECTIVE',color='black')
# plt.xlabel('SALES  [$]')
# plt.ylabel('Objective function terms [$]')
# plt.legend()
# plt.show()


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

    model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    m=model_fun(**kwargs)
    # init_name='case_1_scheduling_and_dynamics_solution' # sequential naive
    # init_name='case_1_scheduling_and_dynamics_solution_seq_iterative' #sequential iterative
    # init_name='case_1_scheduling_and_dynamics_solution_DSDA_naive' #DSDA
    init_name='bestknown' # best known solution


    m=initialize_model(m,from_feasible=True,feasible_model=init_name) 
# # ####--------Objective function summary---------------------------------
    TPC1=sum(sum(sum(  m.fixed_cost[I,J]*pe.value(m.X[I,J,T]) for J in m.J)for I in m.I)for T in m.T)
    TPC2=sum(sum(sum( m.variable_cost[I,J]*pe.value(m.B[I,J,T]) for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    TPC3=sum(sum(sum(pe.value(m.X[I,J,T])*(m.hot_cost*pe.value(m.Integral_hot[I,J][m.N[I,J].last()])   +  m.cold_cost*pe.value(m.Integral_cold[I,J][m.N[I,J].last()])  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors)
    TMC=sum( m.raw_cost[K]*(m.S0[K]-pe.value(m.S[K,m.lastT])) for K in m.K_inputs)
    SALES=sum( m.revenue[K]*pe.value(m.S[K,m.lastT])  for K in m.K_products)
    OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJ:',str(OBJVAL))
    print('----')
    print('TCP1 gams:',str(pe.value(m.TCP1)))
    print('TCP2 gams:',str(pe.value(m.TCP2)))
    print('TCP3 gams:',str(pe.value(m.TCP3)))
    print('TMC gams:',str(pe.value(m.TMC)))
    print('SALES gams:',str(pe.value(m.SALES)))
    # m.E_DEMAND_SATISFACTION.pprint()
    # print('S product 1: ' ,pe.value(m.S['P1',56]))
    # print('S product 2: ' ,pe.value(m.S['P2',56]))

#######-------plots------------------------
    for I in m.I_reactions:
        for J in m.J_reactors:
            case=(I,J)
            t=[]
            c1=[]
            c2=[]
            c3=[]
            Tr=[]
            Tj=[]
            Fhot=[]
            Fcold=[]
            for N in m.N[case]:
                t.append(N*m.varTime[I,J].value)
                Tr.append(m.TRvar[case][N].value)
                Tj.append(m.TJvar[case][N].value)
                Fhot.append(m.Fhot[case][N].value)
                Fcold.append(m.Fcold[case][N].value)
                c1.append( m.Cvar[case][N,list(m.Q_balance[I])[0]].value)
                c2.append( m.Cvar[case][N,list(m.Q_balance[I])[1]].value)
                c3.append( m.Cvar[case][N,list(m.Q_balance[I])[2]].value)
                
                
            plt.plot(t, c1,label=list(m.Q_balance[I])[0],color='red')
            plt.plot(t, c2,label=list(m.Q_balance[I])[1],color='green')
            plt.plot(t, c3,label=list(m.Q_balance[I])[2],color='blue')
            plt.xlabel('Time [h]')
            plt.ylabel('$Concentration [kmol/m^{3}]$')
            title=case[0]+' in '+case[1]+' Concentration'
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            # plt.show()
            plt.savefig("figures/"+title+".svg") 
            plt.clf()
            plt.cla()
            plt.close()

            plt.plot(t,Tr,label='T_reactor',color='red')
            plt.plot(t,Tj,label='T_jacket',color='blue')
            plt.xlabel('Time [h]')
            plt.ylabel('Temperature [K]')
            title=case[0]+' in '+case[1]+' Temperature'
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            # plt.show()
            plt.savefig("figures/"+title+".svg") 
            plt.clf()
            plt.cla()
            plt.close()
            
            plt.plot(t, Fhot,label='F_hot',color='red')
            plt.plot(t,Fcold,label='F_cold',color='blue')
            plt.xlabel('Time [h]')
            plt.ylabel('Flow rate $[m^{3}/h]$')
            title=case[0]+' in '+case[1]+' Flow rate'
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            # plt.show()    
            plt.savefig("figures/"+title+".svg") 
            plt.clf()
            plt.cla()
            plt.close()
    # plot of states
    for k in m.K:
        t_pro=[]
        state=[]
        for t in m.T:
            t_pro.append(m.t_p[t])
            state.append(pe.value(m.S[k,t]))

        plt.plot(t_pro, state,color='red')
        plt.xlabel('Time [h]')
        plt.ylabel('State level $[m^{3}]$')
        title='state '+k
        plt.title(title)
        # plt.show()
        plt.savefig("figures/"+title+".svg") 
        plt.clf()
        plt.cla()
        plt.close()

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
                        if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif round(pe.value(m.X[i,j,t]))==1:
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                                
                    else:
                        if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif round(pe.value(m.X[i,j,t]))==1:
                            gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        
                except:
                    pass 
    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    # plt.show()
    plt.savefig("figures/gantt_minlp.svg")   
    plt.clf()
    plt.cla()
    plt.close()
