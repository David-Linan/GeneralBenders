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
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx_sequential_naive,problem_logic_scheduling as problem_logic_scheduling_case1
import numpy as np
if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


####CASE STUDY 1###############################

    print('******CASE STUDY 1************')


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
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # elif minlp_solver=='OCTERACT':
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    # kwargs={}


# ###############################################################################
# #########--------------sequential naive-------------###########################
# ###############################################################################
# ###############################################################################

#     print('\n-------SEQUENTIAL NAIVE-------------------------------------')
#     kwargs2=kwargs.copy()
#     kwargs2['sequential']=True

#     logic_fun=problem_logic_scheduling_case1
#     model_fun=scheduling_and_control_gdp_N_approx_sequential_naive
#     m=model_fun(**kwargs2)
#     ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1}
#     ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
#     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

#     initialization_test=[]
#     for k in upper_bounds.keys():
#         initialization_test.append(upper_bounds[k]) 
#     print('upper bound ext var related to proc time',initialization_test)
#     ## RUN THIS TO SOLVE
#     m=sequential_non_iterative_2(logic_fun,initialization_test,model_fun,kwargs2,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = True, global_tee= True,rel_tol = 0)
#     ## RUN THIS TO RETRIEVE SOLUTION    

#     m=initialize_model(m,from_feasible=True,feasible_model='case_1_scheduling_solution')
#     m.varTime.pprint()

#     Sol_found=[]
#     for I in m.I_reactions:
#         for J in m.J_reactors:
#             if m.I_i_j_prod[I,J]==1:
#                 for K in m.ordered_set[I,J]:
#                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
#                         Sol_found.append(K-m.minTau[I,J]+1)
#     for I_J in m.I_J:
#         Sol_found.append(1+round(pe.value(m.Nref[I_J])))
#     print('EXT_VARS_FOUND',Sol_found)
#     TPC1=pe.value(m.TCP1)
#     TPC2=pe.value(m.TCP2)
#     TPC3=pe.value(m.TCP3)
#     TMC=pe.value(m.TMC)
#     SALES=pe.value(m.SALES)

#     print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
#     print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
#     print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
#     print('TMC: Total material cost: ',str(TMC))
#     print('SALES: Revenue form selling products: ',str(SALES))



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
#         # plt.show()
#         plt.savefig("figures/"+title+".svg") 
#         plt.clf()
#         plt.cla()
#         plt.close()

#     # --------------------------------- Gantt plot--------------------------------------------
#     fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
#     # Setting Y-axis limits
#     gnt.set_ylim(8, 62)
    
#     # Setting X-axis limits
#     gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
#     # Setting labels for x-axis and y-axis
#     gnt.set_xlabel('Time [h]')
#     gnt.set_ylabel('Units')
    
#     # Setting ticks on y-axis
#     gnt.set_yticks([15, 25, 35, 45, 55])
#     # Labelling tickes of y-axis
#     gnt.set_yticklabels(['Pack', 'Sep', 'R_small', 'R_large','Mix'])
    
    
#     # Setting graph attribute
#     gnt.grid(False)
    
#     # Declaring bars in schedule
#     height=9
#     already_used=[]
#     for j in m.J:
#         if j=='Mix':
#             lower_y_position=50
#         elif j=='R_large':
#             lower_y_position=40    
#         elif j=='R_small':
#             lower_y_position=30    
#         elif j=='Sep':
#             lower_y_position=20
#         elif j=='Pack':
#             lower_y_position=10
#         for i in m.I:
#             if i=='Mix':
#                 bar_color='tab:red'
#             elif i=='R1':
#                 bar_color='tab:green'    
#             elif i=='R2':
#                 bar_color='tab:blue'    
#             elif i=='R3':
#                 bar_color='tab:orange' 
#             elif i=='Sep':
#                 bar_color='tab:olive'
#             elif i=='Pack1':
#                 bar_color='tab:purple'                
#             elif i=='Pack2':
#                 bar_color='teal'
#             for t in m.T:
#                 try:
#                     if i in m.I_reactions and j in m.J_reactors:
#                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
#                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
#                             already_used.append(i)
#                         elif round(pe.value(m.X[i,j,t]))==1:
#                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                                
#                     else:
#                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
#                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
#                             already_used.append(i)
#                         elif round(pe.value(m.X[i,j,t]))==1:
#                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
#                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        
#                 except:
#                     pass 
#     gnt.tick_params(axis='both', which='major', labelsize=15)
#     gnt.tick_params(axis='both', which='minor', labelsize=15) 
#     gnt.yaxis.label.set_size(15)
#     gnt.xaxis.label.set_size(15)
#     plt.legend()
#     # plt.show()
#     plt.savefig("figures/gantt_minlp.svg")   
#     plt.clf()
#     plt.cla()
#     plt.close()


#     m2=model_fun(**kwargs2)
#     m2=initialize_model(m2,from_feasible=True,feasible_model='case_1_min_proc_time_solution')
#     ActualTPC3=sum(sum(pe.value(m.Nref[I,J])*(m2.hot_cost*pe.value(m2.Integral_hot[I, J][m2.N[I, J].last()]) + m2.cold_cost*pe.value(m2.Integral_cold[I, J][m2.N[I, J].last()])) for I in m2.I_reactions)for J in m2.J_reactors)
#     print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(ActualTPC3))
#     OBJVAL=(TPC1+TPC2+ActualTPC3+TMC-SALES)
#     print('OBJ:',str(OBJVAL))

# ######-------plots------------------------
#     for I in m2.I_reactions:
#         for J in m2.J_reactors:
#             case=(I,J)
#             t=[]
#             c1=[]
#             c2=[]
#             c3=[]
#             Tr=[]
#             Tj=[]
#             Fhot=[]
#             Fcold=[]
#             for N in m2.N[case]:
#                 t.append(N*m2.varTime[I,J].value)
#                 Tr.append(m2.TRvar[case][N].value)
#                 Tj.append(m2.TJvar[case][N].value)
#                 Fhot.append(m2.Fhot[case][N].value)
#                 Fcold.append(m2.Fcold[case][N].value)
#                 c1.append( m2.Cvar[case][N,list(m2.Q_balance[I])[0]].value)
#                 c2.append( m2.Cvar[case][N,list(m2.Q_balance[I])[1]].value)
#                 c3.append( m2.Cvar[case][N,list(m2.Q_balance[I])[2]].value)
                
                
#             plt.plot(t, c1,label=list(m2.Q_balance[I])[0],color='red')
#             plt.plot(t, c2,label=list(m2.Q_balance[I])[1],color='green')
#             plt.plot(t, c3,label=list(m2.Q_balance[I])[2],color='blue')
#             plt.xlabel('Time [h]')
#             plt.ylabel('$Concentration [kmol/m^{3}]$')
#             title=case[0]+' in '+case[1]+' Concentration'
#             plt.title(case[0]+' in '+case[1])
#             plt.legend()
#             # plt.show()
#             plt.savefig("figures/"+title+".svg") 
#             plt.clf()
#             plt.cla()
#             plt.close()

#             plt.plot(t,Tr,label='T_reactor',color='red')
#             plt.plot(t,Tj,label='T_jacket',color='blue')
#             plt.xlabel('Time [h]')
#             plt.ylabel('Temperature [K]')
#             title=case[0]+' in '+case[1]+' Temperature'
#             plt.title(case[0]+' in '+case[1])
#             plt.legend()
#             # plt.show()
#             plt.savefig("figures/"+title+".svg") 
#             plt.clf()
#             plt.cla()
#             plt.close()
            
#             plt.plot(t, Fhot,label='F_hot',color='red')
#             plt.plot(t,Fcold,label='F_cold',color='blue')
#             plt.xlabel('Time [h]')
#             plt.ylabel('Flow rate $[m^{3}/h]$')
#             title=case[0]+' in '+case[1]+' Flow rate'
#             plt.title(case[0]+' in '+case[1])
#             plt.legend()
#             # plt.show()    
#             plt.savefig("figures/"+title+".svg") 
#             plt.clf()
#             plt.cla()
#             plt.close()


####CASE STUDY 2###############################

    print('******CASE STUDY 2************')

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
    # elif minlp_solver=='OCTERACT':
    #     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    # LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    # UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    # kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME,'sequential':False}




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



    # ## RUN THIS TO SOLVE
    # m=sequential_non_iterative_2_case2(logic_fun,initialization_test,model_fun,kwargs2,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = True, global_tee= True,rel_tol = 0)
    # ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_solution')

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

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))




    # m2=model_fun(**kwargs2)
    # m2=initialize_model(m2,from_feasible=True,feasible_model='case_2_min_proc_time_solution')
    # m2.varTime.pprint()
    # ActualTPC3=sum(sum(pe.value(m.Nref[I,J])*(m2.hot_cost*pe.value(m2.Integral_hot[I, J,0][m2.N[I, J,0].last()]) + m2.cold_cost*pe.value(m2.Integral_cold[I, J,0][m2.N[I, J,0].last()])) for I in m2.I_dynamics)for J in m2.J_dynamics)
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(ActualTPC3))
    # OBJVAL=(TPC1+TPC2+ActualTPC3+TMC-SALES)
    # print('OBJ:',str(OBJVAL))
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
    # kwargs['x_initial']=Sol_found
    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential
    # m=model_fun(**kwargs)
    # m=initialize_model(m,from_feasible=True,feasible_model='case_2_sequential') 
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
    elif minlp_solver=='OCTERACT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):0.1}
    UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):3}
    kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME,'sequential':False}


    model_witn_distillation_dynamics=True

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

    # m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_solution_with_distillation')

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

    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))




    # m2=model_fun(**kwargs2)
    # m2=initialize_model(m2,from_feasible=True,feasible_model='case_2_min_proc_time_solution_with_distillation')
    # # m2.varTime.pprint()
    # cost_distillation=5/100
    # ActualTPC3=sum(sum(pe.value(m.Nref[I,J])*(m2.hot_cost*pe.value(m2.Integral_hot[I, J,0][m2.N[I, J,0].last()]) + m2.cold_cost*pe.value(m2.Integral_cold[I, J,0][m2.N[I, J,0].last()])) for I in m2.I_dynamics)for J in m2.J_dynamics)  +  sum(sum(pe.value(m.Nref[I,J])*( cost_distillation*m2.dist_models[I,J,0].I_V[m2.dist_models[I,J,0].T.last()]  )  for I in m2.I_distil)for J in m2.J_distil)
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(ActualTPC3))
    # OBJVAL=(TPC1+TPC2+ActualTPC3+TMC-SALES)
    # print('OBJ:',str(OBJVAL))
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
    # m,_=sequential_iterative_2_case2(logic_fun,initialization,model_fun,kwargs,ext_ref,provide_starting_initialization= False, subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,tee = True, global_tee= True,rel_tol = 0,dynamic_dist_model=True)
    # save=generate_initialization(m=m,model_name='case_2_sequential_with_distillation')
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
    # kwargs['x_initial']=Sol_found
    # logic_fun=problem_logic_scheduling
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
    #     # for I_J in m.I_J:
    #     #     Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


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

    print('\n-------DBD-approx solution of subproblems from infeasible-------------------------------------')



    logic_fun=problem_logic_scheduling_complete
    model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
    model_fun_scheduling=case_2_scheduling_only_lower_bound_tau
    infinity_val=1e+4
    maxiter=10000
    neighdef='2'
    kwargs['sequential']=True


    m_scheduling_only=model_fun_scheduling(**kwargs)
    sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','$offecho \n']} 
    m_scheduling_only = solve_with_minlp(m_scheduling_only,transformation='bigm',minlp='cplex',minlp_options=sub_options_cplex_Feas,timelimit=360000000,gams_output=False,tee=True,rel_tol=0)
    Sol_found=[]
    for I in m_scheduling_only.I:
        for J in m_scheduling_only.J:
            if m_scheduling_only.I_i_j_prod[I,J]==1:
                for K in m_scheduling_only.ordered_set[I,J]:
                    if round(pe.value(m_scheduling_only.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m_scheduling_only.minTau[I,J]+1)
    for I_J in m_scheduling_only.I_J:
        Sol_found.append(1+round(pe.value(m_scheduling_only.Nref[I_J])))

    print('Initialization=',Sol_found)



    initialization=Sol_found
    neigh=neighborhood_k_eq_2(len(Sol_found))




    m=model_fun(**kwargs)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)


    start=time.time()

    [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=0,new_case=True,with_distillation=model_witn_distillation_dynamics,provide_starting_initialization=False)
    
    print('Objective value: ',str(pe.value(m.obj)))
    print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    end=time.time()

    solname='case_2_dbd_with_distillation_aprox_subproblems_'+minlp_solver+'_'+neighdef+'_all_neigh_Verified_from_infeasible'
    save=generate_initialization(m=m,model_name=solname) 
    new_Sol_found=[]
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        new_Sol_found.append(K-m.minTau[I,J]+1)
    for I_J in m.I_J:
        new_Sol_found.append(1+round(pe.value(m.Nref[I_J])))
    print(new_Sol_found)
    TPC1=pe.value(m.TCP1)
    TPC2=pe.value(m.TCP2)
    TPC3=pe.value(m.TCP3)
    TMC=pe.value(m.TMC)
    SALES=pe.value(m.SALES)

    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES)) 
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

