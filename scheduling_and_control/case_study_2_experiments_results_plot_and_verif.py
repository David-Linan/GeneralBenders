from __future__ import division
from pickle import TRUE

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min_ref_2
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_minlp
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_all,neighborhood_k_eq_l_natural,neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
from case_study_2_model import case_2_scheduling_control_gdp_var_proc_time,problem_logic_scheduling,case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential,case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
import os
import matplotlib.pyplot as plt
from math import fabs

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

###############################################################################
#########--------------base case ------------------############################
###############################################################################
###############################################################################

    obj_Selected='profit_max'


    # initialization=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    initialization=[1, 1, 1, 1, 1, 1, 1, 1]

    # # DICOPT SOLUTION   
    mip_solver='CPLEX'
    minlp_solver='DICOPT'
    nlp_solver='conopt4'
    transform='bigm'
    last_disc=15
    last_time_h=5
    logic_fun=problem_logic_scheduling
    with_distillation=True
    sequential_naive=False #true if i am ploting results from sequential naive

    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=2;','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}

    sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=7;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    if not with_distillation:
        LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
        UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    else:
        LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):0.1}
        UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):3}        
    kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME,'sequential':False}

    print('\n-------DICOPT-------------------------------------')
    if not with_distillation:
        model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential
    else:
        model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
    m=model_fun(**kwargs)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    # save=generate_initialization(m=m,model_name='case_study_2_opt_dicopt') 
    end=time.time()    
    # UPDATE MODEL WITH INITIALZIATION
    m=model_fun(**kwargs)
    # Transformation step
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # transformation_string = 'gdp.'+transform
    # pe.TransformationFactory(transformation_string).apply_to(m)



    
    # feasible_mod_name2='case_2_scheduling_solution'  #sequential naive: scheduling solution
    # feasible_mod_name='case_2_min_proc_time_solution'     #sequential naive: minimum processing time solution

    # feasible_mod_name='case_2_sequential' #sequential iterative

    # feasible_mod_name='case_2_dsda_DICOPT_2_all_neigh_Verified_all_vars_from_naive'    #DSDA from sequential naive

    # feasible_mod_name= 'case_2_dsda_DICOPT_2_all_neigh_Verified_all_vars'  #DSDA from sequential iterative


    #with distillation model
    # feasible_mod_name='case_2_sequential_With_distillation' #SEQUENTIAL ITERATIVE

    feasible_mod_name='case_2_dbd_with_distillation_aprox_subproblems_DICOPT_2_all_neigh_Verified'


    # feasible_mod_name2='case_2_scheduling_solution_with_distillation'  #sequential naive: scheduling solution
    # feasible_mod_name='case_2_min_proc_time_solution_with_distillation'     #sequential naive: minimum processing time solution
    m=initialize_model(m,from_feasible=True,feasible_model=feasible_mod_name) 

    Sol_found=[]
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m.minTau[I,J]+1)
    for I_J in m.I_J:
        Sol_found.append(1+round(pe.value(m.Nref[I_J])))

    # m = external_ref(m=m,x=Sol_found,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)


    # tol=1e-6
    # sum_infeasibility=0  #sum of infeasibility
    # infeasible_const=[]  #infeasible constraints

    # for constr in m.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
    #     constr_body_value = pe.value(constr.body, exception=False)
    #     constr_lb_value = pe.value(constr.lower, exception=False)
    #     constr_ub_value = pe.value(constr.upper, exception=False)

    #     constr_undefined = False
    #     equality_violated = False
    #     lb_violated = False
    #     ub_violated = False

    #     if constr_body_value is None:
    #         # Undefined constraint body value due to missing variable value
    #         constr_undefined = True
    #         pass
    #     else:
    #         # Check for infeasibilities
    #         if constr.equality:
    #             if fabs(constr_lb_value - constr_body_value) >= tol:
    #                 equality_violated = True
    #                 sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)
    #         else:
    #             if constr.has_lb() and constr_lb_value - constr_body_value >= tol:
    #                 lb_violated = True
    #                 sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)
    #             if constr.has_ub() and constr_body_value - constr_ub_value >= tol:
    #                 ub_violated = True
    #                 sum_infeasibility=sum_infeasibility+fabs(constr_body_value - constr_ub_value)
    #     if not any((constr_undefined, equality_violated, lb_violated, ub_violated)):
    #         # constraint is fine. skip to next constraint
    #         continue

    #     output_dict = dict(name=constr.name)
    #     infeasible_const.append(output_dict)
    # print('If the following list is empty, then everything is feasible')
    # print(infeasible_const)

    Sol_found=[]
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m.minTau[I,J]+1)
    for I_J in m.I_J:
        Sol_found.append(1+round(pe.value(m.Nref[I_J])))

    textbuffer = io.StringIO()
    for v in m.component_objects(pe.Var, descend_into=True):
        v.pprint(textbuffer)
        textbuffer.write('\n')
    textbuffer.write('\n Objective: \n') 
    textbuffer.write(str(pe.value(m.obj)))  
    file_name=feasible_mod_name+'.txt'  
    with open(os.path.join('C:/Users/dlinanro/Desktop/GeneralBenders/scheduling_and_control',file_name), 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())
    if not sequential_naive:
        print('Objective =',pe.value(m.obj),'best =',Sol_found,'cputime =',str(end-start))


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
        if with_distillation:
            cost_distillation=5/100
            DISTIl_COST=sum(sum(sum(pe.value(m.X[I, J, T])*( cost_distillation*pe.value(m.dist_models[I,J,T].I_V[m.dist_models[I,J,T].T.last()])  ) for T in m.T) for I in m.I_distil)for J in m.J_distil)
            print('TPC: Variable cost for unit-tasks that do consider dynamics, distillation only:', str(DISTIl_COST))

# SOLUTION USING DSDA
    # print('\n-------DSDA-------------------------------------')
    # infinity_val=1e+2
    # maxiter=10000
    # neighdef='2'
    # logic_fun=problem_logic_scheduling
    # model_fun=case_2_scheduling_control_gdp_var_proc_time
    # kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME}
    # start=time.time()
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,initialization,ext_ref,logic_fun,k = neighdef,provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= True,global_tee = True,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=False)
    # end=time.time()
    # print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  
    # TPC1=pe.value(D_SDAsol.TCP1)
    # TPC2=pe.value(D_SDAsol.TCP2)
    # TPC3=pe.value(D_SDAsol.TCP3)
    # TMC=pe.value(D_SDAsol.TMC)
    # SALES=pe.value(D_SDAsol.SALES)
    # OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJ:',str(OBJVAL))

#######-------plots------------------------
    if not with_distillation:
        if sequential_naive:
            for I in m.I_dynamics:
                for J in m.J_dynamics:
                    for T in m.T:
                        if pe.value(m.X[I,J,T])==1: 
                            case=(I,J,T)
                            t=[]
                            CA=[]
                            CB=[]
                            CC=[]
                            Tr=[]
                            Tj=[]
                            Fhot=[]
                            Fcold=[]
                            u_input=[]
                            for N in m.N[case]:
                                t.append(N*m.varTime[case].value)
                                Tr.append(m.TRvar[case][N].value)
                                Tj.append(m.TJvar[case][N].value)
                                Fhot.append(m.Fhot[case][N].value)
                                Fcold.append(m.Fcold[case][N].value)
                                CA.append( m.CA[case][N].value)
                                CB.append( m.CB[case][N].value)
                                CC.append( m.CC[case][N].value)
                                u_input.append(m.u_input[case][N].value)
                                
                                
                            plt.plot(t, CA,label='CA',color='red')
                            plt.plot(t, CB,label='CB',color='green')
                            plt.plot(t, CC,label='CC',color='blue')
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

                            plt.plot(t, u_input,color='red')
                            plt.xlabel('Time [h]')
                            plt.ylabel('Flow rate of B $[m^{3}/h]$')
                            title=case[0]+' in '+case[1]+' Flow rate of B'
                            plt.title(case[0]+' in '+case[1])
                            # plt.show()    
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()
        else: 
            for I in m.I_dynamics:
                for J in m.J_dynamics:
                    for T in m.T:
                        if pe.value(m.X[I,J,T])==1: 
                            case=(I,J,T)
                            t=[]
                            CA=[]
                            CB=[]
                            CC=[]
                            Tr=[]
                            Tj=[]
                            Fhot=[]
                            Fcold=[]
                            u_input=[]
                            for N in m.N[case]:
                                t.append(N*m.varTime[case].value)
                                Tr.append(m.TRvar[case][N].value)
                                Tj.append(m.TJvar[case][N].value)
                                Fhot.append(m.Fhot[case][N].value)
                                Fcold.append(m.Fcold[case][N].value)
                                CA.append( m.CA[case][N].value)
                                CB.append( m.CB[case][N].value)
                                CC.append( m.CC[case][N].value)
                                u_input.append(m.u_input[case][N].value)
                                
                                
                            plt.plot(t, CA,label='CA',color='red')
                            plt.plot(t, CB,label='CB',color='green')
                            plt.plot(t, CC,label='CC',color='blue')
                            plt.xlabel('Time [h]')
                            plt.ylabel('$Concentration [kmol/m^{3}]$')
                            title=case[0]+' in '+case[1]+' Concentration'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
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
                            title=case[0]+' in '+case[1]+' Temperature'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
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
                            title=case[0]+' in '+case[1]+' Flow rate'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            plt.legend()
                            # plt.show()    
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

                            plt.plot(t, u_input,color='red')
                            plt.xlabel('Time [h]')
                            plt.ylabel('Flow rate of B $[m^{3}/h]$')
                            title=case[0]+' in '+case[1]+' Flow rate of B'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            # plt.show()    
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

    else:
        if sequential_naive:
            for I in m.I_dynamics:
                for J in m.J_dynamics:
                    for T in m.T:
                        if pe.value(m.X[I,J,T])==1: 
                            case=(I,J,T)
                            t=[]
                            CA=[]
                            CB=[]
                            CC=[]
                            Tr=[]
                            Tj=[]
                            Fhot=[]
                            Fcold=[]
                            u_input=[]
                            for N in m.N[case]:
                                t.append(N*m.varTime[case].value)
                                Tr.append(m.TRvar[case][N].value)
                                Tj.append(m.TJvar[case][N].value)
                                Fhot.append(m.Fhot[case][N].value)
                                Fcold.append(m.Fcold[case][N].value)
                                CA.append( m.CA[case][N].value)
                                CB.append( m.CB[case][N].value)
                                CC.append( m.CC[case][N].value)
                                u_input.append(m.u_input[case][N].value)
                                
                                
                            plt.plot(t, CA,label='CA',color='red')
                            plt.plot(t, CB,label='CB',color='green')
                            plt.plot(t, CC,label='CC',color='blue')
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

                            plt.plot(t, u_input,color='red')
                            plt.xlabel('Time [h]')
                            plt.ylabel('Flow rate of B $[m^{3}/h]$')
                            title=case[0]+' in '+case[1]+' Flow rate of B'
                            plt.title(case[0]+' in '+case[1])
                            # plt.show()    
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()
            for I in m.I_distil:
                for J in m.J_distil:
                    for T in m.T:
                        if pe.value(m.X[I,J,T])==1: 
                            case=(I,J,T)
                            tdist=[]
                            Dist=[]
                            Boil_up=[]
                            Reflux_rate=[]
                            x_instantaneous=[]
                            x_accumulated=[]
                            Reboiler_hold_up=[]
                            Product_accumulated=[]

                            for N in m.dist_models[case].T:
                                tdist.append(N*m.varTime[case].value)
                                Dist.append(m.dist_models[case].D[N].value)
                                Boil_up.append(m.dist_models[case].V[N].value)
                                Reflux_rate.append(m.dist_models[case].R[N].value)
                                x_instantaneous.append(m.dist_models[case].x[m.dist_models[case].N.last(), N].value)
                                x_accumulated.append(m.dist_models[case].xd_average[N].value)
                                Reboiler_hold_up.append(m.dist_models[case].HB[N].value)
                                Product_accumulated.append(m.dist_models[case].I2[N].value)

                            plt.plot(tdist, x_instantaneous,label='Mole fraction of desired product (distillate)',color='red')
                            plt.plot(tdist,  x_accumulated,label='Mole fraction of desired product (reciever)',color='green')
                            plt.xlabel('Time [h]')
                            plt.ylabel('$Mole fraction [kmol/kmol]$')
                            title=case[0]+' in '+case[1]+' Mole fraction'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            plt.legend()
                            # plt.show()
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

                            ax1 = plt.subplot()
                            l1=ax1.plot(tdist,Boil_up,label='Boil-up rate',color='red')
                            ax2=ax1.twinx()
                            l2=ax2.plot(tdist,Reflux_rate,label='Reflux rate',color='blue')
                            plt.xlabel('Time [h]')
                            ax1.set_ylabel('$Boil-up rate [m^{3}/h]$',color='red')
                            ax2.set_ylabel('$Reflux rate [m^{3}/h]$',color='blue')                            
                            title=case[0]+' in '+case[1]+' Boil-up and reflux'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            # plt.legend()
                            # plt.show()
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

                            plt.plot(tdist, Reboiler_hold_up,label='Reboiler level',color='red')
                            plt.plot(tdist,  Product_accumulated,label='Reciever level',color='green')
                            plt.xlabel('Time [h]')
                            plt.ylabel('$Volume [m^{3}]$')
                            title=case[0]+' in '+case[1]+' distillation levels'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            plt.legend()
                            # plt.show()
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()
        else: 
            for I in m.I_dynamics:
                for J in m.J_dynamics:
                    for T in m.T:
                        if pe.value(m.X[I,J,T])==1: 
                            case=(I,J,T)
                            t=[]
                            CA=[]
                            CB=[]
                            CC=[]
                            Tr=[]
                            Tj=[]
                            Fhot=[]
                            Fcold=[]
                            u_input=[]
                            for N in m.N[case]:
                                t.append(N*m.varTime[case].value)
                                Tr.append(m.TRvar[case][N].value)
                                Tj.append(m.TJvar[case][N].value)
                                Fhot.append(m.Fhot[case][N].value)
                                Fcold.append(m.Fcold[case][N].value)
                                CA.append( m.CA[case][N].value)
                                CB.append( m.CB[case][N].value)
                                CC.append( m.CC[case][N].value)
                                u_input.append(m.u_input[case][N].value)
                                
                                
                            plt.plot(t, CA,label='CA',color='red')
                            plt.plot(t, CB,label='CB',color='green')
                            plt.plot(t, CC,label='CC',color='blue')
                            plt.xlabel('Time [h]')
                            plt.ylabel('$Concentration [kmol/m^{3}]$')
                            title=case[0]+' in '+case[1]+' Concentration'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
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
                            title=case[0]+' in '+case[1]+' Temperature'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
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
                            title=case[0]+' in '+case[1]+' Flow rate'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            plt.legend()
                            # plt.show()    
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

                            plt.plot(t, u_input,color='red')
                            plt.xlabel('Time [h]')
                            plt.ylabel('Flow rate of B $[m^{3}/h]$')
                            title=case[0]+' in '+case[1]+' Flow rate of B'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            # plt.show()    
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()       
            for I in m.I_distil:
                for J in m.J_distil:
                    for T in m.T:
                        if pe.value(m.X[I,J,T])==1: 
                            case=(I,J,T)
                            tdist=[]
                            Dist=[]
                            Boil_up=[]
                            Reflux_rate=[]
                            x_instantaneous=[]
                            x_accumulated=[]
                            Reboiler_hold_up=[]
                            Product_accumulated=[]

                            for N in m.dist_models[case].T:
                                tdist.append(N*m.varTime[case].value)
                                Dist.append(m.dist_models[case].D[N].value)
                                Boil_up.append(m.dist_models[case].V[N].value)
                                Reflux_rate.append(m.dist_models[case].R[N].value)
                                x_instantaneous.append(m.dist_models[case].x[m.dist_models[case].N.last(), N].value)
                                x_accumulated.append(m.dist_models[case].xd_average[N].value)
                                Reboiler_hold_up.append(m.dist_models[case].HB[N].value)
                                Product_accumulated.append(m.dist_models[case].I2[N].value)

                            plt.plot(tdist, x_instantaneous,label='Mole fraction of desired product (distillate)',color='red')
                            plt.plot(tdist,  x_accumulated,label='Mole fraction of desired product (reciever)',color='green')
                            plt.xlabel('Time [h]')
                            plt.ylabel('$Mole fraction [kmol/kmol]$')
                            title=case[0]+' in '+case[1]+' Mole fraction'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            plt.legend()
                            # plt.show()
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

                            ax1 = plt.subplot()
                            l1=ax1.plot(tdist,Boil_up,label='Boil-up rate',color='red')
                            ax2=ax1.twinx()
                            l2=ax2.plot(tdist,Reflux_rate,label='Reflux rate',color='blue')
                            plt.xlabel('Time [h]')
                            ax1.set_ylabel('$Boil-up rate [m^{3}/h]$',color='red')
                            ax2.set_ylabel('$Reflux rate [m^{3}/h]$',color='blue')                            
                            title=case[0]+' in '+case[1]+' Boil-up and reflux'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            # plt.legend()
                            # plt.show()
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

                            plt.plot(tdist, Reboiler_hold_up,label='Reboiler level',color='red')
                            plt.plot(tdist,  Product_accumulated,label='Reciever level',color='green')
                            plt.xlabel('Time [h]')
                            plt.ylabel('$Volume [m^{3}]$')
                            title=case[0]+' in '+case[1]+' distillation levels'+' at '+str(m.t_p[T])+' h'
                            plt.title(case[0]+' in '+case[1]+' at '+str(m.t_p[T])+' h')
                            plt.legend()
                            # plt.show()
                            plt.savefig("figures/"+title+".svg") 
                            plt.clf()
                            plt.cla()
                            plt.close()

    if not with_distillation:
        if sequential_naive:
            model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential
            m=model_fun(**kwargs)
            m=initialize_model(m,from_feasible=True,feasible_model=feasible_mod_name2) 

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
            gnt.set_ylim(8, 52) #TODO: change depending case study
            
            # Setting X-axis limits
            gnt.set_xlim(0, m.lastT.value*m.delta.value)
            
            # Setting labels for x-axis and y-axis
            gnt.set_xlabel('Time [h]')
            gnt.set_ylabel('Units')
            
            # Setting ticks on y-axis
            gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
            # Labelling tickes of y-axis
            gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
            
            
            # Setting graph attribute
            gnt.grid(False)
            
            # Declaring bars in schedule
            height=9
            already_used=[]
            for j in m.J:

                if j=='U1':
                    lower_y_position=40    
                elif j=='U2':
                    lower_y_position=30    
                elif j=='U3':
                    lower_y_position=20
                elif j=='U4':
                    lower_y_position=10
                for i in m.I:
                    if i=='T1':
                        bar_color='tab:red'
                    elif i=='T2':
                        bar_color='tab:green'    
                    elif i=='T3':
                        bar_color='tab:blue'    
                    elif i=='T4':
                        bar_color='tab:orange' 
                    elif i=='T5':
                        bar_color='tab:olive'
                    for t in m.T:
                        try:
                            if i in m.I_dynamics and j in m.J_dynamics:
                                if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                    already_used.append(i)
                                elif round(pe.value(m.X[i,j,t]))==1:
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
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


        else:

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
            gnt.set_ylim(8, 52) #TODO: change depending case study
            
            # Setting X-axis limits
            gnt.set_xlim(0, m.lastT.value*m.delta.value)
            
            # Setting labels for x-axis and y-axis
            gnt.set_xlabel('Time [h]')
            gnt.set_ylabel('Units')
            
            # Setting ticks on y-axis
            gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
            # Labelling tickes of y-axis
            gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
            
            
            # Setting graph attribute
            gnt.grid(False)
            
            # Declaring bars in schedule
            height=9
            already_used=[]
            for j in m.J:

                if j=='U1':
                    lower_y_position=40    
                elif j=='U2':
                    lower_y_position=30    
                elif j=='U3':
                    lower_y_position=20
                elif j=='U4':
                    lower_y_position=10
                for i in m.I:
                    if i=='T1':
                        bar_color='tab:red'
                    elif i=='T2':
                        bar_color='tab:green'    
                    elif i=='T3':
                        bar_color='tab:blue'    
                    elif i=='T4':
                        bar_color='tab:orange' 
                    elif i=='T5':
                        bar_color='tab:olive'
                    for t in m.T:
                        try:
                            if i in m.I_dynamics and j in m.J_dynamics:
                                if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                    already_used.append(i)
                                elif round(pe.value(m.X[i,j,t]))==1:
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
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

    else:

        if sequential_naive:
            model_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
            m=model_fun(**kwargs)
            m=initialize_model(m,from_feasible=True,feasible_model=feasible_mod_name2) 
            textbuffer = io.StringIO()
            for v in m.component_objects(pe.Var, descend_into=True):
                v.pprint(textbuffer)
                textbuffer.write('\n')
            textbuffer.write('\n Objective: \n') 
            textbuffer.write(str(pe.value(m.obj)))  
            file_name=feasible_mod_name2+'.txt'  
            with open(os.path.join('C:/Users/dlinanro/Desktop/GeneralBenders/scheduling_and_control',file_name), 'w') as outputfile:
                outputfile.write(textbuffer.getvalue())
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
            gnt.set_ylim(8, 52) #TODO: change depending case study
            
            # Setting X-axis limits
            gnt.set_xlim(0, m.lastT.value*m.delta.value)
            
            # Setting labels for x-axis and y-axis
            gnt.set_xlabel('Time [h]')
            gnt.set_ylabel('Units')
            
            # Setting ticks on y-axis
            gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
            # Labelling tickes of y-axis
            gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
            
            
            # Setting graph attribute
            gnt.grid(False)
            
            # Declaring bars in schedule
            height=9
            already_used=[]
            for j in m.J:

                if j=='U1':
                    lower_y_position=40    
                elif j=='U2':
                    lower_y_position=30    
                elif j=='U3':
                    lower_y_position=20
                elif j=='U4':
                    lower_y_position=10
                for i in m.I:
                    if i=='T1':
                        bar_color='tab:red'
                    elif i=='T2':
                        bar_color='tab:green'    
                    elif i=='T3':
                        bar_color='tab:blue'    
                    elif i=='T4':
                        bar_color='tab:orange' 
                    elif i=='T5':
                        bar_color='tab:olive'
                    for t in m.T:
                        try:
                            if i in m.I_dynamics and j in m.J_dynamics:
                                if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                    already_used.append(i)
                                elif round(pe.value(m.X[i,j,t]))==1:
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
                            elif i in m.I_distil and j in m.J_distil:
                                if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                    already_used.append(i)
                                elif round(pe.value(m.X[i,j,t]))==1:
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                                                          
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


        else:

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
            gnt.set_ylim(8, 52) #TODO: change depending case study
            
            # Setting X-axis limits
            gnt.set_xlim(0, m.lastT.value*m.delta.value)
            
            # Setting labels for x-axis and y-axis
            gnt.set_xlabel('Time [h]')
            gnt.set_ylabel('Units')
            
            # Setting ticks on y-axis
            gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
            # Labelling tickes of y-axis
            gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
            
            
            # Setting graph attribute
            gnt.grid(False)
            
            # Declaring bars in schedule
            height=9
            already_used=[]
            for j in m.J:

                if j=='U1':
                    lower_y_position=40    
                elif j=='U2':
                    lower_y_position=30    
                elif j=='U3':
                    lower_y_position=20
                elif j=='U4':
                    lower_y_position=10
                for i in m.I:
                    if i=='T1':
                        bar_color='tab:red'
                    elif i=='T2':
                        bar_color='tab:green'    
                    elif i=='T3':
                        bar_color='tab:blue'    
                    elif i=='T4':
                        bar_color='tab:orange' 
                    elif i=='T5':
                        bar_color='tab:olive'
                    for t in m.T:
                        try:
                            if i in m.I_dynamics and j in m.J_dynamics:
                                if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                    already_used.append(i)
                                elif round(pe.value(m.X[i,j,t]))==1:
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
                            elif i in m.I_distil and j in m.J_distil:
                                if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                    already_used.append(i)
                                elif round(pe.value(m.X[i,j,t]))==1:
                                    gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                                                          
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


###############################################################################
#########--------------loop over time discretization ------------------########
###############################################################################
###############################################################################
    # obj_Selected='profit_max'


    # # initialization=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    # initialization=[1, 1, 1, 1, 1, 1, 1, 1]

    # # # DICOPT SOLUTION   
    # mip_solver='CPLEX'
    # minlp_solver='DICOPT'
    # nlp_solver='conopt4'
    # transform='bigm'
    # last_time_h=10
    # # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=2;','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}

    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=7;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    # UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}



    # for kb in range(2,100):
    #     last_disc=kb
    #     print('\n----------------------------------------')
    #     print('--NUMBER OF DISCRETIZATION POINTS '+str(kb))

    #     print('\n-------DICOPT-------------------------------------')
    #     m=case_2_scheduling_control_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=last_disc,last_time_hours=last_time_h,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
    #     ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    #     # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    #     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    #     start=time.time()
    #     m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0)
    #     # save=generate_initialization(m=m,model_name='case_study_2_opt_dicopt') 
    #     end=time.time()    
    #     # UPDATE MODEL WITH INITIALZIATION
    #     # m=case_2_scheduling_control_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=last_disc,last_time_hours=last_time_h,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
    #     # Transformation step
    #     # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    #     # transformation_string = 'gdp.'+transform
    #     # pe.TransformationFactory(transformation_string).apply_to(m)
    #     # m=initialize_model(m,from_feasible=True,feasible_model='case_study_2_opt_dicopt')   

    #     Sol_found=[]
    #     for I in m.I:
    #         for J in m.J:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_found.append(K-m.minTau[I,J]+1)
    #     for I_J in m.I_J:
    #         Sol_found.append(1+round(pe.value(m.Nref[I_J])))

    #         textbuffer = io.StringIO()
    #         for v in m.component_objects(pe.Var, descend_into=True):
    #             v.pprint(textbuffer)
    #             textbuffer.write('\n')
    #         textbuffer.write('\n Objective: \n') 
    #         textbuffer.write(str(pe.value(m.obj)))  
    #         file_name='Case_2_results_var_proc_time_dicopt.txt'  
    #         with open(os.path.join('C:/Users/dlinanro/Desktop/GeneralBenders/scheduling_and_control',file_name), 'w') as outputfile:
    #             outputfile.write(textbuffer.getvalue())
    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_found,'cputime DICOPT=',str(end-start))


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


    # # SOLUTION USING DSDA
    #     print('\n-------DSDA-------------------------------------')
    #     infinity_val=1e+2
    #     maxiter=10000
    #     neighdef='2'
    #     logic_fun=problem_logic_scheduling
    #     model_fun=case_2_scheduling_control_gdp_var_proc_time
    #     kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME}
    #     start=time.time()
    #     D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,initialization,ext_ref,logic_fun,k = neighdef,provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = False,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=False)
    #     end=time.time()
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
    #     m=D_SDAsol
    #     for I in m.I_dynamics:
    #         for J in m.J_dynamics:
    #             for K in m.ordered_set[I,J]:
    #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                     variable_bound_found=K*m.delta
    #             print((I,J))
    #             if sum(pe.value(m.X[I,J,T]) for T in m.T)>=1:
    #                 print('maximum variable time [h]: ',max(pe.value(m.varTime[I,J,T]) for T in m.T if round(pe.value(m.X[I,J,T]))==1),'<= current bound (based on discretization) [h]: ',variable_bound_found,'<= initial bound (user) [h]: ',UP_PROC_TIME[(I,J)])

