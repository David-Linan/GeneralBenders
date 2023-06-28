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
    Infinity_aprox=100000


    if minlp_solver=='dicopt' or minlp_solver=='DICOPT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='OCTERACT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','Option SOLVER = OCTERACT;','$onecho > octeract.opt \n','LOCAL_SEARCH true\n','$offecho \n']}
    
    kwargs={}


    print('\n-------GENERALIZED BENDERS TEST-------------------------------------')
    kwargs2=kwargs.copy()

    logic_fun=problem_logic_scheduling_case1
    model_fun=scheduling_and_control_gdp_N_GBD
    ###### MATER PROBLEM ##################
    mas=model_fun(**kwargs2) #master problem
    mas.cuts=pe.ConstraintList() #Initialize benders cuts

    for I in mas.I_reactions:
        for J in mas.J_reactors:
            mas.c_dCdtheta[I,J].deactivate()
            mas.c_dTRdtheta[I,J].deactivate()                        
            mas.c_dTJdtheta[I,J].deactivate()
            mas.c_dIntegral_hotdtheta[I,J].deactivate()
            mas.c_dIntegral_colddtheta[I,J].deactivate()
            mas.Constant_control1[I,J].deactivate()                        
            mas.Constant_control2[I,J].deactivate()
            mas.finalCon[I,J].deactivate()
            mas.finalTemp[I,J].deactivate()

    # for I_J in mas.I_J:
    #     I=I_J[0]
    #     J=I_J[1] 
    #     for disj in mas.ordered_set2[I,J]:
    #         disjunct=mas.YR2_Disjunct[I,J][disj]
    #         for constr in disjunct.component_objects(pe.Constraint, descend_into=True):
    #             if constr.local_name=='finalCon' or constr.local_name=='finalTemp':
    #                 constr.deactivate()
    mas.C_TCP3.deactivate()

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(mas)
    transformation_string = 'gdp.' + transform
    pe.TransformationFactory(transformation_string).apply_to(mas)

    ###### SUBPROBLEM ##################
    sub=model_fun(**kwargs2) #subproblem
    sub.E2_CAPACITY_LOW.deactivate()
    sub.E2_CAPACITY_UP.deactivate()
    sub.E3_BALANCE_INIT.deactivate()
    sub.E_DEMAND_SATISFACTION.deactivate()
    sub.E1_UNIT.deactivate()
    sub.DEF_AUX1_INDEP.deactivate()
    sub.E3_BALANCE.deactivate()
    sub.DEF_AUX2_INDEP.deactivate()

    for I in sub.I_reactions:
        for J in sub.J_reactors:
            for disj in sub.ordered_set[I,J]:
                disjunct=sub.YR_disjunct[I,J][disj]
                
                for constr in disjunct.component_objects(pe.Constraint, descend_into=True):
                    if constr.local_name=='DEF_VAR_TIME' or constr.local_name=='DEF_AUX1' or constr.local_name=='DEF_AUX2': #NOTE that I am deactivating everythin. I am just using this to show that I can compare by name to deactivate what I want
                        constr.deactivate()

            sub.YR_disjunct[I,J].deactivate()
            sub.Disjunction1[I,J].deactivate()
            sub.oneYR[I,J].deactivate()

    sub.X_Z_relation.deactivate()

    # for I_J in sub.I_J:
    #     I=I_J[0]
    #     J=I_J[1] 
    #     for disj in sub.ordered_set2[I,J]:
    #         disjunct=sub.YR2_Disjunct[I,J][disj]
    #         for constr in disjunct.component_objects(pe.Constraint, descend_into=True):
    #             if constr.local_name=='DEF_Nref':
    #                 constr.deactivate()

    #     sub.oneYR2[I,J].deactivate()
    #     sub.Disjunction2[I,J].deactivate()


    sub.obj.deactivate()
    def _obj_dynamic(m):
        return m.TCP3
    sub.obj_dyn = pe.Objective(rule=_obj_dynamic, sense=pe.minimize) 

    #linking variables and constraints in subproblem:
    sub.link_B=pe.Var(sub.I,sub.J,sub.T,within=pe.NonNegativeReals)
    sub.link_varTime=pe.Var(sub.I_reactions,sub.J_reactors,within=pe.NonNegativeReals)

    def _const_link_B(sub,I,J,T):
        return sub.B[I,J,T]-sub.link_B[I,J,T]==0
    sub.const_link_B=pe.Constraint(sub.I,sub.J,sub.T,rule=_const_link_B)

    def _const_link_VarTime(sub,I,J):
        return sub.varTime[I,J]-sub.link_varTime[I,J]==0
    sub.const_link_VarTime=pe.Constraint(sub.I_reactions,sub.J_reactors,rule=_const_link_VarTime)

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(sub)
    transformation_string = 'gdp.' + transform
    pe.TransformationFactory(transformation_string).apply_to(sub)
    # Dual variables
    sub.dual = pe.Suffix(direction=pe.Suffix.IMPORT) #define dual variables


    ###### FEASIBILITY SUBPROBLEM ##################
    feas=sub.clone()
    sum_infeasibility=0  #sum of infeasibility
    feas.elastic_vars={}

    count_elastic=0
    for constr in feas.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
        if constr.parent_component().name != 'const_link_B' and constr.parent_component().name != 'const_link_VarTime': 
            if constr.equality:
                count_elastic=count_elastic+1
                feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                constr._body+=feas.elastic_vars[count_elastic]

                count_elastic=count_elastic+1
                feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                constr._body+=-feas.elastic_vars[count_elastic]
            else:
                count_elastic=count_elastic+1
                if constr.has_lb():
                    feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                    setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                    constr._body+=feas.elastic_vars[count_elastic]
                if constr.has_ub():
                    feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                    setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                    constr._body+=-feas.elastic_vars[count_elastic]
        # constr.pprint()
    feas.obj_dyn.deactivate()
    def _obj_feas(m):
        return sum( feas.elastic_vars[i]  for i in feas.elastic_vars.keys())
    feas.obj_feas = pe.Objective(rule=_obj_feas, sense=pe.minimize)     
    # sub.obj_feas.pprint()

    #TESTS-----------------------------------------------------------------------------------------------------------------------------------------------------
    mas=initialize_model(mas,from_feasible=True,feasible_model='case_1_scheduling_and_dynamics_solution') #THIS IS THE SEQUENTIAL NAIVE APPROACH, WHIH RETURNS THE SAME SOLUTION AS THE SEQUENTIAL ITERATIVE APPROACH IN THIS CASE
    for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
        if v.name=='varTime' or v.name=='B':
            for index in v:
                if index==None:
                    v.fix(pe.value(v))
                else:
                    v[index].fix(pe.value(v[index]))

    mas=solve_with_minlp(mas,transformation='',minlp='cplex',minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
    print(pe.value(mas.obj))
    # mas.varTime.pprint()
    # mas.B.pprint()

    # fix variables in subproblem 
    for I in sub.I:
        for J in sub.J:
            for T in sub.T:
                sub.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))
                sub.X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

    for I_J in sub.I_J:
            I=I_J[0]
            J=I_J[1]
            sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

    for K in sub.K:
        for T in sub.T:
            sub.S[K,T].fix(pe.value(mas.S[K,T]))

    for I in sub.I_reactions:
        for J in sub.J_reactors:
            sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  


    sub=solve_subproblem(sub,subproblem_solver='conopt4',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
    #print dual info
    # for c in sub.component_objects(pe.Constraint, active=True):
    #     if c.name=='const_link_B' or c.name=='const_link_VarTime':
    #         for index in c:
    #             if index==None:
    #                 print(c.body)
    #                 print(c.name,index,str(sub.dual[c]))
    #             else:
    #                 print(c[index].body)
    #                 print(c.name,index,str(sub.dual[c[index]]))   
    
    if sub.dsda_status=='Optimal':
        mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  sub.dual[sub.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(sub.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
        print('Optimality cut to initialize: ')
        mas.cuts.pprint()
        
        TPC1=pe.value(sub.TCP1)
        TPC2=pe.value(sub.TCP2)
        TPC3=pe.value(sub.TCP3)
        TMC=pe.value(sub.TMC)
        SALES=pe.value(sub.SALES)
        sub.UBD=TPC1+TPC2+TPC3+TMC-SALES
    else:

        #TODO: rewrite!!
        mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  sub.dual[sub.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(sub.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)<=0)

        print('Feasibility cut to initialie:')
        mas.cuts.pprint()
        sub.UBD=Infinity_aprox

    # make sure that variables in master are left unfixed
    for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
        if v.name=='varTime' or v.name=='B':
            for index in v:
                if index==None:
                    v.unfix()
                else:
                    v[index].unfix()

    # END OF TEST--------------------------------------------------------------------------------------------------------------------------------------------




# GENERALIZED BENDERS DECOMPOSITION-------------------------------------------------------------------------------------------------------------------------

    start=time.time()
    max_iter=1000
    epsilon=0
    tol=1e-6 #feasibility tolearance
    for k in range(max_iter):
        print('------------------------Iteration ',str(k),'------------------------------------')
    #1: solve the master problem
        mas=solve_with_minlp(mas,transformation='',minlp='cplex',minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
        mas.LBD=pe.value(mas.obj)

    #2: verify stopping criterion
        current=time.time()
        print('Primal obj: ',str(sub.UBD),'Master obj:', str(mas.LBD),'Current time: ',str(current-start))
        if sub.UBD- mas.LBD <=epsilon:
            break

    #3: Solve primal problem
        # fix variables in subproblem 
        for I in sub.I:
            for J in sub.J:
                for T in sub.T:
                    sub.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))
                    sub.X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

        for I_J in sub.I_J:
                I=I_J[0]
                J=I_J[1]
                sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

        for K in sub.K:
            for T in sub.T:
                sub.S[K,T].fix(pe.value(mas.S[K,T]))

        for I in sub.I_reactions:
            for J in sub.J_reactors:
                sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))    

        # solve primal (subprolem)
        sub=solve_subproblem(sub,subproblem_solver='conopt4',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
        print('Subproblem status:',sub.dsda_status)
        if sub.dsda_status=='Optimal':
            
            
            
            TPC1=pe.value(sub.TCP1)
            TPC2=pe.value(sub.TCP2)
            TPC3=pe.value(sub.TCP3)
            TMC=pe.value(sub.TMC)
            SALES=pe.value(sub.SALES)

            sub.UBD=min([sub.UBD,TPC1+TPC2+TPC3+TMC-SALES])

            # if sub.UBD- mas.LBD <=epsilon:
            #     break

            mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  sub.dual[sub.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(sub.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
            print('Optimaliti cut: ')
            mas.cuts.pprint()
            
        else:

            # sum_infeasibility=0  #sum of infeasibility
            # # infeasible_const=[]  #infeasible constraints

            # for constr in sub.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
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

            #     # output_dict = dict(name=constr.name)
            #     # infeasible_const.append(output_dict)
            
            # sum_infeasibility=0
            # for c in sub.component_objects(pe.Constraint, active=True):
            #     # print(c.local_name)
            #     if c.local_name != 'const_link_B' and c.local_name != 'const_link_VarTime':
            #         for index in c:
            #             if index==None:
            #                 # print(c.body)
            #                 # print(c.name,index,str(sub.dual[c]))
            #                 sum_infeasibility=sum_infeasibility+sub.dual[c]*pe.value(c.body)
            #             else:
            #                 # print(c[index].body)
            #                 # print(c.name,index,str(sub.dual[c[index]]))   
            #                 sum_infeasibility=sum_infeasibility+sub.dual[c[index]]*pe.value(c[index].body)






            # fix variables in subproblem 
            for I in feas.I:
                for J in feas.J:
                    for T in feas.T:
                        feas.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))
                        feas.X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

            for I_J in feas.I_J:
                    I=I_J[0]
                    J=I_J[1]
                    feas.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

            for K in feas.K:
                for T in feas.T:
                    feas.S[K,T].fix(pe.value(mas.S[K,T]))

            for I in feas.I_reactions:
                for J in feas.J_reactors:
                    feas.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  


            feas=initialize_model(feas,from_feasible=True,feasible_model='case_1_scheduling_and_dynamics_solution')# TODO: added this because I found an error!!!
            feas=solve_subproblem(feas,subproblem_solver='conopt4',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
            print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
            if feas.dsda_status=='Optimal':
                sum_infeasibility=pe.value(feas.obj_feas)
                mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  feas.dual[feas.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(feas.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                print('Feasibility cut:')
                mas.cuts.pprint()
            else:
                print('Problem with feasibility stage')
                break
















# END OF GENERALIZED BENDERS DECOMPOSITION---------------------------------------------------------------------------------------------------------------------





    ## RUN THIS TO RETRIEVE SOLUTION    

    # m=initialize_model(m,from_feasible=True,feasible_model='case_1_generalized_benders_Test')









    Sol_found=[]
    for I in sub.I_reactions:
        for J in sub.J_reactors:
            if sub.I_i_j_prod[I,J]==1:
                for K in sub.ordered_set[I,J]:
                    if round(pe.value(sub.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-sub.minTau[I,J]+1)
    for I_J in sub.I_J:
        Sol_found.append(1+round(pe.value(sub.Nref[I_J])))
    print('EXT_VARS_FOUND',Sol_found)
    TPC1=pe.value(sub.TCP1)
    TPC2=pe.value(sub.TCP2)
    TPC3=pe.value(sub.TCP3)
    TMC=pe.value(sub.TMC)
    SALES=pe.value(sub.SALES)
    OBJ_FOUND=TPC1+TPC2+TPC3+TMC-SALES

    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJECTIVE:',str(OBJ_FOUND))



