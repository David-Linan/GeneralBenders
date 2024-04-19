import pyomo.environ as pe
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from pyomo.gdp import Disjunct, Disjunction
import logging
import os
import pandas as pd

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/')
from functions.dsda_functions import get_external_information, solve_with_minlp,solve_with_dsda,external_ref_neighborhood_general,solve_with_gdpopt,solve_subproblem,generate_initialization,initialize_model,external_ref


def introductory_robust_model2():
    m=pe.ConcreteModel(name='A survey of nonlinear robust optimization, Illustrative example')

    # VARS
    m.x1=pe.Var(within=pe.Reals,bounds=(-10,10),initialize=0)
    m.x2=pe.Var(within=pe.Reals,bounds=(-10,10),initialize=0)

    # UNCERTAINTY VARIABLES AND SET
    m.u1=pe.Var(within=pe.Reals,bounds=(-1,1),initialize=0)
    m.u2=pe.Var(within=pe.Reals,bounds=(-1,1),initialize=0)   
    
    def _const(m):
        return 2*m.u1*m.x1+2*m.u2*m.x2-(m.u1**2)-(m.u2**2)-1<=0
    m.const=pe.Constraint(rule=_const)


    m.objvar=pe.Var(within=pe.Reals)

    def _obj_def(m):
        return m.x1+m.x2<=m.objvar
    m.obj_def=pe.Constraint(rule=_obj_def)

    def _obj(m):
        return m.objvar
    m.obj=pe.Objective(rule=_obj)
    return m

def auxiliary_robust_model2():
    m=pe.ConcreteModel(name='A survey of nonlinear robust optimization, Illustrative example')

    # VARS
    m.x1=pe.Var(within=pe.Reals,bounds=(-10,10),initialize=0)
    m.x2=pe.Var(within=pe.Reals,bounds=(-10,10),initialize=0)

    # UNCERTAINTY VARIABLES AND SET
    m.u1=pe.Var(within=pe.Reals,bounds=(-1,1),initialize=0)
    m.u2=pe.Var(within=pe.Reals,bounds=(-1,1),initialize=0)   
    
    # def _const(m):
    #     return 2*m.u1*m.x1+2*m.u2*m.x2-(m.u1**2)-(m.u2**2)-1<=0
    # m.const=pe.Constraint(rule=_const)


    m.objvar=pe.Var(within=pe.Reals)

    def _obj_def(m):
        return -(2*m.u1*m.x1+2*m.u2*m.x2-(m.u1**2)-(m.u2**2)-1)<=m.objvar
    m.obj_def=pe.Constraint(rule=_obj_def)

    def _obj(m):
        return m.objvar
    m.obj=pe.Objective(rule=_obj)
    return m
def introductory_robust_model():
    m=pe.ConcreteModel(name='A survey of nonlinear robust optimization, Illustrative example')

    # VARS
    m.x1=pe.Var(within=pe.NonNegativeReals,initialize=0)
    m.x2=pe.Var(within=pe.NonNegativeReals,initialize=0)

    # UNCERTAINTY VARIABLES AND SET
    m.u=pe.Var(within=pe.Reals,bounds=(1/4,2))  
    
    def _const(m):
        return m.x1*(m.u**(1/2))-m.x2*m.u-2<=0
    m.const=pe.Constraint(rule=_const)


    m.objvar=pe.Var(within=pe.Reals)

    def _obj_def(m):
        return (m.x1-4)**2+(m.x2-1)**2<=m.objvar
    m.obj_def=pe.Constraint(rule=_obj_def)

    def _obj(m):
        return m.objvar
    m.obj=pe.Objective(rule=_obj)
    return m

def auxiliary_robust_model():
    m=pe.ConcreteModel(name='A survey of nonlinear robust optimization, Illustrative example')

    # VARS
    m.x1=pe.Var(within=pe.NonNegativeReals,initialize=0)
    m.x2=pe.Var(within=pe.NonNegativeReals,initialize=0)

    # UNCERTAINTY VARIABLES AND SET
    m.u=pe.Var(within=pe.Reals,bounds=(1/4,2))  
    
    # def _const(m):
    #     return m.x1*(m.u**(1/2))-m.x2*m.u-2<=0
    # m.const=pe.Constraint(rule=_const)


    m.objvar=pe.Var(within=pe.Reals)

    def _obj_def(m):
        return -(m.x1*(m.u**(1/2))-m.x2*m.u-2)<=m.objvar
    m.obj_def=pe.Constraint(rule=_obj_def)

    def _obj(m):
        return m.objvar
    m.obj=pe.Objective(rule=_obj)
    return m

def constraint_generator1(m,u):
    return m.x1*(u**(1/2))-m.x2*u-2<=0
def constraint_generator2(m,u):
    return 2*u[0]*m.x1+2*u[1]*m.x2-(u[0]**2)-(u[1]**2)-1<=0
if __name__ == "__main__":

    ''' 
    REF 1: Cutting-set methods for robust convex optimization with pessimizing oracles (methods)
    REF 2: A survey of nonlinear robust optimization (example) 
    '''
    
    nlp='knitro'
    vtol=1e-9 #TOLERANCE
    iterat=1000

    # SIMPLE ITERATIVE GAME
    print('***SIMPLE ZERO SUM GAME: For convex uncertainty set and constraints concave wrt u and convex wrt x')
    print('***CASE 1')
    uprev=1/4
    main=introductory_robust_model()
    aux=auxiliary_robust_model()
    print('initial u: ',uprev)
    for k in range(iterat):
        print('-----ITERATION',k)

        main.u.fix(uprev)
        main=solve_subproblem(main,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
        print('Optimal x from optimizer: x1=',pe.value(main.x1),' ,x2=',pe.value(main.x2))
        print('Objective from optimizer:',pe.value(main.objvar))

        aux.x1.fix(pe.value(main.x1))
        aux.x2.fix(pe.value(main.x2))
        aux=solve_subproblem(aux,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
        print('Worst-case u from opponent: u=',pe.value(aux.u))
        uprev=pe.value(aux.u)

        const_violation=-pe.value(aux.objvar)
        print('Constraint violation from opponent: ',const_violation)
        if const_violation<=vtol:
            break


    print('***CASE 2: THIS SIMPLE METHOD DOES NOT CONVERGE FOR THIS CASE STUDY')
    # main=introductory_robust_model2()
    # aux=auxiliary_robust_model2()
    # uprev1=-1
    # uprev2=-1
    # print('initial u1: ',uprev1,'initial u2: ',uprev2)
    # for k in range(iterat):
    #     print('-----ITERATION',k)

    #     main.u1.fix(uprev1)
    #     main.u2.fix(uprev2)
    #     main=solve_subproblem(main,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
    #     print('Optimal x from optimizer: x1=',pe.value(main.x1),' ,x2=',pe.value(main.x2))

    #     aux.x1.fix(pe.value(main.x1))
    #     aux.x2.fix(pe.value(main.x2))
    #     aux=solve_subproblem(aux,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
    #     print('Worst-case u from opponent: u1=',pe.value(aux.u1),', u2=',pe.value(aux.u2))
    #     uprev1=pe.value(aux.u1)
    #     uprev2=pe.value(aux.u2)

    #     const_violation=-pe.value(aux.objvar)
    #     print('Constraint violation from opponent: ',const_violation)
    #     if const_violation<=vtol:
    #         break

    print('***CUTTING-SET METHOD FOR PROBLEMS WITH A SINGLE CONSTRAINT')
    print('***CASE 1')

    uprev=1/4 #INITIALIZATION OF UNCERTAIN PARAMETER (ALSO CALLED u nominal)
    print('initial u: ',uprev)
    U=[uprev]


    main=introductory_robust_model()
    main.const.deactivate() #INITIALIZE AS UNCONSTRAINED PROBLEM
    main.cuts=pe.ConstraintList() #INITIALIZE LIST WITH CONSTRAINTS

    aux=auxiliary_robust_model()

    for k in range(iterat):
        print('-----ITERATION',k)
        # 1) OPTIMIZATION
        for u in U:
            main.cuts.add(constraint_generator1(main,u))
        main=solve_subproblem(main,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
        print('Optimal x from OPTIMIZATION: x1=',pe.value(main.x1),' ,x2=',pe.value(main.x2))
        print('Objective from OPTIMIZATION:',pe.value(main.objvar))
        # 2) PESSIMIZATION
        aux.x1.fix(pe.value(main.x1))
        aux.x2.fix(pe.value(main.x2))
        aux=solve_subproblem(aux,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
        #3) UPDATE UNCERTAINTY SAMPLED POINTS
        const_violation=-pe.value(aux.objvar)
        print('Constraint violation from PESSIMIZATION: ',const_violation)
        if const_violation>0:
            U.append(pe.value(aux.u))
            print('New worst case parameter from PESSIMIZATION: u=',pe.value(aux.u))
        # 4) VERIFY STOPPING CRITERION
        if const_violation<=vtol:
            break
        
    print('***CASE 2')

    uprev=[-1,-1] #INITIALIZATION OF UNCERTAIN PARAMETER (ALSO CALLED u nominal)
    print('initial u1: ',uprev[0],'initial u2: ',uprev[1])
    U=[uprev]


    main=introductory_robust_model2()
    main.const.deactivate() #INITIALIZE AS UNCONSTRAINED PROBLEM
    main.cuts=pe.ConstraintList() #INITIALIZE LIST WITH CONSTRAINTS

    aux=auxiliary_robust_model2()

    for k in range(iterat):
        print('-----ITERATION',k)
        # 1) OPTIMIZATION
        for u in U:
            main.cuts.add(constraint_generator2(main,u))
        main=solve_subproblem(main,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
        print('Optimal x from OPTIMIZATION: x1=',pe.value(main.x1),' ,x2=',pe.value(main.x2))
        print('Objective from OPTIMIZATION:',pe.value(main.objvar))
        # 2) PESSIMIZATION
        aux.x1.fix(pe.value(main.x1))
        aux.x2.fix(pe.value(main.x2))
        aux=solve_subproblem(aux,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
        #3) UPDATE UNCERTAINTY SAMPLED POINTS
        const_violation=-pe.value(aux.objvar)
        print('Constraint violation from PESSIMIZATION: ',const_violation)
        if const_violation>0:
            U.append([pe.value(aux.u1),pe.value(aux.u2)])
            print('New worst case parameter from PESSIMIZATION: u1=',pe.value(aux.u1),' u2=',pe.value(aux.u2))
        # 4) VERIFY STOPPING CRITERION
        if const_violation<=vtol:
            break
     



    # # MASTER
    # mas=introductory_robust_model()
    # mas.obj_def.deactivate()
    # mas.const.deactivate()
    # mas.cuts=pe.ConstraintList()
    
    # # SUBPROBLEM
    # sub=introductory_robust_model()
    # sub.link_u1=pe.Var(within=pe.Reals,bounds=(-1,1),initialize=0)
    # sub.link_u2=pe.Var(within=pe.Reals,bounds=(-1,1),initialize=0)
    # def _const_link_u1(sub):
    #     return sub.u1-sub.link_u1==0
    # sub.const_link_u1=pe.Constraint(rule=_const_link_u1)
    # def _const_link_u2(sub):
    #     return sub.u2-sub.link_u2==0
    # sub.const_link_u2=pe.Constraint(rule=_const_link_u2)


    # # SUBPROBLEM FOR LAGRANE MULTIPLIERS
    # sub2=sub.clone()
    # sub2.obj.deactivate()
    # def _new_bjj(m):
    #     return -m.x1-m.x2
    # sub2.new_obj=pe.Objective(rule=_new_bjj)


    # sub2.link_x1=pe.Var(within=pe.Reals,bounds=(-10,10),initialize=0)
    # sub2.link_x2=pe.Var(within=pe.Reals,bounds=(-10,10),initialize=0)
    # def _const_link_x1(sub2):
    #     return sub2.x1-sub2.link_x1==0
    # sub2.const_link_x1=pe.Constraint(rule=_const_link_x1)
    # def _const_link_x2(sub2):
    #     return sub2.x2-sub2.link_x2==0
    # sub2.const_link_x2=pe.Constraint(rule=_const_link_x2)



    # sub.dual = pe.Suffix(direction=pe.Suffix.IMPORT) #define dual variables
    # sub2.dual = pe.Suffix(direction=pe.Suffix.IMPORT) #define dual variables    
    # ######---------------------------- FEASIBILITY SUBPROBLEM --------------------------------##################
    # feas=sub.clone()
    # feas.elastic_vars={}
    # # L1-norm minimization is considered
    # count_elastic=0
    # for constr in feas.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
    #     if constr.parent_component().name != 'const_link_u1' and constr.parent_component().name != 'const_link_u2': 
    #         if constr.equality:
    #             count_elastic=count_elastic+1
    #             feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
    #             setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
    #             constr._body+=feas.elastic_vars[count_elastic]

    #             count_elastic=count_elastic+1
    #             feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
    #             setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
    #             constr._body+=-feas.elastic_vars[count_elastic]
    #         else:
    #             count_elastic=count_elastic+1
    #             if constr.has_lb():
    #                 feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
    #                 setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
    #                 constr._body+=feas.elastic_vars[count_elastic]
    #             if constr.has_ub():
    #                 feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
    #                 setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
    #                 constr._body+=-feas.elastic_vars[count_elastic]

    # feas.obj.deactivate()
    # def _obj_feas(m):
    #     return sum( feas.elastic_vars[i]  for i in feas.elastic_vars.keys())
    # feas.obj_feas = pe.Objective(rule=_obj_feas, sense=pe.minimize) 



    # # GBD algorithm
    # best_sol_name='robust_experiment'
    # nlp='conopt'
    # iterations=1000
    # Infinity_aprox=1000
    # sub.UBD=Infinity_aprox
    # mas.LBD=-Infinity_aprox
    # mas.cuts.add(-Infinity_aprox<=mas.obj)
    # start=time.time()
    # epsilon=1E-6
    # time_limit=86400
    # for k in range(iterations):
    #     print('------------------------Iteration ',str(k),'------------------------------------')
    #     #1: SOLVE MASTER PROBLEM
    #     mas=solve_subproblem(mas,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0) 
    #     mas.LBD=pe.value(mas.obj)

    #     #2: verify stopping criterion
    #     current=time.time()
    #     print('Primal obj: ',str(sub.UBD),'Master obj:', str(mas.LBD),'Current time: ',str(current-start),flush=True)

    #     if sub.UBD- mas.LBD <=epsilon:
    #         break
    #     if current-start>=time_limit:
    #         break   

    #     #3: Solve primal problem
    #     sub.link_u1.fix(pe.value(mas.u1))
    #     sub.link_u2.fix(pe.value(mas.u2))
    #     # solve primal (subprolem)
    #     sub=solve_subproblem(sub,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
    #     generate_initialization(m=sub,model_name='GBD_subproblem') #save solution, in case I need an alternative initialization for the feasibility subproblem
    #     print('Subproblem status:',sub.dsda_status,flush=True)

    #     print('x1:',pe.value(sub.x1),'x2:',pe.value(sub.x2))

    #     if sub.dsda_status=='Optimal':

    #         # sub2.link_u1.fix(pe.value(mas.u1))
    #         # sub2.link_u2.fix(pe.value(mas.u2))
    #         if pe.value(sub.x1)>=sub.x1.ub:
    #             sub2.link_x1.fix(sub.x1.ub)            
    #         elif pe.value(sub.x1)<=sub.x1.lb:
    #             sub2.link_x1.fix(sub.x1.lb)
    #         else:
    #             sub2.link_x1.fix(pe.value(sub.x1))
    #         if pe.value(sub.x2)>=sub.x2.ub:
    #             sub2.link_x2.fix(sub.x2.ub)            
    #         elif pe.value(sub.x2)<=sub.x2.lb:
    #             sub2.link_x2.fix(sub.x2.lb)
    #         else:
    #             sub2.link_x2.fix(pe.value(sub.x2))
    #         # solve primal (subprolem)
    #         sub2=solve_subproblem(sub2,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0) 

    #         print('x1:',pe.value(sub2.x1),'x2:',pe.value(sub2.x2))             
    #         current_obj=pe.value(sub2.objvar)

    #         sub.UBD_new=min([sub.UBD,current_obj])
    #         # Update the best known solution if it improved
    #         if sub.UBD_new<sub.UBD:
    #             generate_initialization(m=sub,model_name=best_sol_name)
    #         # Update subproblem solution with the best subproblem solution identified so far  
    #         sub.UBD=sub.UBD_new


    #         mas.cuts.add(sub2.dual[sub2.const_link_u1]*(mas.u1-pe.value(sub2.link_u1))+sub2.dual[sub2.const_link_u2]*(mas.u2-pe.value(sub2.link_u2))  +pe.value(sub2.objvar)<=mas.objvar)

    #         print('Optimality cut added',flush=True)
    #         mas.cuts.pprint()
    #     else:
    #         feas.link_u1.fix(pe.value(mas.u1))
    #         feas.link_u2.fix(pe.value(mas.u2))
    #         # solve primal (subprolem)
    #         feas=solve_subproblem(feas,subproblem_solver=nlp,subproblem_solver_options = {},timelimit = 86400, gams_output = False,tee = False,rel_tol = 0) 
    #         print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition,flush=True)
    #         if feas.dsda_status=='Optimal':
    #             sum_infeasibility=pe.value(feas.obj_feas)
    #             mas.cuts.add(feas.dual[feas.const_link_u1]*(mas.u1-pe.value(feas.link_u1))+feas.dual[feas.const_link_u2]*(mas.u2-pe.value(feas.link_u2)) +sum_infeasibility<=0)
    #             print('Feasibility cut added',flush=True)
    #         else:
    #             print('GBD solver failure: subproblem detected as infeasible, and fatal error with feasibility stage',flush=True)
    #             break 