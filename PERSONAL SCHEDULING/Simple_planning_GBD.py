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

def problem_logic(m):
    logic_expr = []
    for index in m.ordered:
        for i in m.i:
            logic_expr.append([m.YR[index,i],m.YR_Disjunct[index,i].indicator_var])              

    return logic_expr
def planning_model(reformulation: bool=False):
    m=pe.ConcreteModel(name='Production planning by mixed integer programming, Section 1.2')

    # SETS

    m.NI=pe.Param(initialize=12,doc='Number of products')
    m.NK=pe.Param(initialize=3,doc='Number of machines')
    m.NT=pe.Param(initialize=15,doc='Number of time periods')

    m.i=pe.RangeSet(1,m.NI,1,doc='Set of finished products')
    m.t=pe.RangeSet(1,m.NT,1,doc='Set of time periods')
    m.k=pe.Set(initialize=[1,2,3], doc='Machines. 1: mixing, 2: Cereal packaging, 3: Fruit packaging')
    m.F2=pe.Set(initialize=[i for i in m.i if i<=6],within=m.i)
    m.F3=pe.Set(initialize=[i for i in m.i if i>6],within=m.i)

    # PARAMETERS
    
    D_data=np.array([[0, 95, 110, 96, 86,124, 83,108, 114,121, 110,124, 104, 86, 87],
     [98, 96, 96, 98, 103,104, 122,101, 89,108, 101,109, 106,108, 76],
     [106, 0, 89,123, 96,105, 83, 82, 112,109, 119, 85, 99, 80, 123],
     [98,121, 0,105, 98, 96, 101, 81, 117, 76, 103, 81, 95,105, 102],
     [0,124, 113,123, 123, 79, 111, 98, 97, 80, 98,124, 78,108, 109],
     [103,102, 0, 95, 107,105, 107,105, 75, 93, 115,113, 111,105, 85],
     [110, 93, 0,112, 84,124, 98,101, 83, 87, 105,118, 115,106, 78],
     [85, 92, 101,110, 93, 96, 120,109, 121, 87, 92, 85, 91, 93, 109],
     [122,116, 109, 0, 105,108, 88, 98, 77, 90, 110,102, 107, 99, 96],
     [120,124, 94,105, 92, 86, 101,106, 75,109, 83, 95, 79,108, 100],
     [117, 96, 78, 0, 108, 87, 114,107, 110, 94, 104,101, 108,110, 80],
     [125,112, 75, 0, 116,103, 122, 88, 85, 84, 76,102, 84, 88, 82]])
    def _D(m,i,t):
        return D_data[i-1,t-1]
    m.D=pe.Param(m.i,m.t,initialize=_D,doc='Forecast demand for item i in period t')


    def _SS(m,i,t):
        if i in m.F2:
            return 10
        else:
            return 20
    m.SS=pe.Param(m.i,m.t,initialize=_SS,doc='Safety stock of item i needed at the end of period t')
    _SS0={}
    _SS0[1]=83
    _SS0[2]=31
    _SS0[3]=11
    _SS0[4]=93
    _SS0[5]=82
    _SS0[6]=72
    _SS0[7]=23
    _SS0[8]=91
    _SS0[9]=83
    _SS0[10]=34
    _SS0[11]=61
    _SS0[12]=82
    m.SS0=pe.Param(m.i,initialize=_SS0,doc='Initial stock of item i')

    m.alpha=pe.Param(m.i,m.k, initialize=1,doc='capacity consumed on machine k to produce one unit of product i')
    _beta={}
    _beta[1]=30
    _beta[2]=20
    _beta[3]=30
    _beta[4]=40
    _beta[5]=40
    _beta[6]=10
    _beta[7]=30
    _beta[8]=20
    _beta[9]=10
    _beta[10]=50
    _beta[11]=30
    _beta[12]=20
    m.beta=pe.Param(m.i,initialize=_beta,doc='Mixing capacity consumed per cleaning operation at the end of a batch of product i')

    m.L=pe.Param(m.k,initialize={1:1400,2:700,3:700},doc='Capacity available on machine k in each time period')
    def _M(m,i,t):
        if i in m.F2:
            return min([sum(m.D[i,l] for l in m.t if l>=t)+m.SS[i,t],(m.L[1]-m.beta[i])/(m.alpha[i,1]),m.L[2]/m.alpha[i,2]])
        else:
            return min([sum(m.D[i,l] for l in m.t if l>=t)+m.SS[i,t],(m.L[1]-m.beta[i])/(m.alpha[i,1]),m.L[3]/m.alpha[i,3]])
    m.M=pe.Param(m.i,m.t,initialize=_M,doc='Big-M')

    # VARIABLES

    m.x=pe.Var(m.i,m.t,within=pe.NonNegativeReals,initialize=0,doc='amount of product i produced during time period t')
    m.s=pe.Var(m.i,m.t,within=pe.NonNegativeReals,initialize=0,doc='inventory level of product i at the end of time period t')
    m.y=pe.Var(m.i,m.t,within=pe.Binary,initialize=0,doc='1 if there is cleaning operation because production of a batch of product i')

    # CONSTRAINTS
    def _dem_sat1(m,i,t):
        if t!=m.t.first():
            return m.s[i,t]>=m.SS[i,t]
        else:
            return pe.Constraint.Skip
    m.dem_sat1=pe.Constraint(m.i,m.t,rule=_dem_sat1)

    def _dem_sat2(m,i,t):
        if t==m.t.first():
            return m.SS0[i]+m.x[i,t]==m.D[i,t]+m.s[i,t] 
        else:
            return m.s[i,m.t.prev(t)]+m.x[i,t]==m.D[i,t]+m.s[i,t] 
    m.dem_sat2=pe.Constraint(m.i,m.t,rule=_dem_sat2)

    def _vub(m,i,t):
        return m.x[i,t]<=m.M[i,t]*m.y[i,t] 
    m.vub=pe.Constraint(m.i,m.t,rule=_vub)

    def _mix_cap(m,t):
        return sum(m.alpha[i,1]*m.x[i,t]  for i in m.i)+sum( m.beta[i]*m.y[i,t]    for i in m.i)<=m.L[1]
    m.mix_cap=pe.Constraint(m.t,rule=_mix_cap)
    def _pack_cap(m,t,k):
        if k==2:
            return sum(  m.alpha[i,k]*m.x[i,t] for i in m.F2)<=m.L[k]
        elif k==3:
            return sum(  m.alpha[i,k]*m.x[i,t] for i in m.F3)<=m.L[k]
        else:
            return pe.Constraint.Skip 
    m.pack_cap=pe.Constraint(m.t,m.k,rule=_pack_cap)



    m.TCP3=pe.Var(within=pe.Reals)

    def _obf_Defin(m):
        return m.TCP3== sum(   sum(     m.s[i,t]       for t in m.t)         for i in m.i)
    m.C_TCP3=pe.Constraint(rule=_obf_Defin)


    def _obj(m):
        return m.TCP3 
    m.obj=pe.Objective(sense=pe.minimize,rule=_obj)


    # REFORMULATION

    m.tp=pe.RangeSet(1,m.NT,1,doc='Set of time periods (clone)')
    m.P=pe.Var(m.i,m.tp,within=pe.Integers,bounds=(0,m.NT),initialize=0)
    m.yp=pe.Var(m.i,m.t,m.tp,within=pe.Binary,initialize=0)
    m.yp0=pe.Var(m.i,m.tp,within=pe.Binary,initialize=0)

    # m.N=pe.Var(m.i,within=pe.Integers,bounds=(0,m.NT))

    # def _DefN(m,i):
    #     return m.N[i]==sum(m.y[i,t] for t in m.t)
    # m.def_N=pe.Constraint(m.i,rule=_DefN)

    def _Defyp(m,i,t):
        return m.y[i,t]==sum(  m.yp[i,t,tp]     for tp in m.tp)
    m.def_yp=pe.Constraint(m.i,m.t,rule=_Defyp)

    # NOTE: THESE WILL BE UPDATED EVERY TIME THE SUBPROBLEM IS SOLVED
    def _DefP(m,i,tprime):
        return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in m.t)
    m.def_P=pe.Constraint(m.i,m.tp,rule=_DefP)

    def _DefP2(m,i,tprime):
        return 1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in m.t)
    m.def_P2=pe.Constraint(m.i,m.tp,rule=_DefP2)
    
    if reformulation:
        # REFORMULATION VARIABLES




        # DISJUNCTIONS
        m.ordered=pe.RangeSet(0,m.NT)
        m.YR=pe.BooleanVar(m.ordered,m.i,initialize=False)

        def _select_one(m,i):
            return pe.exactly(1,[m.YR[index,i] for index in m.ordered])
        m.oneYR=pe.LogicalConstraint(m.i,rule=_select_one) 
        

        def _build_YR_Disjunct(m,index,i):

            if index==0:
                def _C1_P(m,tprime):
                    return m.model().P[i,tprime]==0
                m.C_1P=pe.Constraint(m.model().tp,rule=_C1_P)

            elif index==1:
                def _C1_P(m,tprime):
                    if tprime>=index+1:
                        return m.model().P[i,tprime]==0
                    elif tprime==1:
                        return m.model().P[i,tprime]>=1
                m.C_1P=pe.Constraint(m.model().tp,rule=_C1_P)
            else:
                def _C1_P(m,tprime):
                    if tprime>=index+1:
                        return m.model().P[i,tprime]==0
                    elif tprime==1:
                        return m.model().P[i,tprime]>=1
                    else:
                        return m.model().P[i,tprime]>=m.model().P[i,m.model().t.prev(tprime)]+1
                m.C_1P=pe.Constraint(m.model().tp,rule=_C1_P)

        m.YR_Disjunct=Disjunct(m.ordered,m.i,rule=_build_YR_Disjunct)


        # Create disjunction
        def _Disjunction(m,i):   
            return [m.YR_Disjunct[index,i] for index in m.ordered]
        m.Disjunction=Disjunction(m.i,rule=_Disjunction,xor=True)


        # Associate disjuncts with boolean variables
        for index in m.ordered:
            for i in m.i:
                m.YR[index,i].associate_binary_var(m.YR_Disjunct[index,i].indicator_var)


    return m
def planning_model_original():
    m=pe.ConcreteModel(name='Production planning by mixed integer programming, Section 1.2')

    # SETS

    m.NI=pe.Param(initialize=12,doc='Number of products')
    m.NK=pe.Param(initialize=3,doc='Number of machines')
    m.NT=pe.Param(initialize=15,doc='Number of time periods')

    m.i=pe.RangeSet(1,m.NI,1,doc='Set of finished products')
    m.t=pe.RangeSet(1,m.NT,1,doc='Set of time periods')
    m.k=pe.Set(initialize=[1,2,3], doc='Machines. 1: mixing, 2: Cereal packaging, 3: Fruit packaging')
    m.F2=pe.Set(initialize=[i for i in m.i if i<=6],within=m.i)
    m.F3=pe.Set(initialize=[i for i in m.i if i>6],within=m.i)

    # PARAMETERS
    
    D_data=np.array([[0, 95, 110, 96, 86,124, 83,108, 114,121, 110,124, 104, 86, 87],
     [98, 96, 96, 98, 103,104, 122,101, 89,108, 101,109, 106,108, 76],
     [106, 0, 89,123, 96,105, 83, 82, 112,109, 119, 85, 99, 80, 123],
     [98,121, 0,105, 98, 96, 101, 81, 117, 76, 103, 81, 95,105, 102],
     [0,124, 113,123, 123, 79, 111, 98, 97, 80, 98,124, 78,108, 109],
     [103,102, 0, 95, 107,105, 107,105, 75, 93, 115,113, 111,105, 85],
     [110, 93, 0,112, 84,124, 98,101, 83, 87, 105,118, 115,106, 78],
     [85, 92, 101,110, 93, 96, 120,109, 121, 87, 92, 85, 91, 93, 109],
     [122,116, 109, 0, 105,108, 88, 98, 77, 90, 110,102, 107, 99, 96],
     [120,124, 94,105, 92, 86, 101,106, 75,109, 83, 95, 79,108, 100],
     [117, 96, 78, 0, 108, 87, 114,107, 110, 94, 104,101, 108,110, 80],
     [125,112, 75, 0, 116,103, 122, 88, 85, 84, 76,102, 84, 88, 82]])
    def _D(m,i,t):
        return D_data[i-1,t-1]
    m.D=pe.Param(m.i,m.t,initialize=_D,doc='Forecast demand for item i in period t')


    def _SS(m,i,t):
        if i in m.F2:
            return 10
        else:
            return 20
    m.SS=pe.Param(m.i,m.t,initialize=_SS,doc='Safety stock of item i needed at the end of period t')
    _SS0={}
    _SS0[1]=83
    _SS0[2]=31
    _SS0[3]=11
    _SS0[4]=93
    _SS0[5]=82
    _SS0[6]=72
    _SS0[7]=23
    _SS0[8]=91
    _SS0[9]=83
    _SS0[10]=34
    _SS0[11]=61
    _SS0[12]=82
    m.SS0=pe.Param(m.i,initialize=_SS0,doc='Initial stock of item i')

    m.alpha=pe.Param(m.i,m.k, initialize=1,doc='capacity consumed on machine k to produce one unit of product i')
    _beta={}
    _beta[1]=30
    _beta[2]=20
    _beta[3]=30
    _beta[4]=40
    _beta[5]=40
    _beta[6]=10
    _beta[7]=30
    _beta[8]=20
    _beta[9]=10
    _beta[10]=50
    _beta[11]=30
    _beta[12]=20
    m.beta=pe.Param(m.i,initialize=_beta,doc='Mixing capacity consumed per cleaning operation at the end of a batch of product i')

    m.L=pe.Param(m.k,initialize={1:1400,2:700,3:700},doc='Capacity available on machine k in each time period')
    def _M(m,i,t):
        if i in m.F2:
            return min([sum(m.D[i,l] for l in m.t if l>=t)+m.SS[i,t],(m.L[1]-m.beta[i])/(m.alpha[i,1]),m.L[2]/m.alpha[i,2]])
        else:
            return min([sum(m.D[i,l] for l in m.t if l>=t)+m.SS[i,t],(m.L[1]-m.beta[i])/(m.alpha[i,1]),m.L[3]/m.alpha[i,3]])
    m.M=pe.Param(m.i,m.t,initialize=_M,doc='Big-M')

    # VARIABLES

    m.x=pe.Var(m.i,m.t,within=pe.NonNegativeReals,initialize=0,doc='amount of product i produced during time period t')
    m.s=pe.Var(m.i,m.t,within=pe.NonNegativeReals,initialize=0,doc='inventory level of product i at the end of time period t')
    m.y=pe.Var(m.i,m.t,within=pe.Binary,initialize=0,doc='1 if there is cleaning operation because production of a batch of product i')

    # CONSTRAINTS
    def _dem_sat1(m,i,t):
        if t!=m.t.first():
            return m.s[i,t]>=m.SS[i,t]
        else:
            return pe.Constraint.Skip
    m.dem_sat1=pe.Constraint(m.i,m.t,rule=_dem_sat1)

    def _dem_sat2(m,i,t):
        if t==m.t.first():
            return m.SS0[i]+m.x[i,t]==m.D[i,t]+m.s[i,t] 
        else:
            return m.s[i,m.t.prev(t)]+m.x[i,t]==m.D[i,t]+m.s[i,t] 
    m.dem_sat2=pe.Constraint(m.i,m.t,rule=_dem_sat2)

    def _vub(m,i,t):
        return m.x[i,t]<=m.M[i,t]*m.y[i,t] 
    m.vub=pe.Constraint(m.i,m.t,rule=_vub)

    def _mix_cap(m,t):
        return sum(m.alpha[i,1]*m.x[i,t]  for i in m.i)+sum( m.beta[i]*m.y[i,t]    for i in m.i)<=m.L[1]
    m.mix_cap=pe.Constraint(m.t,rule=_mix_cap)
    def _pack_cap(m,t,k):
        if k==2:
            return sum(  m.alpha[i,k]*m.x[i,t] for i in m.F2)<=m.L[k]
        elif k==3:
            return sum(  m.alpha[i,k]*m.x[i,t] for i in m.F3)<=m.L[k]
        else:
            return pe.Constraint.Skip 
    m.pack_cap=pe.Constraint(m.t,m.k,rule=_pack_cap)



    m.TCP3=pe.Var(within=pe.Reals)

    def _obf_Defin(m):
        return m.TCP3== sum(   sum(     m.s[i,t]       for t in m.t)         for i in m.i)
    m.C_TCP3=pe.Constraint(rule=_obf_Defin)


    def _obj(m):
        return m.TCP3 
    m.obj=pe.Objective(sense=pe.minimize,rule=_obj)

    return m
def solve_scheduling(
        model_fun,
        solver,
        best_sol_name,
        relaxed: bool=True,
        initialize_with_master: bool=True,
        init_name: str='',
        max_iter: float= 100000,
        epsilon: float=1e-5,
        time_limit: float=86400,
        no_good_cuts: bool=False,
        mip_solver: str='xpress',
        nlp_solver: str='conopt4',
        minlp_solver: str='dicopt',
        transform: str='bigm',
        Infinity_aprox: float=100000,
):

    """
    Args:
        model_fun=scheduling_and_control
        solver='GBD' # GBD or MINLP
        auxiliary_cuts=True # If proposed auxiliary feasibility cuts are going to be used. Select type below
        auxiliary_ctus_type='maxb' #'mint, maxb, all'   
        initial_cuts=1 # To decide the type of initial cuts to be used
            1: Naive execution of GBD
            2: Execution of GBD with initial cuts: minimum processing time s.t. variable capacity
            3: Execution of GBD with initial cuts: minimum processing time s.t. fixed capacity at its maximum
        best_sol_name='sol_test' # Name of the file to save solution of experiment


        relaxed=True #True: start with relaxed MINLP, False: Start with feasible initialization given by "init_name"
        initialize_with_master=True #If problem is solved by simply solving the master problem without cuts first. If True, then relaxed option will not affect the solution of the problem
        init_name='sequential_iterative_init_for_V5' # Used if relaxed is False and initialize_with_master is false
        max_iter=100000 # maximum number of iterations
        epsilon=1e-6 # GBD tolerance
        time_limit=86400 #seconds  
        no_good_cuts=False # If no good cuts are going to be used
        mip_solver='CPLEX'
        nlp_solver='conopt4'
        minlp_solver='dicopt'
        transform='bigm'
        Infinity_aprox=100000
        kwargs={}    
        
    Returns:
        LB_list=[]
        UB_list=[]
        Time_list=[]
        Iter_list=[]
    """



    print('\n-------GENERALIZED BENDERS DECOMPOSITION TEST-------------------------------------',flush=True)
    ######-------------------SOLVER AND MODEL DECLARATION -------------------------##################
    LB_list=[]
    UB_list=[]
    Time_list=[]
    Iter_list=[]
    
    if solver=='GBD':
        sub_options={'add_options':['GAMS_MODEL.threads=0;']} #Subproblem solver options: use all available threads
        ######-------------------------- MATER PROBLEM ------------------------------##################

        mas=model_fun(reformulation=True) #master problem
        mas.cuts=pe.ConstraintList() #Initialize benders cuts

        # MASTER PROBLEM VARIABLES: YR, P
        #Deactivate subproblem constraint

        mas.mix_cap.deactivate()
        mas.pack_cap.deactivate()
        mas.dem_sat1.deactivate()
        mas.dem_sat2.deactivate()
        mas.vub.deactivate()

        mas.def_P.deactivate()
        mas.def_P2.deactivate()
        mas.def_yp.deactivate()

        mas.C_TCP3.deactivate()

        pe.TransformationFactory('core.logical_to_linear').apply_to(mas)
        transformation_string = 'gdp.'+transform
        pe.TransformationFactory(transformation_string).apply_to(mas)
        ######---------------------------- SUBPROBLEM --------------------------------##################
        sub=model_fun(reformulation=False) #subproblem
        
        # SUBPROBLEM VARIABLES: x,s,y,yp,yp0, P
        #Deactivate master problem constraints
        # NOTE: these will be re-updated every time the subproblem is solved, hence they are deactivated. Note that these can be clasified as linking constraints
        # sub.def_P.deactivate()
        # sub.def_P2.deactivate()

        # re-define objective function
        sub.obj.deactivate()
        def _obj_dynamic(m):
            return m.TCP3
        sub.obj_dyn = pe.Objective(rule=_obj_dynamic, sense=pe.minimize) 

        #linking variables and constraints in subproblem:
        sub.link_P=pe.Var(sub.i,sub.tp,within=pe.NonNegativeReals)
        # Change domain of complicating variables in subproblems.Note that we also change the domain of y and yp
        sub.P.domain=pe.NonNegativeReals
        sub.y.domain=pe.NonNegativeReals
        sub.yp.domain=pe.NonNegativeReals
        sub.yp0.domain=pe.NonNegativeReals
        sub.yp.setlb(0)
        sub.yp.setub(1)
        sub.yp0.setlb(0)
        sub.yp0.setub(1)

        def _const_link_P(sub,i,tp):
            return sub.P[i,tp]-sub.link_P[i,tp]==0
        sub.const_link_P=pe.Constraint(sub.i,sub.tp,rule=_const_link_P)

        # Dual variables (Lagrange multipliers)
        sub.dual = pe.Suffix(direction=pe.Suffix.IMPORT) #define dual variables


        ######---------------------------- FEASIBILITY SUBPROBLEM --------------------------------##################
        feas=sub.clone()
        feas.elastic_vars={}
        # L1-norm minimization is considered
        count_elastic=0
        for constr in feas.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
            if constr.parent_component().name != 'const_link_P': 
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

        feas.obj_dyn.deactivate()
        def _obj_feas(m):
            return sum( feas.elastic_vars[i]  for i in feas.elastic_vars.keys())
        feas.obj_feas = pe.Objective(rule=_obj_feas, sense=pe.minimize)     

        ######---------------------------- GBD INITIALIZATION STRATEGY --------------------------------##################
        start=time.time()
        if not initialize_with_master:
            if relaxed:
                relm=model_fun(reformulation=True)
                pe.TransformationFactory('core.logical_to_linear').apply_to(relm)
                transformation_string = 'gdp.'+transform
                pe.TransformationFactory(transformation_string).apply_to(relm)
                for v in relm.component_data_objects(ctype=pe.Var,descend_into=True,active=True):
                    v.domain=pe.NonNegativeReals

                relm=solve_subproblem(relm,subproblem_solver=mip_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = True,rel_tol = 0)
                generate_initialization(m=relm,model_name='relaxed')
                mas=initialize_model(mas,from_feasible=True,feasible_model='relaxed')
                mas.cuts.add(pe.value(relm.TCP3)<=mas.TCP3)
            else: 
                mas=initialize_model(mas,from_feasible=True,feasible_model=init_name) #Initialization from solution that is known to be feasible

        ######---------------------------- GENERALIZED BENDERS DECOMPOSIION ALGORITHM --------------------------------##################

        sub.updated_def_P={}
        sub.updated_def_P2={}
        sub.updated_def_P3={}
        feas.updated_def_P={}
        feas.updated_def_P2={}
        feas.updated_def_P3={}
        sub.UBD=Infinity_aprox
        mas.LBD=-Infinity_aprox
        for k in range(max_iter):
            print('------------------------Iteration ',str(k),'------------------------------------',flush=True)
            if k>=1 or initialize_with_master:
                #1: solve the master problem
                mas=solve_with_minlp(mas,transformation='',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
                mas.LBD=pe.value(mas.obj)
                # mas.P.pprint()

                if no_good_cuts:
                #1.1.: add no-good cuts (cuts that avoid repeating combination of binary variables)
                    expr=0
                    for v in mas.component_data_objects(ctype=pe.Var,descend_into=True,active=True):
                        if v.is_binary() and int(round(pe.value(v)))==int(1):
                            expr+=v-1
                        elif v.is_binary() and int(round(pe.value(v)))==int(0):
                            expr+=-v          
                    
                    mas.cuts.add( expr<= -1)

        #2: verify stopping criterion
            current=time.time()
            print('Primal obj: ',str(sub.UBD),'Master obj:', str(mas.LBD),'Current time: ',str(current-start),flush=True)
            LB_list.append(mas.LBD)
            UB_list.append(sub.UBD)
            Time_list.append(current-start)
            Iter_list.append(k)
            if sub.UBD- mas.LBD <=epsilon:
                break
            if current-start>=time_limit:
                break

        #3: Solve primal problem
            # fix variables in subproblem 
            for i in sub.i:
                for tp in sub.tp:
                    if pe.value(mas.P[i,tp])>=mas.P[i,tp].ub:
                        sub.link_P[i,tp].fix(mas.P[i,tp].ub)  
                    elif pe.value(mas.P[i,tp])<=mas.P[i,tp].lb:
                        sub.link_P[i,tp].fix(mas.P[i,tp].lb)  
                    else:
                        sub.link_P[i,tp].fix(round(pe.value(mas.P[i,tp])))   


            if k!=0:
                sub.updated_def_P[k-1].deactivate()
                sub.updated_def_P2[k-1].deactivate()
                sub.updated_def_P3[k-1].deactivate()

            def _DefP(m,i,tprime):
                if round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb):
                    return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))+1])
                elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb)+1:
                    return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1])
                elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].ub):
                    return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime]))])
                else:
                    return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1])
            sub.updated_def_P[k]=pe.Constraint(sub.i,sub.tp,rule=_DefP)
            setattr(sub,'updated_def_P_%s' %str(k),sub.updated_def_P[k])


            def _DefP2(m,i,tprime):
                if round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb):
                    return 1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))+1])
                elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb)+1:
                    return  1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1]) 
                elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].ub):
                    return 1==sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime]))]) 
                else:
                    return 1==sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1]) 
            sub.updated_def_P2[k]=pe.Constraint(sub.i,sub.tp,rule=_DefP2)
            setattr(sub,'updated_def_P2_%s' %str(k),sub.updated_def_P2[k])

            def _DefP3(m,i,tprime):
                if round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb):
                    return pe.Constraint.Skip
                elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb)+1:
                    return m.yp0[i,tprime]*m.yp[i,round(pe.value(mas.P[i,tprime]))+1,tprime]<=0
                elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].ub):
                    return pe.Constraint.Skip
                else:
                    return m.yp[i,round(pe.value(mas.P[i,tprime]))-1,tprime]*m.yp[i,round(pe.value(mas.P[i,tprime]))+1,tprime]<=0
            sub.updated_def_P3[k]=pe.Constraint(sub.i,sub.tp,rule=_DefP3)
            setattr(sub,'updated_def_P3_%s' %str(k),sub.updated_def_P3[k])



            # solve primal (subprolem)
            sub=solve_subproblem(sub,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = True,rel_tol = 0)     
            generate_initialization(m=sub,model_name='GBD_subproblem') #save solution, in case I need an alternative initialization for the feasibility subproblem
            print('Subproblem status:',sub.dsda_status,flush=True)

            if sub.dsda_status=='Optimal':
                
                TPC3=pe.value(sub.TCP3)

                if k>=1 or (not relaxed) or initialize_with_master:
                    sub.UBD_new=min([sub.UBD,TPC3])
                    # Update the best known solution if it improved
                    if sub.UBD_new<sub.UBD:
                        generate_initialization(m=sub,model_name=best_sol_name)
                    # Update subproblem solution with the best subproblem solution identified so far  
                    sub.UBD=sub.UBD_new


                mas.cuts.add(sum(sum( sub.dual[sub.const_link_P[i,tp]]*(mas.P[i,tp]-pe.value(sub.link_P[i,tp])) for i in mas.i)  for tp in mas.tp)+pe.value(sub.TCP3)<=mas.TCP3)
                # for i in mas.i:
                #     for tp in mas.tp:
                #         mas.cuts.add(sub.dual[sub.const_link_P[i,tp]]*(mas.P[i,tp]-pe.value(sub.link_P[i,tp]))+pe.value(sub.TCP3)<=mas.TCP3)
                print('Optimality cut added',flush=True)
                # mas.cuts.pprint()
            else:
                # Naive feasibility cuts
                # fix variables in subproblem 
                for i in feas.i:
                    for tp in feas.tp:
                        if pe.value(mas.P[i,tp])>=mas.P[i,tp].ub:
                            feas.link_P[i,tp].fix(mas.P[i,tp].ub)  
                        elif pe.value(mas.P[i,tp])<=mas.P[i,tp].lb:
                            feas.link_P[i,tp].fix(mas.P[i,tp].lb)  
                        else:
                            feas.link_P[i,tp].fix(round(pe.value(mas.P[i,tp])))   


                if k!=0:
                    for gg in feas.updated_def_P.keys(): 
                        feas.updated_def_P[gg].deactivate()
                        feas.updated_def_P2[gg].deactivate()
                        feas.updated_def_P3[gg].deactivate()

                def _DefP(m,i,tprime):
                    if round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb):
                        return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))+1])
                    elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb)+1:
                        return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1])
                    elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].ub):
                        return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime]))])
                    else:
                        return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1])
                feas.updated_def_P[k]=pe.Constraint(feas.i,feas.tp,rule=_DefP)
                setattr(feas,'updated_def_P_%s' %str(k),feas.updated_def_P[k])


                def _DefP2(m,i,tprime):
                    if round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb):
                        return 1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))+1])
                    elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb)+1:
                        return  1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1]) 
                    elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].ub):
                        return 1==sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime]))]) 
                    else:
                        return 1==sum(m.yp[i,t,tprime] for t in [round(pe.value(mas.P[i,tprime]))-1,round(pe.value(mas.P[i,tprime])),round(pe.value(mas.P[i,tprime]))+1]) 
                feas.updated_def_P2[k]=pe.Constraint(feas.i,feas.tp,rule=_DefP2)
                setattr(feas,'updated_def_P2_%s' %str(k),feas.updated_def_P2[k])

                def _DefP3(m,i,tprime):
                    if round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb):
                        return pe.Constraint.Skip
                    elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].lb)+1:
                        return m.yp0[i,tprime]*m.yp[i,round(pe.value(mas.P[i,tprime]))+1,tprime]<=0
                    elif round(pe.value(mas.P[i,tprime]))==round(mas.P[i,tprime].ub):
                        return pe.Constraint.Skip
                    else:
                        return m.yp[i,round(pe.value(mas.P[i,tprime]))-1,tprime]*m.yp[i,round(pe.value(mas.P[i,tprime]))+1,tprime]<=0
                feas.updated_def_P3[k]=pe.Constraint(feas.i,feas.tp,rule=_DefP3)
                setattr(feas,'updated_def_P3_%s' %str(k),feas.updated_def_P3[k])


                feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition,flush=True)
                if feas.dsda_status=='Optimal':
                    sum_infeasibility=pe.value(feas.obj_feas)
                    mas.cuts.add(sum(sum( feas.dual[feas.const_link_P[i,tp]]*(mas.P[i,tp]-pe.value(feas.link_P[i,tp])) for i in mas.i)  for tp in mas.tp)+sum_infeasibility<=0)
                    # for i in mas.i:
                    #     for tp in mas.tp:
                    #         mas.cuts.add(feas.dual[feas.const_link_P[i,tp]]*(mas.P[i,tp]-pe.value(feas.link_P[i,tp])) +sum_infeasibility<=0)
                    print('Feasibility cut added',flush=True)
                    # mas.cuts[k+2].pprint()
                else:
                    print('Problem with feasibility stage. Trying a different initialization',flush=True)
                    feas=initialize_model(feas,from_feasible=True,feasible_model='GBD_subproblem')
                    feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                    print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition,flush=True)
                    if feas.dsda_status=='Optimal':
                        sum_infeasibility=pe.value(feas.obj_feas)
                        mas.cuts.add(sum(sum( feas.dual[feas.const_link_P[i,tp]]*(mas.P[i,tp]-pe.value(feas.link_P[i,tp])) for i in mas.i)  for tp in mas.tp)+sum_infeasibility<=0)
                        # for i in mas.i:
                        #     for tp in mas.tp:
                        #         mas.cuts.add(feas.dual[feas.const_link_P[i,tp]]*(mas.P[i,tp]-pe.value(feas.link_P[i,tp])) +sum_infeasibility<=0)
                        print('Feasibility cut added',flush=True)
                        # mas.cuts[k+2].pprint()
                    else:
                        print('GBD solver failure: subproblem detected as infeasible, and fatal error with feasibility stage',flush=True)
                        break                

        ######---------------------------- DISPLAY SOLUTION SUMMARY --------------------------------##################
        try:
            model_fun=planning_model  
            subsol=model_fun(reformulation=False)
            subsol=initialize_model(subsol,from_feasible=True,feasible_model=best_sol_name)
            # subsol.Nref.pprint()

            TPC3=pe.value(subsol.TCP3)
            OBJ_FOUND=TPC3

            print('OBJECTIVE:',str(OBJ_FOUND),flush=True)
        except:
            print('No feasible solution found',flush=True)


    return LB_list, UB_list, Time_list,Iter_list

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    # stdoutOrigin=sys.stdout 
    mip_solver='cplex'
    sub_options={'add_options':['option optcr=0;','option threads=0;','GAMS_MODEL.optfile = 1;','$onecho > cplex.opt', 'intsollim large', '$offecho']}

    m=planning_model(reformulation=True)
    # # opt1 = pe.SolverFactory('gams')
    # # results = opt1.solve(m, solver=mip_solver, add_options = sub_options['add_options'] , tee=True)

    # # m =solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0) 
    # # generate_initialization(m=m,model_name='init_GBD') 
    m=initialize_model(m,from_feasible=True,feasible_model='init_GBD')

    ext_Var_init=[]
    for i in m.i:
        ext_Var_init.append(int(sum(pe.value(m.y[i,t]) for t in m.t)+1))
    print(ext_Var_init)

    m=planning_model(reformulation=True)
    logic_fun=problem_logic
    ext_ref={m.YR:m.ordered}
    sub_options={'add_options':['option optcr=0;','option threads=0;','GAMS_MODEL.optfile = 1;','$onecho > cplex.opt', 'intsollim large', '$offecho']}
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # # m=external_ref(m,ext_Var_init,logic_fun,dict_extvar=reformulation_dict)
    # # m =solve_with_minlp(m,minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0,transform_required=False) 
    # # generate_initialization(m=m,model_name='init_fea')


    # m=planning_model_original()
    # m=solve_subproblem(m,subproblem_solver='cplex',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = True,tee = True,rel_tol = 0)  
    
    sub=planning_model(reformulation=False)


    sub=initialize_model(sub,from_feasible=True,feasible_model='init_GBD')
    # sub=external_ref(sub,ext_Var_init,logic_fun,dict_extvar=reformulation_dict)

    P_val_init={}
    for i in sub.i:
        for tp in sub.tp:
            P_val_init[i,tp]=pe.value(sub.P[i,tp])


    def _DefP(m,i,tprime):
        if round(P_val_init[i,tprime])==round(m.P[i,tprime].lb):
            return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime])+1])
        elif round(P_val_init[i,tprime])==round(m.P[i,tprime].lb)+1:
            return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime]),round(P_val_init[i,tprime])+1])
        elif round(P_val_init[i,tprime])==round(m.P[i,tprime].ub):
            return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime])-1,round(P_val_init[i,tprime])])
        else:
            return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime])-1,round(P_val_init[i,tprime]),round(P_val_init[i,tprime])+1])
    sub.updated_def_P=pe.Constraint(sub.i,sub.tp,rule=_DefP)

    # setattr(sub,'updated_def_P_%s' %str(k),sub.updated_def_P[k])


    def _DefP2(m,i,tprime):
        if round(P_val_init[i,tprime])==round(m.P[i,tprime].lb):
            return 1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime])+1])
        elif round(P_val_init[i,tprime])==round(m.P[i,tprime].lb)+1:
            return  1==m.yp0[i,tprime]+sum(m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime]),round(P_val_init[i,tprime])+1]) 
        elif round(P_val_init[i,tprime])==round(m.P[i,tprime].ub):
            return 1==sum(m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime])-1,round(P_val_init[i,tprime])]) 
        else:
            return 1==sum(m.yp[i,t,tprime] for t in [round(P_val_init[i,tprime])-1,round(P_val_init[i,tprime]),round(P_val_init[i,tprime])+1]) 
    sub.updated_def_P2=pe.Constraint(sub.i,sub.tp,rule=_DefP2)
    # setattr(sub,'updated_def_P2_%s' %str(k),sub.updated_def_P2[k])

    # # m.xi={}
    # # m.xj={}
    # # m.u={}
    # # m.udef={}
    # m.Mc1={}
    # m.Mc2={}
    # m.Mc3={}
    # m.Mc3={}
    # k=0
    # for i in sub.i:
    #     for tprime in sub.tp:
    #         if round(P_val_init[i,tprime])==round(sub.P[i,tprime].lb)+1:
    #             k=k+1
    #             # m.u[k]=pe.Var(within=pe.NonNegativeReals,bounds=(0,1))
    #             # setattr(sub,'u_%s' %str(k),sub.u[k])

    #             # def _udef(m):
    #             #     return m.u[k]<=0 
    #             # m.udef[k]=pe.Constraint(rule=_udef)
    #             def _Mc1(m):
    #                 return 0>=m.yp0[i,tprime].lb*m.yp[i,round(P_val_init[i,tprime])+1,tprime]+m.yp0[i,tprime]*m.yp[i,round(P_val_init[i,tprime])+1,tprime].lb-m.yp0[i,tprime].lb*m.yp[i,round(P_val_init[i,tprime])+1,tprime].lb
    #             m.Mc1[k]=pe.Constraint(rule=_Mc1)

    #             def _Mc2(m):
    #                 return 0>=m.yp0[i,tprime].ub*m.yp[i,round(P_val_init[i,tprime])+1,tprime]+m.yp0[i,tprime]*m.yp[i,round(P_val_init[i,tprime])+1,tprime].ub-m.yp0[i,tprime].ub*m.yp[i,round(P_val_init[i,tprime])+1,tprime].ub
    #             m.Mc2[k]=pe.Constraint(rule=_Mc2)

    #             def _Mc3(m):
    #                 return 0<=m.yp0[i,tprime].lb*m.yp[i,round(P_val_init[i,tprime])+1,tprime]+m.yp0[i,tprime]*m.yp[i,round(P_val_init[i,tprime])+1,tprime].ub-m.yp0[i,tprime].lb*m.yp[i,round(P_val_init[i,tprime])+1,tprime].ub
    #             m.Mc3[k]=pe.Constraint(rule=_Mc3)                
            
    #         elif round(P_val_init[i,tprime])!=round(sub.P[i,tprime].lb) and round(P_val_init[i,tprime])!=round(sub.P[i,tprime].ub):

                

    def _DefP3(m,i,tprime):
        if round(P_val_init[i,tprime])==round(m.P[i,tprime].lb):
            return pe.Constraint.Skip
        elif round(P_val_init[i,tprime])==round(m.P[i,tprime].lb)+1:
            return m.yp0[i,tprime]*m.yp[i,round(P_val_init[i,tprime])+1,tprime]<=0
            # return m.yp0[i,tprime]+m.yp[i,round(P_val_init[i,tprime])+1,tprime]<=1
        elif round(P_val_init[i,tprime])==round(m.P[i,tprime].ub):
            return pe.Constraint.Skip
        else:
            return m.yp[i,round(P_val_init[i,tprime])-1,tprime]*m.yp[i,round(P_val_init[i,tprime])+1,tprime]<=0
            # return m.yp[i,round(P_val_init[i,tprime])-1,tprime]+m.yp[i,round(P_val_init[i,tprime])+1,tprime]<=1
    sub.updated_def_P3=pe.Constraint(sub.i,sub.tp,rule=_DefP3)
    # setattr(sub,'updated_def_P3_%s' %str(k),sub.updated_def_P3[k])

    # for i in sub.updated_def_P.values():
    #     i.pprint()
    #     print(pe.value(i.body))

    for i in sub.i:
        for tp in sub.tp:
            sub.P[i,tp].setlb(max([round(P_val_init[i,tp])-1,sub.P[i,tp].lb]))
            sub.P[i,tp].setub(min([round(P_val_init[i,tp])+1,sub.P[i,tp].ub]))
            # sub.P[i,tp].fix(round(P_val_init[i,tp]))
            # sub.P[i,tp].fix(round(sub.P[i,tp].lb))

    # sub.P.domain=pe.NonNegativeReals
    sub.y.domain=pe.NonNegativeReals
    sub.yp.domain=pe.NonNegativeReals
    sub.yp0.domain=pe.NonNegativeReals
    sub.y.setlb(0)
    sub.y.setub(1)
    sub.yp.setlb(0)
    sub.yp.setub(1)
    sub.yp0.setlb(0)
    sub.yp0.setub(1)

    # sub_options={'add_options':['option optcr=0;','option threads=0;','GAMS_MODEL.optfile = 1;','$onecho > dicopt.opt', 'stop 1', '$offecho']}
    sub=solve_subproblem(sub,subproblem_solver='sbb',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = True,tee = True,rel_tol = 0)  
   
    sub.P.pprint()
    sub.y.pprint()


 
    # sub=planning_model(reformulation=False)
    # sub=initialize_model(sub,from_feasible=True,feasible_model='init_GBD')
    # # sub=external_ref(sub,ext_Var_init,logic_fun,dict_extvar=reformulation_dict)
    # sub.updated_def_P={}
    # sub.updated_def_P2={}
    # sub.updated_def_P3={}
    objective_functions={}

    k=0
    infinity_aprox=1e+10
    for var in P_val_init.keys():
        for direction in [-1,1]:
            k=k+1
            print('---------------Neighborhood evaluation',k,'------------------------------\n')
            for i in sub.i:
                for tp in sub.tp:
                    sub.P[i,tp].fix(round(P_val_init[i,tp]))
            
            if sub.P[var].value+direction>sub.P[var].ub or sub.P[var].value+direction<sub.P[var].lb:
                objective_functions[var,direction]=infinity_aprox
                print('External variables out of bounds')
            else:
                sub.P[var].fix(round(P_val_init[i,tp])+direction) 
                sub=solve_subproblem(sub,subproblem_solver='ipopth',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)  
                print(pe.value(sub.obj))
                print(sub.dsda_status)
                if sub.dsda_status=='Optimal':
                    objective_functions[var,direction]=pe.value(sub.obj)
                else:
                    objective_functions[var,direction]=infinity_aprox



            # if k>=2:
            #     sub.updated_def_P[k-1].deactivate()














    # time_limit=1000 #time limit in seconds
    # # EXPERIEMTS
    # model_fun=planning_model
    # solver='GBD'
    # initialize_with_master=False




    # best_sol_name='GBD_sol'
    # if not initialize_with_master:
    #     best_sol_name=best_sol_name+'__Initialized_RMINLP'

    # # dir_path = os.path.dirname(os.path.abspath(__file__))
    # # file_name=dir_path+'/'+best_sol_name
    # # sys.stdout = open(file_name+".txt", "w")
    # print(best_sol_name,flush=True)
    # solution=solve_scheduling(model_fun,solver,best_sol_name,time_limit=time_limit,initialize_with_master=initialize_with_master,no_good_cuts=False,relaxed=False,init_name='init_GBD')
    # # sys.stdout.close()
    # # sys.stdout=stdoutOrigin

    # # df=pd.DataFrame(solution)
    # # df=df.transpose()
    
    # # writer = pd.ExcelWriter(file_name+'.xlsx', engine='xlsxwriter')
    # # df.to_excel(writer, index=False)
    # # writer.save()





    # ext_Var_init=[]
    # for i in m.i:
    #     ext_Var_init.append(int(sum(pe.value(m.y[i,t]) for t in m.t)+1))

    # # REFORMULATION

    # m=planning_model(reformulation=True)
    # ext_ref={m.YR:m.ordered}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)


    # #SOLUTION
    # sub_options={'add_options':['option threads=0;','GAMS_MODEL.optfile = 1;','$onecho > cplex.opt', 'intsollim large', '$offecho']}

    # model_fun=planning_model
    # logic_fun=problem_logic
    # kwargs={'reformulation':True}
    # current_central=ext_Var_init #initialization of external variables
    # original_m=model_fun(**kwargs) # declaration of original GDP model
    # max_iter_out=1000000 # Maximum number of iterations
    # upper_evaluated={} #evaluated points
    # old_obj=1e+10
    # teed=True


    # start=time.time()
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,ext_Var_init,ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = mip_solver,subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = True,rel_tol = 0.02,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=False)
    # end=time.time()
    # print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  


    # # CG_DSDA algorithm
    # start=time.time()
    # for out in range(max_iter_out):
    #     if teed:
    #         print('------Outer iteration= ',out+1)
    #     upper_evaluated[out+1]=[current_central] #Update evaluated points with current external variable being evaluated
        
    #     # perform the reformulation
    #     for i in original_m.i:
    #         for ind in original_m.ordered:
    #             disjunct=original_m.YR_Disjunct[ind,i]
    #             disjunct.activate()
    #     original_m = external_ref_neighborhood_general(m=original_m,x=current_central,extra_logic_function=logic_fun,dict_extvar=reformulation_dict)
    #     # TODO: had to do this outside reformulation function. Generalize!!
    #     for i in original_m.i:
    #         for ind in original_m.ordered:
    #             disjunct=original_m.YR_Disjunct[ind,i]
    #             if original_m.YR[ind,i].is_fixed() and original_m.YR[ind,i].value==False:
    #                 disjunct.deactivate()
    #                 # disjunct.pprint()
    #                 # print(disjunct.name,' deactivated') 

    #     # solve the problem
    #     m=original_m.clone() #initialize mip problem
    #     # m=solve_with_gdpopt(m,mip=mip_solver,minlp='cplex',mip_options=sub_options,timelimit=86400,rel_tol=0,strategy='LBB',tee=True)
    #     m =solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0.03) 
        
    #     # Extract solution
    #     Sol_found=[]

    #     # TODO: Generalize this part
    #     for i in original_m.i:
    #         for ind in original_m.ordered:
    #             if(round(pe.value(m.YR_Disjunct[ind,i].binary_indicator_var))==1):
    #                 Sol_found.append(ind+1) #NOTE, because my set starts at 0, but ext vars at 1
    #                 break
    #     if teed:
    #         print(Sol_found)

    #     # for v in m.component_data_objects(ctype=pe.Var):
    #     #     if v.parent_component().name!='X' and v.is_binary()==True:
    #     #         print(v.parent_component().name,'=',pe.value(v))
    #     #         # v.pprint()
    #     #         # for vv in v.items():
    #     #         #     vv.pprint()
    

    #     # Generate search direction
    #     direction=[]
    #     for i in range(len(Sol_found)):
    #         direction.append(Sol_found[i]-current_central[i])              

    #     if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #         m.mip_status='Infeasible'
    #         #TODO: IN CASE OF INFEASIBILITIES, I should declare infinity objective, however, it is not possible because new neighborhood always have the previous feasible solution in it.
    #     else:
    #         m.mip_status='Optimal'
    #     if teed:
    #         if m.mip_status == 'Optimal':  

    #             print('   Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                            
    #         else:
    #             print('   Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))        


    #         print('   SEARCH DIRECTION: ', direction)                
        
    #     if round(sum(abs(j) for j in direction))==0 or old_obj<=round(pe.value(m.obj)):    
    #         end=time.time()
    #         print('Objective CG-DSDA='+str(pe.value(m.obj))+', best D-SDSA='+str(Sol_found),'cputime LG-DSDA= '+str(end-start))  
    #         break
    #     else:
    #         # update objective and variables
    #         old_obj=round(pe.value(m.obj))
    #         current_central=Sol_found
    #         # delete most recently solved mip
    #         del m
    #         #update

