from __future__ import division
from pickle import TRUE

import sys
# sys.path.insert(0, '/home/dadapy/GeneralBenders/')
sys.path.append('C:/Users/TEMP/Desktop/GeneralBenders/')
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
def problem_logic_scheduling(m):
    logic_expr = []
    for N in m.N:
        for I_J in m.I_J:
                if N<=m.lastN[I_J]:
                    logic_expr.append([m.Z[N,I_J],m.Z_binary[N,I_J]])
    return logic_expr

def build_scheduling_Boolean_cost_min():
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()


    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    #profit maximization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   


    #BOOLEAN VARIABLES--------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.N=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.N.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    m.Z=pe.BooleanVar(m.N,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.N,m.I_J,within=pe.Binary,doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.N:
        for I_J in m.I_J:
                m.Z[N,I_J].associate_binary_var(m.Z_binary[N,I_J])
                if N>m.lastN[I_J]:
                    m.Z[N,I_J].fix(False)
                    m.Z_binary[N,I_J].set_value(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.N if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return pe.exactly(1,[m.Z[N,I,J] for N in m.N if N<=m.lastN[I,J]])
    m.EXACTLY_CONST=pe.LogicalConstraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()

    return m
   
def build_scheduling_Original_cost_min():
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()


    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    #profit maximization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   

    return m
  
def build_scheduling_Boolean_profit_max():
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    m.demand=pe.Param(m.K,m.T,initialize=0,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()


    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    #profit maximization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   


    #BOOLEAN VARIABLES--------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.N=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.N.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    m.Z=pe.BooleanVar(m.N,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.N,m.I_J,within=pe.Binary,doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.N:
        for I_J in m.I_J:
                m.Z[N,I_J].associate_binary_var(m.Z_binary[N,I_J])
                if N>m.lastN[I_J]:
                    m.Z[N,I_J].fix(False)
                    m.Z_binary[N,I_J].set_value(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.N if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return pe.exactly(1,[m.Z[N,I,J] for N in m.N if N<=m.lastN[I,J]])
    m.EXACTLY_CONST=pe.LogicalConstraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()

    return m
   
def build_scheduling_Original_profit_max():
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    m.demand=pe.Param(m.K,m.T,initialize=0,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()


    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    #profit maximization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   

    return m
 
def build_scheduling_Boolean_cost_min_feasibility(objective,epsilon):
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()


    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    def _obj1(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)<=objective+epsilon
    m.obj1=pe.Constraint(rule=_obj1,doc='feasibility of objective value 1')

    # def _obj2(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)>=objective-epsilon
    # m.obj2=pe.Constraint(rule=_obj2,doc='feasibility of objective value 2')

    def _obj_dummy(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj_dummy=pe.Objective(rule=_obj_dummy,sense=pe.minimize,doc='dummy objective')

    #profit maximization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   


    #BOOLEAN VARIABLES--------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.N=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.N.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    m.Z=pe.BooleanVar(m.N,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.N,m.I_J,within=pe.Binary,doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.N:
        for I_J in m.I_J:
                m.Z[N,I_J].associate_binary_var(m.Z_binary[N,I_J])
                if N>m.lastN[I_J]:
                    m.Z[N,I_J].fix(False)
                    m.Z_binary[N,I_J].set_value(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.N if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return pe.exactly(1,[m.Z[N,I,J] for N in m.N if N<=m.lastN[I,J]])
    m.EXACTLY_CONST=pe.LogicalConstraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()

    return m
  
def build_scheduling_Boolean_cost_min_simplified():
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    # _I_i_k_minus={}
    # _I_i_k_minus['T1','S1']=1

    # _I_i_k_minus['T2','S3']=1
    # _I_i_k_minus['T2','S2']=1

    # _I_i_k_minus['T3','S4']=1
    # _I_i_k_minus['T3','S5']=1

    # _I_i_k_minus['T4','S6']=1
    # _I_i_k_minus['T4','S3']=1

    # _I_i_k_minus['T5','S7']=1
    # m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    # _I_i_k_plus={}
    # _I_i_k_plus['T1','S4']=1

    # _I_i_k_plus['T2','S5']=1

    # _I_i_k_plus['T3','S6']=1
    # _I_i_k_plus['T3','S8']=1

    # _I_i_k_plus['T4','S7']=1

    # _I_i_k_plus['T5','S6']=1
    # _I_i_k_plus['T5','S9']=1

    # m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    # _rho_minus={}
    # _rho_minus['T1','S1']=1

    # _rho_minus['T2','S3']=0.5
    # _rho_minus['T2','S2']=0.5

    # _rho_minus['T3','S4']=0.4
    # _rho_minus['T3','S5']=0.6

    # _rho_minus['T4','S6']=0.8
    # _rho_minus['T4','S3']=0.2

    # _rho_minus['T5','S7']=1  
    # m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    # _rho_plus={}
    # _rho_plus['T1','S4']=1

    # _rho_plus['T2','S5']=1

    # _rho_plus['T3','S6']=0.6
    # _rho_plus['T3','S8']=0.4

    # _rho_plus['T4','S7']=1

    # _rho_plus['T5','S6']=0.1
    # _rho_plus['T5','S9']=0.9
    # m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    # _beta_min={}
    # _beta_min['T1','U1']=10

    # _beta_min['T2','U2']=10
    # _beta_min['T2','U3']=10

    # _beta_min['T3','U2']=10
    # _beta_min['T3','U3']=10

    # _beta_min['T4','U2']=10
    # _beta_min['T4','U3']=10

    # _beta_min['T5','U4']=10
    # m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    # _beta_max={}
    # _beta_max['T1','U1']=100

    # _beta_max['T2','U2']=50
    # _beta_max['T2','U3']=80

    # _beta_max['T3','U2']=50
    # _beta_max['T3','U3']=80

    # _beta_max['T4','U2']=50
    # _beta_max['T4','U3']=80

    # _beta_max['T5','U4']=200
    # m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    # m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    # def _demand(m,K,T):
    #     if K=='S8' and T==m.lastT:
    #         return 1400
    #     elif K=='S9' and T==m.lastT:
    #         return 1500
    #     else:
    #         return 0 
    # m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t")
    # m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    # m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    # m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    # def _S_bounds(m,K,T):
    #     return (None,m.gamma[K])
    # m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    # def _E1_UNIT(m,J,T):
    #     return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    # m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    # #m.E1_UNIT.display()


    # def _E2_CAPACITY_LOW(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    # m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    # def _E2_CAPACITY_UP(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    # m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    # def _E3_BALANCE(m,K,T):
    #     if T==0:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    # m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    # def _E3_BALANCE_INIT(m,K):
    #     return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    # m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    #profit maximization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   


    #BOOLEAN VARIABLES--------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.N=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.N.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    m.Z=pe.BooleanVar(m.N,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.N,m.I_J,within=pe.Binary,doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.N:
        for I_J in m.I_J:
                m.Z[N,I_J].associate_binary_var(m.Z_binary[N,I_J])
                if N>m.lastN[I_J]:
                    m.Z[N,I_J].fix(False)
                    m.Z_binary[N,I_J].set_value(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.N if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return pe.exactly(1,[m.Z[N,I,J] for N in m.N if N<=m.lastN[I,J]])
    m.EXACTLY_CONST=pe.LogicalConstraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()

    return m

def build_scheduling_Boolean_cost_min_relaxed():
    m = pe.ConcreteModel(name='scheduling_model_maravelias')
    #SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=120,doc='last discrete time value')
    #SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    #m.eta.display()
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    #m.t_p.display()
    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")



    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")
    # m.S0.display()

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=(0,1),doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()



    #set priority to the desired integer variables--TODO: this is currently not working. I must find another alternatie to pass priorities
    # m.priority = pe.Suffix(direction=pe.Suffix.EXPORT, datatype=pe.Suffix.INT)
    # m.priority.set_value(m.X['T1','U1',1], 1)

    #CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()


    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #OBJECTIVE----------------
    # cost minimization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    #profit maximization
    # def _obj(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)-sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)   


    #BOOLEAN VARIABLES--------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.N=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.N.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    # m.Z=pe.BooleanVar(m.N,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.N,m.I_J,within=pe.NonNegativeReals,bounds=(0,1),doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.N:
        for I_J in m.I_J:
                if N>m.lastN[I_J]:
                    m.Z_binary[N,I_J].fix(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.N if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return sum(m.Z_binary[N,I,J] for N in m.N if N<=m.lastN[I,J])==1
    m.EXACTLY_CONST=pe.Constraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()

    return m
  
if __name__ == "__main__":
    # #Do not show warnings
    # logging.getLogger('pyomo').setLevel(logging.ERROR)

    # #relaxed problem
    # m=build_scheduling_Boolean_cost_min_relaxed()
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # options={}
    # start=time.time()
    # m_solved=solve_subproblem(m,subproblem_solver = 'cplex',subproblem_solver_options= options,timelimit= 1000000,gams_output = False,tee= True,rel_tol = 0)   
    # end=time.time()
    # print('CPLEX time (reformulated, relaxed)='+str(end-start))
    # for I_J in m_solved.I_J:
    #     print(1+(sum(N*pe.value(m.Z_binary[N,I_J]) for N in m.N if N<=m.lastN[I_J])))
    #     # for N in m_solved.N:
    #         # if  pe.value(m_solved.Z_binary[N,I_J])>=0.0000000001:
    #             # m.Z_binary[N,I_J].pprint()

    ###-----NEW SCHEDULING ALGORITHM FOR COST MINIMIZATION------
    # lower_obj=pe.value(m_solved.obj) #initialization of cut
    # model_fun_simplified=build_scheduling_Boolean_cost_min_simplified
    # model_fun_feasibility=build_scheduling_Boolean_cost_min_feasibility
    # logic_fun=problem_logic_scheduling

    # master_max_iter=1000#master iterations
    # initialization=[15,8,23,71,1,45,1,9]
    # infinity_val=1e+6
    # min_epsilon_improvement=5#todo: this should agree with the minimum coefficient in the objective function
    # nlp_solver='cplex'
    # neigh=neighborhood_k_eq_2(len(initialization))
    # maxiter=1000 #benders decomposition
    # mastertee=True
    # kwargs={}
    # m=model_fun_simplified(**kwargs)
    # ext_ref = {m.Z: m.N} #reformulation sets and variables
    # #lower_obj=1664
    # start=time.time()

    # [important_info,D,x_actual]=run_function_dbd_scheduling_cost_min(model_fun_feasibility,lower_obj,min_epsilon_improvement,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun_simplified,kwargs,use_random=False,sub_solver_opt={}, tee=True)


    ##-----END OF NEW SCHEDULING ALGORITHM FOR COST MINIMIZATION


    # model_fun =build_scheduling_Original_cost_min
    # kwargs={}
    # logic_fun=problem_logic_scheduling
    # #CPLEX solution
    # m=model_fun(**kwargs)
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # options=    {'add_options':[
    #     'GAMS_MODEL.optfile = 1;'
    #     '\n'
    #     '$onecho > cplex.opt \n'
    #     'mipemphasis 4\n'
    #     '$offecho \n']}

    # start=time.time()
    # m_solved=solve_subproblem(m,subproblem_solver = 'cplex',subproblem_solver_options= options,timelimit= 1000000,gams_output = False,tee= True,rel_tol = 0)   
    # end=time.time()
    # print('CPLEX time ='+str(end-start))

    # CPLEX solution reformulated
    # model_fun =build_scheduling_Boolean_cost_min
    # kwargs={}
    # m=model_fun(**kwargs)
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # options=    {'add_options':[
    #     'GAMS_MODEL.optfile = 1;'
    #     '\n'
    #     '$onecho > cplex.opt \n'
    #     'mipemphasis 0\n'
    #     '$offecho \n']}

    # start=time.time()
    # m_solved=solve_subproblem(m,subproblem_solver = 'xpress',subproblem_solver_options= options,timelimit= 1000000,gams_output = False,tee= True,rel_tol = 0)   
    # end=time.time()
    # print('CPLEX time (reformulated)='+str(end-start))



    #### SOLVE USING DSDA AND DBD
    model_fun =build_scheduling_Boolean_cost_min
    logic_fun=problem_logic_scheduling
    kwargs={}
    m=model_fun(**kwargs)
    ext_ref = {m.Z: m.N} #reformulation sets and variables
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)


    max_cplex_sol=1 #maximum number of cplex solutions
    dict_known={}
    for sol in range(1,max_cplex_sol+1):

        m=model_fun(**kwargs)
        pe.TransformationFactory('core.logical_to_linear').apply_to(m)
        options=    {'add_options':[
            'GAMS_MODEL.optfile = 1;'
            '\n'
            '$onecho > cplex.opt \n'
            'intsollim '+str(sol)+'\n'
            'mipemphasis 1\n'
            '$offecho \n']}
        if sol==max_cplex_sol:
            start=time.time()
        m_solved=solve_subproblem(m,subproblem_solver = 'cplex',subproblem_solver_options= options,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)   
        if sol==max_cplex_sol:
            end=time.time()
        #print(m_solved.dsda_status)   
        # solved=generate_initialization(m=m_solved,model_name='maravelias_cplex_reformualted_profit_max')
        empty_list=[]
        for I_J in m_solved.I_J:
            for N in m_solved.N:
                if pe.value(m_solved.Z_binary[N,I_J])==1:
                    # print('ext_Var associated to '+str(I_J)+'is '+str(N+1))
                    empty_list.append(N+1)
        # print(pe.value(m_solved.obj))
        dict_known[tuple(empty_list)]=pe.value(m_solved.obj)
        updated_known=dict_known.copy()

    required_time=end-start
    print('Time required to evaluate '+str(max_cplex_sol)+' MIP solutions: '+str(required_time))
    print(dict_known)
# USE THIS IF THE PROBLEM IS NOT A FEASIBILITY PROBLEM WHEN SOLVING DIFFERENT INITIALZIATION TRIALS (E.G. PROFIT MAX)
    # for trial in dict_known:
    #     m=build_scheduling()       
    #     external_ref(m,list(trial),logic_fun,reformulation_dict,tee=False)
    #     m_solved2=solve_subproblem(m,subproblem_solver = 'cplex',subproblem_solver_options= {},timelimit= 1000000,gams_output = False,tee= True,rel_tol = 0) 
    #     updated_known[trial]=pe.value(m_solved2.obj)
    # print(updated_known)   


    m=model_fun(**kwargs)
    initialization=list(min(dict_known, key=lambda k: dict_known[k]) )
    infinity_val=1e+6
    nlp_solver='cplex'
    sub_options={'add_options':[
        'GAMS_MODEL.optfile = 1;'
        '\n'
        '$onecho > cplex.opt \n'
        'intsollim 1\n'  #TODO: CHANGE THIS FOR COST MIN
        'mipemphasis 1\n'
        '$offecho \n']}
    neigh=neighborhood_k_eq_2(len(initialization))
    maxiter=1000
    # DBD
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    [important_info,important_info_preprocessing,D,x_actual]=run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,known_solutions=updated_known)
    print('obj= ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
    
     #DSDA
    # m_init_fixed = external_ref(m=m,x=initialization,extra_logic_function=logic_fun,dict_extvar=ext_ref,tee=False)
    # m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver='baron', timelimit=10000, tee=False)
    # init_path = generate_initialization(m=m_init_solved)   
    start=time.time()
    D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,{},initialization,ext_ref,logic_fun,k = '2',provide_starting_initialization= True,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 1000,timelimit = 3600,gams_output = False,tee= False,global_tee = True,rel_tol = 1e-3)
    end=time.time()
    print('Objective='+str(pe.value(D_SDAsol.obj))+', best='+str(routeDSDA[-1]))
    print('cputime= '+str(end-start))   



# # # Optimal: 14515.5
# # # 786.0247232913971 secinds 
# # #     ## LOAD RESULTS
# # #     model = build_scheduling()
# # #     m=initialize_model(model,from_feasible=True,feasible_model='maravelias_cplex_reformualted_profit_max')
# # #     for I_J in m.I_J:
# # #         for N in m.N:
# # #             if pe.value(m.Z_binary[N,I_J])==1:
# # #                 print('ext_Var associated to '+str(I_J)+'is '+str(N+1))
# # #     textbuffer = io.StringIO()
# # #     for v in m.component_objects(pe.Var, descend_into=True):
# # #         v.pprint(textbuffer)
# # #         textbuffer.write('\n')
# # #     for v in m.component_data_objects(pe.Objective, descend_into=True):
# # #         textbuffer.write('Objective value '+str(pe.value(v)))
# # #         textbuffer.write('\n')
# # #         # v.pprint(textbuffer)
# # #         # textbuffer.write('\n')
# # #     with open('maravelias_cplex_reformualted_profit_max.txt', 'w') as outputfile:
# # #         outputfile.write(textbuffer.getvalue())