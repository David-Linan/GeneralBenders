from __future__ import division
import pyomo.environ as pe
import pyomo.dae as dae
import math
import os
import io
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory
from pyomo.gdp import Disjunct, Disjunction
import itertools


def scheduling_gdp_var_proc_time(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=120,last_time_hours: float=120,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3}):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') 
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') 
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
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

    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

    _fixed_cost={}
    _fixed_cost['T1','U1']=10

    _fixed_cost['T2','U2']=15
    _fixed_cost['T2','U3']=30

    _fixed_cost['T3','U2']=5
    _fixed_cost['T3','U3']=25

    _fixed_cost['T4','U2']=5
    _fixed_cost['T4','U3']=20

    _fixed_cost['T5','U4']=20
    m.fixed_cost=pe.Param(m.I,m.J,default=0,initialize=_fixed_cost,doc="Fixed cost to run task i in unit j [m.u./batch]")

    _variable_cost_param={}
    _variable_cost_param['T1','U1']=0

    _variable_cost_param['T2','U2']=0
    _variable_cost_param['T2','U3']=0

    _variable_cost_param['T3','U2']=0
    _variable_cost_param['T3','U3']=0

    _variable_cost_param['T4','U2']=0
    _variable_cost_param['T4','U3']=0

    _variable_cost_param['T5','U4']=0
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 3 
        elif K=='S9':
            return 4
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')


    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #NOTE: in this context these are just values I use to calculate sigma 
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,mutable=True,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(pe.value(m.tau_p[I,J])/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,mutable=True,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    # # -----------scheduling variables -----------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    def _B_bounds(m,I,J,T):
        return (0,m.beta_max[I,J])
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_B_bounds,initialize=0,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (0,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,initialize=0,doc='Inventory of material k at time t')

    # Auxiliary ariables required to decrease combinatorial complexity
    m.sumX=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=(0,m.lastT+1),initialize=0,doc='Auxiliary variable 1 for disjunctive section. Appears in UNIT UTILIZATION cosntraints ')
    def _B_shift_bounds(m,I,J,T):
        return (0,m.beta_max[I,J])
    m.B_shift=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_B_shift_bounds,initialize=0,doc='Auxiliary variable 2 for disjunctive section. Appears in MATERIAL BALANCES. This is a time-shifted version of variable B')

    # # ----------Scheduling Constraints that DO NOT depend on disjunctions-----------------------------------------
    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='(1A): UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='(1A): UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='(1C): MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='DOES NOT APPEAR IN MANUSCRIPT: INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='(1B): UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='(1D): MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   

    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')


    def _maxTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(upper_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.maxTau=pe.Param(m.I,m.J,initialize=_maxTau_rule,doc='Maximum number of discrete elements required to complete task [dimensionless]')
    
    ### NEW ###################
    def _varTime_bounds(m,I,J,T):
        if m.I_i_j_prod[I,J]==1:
            return (0,upper_t_h[(I,J)])
        else:
            return (0,0)
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing times [h]')

    #NOTE: that Eq. (1G) in the manuscript is not needed because lower bound for processing time is 0

    ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b.
    def _rule_beta_time(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return pe.value(m.tau_p[I,J])/m.beta_max[I,J]
        else:
            return 0 
    m.beta_time=pe.Param(m.I,m.J,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    def _rule_ineqrel_1(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    def _rule_ineqrel_2(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    m.ineq_rel_1=pe.Constraint(m.I,m.J,m.T,rule=_rule_ineqrel_1,doc='DOES NOT APPEAR IN MANUSCRIPT: Linear Big-M relationship between processing times and size of batches')
    m.ineq_rel_2=pe.Constraint(m.I,m.J,m.T,rule=_rule_ineqrel_2,doc='DOES NOT APPEAR IN MANUSCRIPT: Linear Big-M relationship between processing times and size of batches')



    ### END OF THE SECTION

    m.ordered_set={}
    m.YR={}
    m.oneYR={}
    m.YR_disjunct={}
    m.Disjunction1={}
    positcui=-1
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                positcui=positcui+1
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each I J pair') 
                setattr(m,'ordered_set_%s_%s' %(I,J),m.ordered_set[I,J])

                def _YRinit(m,ordered_set):
                    if ordered_set==(x_initial[positcui]+m.minTau[I,J]-1):
                        return True 
                    else:
                        return False
                m.YR[I,J]=pe.BooleanVar(m.ordered_set[I,J],initialize=_YRinit)
                setattr(m,'YR_%s_%s' %(I,J),m.YR[I,J])            

                #Constraint that allow to apply the reformulation over YR
                def _select_one(m):
                    return pe.exactly(1,m.YR[I,J])
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one,doc='(1H): exactly on (OR) operator between disjunctions') 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J]) #NOTE #when using les or equal this formulation is correct. But when using equal I should use the min operatior in Eq. (1H)
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='(1H): Relationshipt between continuous and discrete processing time')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='(1H):Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='(1H):Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="(1H): each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True,doc='(1H): Declaration of disjunction')
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # # -------Reformulation----------------------------------------------------
    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
            return math.floor((m.T.__len__()-1)/m.minTau[I,J])  #TODO: Note that I am using the minimum, or I can use Tau, but I would have to incorporate this within the disjunction.
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')

    def _Nref_bounds(m,I,J):
        return (0,m.lastN[I,J])
    m.Nref=pe.Var(m.I_J,within=pe.Integers,bounds=_Nref_bounds,doc='reformulation variables from 0 to lastN')

    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==m.Nref[I,J]
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='(1F): constraint that specifies the relationship between Integer and binary variables')   


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in m.J) for I in m.I) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TMC= pe.Var(within=pe.Reals,initialize=0,doc='TMC: Total material cost')
    def _C_TMC(m):
        return m.TMC==sum(m.raw_cost[K]*(m.S0[K]-m.S[K, m.lastT]) for K in m.K_inputs) 
    m.C_TMC=pe.Constraint(rule=_C_TMC)
    m.SALES=pe.Var(within=pe.Reals,initialize=0,doc='SALES: Revenue form selling products')
    def _C_SALES(m):
        return m.SALES==sum(m.revenue[K]*m.S[K, m.lastT] for K in m.K_products)
    m.C_SALES=pe.Constraint(rule=_C_SALES)

    if obj_type=='profit_max':
        def _obj(m):
            return m.TCP1+m.TCP2+m.TMC-m.SALES  
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)      
    return m


def problem_logic_scheduling(m):
    logic_expr = []
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for index in m.ordered_set[I,J]:
                    logic_expr.append([m.YR[I,J][index],m.YR_disjunct[I,J][index].indicator_var])              

    # for I_J in m.I_J:
    #     I=I_J[0]
    #     J=I_J[1]
    #     for index in m.ordered_set2[I,J]:
    #         logic_expr.append([m.YR2[I,J][index],m.YR2_Disjunct[I,J][index].indicator_var])  
    return logic_expr

if __name__ == "__main__":
    m=scheduling_gdp_var_proc_time()
  