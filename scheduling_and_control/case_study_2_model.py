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
from Working_Distillation_Model import create_distillation_model

## V1
def case_2_scheduling_control_gdp_var_proc_time(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3}):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3','U4'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL

    m.CA0=pe.Param(initialize=10,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CB0=pe.Param(initialize=1.1685,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CT0=pe.Param(initialize=m.CA0+m.CB0+m.CC0)

    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Fixed initial volume for dynamic tast [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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
    for T in m.T:
        _rho_minus['T1','S1',T]=1

        _rho_minus['T2','S3',T]=0.5
        _rho_minus['T2','S2',T]=0.5

        _rho_minus['T3','S4',T]=0.4
        _rho_minus['T3','S5',T]=0.6

        _rho_minus['T4','S6',T]=0.8
        _rho_minus['T4','S3',T]=0.2

        _rho_minus['T5','S7',T]=1 
    m.rho_minus=pe.Var(m.I,m.K,m.T,within=pe.NonNegativeReals,bounds=(0,1),initialize=_rho_minus,doc="Fraction of material in state k consumed by task i ")

    def _rho_minus_sum(m,I,T):
        return sum(m.rho_minus[I,K,T] for K in m.K if m.I_i_k_minus[I,K]==1)==1
    m.rho_minus_sum=pe.Constraint(m.I,m.T, rule=_rho_minus_sum,doc='Sum of fractions must be equal to 1')

    #Fixed rhos
    for T in m.T:
        m.rho_minus['T1','S1',T].fix(1)

        m.rho_minus['T3','S4',T].fix(0.4)
        m.rho_minus['T3','S5',T].fix(0.6)

        m.rho_minus['T4','S6',T].fix(0.8)
        m.rho_minus['T4','S3',T].fix(0.2)

        m.rho_minus['T5','S7',T].fix(1)     

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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K,0]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K,T]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    # def _rule_beta_time(m,I,J):
    #     if m.I_i_j_prod[I,J]==1:
    #         return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
    #     else:
    #         return 0 
    # m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    # def _rule_ineqrel_1(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    # def _rule_ineqrel_2(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    # m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    # m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    # ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={}  

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct
                
                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0 # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0 # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0)))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*2 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*2, ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   

    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2',T]*m.Vol[I,J,T][N]-m.V0*(1-(m.CB0/m.CBIN))<=(m.beta_max[I,J]-m.V0*(1-(m.CB0/m.CBIN)))*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*(1-(m.CB0/m.CBIN))-m.rho_minus[I,'S2',T]*m.Vol[I,J,T][N]<=m.V0*(1-(m.CB0/m.CBIN))*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])

                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    # m.ordered_set2={}
    # m.YR2={}
    # m.oneYR2={}
    # m.YR2_Disjunct={}
    # m.Disjunction2={}
    # for I_J in m.I_J:
    #     positcui=positcui+1
    #     I=I_J[0]
    #     J=I_J[1]
    #     m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
    #     setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
    #     def _YR2init(m,ordered_set2):
    #         if ordered_set2== x_initial[positcui]-1:
    #             return True
    #         else:
    #             return False       
    #     m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
    #     setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

    #     def _select_one2(m):
    #         return pe.exactly(1,m.YR2[I,J])
    #     m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
    #     setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

    #     def _build_YR2_Disjunct(m,indexN):
    #         def _DEF_Nref(m):
    #             return m.model().Nref[I,J]==indexN
    #         m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
    #     m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
    #     setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

    #     # Create disjunction
    #     def Disjunction2(m):   
    #         return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
    #     m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
    #     setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # # Associate disjuncts with boolean variables
    #     for index in m.ordered_set2[I,J]:
    #         m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
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
            return m.TCP1+m.TCP2+m.TCP3+m.TMC-m.SALES  
            # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)      
    return m
## V2
#Same as previos, except that in this case I do not have rho variable. I am fixing that instead and optimizing a concentration, which is the ammount of B at the begining of the process
def case_2_scheduling_control_gdp_var_proc_time_simplified(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3}):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3','U4'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CAIN=pe.Param(initialize=10.62,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Fixed initial volume for dynamic tast [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    # def _rule_beta_time(m,I,J):
    #     if m.I_i_j_prod[I,J]==1:
    #         return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
    #     else:
    #         return 0 
    # m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    # def _rule_ineqrel_1(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    # def _rule_ineqrel_2(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    # m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    # m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    # ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={} 

    m.CA0={} 
    m.CB0={}
    m.CT0={}
    m.c_defCT0={}

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct




                m.CA0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CAIN),doc='Initial composition of A [kmol/m^3]')
                setattr(m,'CA0_%s_%s_%s' %(I,J,T),m.CA0[I,J,T])

                m.CB0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CBIN),doc='Initial composition of B [kmol/m^3]')
                setattr(m,'CB0_%s_%s_%s' %(I,J,T),m.CB0[I,J,T])

                m.CT0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(1e-3,m.CAIN+m.CBIN),doc='Initial total composition  [kmol/m^3]')
                setattr(m,'CT0_%s_%s_%s' %(I,J,T),m.CT0[I,J,T])

                def _defCT0(m):
                    return m.CT0[I,J,T]== m.CA0[I,J,T]+m.CB0[I,J,T]
                m.c_defCT0[I,J,T] = pe.Constraint(rule=_defCT0)
                setattr(m,'c_defCT0_%s_%s_%s' %(I,J,T),m.c_defCT0[I,J,T])       

                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0[I,J,T]))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0[I,J,T])))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0[I,J,T]))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*2 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*2, ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   

    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking2_3={} #rho and Vol relationship 
    m.linking2_4={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2']*m.Vol[I,J,T][N]-m.V0*((m.CA0[I,J,T]/m.CAIN))<=(m.beta_max[I,J])*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*((m.CA0[I,J,T]/m.CAIN))-m.rho_minus[I,'S2']*m.Vol[I,J,T][N]<=m.V0*(m.CAIN/m.CAIN)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])



                def _linking2_3(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0<=(m.beta_max[I,J]+m.V0)*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_3,doc='') 
                setattr(m,'linking2_3_%s_%s_%s' %(I,J,T),m.linking2_3[I,J,T])

                def _linking2_4(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0)<=(m.V0*(m.CBIN/m.CBIN)+m.beta_max[I,J]-m.V0)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_4[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_4,doc='') 
                setattr(m,'linking2_4_%s_%s_%s' %(I,J,T),m.linking2_4[I,J,T])




                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    # m.ordered_set2={}
    # m.YR2={}
    # m.oneYR2={}
    # m.YR2_Disjunct={}
    # m.Disjunction2={}
    # for I_J in m.I_J:
    #     positcui=positcui+1
    #     I=I_J[0]
    #     J=I_J[1]
    #     m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
    #     setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
    #     def _YR2init(m,ordered_set2):
    #         if ordered_set2== x_initial[positcui]-1:
    #             return True
    #         else:
    #             return False       
    #     m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
    #     setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

    #     def _select_one2(m):
    #         return pe.exactly(1,m.YR2[I,J])
    #     m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
    #     setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

    #     def _build_YR2_Disjunct(m,indexN):
    #         def _DEF_Nref(m):
    #             return m.model().Nref[I,J]==indexN
    #         m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
    #     m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
    #     setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

    #     # Create disjunction
    #     def Disjunction2(m):   
    #         return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
    #     m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
    #     setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # # Associate disjuncts with boolean variables
    #     for index in m.ordered_set2[I,J]:
    #         m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
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
            return m.TCP1+m.TCP2+m.TCP3+m.TMC-m.SALES  
            # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)      
    return m

# new V2: THIS IS THE FINAL VERSION AND THE ONE I AM USING TO OBTAIN THE RESULTS FOR THE ARTICLE.
def case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3},sequential: bool=False):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3','U4'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CAIN=pe.Param(initialize=10.62,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Fixed initial volume for dynamic tast [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    # def _rule_beta_time(m,I,J):
    #     if m.I_i_j_prod[I,J]==1:
    #         return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
    #     else:
    #         return 0 
    # m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    # def _rule_ineqrel_1(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    # def _rule_ineqrel_2(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    # m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    # m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    # ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={} 

    m.CA0={} 
    m.CB0={}
    m.CT0={}
    m.c_defCT0={}

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct




                m.CA0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CAIN),doc='Initial composition of A [kmol/m^3]')
                setattr(m,'CA0_%s_%s_%s' %(I,J,T),m.CA0[I,J,T])

                m.CB0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CBIN),doc='Initial composition of B [kmol/m^3]')
                setattr(m,'CB0_%s_%s_%s' %(I,J,T),m.CB0[I,J,T])

                m.CT0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(1e-3,m.CAIN+m.CBIN),doc='Initial total composition  [kmol/m^3]')
                setattr(m,'CT0_%s_%s_%s' %(I,J,T),m.CT0[I,J,T])

                def _defCT0(m):
                    return m.CT0[I,J,T]== m.CA0[I,J,T]+m.CB0[I,J,T]
                m.c_defCT0[I,J,T] = pe.Constraint(rule=_defCT0)
                setattr(m,'c_defCT0_%s_%s_%s' %(I,J,T),m.c_defCT0[I,J,T])       

                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0[I,J,T]))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0[I,J,T])))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0[I,J,T]))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*2 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*2, ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   

    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking2_3={} #rho and Vol relationship 
    m.linking2_4={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2']*m.Vol[I,J,T][N]-m.V0*((m.CA0[I,J,T]/m.CAIN))<=(m.beta_max[I,J])*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*((m.CA0[I,J,T]/m.CAIN))-m.rho_minus[I,'S2']*m.Vol[I,J,T][N]<=m.V0*(m.CAIN/m.CAIN)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])



                def _linking2_3(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0<=(m.beta_max[I,J]+m.V0)*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_3,doc='') 
                setattr(m,'linking2_3_%s_%s_%s' %(I,J,T),m.linking2_3[I,J,T])

                def _linking2_4(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0)<=(m.V0*(m.CBIN/m.CBIN)+m.beta_max[I,J]-m.V0)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_4[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_4,doc='') 
                setattr(m,'linking2_4_%s_%s_%s' %(I,J,T),m.linking2_4[I,J,T])




                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    m.ordered_set2={}
    m.YR2={}
    m.oneYR2={}
    m.YR2_Disjunct={}
    m.Disjunction2={}
    for I_J in m.I_J:
        positcui=positcui+1
        I=I_J[0]
        J=I_J[1]
        m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
        setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
        def _YR2init(m,ordered_set2):
            if ordered_set2== x_initial[positcui]-1:
                return True
            else:
                return False       
        m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
        setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

        def _select_one2(m):
            return pe.exactly(1,m.YR2[I,J])
        m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
        setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

        def _build_YR2_Disjunct(m,indexN):
            def _DEF_Nref(m):
                return m.model().Nref[I,J]==indexN
            m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
        m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
        setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

        # Create disjunction
        def Disjunction2(m):   
            return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
        m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
        setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # Associate disjuncts with boolean variables
        for index in m.ordered_set2[I,J]:
            m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.NonNegativeReals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
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
            return m.TCP1+m.TCP2+m.TCP3+m.TMC-m.SALES  
            # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        if sequential:
            def _obj_scheduling(m):
                return ( m.TCP1+m.TCP2+m.TMC-m.SALES  )
            m.obj_scheduling = pe.Objective(rule=_obj_scheduling, sense=pe.minimize)  
            
            def _obj_dummy(m):
                return 1
            m.obj_dummy = pe.Objective(rule=_obj_dummy, sense=pe.minimize) 

    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)      
    return m


# used to minimize variable processing times. s.t. max capacity and satisfy product requirements.
def case_2_scheduling_control_gdp_var_proc_time_min_proc_time(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3},sequential: bool=False,max_capacity: bool=False):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3','U4'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CAIN=pe.Param(initialize=10.62,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Maximum capacity [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Maximum capacity [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    # def _rule_beta_time(m,I,J):
    #     if m.I_i_j_prod[I,J]==1:
    #         return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
    #     else:
    #         return 0 
    # m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    # def _rule_ineqrel_1(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    # def _rule_ineqrel_2(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    # m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    # m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    # ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={} 

    m.CA0={} 
    m.CB0={}
    m.CT0={}
    m.c_defCT0={}

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct




                m.CA0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CAIN),doc='Initial composition of A [kmol/m^3]')
                setattr(m,'CA0_%s_%s_%s' %(I,J,T),m.CA0[I,J,T])

                m.CB0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CBIN),doc='Initial composition of B [kmol/m^3]')
                setattr(m,'CB0_%s_%s_%s' %(I,J,T),m.CB0[I,J,T])

                m.CT0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(1e-3,m.CAIN+m.CBIN),doc='Initial total composition  [kmol/m^3]')
                setattr(m,'CT0_%s_%s_%s' %(I,J,T),m.CT0[I,J,T])

                def _defCT0(m):
                    return m.CT0[I,J,T]== m.CA0[I,J,T]+m.CB0[I,J,T]
                m.c_defCT0[I,J,T] = pe.Constraint(rule=_defCT0)
                setattr(m,'c_defCT0_%s_%s_%s' %(I,J,T),m.c_defCT0[I,J,T])       

                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0[I,J,T]))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0[I,J,T])))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0[I,J,T]))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*(2) #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*(2) #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*(2) #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*(2), ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   

    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking2_3={} #rho and Vol relationship 
    m.linking2_4={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2']*m.Vol[I,J,T][N]-m.V0*((m.CA0[I,J,T]/m.CAIN))<=(m.beta_max[I,J])*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*((m.CA0[I,J,T]/m.CAIN))-m.rho_minus[I,'S2']*m.Vol[I,J,T][N]<=m.V0*(m.CAIN/m.CAIN)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])



                def _linking2_3(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0<=(m.beta_max[I,J]+m.V0)*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_3,doc='') 
                setattr(m,'linking2_3_%s_%s_%s' %(I,J,T),m.linking2_3[I,J,T])

                def _linking2_4(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0)<=(m.V0*(m.CBIN/m.CBIN)+m.beta_max[I,J]-m.V0)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_4[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_4,doc='') 
                setattr(m,'linking2_4_%s_%s_%s' %(I,J,T),m.linking2_4[I,J,T])




                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    # m.ordered_set2={}
    # m.YR2={}
    # m.oneYR2={}
    # m.YR2_Disjunct={}
    # m.Disjunction2={}
    # for I_J in m.I_J:
    #     positcui=positcui+1
    #     I=I_J[0]
    #     J=I_J[1]
    #     m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
    #     setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
    #     def _YR2init(m,ordered_set2):
    #         if ordered_set2== x_initial[positcui]-1:
    #             return True
    #         else:
    #             return False       
    #     m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
    #     setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

    #     def _select_one2(m):
    #         return pe.exactly(1,m.YR2[I,J])
    #     m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
    #     setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

    #     def _build_YR2_Disjunct(m,indexN):
    #         def _DEF_Nref(m):
    #             return m.model().Nref[I,J]==indexN
    #         m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
    #     m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
    #     setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

    #     # Create disjunction
    #     def Disjunction2(m):   
    #         return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
    #     m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
    #     setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # # Associate disjuncts with boolean variables
    #     for index in m.ordered_set2[I,J]:
    #         m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.NonNegativeReals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
    m.TMC= pe.Var(within=pe.Reals,initialize=0,doc='TMC: Total material cost')
    def _C_TMC(m):
        return m.TMC==sum(m.raw_cost[K]*(m.S0[K]-m.S[K, m.lastT]) for K in m.K_inputs) 
    m.C_TMC=pe.Constraint(rule=_C_TMC)
    m.SALES=pe.Var(within=pe.Reals,initialize=0,doc='SALES: Revenue form selling products')
    def _C_SALES(m):
        return m.SALES==sum(m.revenue[K]*m.S[K, m.lastT] for K in m.K_products)
    m.C_SALES=pe.Constraint(rule=_C_SALES)



    if obj_type=='profit_max':
        if max_capacity:
            def _obj(m):
                return -sum(sum(sum(m.X[I, J, T]*m.B[I,J,T]  for J in m.J_dynamics)  for I in m.I_dynamics)  for T in m.T)  
                # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
            m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        else:
            def _obj(m):
                return sum(sum(sum(m.X[I, J, T]*m.varTime[I,J,T]  for J in m.J_dynamics)  for I in m.I_dynamics)  for T in m.T)  
                # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
            m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        if sequential:
            def _obj_scheduling(m):
                return ( m.TCP1+m.TCP2+m.TMC-m.SALES   +   sum(sum(m.YR_disjunct[I,J][index].indicator_var for index in m.ordered_set[I,J]) for (I,J) in m.I_J if m.I_i_j_prod[I,J]==1)  ) 
            m.obj_scheduling = pe.Objective(rule=_obj_scheduling, sense=pe.minimize)  
            
            def _obj_dummy(m):
                return 1
            m.obj_dummy = pe.Objective(rule=_obj_dummy, sense=pe.minimize) 

    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)      
    return m



# WITH DISTILLATION
def case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3},sequential: bool=False):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_distil=pe.Set(initialize=['U4'],within=m.J)
    m.I_distil=pe.Set(initialize=['T5'],within=m.I)
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics-m.I_distil
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CAIN=pe.Param(initialize=10.62,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Fixed initial volume for dynamic tast [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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

    _rho_plus['T5','S6']=0.2
    _rho_plus['T5','S9']=0.8
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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    # _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    # def _rule_beta_time(m,I,J):
    #     if m.I_i_j_prod[I,J]==1:
    #         return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
    #     else:
    #         return 0 
    # m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    # def _rule_ineqrel_1(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    # def _rule_ineqrel_2(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    # m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    # m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    # ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={} 

    m.CA0={} 
    m.CB0={}
    m.CT0={}
    m.c_defCT0={}

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


                m.CA0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CAIN),doc='Initial composition of A [kmol/m^3]')
                setattr(m,'CA0_%s_%s_%s' %(I,J,T),m.CA0[I,J,T])

                m.CB0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CBIN),doc='Initial composition of B [kmol/m^3]')
                setattr(m,'CB0_%s_%s_%s' %(I,J,T),m.CB0[I,J,T])

                m.CT0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(1e-3,m.CAIN+m.CBIN),doc='Initial total composition  [kmol/m^3]')
                setattr(m,'CT0_%s_%s_%s' %(I,J,T),m.CT0[I,J,T])

                def _defCT0(m):
                    return m.CT0[I,J,T]== m.CA0[I,J,T]+m.CB0[I,J,T]
                m.c_defCT0[I,J,T] = pe.Constraint(rule=_defCT0)
                setattr(m,'c_defCT0_%s_%s_%s' %(I,J,T),m.c_defCT0[I,J,T])       

                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0[I,J,T]))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0[I,J,T])))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0[I,J,T]))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*2 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*2, ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   



    # DISTILLATION COLUMN CONSTRAINTS
    N_imp = 10
    x0 = 0.8
    a = 2.5
    V_up = 1000 #upper bound
    HT = 0.01
    HC = 0.1
    CV = 2.309
    ZS = 0.5
    xdset = 0.95

    cost_distillation=5/100

    m.dist_models={} #distillation column models

    m.dist_linking1_1={} #B and Hold-up relationship
    m.dist_linking1_2={} #B and Hold-up relationship

    m.dist_linking2_1={} #rho and hold-up distillate relationship
    m.dist_linking2_2={} #rho and hold-up distillate relationship

    m.dist_linking3_1={} #end point onstraint: final product requirement

    m.dist_linking4_1={} #Processing time
    m.dist_linking4_2={} #Processing time

    for I in m.I_distil:
        for J in m.J_distil:
            for T in m.T:
                m.dist_models[I,J,T]=create_distillation_model(N_imp, upper_t_h[(I,J)], x0, a, V_up, HT, HC, m.beta_max[I,J], CV, ZS, xdset)
                setattr(m,'dist_modelsL_%s_%s_%s' %(I,J,T),m.dist_models[I,J,T]) 

                #Constraints of this model I am not interested
                m.dist_models[I,J,T].del_component(m.dist_models[I,J,T].objective)
                m.dist_models[I,J,T].del_component(m.dist_models[I,J,T].xd_average_final_constraint)
                m.dist_models[I,J,T].del_component(m.dist_models[I,J,T].product_fraction_Rquirement)
    
                # DISTILLATION COLUMN LINKING CONSTRAINTS


                def _dist_linking1_1(m):
                    return m.B[I,J,T]-(m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)<=(m.beta_max[I,J]-(N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC))*(1-m.X[I,J,T])
                m.dist_linking1_1[I,J,T]=pe.Constraint(rule=_dist_linking1_1)
                setattr(m,'dist_linking1_1_%s_%s_%s' %(I,J,T),m.dist_linking1_1[I,J,T]) 

                def _dist_linking1_2(m):
                    return (m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)-m.B[I,J,T]<=(m.beta_max[I,J]+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)*(1-m.X[I,J,T])
                m.dist_linking1_2[I,J,T]=pe.Constraint(rule=_dist_linking1_2)
                setattr(m,'dist_linking1_2_%s_%s_%s' %(I,J,T),m.dist_linking1_2[I,J,T]) 


                def _dist_linking2_1(m):
                    return m.dist_models[I,J,T].I2[m.dist_models[I,J,T].T.last()] - m.rho_plus[I,'S9']*(m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)<=(m.beta_max[I,J]-m.rho_plus[I,'S9']*(N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC))*(1-m.X[I,J,T])
                m.dist_linking2_1[I,J,T]=pe.Constraint(rule=_dist_linking2_1)
                setattr(m,'dist_linking2_1_%s_%s_%s' %(I,J,T),m.dist_linking2_1[I,J,T]) 

                def _dist_linking2_2(m):
                    return m.rho_plus[I,'S9']*(m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)-m.dist_models[I,J,T].I2[m.dist_models[I,J,T].T.last()]<=(m.rho_plus[I,'S9']*(m.beta_max[I,J]+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC))*(1-m.X[I,J,T])
                m.dist_linking2_2[I,J,T]=pe.Constraint(rule=_dist_linking2_2)
                setattr(m,'dist_linking2_2_%s_%s_%s' %(I,J,T),m.dist_linking2_2[I,J,T]) 


                def _dist_linking3_1(m):
                    return m.dist_models[I,J,T].xdset-m.dist_models[I,J,T].xd_average[m.dist_models[I,J,T].T.last()] <= m.dist_models[I,J,T].xdset*(1-m.X[I,J,T]) 
                m.dist_linking3_1[I,J,T]=pe.Constraint(rule=_dist_linking3_1)
                setattr(m,'dist_linking3_1_%s_%s_%s' %(I,J,T),m.dist_linking3_1[I,J,T]) 


                def _dist_linking4_1(m):
                    return  m.dist_models[I,J,T].variableTime-m.varTime[I,J,T] <=(upper_t_h[(I,J)])*(1-m.X[I,J,T])
                m.dist_linking4_1[I,J,T]=pe.Constraint(rule=_dist_linking4_1)
                setattr(m,'dist_linking4_1_%s_%s_%s' %(I,J,T),m.dist_linking4_1[I,J,T]) 

                def _dist_linking4_2(m):
                    return  m.varTime[I,J,T]-m.dist_models[I,J,T].variableTime  <=(upper_t_h[(I,J)])*(1-m.X[I,J,T])
                m.dist_linking4_2[I,J,T]=pe.Constraint(rule=_dist_linking4_2)
                setattr(m,'dist_linking4_2_%s_%s_%s' %(I,J,T),m.dist_linking4_2[I,J,T]) 



    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking2_3={} #rho and Vol relationship 
    m.linking2_4={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2']*m.Vol[I,J,T][N]-m.V0*((m.CA0[I,J,T]/m.CAIN))<=(m.beta_max[I,J])*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*((m.CA0[I,J,T]/m.CAIN))-m.rho_minus[I,'S2']*m.Vol[I,J,T][N]<=m.V0*(m.CAIN/m.CAIN)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])



                def _linking2_3(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0<=(m.beta_max[I,J]+m.V0)*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_3,doc='') 
                setattr(m,'linking2_3_%s_%s_%s' %(I,J,T),m.linking2_3[I,J,T])

                def _linking2_4(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0)<=(m.V0*(m.CBIN/m.CBIN)+m.beta_max[I,J]-m.V0)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_4[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_4,doc='') 
                setattr(m,'linking2_4_%s_%s_%s' %(I,J,T),m.linking2_4[I,J,T])




                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    m.ordered_set2={}
    m.YR2={}
    m.oneYR2={}
    m.YR2_Disjunct={}
    m.Disjunction2={}
    for I_J in m.I_J:
        positcui=positcui+1
        I=I_J[0]
        J=I_J[1]
        m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
        setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
        def _YR2init(m,ordered_set2):
            if ordered_set2== x_initial[positcui]-1:
                return True
            else:
                return False       
        m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
        setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

        def _select_one2(m):
            return pe.exactly(1,m.YR2[I,J])
        m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
        setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

        def _build_YR2_Disjunct(m,indexN):
            def _DEF_Nref(m):
                return m.model().Nref[I,J]==indexN
            m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
        m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
        setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

        # Create disjunction
        def Disjunction2(m):   
            return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
        m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
        setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # Associate disjuncts with boolean variables
        for index in m.ordered_set2[I,J]:
            m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.NonNegativeReals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)   +    sum(sum(sum(m.X[I, J, T]*( cost_distillation*m.dist_models[I,J,T].I_V[m.dist_models[I,J,T].T.last()]  ) for T in m.T) for I in m.I_distil)for J in m.J_distil)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
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
            return m.TCP1+m.TCP2+m.TCP3+m.TMC-m.SALES  
            # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        if sequential:
            def _obj_scheduling(m):
                return ( m.TCP1+m.TCP2+m.TMC-m.SALES  )
            m.obj_scheduling = pe.Objective(rule=_obj_scheduling, sense=pe.minimize)  
            
            def _obj_dummy(m):
                return 1
            m.obj_dummy = pe.Objective(rule=_obj_dummy, sense=pe.minimize) 

    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)      
    return m
#for sequential naive
def case_2_scheduling_control_gdp_var_proc_time_min_proc_time_with_distillation(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3},sequential: bool=False, max_capacity: bool=False):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_distil=pe.Set(initialize=['U4'],within=m.J)
    m.I_distil=pe.Set(initialize=['T5'],within=m.I)
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics-m.I_distil
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CAIN=pe.Param(initialize=10.62,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Fixed initial volume for dynamic tast [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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

    _rho_plus['T5','S6']=0.2
    _rho_plus['T5','S9']=0.8
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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    # _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    # ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    # def _rule_beta_time(m,I,J):
    #     if m.I_i_j_prod[I,J]==1:
    #         return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
    #     else:
    #         return 0 
    # m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


    # def _rule_ineqrel_1(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]>=0
    # def _rule_ineqrel_2(m,I,J,T):
    #     if  m.I_i_j_prod[I,J]!=1:
    #         return pe.Constraint.Skip
    #     else:
    #         return m.varTime[I,J,T]-m.beta_time[I,J]*m.B[I,J,T]<=upper_t_h[(I,J)]*(1-m.X[I,J,T])
    
    # m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    # m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    # ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={} 

    m.CA0={} 
    m.CB0={}
    m.CT0={}
    m.c_defCT0={}

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


                m.CA0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CAIN),doc='Initial composition of A [kmol/m^3]')
                setattr(m,'CA0_%s_%s_%s' %(I,J,T),m.CA0[I,J,T])

                m.CB0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CBIN),doc='Initial composition of B [kmol/m^3]')
                setattr(m,'CB0_%s_%s_%s' %(I,J,T),m.CB0[I,J,T])

                m.CT0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(1e-3,m.CAIN+m.CBIN),doc='Initial total composition  [kmol/m^3]')
                setattr(m,'CT0_%s_%s_%s' %(I,J,T),m.CT0[I,J,T])

                def _defCT0(m):
                    return m.CT0[I,J,T]== m.CA0[I,J,T]+m.CB0[I,J,T]
                m.c_defCT0[I,J,T] = pe.Constraint(rule=_defCT0)
                setattr(m,'c_defCT0_%s_%s_%s' %(I,J,T),m.c_defCT0[I,J,T])       

                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0[I,J,T]))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0[I,J,T])))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0[I,J,T]))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*2 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*2, ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   



    # DISTILLATION COLUMN CONSTRAINTS
    N_imp = 10
    x0 = 0.8
    a = 2.5
    V_up = 1000 #upper bound
    HT = 0.01
    HC = 0.1
    CV = 2.309
    ZS = 0.5
    xdset = 0.95

    cost_distillation=5/100

    m.dist_models={} #distillation column models

    m.dist_linking1_1={} #B and Hold-up relationship
    m.dist_linking1_2={} #B and Hold-up relationship

    m.dist_linking2_1={} #rho and hold-up distillate relationship
    m.dist_linking2_2={} #rho and hold-up distillate relationship

    m.dist_linking3_1={} #end point onstraint: final product requirement

    m.dist_linking4_1={} #Processing time
    m.dist_linking4_2={} #Processing time

    for I in m.I_distil:
        for J in m.J_distil:
            for T in m.T:
                m.dist_models[I,J,T]=create_distillation_model(N_imp, upper_t_h[(I,J)], x0, a, V_up, HT, HC, m.beta_max[I,J], CV, ZS, xdset)
                setattr(m,'dist_modelsL_%s_%s_%s' %(I,J,T),m.dist_models[I,J,T]) 

                #Constraints of this model I am not interested
                m.dist_models[I,J,T].del_component(m.dist_models[I,J,T].objective)
                m.dist_models[I,J,T].del_component(m.dist_models[I,J,T].xd_average_final_constraint)
                m.dist_models[I,J,T].del_component(m.dist_models[I,J,T].product_fraction_Rquirement)
    
                # DISTILLATION COLUMN LINKING CONSTRAINTS


                def _dist_linking1_1(m):
                    return m.B[I,J,T]-(m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)<=(m.beta_max[I,J]-(N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC))*(1-m.X[I,J,T])
                m.dist_linking1_1[I,J,T]=pe.Constraint(rule=_dist_linking1_1)
                setattr(m,'dist_linking1_1_%s_%s_%s' %(I,J,T),m.dist_linking1_1[I,J,T]) 

                def _dist_linking1_2(m):
                    return (m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)-m.B[I,J,T]<=(m.beta_max[I,J]+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)*(1-m.X[I,J,T])
                m.dist_linking1_2[I,J,T]=pe.Constraint(rule=_dist_linking1_2)
                setattr(m,'dist_linking1_2_%s_%s_%s' %(I,J,T),m.dist_linking1_2[I,J,T]) 


                def _dist_linking2_1(m):
                    return m.dist_models[I,J,T].I2[m.dist_models[I,J,T].T.last()] - m.rho_plus[I,'S9']*(m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)<=(m.beta_max[I,J]-m.rho_plus[I,'S9']*(N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC))*(1-m.X[I,J,T])
                m.dist_linking2_1[I,J,T]=pe.Constraint(rule=_dist_linking2_1)
                setattr(m,'dist_linking2_1_%s_%s_%s' %(I,J,T),m.dist_linking2_1[I,J,T]) 

                def _dist_linking2_2(m):
                    return m.rho_plus[I,'S9']*(m.dist_models[I,J,T].HB0var+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC)-m.dist_models[I,J,T].I2[m.dist_models[I,J,T].T.last()]<=(m.rho_plus[I,'S9']*(m.beta_max[I,J]+N_imp*m.dist_models[I,J,T].HT+m.dist_models[I,J,T].HC))*(1-m.X[I,J,T])
                m.dist_linking2_2[I,J,T]=pe.Constraint(rule=_dist_linking2_2)
                setattr(m,'dist_linking2_2_%s_%s_%s' %(I,J,T),m.dist_linking2_2[I,J,T]) 


                def _dist_linking3_1(m):
                    return m.dist_models[I,J,T].xdset-m.dist_models[I,J,T].xd_average[m.dist_models[I,J,T].T.last()] <= m.dist_models[I,J,T].xdset*(1-m.X[I,J,T]) 
                m.dist_linking3_1[I,J,T]=pe.Constraint(rule=_dist_linking3_1)
                setattr(m,'dist_linking3_1_%s_%s_%s' %(I,J,T),m.dist_linking3_1[I,J,T]) 


                def _dist_linking4_1(m):
                    return  m.dist_models[I,J,T].variableTime-m.varTime[I,J,T] <=(upper_t_h[(I,J)])*(1-m.X[I,J,T])
                m.dist_linking4_1[I,J,T]=pe.Constraint(rule=_dist_linking4_1)
                setattr(m,'dist_linking4_1_%s_%s_%s' %(I,J,T),m.dist_linking4_1[I,J,T]) 

                def _dist_linking4_2(m):
                    return  m.varTime[I,J,T]-m.dist_models[I,J,T].variableTime  <=(upper_t_h[(I,J)])*(1-m.X[I,J,T])
                m.dist_linking4_2[I,J,T]=pe.Constraint(rule=_dist_linking4_2)
                setattr(m,'dist_linking4_2_%s_%s_%s' %(I,J,T),m.dist_linking4_2[I,J,T]) 



    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking2_3={} #rho and Vol relationship 
    m.linking2_4={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2']*m.Vol[I,J,T][N]-m.V0*((m.CA0[I,J,T]/m.CAIN))<=(m.beta_max[I,J])*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*((m.CA0[I,J,T]/m.CAIN))-m.rho_minus[I,'S2']*m.Vol[I,J,T][N]<=m.V0*(m.CAIN/m.CAIN)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])



                def _linking2_3(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0<=(m.beta_max[I,J]+m.V0)*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_3,doc='') 
                setattr(m,'linking2_3_%s_%s_%s' %(I,J,T),m.linking2_3[I,J,T])

                def _linking2_4(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0)<=(m.V0*(m.CBIN/m.CBIN)+m.beta_max[I,J]-m.V0)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_4[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_4,doc='') 
                setattr(m,'linking2_4_%s_%s_%s' %(I,J,T),m.linking2_4[I,J,T])




                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    # m.ordered_set2={}
    # m.YR2={}
    # m.oneYR2={}
    # m.YR2_Disjunct={}
    # m.Disjunction2={}
    # for I_J in m.I_J:
    #     positcui=positcui+1
    #     I=I_J[0]
    #     J=I_J[1]
    #     m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
    #     setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
    #     def _YR2init(m,ordered_set2):
    #         if ordered_set2== x_initial[positcui]-1:
    #             return True
    #         else:
    #             return False       
    #     m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
    #     setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

    #     def _select_one2(m):
    #         return pe.exactly(1,m.YR2[I,J])
    #     m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
    #     setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

    #     def _build_YR2_Disjunct(m,indexN):
    #         def _DEF_Nref(m):
    #             return m.model().Nref[I,J]==indexN
    #         m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
    #     m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
    #     setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

    #     # Create disjunction
    #     def Disjunction2(m):   
    #         return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
    #     m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
    #     setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # # Associate disjuncts with boolean variables
    #     for index in m.ordered_set2[I,J]:
    #         m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.NonNegativeReals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)   +    sum(sum(sum(m.X[I, J, T]*( cost_distillation*m.dist_models[I,J,T].I_V[m.dist_models[I,J,T].T.last()]  ) for T in m.T) for I in m.I_distil)for J in m.J_distil)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
    m.TMC= pe.Var(within=pe.Reals,initialize=0,doc='TMC: Total material cost')
    def _C_TMC(m):
        return m.TMC==sum(m.raw_cost[K]*(m.S0[K]-m.S[K, m.lastT]) for K in m.K_inputs) 
    m.C_TMC=pe.Constraint(rule=_C_TMC)
    m.SALES=pe.Var(within=pe.Reals,initialize=0,doc='SALES: Revenue form selling products')
    def _C_SALES(m):
        return m.SALES==sum(m.revenue[K]*m.S[K, m.lastT] for K in m.K_products)
    m.C_SALES=pe.Constraint(rule=_C_SALES)


    if obj_type=='profit_max':
        if max_capacity:
            def _obj(m):
                return -sum(sum(sum(m.X[I, J, T]*m.B[I,J,T]  for J in m.J_dynamics)  for I in m.I_dynamics)  for T in m.T)  
                # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
            m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        else:
            def _obj(m):
                return sum(sum(sum(m.X[I, J, T]*m.varTime[I,J,T]  for J in m.J_dynamics)  for I in m.I_dynamics)  for T in m.T)  
                # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
            m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        if sequential:
            def _obj_scheduling(m):
                return ( m.TCP1+m.TCP2+m.TMC-m.SALES   +   sum(sum(m.YR_disjunct[I,J][index].indicator_var for index in m.ordered_set[I,J]) for (I,J) in m.I_J if m.I_i_j_prod[I,J]==1)  ) 
            m.obj_scheduling = pe.Objective(rule=_obj_scheduling, sense=pe.minimize)  
            
            def _obj_dummy(m):
                return 1
            m.obj_dummy = pe.Objective(rule=_obj_dummy, sense=pe.minimize) 

    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)           

    return m



## DO NOT USE!!!!!
# same as previuous, except that considers the relationship between tau and B for processes without dynamic models i.e. it has more variable processing times
def case_2_scheduling_control_gdp_var_proc_time_simplified_2(x_initial: list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], obj_type: str='profit_max',last_disc_point: float=12,last_time_hours: float=12,lower_t_h: dict={('T1','U1'):1,('T2','U2'):1,('T2','U3'):1,('T3','U2'):1,('T3','U3'):1,('T4','U2'):1,('T4','U3'):4,('T5','U4'):1},upper_t_h: dict={('T1','U1'):2,('T2','U2'):2,('T2','U3'):3,('T3','U2'):2,('T3','U3'):6,('T4','U2'):2,('T4','U3'):6,('T5','U4'):3}):

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_gdp_var_proc_time')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=last_time_hours/last_disc_point,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=last_disc_point,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')
    #Subsets
    m.J_dynamics=pe.Set(initialize=['U2','U3'],within=m.J)
    m.I_dynamics=pe.Set(initialize=['T2'],within=m.I)   
    m.J_noDynamics=pe.Set(initialize=['U1','U2','U3','U4'],within=m.J)
    m.I_noDynamics=m.I-m.I_dynamics
    m.K_inputs=pe.Set(initialize=['S1','S2','S3'],within=m.K)
    m.K_products=pe.Set(initialize=['S8','S9'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=last_time_hours, doc='scheduling horizon [units of nntime]')
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')
    # -----------parameters--------------------------------------------------
    # REACTOR MODEL
    m.CC0=pe.Param(initialize=0,doc='Required initial composition inside reactor for this reaction and component [kmol/m^3]')
    m.CAIN=pe.Param(initialize=10.62,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CBIN=pe.Param(initialize=20,doc='Concentration of B in inlet flow [kmol/m^3]')
    m.CCDESIRED=pe.Param(initialize=4,doc='Desired concentration of C [kmol/m^3]')
    m.TBIN=pe.Param(initialize=293.15, doc='Inlet temperature of feed B [K]')
    m.V0=pe.Param(initialize=1,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax2=pe.Param(initialize=5,doc='Fixed initial volume for dynamic tast [m^3]')
    m.Vmax3=pe.Param(initialize=8,doc='Fixed initial volume for dynamic tast [m^3]')
    # m.qrmax=pe.Param(initialize=(1.5e+5)*(1/1000)*(m.V0/0.001),doc='upper bound on the heat rate produced by the reaction [kJ/h]') #TODO: check if assumed linear relationship holds

    m.k10=pe.Param(initialize=4,doc='[m^3/kmol h]')
    m.k20=pe.Param(initialize=800*(0.001),doc='  [m^3/h]')
    m.E1=pe.Param(initialize=6e+3,doc='  [kJ/kmol]')
    m.E2=pe.Param(initialize=20e+3,doc='  [kJ/kmol]')
    m.R=pe.Param(initialize=8.31,doc='  [kJ/kmol K]')
    m.DH1=pe.Param(initialize=-3e+4,doc='  [kJ/kmol]')
    m.DH2=pe.Param(initialize=-1e+4,doc='  [kJ/kmol]')
    m.CP=pe.Param(initialize=75, doc='kJ/ kmol K')


    m.v_J=pe.Param(m.J_dynamics,initialize={'U3':0.5,'U2':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_dynamics,initialize={'U3':1e+3,'U2':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_dynamics,initialize={'U3':4.2,'U2':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_dynamics,initialize={'U3':3e+4,'U2':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_dynamics,initialize={'U3':293.15,'U2':293.15},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_dynamics,initialize={'U3':323.15,'U2':323.15},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_dynamics,initialize={'U3':10,'U2':8},doc='Maximum flow rate of heating and cooling water [m^3/h]')


        # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_dynamics,initialize={'T2':293.15},doc='Initial condition for jacket temperatures [K]')

    # SCHEDULING
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
    _beta_min['T1','U1']=1

    _beta_min['T2','U2']=m.V0
    _beta_min['T2','U3']=m.V0

    _beta_min['T3','U2']=m.V0
    _beta_min['T3','U3']=m.V0

    _beta_min['T4','U2']=m.V0
    _beta_min['T4','U3']=m.V0

    _beta_min['T5','U4']=1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['T1','U1']=10

    _beta_max['T2','U2']=m.Vmax2
    _beta_max['T2','U3']=m.Vmax3

    _beta_max['T3','U2']=m.Vmax2
    _beta_max['T3','U3']=m.Vmax3

    _beta_max['T4','U2']=m.Vmax2
    _beta_max['T4','U3']=m.Vmax3

    _beta_max['T5','U4']=20
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400,'S4':100,'S5':15,'S6':50,'S7':100,'S8':400,'S9':400},default=0,doc="maximum amount of material k that can be stored [m^3]")
    
    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'S1':400,'S2':400,'S3':400},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

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
    _variable_cost_param['T1','U1']=10

    _variable_cost_param['T3','U2']=20
    _variable_cost_param['T3','U3']=30

    _variable_cost_param['T4','U2']=20
    _variable_cost_param['T4','U3']=35

    _variable_cost_param['T5','U4']=10
    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 50
        elif K=='S2': #A
            return 150
        elif K=='S3 ': #B
            return 200
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')


    def _revenue(m,K):
        if K=='S8':
            return 300 
        elif K=='S9':
            return 400
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
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

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    # TODO, IN THIS CASE I ASSUME AN EQUALITY CONSTRAINT
    if obj_type=='cost_min': 
        def _E_DEMAND_SATISFACTION(m,K):
            return m.S[K,m.lastT]==m.demand[K,m.lastT]
        m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
               
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    #*****DISJUNCTIVE SECTION**********************************   
#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _minTau={}
    # _minTau['T1','U1']=math.ceil(1/m.delta)

    # _minTau['T2','U2']=math.ceil(1/m.delta)
    # _minTau['T2','U3']=math.ceil(1/m.delta)

    # _minTau['T3','U2']=math.ceil(1/m.delta)
    # _minTau['T3','U3']=math.ceil(1/m.delta)

    # _minTau['T4','U2']=math.ceil(1/m.delta)
    # _minTau['T4','U3']=math.ceil(4/m.delta)

    # _minTau['T5','U4']=math.ceil(1/m.delta)

    # _minTau['T1','U1']=1

    # _minTau['T2','U2']=1
    # _minTau['T2','U3']=2

    # _minTau['T3','U2']=1
    # _minTau['T3','U3']=3

    # _minTau['T4','U2']=1
    # _minTau['T4','U3']=5

    # _minTau['T5','U4']=2
    def _minTau_rule(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return math.ceil(lower_t_h[(I,J)]/m.delta)
        else:
            return 0
    m.minTau=pe.Param(m.I,m.J,initialize=_minTau_rule,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    # _maxTau={}
    # _maxTau['T1','U1']=math.ceil(2/m.delta)

    # _maxTau['T2','U2']=math.ceil(2/m.delta)
    # _maxTau['T2','U3']=math.ceil(3/m.delta)

    # _maxTau['T3','U2']=math.ceil(2/m.delta)
    # _maxTau['T3','U3']=math.ceil(6/m.delta)

    # _maxTau['T4','U2']=math.ceil(2/m.delta)
    # _maxTau['T4','U3']=math.ceil(6/m.delta)

    # _maxTau['T5','U4']=math.ceil(3/m.delta)

    # _maxTau['T1','U1']=1

    # _maxTau['T2','U2']=1
    # _maxTau['T2','U3']=2

    # _maxTau['T3','U2']=1
    # _maxTau['T3','U3']=3

    # _maxTau['T4','U2']=1
    # _maxTau['T4','U3']=5

    # _maxTau['T5','U4']=2
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
    m.varTime=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')


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
                m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
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
                m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
                setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

                # Declaration of disjuncts
                def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                    m.model().tau[I,J]=indexTau
                    m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                    # #----------- Variable processing times----------------------------------------------------------------
                    # TODO: CHANGE TO INEQUALITY AND ADD NEW CONSTRAINT RELATING varTime AND B outside disjunction
                    def _DEF_VAR_TIME(m,T):
                        return m.model().varTime[I,J,T]<=pe.value(m.model().tau_p[I,J])
                    m.DEF_VAR_TIME=pe.Constraint(m.model().T,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                    # m.DEF_VAR_TIME.display()

                    # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                    def _DEF_AUX1(m,T):
                        return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                    m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                    # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                    def _DEF_AUX2(m,T):
                        if T==0:        
                            return pe.Constraint.Skip
                        elif T-pe.value(m.model().tau[I,J])>=0:
                            return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                        else:
                            return m.model().B_shift[I,J,T]==0
                    m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
                m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
                setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
                
                #Create disjunction
                def Disjunction1(m):    #Disjunction for first Boolean variable
                    return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
                m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
                setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

                # Associate disjuncts with boolean variables
                for index in m.ordered_set[I,J]:
                    m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************
    ### THIS SECTION CONSIDERS THE RELATIONSHIP BETWEEN varTime and b for noDynamic tasks
    def _rule_beta_time(m,I,J):
        if m.I_i_j_prod[I,J]==1:
            return pe.value(m.tau_p[I,J])/m.beta_max[I,J] #TODO: Instead of writing this relationship, simply indicate the constant used.
        else:
            return 0 
    m.beta_time=pe.Param(m.I_noDynamics,m.J_noDynamics,initialize=_rule_beta_time,doc='constant that relates processing times and size of batches')


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
    
    m.ineq_rel_1=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_1)
    m.ineq_rel_2=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_rule_ineqrel_2)
    ### END OF THE SECTION

    #-----------Reactors dynamic models--------------------------------





    m.N={} #Continuous time set
    m.CA={}
    m.CB={}
    m.CC={}
    m.TRvar={}
    m.u_input={}
    m.Vol={}
    m.dCAdtheta={}
    m.dCBdtheta={}
    m.dCCdtheta={}
    m.dVdtheta={}
    # m.aux1={}
    # m.aux2={}
    # m.aux3={}
    # m.aux4={}

    m.c_dCAdtheta={}
    m.c_dCBdtheta={}
    m.c_dCCdtheta={}
    m.c_dVdtheta={}
    m.Integral={}
    m.dIntegraldtheta={}
    m.c_dIntegraldtheta={}


    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    m.c_dTRdtheta={}
    m.c_dTJdtheta={}

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={} 

    m.CA0={} 
    m.CB0={}
    m.CT0={}
    m.c_defCT0={}

    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.N[I,J,T]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #No units!!
                setattr(m,'N_%s_%s_%s' %(I,J,T),m.N[I,J,T]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct




                m.CA0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CAIN),doc='Initial composition of A [kmol/m^3]')
                setattr(m,'CA0_%s_%s_%s' %(I,J,T),m.CA0[I,J,T])

                m.CB0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(0,m.CBIN),doc='Initial composition of B [kmol/m^3]')
                setattr(m,'CB0_%s_%s_%s' %(I,J,T),m.CB0[I,J,T])

                m.CT0[I,J,T]=pe.Var(within=pe.NonNegativeReals,bounds=(1e-3,m.CAIN+m.CBIN),doc='Initial total composition  [kmol/m^3]')
                setattr(m,'CT0_%s_%s_%s' %(I,J,T),m.CT0[I,J,T])

                def _defCT0(m):
                    return m.CT0[I,J,T]== m.CA0[I,J,T]+m.CB0[I,J,T]
                m.c_defCT0[I,J,T] = pe.Constraint(rule=_defCT0)
                setattr(m,'c_defCT0_%s_%s_%s' %(I,J,T),m.c_defCT0[I,J,T])       

                def _CA_bounds(m,N):
                    return (0,100)
                m.CA[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CA_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CA_%s_%s_%s' %(I,J,T),m.CA[I,J,T])

                def _CB_bounds(m,N):
                    return (0,100)
                m.CB[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CB_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CB_%s_%s_%s' %(I,J,T),m.CB[I,J,T])

                def _CC_bounds(m,N):
                    return (0,100)
                m.CC[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_CC_bounds, doc='Component composition profile [kmol/m^3]') 
                setattr(m,'CC_%s_%s_%s' %(I,J,T),m.CC[I,J,T])

                def _TRvar_bounds(m,N):
                    return (293.15,323.15)  
                m.TRvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
                setattr(m,'TRvar_%s_%s_%s' %(I,J,T),m.TRvar[I,J,T])
                
                def _u_input_bounds(m,N):
                    return (0,5)
                m.u_input[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_u_input_bounds,doc='Feed rate of B with inlet concentration CBIN [m^3/h]')
                setattr(m,'u_input_%s_%s_%s' %(I,J,T),m.u_input[I,J,T])

                def _Vol_bounds(m,N):
                    return (m.model().beta_min[I,J],m.model().beta_max[I,J])
                m.Vol[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_Vol_bounds,doc='Variable reactor volume [m^3]')
                setattr(m,'Vol_%s_%s_%s' %(I,J,T),m.Vol[I,J,T])

                def _TJvar_bounds(m,N):
                    return (293.15,m.T_J_max[J]) 
                m.TJvar[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
                setattr(m,'TJvar_%s_%s_%s' %(I,J,T),m.TJvar[I,J,T])

                m.Fhot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fhot_%s_%s_%s' %(I,J,T),m.Fhot[I,J,T])

                m.Fcold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
                setattr(m,'Fcold_%s_%s_%s' %(I,J,T),m.Fcold[I,J,T])

                # m.aux1[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 1 (CA)')
                # setattr(m,'aux1_%s_%s_%s' %(I,J,T),m.aux1[I,J,T])

                # m.aux2[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 2 (CB)')
                # setattr(m,'aux2_%s_%s_%s' %(I,J,T),m.aux2[I,J,T])

                # m.aux3[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 3 (CC)')
                # setattr(m,'aux3_%s_%s_%s' %(I,J,T),m.aux3[I,J,T])

                # m.aux4[I,J,T]=pe.Var(m.N[I,J,T],within=pe.Reals,bounds=(-100,100),doc='Auxiliary variable for differential equation 4 (V)')
                # setattr(m,'aux4_%s_%s_%s' %(I,J,T),m.aux4[I,J,T])

                m.dCAdtheta[I,J,T] = dae.DerivativeVar(m.CA[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition A')
                setattr(m,'dCAdtheta_%s_%s_%s' %(I,J,T),m.dCAdtheta[I,J,T])               

                m.dCBdtheta[I,J,T] = dae.DerivativeVar(m.CB[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition B')
                setattr(m,'dCBdtheta_%s_%s_%s' %(I,J,T),m.dCBdtheta[I,J,T])

                m.dCCdtheta[I,J,T] = dae.DerivativeVar(m.CC[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dCCdtheta_%s_%s_%s' %(I,J,T),m.dCCdtheta[I,J,T])

                m.dVdtheta[I,J,T] = dae.DerivativeVar(m.Vol[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of composition C')
                setattr(m,'dVdtheta_%s_%s_%s' %(I,J,T),m.dVdtheta[I,J,T])

                m.dTRdtheta[I,J,T]=dae.DerivativeVar(m.TRvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of reactor temperature')
                setattr(m,'dTRdtheta_%s_%s_%s' %(I,J,T),m.dTRdtheta[I,J,T])

                m.dTJdtheta[I,J,T]=dae.DerivativeVar(m.TJvar[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of jacket temperature')
                setattr(m,'dTJdtheta_%s_%s_%s' %(I,J,T),m.dTJdtheta[I,J,T])

                def _dCAdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CA[I,J,T][N] == m.CA0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCAdtheta[I,J,T][N] == m.varTime[I,J,T]*(   -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CA[I,J,T][N]))       ) 
                m.c_dCAdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCAdtheta)
                setattr(m,'c_dCAdtheta_%s_%s_%s' %(I,J,T),m.c_dCAdtheta[I,J,T])

                def _dCBdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CB[I,J,T][N] == m.CB0[I,J,T] # Initial condition
                    else:                                    
                        return m.dCBdtheta[I,J,T][N] == m.varTime[I,J,T]*( -((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))      +    (((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*(m.CBIN-m.CB[I,J,T][N]))  ) 
                m.c_dCBdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCBdtheta)
                setattr(m,'c_dCBdtheta_%s_%s_%s' %(I,J,T),m.c_dCBdtheta[I,J,T])

                def _dCCdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.CC[I,J,T][N] == m.CC0 # Initial condition
                    else:                                    
                        return m.dCCdtheta[I,J,T][N] == m.varTime[I,J,T]*(  ((m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))    -((m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])       -(((m.u_input[I,J,T][N])/(m.Vol[I,J,T][N]))*m.CC[I,J,T][N]) ) 
                m.c_dCCdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dCCdtheta)
                setattr(m,'c_dCCdtheta_%s_%s_%s' %(I,J,T),m.c_dCCdtheta[I,J,T])

                def _dVdtheta(m,N):
                    if N == m.N[I,J,T].first(): 
                        return m.Vol[I,J,T][N] == m.V0 # Initial condition
                    else:                                    
                        return m.dVdtheta[I,J,T][N] == m.varTime[I,J,T]*(  m.u_input[I,J,T][N] ) 
                m.c_dVdtheta[I,J,T] = pe.Constraint(m.N[I,J,T], rule=_dVdtheta)
                setattr(m,'c_dVdtheta_%s_%s_%s' %(I,J,T),m.c_dVdtheta[I,J,T])


                def _dTRdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TRvar[I,J,T][N] == m.T_R_initial[I] #Initial condition
                    else:
                        return m.dTRdtheta[I,J,T][N] == m.varTime[I,J,T]*(((m.ua[J]*(m.TJvar[I,J,T][N]-m.TRvar[I,J,T][N]))/(m.V0*m.CP*m.CT0[I,J,T]))-(m.CBIN*m.u_input[I,J,T][N]*(m.TRvar[I,J,T][N]-m.TBIN)*(1/(m.V0*m.CT0[I,J,T])))-(((m.Vol[I,J,T][N])/(m.V0*m.CP*m.CT0[I,J,T]))*((m.DH1*(m.k10*pe.exp(-((m.E1)/(m.R*m.TRvar[I,J,T][N]))))*(m.CA[I,J,T][N])*(m.CB[I,J,T][N]))  +  (m.DH2*(m.k20*pe.exp(-((m.E2)/(m.R*m.TRvar[I,J,T][N]))))*m.CC[I,J,T][N])))) 
                m.c_dTRdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTRdtheta)
                setattr(m,'c_dTRdtheta_%s_%s_%s' %(I,J,T),m.c_dTRdtheta[I,J,T])
                # m.c_dTRdt[I,J].pprint()

                def _dTJdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.TJvar[I,J,T][N] == m.T_J_initial[I] #Initial condition
                    else:
                        return m.dTJdtheta[I,J,T][N] == m.varTime[I,J,T]*((((m.Fhot[I,J,T][N]*(m.T_H[J]-m.TJvar[I,J,T][N]))+(m.Fcold[I,J,T][N]*(m.T_C[J]-m.TJvar[I,J,T][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J,T][N]-m.TJvar[I,J,T][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
                m.c_dTJdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_dTJdtheta)
                setattr(m,'c_dTJdtheta_%s_%s_%s' %(I,J,T),m.c_dTJdtheta[I,J,T])


                # Integrals for cost calculation
                def _Integral_hot_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_hot[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
                setattr(m,'Integral_hot_%s_%s_%s' %(I,J,T),m.Integral_hot[I,J,T])
                def _Integral_cold_bounds(m,N):
                    return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
                m.Integral_cold[I,J,T]=pe.Var(m.N[I,J,T],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
                setattr(m,'Integral_cold_%s_%s_%s' %(I,J,T),m.Integral_cold[I,J,T])
                
                m.dIntegral_hotdtheta[I,J,T]=dae.DerivativeVar(m.Integral_hot[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of hot integral')
                setattr(m,'dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.dIntegral_hotdtheta[I,J,T])            
                m.dIntegral_colddtheta[I,J,T]=dae.DerivativeVar(m.Integral_cold[I,J,T], withrespectto=m.N[I,J,T], doc='Derivative of cold integral')
                setattr(m,'dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.dIntegral_colddtheta[I,J,T])


                def _c_dIntegral_hotdtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_hot[I,J,T][N]==0
                    else:
                        return m.dIntegral_hotdtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fhot[I,J,T][N]
                m.c_dIntegral_hotdtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_hotdtheta)
                setattr(m,'c_dIntegral_hotdtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_hotdtheta[I,J,T])   
                
                def _c_dIntegral_colddtheta(m,N):
                    if N == m.N[I,J,T].first():
                        return m.Integral_cold[I,J,T][N]==0
                    else:
                        return m.dIntegral_colddtheta[I,J,T][N]==m.varTime[I,J,T]*m.Fcold[I,J,T][N]
                m.c_dIntegral_colddtheta[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_c_dIntegral_colddtheta)
                setattr(m,'c_dIntegral_colddtheta_%s_%s_%s' %(I,J,T),m.c_dIntegral_colddtheta[I,J,T])  
 



    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    m.Constant_control3={}
    keep_constant_u=9*2 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_fcold=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 
    keep_constant_fhot=9*2 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T:
                discretizer.apply_to(m, nfe=30*2, ncp=3, wrt=m.N[I,J,T], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _Constant_control1(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_u!=0) or (N==m.N[I,J,T].last()):
                        return m.u_input[I,J,T][N] == m.u_input[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control1,doc='Constant control action every keep_constant_u discrete points and the last one')
                setattr(m,'Constant_control1_%s_%s_%s' %(I,J,T),m.Constant_control1[I,J,T])

                def _Constant_control2(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fhot!=0) or (N==m.N[I,J,T].last()):
                        return m.Fhot[I,J,T][N] == m.Fhot[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control2,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control2_%s_%s_%s' %(I,J,T),m.Constant_control2[I,J,T])  

                def _Constant_control3(m,N):
                    if (N!=m.N[I,J,T].first() and (m.N[I,J,T].ord(N)-1)%keep_constant_fcold!=0) or (N==m.N[I,J,T].last()):
                        return m.Fcold[I,J,T][N] == m.Fcold[I,J,T][m.N[I,J,T].prev(N)]
                    else:
                        return pe.Constraint.Skip
                m.Constant_control3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_Constant_control3,doc='Constant control action every keep_constant_temp discrete points and the last one')
                setattr(m,'Constant_control3_%s_%s_%s' %(I,J,T),m.Constant_control3[I,J,T])   

    # # ----------Linking constraints-------------------------------------------
# TODO: discretize models before linking constraints
# In this case I will create disjunctions that will activate and deactivate constraints depending on the value of Xijt

    m.linking1_1={} #B and Vol relationship 
    m.linking1_2={} #B and Vol relationship 

    m.linking2_1={} #rho and Vol relationship 
    m.linking2_2={} #rho and Vol relationship 

    m.linking2_3={} #rho and Vol relationship 
    m.linking2_4={} #rho and Vol relationship 

    m.linking3_1={} #end point constraint relationship 
    m.linking3_2={} #end point constraint relationship 

    for I in m.I_dynamics:
        for J in m.J_dynamics: 
            for T in m.T: 
                def _linking1_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.B[I,J,T]-m.Vol[I,J,T][N] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
                    else:
                        return pe.Constraint.Skip
                m.linking1_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_1_%s_%s_%s' %(I,J,T),m.linking1_1[I,J,T])

                def _linking1_2(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.B[I,J,T]-m.Vol[I,J,T][N]) <= m.beta_max[I,J]*(1-m.X[I,J,T]) 
                    else:
                        return pe.Constraint.Skip 
                m.linking1_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
                setattr(m,'linking1_2_%s_%s_%s' %(I,J,T),m.linking1_2[I,J,T])

                def _linking2_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S2']*m.Vol[I,J,T][N]-m.V0*((m.CA0[I,J,T]/m.CAIN))<=(m.beta_max[I,J])*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_1,doc='') 
                setattr(m,'linking2_1_%s_%s_%s' %(I,J,T),m.linking2_1[I,J,T])

                def _linking2_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.V0*((m.CA0[I,J,T]/m.CAIN))-m.rho_minus[I,'S2']*m.Vol[I,J,T][N]<=m.V0*(m.CAIN/m.CAIN)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_2,doc='') 
                setattr(m,'linking2_2_%s_%s_%s' %(I,J,T),m.linking2_2[I,J,T])



                def _linking2_3(m,N):
                    if N==m.N[I,J,T].last():
                        return m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0<=(m.beta_max[I,J]+m.V0)*(1-m.X[I,J,T]) 
                    else:  
                        return pe.Constraint.Skip
                m.linking2_3[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_3,doc='') 
                setattr(m,'linking2_3_%s_%s_%s' %(I,J,T),m.linking2_3[I,J,T])

                def _linking2_4(m,N):
                    if N==m.N[I,J,T].last():
                        return -(m.rho_minus[I,'S3']*m.Vol[I,J,T][N]-m.V0*((m.CB0[I,J,T]/m.CBIN))-m.Vol[I,J,T][N]+m.V0)<=(m.V0*(m.CBIN/m.CBIN)+m.beta_max[I,J]-m.V0)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking2_4[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking2_4,doc='') 
                setattr(m,'linking2_4_%s_%s_%s' %(I,J,T),m.linking2_4[I,J,T])




                def _linking3_1(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CC[I,J,T][N]-m.CCDESIRED<=(100-m.CCDESIRED)*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip
                m.linking3_1[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_1,doc='')
                setattr(m,'linking3_1_%s_%s_%s' %(I,J,T),m.linking3_1[I,J,T]) 

                def _linking3_2(m,N):
                    if N==m.N[I,J,T].last():
                        return m.CCDESIRED-m.CC[I,J,T][N]<=m.CCDESIRED*(1-m.X[I,J,T])
                    else:
                        return pe.Constraint.Skip 
                m.linking3_2[I,J,T]=pe.Constraint(m.N[I,J,T],rule=_linking3_2,doc='') 
                setattr(m,'linking3_2_%s_%s_%s' %(I,J,T),m.linking3_2[I,J,T])
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
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

#-------- this is required to apply dsda and ldbd (however when using variable continuous processing time these disjunctions now serve a purpose!!!!)----------------------------------------
    # m.ordered_set2={}
    # m.YR2={}
    # m.oneYR2={}
    # m.YR2_Disjunct={}
    # m.Disjunction2={}
    # for I_J in m.I_J:
    #     positcui=positcui+1
    #     I=I_J[0]
    #     J=I_J[1]
    #     m.ordered_set2[I,J]=pe.RangeSet(0,m.lastN[I,J],doc='Ordered set for each task-unit pair, related to batching variable') 
    #     setattr(m,'ordered_set2_%s_%s' %(I,J),m.ordered_set2[I,J])
          
    #     def _YR2init(m,ordered_set2):
    #         if ordered_set2== x_initial[positcui]-1:
    #             return True
    #         else:
    #             return False       
    #     m.YR2[I,J]=pe.BooleanVar(m.ordered_set2[I,J],initialize=_YR2init)
    #     setattr(m,'YR2_%s_%s' %(I,J), m.YR2[I,J])

    #     def _select_one2(m):
    #         return pe.exactly(1,m.YR2[I,J])
    #     m.oneYR2[I,J]=pe.LogicalConstraint(rule=_select_one2) 
    #     setattr(m,'oneYR2_%s_%s' %(I,J),m.oneYR2[I,J])        

    #     def _build_YR2_Disjunct(m,indexN):
    #         def _DEF_Nref(m):
    #             return m.model().Nref[I,J]==indexN
    #         m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
    #     m.YR2_Disjunct[I,J]=Disjunct(m.ordered_set2[I,J],rule=_build_YR2_Disjunct)
    #     setattr(m,'YR2_Disjunct_%s_%s' %(I,J),m.YR2_Disjunct[I,J])

    #     # Create disjunction
    #     def Disjunction2(m):   
    #         return [m.YR2_Disjunct[I,J][dis_set] for dis_set in m.ordered_set2[I,J]]
    #     m.Disjunction2[I,J]=Disjunction(rule=Disjunction2,xor=True)
    #     setattr(m,'Disjunction2_%s_%s' %(I,J),m.Disjunction2[I,J])


    # # Associate disjuncts with boolean variables
    #     for index in m.ordered_set2[I,J]:
    #         m.YR2[I,J][index].associate_binary_var(m.YR2_Disjunct[I,J][index].indicator_var)


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    # def _obj(m): 
    #     return  (    
    #       sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J) for I in m.I) for T in m.T)                                                                          #TPC: Fixed costs for all unit-tasks
    #     + sum(sum(sum( m.variable_cost[I,J]*m.B[I,J,T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)                                                #TPC: Variable cost for unit-tasks that do not consider dynamics
    #     + sum(sum(sum(m.X[I,J,T]*(m.hot_cost*m.Integral_hot[I,J][m.N[I,J].last()]   +  m.cold_cost*m.Integral_cold[I,J][m.N[I,J].last()]  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors) #TPC: Variable cost for unit-tasks that do consider dynamics
    #     + sum( m.raw_cost[K]*(m.S0[K]-m.S[K,m.lastT]) for K in m.K_inputs)                                                                                            #TMC: Total material cost
    #     - sum( m.revenue[K]*m.S[K,m.lastT]  for K in m.K_products)                                                                                                    #SALES: Revenue form selling products
    #     )/100 
    # m.obj=pe.Objective(rule=_obj,sense=pe.minimize)

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in  m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J,T][m.N[I, J,T].last()] + m.cold_cost*m.Integral_cold[I, J,T][m.N[I, J,T].last()]) for T in m.T) for I in m.I_dynamics)for J in m.J_dynamics)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3) 
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
            return m.TCP1+m.TCP2+m.TCP3+m.TMC-m.SALES  
            # return -sum(sum(sum(sum(m.CC[I,J,T][N]*m.X[I, J, T] for N in m.N[I,J,T] if N==m.N[I,J,T].last()) for J in m.J_dynamics) for I in m.I_dynamics) for T in m.T) 
        m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
    elif obj_type=='cost_min': 
        def _obj(m):
            return m.TCP1+m.TCP2+m.TCP3+m.TMC 
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

def problem_logic_scheduling_complete(m):
    logic_expr = []
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for index in m.ordered_set[I,J]:
                    logic_expr.append([m.YR[I,J][index],m.YR_disjunct[I,J][index].indicator_var])              

    for I_J in m.I_J:
        I=I_J[0]
        J=I_J[1]
        for index in m.ordered_set2[I,J]:
            logic_expr.append([m.YR2[I,J][index],m.YR2_Disjunct[I,J][index].indicator_var])  
    return logic_expr
if __name__ == "__main__":
    obj_Selected='profit_max'



    initialization=[1, 1, 1, 1, 1, 1, 1, 1]
  
    mip_solver='CPLEX'
    minlp_solver='DICOPT'
    nlp_solver='conopt4'
    transform='bigm'
    #tried 5 and no improvement. With 15 DICOT is unable, and now DSDA can solve the problem.
    last_disc=15
    last_time_h=5

    LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):0.1}
    UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):3}
    kwargs={'obj_type':obj_Selected,'last_disc_point':last_disc,'last_time_hours':last_time_h,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME,'sequential':False}
    m_fun=case_2_scheduling_control_gdp_var_proc_time_simplified_for_sequential_with_distillation
    m=m_fun(**kwargs)
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    pe.TransformationFactory('gdp.bigm').apply_to(m)
    solver = pe.SolverFactory('gams', solver=minlp_solver)
    res = solver.solve(m, tee=True)


  