from __future__ import division
from pickle import TRUE

import sys
sys.path.insert(0, '/home/dadapy/GeneralBenders/')
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min_nonlinear_ref_2
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
import pyomo.dae as dae
def problem_logic_scheduling(m):
    logic_expr = []
    for N in m.Nref:
        for I_J in m.I_J:
                if N<=m.lastN[I_J]:
                    logic_expr.append([m.Z[N,I_J],m.Z_binary[N,I_J]])
    return logic_expr

def dsda_dbd_model():
    # Data
    Infty=1e+8 
    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='reaction_1')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=14,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    

    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)

    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    # -----------parameters--------------------------------------------------
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')

    _I_i_k_minus={}
    _I_i_k_minus['Mix','S1']=1
    _I_i_k_minus['Mix','M1']=1

    _I_i_k_minus['R1','M2']=1
    _I_i_k_minus['R1','M3']=1

    _I_i_k_minus['R2','I1']=1
    _I_i_k_minus['R2','I2']=1

    _I_i_k_minus['R3','I3']=1
    _I_i_k_minus['R3','M3']=1

    _I_i_k_minus['Sep','I4']=1

    _I_i_k_minus['Pack1','I5']=1

    _I_i_k_minus['Pack2','I6']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['Mix','I1']=1

    _I_i_k_plus['R1','I2']=1

    _I_i_k_plus['R2','I3']=1
    _I_i_k_plus['R2','I5']=1

    _I_i_k_plus['R3','I4']=1

    _I_i_k_plus['Sep','W1']=1
    _I_i_k_plus['Sep','I6']=1

    _I_i_k_plus['Pack1','P1']=1

    _I_i_k_plus['Pack2','P2']=1
    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")

    _rho_minus={}
    _rho_minus['Mix','S1']=3/5
    _rho_minus['Mix','M1']=2/5

    _rho_minus['R1','M2']=1/2
    _rho_minus['R1','M3']=1/2

    _rho_minus['R2','I1']=1/2
    _rho_minus['R2','I2']=1/2

    _rho_minus['R3','I3']=1/2
    _rho_minus['R3','M3']=1/2

    _rho_minus['Sep','I4']=1

    _rho_minus['Pack1','I5']=1

    _rho_minus['Pack2','I6']=1
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")

    _rho_plus={}
    _rho_plus['Mix','I1']=1

    _rho_plus['R1','I2']=1

    _rho_plus['R2','I3']=3/5
    _rho_plus['R2','I5']=2/5

    _rho_plus['R3','I4']=1

    _rho_plus['Sep','W1']=2/5
    _rho_plus['Sep','I6']=3/5

    _rho_plus['Pack1','P1']=1

    _rho_plus['Pack2','P2']=1
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")

    _I_i_j_prod={}
    _I_i_j_prod['Mix','Mix']=1

    _I_i_j_prod['R1','R_large']=1
    _I_i_j_prod['R1','R_small']=1

    _I_i_j_prod['R2','R_large']=1
    _I_i_j_prod['R2','R_small']=1

    _I_i_j_prod['R3','R_large']=1
    _I_i_j_prod['R3','R_small']=1

    _I_i_j_prod['Sep','Sep']=1

    _I_i_j_prod['Pack1','Pack']=1
    _I_i_j_prod['Pack2','Pack']=1
    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['R1','R_large']=3 #Approximate value from You results
    _tau_p['R1','R_small']=3 #Approximate value from You results

    _tau_p['R2','R_large']=3 #Approximate value from You results
    _tau_p['R2','R_small']=3 #Approximate value from You results

    _tau_p['R3','R_large']=3 #Approximate value from You results
    _tau_p['R3','R_small']=3 #Approximate value from You results

    _tau_p['Sep','Sep']=3 

    _tau_p['Pack1','Pack']=1.5 
    _tau_p['Pack2','Pack']=1.5 

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    _beta_min={}
    _beta_min['Mix','Mix']=0.2

    _beta_min['R1','R_large']=0.15
    _beta_min['R1','R_small']=0.1

    _beta_min['R2','R_large']=0.15
    _beta_min['R2','R_small']=0.1

    _beta_min['R3','R_large']=0.15
    _beta_min['R3','R_small']=0.1

    _beta_min['Sep','Sep']=0.2

    _beta_min['Pack1','Pack']=0.1
    _beta_min['Pack2','Pack']=0.1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['Mix','Mix']=2

    _beta_max['R1','R_large']=1.5
    _beta_max['R1','R_small']=1

    _beta_max['R2','R_large']=1.5
    _beta_max['R2','R_small']=1

    _beta_max['R3','R_large']=1.5
    _beta_max['R3','R_small']=1

    _beta_max['Sep','Sep']=2

    _beta_max['Pack1','Pack']=1
    _beta_max['Pack2','Pack']=1
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':Infty,'M1':Infty,'M2':Infty,'M3':Infty,'W1':Infty,'P1':Infty,'P2':Infty,'I1':2,'I2':2,'I3':2,'I4':2,'I5':5,'I6':5},default=0,doc="maximum amount of material k that can be stored [m^3]")


    #Parameters of reactor units
    # m.v_R_max = pe.Param(m.J_reactors,initialize={'R_large':1.5,'R_small':1},doc='Maximum capacity of the reactor [m^3]') #TODO: Probably not used
    m.v_J=pe.Param(m.J_reactors,initialize={'R_large':0.5,'R_small':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_reactors,initialize={'R_large':1e+3,'R_small':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_reactors,initialize={'R_large':4.2,'R_small':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_reactors,initialize={'R_large':3e+4,'R_small':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_reactors,initialize={'R_large':300,'R_small':300},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_reactors,initialize={'R_large':10,'R_small':5},doc='Maximum flow rate of heating and cooling water [m^3/h]')

    #Parameters of reaction tasks
    m.z=pe.Param(m.I_reactions,initialize={'R1':1e+7,'R2':1.2e+3,'R3':2e+4},doc='Pre-exponential factors [m^3/kmol h]')
    m.er=pe.Param(m.I_reactions,initialize={'R1':5e+3,'R2':2e+3,'R3':3e+3},doc='Normalized activation energy [K]')
    m.delta_h=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':-2e+3,'R3':5e+3},doc='Heat of reaction [kJ/kmol]')
    m.rho_R=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':1e+3,'R3':1e+3},doc='Density of reaction mixture [kg/m^3]')
    m.c_R=pe.Param(m.I_reactions,initialize={'R1':3,'R2':3.5,'R3':4},doc='Heat capacity of reaction mixture [kJ/kg K]')
    _coef={}
    
    _coef['R1','B']=-1
    _coef['R1','C']=-1
    _coef['R1','D']=2

    _coef['R2','A']=-1
    _coef['R2','D']=-1
    _coef['R2','E']=2

    _coef['R3','C']=-1
    _coef['R3','E']=-1
    _coef['R3','F']=1
    m.coef=pe.Param(m.I_reactions,m.Q,default=0,initialize=_coef,doc='Stoichiometric coefficient')

    #Composition of states #TODO: In general the problem is formulated using mass balances, but in the paper there is an assumption, so balances are performed in terms of volumes
    _C={}
    _C['M1','A']=5

    _C['M2','B']=2

    _C['M3','C']=2

    _C['P1','E']=1.8 #This has the same composition as I3 and I5
    _C['P1','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)    

    _C['P2','F']=1 #Other component compositions are unknown due to unknown conditions in distillation. Perfect separation is assumed

    _C['I1','A']=2

    _C['I2','B']=0.05
    _C['I2','C']=0.05 
    _C['I2','D']=1.9   #TODO: Note that composition of output states from reactors is being specified, i.e., I already know the exact desired composition I want as output from each reactor

    _C['I3','E']=1.8  #TODO (solved): There are "others" component here. Check if there is any assumption, given that inerts are usually considered in balances
    _C['I3','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I4','F']=0.8
    _C['I4','C']=0.2125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','E']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','A']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','D']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','B']=0.0125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I5','E']=1.8 
    _C['I5','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I6','F']=1 #Other component compositions are unknown due to unknown conditions in distillation
    # Waste composition is unknown
    m.C=pe.Param(m.K,m.Q,initialize=_C,default=0,doc='Composition of different reactive components at each state [kmol/m^3]')
    

    #Initial composition and final composition inside reactors. This is important for dynamics, but these parameters are not going to be used in scheduling

    def _C_initial(m,I,Q):
        return sum(m.rho_minus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_minus[I,K]==1)    
    m.C_initial=pe.Param(m.I_reactions,m.Q,initialize=_C_initial,doc='Initial composition inside reactor for this reaction and component [kmol/m^3]') #TODO: Check assumptions that lead to this equation in you article. Same assumptions here
    # m.C_initial.display()


    def _C_final(m,I,Q):
        return sum(m.rho_plus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_plus[I,K]==1)
    m.C_final=pe.Param(m.I_reactions,m.Q,initialize=_C_final,doc='Final composition inside reactor for this reaction and component [kmol/m^3]')


    # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for jacket temperatures [K]')
    # Final temperature of reactions
    m.T_R_final=pe.Param(m.I_reactions,initialize={'R1':320,'R2':320,'R3':320},doc='Maximum temperature at the end of the reaction [K]')
    
    
    def _demand(m,K,T):
        if K=='P1' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        elif K=='P2' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        else:
            return 0
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'M1':Infty,'M2':Infty,'M3':Infty,'S1':Infty},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

    _fixed_cost={}
    _fixed_cost['Mix','Mix']=10

    _fixed_cost['R1','R_large']=30
    _fixed_cost['R1','R_small']=20

    _fixed_cost['R2','R_large']=30
    _fixed_cost['R2','R_small']=20

    _fixed_cost['R3','R_large']=30
    _fixed_cost['R3','R_small']=20

    _fixed_cost['Sep','Sep']=100

    _fixed_cost['Pack1','Pack']=50
    _fixed_cost['Pack2','Pack']=50

    m.fixed_cost=pe.Param(m.I,m.J,default=0,initialize=_fixed_cost,doc="Fixed cost to run task i in unit j [m.u./batch]")

    _variable_cost_param={}
    _variable_cost_param['Mix','Mix']=30

    _variable_cost_param['Sep','Sep']=100

    _variable_cost_param['Pack1','Pack']=50
    _variable_cost_param['Pack2','Pack']=50

    m.variable_cost_param=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") #TODO: define a vaiable called m.variable_cost, which will either take the value of this parameter or the variable cost related to reactor operation

    def _raw_cost(m,K):
        if K=='S1':
            return 0 
        elif K=='M1': #A
            return 100*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M2': #b
            return 150*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M3':#c
            return 200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    def _revenue(m,K):
        if K=='P1':
            return 700*sum(m.C[K,Q] for Q in m.Q)
        elif K=='P2':
            return 1200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    #-----------Reactors dynamic models--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
    
    def _Vreactor_bounds(m,I,J):
        return (m.beta_min[I,J],m.beta_max[I,J])
    m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables
    #Sets
    m.N={} #Continuous time set
    m.Q_balance={} #Species of interest in mole and energy balances

    #Variables
    m.Cvar={} #Composition profiles
    m.TRvar={} #Reactor temperature profiles
    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    #Derivativa variables
    m.dCdt={} # Composition derivatives
    m.dTRdt={} #Reactor temperature derivatives
    m.dTJdt={} #Jacket temperature derivatives

    #Differential equations
    m.c_dCdt={}
    m.c_dTRdt={}
    m.c_dTJdt={}
    
    #Final constraint
    m.finalCon={}
    m.finalTemp={}      

    for I in m.I_reactions:
        m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
        setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
        for J in m.J_reactors:
            m.N[I,J]=dae.ContinuousSet(bounds=(0,m.tau_p[I,J]),doc='Continuous time set for reaction I in reactor J [h]') #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_[%s,%s]' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


            def _Cvar_bounds(m,N,Q):
                return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
            m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
            setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

            def _TRvar_bounds(m,N):
                return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
                return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
            m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
            setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

            m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

            m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

            m.dCdt[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivcative of composition')
            setattr(m,'dCdt_(%s,%s)' %(I,J),m.dCdt[I,J])

            m.dTRdt[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
            setattr(m,'dTRdt_(%s,%s)' %(I,J),m.dTRdt[I,J])

            m.dTJdt[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
            setattr(m,'dTJdt_(%s,%s)' %(I,J),m.dTJdt[I,J])

            def _dCdt(m,N,Q):
                if N == m.N[I,J].first(): 
                    return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
                else:                                         #This is what the author calls Rb
                    return m.dCdt[I,J][N,Q] == m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]) 
            m.c_dCdt[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdt)
            setattr(m,'c_dCdt_(%s,%s)' %(I,J),m.c_dCdt[I,J])


            def _dTRdt(m,N):
                if N == m.N[I,J].first():
                    return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
                else:
                    return m.dTRdt[I,J][N] == (((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I]))  
            m.c_dTRdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdt)
            setattr(m,'c_dTRdt_(%s,%s)' %(I,J),m.c_dTRdt[I,J])
            # m.c_dTRdt[I,J].pprint()

            def _dTJdt(m,N):
                if N == m.N[I,J].first():
                    return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
                else:
                    return m.dTJdt[I,J][N] == (((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J]))  
            m.c_dTJdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdt)
            setattr(m,'c_dTJdt_(%s,%s)' %(I,J),m.c_dTJdt[I,J])
            
            
            #Constraints when finishing reaction tasks
            
            # Final concentration constraint
            def _finalCon(m,N,Q):
                if N==m.N[I,J].last():
                    return m.Cvar[I,J][N,Q] == m.C_final[I,Q]
                else:
                    return pe.Constraint.Skip
            m.finalCon[I,J]=pe.Constraint(m.N[I,J],m.Q_balance[I],rule=_finalCon)
            setattr(m,'finalCon_(%s,%s)' %(I,J),m.finalCon[I,J])
            
            #Final temperature constraints
            
            def _finalTemp(m,N):
                if N==m.N[I,J].last():
                    return m.TRvar[I,J][N]<= m.T_R_final[I]
                else:
                    return pe.Constraint.Skip
            m.finalTemp[I,J]=pe.Constraint(m.N[I,J],rule=_finalTemp)
            setattr(m,'finalTemp_(%s,%s)' %(I,J),m.finalTemp[I,J])

    # m.c_dCdt['R3','R_large'].display()
    # m.Cvar['R3','R_large'].display()  
    # m.Q_balance['R1'].pprint()
    # m.Q_balance['R2'].pprint()
    # m.Q_balance['R3'].pprint()
    # # -----------scheduling variables -----------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,initialize=0,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # # ----------Scheduling Constraints-----------------------------------------

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

    # # ----------Linking constraints-------------------------------------------
    #1) Reactor volumes and scheduling capacities
    def _linking1_1(m,I,J,T):
        return m.B[I,J,T]-m.Vreactor[I,J] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
    m.linking1=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 

    def _linking1_2(m,I,J,T):
        return -(m.B[I,J,T]-m.Vreactor[I,J]) <= m.beta_max[I,J]*(1-m.X[I,J,T])  
    m.linking2=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
      
    #There is an important assumption here (discussed before): If a given task I is executed multiple times in reactor J, then it is always executed the same way, i.e., same batch size, same time 

    #2) Variable batch costs #TODO

    #Note that processing time is constant, hence it was linked before as a parameter to define the discretization of reactor differential equations 

    #TODO: MAKE TIME DIMENTIONLESS IDEA!!!!!!!!!!!!!!!!!. MAYBE THAT WAY I CAN MAKE TIME A VARIABLE!!!!!!

    #-----------Objective function----------------------------------------------
    def _obj(m): #TODO: CONSIDER OTHER TERMS
        return sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)
    
    
    # # -------Reformulation----------------------------------------------------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.Nref=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.Nref.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    m.Z=pe.BooleanVar(m.Nref,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.Nref,m.I_J,within=pe.Binary,doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.Nref:
        for I_J in m.I_J:
                m.Z[N,I_J].associate_binary_var(m.Z_binary[N,I_J])
                if N>m.lastN[I_J]:
                    m.Z[N,I_J].fix(False)
                    m.Z_binary[N,I_J].set_value(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.Nref if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return pe.exactly(1,[m.Z[N,I,J] for N in m.Nref if N<=m.lastN[I,J]])
    m.EXACTLY_CONST=pe.LogicalConstraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()
    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=6 #Keep Fhot constant every three discretization points
    keep_constant_Fcold=6 #Keep Fcold constant every three discretization points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:
            discretizer.apply_to(m, nfe=20, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
            
            
            #------Constant control
            def _Constant_control1(m,N):
                if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fhot!=0) or (N==m.N[I,J].last()):
                    return m.Fhot[I,J][N] == m.Fhot[I,J][m.N[I,J].prev(N)]
                else:
                    return pe.Constraint.Skip
            m.Constant_control1[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control1,doc='Constant control action every keep_constant_Fhot discrete points and the last one')
            setattr(m,'Constant_control1_(%s,%s)' %(I,J),m.Constant_control1[I,J])

            def _Constant_control2(m,N):
                if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fcold!=0) or (N==m.N[I,J].last()):
                    return m.Fcold[I,J][N] == m.Fcold[I,J][m.N[I,J].prev(N)]
                else:
                    return pe.Constraint.Skip
            m.Constant_control2[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control2,doc='Constant control action every keep_constant_Fcold discrete points and the last one')
            setattr(m,'Constant_control2_(%s,%s)' %(I,J),m.Constant_control2[I,J])
            # m.Constant_control1[I,J].pprint()  
            # m.Constant_control2[I,J].pprint()             
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    return m

def build_scheduling_Boolean_cost_min_relaxed():

    # Data
    Infty=1e+8 
    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='reaction_1')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=14,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    

    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)

    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    # -----------parameters--------------------------------------------------
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')

    _I_i_k_minus={}
    _I_i_k_minus['Mix','S1']=1
    _I_i_k_minus['Mix','M1']=1

    _I_i_k_minus['R1','M2']=1
    _I_i_k_minus['R1','M3']=1

    _I_i_k_minus['R2','I1']=1
    _I_i_k_minus['R2','I2']=1

    _I_i_k_minus['R3','I3']=1
    _I_i_k_minus['R3','M3']=1

    _I_i_k_minus['Sep','I4']=1

    _I_i_k_minus['Pack1','I5']=1

    _I_i_k_minus['Pack2','I6']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['Mix','I1']=1

    _I_i_k_plus['R1','I2']=1

    _I_i_k_plus['R2','I3']=1
    _I_i_k_plus['R2','I5']=1

    _I_i_k_plus['R3','I4']=1

    _I_i_k_plus['Sep','W1']=1
    _I_i_k_plus['Sep','I6']=1

    _I_i_k_plus['Pack1','P1']=1

    _I_i_k_plus['Pack2','P2']=1
    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")

    _rho_minus={}
    _rho_minus['Mix','S1']=3/5
    _rho_minus['Mix','M1']=2/5

    _rho_minus['R1','M2']=1/2
    _rho_minus['R1','M3']=1/2

    _rho_minus['R2','I1']=1/2
    _rho_minus['R2','I2']=1/2

    _rho_minus['R3','I3']=1/2
    _rho_minus['R3','M3']=1/2

    _rho_minus['Sep','I4']=1

    _rho_minus['Pack1','I5']=1

    _rho_minus['Pack2','I6']=1
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")

    _rho_plus={}
    _rho_plus['Mix','I1']=1

    _rho_plus['R1','I2']=1

    _rho_plus['R2','I3']=3/5
    _rho_plus['R2','I5']=2/5

    _rho_plus['R3','I4']=1

    _rho_plus['Sep','W1']=2/5
    _rho_plus['Sep','I6']=3/5

    _rho_plus['Pack1','P1']=1

    _rho_plus['Pack2','P2']=1
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")

    _I_i_j_prod={}
    _I_i_j_prod['Mix','Mix']=1

    _I_i_j_prod['R1','R_large']=1
    _I_i_j_prod['R1','R_small']=1

    _I_i_j_prod['R2','R_large']=1
    _I_i_j_prod['R2','R_small']=1

    _I_i_j_prod['R3','R_large']=1
    _I_i_j_prod['R3','R_small']=1

    _I_i_j_prod['Sep','Sep']=1

    _I_i_j_prod['Pack1','Pack']=1
    _I_i_j_prod['Pack2','Pack']=1
    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['R1','R_large']=3 #Approximate value from You results
    _tau_p['R1','R_small']=3 #Approximate value from You results

    _tau_p['R2','R_large']=3 #Approximate value from You results
    _tau_p['R2','R_small']=3 #Approximate value from You results

    _tau_p['R3','R_large']=3 #Approximate value from You results
    _tau_p['R3','R_small']=3 #Approximate value from You results

    _tau_p['Sep','Sep']=3 

    _tau_p['Pack1','Pack']=1.5 
    _tau_p['Pack2','Pack']=1.5 

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    _beta_min={}
    _beta_min['Mix','Mix']=0.2

    _beta_min['R1','R_large']=0.15
    _beta_min['R1','R_small']=0.1

    _beta_min['R2','R_large']=0.15
    _beta_min['R2','R_small']=0.1

    _beta_min['R3','R_large']=0.15
    _beta_min['R3','R_small']=0.1

    _beta_min['Sep','Sep']=0.2

    _beta_min['Pack1','Pack']=0.1
    _beta_min['Pack2','Pack']=0.1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['Mix','Mix']=2

    _beta_max['R1','R_large']=1.5
    _beta_max['R1','R_small']=1

    _beta_max['R2','R_large']=1.5
    _beta_max['R2','R_small']=1

    _beta_max['R3','R_large']=1.5
    _beta_max['R3','R_small']=1

    _beta_max['Sep','Sep']=2

    _beta_max['Pack1','Pack']=1
    _beta_max['Pack2','Pack']=1
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':Infty,'M1':Infty,'M2':Infty,'M3':Infty,'W1':Infty,'P1':Infty,'P2':Infty,'I1':2,'I2':2,'I3':2,'I4':2,'I5':5,'I6':5},default=0,doc="maximum amount of material k that can be stored [m^3]")


    #Parameters of reactor units
    # m.v_R_max = pe.Param(m.J_reactors,initialize={'R_large':1.5,'R_small':1},doc='Maximum capacity of the reactor [m^3]') #TODO: Probably not used
    m.v_J=pe.Param(m.J_reactors,initialize={'R_large':0.5,'R_small':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_reactors,initialize={'R_large':1e+3,'R_small':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_reactors,initialize={'R_large':4.2,'R_small':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_reactors,initialize={'R_large':3e+4,'R_small':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_reactors,initialize={'R_large':300,'R_small':300},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_reactors,initialize={'R_large':10,'R_small':5},doc='Maximum flow rate of heating and cooling water [m^3/h]')

    #Parameters of reaction tasks
    m.z=pe.Param(m.I_reactions,initialize={'R1':1e+7,'R2':1.2e+3,'R3':2e+4},doc='Pre-exponential factors [m^3/kmol h]')
    m.er=pe.Param(m.I_reactions,initialize={'R1':5e+3,'R2':2e+3,'R3':3e+3},doc='Normalized activation energy [K]')
    m.delta_h=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':-2e+3,'R3':5e+3},doc='Heat of reaction [kJ/kmol]')
    m.rho_R=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':1e+3,'R3':1e+3},doc='Density of reaction mixture [kg/m^3]')
    m.c_R=pe.Param(m.I_reactions,initialize={'R1':3,'R2':3.5,'R3':4},doc='Heat capacity of reaction mixture [kJ/kg K]')
    _coef={}
    
    _coef['R1','B']=-1
    _coef['R1','C']=-1
    _coef['R1','D']=2

    _coef['R2','A']=-1
    _coef['R2','D']=-1
    _coef['R2','E']=2

    _coef['R3','C']=-1
    _coef['R3','E']=-1
    _coef['R3','F']=1
    m.coef=pe.Param(m.I_reactions,m.Q,default=0,initialize=_coef,doc='Stoichiometric coefficient')

    #Composition of states #TODO: In general the problem is formulated using mass balances, but in the paper there is an assumption, so balances are performed in terms of volumes
    _C={}
    _C['M1','A']=5

    _C['M2','B']=2

    _C['M3','C']=2

    _C['P1','E']=1.8 #This has the same composition as I3 and I5
    _C['P1','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)    

    _C['P2','F']=1 #Other component compositions are unknown due to unknown conditions in distillation. Perfect separation is assumed

    _C['I1','A']=2

    _C['I2','B']=0.05
    _C['I2','C']=0.05 
    _C['I2','D']=1.9   #TODO: Note that composition of output states from reactors is being specified, i.e., I already know the exact desired composition I want as output from each reactor

    _C['I3','E']=1.8  #TODO (solved): There are "others" component here. Check if there is any assumption, given that inerts are usually considered in balances
    _C['I3','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I4','F']=0.8
    _C['I4','C']=0.2125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','E']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','A']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','D']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','B']=0.0125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I5','E']=1.8 
    _C['I5','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I6','F']=1 #Other component compositions are unknown due to unknown conditions in distillation
    # Waste composition is unknown
    m.C=pe.Param(m.K,m.Q,initialize=_C,default=0,doc='Composition of different reactive components at each state [kmol/m^3]')
    

    #Initial composition and final composition inside reactors. This is important for dynamics, but these parameters are not going to be used in scheduling

    def _C_initial(m,I,Q):
        return sum(m.rho_minus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_minus[I,K]==1)    
    m.C_initial=pe.Param(m.I_reactions,m.Q,initialize=_C_initial,doc='Initial composition inside reactor for this reaction and component [kmol/m^3]') #TODO: Check assumptions that lead to this equation in you article. Same assumptions here
    # m.C_initial.display()


    def _C_final(m,I,Q):
        return sum(m.rho_plus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_plus[I,K]==1)
    m.C_final=pe.Param(m.I_reactions,m.Q,initialize=_C_final,doc='Final composition inside reactor for this reaction and component [kmol/m^3]')


    # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for jacket temperatures [K]')
    # Final temperature of reactions
    m.T_R_final=pe.Param(m.I_reactions,initialize={'R1':320,'R2':320,'R3':320},doc='Maximum temperature at the end of the reaction [K]')
    
    
    def _demand(m,K,T):
        if K=='P1' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        elif K=='P2' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        else:
            return 0
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'M1':Infty,'M2':Infty,'M3':Infty,'S1':Infty},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

    _fixed_cost={}
    _fixed_cost['Mix','Mix']=10

    _fixed_cost['R1','R_large']=30
    _fixed_cost['R1','R_small']=20

    _fixed_cost['R2','R_large']=30
    _fixed_cost['R2','R_small']=20

    _fixed_cost['R3','R_large']=30
    _fixed_cost['R3','R_small']=20

    _fixed_cost['Sep','Sep']=100

    _fixed_cost['Pack1','Pack']=50
    _fixed_cost['Pack2','Pack']=50

    m.fixed_cost=pe.Param(m.I,m.J,default=0,initialize=_fixed_cost,doc="Fixed cost to run task i in unit j [m.u./batch]")

    _variable_cost_param={}
    _variable_cost_param['Mix','Mix']=30

    _variable_cost_param['Sep','Sep']=100

    _variable_cost_param['Pack1','Pack']=50
    _variable_cost_param['Pack2','Pack']=50

    m.variable_cost_param=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") #TODO: define a vaiable called m.variable_cost, which will either take the value of this parameter or the variable cost related to reactor operation

    def _raw_cost(m,K):
        if K=='S1':
            return 0 
        elif K=='M1': #A
            return 100*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M2': #b
            return 150*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M3':#c
            return 200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    def _revenue(m,K):
        if K=='P1':
            return 700*sum(m.C[K,Q] for Q in m.Q)
        elif K=='P2':
            return 1200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    #-----------Reactors dynamic models--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
    
    def _Vreactor_bounds(m,I,J):
        return (m.beta_min[I,J],m.beta_max[I,J])
    m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables
    #Sets
    m.N={} #Continuous time set
    m.Q_balance={} #Species of interest in mole and energy balances

    #Variables
    m.Cvar={} #Composition profiles
    m.TRvar={} #Reactor temperature profiles
    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    #Derivativa variables
    m.dCdt={} # Composition derivatives
    m.dTRdt={} #Reactor temperature derivatives
    m.dTJdt={} #Jacket temperature derivatives

    #Differential equations
    m.c_dCdt={}
    m.c_dTRdt={}
    m.c_dTJdt={}
    
    #Final constraint
    m.finalCon={}
    m.finalTemp={}      

    for I in m.I_reactions:
        m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
        setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
        for J in m.J_reactors:
            m.N[I,J]=dae.ContinuousSet(bounds=(0,m.tau_p[I,J]),doc='Continuous time set for reaction I in reactor J [h]') #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_[%s,%s]' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


            def _Cvar_bounds(m,N,Q):
                return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
            m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
            setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

            def _TRvar_bounds(m,N):
                return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
                return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
            m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
            setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

            m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

            m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

            m.dCdt[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivcative of composition')
            setattr(m,'dCdt_(%s,%s)' %(I,J),m.dCdt[I,J])

            m.dTRdt[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
            setattr(m,'dTRdt_(%s,%s)' %(I,J),m.dTRdt[I,J])

            m.dTJdt[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
            setattr(m,'dTJdt_(%s,%s)' %(I,J),m.dTJdt[I,J])

            def _dCdt(m,N,Q):
                if N == m.N[I,J].first(): 
                    return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
                else:                                         #This is what the author calls Rb
                    return m.dCdt[I,J][N,Q] == m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]) 
            m.c_dCdt[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdt)
            setattr(m,'c_dCdt_(%s,%s)' %(I,J),m.c_dCdt[I,J])


            def _dTRdt(m,N):
                if N == m.N[I,J].first():
                    return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
                else:
                    return m.dTRdt[I,J][N] == (((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I]))  
            m.c_dTRdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdt)
            setattr(m,'c_dTRdt_(%s,%s)' %(I,J),m.c_dTRdt[I,J])
            # m.c_dTRdt[I,J].pprint()

            def _dTJdt(m,N):
                if N == m.N[I,J].first():
                    return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
                else:
                    return m.dTJdt[I,J][N] == (((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J]))  
            m.c_dTJdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdt)
            setattr(m,'c_dTJdt_(%s,%s)' %(I,J),m.c_dTJdt[I,J])
            
            
            #Constraints when finishing reaction tasks
            
            # Final concentration constraint
            def _finalCon(m,N,Q):
                if N==m.N[I,J].last():
                    return m.Cvar[I,J][N,Q] == m.C_final[I,Q]
                else:
                    return pe.Constraint.Skip
            m.finalCon[I,J]=pe.Constraint(m.N[I,J],m.Q_balance[I],rule=_finalCon)
            setattr(m,'finalCon_(%s,%s)' %(I,J),m.finalCon[I,J])
            
            #Final temperature constraints
            
            def _finalTemp(m,N):
                if N==m.N[I,J].last():
                    return m.TRvar[I,J][N]<= m.T_R_final[I]
                else:
                    return pe.Constraint.Skip
            m.finalTemp[I,J]=pe.Constraint(m.N[I,J],rule=_finalTemp)
            setattr(m,'finalTemp_(%s,%s)' %(I,J),m.finalTemp[I,J])

    # m.c_dCdt['R3','R_large'].display()
    # m.Cvar['R3','R_large'].display()  
    # m.Q_balance['R1'].pprint()
    # m.Q_balance['R2'].pprint()
    # m.Q_balance['R3'].pprint()
    # # -----------scheduling variables -----------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=(0,1),doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # m.S.display()
    # # ----------Scheduling Constraints-----------------------------------------

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

    # # ----------Linking constraints-------------------------------------------
    #1) Reactor volumes and scheduling capacities
    def _linking1_1(m,I,J,T):
        return m.B[I,J,T]-m.Vreactor[I,J] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
    m.linking1=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 

    def _linking1_2(m,I,J,T):
        return -(m.B[I,J,T]-m.Vreactor[I,J]) <= m.beta_max[I,J]*(1-m.X[I,J,T])  
    m.linking2=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
      
    #There is an important assumption here (discussed before): If a given task I is executed multiple times in reactor J, then it is always executed the same way, i.e., same batch size, same time 

    #2) Variable batch costs #TODO

    #Note that processing time is constant, hence it was linked before as a parameter to define the discretization of reactor differential equations 

    #TODO: MAKE TIME DIMENTIONLESS IDEA!!!!!!!!!!!!!!!!!. MAYBE THAT WAY I CAN MAKE TIME A VARIABLE!!!!!!

    #-----------Objective function----------------------------------------------
    def _obj(m): #TODO: CONSIDER OTHER TERMS
        return sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)
    
    
    # # -------Reformulation----------------------------------------------------
    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    def _Nref_bounds(m,I,J):
        return (0,m.lastN[I,J])
    m.Nref=pe.Var(m.I_J,within=pe.NonNegativeReals,bounds=_Nref_bounds,doc='reformulation variables from 0 to lastN')
    
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==m.Nref[I,J]
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')
     
    #m.EXACTLY_CONST.pprint()
    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=6 #Keep Fhot constant every three discretization points
    keep_constant_Fcold=6 #Keep Fcold constant every three discretization points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:
            discretizer.apply_to(m, nfe=20, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
            
            
            #------Constant control
            def _Constant_control1(m,N):
                if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fhot!=0) or (N==m.N[I,J].last()):
                    return m.Fhot[I,J][N] == m.Fhot[I,J][m.N[I,J].prev(N)]
                else:
                    return pe.Constraint.Skip
            m.Constant_control1[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control1,doc='Constant control action every keep_constant_Fhot discrete points and the last one')
            setattr(m,'Constant_control1_(%s,%s)' %(I,J),m.Constant_control1[I,J])

            def _Constant_control2(m,N):
                if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fcold!=0) or (N==m.N[I,J].last()):
                    return m.Fcold[I,J][N] == m.Fcold[I,J][m.N[I,J].prev(N)]
                else:
                    return pe.Constraint.Skip
            m.Constant_control2[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control2,doc='Constant control action every keep_constant_Fcold discrete points and the last one')
            setattr(m,'Constant_control2_(%s,%s)' %(I,J),m.Constant_control2[I,J])
            # m.Constant_control1[I,J].pprint()  
            # m.Constant_control2[I,J].pprint()             
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------





    return m

def build_scheduling_Boolean_cost_min_simplified():
    # Data
    Infty=1e+8 
    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='reaction_1')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=14,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    

    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)

    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    # -----------parameters--------------------------------------------------
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')

    _I_i_k_minus={}
    _I_i_k_minus['Mix','S1']=1
    _I_i_k_minus['Mix','M1']=1

    _I_i_k_minus['R1','M2']=1
    _I_i_k_minus['R1','M3']=1

    _I_i_k_minus['R2','I1']=1
    _I_i_k_minus['R2','I2']=1

    _I_i_k_minus['R3','I3']=1
    _I_i_k_minus['R3','M3']=1

    _I_i_k_minus['Sep','I4']=1

    _I_i_k_minus['Pack1','I5']=1

    _I_i_k_minus['Pack2','I6']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['Mix','I1']=1

    _I_i_k_plus['R1','I2']=1

    _I_i_k_plus['R2','I3']=1
    _I_i_k_plus['R2','I5']=1

    _I_i_k_plus['R3','I4']=1

    _I_i_k_plus['Sep','W1']=1
    _I_i_k_plus['Sep','I6']=1

    _I_i_k_plus['Pack1','P1']=1

    _I_i_k_plus['Pack2','P2']=1
    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")

    _rho_minus={}
    _rho_minus['Mix','S1']=3/5
    _rho_minus['Mix','M1']=2/5

    _rho_minus['R1','M2']=1/2
    _rho_minus['R1','M3']=1/2

    _rho_minus['R2','I1']=1/2
    _rho_minus['R2','I2']=1/2

    _rho_minus['R3','I3']=1/2
    _rho_minus['R3','M3']=1/2

    _rho_minus['Sep','I4']=1

    _rho_minus['Pack1','I5']=1

    _rho_minus['Pack2','I6']=1
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")

    _rho_plus={}
    _rho_plus['Mix','I1']=1

    _rho_plus['R1','I2']=1

    _rho_plus['R2','I3']=3/5
    _rho_plus['R2','I5']=2/5

    _rho_plus['R3','I4']=1

    _rho_plus['Sep','W1']=2/5
    _rho_plus['Sep','I6']=3/5

    _rho_plus['Pack1','P1']=1

    _rho_plus['Pack2','P2']=1
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")

    _I_i_j_prod={}
    _I_i_j_prod['Mix','Mix']=1

    _I_i_j_prod['R1','R_large']=1
    _I_i_j_prod['R1','R_small']=1

    _I_i_j_prod['R2','R_large']=1
    _I_i_j_prod['R2','R_small']=1

    _I_i_j_prod['R3','R_large']=1
    _I_i_j_prod['R3','R_small']=1

    _I_i_j_prod['Sep','Sep']=1

    _I_i_j_prod['Pack1','Pack']=1
    _I_i_j_prod['Pack2','Pack']=1
    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['R1','R_large']=3 #Approximate value from You results
    _tau_p['R1','R_small']=3 #Approximate value from You results

    _tau_p['R2','R_large']=3 #Approximate value from You results
    _tau_p['R2','R_small']=3 #Approximate value from You results

    _tau_p['R3','R_large']=3 #Approximate value from You results
    _tau_p['R3','R_small']=3 #Approximate value from You results

    _tau_p['Sep','Sep']=3 

    _tau_p['Pack1','Pack']=1.5 
    _tau_p['Pack2','Pack']=1.5 

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    _beta_min={}
    _beta_min['Mix','Mix']=0.2

    _beta_min['R1','R_large']=0.15
    _beta_min['R1','R_small']=0.1

    _beta_min['R2','R_large']=0.15
    _beta_min['R2','R_small']=0.1

    _beta_min['R3','R_large']=0.15
    _beta_min['R3','R_small']=0.1

    _beta_min['Sep','Sep']=0.2

    _beta_min['Pack1','Pack']=0.1
    _beta_min['Pack2','Pack']=0.1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['Mix','Mix']=2

    _beta_max['R1','R_large']=1.5
    _beta_max['R1','R_small']=1

    _beta_max['R2','R_large']=1.5
    _beta_max['R2','R_small']=1

    _beta_max['R3','R_large']=1.5
    _beta_max['R3','R_small']=1

    _beta_max['Sep','Sep']=2

    _beta_max['Pack1','Pack']=1
    _beta_max['Pack2','Pack']=1
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':Infty,'M1':Infty,'M2':Infty,'M3':Infty,'W1':Infty,'P1':Infty,'P2':Infty,'I1':2,'I2':2,'I3':2,'I4':2,'I5':5,'I6':5},default=0,doc="maximum amount of material k that can be stored [m^3]")


    #Parameters of reactor units
    # m.v_R_max = pe.Param(m.J_reactors,initialize={'R_large':1.5,'R_small':1},doc='Maximum capacity of the reactor [m^3]') #TODO: Probably not used
    m.v_J=pe.Param(m.J_reactors,initialize={'R_large':0.5,'R_small':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_reactors,initialize={'R_large':1e+3,'R_small':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_reactors,initialize={'R_large':4.2,'R_small':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_reactors,initialize={'R_large':3e+4,'R_small':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_reactors,initialize={'R_large':300,'R_small':300},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_reactors,initialize={'R_large':10,'R_small':5},doc='Maximum flow rate of heating and cooling water [m^3/h]')

    #Parameters of reaction tasks
    m.z=pe.Param(m.I_reactions,initialize={'R1':1e+7,'R2':1.2e+3,'R3':2e+4},doc='Pre-exponential factors [m^3/kmol h]')
    m.er=pe.Param(m.I_reactions,initialize={'R1':5e+3,'R2':2e+3,'R3':3e+3},doc='Normalized activation energy [K]')
    m.delta_h=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':-2e+3,'R3':5e+3},doc='Heat of reaction [kJ/kmol]')
    m.rho_R=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':1e+3,'R3':1e+3},doc='Density of reaction mixture [kg/m^3]')
    m.c_R=pe.Param(m.I_reactions,initialize={'R1':3,'R2':3.5,'R3':4},doc='Heat capacity of reaction mixture [kJ/kg K]')
    _coef={}
    
    _coef['R1','B']=-1
    _coef['R1','C']=-1
    _coef['R1','D']=2

    _coef['R2','A']=-1
    _coef['R2','D']=-1
    _coef['R2','E']=2

    _coef['R3','C']=-1
    _coef['R3','E']=-1
    _coef['R3','F']=1
    m.coef=pe.Param(m.I_reactions,m.Q,default=0,initialize=_coef,doc='Stoichiometric coefficient')

    #Composition of states #TODO: In general the problem is formulated using mass balances, but in the paper there is an assumption, so balances are performed in terms of volumes
    _C={}
    _C['M1','A']=5

    _C['M2','B']=2

    _C['M3','C']=2

    _C['P1','E']=1.8 #This has the same composition as I3 and I5
    _C['P1','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)    

    _C['P2','F']=1 #Other component compositions are unknown due to unknown conditions in distillation. Perfect separation is assumed

    _C['I1','A']=2

    _C['I2','B']=0.05
    _C['I2','C']=0.05 
    _C['I2','D']=1.9   #TODO: Note that composition of output states from reactors is being specified, i.e., I already know the exact desired composition I want as output from each reactor

    _C['I3','E']=1.8  #TODO (solved): There are "others" component here. Check if there is any assumption, given that inerts are usually considered in balances
    _C['I3','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I4','F']=0.8
    _C['I4','C']=0.2125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','E']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','A']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','D']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','B']=0.0125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I5','E']=1.8 
    _C['I5','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I6','F']=1 #Other component compositions are unknown due to unknown conditions in distillation
    # Waste composition is unknown
    m.C=pe.Param(m.K,m.Q,initialize=_C,default=0,doc='Composition of different reactive components at each state [kmol/m^3]')
    

    #Initial composition and final composition inside reactors. This is important for dynamics, but these parameters are not going to be used in scheduling

    def _C_initial(m,I,Q):
        return sum(m.rho_minus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_minus[I,K]==1)    
    m.C_initial=pe.Param(m.I_reactions,m.Q,initialize=_C_initial,doc='Initial composition inside reactor for this reaction and component [kmol/m^3]') #TODO: Check assumptions that lead to this equation in you article. Same assumptions here
    # m.C_initial.display()


    def _C_final(m,I,Q):
        return sum(m.rho_plus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_plus[I,K]==1)
    m.C_final=pe.Param(m.I_reactions,m.Q,initialize=_C_final,doc='Final composition inside reactor for this reaction and component [kmol/m^3]')


    # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for jacket temperatures [K]')
    # Final temperature of reactions
    m.T_R_final=pe.Param(m.I_reactions,initialize={'R1':320,'R2':320,'R3':320},doc='Maximum temperature at the end of the reaction [K]')
    
    
    def _demand(m,K,T):
        if K=='P1' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        elif K=='P2' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        else:
            return 0
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'M1':Infty,'M2':Infty,'M3':Infty,'S1':Infty},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

    _fixed_cost={}
    _fixed_cost['Mix','Mix']=10

    _fixed_cost['R1','R_large']=30
    _fixed_cost['R1','R_small']=20

    _fixed_cost['R2','R_large']=30
    _fixed_cost['R2','R_small']=20

    _fixed_cost['R3','R_large']=30
    _fixed_cost['R3','R_small']=20

    _fixed_cost['Sep','Sep']=100

    _fixed_cost['Pack1','Pack']=50
    _fixed_cost['Pack2','Pack']=50

    m.fixed_cost=pe.Param(m.I,m.J,default=0,initialize=_fixed_cost,doc="Fixed cost to run task i in unit j [m.u./batch]")

    _variable_cost_param={}
    _variable_cost_param['Mix','Mix']=30

    _variable_cost_param['Sep','Sep']=100

    _variable_cost_param['Pack1','Pack']=50
    _variable_cost_param['Pack2','Pack']=50

    m.variable_cost_param=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") #TODO: define a vaiable called m.variable_cost, which will either take the value of this parameter or the variable cost related to reactor operation

    def _raw_cost(m,K):
        if K=='S1':
            return 0 
        elif K=='M1': #A
            return 100*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M2': #b
            return 150*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M3':#c
            return 200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    def _revenue(m,K):
        if K=='P1':
            return 700*sum(m.C[K,Q] for Q in m.Q)
        elif K=='P2':
            return 1200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    # # -----------scheduling variables -----------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,doc='1 if unit j processes task i starting at time t')   
    # # ----------Scheduling Constraints-----------------------------------------


    #There is an important assumption here (discussed before): If a given task I is executed multiple times in reactor J, then it is always executed the same way, i.e., same batch size, same time 

    #2) Variable batch costs #TODO

    #Note that processing time is constant, hence it was linked before as a parameter to define the discretization of reactor differential equations 

    #TODO: MAKE TIME DIMENTIONLESS IDEA!!!!!!!!!!!!!!!!!. MAYBE THAT WAY I CAN MAKE TIME A VARIABLE!!!!!!

    #-----------Objective function----------------------------------------------
    def _obj(m): #TODO: CONSIDER OTHER TERMS
        return sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)
    
    
    # # -------Reformulation----------------------------------------------------
    #TODO: THIS IS THE CORRECT DEFINITION WHEN USING THE CONTINUOUS TIME FORMULATION !!!!! : ### m.min_taup=pe.Param(initialize=math.floor(m.eta/min([pe.value(m.tau_p[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    m.maxN=pe.Param(initialize=math.floor((m.T.__len__()-1)/min([pe.value(m.tau[I,J]) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1])),doc='minimum Physical processing time [units of time]')
    #m.lastN.display()
    m.Nref=pe.RangeSet(0,m.maxN,1, doc='ordered set')
    #m.Nref.display()

    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    
    m.Z=pe.BooleanVar(m.Nref,m.I_J,doc='Boolean variables that can be reformulated with external variables')
    m.Z_binary=pe.Var(m.Nref,m.I_J,within=pe.Binary,doc='Binary variable associated to Z')   #TODO: there are no disjuncts for the moment, so we can do this. In the future we have to create disjuncts
    
    # Associate boolean to binary variables and fix Boolean and bynary variable that are redundant in the formulation
    for N in m.Nref:
        for I_J in m.I_J:
                m.Z[N,I_J].associate_binary_var(m.Z_binary[N,I_J])
                if N>m.lastN[I_J]:
                    m.Z[N,I_J].fix(False)
                    m.Z_binary[N,I_J].set_value(0)
        
    # m.Z.display()
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==sum(N*m.Z_binary[N,I,J] for N in m.Nref if N<=m.lastN[I,J])
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Boolean and binary variables')
    #m.X_Z_relation.pprint()


    #Exactly k constraints
    def _EXACTLY_CONST(m,I,J):
        return pe.exactly(1,[m.Z[N,I,J] for N in m.Nref if N<=m.lastN[I,J]])
    m.EXACTLY_CONST=pe.LogicalConstraint(m.I_J,rule=_EXACTLY_CONST)   
    #m.EXACTLY_CONST.pprint()
    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=6 #Keep Fhot constant every three discretization points
    keep_constant_Fcold=6 #Keep Fcold constant every three discretization points 
           
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------

    return m 

def build_scheduling_Boolean_cost_min_feasibility(objective,epsilon):
    # Data
    Infty=1e+8 
    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='reaction_1')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=14,doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    

    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)

    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')
    # -----------parameters--------------------------------------------------
    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')

    _I_i_k_minus={}
    _I_i_k_minus['Mix','S1']=1
    _I_i_k_minus['Mix','M1']=1

    _I_i_k_minus['R1','M2']=1
    _I_i_k_minus['R1','M3']=1

    _I_i_k_minus['R2','I1']=1
    _I_i_k_minus['R2','I2']=1

    _I_i_k_minus['R3','I3']=1
    _I_i_k_minus['R3','M3']=1

    _I_i_k_minus['Sep','I4']=1

    _I_i_k_minus['Pack1','I5']=1

    _I_i_k_minus['Pack2','I6']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['Mix','I1']=1

    _I_i_k_plus['R1','I2']=1

    _I_i_k_plus['R2','I3']=1
    _I_i_k_plus['R2','I5']=1

    _I_i_k_plus['R3','I4']=1

    _I_i_k_plus['Sep','W1']=1
    _I_i_k_plus['Sep','I6']=1

    _I_i_k_plus['Pack1','P1']=1

    _I_i_k_plus['Pack2','P2']=1
    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")

    _rho_minus={}
    _rho_minus['Mix','S1']=3/5
    _rho_minus['Mix','M1']=2/5

    _rho_minus['R1','M2']=1/2
    _rho_minus['R1','M3']=1/2

    _rho_minus['R2','I1']=1/2
    _rho_minus['R2','I2']=1/2

    _rho_minus['R3','I3']=1/2
    _rho_minus['R3','M3']=1/2

    _rho_minus['Sep','I4']=1

    _rho_minus['Pack1','I5']=1

    _rho_minus['Pack2','I6']=1
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")

    _rho_plus={}
    _rho_plus['Mix','I1']=1

    _rho_plus['R1','I2']=1

    _rho_plus['R2','I3']=3/5
    _rho_plus['R2','I5']=2/5

    _rho_plus['R3','I4']=1

    _rho_plus['Sep','W1']=2/5
    _rho_plus['Sep','I6']=3/5

    _rho_plus['Pack1','P1']=1

    _rho_plus['Pack2','P2']=1
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")

    _I_i_j_prod={}
    _I_i_j_prod['Mix','Mix']=1

    _I_i_j_prod['R1','R_large']=1
    _I_i_j_prod['R1','R_small']=1

    _I_i_j_prod['R2','R_large']=1
    _I_i_j_prod['R2','R_small']=1

    _I_i_j_prod['R3','R_large']=1
    _I_i_j_prod['R3','R_small']=1

    _I_i_j_prod['Sep','Sep']=1

    _I_i_j_prod['Pack1','Pack']=1
    _I_i_j_prod['Pack2','Pack']=1
    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['R1','R_large']=3 #Approximate value from You results
    _tau_p['R1','R_small']=3 #Approximate value from You results

    _tau_p['R2','R_large']=3 #Approximate value from You results
    _tau_p['R2','R_small']=3 #Approximate value from You results

    _tau_p['R3','R_large']=3 #Approximate value from You results
    _tau_p['R3','R_small']=3 #Approximate value from You results

    _tau_p['Sep','Sep']=3 

    _tau_p['Pack1','Pack']=1.5 
    _tau_p['Pack2','Pack']=1.5 

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    _beta_min={}
    _beta_min['Mix','Mix']=0.2

    _beta_min['R1','R_large']=0.15
    _beta_min['R1','R_small']=0.1

    _beta_min['R2','R_large']=0.15
    _beta_min['R2','R_small']=0.1

    _beta_min['R3','R_large']=0.15
    _beta_min['R3','R_small']=0.1

    _beta_min['Sep','Sep']=0.2

    _beta_min['Pack1','Pack']=0.1
    _beta_min['Pack2','Pack']=0.1
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

    _beta_max={}
    _beta_max['Mix','Mix']=2

    _beta_max['R1','R_large']=1.5
    _beta_max['R1','R_small']=1

    _beta_max['R2','R_large']=1.5
    _beta_max['R2','R_small']=1

    _beta_max['R3','R_large']=1.5
    _beta_max['R3','R_small']=1

    _beta_max['Sep','Sep']=2

    _beta_max['Pack1','Pack']=1
    _beta_max['Pack2','Pack']=1
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.
    
    m.gamma=pe.Param(m.K,initialize={'S1':Infty,'M1':Infty,'M2':Infty,'M3':Infty,'W1':Infty,'P1':Infty,'P2':Infty,'I1':2,'I2':2,'I3':2,'I4':2,'I5':5,'I6':5},default=0,doc="maximum amount of material k that can be stored [m^3]")


    #Parameters of reactor units
    # m.v_R_max = pe.Param(m.J_reactors,initialize={'R_large':1.5,'R_small':1},doc='Maximum capacity of the reactor [m^3]') #TODO: Probably not used
    m.v_J=pe.Param(m.J_reactors,initialize={'R_large':0.5,'R_small':0.3},doc='Volume of the Jacket [m^3]')
    m.rho_J=pe.Param(m.J_reactors,initialize={'R_large':1e+3,'R_small':1e+3},doc='Density of the jacket [kg/m^3]')
    m.c_J=pe.Param(m.J_reactors,initialize={'R_large':4.2,'R_small':4.2},doc='Heat capacity of jacket [kJ/kg K]')
    m.ua=pe.Param(m.J_reactors,initialize={'R_large':3e+4,'R_small':2e+4},doc='Heat transfer coefficient [kJ/h K]')
    m.T_H= pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Temperature of heating water [K]')
    m.T_C=pe.Param(m.J_reactors,initialize={'R_large':300,'R_small':300},doc='Temperature of cooling water [K]')
    m.T_R_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of reactor [K]')
    m.T_J_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of jacket [K]')
    m.F_max=pe.Param(m.J_reactors,initialize={'R_large':10,'R_small':5},doc='Maximum flow rate of heating and cooling water [m^3/h]')

    #Parameters of reaction tasks
    m.z=pe.Param(m.I_reactions,initialize={'R1':1e+7,'R2':1.2e+3,'R3':2e+4},doc='Pre-exponential factors [m^3/kmol h]')
    m.er=pe.Param(m.I_reactions,initialize={'R1':5e+3,'R2':2e+3,'R3':3e+3},doc='Normalized activation energy [K]')
    m.delta_h=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':-2e+3,'R3':5e+3},doc='Heat of reaction [kJ/kmol]')
    m.rho_R=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':1e+3,'R3':1e+3},doc='Density of reaction mixture [kg/m^3]')
    m.c_R=pe.Param(m.I_reactions,initialize={'R1':3,'R2':3.5,'R3':4},doc='Heat capacity of reaction mixture [kJ/kg K]')
    _coef={}
    
    _coef['R1','B']=-1
    _coef['R1','C']=-1
    _coef['R1','D']=2

    _coef['R2','A']=-1
    _coef['R2','D']=-1
    _coef['R2','E']=2

    _coef['R3','C']=-1
    _coef['R3','E']=-1
    _coef['R3','F']=1
    m.coef=pe.Param(m.I_reactions,m.Q,default=0,initialize=_coef,doc='Stoichiometric coefficient')

    #Composition of states #TODO: In general the problem is formulated using mass balances, but in the paper there is an assumption, so balances are performed in terms of volumes
    _C={}
    _C['M1','A']=5

    _C['M2','B']=2

    _C['M3','C']=2

    _C['P1','E']=1.8 #This has the same composition as I3 and I5
    _C['P1','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['P1','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)    

    _C['P2','F']=1 #Other component compositions are unknown due to unknown conditions in distillation. Perfect separation is assumed

    _C['I1','A']=2

    _C['I2','B']=0.05
    _C['I2','C']=0.05 
    _C['I2','D']=1.9   #TODO: Note that composition of output states from reactors is being specified, i.e., I already know the exact desired composition I want as output from each reactor

    _C['I3','E']=1.8  #TODO (solved): There are "others" component here. Check if there is any assumption, given that inerts are usually considered in balances
    _C['I3','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I3','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I4','F']=0.8
    _C['I4','C']=0.2125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','E']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','A']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','D']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I4','B']=0.0125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I5','E']=1.8 
    _C['I5','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
    _C['I5','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

    _C['I6','F']=1 #Other component compositions are unknown due to unknown conditions in distillation
    # Waste composition is unknown
    m.C=pe.Param(m.K,m.Q,initialize=_C,default=0,doc='Composition of different reactive components at each state [kmol/m^3]')
    

    #Initial composition and final composition inside reactors. This is important for dynamics, but these parameters are not going to be used in scheduling

    def _C_initial(m,I,Q):
        return sum(m.rho_minus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_minus[I,K]==1)    
    m.C_initial=pe.Param(m.I_reactions,m.Q,initialize=_C_initial,doc='Initial composition inside reactor for this reaction and component [kmol/m^3]') #TODO: Check assumptions that lead to this equation in you article. Same assumptions here
    # m.C_initial.display()


    def _C_final(m,I,Q):
        return sum(m.rho_plus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_plus[I,K]==1)
    m.C_final=pe.Param(m.I_reactions,m.Q,initialize=_C_final,doc='Final composition inside reactor for this reaction and component [kmol/m^3]')


    # Initial temperature of reactors and heating medium for each task
    m.T_R_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for reaction temperatures inside reactor [K]')
    m.T_J_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for jacket temperatures [K]')
    # Final temperature of reactions
    m.T_R_final=pe.Param(m.I_reactions,initialize={'R1':320,'R2':320,'R3':320},doc='Maximum temperature at the end of the reaction [K]')
    
    
    def _demand(m,K,T):
        if K=='P1' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        elif K=='P2' and T==m.lastT:
            return (1)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        else:
            return 0
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'M1':Infty,'M2':Infty,'M3':Infty,'S1':Infty},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

    _fixed_cost={}
    _fixed_cost['Mix','Mix']=10

    _fixed_cost['R1','R_large']=30
    _fixed_cost['R1','R_small']=20

    _fixed_cost['R2','R_large']=30
    _fixed_cost['R2','R_small']=20

    _fixed_cost['R3','R_large']=30
    _fixed_cost['R3','R_small']=20

    _fixed_cost['Sep','Sep']=100

    _fixed_cost['Pack1','Pack']=50
    _fixed_cost['Pack2','Pack']=50

    m.fixed_cost=pe.Param(m.I,m.J,default=0,initialize=_fixed_cost,doc="Fixed cost to run task i in unit j [m.u./batch]")

    _variable_cost_param={}
    _variable_cost_param['Mix','Mix']=30

    _variable_cost_param['Sep','Sep']=100

    _variable_cost_param['Pack1','Pack']=50
    _variable_cost_param['Pack2','Pack']=50

    m.variable_cost_param=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") #TODO: define a vaiable called m.variable_cost, which will either take the value of this parameter or the variable cost related to reactor operation

    def _raw_cost(m,K):
        if K=='S1':
            return 0 
        elif K=='M1': #A
            return 100*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M2': #b
            return 150*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M3':#c
            return 200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    def _revenue(m,K):
        if K=='P1':
            return 700*sum(m.C[K,Q] for Q in m.Q)
        elif K=='P2':
            return 1200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')

    #-----------Reactors dynamic models--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
    
    def _Vreactor_bounds(m,I,J):
        return (m.beta_min[I,J],m.beta_max[I,J])
    m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables
    #Sets
    m.N={} #Continuous time set
    m.Q_balance={} #Species of interest in mole and energy balances

    #Variables
    m.Cvar={} #Composition profiles
    m.TRvar={} #Reactor temperature profiles
    m.TJvar={} #Jacket temperature profile
    m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
    m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

    #Derivativa variables
    m.dCdt={} # Composition derivatives
    m.dTRdt={} #Reactor temperature derivatives
    m.dTJdt={} #Jacket temperature derivatives

    #Differential equations
    m.c_dCdt={}
    m.c_dTRdt={}
    m.c_dTJdt={}
    
    #Final constraint
    m.finalCon={}
    m.finalTemp={}      

    for I in m.I_reactions:
        m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
        setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
        for J in m.J_reactors:
            m.N[I,J]=dae.ContinuousSet(bounds=(0,m.tau_p[I,J]),doc='Continuous time set for reaction I in reactor J [h]') #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_[%s,%s]' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


            def _Cvar_bounds(m,N,Q):
                return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
            m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
            setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

            def _TRvar_bounds(m,N):
                return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
                return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
            m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
            setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

            m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

            m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

            m.dCdt[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivcative of composition')
            setattr(m,'dCdt_(%s,%s)' %(I,J),m.dCdt[I,J])

            m.dTRdt[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
            setattr(m,'dTRdt_(%s,%s)' %(I,J),m.dTRdt[I,J])

            m.dTJdt[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
            setattr(m,'dTJdt_(%s,%s)' %(I,J),m.dTJdt[I,J])

            def _dCdt(m,N,Q):
                if N == m.N[I,J].first(): 
                    return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
                else:                                         #This is what the author calls Rb
                    return m.dCdt[I,J][N,Q] == m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]) 
            m.c_dCdt[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdt)
            setattr(m,'c_dCdt_(%s,%s)' %(I,J),m.c_dCdt[I,J])


            def _dTRdt(m,N):
                if N == m.N[I,J].first():
                    return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
                else:
                    return m.dTRdt[I,J][N] == (((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I]))  
            m.c_dTRdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdt)
            setattr(m,'c_dTRdt_(%s,%s)' %(I,J),m.c_dTRdt[I,J])
            # m.c_dTRdt[I,J].pprint()

            def _dTJdt(m,N):
                if N == m.N[I,J].first():
                    return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
                else:
                    return m.dTJdt[I,J][N] == (((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J]))  
            m.c_dTJdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdt)
            setattr(m,'c_dTJdt_(%s,%s)' %(I,J),m.c_dTJdt[I,J])
            
            
            #Constraints when finishing reaction tasks
            
            # Final concentration constraint
            def _finalCon(m,N,Q):
                if N==m.N[I,J].last():
                    return m.Cvar[I,J][N,Q] == m.C_final[I,Q]
                else:
                    return pe.Constraint.Skip
            m.finalCon[I,J]=pe.Constraint(m.N[I,J],m.Q_balance[I],rule=_finalCon)
            setattr(m,'finalCon_(%s,%s)' %(I,J),m.finalCon[I,J])
            
            #Final temperature constraints
            
            def _finalTemp(m,N):
                if N==m.N[I,J].last():
                    return m.TRvar[I,J][N]<= m.T_R_final[I]
                else:
                    return pe.Constraint.Skip
            m.finalTemp[I,J]=pe.Constraint(m.N[I,J],rule=_finalTemp)
            setattr(m,'finalTemp_(%s,%s)' %(I,J),m.finalTemp[I,J])

    # m.c_dCdt['R3','R_large'].display()
    # m.Cvar['R3','R_large'].display()  
    # m.Q_balance['R1'].pprint()
    # m.Q_balance['R2'].pprint()
    # m.Q_balance['R3'].pprint()
    # # -----------scheduling variables -----------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,initialize=0,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')
    # # ----------Scheduling Constraints-----------------------------------------

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

    # # ----------Linking constraints-------------------------------------------
    #1) Reactor volumes and scheduling capacities
    def _linking1_1(m,I,J,T):
        return m.B[I,J,T]-m.Vreactor[I,J] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
    m.linking1=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 

    def _linking1_2(m,I,J,T):
        return -(m.B[I,J,T]-m.Vreactor[I,J]) <= m.beta_max[I,J]*(1-m.X[I,J,T])  
    m.linking2=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
      
    #There is an important assumption here (discussed before): If a given task I is executed multiple times in reactor J, then it is always executed the same way, i.e., same batch size, same time 

    #2) Variable batch costs #TODO

    #Note that processing time is constant, hence it was linked before as a parameter to define the discretization of reactor differential equations 

    #TODO: MAKE TIME DIMENTIONLESS IDEA!!!!!!!!!!!!!!!!!. MAYBE THAT WAY I CAN MAKE TIME A VARIABLE!!!!!!

    #-----------Objective function----------------------------------------------
    # cost minimization
    def _obj1(m):
        return sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)<=objective+epsilon
    m.obj1=pe.Constraint(rule=_obj1,doc='feasibility of objective value 1')

    # def _obj2(m):
    #     return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)>=objective-epsilon
    # m.obj2=pe.Constraint(rule=_obj2,doc='feasibility of objective value 2')

    def _obj_dummy(m):
        return sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj_dummy=pe.Objective(rule=_obj_dummy,sense=pe.minimize,doc='dummy objective')
    
    
    # # -------Reformulation----------------------------------------------------
    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        return math.floor((m.T.__len__()-1)/m.tau[I,J])  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    def _Nref_bounds(m,I,J):
        return (0,m.lastN[I,J])
    m.Nref=pe.Var(m.I_J,within=pe.Integers,bounds=_Nref_bounds,doc='reformulation variables from 0 to lastN')
    
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==m.Nref[I,J]
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')
  
    #m.EXACTLY_CONST.pprint()
    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=6 #Keep Fhot constant every three discretization points
    keep_constant_Fcold=6 #Keep Fcold constant every three discretization points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:
            discretizer.apply_to(m, nfe=20, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
            
            
            #------Constant control
            def _Constant_control1(m,N):
                if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fhot!=0) or (N==m.N[I,J].last()):
                    return m.Fhot[I,J][N] == m.Fhot[I,J][m.N[I,J].prev(N)]
                else:
                    return pe.Constraint.Skip
            m.Constant_control1[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control1,doc='Constant control action every keep_constant_Fhot discrete points and the last one')
            setattr(m,'Constant_control1_(%s,%s)' %(I,J),m.Constant_control1[I,J])

            def _Constant_control2(m,N):
                if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fcold!=0) or (N==m.N[I,J].last()):
                    return m.Fcold[I,J][N] == m.Fcold[I,J][m.N[I,J].prev(N)]
                else:
                    return pe.Constraint.Skip
            m.Constant_control2[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control2,doc='Constant control action every keep_constant_Fcold discrete points and the last one')
            setattr(m,'Constant_control2_(%s,%s)' %(I,J),m.Constant_control2[I,J])
            # m.Constant_control1[I,J].pprint()  
            # m.Constant_control2[I,J].pprint()             
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    return m

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    # relaxed problem
    m=build_scheduling_Boolean_cost_min_relaxed()
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    options={}
    start=time.time()
    m_solved=solve_subproblem(m,subproblem_solver = 'conopt4',subproblem_solver_options= options,timelimit= 1000000,gams_output = False,tee=True,rel_tol = 1e-6)   
    end=time.time()
    print('CONOPT4 time (reformulated, relaxed)='+str(end-start))
    for I_J in m_solved.I_J:
        print(1+pe.value(m_solved.Nref[I_J]))

    # ###-----NEW SCHEDULING ALGORITHM FOR COST MINIMIZATION------
    lower_obj=pe.value(m_solved.obj) #initialization of cut
    model_fun_simplified=build_scheduling_Boolean_cost_min_simplified
    model_fun_feasibility=build_scheduling_Boolean_cost_min_feasibility
    logic_fun=problem_logic_scheduling

    # master_max_iter=1000#master iterations
    initialization=[1,1,1,2,2,2,2,2,2,2]
    infinity_val=1e+6
    min_epsilon_improvement=5#todo: this should agree with the minimum coefficient in the objective function
    absolute_gap=0.01
    nlp_solver='dicopt'
    neigh=neighborhood_k_eq_2(len(initialization))
    maxiter=1000 #benders decomposition
    mastertee=True
    kwargs={}
    m=model_fun_simplified(**kwargs)
    ext_ref = {m.Z: m.Nref} #reformulation sets and variables


    #lower_obj=1664
    start=time.time()

    [important_info,D,x_actual,m_solved]=run_function_dbd_scheduling_cost_min_nonlinear_ref_2(model_fun_feasibility,lower_obj,absolute_gap,min_epsilon_improvement,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun_simplified,kwargs,use_random=False,sub_solver_opt={}, tee=True)
   
   

    ##-----END OF NEW SCHEDULING ALGORITHM FOR COST MINIMIZATION



    # kwargs={}
    # # DICOPT or SBB solution reformulated
    # model_fun =build_scheduling_Boolean_cost_min
    # m=model_fun(**kwargs)
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # options=    {'add_options':[
    #     'GAMS_MODEL.optfile = 1;'
    #     '\n'
    #     '$onecho > sbb.opt \n'
    #     'nodlim 1000000\n'
    #     'memnodes 1000000\n'
    #     '$offecho \n']}
    # start=time.time()
    # m_solved=solve_subproblem(m,subproblem_solver = 'bonmin',subproblem_solver_options= options,timelimit= 1000,gams_output = False,tee= True,rel_tol = 0)   
    # end=time.time()
    # print('DICOPT time (reformulated)='+str(end-start))


    # model_fun =build_scheduling_Original_cost_min
    # logic_fun=problem_logic_scheduling
    # #DICOPT solution
    # m=model_fun(**kwargs)
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # options=    {'add_options':[
    #     'GAMS_MODEL.optfile = 1;'
    #     '\n'
    #     '$onecho > dicopt.opt \n'
    #     'feaspump 2\n'        
    #     '$offecho \n']}

    # start=time.time()
    # m_solved=solve_subproblem(m,subproblem_solver = 'antigone',subproblem_solver_options= options,timelimit= 1000,gams_output = False,tee= True,rel_tol = 0)   
    # end=time.time()
    # print('DICOPT time ='+str(end-start))










    # textbuffer = io.StringIO()
    # for v in m_solved.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # for v in m_solved.component_data_objects(pe.Objective, descend_into=True):
    #     textbuffer.write('Objective value '+str(pe.value(v)))
    #     textbuffer.write('\n')
    #     # v.pprint(textbuffer)
    #     # textbuffer.write('\n')
    # with open('MINLP_solution.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

#     ### SOLVE USING DSDA AND DBD
#     kwargs={}
#     model_fun =dsda_dbd_model
#     logic_fun=problem_logic_scheduling
#     m=model_fun(**kwargs)
#     ext_ref = {m.Z: m.Nref} #reformulation sets and variables
#     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
#     max_cplex_sol=1 #maximum number of solutions
#     dict_known={}
#     for sol in range(1,max_cplex_sol+1):

#         m=model_fun(**kwargs)
#         pe.TransformationFactory('core.logical_to_linear').apply_to(m)
#         options=   {'add_options':['option nlp=conopt4;','GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}
#         if sol==max_cplex_sol:
#             start=time.time()
#         m_solved=solve_subproblem(m,subproblem_solver = 'dicopt',subproblem_solver_options= options,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)   
#         if sol==max_cplex_sol:
#             end=time.time()
#         #print(m_solved.dsda_status)   
#         # solved=generate_initialization(m=m_solved,model_name='maravelias_cplex_reformualted_profit_max')
#         empty_list=[]
#         for I_J in m_solved.I_J:
#             for N in m_solved.Nref:
#                 if pe.value(m_solved.Z_binary[N,I_J])==1:
#                     # print('ext_Var associated to '+str(I_J)+'is '+str(N+1))
#                     empty_list.append(N+1)
#         # print(pe.value(m_solved.obj))
#         dict_known[tuple(empty_list)]=pe.value(m_solved.obj)
#         updated_known=dict_known.copy()

#     required_time=end-start
#     print('Time required to evaluate '+str(max_cplex_sol)+' MIP solutions: '+str(required_time))
#     print(dict_known)
# # USE THIS IF THE PROBLEM IS NOT A FEASIBILITY PROBLEM WHEN SOLVING DIFFERENT INITIALZIATION TRIALS (E.G. PROFIT MAX)
#     # for trial in dict_known:
#     #     m=build_scheduling()       
#     #     external_ref(m,list(trial),logic_fun,reformulation_dict,tee=False)
#     #     m_solved2=solve_subproblem(m,subproblem_solver = 'cplex',subproblem_solver_options= {},timelimit= 1000000,gams_output = False,tee= True,rel_tol = 0) 
#     #     updated_known[trial]=pe.value(m_solved2.obj)
#     # print(updated_known)   


#     m=model_fun(**kwargs)
#     initialization=list(min(dict_known, key=lambda k: dict_known[k]) )
#     infinity_val=1e+6
#     nlp_solver='dicopt'
#     sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver conopt4\n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}
#     neigh=neighborhood_k_eq_2(len(initialization))
#     maxiter=1000
#     # DBD
#     [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
#     [important_info,important_info_preprocessing,D,x_actual]=run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,known_solutions=updated_known)
#     print('obj= ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
    
#      #DSDA
#     # m_init_fixed = external_ref(m=m,x=initialization,extra_logic_function=logic_fun,dict_extvar=ext_ref,tee=False)
#     # m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver='baron', timelimit=10000, tee=False)
#     # init_path = generate_initialization(m=m_init_solved)   
#     start=time.time()
#     D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,{},initialization,ext_ref,logic_fun,k = '2',provide_starting_initialization= True,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 1000,timelimit = 3600,gams_output = False,tee= False,global_tee = True,rel_tol = 1e-3)
#     end=time.time()
#     print('Objective='+str(pe.value(D_SDAsol.obj))+', best='+str(routeDSDA[-1]))
#     print('cputime= '+str(end-start))   



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