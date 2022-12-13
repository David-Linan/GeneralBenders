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

def reactor_dynamics():

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling')

    m.delta=pe.Param(initialize=0.5,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required

    #Main sets
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)  

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


    #Cost info
    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')


#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    _minTau={}
    _minTau['R1','R_large']=math.ceil(1/m.delta)
    _minTau['R1','R_small']=math.ceil(1/m.delta)

    _minTau['R2','R_large']=math.ceil(1/m.delta) 
    _minTau['R2','R_small']=math.ceil(1/m.delta)

    _minTau['R3','R_large']=math.ceil(1/m.delta)
    _minTau['R3','R_small']=math.ceil(1/m.delta)
    m.minTau=pe.Param(m.I_reactions,m.J_reactors,initialize=_minTau,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    _maxTau={}
    _maxTau['R1','R_large']=math.ceil(4/m.delta)
    _maxTau['R1','R_small']=math.ceil(4/m.delta) 

    _maxTau['R2','R_large']=math.ceil(4/m.delta) 
    _maxTau['R2','R_small']=math.ceil(4/m.delta) 

    _maxTau['R3','R_large']=math.ceil(4/m.delta)
    _maxTau['R3','R_small']=math.ceil(4/m.delta)
    m.maxTau=pe.Param(m.I_reactions,m.J_reactors,initialize=_maxTau,doc='Maximum number of discrete elements required to complete task [dimensionless]')


    def _varTime_bounds(m,I,J):
        # return (m.minTau[I,J]*m.delta,m.maxTau[I,J]*m.delta)
        return (0,m.maxTau[I,J]*m.delta)
    m.varTime=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')
    def _Vreactor_bounds(m,I,J):
        return (m.beta_min[I,J],m.beta_max[I,J])
    m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables


    for I in m.I_reactions:
        for J in m.J_reactors:
            m.Vreactor[I,J].fix(m.beta_max[I,J])
    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         m.Vreactor[I,J].fix((m.beta_max[I,J]+m.beta_min[I,J])/2)
    # m.Vreactor['R1','R_large'].fix(1.25)
    # m.Vreactor['R1','R_small'].fix(0.1)
    # m.Vreactor['R2','R_large'].fix(1.5)
    # m.Vreactor['R2','R_small'].fix(0.9999999999999999)
    # m.Vreactor['R3','R_large'].fix(0.6666666666666667)
    # m.Vreactor['R3','R_small'].fix(0.9999999999999999)
    #-----------Reactors dynamic models--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
        
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
    m.dCdtheta={} # Composition derivatives
    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    #Differential equations
    m.c_dCdtheta={}
    m.c_dTRdtheta={}
    m.c_dTJdtheta={}
    
    #Final constraint
    m.finalCon={}
    m.finalTemp={}    

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={}    

    for I in m.I_reactions:
        m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
        setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
        for J in m.J_reactors:
            m.N[I,J]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_%s_%s' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


            def _Cvar_bounds(m,N,Q):
                return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
            m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
            setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

            def _TRvar_bounds(m,N):
                # return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
                return (295,m.T_R_max[J])
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
                # return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
                return (295,m.T_J_max[J])
            m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
            setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

            m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

            m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

            m.dCdtheta[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivative of composition')
            setattr(m,'dCdtheta_(%s,%s)' %(I,J),m.dCdtheta[I,J])

            m.dTRdtheta[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
            setattr(m,'dTRdtheta_(%s,%s)' %(I,J),m.dTRdtheta[I,J])

            m.dTJdtheta[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
            setattr(m,'dTJdtheta_(%s,%s)' %(I,J),m.dTJdtheta[I,J])

            def _dCdtheta(m,N,Q):
                if N == m.N[I,J].first(): 
                    return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
                else:                                         #This is what the author calls Rb
                    return m.dCdtheta[I,J][N,Q] == m.varTime[I,J]*(m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1])) 
            m.c_dCdtheta[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdtheta)
            setattr(m,'c_dCdtheta_(%s,%s)' %(I,J),m.c_dCdtheta[I,J])


            def _dTRdtheta(m,N):
                if N == m.N[I,J].first():
                    return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
                else:
                    return m.dTRdtheta[I,J][N] == m.varTime[I,J]*((((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I])) ) 
            m.c_dTRdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdtheta)
            setattr(m,'c_dTRdtheta_(%s,%s)' %(I,J),m.c_dTRdtheta[I,J])
            # m.c_dTRdt[I,J].pprint()

            def _dTJdtheta(m,N):
                if N == m.N[I,J].first():
                    return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
                else:
                    return m.dTJdtheta[I,J][N] == m.varTime[I,J]*((((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
            m.c_dTJdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdtheta)
            setattr(m,'c_dTJdtheta_(%s,%s)' %(I,J),m.c_dTJdtheta[I,J])
            
            
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
           
            
           # Integrals for cost calculation
            def _Integral_hot_bounds(m,N):
                return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
            m.Integral_hot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
            setattr(m,'Integral_hot_%s_%s' %(I,J),m.Integral_hot[I,J])
            def _Integral_cold_bounds(m,N):
                return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
            m.Integral_cold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
            setattr(m,'Integral_cold_%s_%s' %(I,J),m.Integral_cold[I,J])
            
            m.dIntegral_hotdtheta[I,J]=dae.DerivativeVar(m.Integral_hot[I,J], withrespectto=m.N[I,J], doc='Derivative of hot integral')
            setattr(m,'dIntegral_hotdtheta_(%s,%s)' %(I,J),m.dIntegral_hotdtheta[I,J])            
            m.dIntegral_colddtheta[I,J]=dae.DerivativeVar(m.Integral_cold[I,J], withrespectto=m.N[I,J], doc='Derivative of cold integral')
            setattr(m,'dIntegral_colddtheta_(%s,%s)' %(I,J),m.dIntegral_colddtheta[I,J])


            def _c_dIntegral_hotdtheta(m,N):
                if N == m.N[I,J].first():
                    return m.Integral_hot[I,J][N]==0
                else:
                    return m.dIntegral_hotdtheta[I,J][N]==m.varTime[I,J]*m.Fhot[I,J][N]
            m.c_dIntegral_hotdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_c_dIntegral_hotdtheta)
            setattr(m,'c_dIntegral_hotdtheta_(%s,%s)' %(I,J),m.c_dIntegral_hotdtheta[I,J])   
            
            def _c_dIntegral_colddtheta(m,N):
                if N == m.N[I,J].first():
                    return m.Integral_cold[I,J][N]==0
                else:
                    return m.dIntegral_colddtheta[I,J][N]==m.varTime[I,J]*m.Fcold[I,J][N]
            m.c_dIntegral_colddtheta[I,J]=pe.Constraint(m.N[I,J],rule=_c_dIntegral_colddtheta)
            setattr(m,'c_dIntegral_colddtheta_(%s,%s)' %(I,J),m.c_dIntegral_colddtheta[I,J])  
            
            # m.c_dCdtheta['R3','R_large'].display()  
            # m.Cvar['R3','R_large'].display()  
            # m.Q_balance['R1'].pprint()
            # m.Q_balance['R2'].pprint()
            # m.Q_balance['R3'].pprint()

    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=9 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_Fcold=9 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:        #TODO: Depending on selected variable time the number of discretization points must change accordingly
            discretizer.apply_to(m, nfe=30, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_reactions:
        for J in m.J_reactors:  
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

    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    def _I_J(m):
        return ((I,J) for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    
    _Nref={} #TODO: This should be an input to the model function. Other inputs may be m.delta, processing times and reactor volumes which are related to B
    _Nref['R1','R_large']=1
    _Nref['R1','R_small']=1

    _Nref['R2','R_large']=1
    _Nref['R2','R_small']=1

    _Nref['R3','R_large']=1
    _Nref['R3','R_small']=1 
    m.Nref=pe.Param(m.I_J,initialize=_Nref,doc='reformulation variables from 0 to lastN')


    m.TCP3=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        # return m.TCP3== sum(sum(m.Nref[I,J]*(m.hot_cost*m.Integral_hot[I, J][m.N[I, J].last()] + m.cold_cost*m.Integral_cold[I, J][m.N[I, J].last()]) for I in m.I_reactions)for J in m.J_reactors)
        return m.TCP3==sum(sum(m.Nref[I,J]*m.varTime[I,J] for I in m.I_reactions)for J in m.J_reactors) 
    m.C_TCP3=pe.Constraint(rule=_C_TCP3)  

    def _obj(m):
        return (m.TCP3)/100
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)   
    return m

def reactor_dynamics_verif():

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling')

    m.delta=pe.Param(initialize=0.5,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required

    #Main sets
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)  

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


    #Cost info
    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')


#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    _minTau={}
    _minTau['R1','R_large']=math.ceil(2.1366220886694527/m.delta)
    _minTau['R1','R_small']=math.ceil(2.2294153194353483/m.delta) 

    _minTau['R2','R_large']=math.ceil(2.8556474598625035/m.delta)  
    _minTau['R2','R_small']=math.ceil(2.9181422480152954/m.delta) 

    _minTau['R3','R_large']=math.ceil(1.5917112584529056/m.delta)
    _minTau['R3','R_small']=math.ceil(1.675857698391256/m.delta)
    m.minTau=pe.Param(m.I_reactions,m.J_reactors,initialize=_minTau,doc='Minimum number of discrete elements required to complete task [dimensionless]')

#TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
    _maxTau={}
    _maxTau['R1','R_large']=math.ceil(2.1366220886694527/m.delta)
    _maxTau['R1','R_small']=math.ceil(2.2294153194353483/m.delta) 

    _maxTau['R2','R_large']=math.ceil(2.8556474598625035/m.delta) 
    _maxTau['R2','R_small']=math.ceil(2.9181422480152954/m.delta) 

    _maxTau['R3','R_large']=math.ceil(1.5917112584529056/m.delta)
    _maxTau['R3','R_small']=math.ceil(1.675857698391256/m.delta)
    m.maxTau=pe.Param(m.I_reactions,m.J_reactors,initialize=_maxTau,doc='Maximum number of discrete elements required to complete task [dimensionless]')


    def _varTime_bounds(m,I,J):
        return (m.minTau[I,J]*m.delta,m.maxTau[I,J]*m.delta)
        # return (0,m.maxTau[I,J]*m.delta)
    m.varTime=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')
    def _Vreactor_bounds(m,I,J):
        return (m.beta_min[I,J],m.beta_max[I,J])
    m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables


    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         m.Vreactor[I,J].fix(m.beta_max[I,J])

    m.Vreactor['R1','R_large'].fix(1.5)
    m.Vreactor['R1','R_small'].fix(1)
    m.Vreactor['R2','R_large'].fix(1.5)
    m.Vreactor['R2','R_small'].fix(1)
    m.Vreactor['R3','R_large'].fix(1)
    m.Vreactor['R3','R_small'].fix(1)
    #-----------Reactors dynamic models--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
        
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
    m.dCdtheta={} # Composition derivatives
    m.dTRdtheta={} #Reactor temperature derivatives
    m.dTJdtheta={} #Jacket temperature derivatives

    #Differential equations
    m.c_dCdtheta={}
    m.c_dTRdtheta={}
    m.c_dTJdtheta={}
    
    #Final constraint
    m.finalCon={}
    m.finalTemp={}    

    #Integrals for cost calcualtion
    m.Integral_hot={}
    m.Integral_cold={}
    
    m.dIntegral_hotdtheta={}
    m.dIntegral_colddtheta={}
    m.c_dIntegral_hotdtheta={}
    m.c_dIntegral_colddtheta={}    

    for I in m.I_reactions:
        m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
        setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
        for J in m.J_reactors:
            m.N[I,J]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_%s_%s' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


            def _Cvar_bounds(m,N,Q):
                return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
            m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
            setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

            def _TRvar_bounds(m,N):
                # return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
                return (295,m.T_R_max[J])
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
                # return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
                return (295,m.T_J_max[J])
            m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
            setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

            m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

            m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

            m.dCdtheta[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivative of composition')
            setattr(m,'dCdtheta_(%s,%s)' %(I,J),m.dCdtheta[I,J])

            m.dTRdtheta[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
            setattr(m,'dTRdtheta_(%s,%s)' %(I,J),m.dTRdtheta[I,J])

            m.dTJdtheta[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
            setattr(m,'dTJdtheta_(%s,%s)' %(I,J),m.dTJdtheta[I,J])

            def _dCdtheta(m,N,Q):
                if N == m.N[I,J].first(): 
                    return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
                else:                                         #This is what the author calls Rb
                    return m.dCdtheta[I,J][N,Q] == m.varTime[I,J]*(m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1])) 
            m.c_dCdtheta[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdtheta)
            setattr(m,'c_dCdtheta_(%s,%s)' %(I,J),m.c_dCdtheta[I,J])


            def _dTRdtheta(m,N):
                if N == m.N[I,J].first():
                    return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
                else:
                    return m.dTRdtheta[I,J][N] == m.varTime[I,J]*((((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I])) ) 
            m.c_dTRdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdtheta)
            setattr(m,'c_dTRdtheta_(%s,%s)' %(I,J),m.c_dTRdtheta[I,J])
            # m.c_dTRdt[I,J].pprint()

            def _dTJdtheta(m,N):
                if N == m.N[I,J].first():
                    return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
                else:
                    return m.dTJdtheta[I,J][N] == m.varTime[I,J]*((((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
            m.c_dTJdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdtheta)
            setattr(m,'c_dTJdtheta_(%s,%s)' %(I,J),m.c_dTJdtheta[I,J])
            
            
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
           
            
           # Integrals for cost calculation
            def _Integral_hot_bounds(m,N):
                return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
            m.Integral_hot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
            setattr(m,'Integral_hot_%s_%s' %(I,J),m.Integral_hot[I,J])
            def _Integral_cold_bounds(m,N):
                return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
            m.Integral_cold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
            setattr(m,'Integral_cold_%s_%s' %(I,J),m.Integral_cold[I,J])
            
            m.dIntegral_hotdtheta[I,J]=dae.DerivativeVar(m.Integral_hot[I,J], withrespectto=m.N[I,J], doc='Derivative of hot integral')
            setattr(m,'dIntegral_hotdtheta_(%s,%s)' %(I,J),m.dIntegral_hotdtheta[I,J])            
            m.dIntegral_colddtheta[I,J]=dae.DerivativeVar(m.Integral_cold[I,J], withrespectto=m.N[I,J], doc='Derivative of cold integral')
            setattr(m,'dIntegral_colddtheta_(%s,%s)' %(I,J),m.dIntegral_colddtheta[I,J])


            def _c_dIntegral_hotdtheta(m,N):
                if N == m.N[I,J].first():
                    return m.Integral_hot[I,J][N]==0
                else:
                    return m.dIntegral_hotdtheta[I,J][N]==m.varTime[I,J]*m.Fhot[I,J][N]
            m.c_dIntegral_hotdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_c_dIntegral_hotdtheta)
            setattr(m,'c_dIntegral_hotdtheta_(%s,%s)' %(I,J),m.c_dIntegral_hotdtheta[I,J])   
            
            def _c_dIntegral_colddtheta(m,N):
                if N == m.N[I,J].first():
                    return m.Integral_cold[I,J][N]==0
                else:
                    return m.dIntegral_colddtheta[I,J][N]==m.varTime[I,J]*m.Fcold[I,J][N]
            m.c_dIntegral_colddtheta[I,J]=pe.Constraint(m.N[I,J],rule=_c_dIntegral_colddtheta)
            setattr(m,'c_dIntegral_colddtheta_(%s,%s)' %(I,J),m.c_dIntegral_colddtheta[I,J])  
            
            # m.c_dCdtheta['R3','R_large'].display()  
            # m.Cvar['R3','R_large'].display()  
            # m.Q_balance['R1'].pprint()
            # m.Q_balance['R2'].pprint()
            # m.Q_balance['R3'].pprint()

    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=9 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_Fcold=9 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:        #TODO: Depending on selected variable time the number of discretization points must change accordingly
            discretizer.apply_to(m, nfe=30, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # print(dir(m.N[I,J]))
            # print(m.N[I,J].value_list)
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
            #------Constant control
    for I in m.I_reactions:
        for J in m.J_reactors:  
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

    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------
    def _I_J(m):
        return ((I,J) for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    
    _Nref={} #TODO: This should be an input to the model function. Other inputs may be m.delta, processing times and reactor volumes which are related to B
    _Nref['R1','R_large']=1
    _Nref['R1','R_small']=1

    _Nref['R2','R_large']=2
    _Nref['R2','R_small']=2

    _Nref['R3','R_large']=1
    _Nref['R3','R_small']=1 


    m.Nref=pe.Param(m.I_J,initialize=_Nref,doc='reformulation variables from 0 to lastN')


    m.TCP3=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(m.Nref[I,J]*(m.hot_cost*m.Integral_hot[I, J][m.N[I, J].last()] + m.cold_cost*m.Integral_cold[I, J][m.N[I, J].last()]) for I in m.I_reactions)for J in m.J_reactors)
        # return m.TCP3==sum(sum(m.Nref[I,J]*m.varTime[I,J] for I in m.I_reactions)for J in m.J_reactors) 
    m.C_TCP3=pe.Constraint(rule=_C_TCP3)  

    def _obj(m):
        return (m.TCP3)/100
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)   
    return m

if __name__ == "__main__":
    m=reactor_dynamics()