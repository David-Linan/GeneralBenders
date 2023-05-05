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
import pickle
import numpy as np
from functions.cuts_functions import convex_clousure
from itertools import product
import time

# def reactor_dynamics_one_time_step():

#     # ------------pyomo model------------------------------------------------
#     #------------------------------------------------------------------------

#     m = pe.ConcreteModel(name='reactor_dynamics_one_time_Step')

#     m.delta=pe.Param(initialize=0.5,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required

#     #Main sets
#     m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
#     m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
#     m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
#     m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
#     #Subsets
#     m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
#     m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)  

#     _I_i_k_minus={}
#     _I_i_k_minus['Mix','S1']=1
#     _I_i_k_minus['Mix','M1']=1

#     _I_i_k_minus['R1','M2']=1
#     _I_i_k_minus['R1','M3']=1

#     _I_i_k_minus['R2','I1']=1
#     _I_i_k_minus['R2','I2']=1

#     _I_i_k_minus['R3','I3']=1
#     _I_i_k_minus['R3','M3']=1

#     _I_i_k_minus['Sep','I4']=1

#     _I_i_k_minus['Pack1','I5']=1

#     _I_i_k_minus['Pack2','I6']=1
#     m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

#     _I_i_k_plus={}
#     _I_i_k_plus['Mix','I1']=1

#     _I_i_k_plus['R1','I2']=1

#     _I_i_k_plus['R2','I3']=1
#     _I_i_k_plus['R2','I5']=1

#     _I_i_k_plus['R3','I4']=1

#     _I_i_k_plus['Sep','W1']=1
#     _I_i_k_plus['Sep','I6']=1

#     _I_i_k_plus['Pack1','P1']=1

#     _I_i_k_plus['Pack2','P2']=1
#     m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")

#     _rho_minus={}
#     _rho_minus['Mix','S1']=3/5
#     _rho_minus['Mix','M1']=2/5

#     _rho_minus['R1','M2']=1/2
#     _rho_minus['R1','M3']=1/2

#     _rho_minus['R2','I1']=1/2
#     _rho_minus['R2','I2']=1/2

#     _rho_minus['R3','I3']=1/2
#     _rho_minus['R3','M3']=1/2

#     _rho_minus['Sep','I4']=1

#     _rho_minus['Pack1','I5']=1

#     _rho_minus['Pack2','I6']=1
#     m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")

#     _rho_plus={}
#     _rho_plus['Mix','I1']=1

#     _rho_plus['R1','I2']=1

#     _rho_plus['R2','I3']=3/5
#     _rho_plus['R2','I5']=2/5

#     _rho_plus['R3','I4']=1

#     _rho_plus['Sep','W1']=2/5
#     _rho_plus['Sep','I6']=3/5

#     _rho_plus['Pack1','P1']=1

#     _rho_plus['Pack2','P2']=1
#     m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")

#     _beta_min={}
#     _beta_min['Mix','Mix']=0.2

#     _beta_min['R1','R_large']=0.15
#     _beta_min['R1','R_small']=0.1

#     _beta_min['R2','R_large']=0.15
#     _beta_min['R2','R_small']=0.1

#     _beta_min['R3','R_large']=0.15
#     _beta_min['R3','R_small']=0.1

#     _beta_min['Sep','Sep']=0.2

#     _beta_min['Pack1','Pack']=0.1
#     _beta_min['Pack2','Pack']=0.1
#     m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

#     _beta_max={}
#     _beta_max['Mix','Mix']=2

#     _beta_max['R1','R_large']=1.5
#     _beta_max['R1','R_small']=1

#     _beta_max['R2','R_large']=1.5
#     _beta_max['R2','R_small']=1

#     _beta_max['R3','R_large']=1.5
#     _beta_max['R3','R_small']=1

#     _beta_max['Sep','Sep']=2

#     _beta_max['Pack1','Pack']=1
#     _beta_max['Pack2','Pack']=1
#     m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i [m^3]")# Note that I am using volumes, altough mass would be more general.

#     _I_i_j_prod={}
#     _I_i_j_prod['Mix','Mix']=1

#     _I_i_j_prod['R1','R_large']=1
#     _I_i_j_prod['R1','R_small']=1

#     _I_i_j_prod['R2','R_large']=1
#     _I_i_j_prod['R2','R_small']=1

#     _I_i_j_prod['R3','R_large']=1
#     _I_i_j_prod['R3','R_small']=1

#     _I_i_j_prod['Sep','Sep']=1

#     _I_i_j_prod['Pack1','Pack']=1
#     _I_i_j_prod['Pack2','Pack']=1
#     m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

#     #Parameters of reactor units
#     # m.v_R_max = pe.Param(m.J_reactors,initialize={'R_large':1.5,'R_small':1},doc='Maximum capacity of the reactor [m^3]') #TODO: Probably not used
#     m.v_J=pe.Param(m.J_reactors,initialize={'R_large':0.5,'R_small':0.3},doc='Volume of the Jacket [m^3]')
#     m.rho_J=pe.Param(m.J_reactors,initialize={'R_large':1e+3,'R_small':1e+3},doc='Density of the jacket [kg/m^3]')
#     m.c_J=pe.Param(m.J_reactors,initialize={'R_large':4.2,'R_small':4.2},doc='Heat capacity of jacket [kJ/kg K]')
#     m.ua=pe.Param(m.J_reactors,initialize={'R_large':3e+4,'R_small':2e+4},doc='Heat transfer coefficient [kJ/h K]')
#     m.T_H= pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Temperature of heating water [K]')
#     m.T_C=pe.Param(m.J_reactors,initialize={'R_large':300,'R_small':300},doc='Temperature of cooling water [K]')
#     m.T_R_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of reactor [K]')
#     m.T_J_max=pe.Param(m.J_reactors,initialize={'R_large':370,'R_small':370},doc='Maximum temperature of jacket [K]')
#     m.F_max=pe.Param(m.J_reactors,initialize={'R_large':10,'R_small':5},doc='Maximum flow rate of heating and cooling water [m^3/h]')

#     #Parameters of reaction tasks
#     m.z=pe.Param(m.I_reactions,initialize={'R1':1e+7,'R2':1.2e+3,'R3':2e+4},doc='Pre-exponential factors [m^3/kmol h]')
#     m.er=pe.Param(m.I_reactions,initialize={'R1':5e+3,'R2':2e+3,'R3':3e+3},doc='Normalized activation energy [K]')
#     m.delta_h=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':-2e+3,'R3':5e+3},doc='Heat of reaction [kJ/kmol]')
#     m.rho_R=pe.Param(m.I_reactions,initialize={'R1':1e+3,'R2':1e+3,'R3':1e+3},doc='Density of reaction mixture [kg/m^3]')
#     m.c_R=pe.Param(m.I_reactions,initialize={'R1':3,'R2':3.5,'R3':4},doc='Heat capacity of reaction mixture [kJ/kg K]')
#     _coef={}
    
#     _coef['R1','B']=-1
#     _coef['R1','C']=-1
#     _coef['R1','D']=2

#     _coef['R2','A']=-1
#     _coef['R2','D']=-1
#     _coef['R2','E']=2

#     _coef['R3','C']=-1
#     _coef['R3','E']=-1
#     _coef['R3','F']=1
#     m.coef=pe.Param(m.I_reactions,m.Q,default=0,initialize=_coef,doc='Stoichiometric coefficient')

#     #Composition of states #TODO: In general the problem is formulated using mass balances, but in the paper there is an assumption, so balances are performed in terms of volumes
#     _C={}
#     _C['M1','A']=5

#     _C['M2','B']=2

#     _C['M3','C']=2

#     _C['P1','E']=1.8 #This has the same composition as I3 and I5
#     _C['P1','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['P1','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['P1','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['P1','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)    

#     _C['P2','F']=1 #Other component compositions are unknown due to unknown conditions in distillation. Perfect separation is assumed

#     _C['I1','A']=2

#     _C['I2','B']=0.05
#     _C['I2','C']=0.05 
#     _C['I2','D']=1.9   #TODO: Note that composition of output states from reactors is being specified, i.e., I already know the exact desired composition I want as output from each reactor

#     _C['I3','E']=1.8  #TODO (solved): There are "others" component here. Check if there is any assumption, given that inerts are usually considered in balances
#     _C['I3','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I3','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I3','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I3','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

#     _C['I4','F']=0.8
#     _C['I4','C']=0.2125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I4','E']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I4','A']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I4','D']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I4','B']=0.0125 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

#     _C['I5','E']=1.8 
#     _C['I5','A']=0.1 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I5','D']=0.05 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I5','B']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)
#     _C['I5','C']=0.025 #Based on the abouve comment and a balance, I had to add this (based on mole balance)

#     _C['I6','F']=1 #Other component compositions are unknown due to unknown conditions in distillation
#     # Waste composition is unknown
#     m.C=pe.Param(m.K,m.Q,initialize=_C,default=0,doc='Composition of different reactive components at each state [kmol/m^3]')
    

#     #Initial composition and final composition inside reactors. This is important for dynamics, but these parameters are not going to be used in scheduling

#     def _C_initial(m,I,Q):
#         return sum(m.rho_minus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_minus[I,K]==1)    
#     m.C_initial=pe.Param(m.I_reactions,m.Q,initialize=_C_initial,doc='Initial composition inside reactor for this reaction and component [kmol/m^3]') #TODO: Check assumptions that lead to this equation in you article. Same assumptions here
#     # m.C_initial.display()


#     def _C_final(m,I,Q):
#         return sum(m.rho_plus[I,K]*m.C[K,Q] for K in m.K if m.I_i_k_plus[I,K]==1)
#     m.C_final=pe.Param(m.I_reactions,m.Q,initialize=_C_final,doc='Final composition inside reactor for this reaction and component [kmol/m^3]')


#     # Initial temperature of reactors and heating medium for each task
#     m.T_R_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for reaction temperatures inside reactor [K]')
#     m.T_J_initial=pe.Param(m.I_reactions,initialize={'R1':300,'R2':300,'R3':300},doc='Initial condition for jacket temperatures [K]')
#     # Final temperature of reactions
#     m.T_R_final=pe.Param(m.I_reactions,initialize={'R1':320,'R2':320,'R3':320},doc='Maximum temperature at the end of the reaction [K]')


#     #Cost info
#     m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
#     m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')


# #TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
#     _minTau={}
#     _minTau['R1','R_large']=math.ceil(1/m.delta)
#     _minTau['R1','R_small']=math.ceil(1/m.delta)

#     _minTau['R2','R_large']=math.ceil(1/m.delta) 
#     _minTau['R2','R_small']=math.ceil(1/m.delta)

#     _minTau['R3','R_large']=math.ceil(1/m.delta)
#     _minTau['R3','R_small']=math.ceil(1/m.delta)
#     m.minTau=pe.Param(m.I_reactions,m.J_reactors,initialize=_minTau,doc='Minimum number of discrete elements required to complete task [dimensionless]')

# #TODO: note that I am using the discrete varions of tau here. Hence, these bounds depend on the discretization step. Whenever I try a differnt discretization step I have to change these bounds accordingly
#     _maxTau={}
#     _maxTau['R1','R_large']=math.ceil(4/m.delta)
#     _maxTau['R1','R_small']=math.ceil(4/m.delta) 

#     _maxTau['R2','R_large']=math.ceil(4/m.delta) 
#     _maxTau['R2','R_small']=math.ceil(4/m.delta) 

#     _maxTau['R3','R_large']=math.ceil(4/m.delta)
#     _maxTau['R3','R_small']=math.ceil(4/m.delta)
#     m.maxTau=pe.Param(m.I_reactions,m.J_reactors,initialize=_maxTau,doc='Maximum number of discrete elements required to complete task [dimensionless]')


#     def _varTime_bounds(m,I,J):
#         # return (m.minTau[I,J]*m.delta,m.maxTau[I,J]*m.delta)
#         return (0,m.maxTau[I,J]*m.delta)
#     m.varTime=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_varTime_bounds,doc='Variable processing time for units that consider dynamics [h]')
#     def _Vreactor_bounds(m,I,J):
#         return (m.beta_min[I,J],m.beta_max[I,J])
#     m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables


#     for I in m.I_reactions:
#         for J in m.J_reactors:
#             m.Vreactor[I,J].fix(m.beta_max[I,J])
#     # for I in m.I_reactions:
#     #     for J in m.J_reactors:
#     #         m.Vreactor[I,J].fix((m.beta_max[I,J]+m.beta_min[I,J])/2)
#     # m.Vreactor['R1','R_large'].fix(1.25)
#     # m.Vreactor['R1','R_small'].fix(0.1)
#     # m.Vreactor['R2','R_large'].fix(1.5)
#     # m.Vreactor['R2','R_small'].fix(0.9999999999999999)
#     # m.Vreactor['R3','R_large'].fix(0.6666666666666667)
#     # m.Vreactor['R3','R_small'].fix(0.9999999999999999)
#     #-----------Reactors dynamic models--------------------------------
#     # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
#     # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
#     # !!! The six time grids stand for:
#     # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
#     # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
        
#         #Sets
#     m.N={} #Continuous time set
#     m.Q_balance={} #Species of interest in mole and energy balances

#     #Variables
#     m.Cvar={} #Composition profiles
#     m.TRvar={} #Reactor temperature profiles
#     m.TJvar={} #Jacket temperature profile
#     m.Fhot={} #Hot fluid volumetric flow rate profile (manipulated variable)
#     m.Fcold={} #Cold fluid volumetric flow rate profile (manipulated variable)

#     #Derivativa variables
#     m.dCdtheta={} # Composition derivatives
#     m.dTRdtheta={} #Reactor temperature derivatives
#     m.dTJdtheta={} #Jacket temperature derivatives

#     #Differential equations
#     m.c_dCdtheta={}
#     m.c_dTRdtheta={}
#     m.c_dTJdtheta={}
    
#     #Final constraint
#     m.finalCon={}
#     m.finalTemp={}    

#     #Integrals for cost calcualtion
#     m.Integral_hot={}
#     m.Integral_cold={}
    
#     m.dIntegral_hotdtheta={}
#     m.dIntegral_colddtheta={}
#     m.c_dIntegral_hotdtheta={}
#     m.c_dIntegral_colddtheta={}    

#     for I in m.I_reactions:
#         m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
#         setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
#         for J in m.J_reactors:
#             m.N[I,J]=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set for reaction I in reactor J [-]') #TODO: chek units of time, are they consistent? should I use hours? 
#             setattr(m,'N_%s_%s' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


#             def _Cvar_bounds(m,N,Q):
#                 return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
#             m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
#             setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

#             def _TRvar_bounds(m,N):
#                 # return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
#                 return (295,m.T_R_max[J])
#             m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
#             setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

#             def _TJvar_bounds(m,N):
#                 # return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
#                 return (295,m.T_J_max[J])
#             m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
#             setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

#             m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
#             setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

#             m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
#             setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

#             m.dCdtheta[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivative of composition')
#             setattr(m,'dCdtheta_(%s,%s)' %(I,J),m.dCdtheta[I,J])

#             m.dTRdtheta[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
#             setattr(m,'dTRdtheta_(%s,%s)' %(I,J),m.dTRdtheta[I,J])

#             m.dTJdtheta[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
#             setattr(m,'dTJdtheta_(%s,%s)' %(I,J),m.dTJdtheta[I,J])

#             def _dCdtheta(m,N,Q):
#                 if N == m.N[I,J].first(): 
#                     return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
#                 else:                                         #This is what the author calls Rb
#                     return m.dCdtheta[I,J][N,Q] == m.varTime[I,J]*(m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1])) 
#             m.c_dCdtheta[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdtheta)
#             setattr(m,'c_dCdtheta_(%s,%s)' %(I,J),m.c_dCdtheta[I,J])


#             def _dTRdtheta(m,N):
#                 if N == m.N[I,J].first():
#                     return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
#                 else:
#                     return m.dTRdtheta[I,J][N] == m.varTime[I,J]*((((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I])) ) 
#             m.c_dTRdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdtheta)
#             setattr(m,'c_dTRdtheta_(%s,%s)' %(I,J),m.c_dTRdtheta[I,J])
#             # m.c_dTRdt[I,J].pprint()

#             def _dTJdtheta(m,N):
#                 if N == m.N[I,J].first():
#                     return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
#                 else:
#                     return m.dTJdtheta[I,J][N] == m.varTime[I,J]*((((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J])) ) 
#             m.c_dTJdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdtheta)
#             setattr(m,'c_dTJdtheta_(%s,%s)' %(I,J),m.c_dTJdtheta[I,J])
            
            
#             #Constraints when finishing reaction tasks
            
#             # Final concentration constraint
#             def _finalCon(m,N,Q):
#                 if N==m.N[I,J].last():
#                     return m.Cvar[I,J][N,Q] == m.C_final[I,Q]
#                 else:
#                     return pe.Constraint.Skip
#             m.finalCon[I,J]=pe.Constraint(m.N[I,J],m.Q_balance[I],rule=_finalCon)
#             setattr(m,'finalCon_(%s,%s)' %(I,J),m.finalCon[I,J])
            
#             #Final temperature constraints
            
#             def _finalTemp(m,N):
#                 if N==m.N[I,J].last():
#                     return m.TRvar[I,J][N]<= m.T_R_final[I]
#                 else:
#                     return pe.Constraint.Skip
#             m.finalTemp[I,J]=pe.Constraint(m.N[I,J],rule=_finalTemp)
#             setattr(m,'finalTemp_(%s,%s)' %(I,J),m.finalTemp[I,J])
           
            
#            # Integrals for cost calculation
#             def _Integral_hot_bounds(m,N):
#                 return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
#             m.Integral_hot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_hot_bounds,doc='Integral of F_hot evaluated at every point [m^3]')
#             setattr(m,'Integral_hot_%s_%s' %(I,J),m.Integral_hot[I,J])
#             def _Integral_cold_bounds(m,N):
#                 return (0,m.F_max[J]*m.maxTau[I,J]*m.delta)
#             m.Integral_cold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,initialize=0,bounds=_Integral_cold_bounds,doc='Integral of F_cold evaluated at every point [m^3]')
#             setattr(m,'Integral_cold_%s_%s' %(I,J),m.Integral_cold[I,J])
            
#             m.dIntegral_hotdtheta[I,J]=dae.DerivativeVar(m.Integral_hot[I,J], withrespectto=m.N[I,J], doc='Derivative of hot integral')
#             setattr(m,'dIntegral_hotdtheta_(%s,%s)' %(I,J),m.dIntegral_hotdtheta[I,J])            
#             m.dIntegral_colddtheta[I,J]=dae.DerivativeVar(m.Integral_cold[I,J], withrespectto=m.N[I,J], doc='Derivative of cold integral')
#             setattr(m,'dIntegral_colddtheta_(%s,%s)' %(I,J),m.dIntegral_colddtheta[I,J])


#             def _c_dIntegral_hotdtheta(m,N):
#                 if N == m.N[I,J].first():
#                     return m.Integral_hot[I,J][N]==0
#                 else:
#                     return m.dIntegral_hotdtheta[I,J][N]==m.varTime[I,J]*m.Fhot[I,J][N]
#             m.c_dIntegral_hotdtheta[I,J]=pe.Constraint(m.N[I,J],rule=_c_dIntegral_hotdtheta)
#             setattr(m,'c_dIntegral_hotdtheta_(%s,%s)' %(I,J),m.c_dIntegral_hotdtheta[I,J])   
            
#             def _c_dIntegral_colddtheta(m,N):
#                 if N == m.N[I,J].first():
#                     return m.Integral_cold[I,J][N]==0
#                 else:
#                     return m.dIntegral_colddtheta[I,J][N]==m.varTime[I,J]*m.Fcold[I,J][N]
#             m.c_dIntegral_colddtheta[I,J]=pe.Constraint(m.N[I,J],rule=_c_dIntegral_colddtheta)
#             setattr(m,'c_dIntegral_colddtheta_(%s,%s)' %(I,J),m.c_dIntegral_colddtheta[I,J])  
            
#             # m.c_dCdtheta['R3','R_large'].display()  
#             # m.Cvar['R3','R_large'].display()  
#             # m.Q_balance['R1'].pprint()
#             # m.Q_balance['R2'].pprint()
#             # m.Q_balance['R3'].pprint()

#     # # -------Discretization---------------------------------------------------
#     # discretizer = pe.TransformationFactory('dae.finite_difference')
#     # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
#     # # discretizer = TransformationFactory('dae.collocation')
#     # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
#     #Constant control actions
#     m.Constant_control1={}
#     m.Constant_control2={}
#     keep_constant_Fhot=9 #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
#     keep_constant_Fcold=9 #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


#     discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

#     for I in m.I_reactions:
#         for J in m.J_reactors:        #TODO: Depending on selected variable time the number of discretization points must change accordingly
#             discretizer.apply_to(m, nfe=30, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
#             # print(dir(m.N[I,J]))
#             # print(m.N[I,J].value_list)
#             # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
                        
#             #------Constant control
#     for I in m.I_reactions:
#         for J in m.J_reactors:  
#             def _Constant_control1(m,N):
#                 if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fhot!=0) or (N==m.N[I,J].last()):
#                     return m.Fhot[I,J][N] == m.Fhot[I,J][m.N[I,J].prev(N)]
#                 else:
#                     return pe.Constraint.Skip
#             m.Constant_control1[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control1,doc='Constant control action every keep_constant_Fhot discrete points and the last one')
#             setattr(m,'Constant_control1_(%s,%s)' %(I,J),m.Constant_control1[I,J])

#             def _Constant_control2(m,N):
#                 if (N!=m.N[I,J].first() and (m.N[I,J].ord(N)-1)%keep_constant_Fcold!=0) or (N==m.N[I,J].last()):
#                     return m.Fcold[I,J][N] == m.Fcold[I,J][m.N[I,J].prev(N)]
#                 else:
#                     return pe.Constraint.Skip
#             m.Constant_control2[I,J]=pe.Constraint(m.N[I,J],rule=_Constant_control2,doc='Constant control action every keep_constant_Fcold discrete points and the last one')
#             setattr(m,'Constant_control2_(%s,%s)' %(I,J),m.Constant_control2[I,J])            

#     # # -----------------------------------------------------------------------
#     # # -----------------------------------------------------------------------
#     #-----------Objective function----------------------------------------------
#     def _I_J(m):
#         return ((I,J) for I in m.I_reactions for J in m.J_reactors if m.I_i_j_prod[I,J]==1)
#     m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    
#     _Nref={} #TODO: This should be an input to the model function. Other inputs may be m.delta, processing times and reactor volumes which are related to B
#     _Nref['R1','R_large']=1
#     _Nref['R1','R_small']=1

#     _Nref['R2','R_large']=1
#     _Nref['R2','R_small']=1

#     _Nref['R3','R_large']=1
#     _Nref['R3','R_small']=1 
#     m.Nref=pe.Param(m.I_J,initialize=_Nref,doc='reformulation variables from 0 to lastN')


#     m.TCP3=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
#     def _C_TCP3(m):
#         # return m.TCP3== sum(sum(m.Nref[I,J]*(m.hot_cost*m.Integral_hot[I, J][m.N[I, J].last()] + m.cold_cost*m.Integral_cold[I, J][m.N[I, J].last()]) for I in m.I_reactions)for J in m.J_reactors)
#         return m.TCP3==sum(sum(m.Nref[I,J]*m.varTime[I,J] for I in m.I_reactions)for J in m.J_reactors) 
#     m.C_TCP3=pe.Constraint(rule=_C_TCP3)  

#     def _obj(m):
#         return (m.TCP3)/100
#     m.obj = pe.Objective(rule=_obj, sense=pe.minimize)   
#     return m




def test_process():
    m = pe.ConcreteModel(name='reactor_dynamics')


    m.t=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set [units of time]')


    #PARAMETERS

    w1=13.1
    w2=0.005
    w3=30
    w4=0.94
    w5=1.71
    w6=20
    y10=0.03
    y20=0
    # y1f=1
    # y2f=1


    #VARIABLES

    m.y1=pe.Var(m.t,within=pe.NonNegativeReals,initialize=0.5,bounds=(0,1.075)) #TODO:  how to define these bounds (here I defined them maximizing the value of the state at the end, with negligible sampling time)
    m.y2=pe.Var(m.t,within=pe.NonNegativeReals,initialize=0.5,bounds=(0,1.243)) #TODO: how to define these bounds (here I defined them maximizing the value of the state at the end, with negligible sampling time)
    m.temp=pe.Var(m.t,within=pe.NonNegativeReals,bounds=(20,30))
    m.b1=pe.Var(m.t,within=pe.NonNegativeReals,initialize=1)
    m.b2=pe.Var(m.t,within=pe.NonNegativeReals,initialize=1)
    m.b3=pe.Var(m.t,within=pe.NonNegativeReals,initialize=1)

    m.dy1dt= dae.DerivativeVar(m.y1, withrespectto=m.t, doc='Derivative of y1')
    m.dy2dt= dae.DerivativeVar(m.y2, withrespectto=m.t,doc='Derivative of y2')

    #CONSTRAINTS
    def cdy1(m,t):
        if t == m.t.first(): 
            return m.y1[t] ==y10  # Initial condition
        else:
            return m.dy1dt[t]==m.b1[t]*m.y1[t]-((m.b1[t]/m.b2[t])*((m.y1[t])**2))
    m.cdy1dt=pe.Constraint(m.t,rule=cdy1)

    def cdy2(m,t):
        if t == m.t.first(): 
            return m.y2[t] ==y20  # Initial condition
        else:
            return m.dy2dt[t]==m.b3[t]*m.y1[t]
    m.cdy2dt=pe.Constraint(m.t,rule=cdy2)

    def b1m(m,t):
        return m.b1[t]==w1*((1-w2*(m.temp[t]-w3)**2)/(1-w2*(25-w3)**2))
    m.cb1=pe.Constraint(m.t,rule=b1m)

    def b2m(m,t):
        return m.b2[t]==w4*((1-w2*(m.temp[t]-w3)**2)/(1-w2*(25-w3)**2))
    m.cb2=pe.Constraint(m.t,rule=b2m)

    def b3m(m,t):
        return m.b3[t]==w5*((1-w2*(m.temp[t]-w6)**2)/(1-w2*(25-w6)**2))
    m.cb3=pe.Constraint(m.t,rule=b3m)



    #ENDPOINT CONSTRAINTS
    # def _end1(m,t):
    #     if t == m.t.last(): 
    #         return m.y1[t] ==y1f
    #     else:                            
    #         return pe.Constraint.Skip      
    # m.endpoint1=pe.Constraint(m.t,rule=_end1)

    # def _end2(m,t):
    #     if t == m.t.last(): 
    #         return m.y2[t] ==y2f
    #     else:                            
    #         return pe.Constraint.Skip      
    # m.endpoint2=pe.Constraint(m.t,rule=_end2)
    #OBJECTIVE FUNCTION

    # def _integralu(m,t):
    #     return m.temp[t]
    # m.integraltemp=dae.Integral(m.t,wrt=m.t,rule=_integralu)

    def _obj(m):
        return -m.y2[max(m.t)]#m.integraltemp
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        
    discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m,nfe=100,wrt=m.t,scheme='BACKWARD') 

    keep_constant=10

    def _Constant_control1(m,t):
        if (t!=m.t.first() and (m.t.ord(t)-1)%keep_constant!=0) or (t==m.t.last()):
            return m.temp[t] == m.temp[m.t.prev(t)]
        else:
            return pe.Constraint.Skip
    m.Constant_control1=pe.Constraint(m.t,rule=_Constant_control1,doc='Constant control action every keep_constant discrete points and the last one')

    return m

def test_process_one_time_step(inputval,y10,y20):

    m = pe.ConcreteModel(name='reactor_dynamics_one_time_Step')


    m.t=dae.ContinuousSet(bounds=(0,0.1),doc='Continuous time set [units of time]')


    #PARAMETERS

    w1=13.1
    w2=0.005
    w3=30
    w4=0.94
    w5=1.71
    w6=20
    # y10=0.03
    # y20=0
    # y1f=1
    # y2f=1


    #VARIABLES

    m.y1=pe.Var(m.t,within=pe.NonNegativeReals,initialize=0.5) #TODO: Here i dont need to define bounds (or that is what I think, but maybe if there are really some constraints over the states I can accout for them here to avoid exploring certain states) 
    m.y2=pe.Var(m.t,within=pe.NonNegativeReals,initialize=0.5) #TODO: 
    m.temp=pe.Var(m.t,within=pe.NonNegativeReals,bounds=(20,30))
    m.b1=pe.Var(m.t,within=pe.NonNegativeReals,initialize=1)
    m.b2=pe.Var(m.t,within=pe.NonNegativeReals,initialize=1)
    m.b3=pe.Var(m.t,within=pe.NonNegativeReals,initialize=1)

    m.dy1dt= dae.DerivativeVar(m.y1, withrespectto=m.t, doc='Derivative of y1')
    m.dy2dt= dae.DerivativeVar(m.y2, withrespectto=m.t,doc='Derivative of y2')

    #CONSTRAINTS
    def cdy1(m,t):
        if t == m.t.first(): 
            return m.y1[t] ==y10  # Initial condition
        else:
            return m.dy1dt[t]==m.b1[t]*m.y1[t]-((m.b1[t]/m.b2[t])*((m.y1[t])**2))
    m.cdy1dt=pe.Constraint(m.t,rule=cdy1)

    def cdy2(m,t):
        if t == m.t.first(): 
            return m.y2[t] ==y20  # Initial condition
        else:
            return m.dy2dt[t]==m.b3[t]*m.y1[t]
    m.cdy2dt=pe.Constraint(m.t,rule=cdy2)

    def b1m(m,t):
        return m.b1[t]==w1*((1-w2*(m.temp[t]-w3)**2)/(1-w2*(25-w3)**2))
    m.cb1=pe.Constraint(m.t,rule=b1m)

    def b2m(m,t):
        return m.b2[t]==w4*((1-w2*(m.temp[t]-w3)**2)/(1-w2*(25-w3)**2))
    m.cb2=pe.Constraint(m.t,rule=b2m)

    def b3m(m,t):
        return m.b3[t]==w5*((1-w2*(m.temp[t]-w6)**2)/(1-w2*(25-w6)**2))
    m.cb3=pe.Constraint(m.t,rule=b3m)



    #ENDPOINT CONSTRAINTS
    # def _end1(m,t):
    #     if t == m.t.last(): 
    #         return m.y1[t] ==y1f
    #     else:                            
    #         return pe.Constraint.Skip      
    # m.endpoint1=pe.Constraint(m.t,rule=_end1)

    # def _end2(m,t):
    #     if t == m.t.last(): 
    #         return m.y2[t] ==y2f
    #     else:                            
    #         return pe.Constraint.Skip      
    # m.endpoint2=pe.Constraint(m.t,rule=_end2)
    #OBJECTIVE FUNCTION

    # def _integralu(m,t):
    #     return m.temp[t]
    # m.integraltemp=dae.Integral(m.t,wrt=m.t,rule=_integralu)

    def _obj(m):
        return 1#-m.y2[max(m.t)]#m.integraltemp
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        
    discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m,nfe=10,wrt=m.t,scheme='BACKWARD') 

    # keep_constant=10

    # def _Constant_control1(m,t):
    #     if (t!=m.t.first() and (m.t.ord(t)-1)%keep_constant!=0) or (t==m.t.last()):
    #         return m.temp[t] == m.temp[m.t.prev(t)]
    #     else:
    #         return pe.Constraint.Skip
    # m.Constant_control1=pe.Constraint(m.t,rule=_Constant_control1,doc='Constant control action every keep_constant discrete points and the last one')
    for t in m.t:
        m.temp[t].fix(inputval)
    return m

def test_process_mip(neighbors_k,alpha_k_state,rho_k_state,theta_k_state,n_states,n_inputs,bounds_states,bounds_inputs,initial_state):
    m = pe.ConcreteModel(name='mip representation of dynamic process')
    m.k=pe.Set(initialize=neighbors_k.keys(),doc='Set of regions, each one having a linearization')
    m.t=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set [units of time]')
    m.s=pe.RangeSet(0,n_states-1,doc='states') #starts from 0
    m.i=pe.RangeSet(0,n_inputs-1,doc='inputs') #starts from 0
    m.s_alias=pe.SetOf(m.s)

    number_finite=100
    keep_constant=10 #10 elements cosntant
    discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m,nfe=round(number_finite/keep_constant),wrt=m.t,scheme='BACKWARD') 

    # m.i=pe.RangeSet(0,m.t.__len__()-1,1)

    m.alpha_param=pe.Param(m.k,m.s,initialize=alpha_k_state)
    def _rule_rho_param(m,k,state,inp):
        return rho_k_state[(k,state)][inp]
    m.rho_param=pe.Param(m.k,m.s,m.i,initialize=_rule_rho_param)

    def _rule_theta_param(m,k,state,alias):
        return theta_k_state[(k,state)][alias]
    m.theta_param=pe.Param(m.k,m.s,m.s_alias,initialize=_rule_theta_param)


    def _rule_lower_input(m,k,inp):
        min_bound_sampling=min([min([b[inp] for b in neighbors_k[kprima]]) for kprima in neighbors_k.keys()])
        bound_to_return=min([b[inp] for b in neighbors_k[k]])

        if min_bound_sampling==bound_to_return:
            return bounds_inputs[inp][0]
        else:
            return bound_to_return
    m.lower_input=pe.Param(m.k,m.i,initialize=_rule_lower_input)

    def _rule_upper_input(m,k,inp):
        max_bound_sampling=max([max([b[inp] for b in neighbors_k[kprima]]) for kprima in neighbors_k.keys()])
        bound_to_return=max([b[inp] for b in neighbors_k[k]])

        if max_bound_sampling==bound_to_return:
            return bounds_inputs[inp][1]
        else:
            return bound_to_return
    m.upper_input=pe.Param(m.k,m.i,initialize=_rule_upper_input)



    def _rule_lower_state(m,k,state):
        loc=n_inputs+state
        min_bound_sampling=min([min([b[loc] for b in neighbors_k[kprima]]) for kprima in neighbors_k.keys()])
        bound_to_return=min([b[loc] for b in neighbors_k[k]])
        if min_bound_sampling==bound_to_return:
            return bounds_states[state][0]
        else:
            return bound_to_return
    m.lower_state=pe.Param(m.k,m.s,initialize=_rule_lower_state)

    def _rule_upper_state(m,k,state):
        loc=n_inputs+state
        max_bound_sampling=max([max([b[loc] for b in neighbors_k[kprima]]) for kprima in neighbors_k.keys()])
        bound_to_return=max([b[loc] for b in neighbors_k[k]])
        if max_bound_sampling==bound_to_return:
            return bounds_states[state][1]
        else:
            return bound_to_return
    m.upper_state=pe.Param(m.k,m.s,initialize=_rule_upper_state)





    m.Y=pe.Var(m.k,m.t,within=pe.Binary,initialize=1)
    m.Xk=pe.Var(m.k,m.t,m.s,within=pe.Reals,initialize=0)
    m.Uk=pe.Var(m.k,m.t,m.i,within=pe.Reals,initialize=0)
    m.alpha=pe.Var(m.k,m.t,m.s,within=pe.Reals,initialize=0)
    m.X=pe.Var(m.t,m.s,within=pe.Reals,initialize=0)
    m.U=pe.Var(m.t,m.i,within=pe.Reals,initialize=0)



    #  CONSTRAINTS
    # ONLY ONE Y IS SELECTED
    def _oneY(m,t):
        if t==m.t.first():
            return pe.Constraint.Skip
        else:
            return sum(m.Y[k,t] for k in m.k)== 1 
    m.oneY=pe.Constraint(m.t,rule=_oneY)

    # Constraint to update dynamic process

    def _xupdate(m,s,t):
        if t==m.t.first():
            return pe.Constraint.Skip#m.Xk[k,t,s]==m.Y[k,t]*initial_state[s]
        else:
            return m.X[t,s]==sum(m.alpha[k,t,s]+sum(m.rho_param[k,s,i]*m.Uk[k,m.t.prev(t),i] for i in m.i)+sum(m.theta_param[k,s,sprim]*m.Xk[k,m.t.prev(t),sprim] for sprim in m.s_alias) for k in m.k)
    m.xupdate=pe.Constraint(m.s,m.t,rule=_xupdate)



    def _defalpha(m,s,k,t):
        if t==m.t.first():
            return pe.Constraint.Skip
        else:
            return m.alpha[k,t,s]==m.Y[k,t]*m.alpha_param[k,s]
    m.defalpha=pe.Constraint(m.s,m.k,m.t, rule=_defalpha)




    # def _u_lower(m,i,k,t):
    #     if t==m.t.last():
    #         return pe.Constraint.Skip
    #     else:
    #         return m.Y[k,t]*m.lower_input[k,i]<=m.Uk[k,t,i]      
    # m.u_lower=pe.Constraint(m.i,m.k,m.t,rule=_u_lower)

    # def _u_upper(m,i,k,t):
    #     if t==m.t.last():
    #         return pe.Constraint.Skip
    #     else:
    #         return m.Uk[k,t,i]<=m.Y[k,t]*m.upper_input[k,i]      
    # m.u_upper=pe.Constraint(m.i,m.k,m.t,rule=_u_upper)

    # def _x_lower(m,s,k,t):
    #     if t==m.t.last():
    #         return pe.Constraint.Skip
    #     else:
    #         return m.Y[k,t]*m.lower_state[k,s]<=m.Xk[k,t,s]      
    # m.x_lower=pe.Constraint(m.s,m.k,m.t,rule=_x_lower)

    # def _x_upper(m,s,k,t):
    #     if t==m.t.last():
    #         return pe.Constraint.Skip
    #     else:
    #         return m.Xk[k,t,s]<=m.Y[k,t]*m.upper_state[k,s]      
    # m.x_upper=pe.Constraint(m.s,m.k,m.t,rule=_x_upper)


    def _u_lower(m,i,k,t):
        if t==m.t.first():
            return pe.Constraint.Skip
        else:
            return m.Y[k,t]*m.lower_input[k,i]<=m.Uk[k,m.t.prev(t),i]      
    m.u_lower=pe.Constraint(m.i,m.k,m.t,rule=_u_lower)

    def _u_upper(m,i,k,t):
        if t==m.t.first():
            return pe.Constraint.Skip
        else:
            return m.Uk[k,m.t.prev(t),i]<=m.Y[k,t]*m.upper_input[k,i]      
    m.u_upper=pe.Constraint(m.i,m.k,m.t,rule=_u_upper)

    def _x_lower(m,s,k,t):
        if t==m.t.first():
            return pe.Constraint.Skip
        else:
            return m.Y[k,t]*m.lower_state[k,s]<=m.Xk[k,m.t.prev(t),s]      
    m.x_lower=pe.Constraint(m.s,m.k,m.t,rule=_x_lower)

    def _x_upper(m,s,k,t):
        if t==m.t.first():
            return pe.Constraint.Skip
        else:
            return m.Xk[k,m.t.prev(t),s]<=m.Y[k,t]*m.upper_state[k,s]      
    m.x_upper=pe.Constraint(m.s,m.k,m.t,rule=_x_upper)

    # Definition of X
    def _defXINIT(m,s,t):
        if t==m.t.first():
            return m.X[t,s]==initial_state[s]
        else:
            return pe.Constraint.Skip
    m.defXINIT=pe.Constraint(m.s,m.t,rule=_defXINIT)

    def _defX(m,s,t):
        return m.X[t,s]==sum(m.Xk[k,t,s] for k in m.k)
    m.defX=pe.Constraint(m.s,m.t,rule=_defX)
    #Definition of U
    def _defU(m,i,t):
        # if t==m.t.first():
        #     return pe.Constraint.Skip
        # else:
        return m.U[t,i]==sum(m.Uk[k,t,i] for k in m.k)       
    m.defU=pe.Constraint(m.i,m.t,rule=_defU)

    def _obj(m):
        return -m.X[max(m.t),1]#m.integraltemp
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  

    return m

def test_process_lp():
    m = pe.ConcreteModel(name='reactor_dynamics_one_time_Step')


    m.t=dae.ContinuousSet(bounds=(0,1),doc='Continuous time set [units of time]')


    #PARAMETERS

    w1=13.1
    w2=0.005
    w3=30
    w4=0.94
    w5=1.71
    w6=20
    y10=0.03
    y20=0
    
    ul=29.33
    y1l=y10
    y2l=y20

    b1=w1*((1-w2*(ul-w3)**2)/(1-w2*(25-w3)**2))
    b2=w4*((1-w2*(ul-w3)**2)/(1-w2*(25-w3)**2))
    b3=w5*((1-w2*(ul-w6)**2)/(1-w2*(25-w6)**2))

    df1dy1=b1-(b1/b2)*2*y1l

    df1db1=y1l-(y1l**2)/b2
    df1db2=(b1*(y1l**2))/(b2**2)
    db1dtemp=w1*((-w2*(ul-w3)*2)/(1-w2*(25-w3)**2))
    db2dtemp=w4*((-w2*(ul-w3)*2)/(1-w2*(25-w3)**2))
    db3dtemp=w5*((-w2*(ul-w6)*2)/(1-w2*(25-w6)**2))

    df2dy1=b3

    df1dtemp=df1db1*db1dtemp+df1db2*db2dtemp
    df2dtemp=y1l*db3dtemp

    f1l=b1*y1l-((b1/b2)*((y1l)**2))
    f2l=b3*y1l


    #VARIABLES

    m.y1=pe.Var(m.t,within=pe.NonNegativeReals,initialize=0.5)#,bounds=(0,1.075)) #TODO:  how to define these bounds (here I defined them maximizing the value of the state at the end, with negligible sampling time)
    m.y2=pe.Var(m.t,within=pe.NonNegativeReals,initialize=0.5)#,bounds=(0,1.243)) #TODO: how to define these bounds (here I defined them maximizing the value of the state at the end, with negligible sampling time)
    m.temp=pe.Var(m.t,within=pe.NonNegativeReals,bounds=(20,29.33))


    m.dy1dt= dae.DerivativeVar(m.y1, withrespectto=m.t, doc='Derivative of y1')
    m.dy2dt= dae.DerivativeVar(m.y2, withrespectto=m.t,doc='Derivative of y2')

    #CONSTRAINTS
    def cdy1(m,t):
        if t == m.t.first(): 
            return m.y1[t] ==y10  # Initial condition
        else:
            return m.dy1dt[t]==f1l+df1dy1*(m.y1[t]-y1l)+df1dtemp*(m.temp[t]-ul)        
    m.cdy1dt=pe.Constraint(m.t,rule=cdy1)

    def cdy2(m,t):
        if t == m.t.first(): 
            return m.y2[t] ==y20  # Initial condition
        else:
            return m.dy2dt[t]==f2l+df2dy1*(m.y1[t]-y1l)+df2dtemp*(m.temp[t]-ul)                      
    m.cdy2dt=pe.Constraint(m.t,rule=cdy2)





    #ENDPOINT CONSTRAINTS
    # def _end1(m,t):
    #     if t == m.t.last(): 
    #         return m.y1[t] ==y1f
    #     else:                            
    #         return pe.Constraint.Skip      
    # m.endpoint1=pe.Constraint(m.t,rule=_end1)

    # def _end2(m,t):
    #     if t == m.t.last(): 
    #         return m.y2[t] ==y2f
    #     else:                            
    #         return pe.Constraint.Skip      
    # m.endpoint2=pe.Constraint(m.t,rule=_end2)
    #OBJECTIVE FUNCTION

    # def _integralu(m,t):
    #     return m.temp[t]
    # m.integraltemp=dae.Integral(m.t,wrt=m.t,rule=_integralu)

    def _obj(m):
        return -m.y2[max(m.t)]#m.integraltemp
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)  
        
    discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m,nfe=1000,wrt=m.t,scheme='BACKWARD') 

    keep_constant=10

    def _Constant_control1(m,t):
        if (t!=m.t.first() and (m.t.ord(t)-1)%keep_constant!=0) or (t==m.t.last()):
            return m.temp[t] == m.temp[m.t.prev(t)]
        else:
            return pe.Constraint.Skip
    m.Constant_control1=pe.Constraint(m.t,rule=_Constant_control1,doc='Constant control action every keep_constant discrete points and the last one')
    return m


def mystep(x,y, ax=None, **kwargs):
    X = np.c_[x[:-1],x[1:],x[1:]]
    Y = np.c_[y[:-1],y[:-1],np.zeros_like(x[:-1])*np.nan]
    if not ax: ax=plt.gca()
    return ax.plot(X.flatten(), Y.flatten(), **kwargs)

if __name__ == "__main__":

    ##--------TEST 1: ORIGINAL DYNAMIC MODEL---------------------------------------
    # m=test_process()

    # opt = SolverFactory('gams', solver='conopt')

    # m.results = opt.solve(m, tee=False,skip_trivial_constraints=True)

    # y1=[pe.value(m.y1[i]) for i in m.t]
    # y2=[pe.value(m.y2[i]) for i in m.t]
    # T=[pe.value(m.temp[i]) for i in m.t]


    # t=list(m.t)

    # plt.figure(1)
    # plt.plot(t,y1,'b',t,y2,'g')
    # plt.xlabel('time')
    # plt.ylabel('Concentration')
    # plt.title('Production Profile')
    # plt.legend(['Cell_Mass','Penicillin'],loc=0)
    # plt.grid(True)
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()

    # plt.figure(2)
    # # plt.plot(t,T,'r')
    # mystep(t,T,color="red")
    # plt.xlabel('time')
    # plt.ylabel('Value')
    # plt.title('Temperature Profile')
    # plt.legend(['Temperature'],loc=0)
    # plt.grid(True)
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()


    ###--------TEST 2: ONE TIME STEP---------------------------------------
    # inputval=22
    # inity1=0.5
    # inity2=0.6
    # m=test_process_one_time_step(inputval,inity1,inity2)

    # opt = SolverFactory('gams', solver='conopt')
    # m.results = opt.solve(m, tee=False,skip_trivial_constraints=True)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     outstatus='Infeasible'
    # else:
    #     outstatus='Optimal'

    # outy1=pe.value(m.y1[m.t.last()])
    # outy2=pe.value(m.y2[m.t.last()])
    # print('output: ','y1=',outy1,'y2=',outy2,'output status=',outstatus)
    # # y1=[pe.value(m.y1[i]) for i in m.t]
    # # y2=[pe.value(m.y2[i]) for i in m.t]
    # # T=[pe.value(m.temp[i]) for i in m.t]

    # # t=list(m.t)

    # # plt.figure(1)
    # # plt.plot(t,y1,'b',t,y2,'g')
    # # plt.xlabel('time')
    # # plt.ylabel('Concentration')
    # # plt.title('Production Profile')
    # # plt.legend(['Cell_Mass','Penicillin'],loc=0)
    # # plt.grid(True)
    # # plt.show()
    # # plt.clf()
    # # plt.cla()
    # # plt.close()

    # # plt.figure(2)
    # # plt.plot(t,T,'r')
    # # plt.xlabel('time')
    # # plt.ylabel('Value')
    # # plt.title('Temperature Profile')
    # # plt.legend(['Temperature'],loc=0)
    # # plt.grid(True)
    # # plt.show()
    # # plt.clf()
    # # plt.cla()
    # # plt.close()


    ###--------TEST 3: SAMPLING LOOP---------------------------------------
    # input_val_range=pe.RangeSet(20,30,1) # 5, 2, 1
    # initial_y1_range=pe.RangeSet(0,1.075,0.1)#0.5, 0.2, 0.1
    # initial_y2_range=pe.RangeSet(0,1.243,0.1)#0.5, 0.2, 0.1
    # dictvals={}
    # for inp in input_val_range:
    #     for y10 in initial_y1_range:
    #         for y20 in initial_y2_range:

    #             inputval=inp
    #             inity1=y10
    #             inity2=y20
    #             m=test_process_one_time_step(inputval,inity1,inity2)
    #             opt = SolverFactory('gams', solver='conopt')
    #             m.results = opt.solve(m, tee=False,skip_trivial_constraints=True)
    #             if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #                 outstatus='Infeasible'
    #             else:
    #                 outstatus='Optimal'
    #             outy1=pe.value(m.y1[m.t.last()])
    #             outy2=pe.value(m.y2[m.t.last()])
    #             dictvals[(inp,y10,y20)]=[outy1,outy2,outstatus]
    #             print('Input: ','u1=',round(inp,4),'y10=',round(y10,4),'y20=',round(y20,4),'      |     output: ','y1=',round(outy1,4),'y2=',round(outy2,4),'output status=',outstatus)        


    # a_file = open("sample_dict_larger_larger.pkl", "wb")
    # pickle.dump(dictvals, a_file)
    # a_file.close()


        ###--------TEST 4: GENERATION OF PLANES---------------------------------------
#TODO: A STEP IS MISSING WHERE I DELETE FROM DICTVALS THOSE SIMULATIONS THAT WERE INFEASIBLE. IN THIS CASE EVERYTHING IS FEASIBLE, SO I DO NOT HAVE TO DO THAT!!!!
#TODO: HOW TO CONSIDER SUROUNDINGS??? FOR THE MOMENT I AM NOT CONSIDERING THE BOUNDS OF INPUTS AND STATES IN MY CALCULATIONS. 

    # f_name="sample_dict_larger_larger"
    # a_file = open(f_name+".pkl", "rb")
    # dictvals = pickle.load(a_file)   
    # print(dictvals)    

    # #TODO: THINK HOW TO INCLUDE FRONTERA
    # new_input_range=pe.RangeSet(20+1/2,30-1/2,1) # 5, 2, 1 #TODO: GENERALIZE
    # new_y1_range=pe.RangeSet(0+0.1/2,1.075-0.1/2,0.1)#0.5, 0.2, 0.1 #TODO: GENERALIZE
    # new_y2_range=pe.RangeSet(0+0.1/2,1.243-0.1/2,0.1)#0.5, 0.2, 0.1 #TODO: GENERALIZE


    # k=0
    # num_states=2# Number of states
    # num_inputs=1#number of inputs

    # #Information that needs to be saved
    # central_points_k={}
    # neighbors_k={}
    # alpha_k_state={}
    # rho_k_state={}
    # theta_k_state={}

    # for inp in new_input_range:
    #     for y10 in new_y1_range:
    #         for y20 in new_y2_range:
    #             k=k+1
    #             central_point=[inp,y10,y20]
    #             central_points_k[k]=central_point

    #             neighbors=[]

    #             #TODO: GENERALIZE
    #             central_up=[inp+1/2,y10+0.1/2,y20+0.1/2]
    #             central_lo=[inp-1/2,y10-0.1/2,y20-0.1/2]
    #             neigh_points=[(x,y,z) for x in [central_lo[0],central_up[0]] for y in [central_lo[1],central_up[1]] for z in [central_lo[2],central_up[2]]]
    #             # print(neigh_points)
    #             for state in range(num_states):
                    
    #                 # neighbors={j:dictvals[j][state] for j in dictvals.keys() if np.linalg.norm(np.asarray((inp,y10,y20))-np.asarray(j))<=min([np.linalg.norm(np.asarray((inp,y10,y20))-np.asarray(k)) for k in dictvals.keys() ])}
    #                 neighbors={j:dictvals[j][state] for j in dictvals.keys() if any([sum(abs(j[k]-hola[k]) for k in range(len(central_point)))<=1e-4 for hola in neigh_points])}
    #                 if len(neighbors.keys())!=8:
    #                     print('error!!!')
    #                 # print('central point:',central_point)
    #                 # print(neighbors)
    #                 coef=convex_clousure(neighbors,central_point)  #TODO: define planes using more data, minimizatio of square error with bounds to guarantee that  # In general, other strategies can be used, here I am using the convex clousure strategy
    #                 # print(coef)
    #                 alpha_k_state[(k,state)]=coef[-1]
    #                 rho_k_state[(k,state)]=coef[0:num_inputs]
    #                 theta_k_state[(k,state)]=coef[num_inputs:num_states+1]
    #                 # neighbors_approx={j:coef[-1]+coef[0]*j[0]+coef[1]*j[1]+coef[2]*j[2] for j in neighbors.keys()}
    #                 # print(neighbors_approx)
    #                 # print(alpha_k_state[(k,state)],rho_k_state[(k,state)],theta_k_state[(k,state)])

    #             neighbors_k[k]=[j for j in neighbors.keys()]
    
    # print('central poits:', central_points_k)
    # print('neighbors:',neighbors_k)


    # generated_info=[central_points_k,neighbors_k,alpha_k_state,rho_k_state,theta_k_state]


    # a_file = open("sample_dict_info_larger_larger.pkl", "wb")
    # pickle.dump(generated_info, a_file)
    # a_file.close()

        ###--------TEST 5: GENERATION AND SOLUTION OF MIP---------------------------------------
    start=time.time()
    m=test_process()

    opt = SolverFactory('gams', solver='conopt')

    m.results = opt.solve(m, tee=False,skip_trivial_constraints=True)
    end=time.time()

    print('CPU time NLP=',end-start)
    y1_base=[pe.value(m.y1[i]) for i in m.t]
    y2_base=[pe.value(m.y2[i]) for i in m.t]
    T_base=[pe.value(m.temp[i]) for i in m.t]


    t_base=list(m.t)

    
    start=time.time()
    m=test_process_lp()

    opt = SolverFactory('gams', solver='cplex')

    m.results = opt.solve(m, tee=True,skip_trivial_constraints=True)
    end=time.time()

    print('CPU time LP=',end-start)
    y1_lp=[pe.value(m.y1[i]) for i in m.t]
    y2_lp=[pe.value(m.y2[i]) for i in m.t]
    T_lp=[pe.value(m.temp[i]) for i in m.t]


    t_lp=list(m.t)

    # plt.figure(1)
    # plt.plot(t_base,y1_base,'b--',t_base,y2_base,'g--')
    # plt.xlabel('time')
    # plt.ylabel('Concentration')
    # plt.title('Production Profile')
    # plt.legend(['Cell_Mass (NLP)','Penicillin (NLP)'],loc=0)
    # plt.grid(False)
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()

    # plt.figure(2)
    # plt.plot(t_base,T_base,'r--')
    # # mystep(t_base,T_base,color="red",marker='.')
    # plt.xlabel('time')
    # plt.ylabel('Value')
    # plt.title('Temperature Profile')
    # plt.legend(['Temperature (NLP)'],loc=0)
    # plt.grid(False)
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()


    f_name="sample_dict"
    a_file = open(f_name+".pkl", "rb")
    dictvals = pickle.load(a_file) 

    print('number points sampled: ',len(dictvals.keys())) 

    f_name="sample_dict_info"
    a_file = open(f_name+".pkl", "rb")
    generated_info = pickle.load(a_file)   
    central_points_k=generated_info[0]
    neighbors_k=generated_info[1]
    alpha_k_state=generated_info[2]
    rho_k_state=generated_info[3]
    theta_k_state=generated_info[4]
    # print('central poits:', central_points_k)
    # print('neighbors:',neighbors_k) 
    # print('alpha:',alpha_k_state) 
    # print('rho:',rho_k_state) 
    # print('theta:',theta_k_state) 

    num_states=2
    num_inputs=1
    bounds_states=[(0,1.075),(0,1.243)]
    bounds_inputs=[(20,30)]
    initial_state=[0.03,0]
    start=time.time()
    m=test_process_mip(neighbors_k,alpha_k_state,rho_k_state,theta_k_state,num_states,num_inputs,bounds_states,bounds_inputs,initial_state)


    opt = SolverFactory('gams', solver='cplex')
    m.results = opt.solve(m, tee=False,skip_trivial_constraints=True)
    # m.pprint()
    end=time.time()
    print('CPU time MIP=',end-start)
    textbuffer = io.StringIO()


    for v in m.component_objects(pe.Var, descend_into=True):
        v.pprint(textbuffer)
        textbuffer.write('\n')
    textbuffer.write('\n Objective: \n') 
    textbuffer.write(str(pe.value(m.obj)))    
    with open('results_penicilin.txt', 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())

    num_continuous=0
    num_binary=0
    num_constraints=0
    for v in m.component_data_objects(pe.Var, descend_into=True):
        if not v.is_continuous():
            num_binary=num_binary+1
        else:
            num_continuous=num_continuous+1
    for v in m.component_data_objects(pe.Constraint, descend_into=True):
            num_constraints=    num_constraints+1
    print('num_continuous_Vars=',num_continuous)
    print('num_binary_Vars=',num_binary)
    print('num_constraints=',num_constraints)

    y1=[pe.value(m.X[i,0]) for i in m.t]
    y2=[pe.value(m.X[i,1]) for i in m.t]
    T=[pe.value(m.U[i,0]) for i in m.t]
    T[-1]=T[-2]

    t=list(m.t)

    plt.figure(1)
    plt.plot(t,y1,'b',t,y2,'g')
    plt.xlabel('time')
    plt.ylabel('Concentration')
    plt.title('Production Profile')
    # plt.legend(['Cell_Mass (MIP)','Penicillin (MIP)'],loc=0)


    plt.plot(t_base,y1_base,'b--',t_base,y2_base,'g--')
    plt.xlabel('time')
    plt.ylabel('Concentration')
    plt.title('Production Profile')

    plt.plot(t_lp,y1_lp,'b.',t_lp,y2_lp,'g.')
    plt.xlabel('time')
    plt.ylabel('Concentration')
    plt.title('Production Profile')


    plt.legend(['Cell_Mass (MIP)','Penicillin (MIP)','Cell_Mass (NLP)','Penicillin (NLP)','Cell_Mass (LP)','Penicillin (LP)'],loc=0)
    plt.ylim([0, 1.3])
    plt.grid(False)  
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(2)
    
    # plt.plot(t,T,'r')
    mystep(t,T,color="red")
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.title('Temperature Profile')
    # plt.legend(['Temperature (MIP)'],loc=0)

    plt.plot(t_base,T_base,'r--')
    # mystep(t_base,T_base,color="red",marker='.')
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.title('Temperature Profile')

    plt.plot(t_lp,T_lp,'r.')
    # mystep(t_base,T_base,color="red",marker='.')
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.title('Temperature Profile')


    plt.legend(['Temperature (MIP)','Temperature (NLP)','Temperature (LP)'],loc=0)    


    plt.grid(False)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
