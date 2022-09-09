from __future__ import division
import pyomo.environ as pe
import pyomo.dae as dae
import math

#IDEAS
#0) If the ammount processed whenever a task repeats is the same, processing times are the same and, etc, then I can consider dynamic model once 
#1) can I make things dimmensionless and independent on the ammount processes for every batch? that way probably I can consider one dynamic model to represent multiple operations

def reaction_1():
    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='reaction_1')
    # ------------scalars    ------------------------------------------------
    
    
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=100,doc='last discrete time value in the scheduling time grid') #TODO: Update as required


    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)

    # -----------parameters--------------------------------------------------
    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['R1','R_large']=1
    _tau_p['R1','R_small']=1

    _tau_p['R2','R_large']=1.5
    _tau_p['R2','R_small']=1.5

    _tau_p['R3','R_large']=1
    _tau_p['R3','R_small']=1

    _tau_p['Sep','Sep']=3

    _tau_p['Pack1','Pack']=1.5
    _tau_p['Pack2','Pack']=1.5

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")


    #Parameters of reactor units
    m.v_R_max = pe.Param(m.J_reactors,initialize={'R_large':1.5,'R_small':1},doc='Maximum capacity of the reactor [m^3]')
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
    m.z=pe.Param(m.I_reactions,initialize={'R_1':1e+7,'R_2':1.2e+3,'R_3':2e+4},doc='Pre-exponential factors [m^3/kmol h]')
    m.er=pe.Param(m.I_reactions,initialize={'R_1':5e+3,'R_2':2e+3,'R_3':3e+3},doc='Normalized activation energy [K]')
    m.delta_h=pe.Param(m.I_reactions,initialize={'R_1':1e+3,'R_2':-2e+3,'R_3':5e+3},doc='Heat of reaction [kJ/kmol]')
    m.rho_R=pe.Param(m.I_reactions,initialize={'R_1':1e+3,'R_2':1e+3,'R_3':1e+3},doc='Density of reaction mixture [kg/m^3]')
    m.c_R=pe.Param(m.I_reactions,initialize={'R_1':3,'R_2':3.5,'R_3':4},doc='Heat capacity of reaction mixture [kJ/kg K]')
    
    #-----------Sets that depend on parameters--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense  
    m.N={}
    for I in m.I_reactions:
        for J in m.J_reactors:
            m.N[(I,J)]=dae.ContinuousSet(bounds=(0,m.tau_p[I,J])) #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_(%s,%s)' %(I,J),m.N[(I,J)]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct

    #----------Parameters defined over sets that depend on parameters---------


    # # -----------variables ---------------------------------------------------
    # m. = pe.Var(m., initialize=,
    #           within=pe.NonNegativeReals, bounds=(,))
    # # ----------Constraints--------------------------------------------------

 
    # m. = Constraint(m, rule=)


    # m.obj = pe.Objective()

    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')

    # # -------Discretization---------------------------------------------------
    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible
    for continuous_sets in m.N:
        discretizer.apply_to(m, nfe=3, ncp=3, wrt=m.N[continuous_sets], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
    return m
if __name__ == "__main__":
    m=reaction_1()
    m.pprint()