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
    
    #TODO: Update these values below
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [units of time]')
    m.lastT=pe.Param(initialize=100,doc='last discrete time value in the scheduling time grid')

    # #Reactor I (large)
    # m.v_R_max = pe.Param(initialize=1.5,doc='Maximum capacity of the reactor [m^3]')
    # m.v_J=pe.Param(initialize=0.5,doc='Volume of the Jacket [m^3]')
    # m.rho_J=pe.Param(initialize=1e+3,doc='Density of the jacket [kg/m^3]')
    # m.c_J 


    # #Reactor II (small)


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

    #-----------Sets that depend on parameters--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense
    #TODO: chek units of time, are they consistent? should I use hours?


    m.N = dae.ContinuousSet(bounds=(0, max([m.tau_p[I,J] for I in m.I_reactions for J in m.J_reactors])),doc='Cuntinuous time set for tasks-units that consider dynamics')
    def _I_J(m):
        return ((I,J) for I in m.I_reactions for J in m.J_reactors) 
    m.F=pe.Set(m.N,dimen=2,initialize=_I_J)

    # discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer = pe.TransformationFactory('dae.collocation')
    # discretizer.apply_to(m, nfe=10, wrt=m.N, scheme='BACKWARD')
    discretizer.apply_to(m, nfe=10, ncp=3, wrt=m.N, scheme='LAGRANGE-RADAU')

    # #TODO: check if I can do this here, or if I have to do it after discretization
    # def _I_J_N(m):
    #     return ((I,J,N) for I in m.I_reactions for J in m.J_reactors for N in m.N if N<=m.tau_p[I,J])    
    # m.I_J_N=pe.Set(dimen=3,initialize=_I_J_N,doc='task-unit-time mapping, for task-units that consider dynamics')

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
    return m
if __name__ == "__main__":
    m=reaction_1()
    m.pprint()