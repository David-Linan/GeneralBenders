import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory
import matplotlib.pyplot as plt



def build_hydrolisis() -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
    """
    Function that 

    Args:

    Returns:

    """

    #designed for around 35% of dry matter
    #hydrolisis time=140 hours
    #hydrolysis model: conventional CSTRs in series. reactor is modelled as a plug flow reactor.... why
    # all reactors in the series have temperature and ph controllers
    # concentration of acetic acid is wath affects pH. NaOH is used for pH adjustment
    # enzimes added from storage tank.

    # Fff=inflow of pretreated fiblers, initial composition C0 (measured with online NIR, measurements every 5 minutes)
    # FB= known flow of NaOH, CB NaOH concentration are known
    # FE, CB: flow of enzimes
    # All inflows measured with sampling period of 2 s
    # water added with flowrate FW
    # Output control: mass measurement and manipulate outflow of fiber mash (FFM), to keep connstant hold up. Outflow concentration C, sampled every 6 h, analized with HPLC.
    # output pH measured online with a sampling period of 10 seconds.

    # reactor tank exposed to atmospheric pressure

    # ALL FLOWS ARE MEAASURED!!!!!!


    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='reactor_model')
    # ------------scalars    ------------------------------------------------
    m.final_time = pe.Param(initialize=170*(3600),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    m.tR = pe.Param(initialize=7.8*(3600),doc='retention time [s]')  # NOTE:  based on "the reactor retention time in this simulation scenario is 7.8h..."
    m.FFM= pe.Param(initialize=1.16,doc='fiber mash outflow [kg/s]') #NOTE: based on table B.4
    m.vx = pe.Param(initialize=m.LR/m.tR,doc='horizontal speed [m/s]') #NOTE: according to equation B5 is considered constant along the length of the reactor

    m.MFM = pe.Param(initialize=m.tR*m.FFM,doc='total mass of fiber mash inside the tank [kg]')  # TODO: according to the article, vx is assumed constant, which means tR constant, which means ration between MFM=FFM is constant. To simplify, I assume that both MFM and FFM are constant
    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, m.final_time))
    # spatial coordinate index (x)
    m.x = dae.ContinuousSet(bounds=(0, m.LR))
    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment.
    m.j = pe.Set(initialize=['CS', 'XS', 'LS', 'C','G', 'X', 'F', 'E']) 






    # ---------parameters----------------------------------------------------
    _C0={}
    _C0['CS']=1
    _C0['XS']=1
    _C0['LS']=1
    _C0['C']=1
    _C0['G']=1
    _C0['X']=1
    _C0['F']=1
    _C0['E']=1
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the differnt species []')  # TODO: check units and declare correct values


    _CFEED={}
    _CFEED['CS']=0.5
    _CFEED['XS']=0.5
    _CFEED['LS']=0.5
    _CFEED['C']=0.5
    _CFEED['G']=0.5
    _CFEED['X']=0.5
    _CFEED['F']=0.5
    _CFEED['E']=0.5
    m.CFEED=pe.Param(m.j,initialize=_CFEED,doc='Feed concentrations []') # TODO: check units and declare correct values


    # ---------variables-----------------------------------------------------
    m.C=pe.Var(m.t, m.x, m.j, initialize=1,within=pe.NonNegativeReals) #bounds=(0, 10000))
    m.R = pe.Var(m.t, m.x, m.j, initialize=1, within=pe.Reals)
    m.D = pe.Var(m.t,m.x, initialize=1, within=pe.NonNegativeReals)



    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dCdx=dae.DerivativeVar(m.C,wrt=m.x)
    m.dC2dx2=dae.DerivativeVar(m.C,wrt=(m.x,m.x))
    m.dDdx=dae.DerivativeVar(m.D,wrt=m.x)
    #--------constraitns----------------------------------------------------


    def _partialDiff(m,t,x,j):
        if t==m.t.first() and x>m.x.first() and x<m.x.last(): #Initial condition
            return m.C[t,x,j] == m.C0[j]
        elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
            return m.C[t,x,j] == m.CFEED[j]
        elif x==m.x.last():  #boundary condition 2 #TODO: CHECK IF THIS IS CORRECT
            return m.dCdx[t,x,j] == 0
        else:  # Partial differential equation
            return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] + m.D[t,x]*m.dC2dx2[t,x,j]+m.dCdx[t,x,j]*m.dDdx[t,x]+m.R[t,x,j]
    m.partialDiff=pe.Constraint(m.t,m.x,m.j,rule=_partialDiff)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)
    
    
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=10, wrt=m.x, scheme='BACKWARD')


    discretizer_x = pe.TransformationFactory('dae.collocation')
    discretizer_x.apply_to(m, nfe=10, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')


    discretizer_t = pe.TransformationFactory('dae.collocation')
    discretizer_t.apply_to(m, nfe=10, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')

    for t in m.t:
        for x in m.x:
            m.D[t,x].fix(1e-3)
            for j in m.j:
                m.R[t,x,j].fix(0)

    return m

if __name__ == '__main__':

    m=build_hydrolisis()
    opt1 = SolverFactory('gams')
    results = opt1.solve(m, solver='conopt', tee=True)


    time=[]
    space=[]
    vec_CS={}



    for x in m.x:
        space.append(x)

    for t in m.t:
        time.append(t)
        vec_CS[t]=[]
        for x in m.x:
            vec_CS[t].append(m.C[t,x,'CS'].value)


    for t in m.t:
        plt.plot(space,vec_CS[t],label=str(t))
    plt.legend()
    plt.show()