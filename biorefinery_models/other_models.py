import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory
import matplotlib.pyplot as plt
import math



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
    
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=50+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose

    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, m.final_time))
    # spatial coordinate index (x)
    m.x = dae.ContinuousSet(bounds=(0, m.LR))
    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E'])
                            # Solid part of the slurry       # Liquid part of the slurry 






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
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the differnt species [g / kg]')  # TODO: check units and declare correct values


    _CFEED={}
    _CFEED['CS']=0.5
    _CFEED['XS']=0.5
    _CFEED['LS']=0.5
    _CFEED['C']=0.5
    _CFEED['G']=0.5
    _CFEED['X']=0.5
    _CFEED['F']=0.5
    _CFEED['E']=0.5
    m.CFEED=pe.Param(m.j,initialize=_CFEED,doc='Feed concentrations [g / kg]') # TODO: check units and declare correct values


    # ---------variables-----------------------------------------------------
    m.C=pe.Var(m.t, m.x, m.j, initialize=1,within=pe.NonNegativeReals, doc='units of g/kg') #bounds=(0, 10000))
    m.R = pe.Var(m.t, m.x, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')
    m.D = pe.Var(m.t,m.x, initialize=1, within=pe.NonNegativeReals, doc='units of m^2 / s')



    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dCdx=dae.DerivativeVar(m.C,wrt=m.x)
    m.dC2dx2=dae.DerivativeVar(m.C,wrt=(m.x,m.x))
    m.dDdx=dae.DerivativeVar(m.D,wrt=m.x)
    #--------constraitns----------------------------------------------------

    # MAIN PARTIAL DIFFERENTIAL EQUATION
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
    
    # DIFFUSION MODEL
    #NOTE: affects only soluble particles, not solids. Meaning that we can neglegt solid fractions in these calculations!
    #NOTE: varying D across x axis because slurry viscosity is expected to decrease as liquefaction progresses, while liquid viscosity increases as sugars are formed and disolved
    def _D_definition(m,t,x):                              
        A_W=2.41E-3 # g/(m*s)
        B_W=1774.9 # K
        A_G=8.65E-10 # m^2/s
        B_G=2502 #K
                                                          #Molecular radius                                                                 #Liquid viscosity [g/(m*s)]*[1 kg/1000 g]..... Concentration multiplied by density to obtain in g/m^3
        return m.D[t,x]== (m.Boltzmann*m.T)/(6*math.pi*   (((3*m.MW_soluble)/(4*math.pi*m.Avogadro*m.rho_soluble))**(1/3))           *     (((A_W*pe.exp(B_W/m.T))          +       (m.C[t,x,'G']*m.rho_soluble*A_G*pe.exp(B_G/m.T))    )*(1/1000))             )
    m.D_definition=pe.Constraint(m.t,m.x, rule=_D_definition)

    # pH MODELING
    #TODO: improve this model part



    m.eta_T=pe.Param(initialize=1, doc='Temperature efficiency factor. Value between 0 and 1') #TODO: find a parameter from the plot in figure B.8. Temperature can be assumed constant at 50 C
    m.eta_pH=pe.Param(initialize=1, doc='pH efficiency factor. Value between 0 and 1') #TODO: find a parameter from the plot in figure B.8 assuming constant pH, or improve the pH modeling. 
    m.eta=pe.Param(initialize=m.eta_T*m.eta_pH,doc='temperature and pH dependence of reaction rate r1') 

    # MODELING OF REACTIONS


    def _R_definition(m,t,x):
        if m.j=='CS':
            return 
        elif m.j=='':
            return

    m.R_definition=pe.Constraint(m.t,m.x,m.j, rule=_R_definition)

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