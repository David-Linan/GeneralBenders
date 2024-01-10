import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/')
from functions.dsda_functions import initialize_model, generate_initialization



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
    m.final_time = pe.Param(initialize=170*(3600),mutable=True,doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,mutable=True,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    # m.final_time=1000
    # m.LR=1000
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
    m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------
    _C0={}
    _C0['CS']=0.1
    _C0['XS']=0.1
    _C0['LS']=0.1
    _C0['C']=0.1
    _C0['G']=0.1
    _C0['X']=0.1
    _C0['F']=0.1
    _C0['E']=0.1
    _C0['AC']=0.1
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the differnt species [g / kg]')  # TODO: check units and declare correct values


    _CFEED={}
    _CFEED['CS']=112.5
    _CFEED['XS']=20
    _CFEED['LS']=80
    _CFEED['C']=0
    _CFEED['G']=0.5
    _CFEED['X']=2.5
    _CFEED['F']=1.8
    _CFEED['E']=500
    _CFEED['AC']=5
    m.CFEED=pe.Param(m.j,initialize=_CFEED,doc='Feed concentrations [g / kg]') # TODO: check units and declare correct values

    # parameters for enzyme balance
    _alpha_enzymes={}
    _alpha_enzymes['1']=0.5
    _alpha_enzymes['2']=0.3
    _alpha_enzymes['3']=0.2
    m.alpha_enzymes=pe.Param(m.e,initialize=_alpha_enzymes,doc='Fraction of each enzyme type (between 0 and 1)')

    _max_ads_enz={}
    _max_ads_enz['1']=0.015
    _max_ads_enz['2']=0.01
    _max_ads_enz['3']=0.01
    m.max_ads_enz=pe.Param(m.e,initialize=_max_ads_enz,doc='Maximum adsorbed enzymes [-]')

    _k_ads={}
    _k_ads['1']=0.84
    _k_ads['2']=0.1
    _k_ads['3']=0.1

    m.k_ads=pe.Param(m.e,initialize=_k_ads,doc='Adsorption constant [-]')

    # ---------variables-----------------------------------------------------
    m.C=pe.Var(m.t, m.x, m.j, initialize=1,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ce=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')
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
    #TODO: improve this model part: currently assuming factor of one

    m.eta_T=pe.Param(initialize=1, doc='Temperature efficiency factor. Value between 0 and 1') #TODO: find a parameter from the plot in figure B.8. Temperature can be assumed constant at 50 C
    m.eta_pH=pe.Param(initialize=1, doc='pH efficiency factor. Value between 0 and 1') #TODO: find a parameter from the plot in figure B.8 assuming constant pH, or improve the pH modeling. 
    m.eta=pe.Param(initialize=m.eta_T*m.eta_pH,doc='temperature and pH dependence of reaction rates') 



    # ENZYME BALANCES
    # NOTE: I do not neet equation below because _enzyme_fractions constraint guarantees this 
    # def _enzyme_balance(m,t,x):
    #     return m.C[t,x,'E'] == sum(m.Ce[t,x,e] for e in m.e)
    # m.enzyme_balance=pe.Constraint(m.t,m.x,rule=_enzyme_balance)

    def _enzyme_fractions(m,t,x,e):
        return m.Ce[t,x,e] == m.alpha_enzymes[e]*m.C[t,x,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.x,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,x,e):
        return m.Ce[t,x,e] == m.Ceb[t,x,e]  +    m.Cef[t,x,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.x,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,x,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,x,e])/(m.C[t,x,'CS']+ m.C[t,x,'XS']+m.C[t,x,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,x,e])/(1+m.k_ads[e]*m.Cef[t,x,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.x,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,x,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,x,e] == m.Ceb[t,x,e]*((m.C[t,x,'CS'])/(m.C[t,x,'CS']+ m.C[t,x,'XS']+m.C[t,x,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,x,e] == m.Ceb[t,x,e]*((m.C[t,x,'XS'])/(m.C[t,x,'CS']+ m.C[t,x,'XS']+m.C[t,x,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.x,m.e,rule=_bounded_enzyme_concentration)





    # MODELING OF REACTIONS
    def _r1_definition(m,t,x):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t,x] == (K1_r1*m.eta*m.CebC[t,x,'1']*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC1_r1)+(m.C[t,x,'X']/IX1_r1)+(m.C[t,x,'G']/IG1_r1)+(m.C[t,x,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,m.x,rule=_r1_definition)

    def _r2_definition(m,t,x):
        K2_r2=0.0053          # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t,x] == (K2_r2*m.eta*(m.CebC[t,x,'1']+m.CebC[t,x,'2'])*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC2_r2)+(m.C[t,x,'X']/IX2_r2)+(m.C[t,x,'G']/IG2_r2)+(m.C[t,x,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,m.x,rule=_r2_definition)

    def _r3_definition(m,t,x):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t,x] == (K3_r3*m.eta* m.Cef[t,x,'2']*m.C[t,x,'C'])/(I3_r3*(1+(m.C[t,x,'X']/IX3_r3)+(m.C[t,x,'G']/IG3_r3)+(m.C[t,x,'F']/IF3_r3))+m.C[t,x,'C'])
    m.r3_definition=pe.Constraint(m.t,m.x,rule=_r3_definition)

    def _r4_definition(m,t,x):
        K4_r4=0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t,x] == (K4_r4*m.eta*m.CebX[t,x,'3']*m.C[t,x,'XS'])/(1+(m.C[t,x,'C']/IC4_r4)+(m.C[t,x,'X']/IX4_r4)+(m.C[t,x,'G']/IG4_r4)+(m.C[t,x,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,m.x,rule=_r4_definition)

    def _r5_definition(m,t,x):
        Beta_r5=0.2     # acetic acid to xylose ratio
        return m.r5[t,x] ==Beta_r5*m.r4[t,x] 
    m.r5_definition=pe.Constraint(m.t,m.x,rule=_r5_definition)
    # ['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC']
    def _R_definition(m,t,x,j):
        if j=='CS':              
            return m.R[t,x,j] == -m.r1[t,x]-m.r2[t,x] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,x,j] == -m.r4[t,x]-m.r5[t,x] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,x,j] == 0 
        elif j=='C':
            return m.R[t,x,j] == m.r1[t,x]-m.r3[t,x]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,x,j] == m.r2[t,x]+m.r3[t,x]      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3)
        elif j=='X':
            return m.R[t,x,j] == m.r4[t,x] #Xylan->Xylose (r4)
        elif j=='F':
            return m.R[t,x,j] == 0
        elif j=='E':
            return m.R[t,x,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,x,j] == m.r5[t,x] #Xylan->Acetic Acid (r5)         
    m.R_definition=pe.Constraint(m.t,m.x,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)
    
    
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=10, wrt=m.x, scheme='BACKWARD')


    discretizer_x = pe.TransformationFactory('dae.collocation')
    discretizer_x.apply_to(m, nfe=10, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')


    discretizer_t = pe.TransformationFactory('dae.collocation')
    discretizer_t.apply_to(m, nfe=10, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')


    return m


def build_hydrolisis_convergence_tests1() -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
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
    m.final_time = pe.Param(initialize=170*(3600),mutable=True,doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,mutable=True,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    # m.final_time=1
    # m.LR=2
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
    m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------
    _C0={}
    _C0['CS']=112.5
    _C0['XS']=20
    _C0['LS']=80
    _C0['C']=0
    _C0['G']=0.5
    _C0['X']=2.5
    _C0['F']=1.8
    _C0['E']=500
    _C0['AC']=5
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the differnt species [g / kg]')  # TODO: check units and declare correct values


    _CFEED={}
    _CFEED['CS']=112.5
    _CFEED['XS']=20
    _CFEED['LS']=80
    _CFEED['C']=0
    _CFEED['G']=0.5
    _CFEED['X']=2.5
    _CFEED['F']=1.8
    _CFEED['E']=500
    _CFEED['AC']=5
    m.CFEED=pe.Param(m.j,initialize=_CFEED,doc='Feed concentrations [g / kg]') # TODO: check units and declare correct values

    # parameters for enzyme balance
    _alpha_enzymes={}
    _alpha_enzymes['1']=0.5
    _alpha_enzymes['2']=0.3
    _alpha_enzymes['3']=0.2
    m.alpha_enzymes=pe.Param(m.e,initialize=_alpha_enzymes,doc='Fraction of each enzyme type (between 0 and 1)')

    _max_ads_enz={}
    _max_ads_enz['1']=0.015
    _max_ads_enz['2']=0.01
    _max_ads_enz['3']=0.01
    m.max_ads_enz=pe.Param(m.e,initialize=_max_ads_enz,doc='Maximum adsorbed enzymes [-]')

    _k_ads={}
    _k_ads['1']=0.84
    _k_ads['2']=0.1
    _k_ads['3']=0.1

    m.k_ads=pe.Param(m.e,initialize=_k_ads,doc='Adsorption constant [-]')

    # ---------variables-----------------------------------------------------
    m.C=pe.Var(m.t, m.x, m.j, initialize=1,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ce=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.x, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t, m.x,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')
    m.R = pe.Var(m.t, m.x, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')
    # m.D = pe.Var(m.t,m.x, initialize=1, within=pe.NonNegativeReals, doc='units of m^2 / s')



    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dCdx=dae.DerivativeVar(m.C,wrt=m.x)
    # m.dC2dx2=dae.DerivativeVar(m.C,wrt=(m.x,m.x))
    # m.dDdx=dae.DerivativeVar(m.D,wrt=m.x)
    #--------constraitns----------------------------------------------------

    # MAIN PARTIAL DIFFERENTIAL EQUATION
    def _partialDiff(m,t,x,j):
        if t==m.t.first() and x>m.x.first(): #and x<m.x.last(): #Initial condition
            return m.C[t,x,j] == m.C0[j]
        elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
            return m.C[t,x,j] == m.CFEED[j]
        # elif x==m.x.last():  #boundary condition 2 #TODO: CHECK IF THIS IS CORRECT
        #     return m.dCdx[t,x,j] == 0
        else:  # Partial differential equation
            return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]
    m.partialDiff=pe.Constraint(m.t,m.x,m.j,rule=_partialDiff)
    
    # DIFFUSION MODEL
    #NOTE: affects only soluble particles, not solids. Meaning that we can neglegt solid fractions in these calculations!
    #NOTE: varying D across x axis because slurry viscosity is expected to decrease as liquefaction progresses, while liquid viscosity increases as sugars are formed and disolved
    # def _D_definition(m,t,x):                              
    #     A_W=2.41E-3 # g/(m*s)
    #     B_W=1774.9 # K
    #     A_G=8.65E-10 # m^2/s
    #     B_G=2502 #K
    #                                                       #Molecular radius                                                                 #Liquid viscosity [g/(m*s)]*[1 kg/1000 g]..... Concentration multiplied by density to obtain in g/m^3
    #     return m.D[t,x]== (m.Boltzmann*m.T)/(6*math.pi*   (((3*m.MW_soluble)/(4*math.pi*m.Avogadro*m.rho_soluble))**(1/3))           *     (((A_W*pe.exp(B_W/m.T))          +       (m.C[t,x,'G']*m.rho_soluble*A_G*pe.exp(B_G/m.T))    )*(1/1000))             )
    # m.D_definition=pe.Constraint(m.t,m.x, rule=_D_definition)

    # pH MODELING
    #TODO: improve this model part: currently assuming factor of one

    m.eta_T=pe.Param(initialize=1, doc='Temperature efficiency factor. Value between 0 and 1') #TODO: find a parameter from the plot in figure B.8. Temperature can be assumed constant at 50 C
    m.eta_pH=pe.Param(initialize=1, doc='pH efficiency factor. Value between 0 and 1') #TODO: find a parameter from the plot in figure B.8 assuming constant pH, or improve the pH modeling. 
    m.eta=pe.Param(initialize=m.eta_T*m.eta_pH,doc='temperature and pH dependence of reaction rates') 



    # ENZYME BALANCES
    # NOTE: I do not neet equation below because _enzyme_fractions constraint guarantees this 
    # def _enzyme_balance(m,t,x):
    #     return m.C[t,x,'E'] == sum(m.Ce[t,x,e] for e in m.e)
    # m.enzyme_balance=pe.Constraint(m.t,m.x,rule=_enzyme_balance)

    def _enzyme_fractions(m,t,x,e):
        return m.Ce[t,x,e] == m.alpha_enzymes[e]*m.C[t,x,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.x,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,x,e):
        return m.Ce[t,x,e] == m.Ceb[t,x,e]  +    m.Cef[t,x,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.x,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,x,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,x,e])/(m.C[t,x,'CS']+ m.C[t,x,'XS']+m.C[t,x,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,x,e])/(1+m.k_ads[e]*m.Cef[t,x,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.x,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,x,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,x,e] == m.Ceb[t,x,e]*((m.C[t,x,'CS'])/(m.C[t,x,'CS']+ m.C[t,x,'XS']+m.C[t,x,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,x,e] == m.Ceb[t,x,e]*((m.C[t,x,'XS'])/(m.C[t,x,'CS']+ m.C[t,x,'XS']+m.C[t,x,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.x,m.e,rule=_bounded_enzyme_concentration)





    # MODELING OF REACTIONS
    def _r1_definition(m,t,x):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t,x] == (K1_r1*m.eta*m.CebC[t,x,'1']*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC1_r1)+(m.C[t,x,'X']/IX1_r1)+(m.C[t,x,'G']/IG1_r1)+(m.C[t,x,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,m.x,rule=_r1_definition)

    def _r2_definition(m,t,x):
        K2_r2=0.0053          # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t,x] == (K2_r2*m.eta*(m.CebC[t,x,'1']+m.CebC[t,x,'2'])*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC2_r2)+(m.C[t,x,'X']/IX2_r2)+(m.C[t,x,'G']/IG2_r2)+(m.C[t,x,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,m.x,rule=_r2_definition)

    def _r3_definition(m,t,x):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t,x] == (K3_r3*m.eta* m.Cef[t,x,'2']*m.C[t,x,'C'])/(I3_r3*(1+(m.C[t,x,'X']/IX3_r3)+(m.C[t,x,'G']/IG3_r3)+(m.C[t,x,'F']/IF3_r3))+m.C[t,x,'C'])
    m.r3_definition=pe.Constraint(m.t,m.x,rule=_r3_definition)

    def _r4_definition(m,t,x):
        K4_r4=0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t,x] == (K4_r4*m.eta*m.CebX[t,x,'3']*m.C[t,x,'XS'])/(1+(m.C[t,x,'C']/IC4_r4)+(m.C[t,x,'X']/IX4_r4)+(m.C[t,x,'G']/IG4_r4)+(m.C[t,x,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,m.x,rule=_r4_definition)

    def _r5_definition(m,t,x):
        Beta_r5=0.2     # acetic acid to xylose ratio
        return m.r5[t,x] ==Beta_r5*m.r4[t,x] 
    m.r5_definition=pe.Constraint(m.t,m.x,rule=_r5_definition)
    # ['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC']
    def _R_definition(m,t,x,j):
        if j=='CS':              
            return m.R[t,x,j] == -m.r1[t,x]-m.r2[t,x] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,x,j] == -m.r4[t,x]-m.r5[t,x] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,x,j] == 0 
        elif j=='C':
            return m.R[t,x,j] == m.r1[t,x]-m.r3[t,x]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,x,j] == m.r2[t,x]+m.r3[t,x]      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3)
        elif j=='X':
            return m.R[t,x,j] == m.r4[t,x] #Xylan->Xylose (r4)
        elif j=='F':
            return m.R[t,x,j] == 0
        elif j=='E':
            return m.R[t,x,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,x,j] == m.r5[t,x] #Xylan->Acetic Acid (r5)         
    m.R_definition=pe.Constraint(m.t,m.x,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)
    
    
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=10, wrt=m.x, scheme='BACKWARD')


    discretizer_x = pe.TransformationFactory('dae.collocation')
    discretizer_x.apply_to(m, nfe=10, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')


    discretizer_t = pe.TransformationFactory('dae.collocation')
    discretizer_t.apply_to(m, nfe=10, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')


    return m


if __name__ == '__main__':

    m=build_hydrolisis_convergence_tests1() # Simplest version of the model that completely ignores diffusion effects

    m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis')

    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # solved=generate_initialization(m=m,model_name='validation_hydrolisis')

    time=[]
    space=[]
    vec={}



    for x in m.x:
        space.append(x)

    for t in m.t:
        time.append(t)
        for j in m.j:
            vec[(j,t)]=[]
            for x in m.x:
                vec[(j,t)].append(m.C[t,x,j].value)

    # PLOT AT STEADY STATE
    t=m.t.last()
    for j in m.j:
        if j !='E':
        
            plt.plot(space,vec[(j,t)],label=j)
            plt.xlabel('length [m]')
            plt.ylabel('Concentration [g/kg]')
    plt.legend()
    plt.show()


    # PLOTS FOR EVERY COMPONENT AND EVERY POINT IN TIME
    for j in m.j:
        for t in m.t:
            plt.plot(space,vec[(j,t)],label=str(t)+' [s]')
            plt.xlabel('length [m]')
            plt.ylabel(j+' [g/kg]')
        plt.legend()
        plt.show()