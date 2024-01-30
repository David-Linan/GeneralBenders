import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/')
from functions.dsda_functions import initialize_model, generate_initialization

def build_hydrolisis_convergence_tests1(time: float=170*(3600),discretization: str='collocation',n_f_elements_x: int=10,n_f_elements_t: int=10) -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
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
    m.final_time = pe.Param(initialize=time,doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    m.tR = pe.Param(initialize=7.8*(3600),doc='retention time [s]')  # NOTE:  based on "the reactor retention time in this simulation scenario is 7.8h..."
    m.FFM= pe.Param(initialize=1.16,doc='fiber mash outflow [kg/s]') #NOTE: based on table B.4
    m.vx = pe.Param(initialize=m.LR/m.tR,doc='horizontal speed [m/s]') #NOTE: according to equation B5 is considered constant along the length of the reactor
    m.MFM = pe.Param(initialize=m.tR*m.FFM,doc='total mass of fiber mash inside the tank [kg]')  # TODO: according to the article, vx is assumed constant, which means tR constant, which means ration between MFM=FFM is constant. To simplify, I assume that both MFM and FFM are constant
    m.FE=pe.Param(initialize=0.025,doc='enzyme flow [kg/s]')
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
    # _C0['E']=500*(m.FE/m.FFM)
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
    # _CFEED['E']=500*(m.FE/m.FFM)
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
    
    if discretization=='collocation':
        discretizer_x = pe.TransformationFactory('dae.collocation')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')

        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_x = pe.TransformationFactory('dae.finite_difference')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, wrt=m.x, scheme='BACKWARD')

        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')

    return m

def build_hydrolisis_convergence_test2(time: float=170*(3600),discretization: str='collocation',n_f_elements_x: int=10,n_f_elements_t: int=10) -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
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
    m.final_time = pe.Param(initialize=time,doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    m.tR = pe.Param(initialize=7.8*(3600),doc='retention time [s]')  # NOTE:  based on "the reactor retention time in this simulation scenario is 7.8h..."
    m.FFM= pe.Param(initialize=1.16,doc='fiber mash outflow [kg/s]') #NOTE: based on table B.4
    m.vx = pe.Param(initialize=m.LR/m.tR,doc='horizontal speed [m/s]') #NOTE: according to equation B5 is considered constant along the length of the reactor
    m.MFM = pe.Param(initialize=m.tR*m.FFM,doc='total mass of fiber mash inside the tank [kg]')  # TODO: according to the article, vx is assumed constant, which means tR constant, which means ration between MFM=FFM is constant. To simplify, I assume that both MFM and FFM are constant
    m.FE=pe.Param(initialize=0.025,doc='enzyme flow [kg/s]')
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
    # _C0['E']=500*(m.FE/m.FFM)
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
    # _CFEED['E']=500*(m.FE/m.FFM)
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
        if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry
            if t==m.t.first() and x>m.x.first() and x<m.x.last(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
                return m.C[t,x,j] == m.CFEED[j]
            elif x==m.x.last():  #boundary condition 2 #TODO: CHECK IF THIS IS CORRECT
                return m.dCdx[t,x,j] == 0
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] + m.D[t,x]*m.dC2dx2[t,x,j]+m.dCdx[t,x,j]*m.dDdx[t,x]+m.R[t,x,j]
        else:
            if t==m.t.first() and x>m.x.first(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
                return m.C[t,x,j] == m.CFEED[j]
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
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
        K2_r2=0.0023          # reaction rate constant, kg/(g*s)
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
    
    
    if discretization=='collocation':
        discretizer_x = pe.TransformationFactory('dae.collocation')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')

        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_x = pe.TransformationFactory('dae.finite_difference')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, wrt=m.x, scheme='BACKWARD')

        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    return m

def build_hydrolisis_convergence_test3(time: float=170*(3600),discretization: str='collocation',n_f_elements_x: int=10,n_f_elements_t: int=10) -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
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
    m.final_time = pe.Param(initialize=time,doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    m.tR = pe.Param(initialize=7.8*(3600),doc='retention time [s]')  # NOTE:  based on "the reactor retention time in this simulation scenario is 7.8h..."
    m.FFM= pe.Param(initialize=1.16,doc='fiber mash outflow [kg/s]') #NOTE: based on table B.4
    m.FE=pe.Param(initialize=0.025,doc='enzyme flow [kg/s]')
    m.vx = pe.Param(initialize=m.LR/m.tR,doc='horizontal speed [m/s]') #NOTE: according to equation B5 is considered constant along the length of the reactor
    m.MFM = pe.Param(initialize=m.tR*m.FFM,doc='total mass of fiber mash inside the tank [kg]')  # TODO: according to the article, vx is assumed constant, which means tR constant, which means ration between MFM=FFM is constant. To simplify, I assume that both MFM and FFM are constant
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=50+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    m.conv_fact_sol=pe.Param(initialize=1.17,doc='Conversion factor from solid mass fraction to solid volume fraction for viscosity calculation')
    m.liquid_viscosity_factor=pe.Param(initialize=0.5,doc='kg of fibers/kg of liquid part of slurry') #NOTE: Approximated as 1-dry fiver fraction based on supp. material of optimization article, it can bee as high as 40%.
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
    # _C0['E']=500*(m.FE/m.FFM)
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
    # _CFEED['E']=500*(m.FE/m.FFM)
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
    m.liquid_mu=pe.Var(m.t,m.x,initialize=1,within=pe.NonNegativeReals, doc='units of g/ m s')
    m.slurry_mu=pe.Var(m.t,m.x,initialize=1,within=pe.NonNegativeReals, doc='units of g/ m s')


    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dCdx=dae.DerivativeVar(m.C,wrt=m.x)
    m.dC2dx2=dae.DerivativeVar(m.C,wrt=(m.x,m.x))
    m.dDdx=dae.DerivativeVar(m.D,wrt=m.x)
    #--------constraitns----------------------------------------------------

    # MAIN PARTIAL DIFFERENTIAL EQUATION
    def _partialDiff(m,t,x,j):
        if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry
            if t==m.t.first() and x>m.x.first() and x<m.x.last(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
                return m.C[t,x,j] == m.CFEED[j]
            elif x==m.x.last():  #boundary condition 2 #TODO: CHECK IF THIS IS CORRECT
                return m.dCdx[t,x,j] == 0
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] + m.D[t,x]*m.dC2dx2[t,x,j]+m.dCdx[t,x,j]*m.dDdx[t,x]+m.R[t,x,j]
        else:
            if t==m.t.first() and x>m.x.first(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
                return m.C[t,x,j] == m.CFEED[j]
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.partialDiff=pe.Constraint(m.t,m.x,m.j,rule=_partialDiff)


    if discretization=='collocation':
        discretizer_x = pe.TransformationFactory('dae.collocation')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')

        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_x = pe.TransformationFactory('dae.finite_difference')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, wrt=m.x, scheme='BACKWARD')

        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')

    # DIFFUSION MODEL
    #NOTE: affects only soluble particles, not solids. Meaning that we can neglegt solid fractions in these calculations!
    #NOTE: varying D across x axis because slurry viscosity is expected to decrease as liquefaction progresses, while liquid viscosity increases as sugars are formed and disolved


    
    def _liquid_mu_definition(m,t,x):
        A_W=2.41E-3 # g/(m*s)
        B_W=1774.9 # K
        A_G=8.65E-10 # m^2/s
        B_G=2502 #K                                                         # Glucose concentration in the liquid part of the slurry in mass/volume units. C[t,x,G] is glucose concentration in g/(kg  of non-dry fibers). Thus, I need to multiply by (kg  of non-dry fibers)/(kg in the liquid part of the slurry)=1-dry fiver fraction aprox, as well as by the density
        return m.liquid_mu[t,x] == (A_W*pe.exp(B_W/m.T))          +       (m.C[t,x,'G']*(1-m.liquid_viscosity_factor)*m.rho_soluble*A_G*pe.exp(B_G/m.T))
    m.liquid_mu_definition=pe.Constraint(m.t,m.x, rule=_liquid_mu_definition)
    

    # m.solid_fraction=pe.Var(m.t,m.x,initialize=1,within=pe.NonNegativeReals, doc='[-]',bounds=(0,1))

    # def _solid_fraction_definition(m,t,x): #NOTE: Here I am using article information that assumes that initial solid content is 25%
    #     return m.solid_fraction[t,x]==(1/1000)*(m.C[t,x,'CS']+m.C[t,x,'XS']+m.C[t,x,'LS'])
    # m.solid_fraction_definition=pe.Constraint(m.t,m.x,rule=_solid_fraction_definition)

    # def _slurry_mu_definition(m,t,x): 
    #     a_0=1
    #     a_1=2.5
    #     a_2=10.05
    #     return m.slurry_mu[t,x] == (a_0+a_1*(m.solid_fraction[t,x])+a_2*((m.solid_fraction[t,x])**2))*m.liquid_mu[t,x]
    # m.slurry_mu_definition=pe.Constraint(m.t,m.x, rule=_slurry_mu_definition)

    def _slurry_mu_definition(m,t,x): 
        a_0=1
        a_1=2.5
        a_2=10.05
        return m.slurry_mu[t,x] == (a_0+a_1*((1/1000)*m.conv_fact_sol*(m.C[t,x,'CS']+m.C[t,x,'XS']+m.C[t,x,'LS']))+a_2*(((1/1000)*m.conv_fact_sol*(m.C[t,x,'CS']+m.C[t,x,'XS']+m.C[t,x,'LS']))**2))*m.liquid_mu[t,x]
    m.slurry_mu_definition=pe.Constraint(m.t,m.x, rule=_slurry_mu_definition)

    def _D_definition(m,t,x):                              
                                                          #Molecular radius                                                                 #Liquid viscosity [g/(m*s)]*[1 kg/1000 g]..... Concentration multiplied by density to obtain in g/m^3
        return m.D[t,x]== (m.Boltzmann*m.T)/(6*math.pi*   (((3*m.MW_soluble)/(4*math.pi*m.Avogadro*m.rho_soluble))**(1/3))           *     (( m.liquid_mu[t,x]   )*(1/1000))             )
    m.D_definition=pe.Constraint(m.t,m.x, rule=_D_definition)

    # pH MODELING

    m.eta_T=pe.Param(initialize=1, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
    

    def _pH(m,x): #TODO: either leave pH constant or include electrolyte balance
        return 0.00001265*(x**4) - 0.00065130*(x**3) + 0.01255702*(x**2) - 0.11425695*x + 4.99306482
    m.pH=pe.Param(m.x,initialize=_pH,doc='pH profile for model validation')


    def _eta_pH_init(m,x):  
        return pe.exp(-((m.pH[x]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Param(m.x,initialize=_eta_pH_init, doc='pH efficiency factor. Value between 0 and 1')  

    def _eta_init(m,x):
        return m.eta_T*m.eta_pH[x]
    m.eta=pe.Param(m.x,initialize=_eta_init,doc='temperature and pH dependence of reaction rates') 


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
        
        return m.r1[t,x] == (K1_r1*m.eta[x]*m.CebC[t,x,'1']*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC1_r1)+(m.C[t,x,'X']/IX1_r1)+(m.C[t,x,'G']/IG1_r1)+(m.C[t,x,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,m.x,rule=_r1_definition)

    def _r2_definition(m,t,x):
        K2_r2=0.0023          # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t,x] == (K2_r2*m.eta[x]*(m.CebC[t,x,'1']+m.CebC[t,x,'2'])*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC2_r2)+(m.C[t,x,'X']/IX2_r2)+(m.C[t,x,'G']/IG2_r2)+(m.C[t,x,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,m.x,rule=_r2_definition)

    def _r3_definition(m,t,x):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t,x] == (K3_r3*m.eta[x]* m.Cef[t,x,'2']*m.C[t,x,'C'])/(I3_r3*(1+(m.C[t,x,'X']/IX3_r3)+(m.C[t,x,'G']/IG3_r3)+(m.C[t,x,'F']/IF3_r3))+m.C[t,x,'C'])
    m.r3_definition=pe.Constraint(m.t,m.x,rule=_r3_definition)

    def _r4_definition(m,t,x):
        K4_r4=0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t,x] == (K4_r4*m.eta[x]*m.CebX[t,x,'3']*m.C[t,x,'XS'])/(1+(m.C[t,x,'C']/IC4_r4)+(m.C[t,x,'X']/IX4_r4)+(m.C[t,x,'G']/IG4_r4)+(m.C[t,x,'F']/IF4_r4))
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
    





    return m

def build_hydrolisis_convergence_test4(time: float=170*(3600),discretization: str='collocation',n_f_elements_x: int=10,n_f_elements_t: int=10) -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
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
    m.final_time = pe.Param(initialize=time,doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.LR = pe.Param(initialize=20,doc='reactor length [m]')  # NOTE: this is the reactor length based on Fig. 4 of the hydrolisis article by prunescu.
    m.tR = pe.Param(initialize=7.8*(3600),doc='retention time [s]')  # NOTE:  based on "the reactor retention time in this simulation scenario is 7.8h..."
    m.FFF=pe.Param(initialize=1.11,doc='fiber flow [kg/s]') #NOTE: based on table B.4
    m.FFM= pe.Param(initialize=1.16,doc='fiber mash outflow [kg/s]') #NOTE: based on table B.4
    m.FE=pe.Param(initialize=0.025,doc='enzyme flow [kg/s]')
    m.FB=pe.Param(initialize=0.012,doc='Base flow [kg/s]')
    m.vx = pe.Param(initialize=m.LR/m.tR,doc='horizontal speed [m/s]') #NOTE: according to equation B5 is considered constant along the length of the reactor
    m.MFM = pe.Param(initialize=m.tR*m.FFM,doc='total mass of fiber mash inside the tank [kg]')  # TODO: according to the article, vx is assumed constant, which means tR constant, which means ration between MFM=FFM is constant. To simplify, I assume that both MFM and FFM are constant
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=50+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    m.conv_fact_sol=pe.Param(initialize=1.17,doc='Conversion factor from solid mass fraction to solid volume fraction for viscosity calculation')
    m.liquid_viscosity_factor=pe.Param(initialize=0.5,doc='kg of fibers/kg of liquid part of slurry') #NOTE: Approximated as 1-dry fiver fraction based on supp. material of optimization article, it can bee as high as 40%.
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
    _C0['CS']=112.5*(m.FFF/m.FFM)
    _C0['XS']=20*(m.FFF/m.FFM)
    _C0['LS']=80*(m.FFF/m.FFM)
    _C0['C']=0*(m.FFF/m.FFM)
    _C0['G']=0.5*(m.FFF/m.FFM)
    _C0['X']=2.5*(m.FFF/m.FFM)
    _C0['F']=1.8*(m.FFF/m.FFM)
    _C0['E']=500*(m.FE/m.FFM)
    _C0['AC']=5*(m.FFF/m.FFM)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the differnt species [g / kg]')  # TODO: check units and declare correct values


    _CFEED={}
    _CFEED['CS']=112.5*(m.FFF/m.FFM)
    _CFEED['XS']=20*(m.FFF/m.FFM)
    _CFEED['LS']=80*(m.FFF/m.FFM)
    _CFEED['C']=0*(m.FFF/m.FFM)
    _CFEED['G']=0.5*(m.FFF/m.FFM)
    _CFEED['X']=2.5*(m.FFF/m.FFM)
    _CFEED['F']=1.8*(m.FFF/m.FFM)
    _CFEED['E']=500*(m.FE/m.FFM)
    _CFEED['AC']=5*(m.FFF/m.FFM)
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
    m.liquid_mu=pe.Var(m.t,m.x,initialize=1,within=pe.NonNegativeReals, doc='units of g/ m s')
    m.slurry_mu=pe.Var(m.t,m.x,initialize=1,within=pe.NonNegativeReals, doc='units of g/ m s')


    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dCdx=dae.DerivativeVar(m.C,wrt=m.x)
    m.dC2dx2=dae.DerivativeVar(m.C,wrt=(m.x,m.x))
    m.dDdx=dae.DerivativeVar(m.D,wrt=m.x)
    #--------constraitns----------------------------------------------------

    # MAIN PARTIAL DIFFERENTIAL EQUATION
    def _partialDiff(m,t,x,j):
        if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry
            if t==m.t.first() and x>m.x.first() and x<m.x.last(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
                return m.C[t,x,j] == m.CFEED[j]
            elif x==m.x.last():  #boundary condition 2 #TODO: CHECK IF THIS IS CORRECT
                return m.dCdx[t,x,j] == 0
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] + m.D[t,x]*m.dC2dx2[t,x,j]+m.dCdx[t,x,j]*m.dDdx[t,x]+m.R[t,x,j]
        else:
            if t==m.t.first() and x>m.x.first(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 #TODO: CHECK IF THIS IS CORRECT
                return m.C[t,x,j] == m.CFEED[j]
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.partialDiff=pe.Constraint(m.t,m.x,m.j,rule=_partialDiff)


    if discretization=='collocation':
        discretizer_x = pe.TransformationFactory('dae.collocation')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, ncp=3, wrt=m.x, scheme='LAGRANGE-RADAU')

        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_x = pe.TransformationFactory('dae.finite_difference')
        discretizer_x.apply_to(m, nfe=n_f_elements_x, wrt=m.x, scheme='BACKWARD')

        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')

    # DIFFUSION MODEL
    #NOTE: affects only soluble particles, not solids. Meaning that we can neglegt solid fractions in these calculations!
    #NOTE: varying D across x axis because slurry viscosity is expected to decrease as liquefaction progresses, while liquid viscosity increases as sugars are formed and disolved


    
    def _liquid_mu_definition(m,t,x):
        A_W=2.41E-3 # g/(m*s)
        B_W=1774.9 # K
        A_G=8.65E-10 # m^2/s
        B_G=2502 #K                                                         # Glucose concentration in the liquid part of the slurry in mass/volume units. C[t,x,G] is glucose concentration in g/(kg  of non-dry fibers). Thus, I need to multiply by (kg  of non-dry fibers)/(kg in the liquid part of the slurry)=1-dry fiver fraction aprox, as well as by the density
        return m.liquid_mu[t,x] == (A_W*pe.exp(B_W/m.T))          +       (m.C[t,x,'G']*(1-m.liquid_viscosity_factor)*m.rho_soluble*A_G*pe.exp(B_G/m.T))
    m.liquid_mu_definition=pe.Constraint(m.t,m.x, rule=_liquid_mu_definition)
    

    # m.solid_fraction=pe.Var(m.t,m.x,initialize=1,within=pe.NonNegativeReals, doc='[-]',bounds=(0,1))

    # def _solid_fraction_definition(m,t,x): #NOTE: Here I am using article information that assumes that initial solid content is 25%
    #     return m.solid_fraction[t,x]==(1/1000)*(m.C[t,x,'CS']+m.C[t,x,'XS']+m.C[t,x,'LS'])
    # m.solid_fraction_definition=pe.Constraint(m.t,m.x,rule=_solid_fraction_definition)

    # def _slurry_mu_definition(m,t,x): 
    #     a_0=1
    #     a_1=2.5
    #     a_2=10.05
    #     return m.slurry_mu[t,x] == (a_0+a_1*(m.solid_fraction[t,x])+a_2*((m.solid_fraction[t,x])**2))*m.liquid_mu[t,x]
    # m.slurry_mu_definition=pe.Constraint(m.t,m.x, rule=_slurry_mu_definition)

    def _slurry_mu_definition(m,t,x): 
        a_0=1
        a_1=2.5
        a_2=10.05
        return m.slurry_mu[t,x] == (a_0+a_1*((1/1000)*m.conv_fact_sol*(m.C[t,x,'CS']+m.C[t,x,'XS']+m.C[t,x,'LS']))+a_2*(((1/1000)*m.conv_fact_sol*(m.C[t,x,'CS']+m.C[t,x,'XS']+m.C[t,x,'LS']))**2))*m.liquid_mu[t,x]
    m.slurry_mu_definition=pe.Constraint(m.t,m.x, rule=_slurry_mu_definition)

    def _D_definition(m,t,x):                              
                                                          #Molecular radius                                                                 #Liquid viscosity [g/(m*s)]*[1 kg/1000 g]..... Concentration multiplied by density to obtain in g/m^3
        return m.D[t,x]== (m.Boltzmann*m.T)/(6*math.pi*   (((3*m.MW_soluble)/(4*math.pi*m.Avogadro*m.rho_soluble))**(1/3))           *     (( m.liquid_mu[t,x]   )*(1/1000))             )
    m.D_definition=pe.Constraint(m.t,m.x, rule=_D_definition)

    # pH MODELING

    m.eta_T=pe.Param(initialize=1, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
    m.eta_severity=pe.Param(initialize=1, doc='Severity factor') 
    

    m.j_elect = pe.Set(initialize=['C2H4O2', 'H+', 'C2H3O2-', 'OH-','CO2aq','HCO3-','CO3-2','C4H6O4','C4H5O4-','C4H4O4-2','C3H6O3','C3H5O3-','NaOH','Na+'],doc='components for pH calculations')
    m.r_elect = pe.Set(initialize=['1','2','3','4','5','6','7','8'],doc='set of reactions for pH calculations')

    _MW_elect={}
    _MW_elect['C2H4O2']=60.05
    _MW_elect['H+']=1.007825032
    _MW_elect[ 'C2H3O2-']=59.04
    _MW_elect['OH-']=17.007 
    _MW_elect['CO2aq']=44.009
    _MW_elect['HCO3-']=61.017
    _MW_elect['CO3-2']=60.009
    _MW_elect['C4H6O4']=118.09
    _MW_elect['C4H5O4-']=117.08
    _MW_elect['C4H4O4-2']=116.07
    _MW_elect['C3H6O3']=90.08
    _MW_elect['C3H5O3-']=89.07
    _MW_elect['NaOH']=39.997
    _MW_elect['Na+']= 22.9897693   
    m.MW_elect=pe.Param(m.j_elect,initialize=_MW_elect,doc='Molecular weight of electrolytes [g/mol]')

    _Equil_lhs={}
    _Equil_lhs['1']=1.63E-5
    _Equil_lhs['2']=1
    _Equil_lhs['3']=5.39E-14
    _Equil_lhs['4']=5.14E-7
    _Equil_lhs['5']=6.69E-11
    _Equil_lhs['6']=6.51E-5
    _Equil_lhs['7']=2.08E-6
    _Equil_lhs['8']=1.27E-4

    m.Equil_lhs=pe.Param(m.r_elect,initialize=_Equil_lhs,doc='lhs Equilibrium constants for pH calculations')
    _Equil_rhs={}
    _Equil_rhs['1']=1
    _Equil_rhs['2']=0
    _Equil_rhs['3']=1
    _Equil_rhs['4']=1
    _Equil_rhs['5']=1
    _Equil_rhs['6']=1
    _Equil_rhs['7']=1
    _Equil_rhs['8']=1
    m.Equil_rhs=pe.Param(m.r_elect,initialize=_Equil_rhs,doc='rhs constants for pH calculations')

    _coef_elect={}
    
    _coef_elect['C2H4O2','1']=-1
    _coef_elect['C2H3O2-','1']=1
    _coef_elect['H+','1']=1

    _coef_elect['NaOH','2']=-1
    _coef_elect['Na+','2']=1
    _coef_elect['OH-','2']=1

    _coef_elect['H+','3']=1
    _coef_elect['OH-','3']=1

    _coef_elect['CO2aq','4']=-1
    _coef_elect['HCO3-','4']=1
    _coef_elect['H+','4']=1

    _coef_elect['HCO3-','5']=-1
    _coef_elect['CO3-2','5']=1
    _coef_elect['H+','5']=1

    _coef_elect['C4H6O4','6']=-1
    _coef_elect['C4H5O4-','6']=1
    _coef_elect['H+','6']=1

    _coef_elect['C4H5O4-','7']=-1
    _coef_elect['C4H4O4-2','7']=1
    _coef_elect['H+','7']=1

    _coef_elect['C3H6O3','8']=-1
    _coef_elect['C3H5O3-','8']=1
    _coef_elect['H+','8']=1

    m.coef_elect=pe.Param(m.j_elect,m.r_elect,initialize=_coef_elect,default=0,doc='Stoichiometry coefficient of species j in reaction r')

    _C_elect_init_param={}

    _C_elect_init_param['CO2aq']=0.0011*(m.FFF/m.FFM)*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0.4*(m.FFF/m.FFM)* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0.7*(m.FFF/m.FFM)* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=270*(m.FB/m.FFM)* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0.01#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.x,m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.x,m.t,m.j_elect,within=pe.NonNegativeReals,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,x,t,j):
        if j=='C2H4O2':
            return m.C_elect_init[x,t,j]==m.C[t,x,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        else:
            return m.C_elect_init[x,t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.x,m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.x,m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,x,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[x,t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[x,t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[x,t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[x,t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[x,t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[x,t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.x,m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,x,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[x,t,j]==m.C_elect_init[x,t,j] + sum(m.coef_elect[j,r]*m.avance[x,t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[x,t,j]==m.C_elect_init[x,t,j] + sum(m.coef_elect[j,r]*m.avance[x,t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.x,m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,x,t): #TODO: either leave pH constant or include electrolyte balance
        return 0.00001265*(x**4) - 0.00065130*(x**3) + 0.01255702*(x**2) - 0.11425695*x + 4.99306482
    m.pH=pe.Var(m.x,m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,x,t):
        return m.pH[x,t]==-pe.log10(m.C_elect_equil[x,t,'H+'])
    m.pH_definition=pe.Constraint(m.x,m.t,rule=_pH_definition)


    def _eta_pH_init(m,x,t):  
        return pe.exp(-((pe.value(m.pH[x,t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.x,m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,x,t):
        return m.eta_pH[x,t]== pe.exp(-((m.pH[x,t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.x,m.t,rule=_eq_eta_pH) 

    def _eta_init(m,x,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[x,t])
    m.eta=pe.Var(m.x,m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,x,t):
        return m.eta[x,t]==m.eta_severity*m.eta_T*m.eta_pH[x,t]
    m.eq_eta=pe.Constraint(m.x,m.t,rule=_eq_eta)

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
        
        return m.r1[t,x] == (K1_r1*m.eta[x,t]*m.CebC[t,x,'1']*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC1_r1)+(m.C[t,x,'X']/IX1_r1)+(m.C[t,x,'G']/IG1_r1)+(m.C[t,x,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,m.x,rule=_r1_definition)

    def _r2_definition(m,t,x):
        K2_r2=0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t,x] == (K2_r2*m.eta[x,t]*(m.CebC[t,x,'1']+m.CebC[t,x,'2'])*m.C[t,x,'CS'])/(1+(m.C[t,x,'C']/IC2_r2)+(m.C[t,x,'X']/IX2_r2)+(m.C[t,x,'G']/IG2_r2)+(m.C[t,x,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,m.x,rule=_r2_definition)

    def _r3_definition(m,t,x):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t,x] == (K3_r3*m.eta[x,t]* m.Cef[t,x,'2']*m.C[t,x,'C'])/(I3_r3*(1+(m.C[t,x,'X']/IX3_r3)+(m.C[t,x,'G']/IG3_r3)+(m.C[t,x,'F']/IF3_r3))+m.C[t,x,'C'])
    m.r3_definition=pe.Constraint(m.t,m.x,rule=_r3_definition)

    def _r4_definition(m,t,x):
        K4_r4=0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t,x] == (K4_r4*m.eta[x,t]*m.CebX[t,x,'3']*m.C[t,x,'XS'])/(1+(m.C[t,x,'C']/IC4_r4)+(m.C[t,x,'X']/IX4_r4)+(m.C[t,x,'G']/IG4_r4)+(m.C[t,x,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,m.x,rule=_r4_definition)

    def _r5_definition(m,t,x):
        Beta_r5=0.5     # acetic acid to xylose ratio
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
    





    return m


# def build_fermentation_convergence_test1():

#     # ------------pyomo model------------------------------------------------
#     m = pe.ConcreteModel(name='fermentation_model')
#     # ------------shared scalars with hydrolisis model ----------------------
#     m.final_time = pe.Param(initialize=200*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
#     m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
#     m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
#     m.T=pe.Param(initialize=50+273.15, doc='Optimal enzymatic activity temperature [K]')
#     m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
#     m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
#     m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
#     #------------ new scalars -----------------------------------------------


#     # -----------sets--------------------------------------------------------
#     # Continuous time set
#     m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

#     # chemical species
#     # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
#     # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
#                             # Solid part of the slurry       # Liquid part of the slurry 
    
#     m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
#     # enzime types
#     m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
#     # ---------parameters----------------------------------------------------

#     m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
#     m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
#     m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
#     m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
#     m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
#     m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
#     m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
#     m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
#     m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
#     m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
#     m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
#     m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
#     m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
#     m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
#     m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
#     m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
#     m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
#     m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
#     m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
#     m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
#     m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
#     m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
#     m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
#     m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
#     m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
#     m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
#     m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
#     m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
#     m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
#     m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
#     m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
#     m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
#     m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

#     # TODO: NOT PROVIDED!!
#     m.K0G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K1G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K2G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     #----- Initical conditions  ----------------------------------

#     _C0={}
#     ...
#     m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
#     m.M0=pe.Param(m.j,initialize=,doc='Initial hold up in the reactor [kg]')


#     # ----- Feed --------------------------------------------------
#     _Fin={}
#     ...
#     m.Fin=pe.Param(m.t,initialize=_Fin,doc='Feed flow [kg/s]')

#     _Cin={}
#     ...
#     m.Cin=pe.Param(m.t,m.j,initialize=_Cin,doc='Feed composition [g/kg]')

#     _Fout={}
#     ...
#     m.Fout=pe.Param(m.t,initialize=_Fout,doc='Output flow [kg/s]')

#     #---- Variables from hydrolisis model
#     m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
#     m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
#     m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
#     m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
#     m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
#     m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
#     m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

#     #---- main variables -------------------------------------------------------------
#     m.C=pe.Var(m.t, m.j, initialize=1,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
#     m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
#     m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

#     # ---------Reaction kinetic expresions for fermentation part -------------------------

#     m.q=pe.Var(m.t,m.j,initialize=1,within=pe.NonNegativeReals,doc='fermentation reactions kinetic expresions [g/kg s]')

#     #---------derivative variables-------------------------------------------
#     m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
#     m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

#     #--------constraitns----------------------------------------------------

#     # Total balance differential equation
#     def _Diff_mass(m,t):    
#         if t==m.t.first(): #Initial condition
#             return m.M[t] == m.M0
#         else:
#             return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
#         -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
#     m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

#     # Balance per component equation
#     def _Diff_comp(m,t,j):
#     #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
#         if t==m.t.first(): #Initial condition
#             return m.C[t,j] == m.C0[j]
#         else:
#             return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j])) + m.R[t,j] 
#     m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)





#     # Definition of fermentation kinetic expresions
#     def _q_definition(m,t,j):
#         if j=='G': 
#             # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
#             # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
#             # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
#             # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
#             # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
#             # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
#             # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
#             # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
#             return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
#         elif j=='X':


#             return m.q[t,j]=()*



#     m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

















#     # TODO: INCLUDE HERE ALL THE HYDROLISIS EQUATIONS THAT ARE NEEDED HERE TOO!!!!





#     # Definition of reaction rates
#     def _R_definition(m,t,j):
#         if j=='CS':              
#             return m.R[t,x,j] == -m.r1[t,x]-m.r2[t,x] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
#         elif j=='XS':
#             return m.R[t,x,j] == -m.r4[t,x]-m.r5[t,x] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
#         elif j=='LS':
#             return m.R[t,x,j] == 0 
#         elif j=='C':
#             return m.R[t,x,j] == m.r1[t,x]-m.r3[t,x]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
#         elif j=='G':
#             return m.R[t,x,j] == m.r2[t,x]+m.r3[t,x]      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3)
#         elif j=='X':
#             return m.R[t,x,j] == m.r4[t,x] #Xylan->Xylose (r4)
#         elif j=='F':
#             return m.R[t,x,j] == 0
#         elif j=='E':
#             return m.R[t,x,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
#         elif j=='AC':
#             return m.R[t,x,j] == m.r5[t,x] #Xylan->Acetic Acid (r5)   

#     m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)



if __name__ == '__main__':

    # PARAMETERS
    sim_time=72000 #seconds
    discretization_type='collocation'#'collocation' #'differences'
    finite_elem_x=6 # According to prunescu the first hydrolisis reactor is discretized into 6 cells
    finite_elem_t=5

    # # 1: SOLUTION OF THE PROBLEM USING A MODEL THAT IGNORES DIFFUSION EFFECTS
    # m=build_hydrolisis_convergence_tests1(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t) # Simplest version of the model that completely ignores diffusion effects
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='knitro', tee=False)
    # generate_initialization(m=m,model_name='validation_hydrolisis_1')

    # # # PLOT GENERATION
    # # time=[]
    # # space=[]
    # # vec={}

    # # for x in m.x:
    # #     space.append(x)

    # # for t in m.t:
    # #     time.append(t)
    # #     for j in m.j:
    # #         vec[(j,t)]=[]
    # #         for x in m.x:
    # #             vec[(j,t)].append(m.C[t,x,j].value)

    # # # PLOT AT STEADY STATE
    # # t=m.t.last()
    # # for j in m.j:
    # #     if j !='E':
        
    # #         plt.plot(space,vec[(j,t)],label=j)
    # #         plt.xlabel('length [m]')
    # #         plt.ylabel('Concentration [g/kg]')
    # # plt.legend()
    # # plt.show()

    # # 2: SOLUTION OF THE PROBLEM USING A MODEL THAT CONSIDERS DIFFUSION EFFECTS
    # m=build_hydrolisis_convergence_test2(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis_1')
    # # Initialization of diffusivity coefficient
    # A_W=2.41E-3 # g/(m*s)
    # B_W=1774.9 # K
    # A_G=8.65E-10 # m^2/s
    # B_G=2502 #K
    # for t in m.t:
    #     for x in m.x:
    #         m.D[t,x].set_value((m.Boltzmann*m.T)/(6*math.pi*   (((3*m.MW_soluble)/(4*math.pi*m.Avogadro*m.rho_soluble))**(1/3))           *     (((A_W*pe.exp(B_W/m.T))          +       (pe.value(m.C[t,x,'G'])*m.rho_soluble*A_G*pe.exp(B_G/m.T))    )*(1/1000))             ))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=False)
    # generate_initialization(m=m,model_name='validation_hydrolisis_2')

    # # # PLOT GENERATION
    # # time=[]
    # # space=[]
    # # vec={}

    # # for x in m.x:
    # #     space.append(x)

    # # for t in m.t:
    # #     time.append(t)
    # #     for j in m.j:
    # #         vec[(j,t)]=[]
    # #         for x in m.x:
    # #             vec[(j,t)].append(m.C[t,x,j].value)

    # # # PLOT AT STEADY STATE
    # # t=m.t.last()
    # # for j in m.j:
    # #     if j !='E':
        
    # #         plt.plot(space,vec[(j,t)],label=j)
    # #         plt.xlabel('length [m]')
    # #         plt.ylabel('Concentration [g/kg]')
    # # plt.legend()
    # # plt.show()
    # # 3: SOLUTION OF THE PROBLEM USING A MODEL THAT CONSIDERS DIFFUSION EFFECTS AND A FIXED PH PROFILE THAT DETERMINES PH EFFICIENCY
    # m=build_hydrolisis_convergence_test3(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis_2')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=False)
    # generate_initialization(m=m,model_name='validation_hydrolisis_3')


    


    # # PLOT GENERATION
    # time=[]
    # space=[]
    # vec={}
    # slurry_mu={}
    # solid_fract={}

    # for x in m.x:
    #     space.append(x)

    # for t in m.t:
    #     time.append(t)
    #     for j in m.j:
    #         vec[(j,t)]=[]
    #         for x in m.x:
    #             vec[(j,t)].append(m.C[t,x,j].value)

    # for t in m.t:
    #     slurry_mu[t]=[]
    #     solid_fract[t]=[]
    #     for x in m.x:
    #         slurry_mu[t].append(m.slurry_mu[t,x].value)
    #         solid_fract[t].append(100*(1/1000)*m.conv_fact_sol*(m.C[t,x,'CS'].value+m.C[t,x,'XS'].value+m.C[t,x,'LS'].value))
             

    # # # PLOT AT STEADY STATE
    # # t=m.t.last()
    # # for j in m.j:
    # #     if j !='E':
        
    # #         plt.plot(space,vec[(j,t)],label=j)
    # #         plt.xlabel('length [m]')
    # #         plt.ylabel('Concentration [g/kg]')
    # # plt.legend()
    # # plt.show()

    # # VALIDATION PLOT
    # colors=['b','g','m','r','k']
    # t=m.t.last()
    # contador=-1
    # for j in m.j:
    #     if any(j==c for c in ['XS','CS','C','G','X']):
    #         contador=contador+1
    #         plt.plot(space,vec[(j,t)],colors[contador],label=j)
    #         original = pd.read_csv('biorefinery_models/'+j+'_hydrolisis.csv', header=None)
    #         plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
    #         plt.xlabel('length [m]')
    #         plt.ylabel('Concentration [g/kg]')
    # plt.legend()
    # plt.show()

    # # VISCOSITY PLOT
    # plt.plot(space,slurry_mu[t],'k')
    # original = pd.read_csv('biorefinery_models/viscosity_hydrolisis.csv', header=None)
    # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
    # plt.xlabel('length [m]')
    # plt.ylabel('Slurry viscosity [g/m s]')
    # plt.legend()
    # plt.show()

    # # SOLID FRACTION
    # plt.plot(space,solid_fract[t],'k')
    # # original = pd.read_csv('biorefinery_models/solfrac_hydrolisis.csv', header=None)
    # # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
    # plt.xlabel('length [m]')
    # plt.ylabel('Solid fraction [%]')
    # plt.legend()
    # plt.show()


    # # # PLOTS FOR EVERY COMPONENT AND EVERY POINT IN TIME
    # # for j in m.j:
    # #     for t in m.t:
    # #         plt.plot(space,vec[(j,t)],label=str(t)+' [s]')
    # #         plt.xlabel('length [m]')
    # #         plt.ylabel(j+' [g/kg]')
    # #     plt.legend()
    # #     plt.show()


    # 4: SOLUTION OF THE PROBLEM USING A MODEL THAT CONSIDERS DIFFUSION EFFECTS AND pH MODEL
    m=build_hydrolisis_convergence_test4(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t)
    m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis_3')
    opt1 = SolverFactory('gams')
    results = opt1.solve(m, solver='conopt4', tee=True)
    generate_initialization(m=m,model_name='validation_hydrolisis_5')

 # PLOT GENERATION
    time=[]
    space=[]
    vec={}
    slurry_mu={}
    solid_fract={}
    pH={}

    for x in m.x:
        space.append(x)

    for t in m.t:
        time.append(t)
        for j in m.j:
            vec[(j,t)]=[]
            for x in m.x:
                vec[(j,t)].append(m.C[t,x,j].value)

    for t in m.t:
        slurry_mu[t]=[]
        solid_fract[t]=[]
        pH[t]=[]
        for x in m.x:
            slurry_mu[t].append(m.slurry_mu[t,x].value)
            solid_fract[t].append(100*(1/1000)*m.conv_fact_sol*(m.C[t,x,'CS'].value+m.C[t,x,'XS'].value+m.C[t,x,'LS'].value))
            pH[t].append(m.pH[x,t].value)
             

    # # PLOT AT STEADY STATE
    # t=m.t.last()
    # for j in m.j:
    #     if j !='E':
        
    #         plt.plot(space,vec[(j,t)],label=j)
    #         plt.xlabel('length [m]')
    #         plt.ylabel('Concentration [g/kg]')
    # plt.legend()
    # plt.show()

    # VALIDATION PLOT
    colors=['b','g','m','r','k']
    t=m.t.last()
    contador=-1
    for j in m.j:
        if any(j==c for c in ['XS','CS','C','G','X']):
            contador=contador+1
            plt.plot(space,vec[(j,t)],colors[contador],label=j)
            original = pd.read_csv('biorefinery_models/'+j+'_hydrolisis.csv', header=None)
            plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
            plt.xlabel('length [m]')
            plt.ylabel('Concentration [g/kg]')
    plt.legend()
    plt.show()

    # VISCOSITY PLOT
    plt.plot(space,slurry_mu[t],'k')
    original = pd.read_csv('biorefinery_models/viscosity_hydrolisis.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
    plt.xlabel('length [m]')
    plt.ylabel('Slurry viscosity [g/m s]')
    plt.legend()
    plt.show()

    # SOLID FRACTION
    plt.plot(space,solid_fract[t],'k')
    original = pd.read_csv('biorefinery_models/solid_hydrolisis.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
    plt.xlabel('length [m]')
    plt.ylabel('Solid fraction [%]')
    plt.legend()
    plt.show()

    # pH
    plt.plot(space,pH[t],'k')
    original = pd.read_csv('biorefinery_models/ph_hydrolisis.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
    plt.xlabel('length [m]')
    plt.ylabel('pH')
    plt.legend()
    plt.show()

    # # PLOTS FOR EVERY COMPONENT AND EVERY POINT IN TIME
    # for j in m.j:
    #     for t in m.t:
    #         plt.plot(space,vec[(j,t)],label=str(t)+' [s]')
    #         plt.xlabel('length [m]')
    #         plt.ylabel(j+' [g/kg]')
    #     plt.legend()
    #     plt.show()
