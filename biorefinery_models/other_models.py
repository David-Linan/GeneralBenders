import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import numpy as np
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

# No reactions
def build_fermentation_convergence_test1(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=200*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=1,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=1,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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


    
    #----- Initical conditions  ----------------------------------
    _C0={}
    _C0['CS']=50
    _C0['XS']=1
    _C0['LS']=78
    _C0['C']=0   # NOT reported
    _C0['G']=98
    _C0['X']=59
    _C0['F']=0.2
    _C0['E']=4.9
    _C0['AC']=16 # this may be the mixture of acids
    _C0['Cell']=20 # Same as yeast (?)
    _C0['Eth']=0
    _C0['CO2']=0
    _C0['ACT']=0.1
    _C0['HMF']=0.1
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    m.M0=pe.Param(initialize=220000,doc='Initial hold up in the reactor [kg]')


    # ----- Feed parameters --------------------------------------------------
    # _Fin={}
    # ...
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')

    # _Cin={}
    # ...
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')

    # _Fout={}
    # ...
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]))
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        m.Fin[t]=0
        m.Fout[t]=0
        for j in m.j:
            m.Cin[t,j]=0
    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2':
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        else:
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,t):
        return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]==(m.C[t,'G']/(m.C[t,'G']+m.C[t,'X']))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(m.C[t,'G']+m.C[t,'X']))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']+m.q[t,'X']    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m

# With reactions, no pH dependency
def build_fermentation_convergence_test2(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=1) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=200*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=100,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=100,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=100,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=100,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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


    
    #----- Initical conditions  ----------------------------------
    _C0={}
    _C0['CS']=50
    _C0['XS']=1
    _C0['LS']=78
    _C0['C']=0   # NOT reported
    _C0['G']=98
    _C0['X']=59
    _C0['F']=0.2
    _C0['E']=4.9
    _C0['AC']=16 # this may be the mixture of acids
    _C0['Cell']=40 # Same as yeast (?)
    _C0['Eth']=0
    _C0['CO2']=0
    _C0['ACT']=0.1
    _C0['HMF']=0.1
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    m.M0=pe.Param(initialize=220000,doc='Initial hold up in the reactor [kg]')


    # ----- Feed parameters --------------------------------------------------
    # _Fin={}
    # ...
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')

    # _Cin={}
    # ...
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')

    # _Fout={}
    # ...
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + conv_param*m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        m.Fin[t]=0
        m.Fout[t]=0
        for j in m.j:
            m.Cin[t,j]=0
    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0.01#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2':
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        else:
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*1  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']+m.q[t,'X']    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m
# With reactions, and pH dependency
def build_fermentation_convergence_test3(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=1) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=200*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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


    
    #----- Initical conditions  ----------------------------------
    _C0={}
    _C0['CS']=50
    _C0['XS']=1
    _C0['LS']=78
    _C0['C']=0   # NOT reported
    _C0['G']=98
    _C0['X']=59
    _C0['F']=0.2
    _C0['E']=4.9
    _C0['AC']=16 # this may be the mixture of acids
    _C0['Cell']=40 # Same as yeast (?)
    _C0['Eth']=0
    _C0['CO2']=0
    _C0['ACT']=0.1
    _C0['HMF']=0.1
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    m.M0=pe.Param(initialize=220000,doc='Initial hold up in the reactor [kg]')


    # ----- Feed parameters --------------------------------------------------
    # _Fin={}
    # ...
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')

    # _Cin={}
    # ...
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')

    # _Fout={}
    # ...
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + conv_param*m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        m.Fin[t]=0
        m.Fout[t]=0
        for j in m.j:
            m.Cin[t,j]=0
    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0.01#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2':
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        else:
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*1  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']+m.q[t,'X']    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m

# With reactions, and pH dependency, and fed batch modeling
def build_fermentation_convergence_test4(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*1  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']+m.q[t,'X']    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m

# With reactions, and pH dependency, and fed batch modeling, adjusting
def build_fermentation_convergence_test5(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.098#0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0097#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.03  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m

# # With reactions, and pH dependency, and fed batch modeling, adjusting, and new inhibition and reaction terms for hydrolisis
# def build_fermentation_convergence_test6(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

#     # ------------pyomo model------------------------------------------------
#     m = pe.ConcreteModel(name='fermentation_model')
#     # ------------shared scalars with hydrolisis model ----------------------
#     m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
#     m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
#     m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
#     m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
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
    
#     m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','XO']) #Cell is cell biomass, ACT is acetate, XO is Xyloligomers
#     # enzime types
#     m.e = pe.Set(initialize=['1','2','3','4']) #NOTE: Enzyme type 4 was included at this stage
    
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
#     m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

#     m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
#     m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
#     m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
#     # ----- Enzymatic hydrolisis parameters-----------------------

#     # parameters for enzyme balance
#     _alpha_enzymes={}
#     _alpha_enzymes['1']=0.25
#     _alpha_enzymes['2']=0.25
#     _alpha_enzymes['3']=0.25
#     _alpha_enzymes['4']=0.25
#     m.alpha_enzymes=pe.Param(m.e,initialize=_alpha_enzymes,doc='Fraction of each enzyme type (between 0 and 1)')

#     _max_ads_enz={}
#     _max_ads_enz['1']=0.016042
#     _max_ads_enz['2']=1.5E-5
#     _max_ads_enz['3']=0.38978
#     _max_ads_enz['4']=0.51178
#     m.max_ads_enz=pe.Param(m.e,initialize=_max_ads_enz,doc='Maximum adsorbed enzymes [-]')

#     _k_ads={}
#     _k_ads['1']=1.0444
#     _k_ads['2']=0.056976
#     _k_ads['3']=0.37844
#     _k_ads['4']=0.093253
#     m.k_ads=pe.Param(m.e,initialize=_k_ads,doc='Adsorption constant [-]')



#     #----- Input feed streams properties--------------------------

#     m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

#     m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
#     _C_C5liquid={}
#     _C_C5liquid['CS']=1.2
#     _C_C5liquid['XS']=0.5
#     _C_C5liquid['LS']=0.7
#     _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
#     _C_C5liquid['XO']=1.2
#     _C_C5liquid['G']=10
#     _C_C5liquid['X']=29.7
#     _C_C5liquid['F']=0.5
#     _C_C5liquid['E']=0
#     _C_C5liquid['AC']=4.1 # this may be the mixture of acids
#     _C_C5liquid['Cell']=0 # Same as yeast (?)
#     _C_C5liquid['Eth']=0
#     _C_C5liquid['CO2']=0
#     _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
#     _C_C5liquid['HMF']=0.3 
#     m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

#     _C_liquified_fibers={}
#     _C_liquified_fibers['CS']=50
#     _C_liquified_fibers['XS']=1
#     _C_liquified_fibers['LS']=78
#     _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
#     _C_liquified_fibers['XO']=5.8
#     _C_liquified_fibers['G']=98
#     _C_liquified_fibers['X']=59
#     _C_liquified_fibers['F']=0.2
#     _C_liquified_fibers['E']=4.9
#     _C_liquified_fibers['AC']=16 # this may be the mixture of acids
#     _C_liquified_fibers['Cell']=0 # Same as yeast (?)
#     _C_liquified_fibers['Eth']=0
#     _C_liquified_fibers['CO2']=0
#     _C_liquified_fibers['ACT']=0.1
#     _C_liquified_fibers['HMF']= 0.1
#     m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
#     #----- Initical conditions  ----------------------------------


#     m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
#     m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
#     m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
#     m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

#     def _C0(m,j):
#         if j=='Cell':
#             return (1000*m.M0_yeast)/(m.M0)
#         else:
#             return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
#     m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
#     #----- Maximum reactor hold up------------------------------------------------
#     m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

#     # ----- Feed parameters --------------------------------------------------
#     m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
#     m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
#     m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

#     #---- Variables from hydrolisis model--------------------------------------------------
#     m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
#     m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
#     m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
#     m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
#     m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
#     m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylooligomers rate, g/kg s')
#     m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
#     m.r6=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylooligomers to xylose rate, g/kg s')
#     m.r7=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Acetic acid production rate, g/kg s')
#     m.r8=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Enzymes deactivation due to thermal and exposure to ethanol, g/kg s')

#     #---- main variables -------------------------------------------------------------
#     def _C_init(m,t,j):
#         return m.C0[j]
#     m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
#     m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
#     m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

#     # ---------Reaction kinetic expresions for fermentation part -------------------------

#     m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

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
#             return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + 0*m.M[t]*m.R[t,j]) 
#     m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

#     if discretization=='collocation':
#         discretizer_t = pe.TransformationFactory('dae.collocation')
#         discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
#     else:
#         discretizer_t = pe.TransformationFactory('dae.finite_difference')
#         discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


#     # ------------------Re definition of feed flow and output flow information---------------------
#     for t in m.t:
#         if t*m.final_time<=10*60*60: # Inoculum phase
#             m.Fin[t]=m.F_liquified_fibers
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=m.C_liquified_fibers[j]
#         elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
#             m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
#         elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
#             m.Fin[t]=0
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=0

#     #------------------- pH modeling (from hydrolysis) ----------------------------------------------
#     m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
#     m.eta_severity=pe.Param(initialize=1, doc='Severity factor') 
    

#     m.j_elect = pe.Set(initialize=['C2H4O2', 'H+', 'C2H3O2-', 'OH-','CO2aq','HCO3-','CO3-2','C4H6O4','C4H5O4-','C4H4O4-2','C3H6O3','C3H5O3-','NaOH','Na+'],doc='components for pH calculations')
#     m.r_elect = pe.Set(initialize=['1','2','3','4','5','6','7','8'],doc='set of reactions for pH calculations')

#     _MW_elect={}
#     _MW_elect['C2H4O2']=60.05
#     _MW_elect['H+']=1.007825032
#     _MW_elect[ 'C2H3O2-']=59.04
#     _MW_elect['OH-']=17.007 
#     _MW_elect['CO2aq']=44.009
#     _MW_elect['HCO3-']=61.017
#     _MW_elect['CO3-2']=60.009
#     _MW_elect['C4H6O4']=118.09
#     _MW_elect['C4H5O4-']=117.08
#     _MW_elect['C4H4O4-2']=116.07
#     _MW_elect['C3H6O3']=90.08
#     _MW_elect['C3H5O3-']=89.07
#     _MW_elect['NaOH']=39.997
#     _MW_elect['Na+']= 22.9897693   
#     m.MW_elect=pe.Param(m.j_elect,initialize=_MW_elect,doc='Molecular weight of electrolytes [g/mol]')

#     _Equil_lhs={}
#     _Equil_lhs['1']=1.63E-5
#     _Equil_lhs['2']=1
#     _Equil_lhs['3']=5.39E-14
#     _Equil_lhs['4']=5.14E-7
#     _Equil_lhs['5']=6.69E-11
#     _Equil_lhs['6']=6.51E-5
#     _Equil_lhs['7']=2.08E-6
#     _Equil_lhs['8']=1.27E-4

#     m.Equil_lhs=pe.Param(m.r_elect,initialize=_Equil_lhs,doc='lhs Equilibrium constants for pH calculations')
#     _Equil_rhs={}
#     _Equil_rhs['1']=1
#     _Equil_rhs['2']=0
#     _Equil_rhs['3']=1
#     _Equil_rhs['4']=1
#     _Equil_rhs['5']=1
#     _Equil_rhs['6']=1
#     _Equil_rhs['7']=1
#     _Equil_rhs['8']=1
#     m.Equil_rhs=pe.Param(m.r_elect,initialize=_Equil_rhs,doc='rhs constants for pH calculations')

#     _coef_elect={}
    
#     _coef_elect['C2H4O2','1']=-1
#     _coef_elect['C2H3O2-','1']=1
#     _coef_elect['H+','1']=1

#     _coef_elect['NaOH','2']=-1
#     _coef_elect['Na+','2']=1
#     _coef_elect['OH-','2']=1

#     _coef_elect['H+','3']=1
#     _coef_elect['OH-','3']=1

#     _coef_elect['CO2aq','4']=-1
#     _coef_elect['HCO3-','4']=1
#     _coef_elect['H+','4']=1

#     _coef_elect['HCO3-','5']=-1
#     _coef_elect['CO3-2','5']=1
#     _coef_elect['H+','5']=1

#     _coef_elect['C4H6O4','6']=-1
#     _coef_elect['C4H5O4-','6']=1
#     _coef_elect['H+','6']=1

#     _coef_elect['C4H5O4-','7']=-1
#     _coef_elect['C4H4O4-2','7']=1
#     _coef_elect['H+','7']=1

#     _coef_elect['C3H6O3','8']=-1
#     _coef_elect['C3H5O3-','8']=1
#     _coef_elect['H+','8']=1

#     m.coef_elect=pe.Param(m.j_elect,m.r_elect,initialize=_coef_elect,default=0,doc='Stoichiometry coefficient of species j in reaction r')

#     _C_elect_init_param={}

#     _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
#     _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
#     _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
#     _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
#     _C_elect_init_param['H+']=0#0.01
#     m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
#     # m.C_elect_init_param.pprint()

#     m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
#     m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
#     m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

#     m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

#     m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

#     def _C_elect_init_constraint(m,t,j):
#         if t==m.t.last():
#             if j=='C2H4O2': #Acetic acid
#                 return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
#             elif j=='C2H3O2-': #Acetate
#                 return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
#             else: #TODO: this model is not rigurous enouugh? 
#                 return m.C_elect_init[t,j]==m.C_elect_init_param[j]
#         else:
#             return pe.Constraint.Skip
#     m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

#     m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


#     def _equilibrium_relationships(m,t,r):
#         if t==m.t.last():
#             # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
#             if r=='4':
#                 return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
#             # For the remaining reactions we only consider the normal equilibrium calculation
#             else:
#                 # If it is only a fordward reaction
#                 if m.Equil_rhs[r]==0:
#                     return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
#                 # If the reaction is an equilibrium reaction
#                 else:
#                     return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
#         else:
#             return pe.Constraint.Skip
#     m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

#     def _elect_balances(m,t,j):
#         if t==m.t.last():
#             if j=='CO2aq':
#                 return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
#             else:
#                 return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
#         else:
#             return pe.Constraint.Skip
#     m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

#     def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
#         return 5.5
#     m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


#     def _pH_definition(m,t):
#         # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
#         # return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
#         return 10**(-m.pH[t])==m.C_elect_equil[m.t.last(),'H+']
#     m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


#     def _eta_pH_init(m,t):  
#         return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
#     m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

#     def _eq_eta_pH(m,t):
#         return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
#     m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

#     def _eta_init(m,t):
#         return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
#     m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

#     def _eq_eta(m,t):
#         return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
#     m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

#     #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

#     def _enzyme_fractions(m,t,e):
#         return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
#     m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

#     def _bounded_free_equilibrium(m,t,e):
#         return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
#     m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

#     def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
#         # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
#         return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
#         # else:
#         #     return pe.Constraint.Skip
#     m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

#     def _bounded_enzyme_concentration(m,t,e):
#         if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
#             return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
#         else:                                           # NOTE: that denominator is Solid concentration. modify if needed
#             return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
#     m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

#     # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
#     def _r1_definition(m,t):
#         K1_r1=  0.005916     # reaction rate constant, kg/(g*s)
#         IC1_r1= 0.02014      # Inhibition of r1 by cellobiose, g/kg
#         IX1_r1= 0.01503       # Inhibition of r1 by xylose, g/kg
#         IG1_r1= 0.10255       # Inhibition of r1 by glucose, g/kg
#         IXO1_r1= 0.0078145          #  Inhibition of r1 by Xyloligomers, g/kg
#         IEth1_r1= 0.15         #  Inhibition of r1 by Ethanol, g/kg
#         return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'XO']/IXO1_r1)+(m.C[t,'Eth']/IEth1_r1))
#     m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

#     def _r2_definition(m,t):
#         K2_r2= 0.0065075              # reaction rate constant, kg/(g*s)
#         IC2_r2=  69.539         # Inhibition of r2 by cellobiose, g/kg
#         IX2_r2= 0.14843          # Inhibition of r2 by xylose, g/kg
#         IG2_r2= 0.067554          # Inhibition of r2 by glucose, g/kg
#         IXO2_r2= 0.059612
#         IEth2_r2=0.15              # NOT GIVEN            
#         return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'XO']/IXO2_r2)+(m.C[t,'Eth']/IEth2_r2))
#     m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

#     def _r3_definition(m,t):
#         K3_r3= 0.0055227               # reaction rate constant, kg/(g*s)
#         I3_r3= 15.949              #overall inhibition term for r3, g/kg
#         IX3_r3= 210.1911               # Inhibition of r3 by xylose, g/kg
#         IG3_r3= 8.7211             # Inhibition of r3 by glucose, g/kg
#         IXO3_r3=111.6822
#         IEth3_r3=0.15               
#         return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'XO']/IXO3_r3)+(m.C[t,'Eth']/IEth3_r3))+m.C[t,'C'])
#     m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

#     def _r4_definition(m,t):
#         K4_r4=0.0020026
#         IC4_r4=53.4804
#         IX4_r4=233.0874 
#         IG4_r4=2.0899
#         IXO4_r4=113.4492
#         IEth4_r4=0.15
#         return m.r4[t] == (K4_r4*m.eta[t]* m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'XO']/IXO4_r4)+(m.C[t,'Eth']/IEth4_r4))
#     m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

#     def _r5_definition(m,t):
#         K5_r5=0.0033936            # reaction rate constant, kg/(g*s)
#         IC5_r5= 2.7413         # Inhibition of r4 by cellobiose, g/kg
#         IX5_r5= 271.2334         # Inhibition of r4 by xylose, g/kg 
#         IG5_r5=   4.7951       # Inhibition of r4 by glucose, g/kg
#         IXO5_r5=  83.5479           
#         IEth5_r5=0.15
#         return m.r5[t] == (K5_r5*m.eta[t]*(m.CebX[t,'3']+m.CebX[t,'4'])*m.C[t,'XS'])/(1+(m.C[t,'C']/IC5_r5)+(m.C[t,'X']/IX5_r5)+(m.C[t,'G']/IG5_r5)+(m.C[t,'XO']/IXO5_r5)+(m.C[t,'Eth']/IEth5_r5))
#     m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)


#     def _r6_definition(m,t):
#         K6_r6=0.0028228
#         I6_r6=28.2079
#         IC6_r6=46.9663
#         IX6_r6=198.3351 
#         IG6_r6=3.0412
#         IEth6_r6=0.15
#         return m.r6[t] == (K6_r6*m.eta[t]*m.Cef[t,'4']*m.C[t,'XO'])/(I6_r6*(1+(m.C[t,'C']/IC6_r6)+(m.C[t,'X']/IX6_r6)+(m.C[t,'G']/IG6_r6)+(m.C[t,'Eth']/IEth6_r6))+m.C[t,'XO'])
#     m.r6_definition=pe.Constraint(m.t,rule=_r6_definition)


#     def _r7_definition(m,t):
#         Beta_r7=0.5     # acetic acid to xylose ratio. Not given
#         return m.r7[t] ==Beta_r7*(m.r4[t]+m.r5[t]) 
#     m.r7_definition=pe.Constraint(m.t,rule=_r7_definition)


#     def _r8_definition(m,t):
#         K8_r8= 2.5E-7  # Enzyme deactivation reaction constant kg/(g*s)
#         return m.r8[t] == K8_r8*(m.C[t,'E']**2)
#     m.r8_definition=pe.Constraint(m.t,rule=_r8_definition)


#     # --------------Definition of fermentation kinetic expresions---------------------------
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
#             # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
#             return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
#         elif j=='X':

#             # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
#             # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
#             # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
#             # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
#             # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
#             # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
#             # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
#             # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
#             # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
#             return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*1  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

#         elif j=='F':
#             return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
#         elif j=='HMF':
#             return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
#         elif j=='ACT':
#             return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
#         elif j =='Cell':
#             return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
#         else:
#             return pe.Constraint.Skip
#     m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

#     #---------------------- Definition of reaction rates------------------------------------------------
#     def _R_definition(m,t,j):

#         # Components from hydrolisis
#         if j=='CS':              
#             return m.R[t,j] == -m.r1[t]-m.r2[t] 
#         elif j=='XS':
#             return m.R[t,j] == -m.r4[t]-m.r5[t] 
#         elif j=='LS':
#             return m.R[t,j] == 0 
#         elif j=='C':
#             return m.R[t,j] == m.r1[t]-m.r3[t]    
#         elif j=='XO':
#             return m.R[t,j] == m.r4[t]-m.r6[t] 
#         elif j=='G':
#             return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      
#         elif j=='X':
#             return m.R[t,j] == m.r5[t]+m.r6[t]-m.q[t,'X'] 
#         elif j=='F':
#             return m.R[t,j] == 0 - m.q[t,'F']    
#         elif j=='E':
#             return m.R[t,j] == -m.r8[t] 
#         elif j=='AC': #Acetic acid
#             return m.R[t,j] == m.r7[t]  
        
#         # New components included in fermentation
#         elif j=='Eth':
#             return m.R[t,j] == m.q[t,'G']+m.q[t,'X']#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
#         elif j=='HMF':
#             return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
#         elif j=='ACT': #Acetate (or acetyl)
#             return m.R[t,j] ==-m.r7[t]+m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
#         elif j=='CO2':
#             return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
#         elif j=='Cell':
#             return m.R[t,j] == m.q[t,'Cell']
#         else:
#             return m.R[t,j] == 0
#         # elif j=='O':
#         #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
#     m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

#     #-------objective function--------------------------------------------

#     m.obj = pe.Objective(expr=1)

#     return m

# # With reactions, and pH dependency, and fed batch modeling, adjusting, and new inhibition and reaction terms for hydrolisis
# def build_fermentation_convergence_test7(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=0) -> pe.ConcreteModel():

#     # ------------pyomo model------------------------------------------------
#     m = pe.ConcreteModel(name='fermentation_model')
#     # ------------shared scalars with hydrolisis model ----------------------
#     m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
#     m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
#     m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
#     m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
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
    
#     m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','XO']) #Cell is cell biomass, ACT is acetate, XO is Xyloligomers
#     # enzime types
#     m.e = pe.Set(initialize=['1','2','3','4']) #NOTE: Enzyme type 4 was included at this stage
    
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
#     m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

#     m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
#     m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
#     m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
#     # ----- Enzymatic hydrolisis parameters-----------------------

#     # parameters for enzyme balance
#     _alpha_enzymes={}
#     _alpha_enzymes['1']=0.25
#     _alpha_enzymes['2']=0.25
#     _alpha_enzymes['3']=0.25
#     _alpha_enzymes['4']=0.25
#     m.alpha_enzymes=pe.Param(m.e,initialize=_alpha_enzymes,doc='Fraction of each enzyme type (between 0 and 1)')

#     _max_ads_enz={}
#     _max_ads_enz['1']=0.016042
#     _max_ads_enz['2']=1.5E-5
#     _max_ads_enz['3']=0.38978
#     _max_ads_enz['4']=0.51178
#     m.max_ads_enz=pe.Param(m.e,initialize=_max_ads_enz,doc='Maximum adsorbed enzymes [-]')

#     _k_ads={}
#     _k_ads['1']=1.0444
#     _k_ads['2']=0.056976
#     _k_ads['3']=0.37844
#     _k_ads['4']=0.093253
#     m.k_ads=pe.Param(m.e,initialize=_k_ads,doc='Adsorption constant [-]')



#     #----- Input feed streams properties--------------------------

#     m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

#     m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
#     _C_C5liquid={}
#     _C_C5liquid['CS']=1.2
#     _C_C5liquid['XS']=0.5
#     _C_C5liquid['LS']=0.7
#     _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
#     _C_C5liquid['XO']=1.2
#     _C_C5liquid['G']=10
#     _C_C5liquid['X']=29.7
#     _C_C5liquid['F']=0.5
#     _C_C5liquid['E']=0
#     _C_C5liquid['AC']=4.1 # this may be the mixture of acids
#     _C_C5liquid['Cell']=0 # Same as yeast (?)
#     _C_C5liquid['Eth']=0
#     _C_C5liquid['CO2']=0
#     _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
#     _C_C5liquid['HMF']=0.3 
#     m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

#     _C_liquified_fibers={}
#     _C_liquified_fibers['CS']=50
#     _C_liquified_fibers['XS']=1
#     _C_liquified_fibers['LS']=78
#     _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
#     _C_liquified_fibers['XO']=5.8
#     _C_liquified_fibers['G']=98
#     _C_liquified_fibers['X']=59
#     _C_liquified_fibers['F']=0.2
#     _C_liquified_fibers['E']=4.9
#     _C_liquified_fibers['AC']=16 # this may be the mixture of acids
#     _C_liquified_fibers['Cell']=0 # Same as yeast (?)
#     _C_liquified_fibers['Eth']=0
#     _C_liquified_fibers['CO2']=0
#     _C_liquified_fibers['ACT']=0.1
#     _C_liquified_fibers['HMF']= 0.1
#     m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
#     #----- Initical conditions  ----------------------------------


#     m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
#     m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
#     m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
#     m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

#     def _C0(m,j):
#         if j=='Cell':
#             return (1000*m.M0_yeast)/(m.M0)
#         else:
#             return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
#     m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
#     #----- Maximum reactor hold up------------------------------------------------
#     m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

#     # ----- Feed parameters --------------------------------------------------
#     m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
#     m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
#     m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

#     #---- Variables from hydrolisis model--------------------------------------------------
#     m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
#     m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
#     m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
#     m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
#     m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
#     m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
#     m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylooligomers rate, g/kg s')
#     m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
#     m.r6=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylooligomers to xylose rate, g/kg s')
#     m.r7=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Acetic acid production rate, g/kg s')
#     m.r8=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Enzymes deactivation due to thermal and exposure to ethanol, g/kg s')

#     #---- main variables -------------------------------------------------------------
#     def _C_init(m,t,j):
#         return m.C0[j]
#     m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
#     m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
#     m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

#     # ---------Reaction kinetic expresions for fermentation part -------------------------

#     m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

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
#             return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + conv_param*m.M[t]*m.R[t,j]) 
#     m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

#     if discretization=='collocation':
#         discretizer_t = pe.TransformationFactory('dae.collocation')
#         discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
#     else:
#         discretizer_t = pe.TransformationFactory('dae.finite_difference')
#         discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


#     # ------------------Re definition of feed flow and output flow information---------------------
#     for t in m.t:
#         if t*m.final_time<=10*60*60: # Inoculum phase
#             m.Fin[t]=m.F_liquified_fibers
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=m.C_liquified_fibers[j]
#         elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
#             m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
#         elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
#             m.Fin[t]=0
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=0

#     #------------------- pH modeling (from hydrolysis) ----------------------------------------------
#     m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
#     m.eta_severity=pe.Param(initialize=1, doc='Severity factor') 
    

#     m.j_elect = pe.Set(initialize=['C2H4O2', 'H+', 'C2H3O2-', 'OH-','CO2aq','HCO3-','CO3-2','C4H6O4','C4H5O4-','C4H4O4-2','C3H6O3','C3H5O3-','NaOH','Na+'],doc='components for pH calculations')
#     m.r_elect = pe.Set(initialize=['1','2','3','4','5','6','7','8'],doc='set of reactions for pH calculations')

#     _MW_elect={}
#     _MW_elect['C2H4O2']=60.05
#     _MW_elect['H+']=1.007825032
#     _MW_elect[ 'C2H3O2-']=59.04
#     _MW_elect['OH-']=17.007 
#     _MW_elect['CO2aq']=44.009
#     _MW_elect['HCO3-']=61.017
#     _MW_elect['CO3-2']=60.009
#     _MW_elect['C4H6O4']=118.09
#     _MW_elect['C4H5O4-']=117.08
#     _MW_elect['C4H4O4-2']=116.07
#     _MW_elect['C3H6O3']=90.08
#     _MW_elect['C3H5O3-']=89.07
#     _MW_elect['NaOH']=39.997
#     _MW_elect['Na+']= 22.9897693   
#     m.MW_elect=pe.Param(m.j_elect,initialize=_MW_elect,doc='Molecular weight of electrolytes [g/mol]')

#     _Equil_lhs={}
#     _Equil_lhs['1']=1.63E-5
#     _Equil_lhs['2']=1
#     _Equil_lhs['3']=5.39E-14
#     _Equil_lhs['4']=5.14E-7
#     _Equil_lhs['5']=6.69E-11
#     _Equil_lhs['6']=6.51E-5
#     _Equil_lhs['7']=2.08E-6
#     _Equil_lhs['8']=1.27E-4

#     m.Equil_lhs=pe.Param(m.r_elect,initialize=_Equil_lhs,doc='lhs Equilibrium constants for pH calculations')
#     _Equil_rhs={}
#     _Equil_rhs['1']=1
#     _Equil_rhs['2']=0
#     _Equil_rhs['3']=1
#     _Equil_rhs['4']=1
#     _Equil_rhs['5']=1
#     _Equil_rhs['6']=1
#     _Equil_rhs['7']=1
#     _Equil_rhs['8']=1
#     m.Equil_rhs=pe.Param(m.r_elect,initialize=_Equil_rhs,doc='rhs constants for pH calculations')

#     _coef_elect={}
    
#     _coef_elect['C2H4O2','1']=-1
#     _coef_elect['C2H3O2-','1']=1
#     _coef_elect['H+','1']=1

#     _coef_elect['NaOH','2']=-1
#     _coef_elect['Na+','2']=1
#     _coef_elect['OH-','2']=1

#     _coef_elect['H+','3']=1
#     _coef_elect['OH-','3']=1

#     _coef_elect['CO2aq','4']=-1
#     _coef_elect['HCO3-','4']=1
#     _coef_elect['H+','4']=1

#     _coef_elect['HCO3-','5']=-1
#     _coef_elect['CO3-2','5']=1
#     _coef_elect['H+','5']=1

#     _coef_elect['C4H6O4','6']=-1
#     _coef_elect['C4H5O4-','6']=1
#     _coef_elect['H+','6']=1

#     _coef_elect['C4H5O4-','7']=-1
#     _coef_elect['C4H4O4-2','7']=1
#     _coef_elect['H+','7']=1

#     _coef_elect['C3H6O3','8']=-1
#     _coef_elect['C3H5O3-','8']=1
#     _coef_elect['H+','8']=1

#     m.coef_elect=pe.Param(m.j_elect,m.r_elect,initialize=_coef_elect,default=0,doc='Stoichiometry coefficient of species j in reaction r')

#     _C_elect_init_param={}

#     _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
#     _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
#     _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
#     _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
#     _C_elect_init_param['H+']=0#0.01
#     m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
#     # m.C_elect_init_param.pprint()

#     m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
#     m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
#     m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

#     m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

#     m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

#     def _C_elect_init_constraint(m,t,j):
#         if t==m.t.last():
#             if j=='C2H4O2': #Acetic acid
#                 return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
#             elif j=='C2H3O2-': #Acetate
#                 return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
#             else: #TODO: this model is not rigurous enouugh? 
#                 return m.C_elect_init[t,j]==m.C_elect_init_param[j]
#         else:
#             return pe.Constraint.Skip
#     m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

#     m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


#     def _equilibrium_relationships(m,t,r):
#         if t==m.t.last():
#             # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
#             if r=='4':
#                 return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
#             # For the remaining reactions we only consider the normal equilibrium calculation
#             else:
#                 # If it is only a fordward reaction
#                 if m.Equil_rhs[r]==0:
#                     return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
#                 # If the reaction is an equilibrium reaction
#                 else:
#                     return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
#         else:
#             return pe.Constraint.Skip
#     m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

#     def _elect_balances(m,t,j):
#         if t==m.t.last():
#             if j=='CO2aq':
#                 return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
#             else:
#                 return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
#         else:
#             return pe.Constraint.Skip
#     m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

#     def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
#         return 5.5
#     m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


#     def _pH_definition(m,t):
#         # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
#         # return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
#         return 10**(-m.pH[t])==m.C_elect_equil[m.t.last(),'H+']
#     m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


#     def _eta_pH_init(m,t):  
#         return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
#     m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

#     def _eq_eta_pH(m,t):
#         return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
#     m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

#     def _eta_init(m,t):
#         return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
#     m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

#     def _eq_eta(m,t):
#         return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
#     m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

#     #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

#     def _enzyme_fractions(m,t,e):
#         return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
#     m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

#     def _bounded_free_equilibrium(m,t,e):
#         return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
#     m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

#     def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
#         # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
#         return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
#         # else:
#         #     return pe.Constraint.Skip
#     m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

#     def _bounded_enzyme_concentration(m,t,e):
#         if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
#             return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
#         else:                                           # NOTE: that denominator is Solid concentration. modify if needed
#             return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
#     m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

#     # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
#     def _r1_definition(m,t):
#         K1_r1=  0.005916     # reaction rate constant, kg/(g*s)
#         IC1_r1= 0.02014      # Inhibition of r1 by cellobiose, g/kg
#         IX1_r1= 0.01503       # Inhibition of r1 by xylose, g/kg
#         IG1_r1= 0.10255       # Inhibition of r1 by glucose, g/kg
#         IXO1_r1= 0.0078145          #  Inhibition of r1 by Xyloligomers, g/kg
#         IEth1_r1= 0.15         #  Inhibition of r1 by Ethanol, g/kg
#         return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'XO']/IXO1_r1)+(m.C[t,'Eth']/IEth1_r1))
#     m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

#     def _r2_definition(m,t):
#         K2_r2= 0.0065075              # reaction rate constant, kg/(g*s)
#         IC2_r2=  69.539         # Inhibition of r2 by cellobiose, g/kg
#         IX2_r2= 0.14843          # Inhibition of r2 by xylose, g/kg
#         IG2_r2= 0.067554          # Inhibition of r2 by glucose, g/kg
#         IXO2_r2= 0.059612
#         IEth2_r2=0.15              # NOT GIVEN            
#         # return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'XO']/IXO2_r2)+(m.C[t,'Eth']/IEth2_r2))
#         return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'XO']/IXO2_r2))

#     m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

#     def _r3_definition(m,t):
#         K3_r3= 0.0055227               # reaction rate constant, kg/(g*s)
#         I3_r3= 15.949              #overall inhibition term for r3, g/kg
#         IX3_r3= 210.1911               # Inhibition of r3 by xylose, g/kg
#         IG3_r3= 8.7211             # Inhibition of r3 by glucose, g/kg
#         IXO3_r3=111.6822
#         IEth3_r3=0.15               
#         # return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'XO']/IXO3_r3)+(m.C[t,'Eth']/IEth3_r3))+m.C[t,'C'])
#         return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'XO']/IXO3_r3))+m.C[t,'C'])

#     m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

#     def _r4_definition(m,t):
#         K4_r4=0.0020026
#         IC4_r4=53.4804
#         IX4_r4=233.0874 
#         IG4_r4=2.0899
#         IXO4_r4=113.4492
#         IEth4_r4=0.15
#         # return m.r4[t] == (K4_r4*m.eta[t]* m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'XO']/IXO4_r4)+(m.C[t,'Eth']/IEth4_r4))
#         return m.r4[t] == (K4_r4*m.eta[t]* m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'XO']/IXO4_r4))

#     m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

#     def _r5_definition(m,t):
#         K5_r5=0.0033936            # reaction rate constant, kg/(g*s)
#         IC5_r5= 2.7413         # Inhibition of r4 by cellobiose, g/kg
#         IX5_r5= 271.2334         # Inhibition of r4 by xylose, g/kg 
#         IG5_r5=   4.7951       # Inhibition of r4 by glucose, g/kg
#         IXO5_r5=  83.5479           
#         IEth5_r5=0.15
#         # return m.r5[t] == (K5_r5*m.eta[t]*(m.CebX[t,'3']+m.CebX[t,'4'])*m.C[t,'XS'])/(1+(m.C[t,'C']/IC5_r5)+(m.C[t,'X']/IX5_r5)+(m.C[t,'G']/IG5_r5)+(m.C[t,'XO']/IXO5_r5)+(m.C[t,'Eth']/IEth5_r5))
#         return m.r5[t] == (K5_r5*m.eta[t]*(m.CebX[t,'3']+m.CebX[t,'4'])*m.C[t,'XS'])/(1+(m.C[t,'C']/IC5_r5)+(m.C[t,'X']/IX5_r5)+(m.C[t,'G']/IG5_r5)+(m.C[t,'XO']/IXO5_r5))

#     m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)


#     def _r6_definition(m,t):
#         K6_r6=0.0028228
#         I6_r6=28.2079
#         IC6_r6=46.9663
#         IX6_r6=198.3351 
#         IG6_r6=3.0412
#         IEth6_r6=0.15
#         # return m.r6[t] == (K6_r6*m.eta[t]*m.Cef[t,'4']*m.C[t,'XO'])/(I6_r6*(1+(m.C[t,'C']/IC6_r6)+(m.C[t,'X']/IX6_r6)+(m.C[t,'G']/IG6_r6)+(m.C[t,'Eth']/IEth6_r6))+m.C[t,'XO'])
#         return m.r6[t] == (K6_r6*m.eta[t]*m.Cef[t,'4']*m.C[t,'XO'])/(I6_r6*(1+(m.C[t,'C']/IC6_r6)+(m.C[t,'X']/IX6_r6)+(m.C[t,'G']/IG6_r6))+m.C[t,'XO'])

#     m.r6_definition=pe.Constraint(m.t,rule=_r6_definition)


#     def _r7_definition(m,t):
#         Beta_r7=0.7     # acetic acid to xylose ratio. Not given
#         return m.r7[t] ==Beta_r7*(m.r4[t]+m.r5[t]) 
#     m.r7_definition=pe.Constraint(m.t,rule=_r7_definition)


#     def _r8_definition(m,t):
#         K8_r8= 2.5E-7  # Enzyme deactivation reaction constant kg/(g*s)
#         return m.r8[t] == K8_r8*(m.C[t,'E']**2)
#     m.r8_definition=pe.Constraint(m.t,rule=_r8_definition)


#     # --------------Definition of fermentation kinetic expresions---------------------------
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
#             # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
#             return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0  )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
#         elif j=='X':

#             # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
#             # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
#             # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
#             # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
#             # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
#             # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
#             # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
#             # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
#             # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
#             return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

#         elif j=='F':
#             return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
#         elif j=='HMF':
#             return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
#         elif j=='ACT':
#             return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
#         elif j =='Cell':
#             return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
#         else:
#             return pe.Constraint.Skip
#     m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

#     #---------------------- Definition of reaction rates------------------------------------------------
#     def _R_definition(m,t,j):

#         # Components from hydrolisis
#         if j=='CS':              
#             return m.R[t,j] == -m.r1[t]-m.r2[t] 
#         elif j=='XS':
#             return m.R[t,j] == -m.r4[t]-m.r5[t] 
#         elif j=='LS':
#             return m.R[t,j] == 0 
#         elif j=='C':
#             return m.R[t,j] == m.r1[t]-m.r3[t]    
#         elif j=='XO':
#             return m.R[t,j] == m.r4[t]-m.r6[t] 
#         elif j=='G':
#             return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      
#         elif j=='X':
#             return m.R[t,j] == m.r5[t]+m.r6[t]-m.q[t,'X'] 
#         elif j=='F':
#             return m.R[t,j] == 0 - m.q[t,'F']    
#         elif j=='E':
#             return m.R[t,j] == -m.r8[t] 
#         elif j=='AC': #Acetic acid
#             return m.R[t,j] == m.r7[t]  
        
#         # New components included in fermentation
#         elif j=='Eth':
#             return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
#         elif j=='HMF':
#             return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
#         elif j=='ACT': #Acetate (or acetyl)
#             return m.R[t,j] ==-m.r7[t]+m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
#         elif j=='CO2':
#             return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
#         elif j=='Cell':
#             return m.R[t,j] == m.q[t,'Cell']
#         else:
#             return m.R[t,j] == 0
#         # elif j=='O':
#         #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
#     m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

#     #-------objective function--------------------------------------------

#     m.obj = pe.Objective(expr=1)

#     return m

# def build_fermentation_convergence_test8(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

#     # ------------pyomo model------------------------------------------------
#     m = pe.ConcreteModel(name='fermentation_model')
#     # ------------shared scalars with hydrolisis model ----------------------
#     m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
#     m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
#     m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
#     m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
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
    
#     m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','XO']) #Cell is cell biomass, ACT is acetate, XO is Xyloligomers
#     # enzime types
#     m.e = pe.Set(initialize=['1','2','3','4']) #NOTE: Enzyme type 4 was included at this stage
    
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
#     m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
#     m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

#     m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
#     m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
#     m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
#     # ----- Enzymatic hydrolisis parameters-----------------------

#     # parameters for enzyme balance
#     _alpha_enzymes={}
#     _alpha_enzymes['1']=0.25
#     _alpha_enzymes['2']=0.25
#     _alpha_enzymes['3']=0.25
#     _alpha_enzymes['4']=0.25
#     m.alpha_enzymes=pe.Param(m.e,initialize=_alpha_enzymes,doc='Fraction of each enzyme type (between 0 and 1)')

#     _max_ads_enz={}
#     _max_ads_enz['1']=0.016042
#     _max_ads_enz['2']=1.5E-5
#     _max_ads_enz['3']=0.38978
#     _max_ads_enz['4']=0.51178
#     m.max_ads_enz=pe.Param(m.e,initialize=_max_ads_enz,doc='Maximum adsorbed enzymes [-]')

#     _k_ads={}
#     _k_ads['1']=1.0444
#     _k_ads['2']=0.056976
#     _k_ads['3']=0.37844
#     _k_ads['4']=0.093253
#     m.k_ads=pe.Param(m.e,initialize=_k_ads,doc='Adsorption constant [-]')



#     #----- Input feed streams properties--------------------------

#     m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

#     m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
#     _C_C5liquid={}
#     _C_C5liquid['CS']=1.2
#     _C_C5liquid['XS']=0.5
#     _C_C5liquid['LS']=0.7
#     _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
#     _C_C5liquid['XO']=1.2
#     _C_C5liquid['G']=10
#     _C_C5liquid['X']=29.7
#     _C_C5liquid['F']=0.5
#     _C_C5liquid['E']=0
#     _C_C5liquid['AC']=4.1 # this may be the mixture of acids
#     _C_C5liquid['Cell']=0 # Same as yeast (?)
#     _C_C5liquid['Eth']=0
#     _C_C5liquid['CO2']=0
#     _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
#     _C_C5liquid['HMF']=0.3 
#     m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

#     _C_liquified_fibers={}
#     _C_liquified_fibers['CS']=50
#     _C_liquified_fibers['XS']=1
#     _C_liquified_fibers['LS']=78
#     _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
#     _C_liquified_fibers['XO']=5.8
#     _C_liquified_fibers['G']=98
#     _C_liquified_fibers['X']=59
#     _C_liquified_fibers['F']=0.2
#     _C_liquified_fibers['E']=4.9
#     _C_liquified_fibers['AC']=16 # this may be the mixture of acids
#     _C_liquified_fibers['Cell']=0 # Same as yeast (?)
#     _C_liquified_fibers['Eth']=0
#     _C_liquified_fibers['CO2']=0
#     _C_liquified_fibers['ACT']=0.1
#     _C_liquified_fibers['HMF']= 0.1
#     m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
#     #----- Initical conditions  ----------------------------------


#     m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
#     m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
#     m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
#     m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

#     def _C0(m,j):
#         if j=='Cell':
#             return (1000*m.M0_yeast)/(m.M0)
#         else:
#             return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
#     m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
#     #----- Maximum reactor hold up------------------------------------------------
#     m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

#     # ----- Feed parameters --------------------------------------------------
#     m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
#     m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
#     m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

#     #---- Variables from hydrolisis model--------------------------------------------------
#     m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg',bounds=(0, 1000)) #bounds=(0, 10000))
#     m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg',bounds=(0, 1000)) #bounds=(0, 10000))
#     m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg',bounds=(0, 1000)) #bounds=(0, 10000))
#     m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg',bounds=(0, 1000))
#     m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg',bounds=(0, 1000))
#     m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
#     m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
#     m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
#     m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylooligomers rate, g/kg s')
#     m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
#     m.r6=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylooligomers to xylose rate, g/kg s')
#     m.r7=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Acetic acid production rate, g/kg s')
#     m.r8=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Enzymes deactivation due to thermal and exposure to ethanol, g/kg s')

#     #---- main variables -------------------------------------------------------------
#     def _C_init(m,t,j):
#         return m.C0[j]
#     m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg',bounds=(0, 1000)) #bounds=(0, 10000))
#     m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
#     m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

#     # ---------Reaction kinetic expresions for fermentation part -------------------------

#     m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

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
#             return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
#     m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

#     if discretization=='collocation':
#         discretizer_t = pe.TransformationFactory('dae.collocation')
#         discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
#     else:
#         discretizer_t = pe.TransformationFactory('dae.finite_difference')
#         discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


#     # ------------------Re definition of feed flow and output flow information---------------------
#     for t in m.t:
#         if t*m.final_time<=10*60*60: # Inoculum phase
#             m.Fin[t]=m.F_liquified_fibers
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=m.C_liquified_fibers[j]
#         elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
#             m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
#         elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
#             m.Fin[t]=0
#             m.Fout[t]=0
#             for j in m.j:
#                 m.Cin[t,j]=0

#     #------------------- pH modeling (from hydrolysis) ----------------------------------------------
#     m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
#     m.eta_severity=pe.Param(initialize=1, doc='Severity factor') 
    

#     # m.j_elect = pe.Set(initialize=['C2H4O2', 'H+', 'C2H3O2-', 'OH-','CO2aq','HCO3-','CO3-2','C4H6O4','C4H5O4-','C4H4O4-2','C3H6O3','C3H5O3-','NaOH','Na+'],doc='components for pH calculations')
#     # m.r_elect = pe.Set(initialize=['1','2','3','4','5','6','7','8'],doc='set of reactions for pH calculations')

#     # _MW_elect={}
#     # _MW_elect['C2H4O2']=60.05
#     # _MW_elect['H+']=1.007825032
#     # _MW_elect[ 'C2H3O2-']=59.04
#     # _MW_elect['OH-']=17.007 
#     # _MW_elect['CO2aq']=44.009
#     # _MW_elect['HCO3-']=61.017
#     # _MW_elect['CO3-2']=60.009
#     # _MW_elect['C4H6O4']=118.09
#     # _MW_elect['C4H5O4-']=117.08
#     # _MW_elect['C4H4O4-2']=116.07
#     # _MW_elect['C3H6O3']=90.08
#     # _MW_elect['C3H5O3-']=89.07
#     # _MW_elect['NaOH']=39.997
#     # _MW_elect['Na+']= 22.9897693   
#     # m.MW_elect=pe.Param(m.j_elect,initialize=_MW_elect,doc='Molecular weight of electrolytes [g/mol]')

#     # _Equil_lhs={}
#     # _Equil_lhs['1']=1.63E-5
#     # _Equil_lhs['2']=1
#     # _Equil_lhs['3']=5.39E-14
#     # _Equil_lhs['4']=5.14E-7
#     # _Equil_lhs['5']=6.69E-11
#     # _Equil_lhs['6']=6.51E-5
#     # _Equil_lhs['7']=2.08E-6
#     # _Equil_lhs['8']=1.27E-4

#     # m.Equil_lhs=pe.Param(m.r_elect,initialize=_Equil_lhs,doc='lhs Equilibrium constants for pH calculations')
#     # _Equil_rhs={}
#     # _Equil_rhs['1']=1
#     # _Equil_rhs['2']=0
#     # _Equil_rhs['3']=1
#     # _Equil_rhs['4']=1
#     # _Equil_rhs['5']=1
#     # _Equil_rhs['6']=1
#     # _Equil_rhs['7']=1
#     # _Equil_rhs['8']=1
#     # m.Equil_rhs=pe.Param(m.r_elect,initialize=_Equil_rhs,doc='rhs constants for pH calculations')

#     # _coef_elect={}
    
#     # _coef_elect['C2H4O2','1']=-1
#     # _coef_elect['C2H3O2-','1']=1
#     # _coef_elect['H+','1']=1

#     # _coef_elect['NaOH','2']=-1
#     # _coef_elect['Na+','2']=1
#     # _coef_elect['OH-','2']=1

#     # _coef_elect['H+','3']=1
#     # _coef_elect['OH-','3']=1

#     # _coef_elect['CO2aq','4']=-1
#     # _coef_elect['HCO3-','4']=1
#     # _coef_elect['H+','4']=1

#     # _coef_elect['HCO3-','5']=-1
#     # _coef_elect['CO3-2','5']=1
#     # _coef_elect['H+','5']=1

#     # _coef_elect['C4H6O4','6']=-1
#     # _coef_elect['C4H5O4-','6']=1
#     # _coef_elect['H+','6']=1

#     # _coef_elect['C4H5O4-','7']=-1
#     # _coef_elect['C4H4O4-2','7']=1
#     # _coef_elect['H+','7']=1

#     # _coef_elect['C3H6O3','8']=-1
#     # _coef_elect['C3H5O3-','8']=1
#     # _coef_elect['H+','8']=1

#     # m.coef_elect=pe.Param(m.j_elect,m.r_elect,initialize=_coef_elect,default=0,doc='Stoichiometry coefficient of species j in reaction r')

#     # _C_elect_init_param={}

#     # _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
#     # _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
#     # _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
#     # _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
#     # _C_elect_init_param['H+']=0#0.01
#     # m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
#     # # m.C_elect_init_param.pprint()

#     # m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
#     # m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
#     # m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

#     # m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

#     # m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

#     # def _C_elect_init_constraint(m,t,j):
#     #     if t==m.t.last():
#     #         if j=='C2H4O2': #Acetic acid
#     #             return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
#     #         elif j=='C2H3O2-': #Acetate
#     #             return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
#     #         else: #TODO: this model is not rigurous enouugh? 
#     #             return m.C_elect_init[t,j]==m.C_elect_init_param[j]
#     #     else:
#     #         return pe.Constraint.Skip
#     # m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

#     # m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


#     # def _equilibrium_relationships(m,t,r):
#     #     if t==m.t.last():
#     #         # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
#     #         if r=='4':
#     #             return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
#     #         # For the remaining reactions we only consider the normal equilibrium calculation
#     #         else:
#     #             # If it is only a fordward reaction
#     #             if m.Equil_rhs[r]==0:
#     #                 return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
#     #             # If the reaction is an equilibrium reaction
#     #             else:
#     #                 return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
#     #     else:
#     #         return pe.Constraint.Skip
#     # m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

#     # def _elect_balances(m,t,j):
#     #     if t==m.t.last():
#     #         if j=='CO2aq':
#     #             return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
#     #         else:
#     #             return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
#     #     else:
#     #         return pe.Constraint.Skip
#     # m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

#     def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
#         return 5.5
#     m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


#     def _pH_definition(m,t):
#         # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
#         # return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
#         # return 10**(-m.pH[t])==m.C_elect_equil[m.t.last(),'H+']
#         return m.pH[t]==5.5
#     m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


#     def _eta_pH_init(m,t):  
#         return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
#     m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

#     def _eq_eta_pH(m,t):
#         return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
#     m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

#     def _eta_init(m,t):
#         return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
#     m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

#     def _eq_eta(m,t):
#         return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
#     m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

#     #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

#     def _enzyme_fractions(m,t,e):
#         return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
#     m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

#     def _bounded_free_equilibrium(m,t,e):
#         return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
#     m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

#     def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
#         # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
#         return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
#         # else:
#         #     return pe.Constraint.Skip
#     m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

#     def _bounded_enzyme_concentration(m,t,e):
#         if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
#             return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
#         else:                                           # NOTE: that denominator is Solid concentration. modify if needed
#             return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
#     m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

#     # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
#     def _r1_definition(m,t):
#         K1_r1=  0.005916     # reaction rate constant, kg/(g*s)
#         IC1_r1= 0.02014      # Inhibition of r1 by cellobiose, g/kg
#         IX1_r1= 0.01503       # Inhibition of r1 by xylose, g/kg
#         IG1_r1= 0.10255       # Inhibition of r1 by glucose, g/kg
#         IXO1_r1= 0.0078145          #  Inhibition of r1 by Xyloligomers, g/kg
#         IEth1_r1= 0.15         #  Inhibition of r1 by Ethanol, g/kg
#         return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'XO']/IXO1_r1)+(m.C[t,'Eth']/IEth1_r1))
#     m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

#     def _r2_definition(m,t):
#         K2_r2= 0.0065075              # reaction rate constant, kg/(g*s)
#         IC2_r2=  69.539         # Inhibition of r2 by cellobiose, g/kg
#         IX2_r2= 0.14843          # Inhibition of r2 by xylose, g/kg
#         IG2_r2= 0.067554          # Inhibition of r2 by glucose, g/kg
#         IXO2_r2= 0.059612
#         IEth2_r2=0.15              # NOT GIVEN            
#         # return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'XO']/IXO2_r2)+(m.C[t,'Eth']/IEth2_r2))
#         return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'XO']/IXO2_r2))

#     m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

#     def _r3_definition(m,t):
#         K3_r3= 0.0055227               # reaction rate constant, kg/(g*s)
#         I3_r3= 15.949              #overall inhibition term for r3, g/kg
#         IX3_r3= 210.1911               # Inhibition of r3 by xylose, g/kg
#         IG3_r3= 8.7211             # Inhibition of r3 by glucose, g/kg
#         IXO3_r3=111.6822
#         IEth3_r3=0.15               
#         return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'XO']/IXO3_r3)+(m.C[t,'Eth']/IEth3_r3))+m.C[t,'C'])
#         # return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'XO']/IXO3_r3))+m.C[t,'C'])

#     m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

#     def _r4_definition(m,t):
#         K4_r4=0.0020026
#         IC4_r4=53.4804
#         IX4_r4=233.0874 
#         IG4_r4=2.0899
#         IXO4_r4=113.4492
#         IEth4_r4=0.15
#         return m.r4[t] == (K4_r4*m.eta[t]* m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'XO']/IXO4_r4)+(m.C[t,'Eth']/IEth4_r4))
#         # return m.r4[t] == (K4_r4*m.eta[t]* m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'XO']/IXO4_r4))

#     m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

#     def _r5_definition(m,t):
#         K5_r5=0.0033936            # reaction rate constant, kg/(g*s)
#         IC5_r5= 2.7413         # Inhibition of r4 by cellobiose, g/kg
#         IX5_r5= 271.2334         # Inhibition of r4 by xylose, g/kg 
#         IG5_r5=   4.7951       # Inhibition of r4 by glucose, g/kg
#         IXO5_r5=  83.5479           
#         IEth5_r5=0.15
#         return m.r5[t] == (K5_r5*m.eta[t]*(m.CebX[t,'3']+m.CebX[t,'4'])*m.C[t,'XS'])/(1+(m.C[t,'C']/IC5_r5)+(m.C[t,'X']/IX5_r5)+(m.C[t,'G']/IG5_r5)+(m.C[t,'XO']/IXO5_r5)+(m.C[t,'Eth']/IEth5_r5))
#         # return m.r5[t] == (K5_r5*m.eta[t]*(m.CebX[t,'3']+m.CebX[t,'4'])*m.C[t,'XS'])/(1+(m.C[t,'C']/IC5_r5)+(m.C[t,'X']/IX5_r5)+(m.C[t,'G']/IG5_r5)+(m.C[t,'XO']/IXO5_r5))

#     m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)


#     def _r6_definition(m,t):
#         K6_r6=0.0028228
#         I6_r6=28.2079
#         IC6_r6=46.9663
#         IX6_r6=198.3351 
#         IG6_r6=3.0412
#         IEth6_r6=0.15
#         return m.r6[t] == (K6_r6*m.eta[t]*m.Cef[t,'4']*m.C[t,'XO'])/(I6_r6*(1+(m.C[t,'C']/IC6_r6)+(m.C[t,'X']/IX6_r6)+(m.C[t,'G']/IG6_r6)+(m.C[t,'Eth']/IEth6_r6))+m.C[t,'XO'])
#         # return m.r6[t] == (K6_r6*m.eta[t]*m.Cef[t,'4']*m.C[t,'XO'])/(I6_r6*(1+(m.C[t,'C']/IC6_r6)+(m.C[t,'X']/IX6_r6)+(m.C[t,'G']/IG6_r6))+m.C[t,'XO'])

#     m.r6_definition=pe.Constraint(m.t,rule=_r6_definition)


#     def _r7_definition(m,t):
#         Beta_r7=0.5     # acetic acid to xylose ratio. Not given
#         return m.r7[t] ==Beta_r7*(m.r4[t]+m.r5[t]) 
#     m.r7_definition=pe.Constraint(m.t,rule=_r7_definition)


#     def _r8_definition(m,t):
#         K8_r8= 2.5E-7  # Enzyme deactivation reaction constant kg/(g*s)
#         return m.r8[t] == K8_r8*(m.C[t,'E']**2)
#     m.r8_definition=pe.Constraint(m.t,rule=_r8_definition)


#     # --------------Definition of fermentation kinetic expresions---------------------------
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
#             # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
#             return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*1  )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
#         elif j=='X':

#             # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
#             # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
#             # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
#             # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
#             # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
#             # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
#             # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
#             # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
#             # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
#             return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

#         elif j=='F':
#             return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
#         elif j=='HMF':
#             return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
#         elif j=='ACT':
#             return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
#         elif j =='Cell':
#             return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
#         else:
#             return pe.Constraint.Skip
#     m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

#     #---------------------- Definition of reaction rates------------------------------------------------
#     def _R_definition(m,t,j):

#         # Components from hydrolisis
#         if j=='CS':              
#             return m.R[t,j] == -m.r1[t]-m.r2[t] 
#         elif j=='XS':
#             return m.R[t,j] == -m.r4[t]-m.r5[t] 
#         elif j=='LS':
#             return m.R[t,j] == 0 
#         elif j=='C':
#             return m.R[t,j] == m.r1[t]-m.r3[t]    
#         elif j=='XO':
#             return m.R[t,j] == m.r4[t]-m.r6[t] 
#         elif j=='G':
#             return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      
#         elif j=='X':
#             return m.R[t,j] == m.r5[t]+m.r6[t]-m.q[t,'X'] 
#         elif j=='F':
#             return m.R[t,j] == 0 - m.q[t,'F']    
#         elif j=='E':
#             return m.R[t,j] == -m.r8[t] 
#         elif j=='AC': #Acetic acid
#             return m.R[t,j] == m.r7[t]  
        
#         # New components included in fermentation
#         elif j=='Eth':
#             return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
#         elif j=='HMF':
#             return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
#         elif j=='ACT': #Acetate (or acetyl)
#             return m.R[t,j] ==-m.r7[t]+m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
#         elif j=='CO2':
#             return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
#         elif j=='Cell':
#             return m.R[t,j] == m.q[t,'Cell']
#         else:
#             return m.R[t,j] == 0
#         # elif j=='O':
#         #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
#     m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

#     #-------objective function--------------------------------------------

#     m.obj = pe.Objective(expr=1)

#     return m

def build_fermentation_convergence_testFinal(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=6.6* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=_C_elect_init_param,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        IEth1_r1=0.15
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1)+(m.C[t,'Eth']/IEth1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0053#0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.97#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.34         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            if t>=0.8:
                return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            else:
                return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.007  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m
#IMPROVED PH modeling
def build_fermentation_convergence_testFinal2(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=1) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=1.06325,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=1E+4,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    _C_C5liquid['Base']=0
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    _C_liquified_fibers['Base']=8.6 
    
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        elif j=='NaOH': #Base
            return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,bounds=(0,30),doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        if t==m.t.first():
            return m.pH[t]==m.pH[m.t.next(t)]
        else:
            return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        IEth1_r1=0.15
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1)+(m.C[t,'Eth']/IEth1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0053#0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.97#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.34         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            if t>=0.8:
                return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            else:
                return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.007  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m

def build_fermentation_convergence_testFinal3(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=1) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=0.1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+2,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=0.5,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=5.25,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=0.006,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    _C_C5liquid['Base']=0
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    _C_liquified_fibers['Base']=8.6 
    
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        elif j=='NaOH': #Base
            return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,bounds=(0,30),doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        if t==m.t.first():
            return m.pH[t]==m.pH[m.t.next(t)]
        else:
            return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        IEth1_r1=0.15
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1)+(m.C[t,'Eth']/IEth1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0053#0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.97#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.34         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*(m.K0X*pe.exp(-(((m.pH[t]-m.K1X)**2)/(2*(m.K2X**2))))   )  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # if t>=0.8:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # else:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.007  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m

def build_fermentation_convergence_testFinal4(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=1) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    m.K0G=pe.Param(initialize=0.1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=1E+7,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Param(initialize=1E+2,doc='Parameter for pH dependency in glucose rate of fermentation model')

    m.K0X=pe.Param(initialize=0.5,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=5.25,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=0.0065,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    _C_C5liquid['Base']=0
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    _C_liquified_fibers['Base']=8.6 
    
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        elif j=='NaOH': #Base
            return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,bounds=(0,30),doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        if t==m.t.first():
            return m.pH[t]==m.pH[m.t.next(t)]
        else:
            return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        IEth1_r1=0.15
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1)+(m.C[t,'Eth']/IEth1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0026#0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.34         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*(m.K0X*pe.exp(-(((m.pH[t]-m.K1X)**2)/(2*(m.K2X**2))))   )  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # if t>=0.8:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # else:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.007  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X#m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------

    m.obj = pe.Objective(expr=1)

    return m



def build_fermentation_convergence_testFinal5_adjust(discretization: str='collocation',n_f_elements_t: int=10, conv_param: float=1,data: dict={}) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    # m.K0G=pe.Var(initialize=0.1,within=pe.NonNegativeReals,doc='Parameter for pH dependency in glucose rate of fermentation model')
    # m.K1G=pe.Var(initialize=1E+7,within=pe.NonNegativeReals,doc='Parameter for pH dependency in glucose rate of fermentation model')
    # m.K2G=pe.Var(initialize=1E+2,within=pe.NonNegativeReals,doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0G=pe.Var(initialize=0.1,within=pe.NonNegativeReals,bounds=(0,1),doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Var(initialize=5,within=pe.NonNegativeReals,bounds=(4,6),doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Var(initialize=0.1,within=pe.NonNegativeReals,bounds=(1e-9,10),doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0X=pe.Var(initialize=0.5,within=pe.NonNegativeReals,bounds=(0,1),doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Var(initialize=5,within=pe.NonNegativeReals,bounds=(5,6),doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Var(initialize=0.1,within=pe.NonNegativeReals,bounds=(1e-9,10),doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    _C_C5liquid['Base']=0
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    _C_liquified_fibers['Base']=8.6 
    
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        elif j=='NaOH': #Base
            return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,bounds=(0,30),doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        if t==m.t.first():
            return m.pH[t]==m.pH[m.t.next(t)]
        else:
            return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        IEth1_r1=0.15
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1)+(m.C[t,'Eth']/IEth1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0023 #0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.34         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G*pe.exp(-(((m.pH[t]-m.K1G)**2)/(2*(m.K2G**2)))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )

            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*(m.K0X*pe.exp(-(((m.pH[t]-m.K1X)**2)/(2*(m.K2X**2))))   )  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # if t>=0.8:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # else:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.007  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------


    def _obj(m):
        # return sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        w={}
        w['Eth']=1
        w['G']=1
        w['X']=1
        return sum(sum( w[j]*((m.C[t,j]-data[j][m.t.ord(t)-1])**2)    for j in ['G','X','Eth']) for t in m.t) 
    m.obj = pe.Objective(rule=_obj)

    return m

def build_fermentation_convergence_testFinal6(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=190*(60)*(60),doc='final simulation time [s]')  # NOTE: this is the time considered in one of the simulation experiments by prunescu.
    m.Boltzmann=pe.Param(initialize=1.380649E-23, doc='[J/K]')
    m.Avogadro=pe.Param(initialize= 6.02214076E+23 ,doc='[1/mol]')
    m.T=pe.Param(initialize=35+273.15, doc='Optimal enzymatic activity temperature [K]')
    m.rho_soluble=pe.Param(initialize=1.05*1000 , doc='Soluble fraction density [kg/ m^3]') #TODO: soluble liquid fraction assumed to have constant density of "Fiber mash density" in Table E2, page 198. Express as correlation!
    m.rho_soluble_kg_L=pe.Param(initialize=m.rho_soluble/1000, doc='Soluble fraction density [kg/ L]') 
    m.MW_soluble=pe.Param(initialize= 0.180156 ,doc='Molecular mass of soluble components in liquid fraction [kg/mol]') #TODO: same as rho_Soluble. Currently using molecular weight of glucose
    #------------ new scalars -----------------------------------------------


    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, 1))   # NOTE: Dimentionless form so that I can optimize time in the future. 

    # chemical species
    # m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS','G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O']) #TODO: this is the list of components from the pretreatment model
    # m.j = pe.Set(initialize=['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC'])  #NOTE: In pretreatment model AC is organic acids, here it is acetic acid, given that according to the pretreatment article "Organic acids, mostly represented by acetic acid"
                            # Solid part of the slurry       # Liquid part of the slurry 
    
    m.j = pe.Set(initialize=['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']) #Cell is cell biomass, ACT is acetate
    # enzime types
    m.e = pe.Set(initialize=['1','2','3']) #NOTE: Enzyme type 4 was not included because, according to Prunescu's hydrolisis paper, their concentration is negligible
    
    # ---------parameters----------------------------------------------------

    m.Y_CO2_G=pe.Param(initialize=0.47,doc='CO2 production from glucose uptake [kg/kg]')
    m.Y_CO2_X=pe.Param(initialize=0.4,doc='CO2 production from xylose uptake [kg/kg]')
    m.KI_F_S=pe.Param(initialize=0.05,doc='Furfural uptake self inhibition constant [g/kg]')
    m.KI_F_G=pe.Param(initialize=0.75,doc='Glucose inhibition on furfural uptake [g/kg]')
    m.KI_HMF_F=pe.Param(initialize=0.25,doc='Furfural inhibition on 5-HMF uptake [g/kg]')
    m.KI_F_X=pe.Param(initialize=0.35,doc='Xylose inhibition on furfural uptake [g/kg]')
    m.qmax_F=pe.Param(initialize=4.6706E-5,doc='Maximum furfural uptake [1/s]')
    m.KIP_G=pe.Param(initialize=4890,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.KSP_G=pe.Param(initialize=1.342,doc='Glucose uptake self inhibition parameter [g/kg]')
    m.PMP_G=pe.Param(initialize=103,doc='Ethanol inhibition in glucose uptake [g/kg]')
    m.gamma_G=pe.Param(initialize=1.42,doc='Ethanol inhibition in glucose uptake [-]')
    m.Y_Eth_G=pe.Param(initialize=0.47,doc='Ethanol production from glucoe uptake [kg/kg]')
    m.Y_Cell_G=pe.Param(initialize=0.115,doc='Biomass growth on glucose [kg/kg]')
    m.m_G=pe.Param(initialize=2.6944E-5,doc='Maintenance coefficient for biomass growth on glucose [1/s]')
    m.qmax_G=pe.Param(initialize=0.000318,doc='Maximum glucose uptake rate [1/s]')
    m.KIP_X=pe.Param(initialize=81.3,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.KSP_X=pe.Param(initialize=3.4,doc='Xylose uptake self inhibition parameter [g/kg]')
    m.PMP_X=pe.Param(initialize=100.2,doc='Ethanol inhibition on xylose uptake [g/kg]')
    m.gamma_X=pe.Param(initialize=0.608,doc='Ethanol inhibition on xylose uptake[-]')
    m.Y_Eth_X=pe.Param(initialize=0.4,doc='Ethanol production from xylose uptake [kg/kg]')
    m.Y_Cell_X=pe.Param(initialize=0.162,doc='Biomass growth on xylose [kg/kg]')
    m.m_X=pe.Param(initialize=1.8611E-5,doc='Maintenance coefficient for biomass growth on xylose [1/s]')
    m.qmax_X=pe.Param(initialize=0.00083444,doc='Maximum xylose uptake rate [1/s]')
    m.KIP_ACT=pe.Param(initialize=2.5,doc='Acetate uptake self inhibition [g/kg]') #KACS in manuscript
    m.KI_ACT_G=pe.Param(initialize=2.74,doc='Acetate inhibition on glucose uptake [g/kg]')
    m.KI_ACT_X=pe.Param(initialize=0.2,doc='Acetate inhibition on xylose uptake [g/kg]')
    m.Y_ACT_HMF=pe.Param(initialize=0.23392,doc='Acetate production from 5HMF uptake [kg/kg]')
    m.Y_CO2_HMF=pe.Param(initialize=0.1,doc='CO2 production from 5HMF uptake [kg/kg]') #YCO2S in table
    m.qmax_ATC=pe.Param(initialize=1.2292E-5,doc='Maximum acetate uptake rate [1/s]')
    m.KIP_HMF=pe.Param(initialize=0.5,doc='5HMF uptake self inhibition [g/kg]') #KHMF_S in table
    m.KI_HMF_G=pe.Param(initialize=2,doc='5HMF inhibition on glucose uptake [g/kg]')
    m.KI_HMF_X=pe.Param(initialize=10,doc='5HMF inhibition on xylose uptake [g/kg]')
    m.qmax_HMF=pe.Param(initialize=8.7576E-5,doc='Maximum 5HMF uptake rate [1/s]')

    # TODO: NOT PROVIDED!!
    # m.K0G=pe.Var(initialize=0.1,within=pe.NonNegativeReals,doc='Parameter for pH dependency in glucose rate of fermentation model')
    # m.K1G=pe.Var(initialize=1E+7,within=pe.NonNegativeReals,doc='Parameter for pH dependency in glucose rate of fermentation model')
    # m.K2G=pe.Var(initialize=1E+2,within=pe.NonNegativeReals,doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=4.846165818555403,doc='Parameter for pH dependency in glucose rate of fermentation model') 
    m.K2G=pe.Param(initialize=0.220507978244498,doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0X=pe.Param(initialize=1.0,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=5.263611373537922,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=0.010727558852723,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    # ----- Enzymatic hydrolisis parameters-----------------------

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



    #----- Input feed streams properties--------------------------

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')

    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')
    _C_C5liquid={}
    _C_C5liquid['CS']=1.2
    _C_C5liquid['XS']=0.5
    _C_C5liquid['LS']=0.7
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=10
    _C_C5liquid['X']=29.7
    _C_C5liquid['F']=0.5
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=4.1 # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=0.2/2 # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=0.3 
    _C_C5liquid['Base']=0
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=50
    _C_liquified_fibers['XS']=1
    _C_liquified_fibers['LS']=78
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=98
    _C_liquified_fibers['X']=59
    _C_liquified_fibers['F']=0.2
    _C_liquified_fibers['E']=4.9
    _C_liquified_fibers['AC']=16 # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=0.1
    _C_liquified_fibers['HMF']= 0.1
    _C_liquified_fibers['Base']=8.6 
    
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')
    #----- Initical conditions  ----------------------------------


    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2300,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Param(initialize=m.M0_fibers+m.M0_water+m.M0_yeast,doc='Initial hold up in the reactor [kg]')

    def _C0(m,j):
        if j=='Cell':
            return (1000*m.M0_yeast)/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Param(m.t,initialize=0,mutable=True,doc='Feed flow [kg/s]')
    m.Cin=pe.Param(m.t,m.j,initialize=0,mutable=True,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals, doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t] - m.Fout[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
    #   if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Re definition of feed flow and output flow information---------------------
    for t in m.t:
        if t*m.final_time<=10*60*60: # Inoculum phase
            m.Fin[t]=m.F_liquified_fibers
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            m.Fin[t]=m.F_C5liquid + m.F_liquified_fibers             #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j])/(m.F_C5liquid + m.F_liquified_fibers)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            m.Fin[t]=0
            m.Fout[t]=0
            for j in m.j:
                m.Cin[t,j]=0

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
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

    _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    _C_elect_init_param['H+']=0#0.01
    m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # m.C_elect_init_param.pprint()

    m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=0.001,doc='Initial concentration of electrolytes')

    def _C_elect_init_constraint(m,t,j):
        if j=='C2H4O2': #Acetic acid
            return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
        elif j=='C2H3O2-': #Acetate
            return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
        elif j=='NaOH': #Base
            return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
        else: #TODO: this model is not rigurous enouugh? 
            return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    def _equilibrium_relationships(m,t,r):
        # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
        if r=='4':
            return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
        # For the remaining reactions we only consider the normal equilibrium calculation
        else:
            # If it is only a fordward reaction
            if m.Equil_rhs[r]==0:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
            # If the reaction is an equilibrium reaction
            else:
                return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    def _elect_balances(m,t,j):
        if j=='CO2aq':
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
        else:
            return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
        return 5.5
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,bounds=(0,30),doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        if t==m.t.first():
            return m.pH[t]==m.pH[m.t.next(t)]
        else:
            return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH[t])-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH[t]-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eq_eta_pH=pe.Constraint(m.t,rule=_eq_eta_pH) 

    def _eta_init(m,t):
        return m.eta_severity*m.eta_T*pe.value(m.eta_pH[t])
    m.eta=pe.Var(m.t,initialize=_eta_init,bounds=(0,1.1),doc='temperature and pH dependence of reaction rates') 

    def _eq_eta(m,t):
        return m.eta[t]==m.eta_severity*m.eta_T*m.eta_pH[t]
    m.eq_eta=pe.Constraint(m.t,rule=_eq_eta)

    #----------------- ENZYME BALANCES (from hydrolisis)----------------------------------

    def _enzyme_fractions(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m.enzyme_fractions=pe.Constraint(m.t,m.e,rule=_enzyme_fractions)

    def _bounded_free_equilibrium(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m.bounded_free_equilibrium=pe.Constraint(m.t,m.e,rule=_bounded_free_equilibrium)

    def _adsorbed_free_equilibrium(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m.Ceb[t,x,e] for e in m.e)
        
        # if e=='1' or e=='2': #TODO: Check if this is for every enzyme, or just for 1 and 2. I think it should be for every enzyme, because we have all info needed for calculations 
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m.bounded_enzyme_concentration=pe.Constraint(m.t,m.e,rule=_bounded_enzyme_concentration)

    # ------------------- MODELING OF REACTION RATES (from hyfrolisis)-------------------------------------
    def _r1_definition(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        IEth1_r1=0.15
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1)+(m.C[t,'Eth']/IEth1_r1))
    m.r1_definition=pe.Constraint(m.t,rule=_r1_definition)

    def _r2_definition(m,t):
        K2_r2=0.0023 #0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m.r2_definition=pe.Constraint(m.t,rule=_r2_definition)

    def _r3_definition(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m.r3_definition=pe.Constraint(m.t,rule=_r3_definition)

    def _r4_definition(m,t):
        K4_r4=0.0087#0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.34         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m.r4_definition=pe.Constraint(m.t,rule=_r4_definition)

    def _r5_definition(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m.r5_definition=pe.Constraint(m.t,rule=_r5_definition)

    # --------------Definition of fermentation kinetic expresions---------------------------
    def _q_definition(m,t,j):
        if j=='G': 
            # qmaxGpH=(   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )
            # qEthG=(   qmaxGpH*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )
            # IEthG=(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )          
            # IFG=(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )
            # IAG=(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )
            # IHMFG=(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            # qEthGI=qEthG*IEthG*IFG*IAG*IHMFG
            # return m.q[t,j] == (1/m.Y_Eth_G)*qEthGI        
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G*pe.exp(-(((m.pH[t]-m.K1G)**2)/(2*(m.K2G**2)))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G/(1+((10**m.pH[t])/m.K1G)+(m.K2G/(10**m.pH[t]))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )

            # return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*0.1   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
            
        elif j=='X':

            # qmaxXpH=(  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )
            # qEthX=(  qmaxXpH*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )
            # IEthX=(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )
            # IFX=(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )
            # IACX=(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )
            # IHMFX=(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            # qHthXI=qEthX*IEthX*IFX*IACX*IHMFX
            # return m.q[t,j]== (1/m.Y_Eth_X)*qHthXI
            # return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*((m.K0X)/(1+((10**m.pH[t])/(m.K1X))+((m.K2X)/(10**m.pH[t]))))  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*(m.K0X*pe.exp(-(((m.pH[t]-m.K1X)**2)/(2*(m.K2X**2))))   )  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # if t>=0.8:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.5  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

            # else:
            #     return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*0.007  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

        elif j=='F':
            return m.q[t,j]==m.qmax_F*m.C[t,'Cell']*((m.C[t,'F'])/(m.KI_F_S+m.C[t,'F']))
        elif j=='HMF':
            return m.q[t,j]== (m.qmax_HMF*m.C[t,'Cell']*(m.C[t,'HMF']/(m.C[t,'HMF']+m.KIP_HMF)))   *    (m.KI_HMF_F/(m.KI_HMF_F+m.C[t,'F']))
        elif j=='ACT':
            return m.q[t,j]==m.qmax_ATC*m.C[t,'Cell']*(m.C[t,'ACT']/(m.C[t,'ACT']+m.KIP_ACT))
        elif j =='Cell':
            return  m.q[t,j]*(m.C[t,'G']+m.C[t,'X'])==(m.C[t,'G']/(1))*((m.q[t,'G']-m.m_G*m.C[t,'Cell'])*m.Y_Cell_G)+(m.C[t,'X']/(1))*((m.q[t,'X']-m.m_X*m.C[t,'Cell'])*m.Y_Cell_X)
        else:
            return pe.Constraint.Skip
    m.q_definition=pe.Constraint(m.t,m.j, rule=_q_definition)

    #---------------------- Definition of reaction rates------------------------------------------------
    def _R_definition(m,t,j):

        # Components from hydrolisis
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]-m.q[t,'G']      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3),    #Glucose->Ethanol (q[t,G])
        elif j=='X':
            return m.R[t,j] == m.r4[t]-m.q[t,'X'] #Xylan->Xylose (r4)    ,     #Xylose-> Ethanol (q[t,X])
        elif j=='F':
            return m.R[t,j] == 0 - m.q[t,'F']    #Furfural -> Other (q[t,F])
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)   
        
        # New components included in fermentation
        elif j=='Eth':
            return m.R[t,j] == m.q[t,'G']*m.Y_Eth_G+m.q[t,'X']*m.Y_Eth_X    #Glucose->Ethanol (q[t,G]) ,     #Xylose-> Ethanol (q[t,X])
        elif j=='HMF':
            return m.R[t,j] ==-m.q[t,'HMF']      #HMF->Other +  Acetate (q[t,HMF])
        elif j=='ACT':
            return m.R[t,j] ==m.q[t,'HMF']*m.Y_ACT_HMF   -  m.q[t,'ACT']    #HMF->Acetate (m.q[t,'HMF']*m.Y_ACT_HMF)        #Acetate->CO2   +    Other
        elif j=='CO2':
            return m.R[t,j] == m.q[t,'G']*m.Y_CO2_G   +    m.q[t,'X']*m.Y_CO2_X    +     m.q[t,'ACT']*m.Y_CO2_HMF  #NOTE: last term is not clear if it is from HMF or ACT
        elif j=='Cell':
            return m.R[t,j] == m.q[t,'Cell']
        else:
            return m.R[t,j] == 0
        # elif j=='O':
        #     return m.R[t,j] == m.q[t,'F']+m.q[t,'HMF']*(1-m.Y_ACT_HMF)    #Furfural -> Other (q[t,F])    ,     #HMF->Other (m.q[t,'HMF']*(1-m.Y_ACT_HMF))
    m.R_definition=pe.Constraint(m.t,m.j, rule=_R_definition)

    #-------objective function--------------------------------------------
    m.obj = pe.Objective(expr=1)

    return m



if __name__ == '__main__':

    # PARAMETERS
    sim_time=72000 #seconds
    discretization_type='collocation'#'collocation' #'differences'
    finite_elem_x=6 # According to prunescu the first hydrolisis reactor is discretized into 6 cells
    finite_elem_t=5

#     # # 1: SOLUTION OF THE PROBLEM USING A MODEL THAT IGNORES DIFFUSION EFFECTS
#     # m=build_hydrolisis_convergence_tests1(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t) # Simplest version of the model that completely ignores diffusion effects
#     # opt1 = SolverFactory('gams')
#     # results = opt1.solve(m, solver='knitro', tee=False)
#     # generate_initialization(m=m,model_name='validation_hydrolisis_1')

#     # # # PLOT GENERATION
#     # # time=[]
#     # # space=[]
#     # # vec={}

#     # # for x in m.x:
#     # #     space.append(x)

#     # # for t in m.t:
#     # #     time.append(t)
#     # #     for j in m.j:
#     # #         vec[(j,t)]=[]
#     # #         for x in m.x:
#     # #             vec[(j,t)].append(m.C[t,x,j].value)

#     # # # PLOT AT STEADY STATE
#     # # t=m.t.last()
#     # # for j in m.j:
#     # #     if j !='E':
        
#     # #         plt.plot(space,vec[(j,t)],label=j)
#     # #         plt.xlabel('length [m]')
#     # #         plt.ylabel('Concentration [g/kg]')
#     # # plt.legend()
#     # # plt.show()

#     # # 2: SOLUTION OF THE PROBLEM USING A MODEL THAT CONSIDERS DIFFUSION EFFECTS
#     # m=build_hydrolisis_convergence_test2(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t)
#     # m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis_1')
#     # # Initialization of diffusivity coefficient
#     # A_W=2.41E-3 # g/(m*s)
#     # B_W=1774.9 # K
#     # A_G=8.65E-10 # m^2/s
#     # B_G=2502 #K
#     # for t in m.t:
#     #     for x in m.x:
#     #         m.D[t,x].set_value((m.Boltzmann*m.T)/(6*math.pi*   (((3*m.MW_soluble)/(4*math.pi*m.Avogadro*m.rho_soluble))**(1/3))           *     (((A_W*pe.exp(B_W/m.T))          +       (pe.value(m.C[t,x,'G'])*m.rho_soluble*A_G*pe.exp(B_G/m.T))    )*(1/1000))             ))
#     # opt1 = SolverFactory('gams')
#     # results = opt1.solve(m, solver='conopt4', tee=False)
#     # generate_initialization(m=m,model_name='validation_hydrolisis_2')

#     # # # PLOT GENERATION
#     # # time=[]
#     # # space=[]
#     # # vec={}

#     # # for x in m.x:
#     # #     space.append(x)

#     # # for t in m.t:
#     # #     time.append(t)
#     # #     for j in m.j:
#     # #         vec[(j,t)]=[]
#     # #         for x in m.x:
#     # #             vec[(j,t)].append(m.C[t,x,j].value)

#     # # # PLOT AT STEADY STATE
#     # # t=m.t.last()
#     # # for j in m.j:
#     # #     if j !='E':
        
#     # #         plt.plot(space,vec[(j,t)],label=j)
#     # #         plt.xlabel('length [m]')
#     # #         plt.ylabel('Concentration [g/kg]')
#     # # plt.legend()
#     # # plt.show()
#     # # 3: SOLUTION OF THE PROBLEM USING A MODEL THAT CONSIDERS DIFFUSION EFFECTS AND A FIXED PH PROFILE THAT DETERMINES PH EFFICIENCY
#     # m=build_hydrolisis_convergence_test3(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t)
#     # m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis_2')
#     # opt1 = SolverFactory('gams')
#     # results = opt1.solve(m, solver='conopt4', tee=False)
#     # generate_initialization(m=m,model_name='validation_hydrolisis_3')


    


#     # # PLOT GENERATION
#     # time=[]
#     # space=[]
#     # vec={}
#     # slurry_mu={}
#     # solid_fract={}

#     # for x in m.x:
#     #     space.append(x)

#     # for t in m.t:
#     #     time.append(t)
#     #     for j in m.j:
#     #         vec[(j,t)]=[]
#     #         for x in m.x:
#     #             vec[(j,t)].append(m.C[t,x,j].value)

#     # for t in m.t:
#     #     slurry_mu[t]=[]
#     #     solid_fract[t]=[]
#     #     for x in m.x:
#     #         slurry_mu[t].append(m.slurry_mu[t,x].value)
#     #         solid_fract[t].append(100*(1/1000)*m.conv_fact_sol*(m.C[t,x,'CS'].value+m.C[t,x,'XS'].value+m.C[t,x,'LS'].value))
             

#     # # # PLOT AT STEADY STATE
#     # # t=m.t.last()
#     # # for j in m.j:
#     # #     if j !='E':
        
#     # #         plt.plot(space,vec[(j,t)],label=j)
#     # #         plt.xlabel('length [m]')
#     # #         plt.ylabel('Concentration [g/kg]')
#     # # plt.legend()
#     # # plt.show()

#     # # VALIDATION PLOT
#     # colors=['b','g','m','r','k']
#     # t=m.t.last()
#     # contador=-1
#     # for j in m.j:
#     #     if any(j==c for c in ['XS','CS','C','G','X']):
#     #         contador=contador+1
#     #         plt.plot(space,vec[(j,t)],colors[contador],label=j)
#     #         original = pd.read_csv('biorefinery_models/'+j+'_hydrolisis.csv', header=None)
#     #         plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
#     #         plt.xlabel('length [m]')
#     #         plt.ylabel('Concentration [g/kg]')
#     # plt.legend()
#     # plt.show()

#     # # VISCOSITY PLOT
#     # plt.plot(space,slurry_mu[t],'k')
#     # original = pd.read_csv('biorefinery_models/viscosity_hydrolisis.csv', header=None)
#     # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
#     # plt.xlabel('length [m]')
#     # plt.ylabel('Slurry viscosity [g/m s]')
#     # plt.legend()
#     # plt.show()

#     # # SOLID FRACTION
#     # plt.plot(space,solid_fract[t],'k')
#     # # original = pd.read_csv('biorefinery_models/solfrac_hydrolisis.csv', header=None)
#     # # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
#     # plt.xlabel('length [m]')
#     # plt.ylabel('Solid fraction [%]')
#     # plt.legend()
#     # plt.show()


#     # # # PLOTS FOR EVERY COMPONENT AND EVERY POINT IN TIME
#     # # for j in m.j:
#     # #     for t in m.t:
#     # #         plt.plot(space,vec[(j,t)],label=str(t)+' [s]')
#     # #         plt.xlabel('length [m]')
#     # #         plt.ylabel(j+' [g/kg]')
#     # #     plt.legend()
#     # #     plt.show()


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


    # 5: Fermentation
    discretization_type_fer='differences'
    finite_elem_t_fer=50


    # m=build_fermentation_convergence_test1(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_1')


    # step=0.1
    # ending=1

    # count=1
    # for conv_p in np.arange(0,ending+step,step):
    #     count=count+1
    #     print('convergence_param_is ',conv_p)
    #     m=build_fermentation_convergence_test2(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=conv_p)
    #     m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count-1))
    #     opt1 = SolverFactory('gams')
    #     results = opt1.solve(m, solver='conopt4', tee=True)
    #     generate_initialization(m=m,model_name='validation_fermentation_'+str(count))
    # # m.C.pprint()
    # # PLOT GENERATION

    # m=build_fermentation_convergence_test3(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+1))


    count=12
    # m=build_fermentation_convergence_test4(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count+1))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+2))

    # m=build_fermentation_convergence_test5(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count+2))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+3))


    # m=build_fermentation_convergence_testFinal(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count+3))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+4))

    # m=build_fermentation_convergence_testFinal2(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=0)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_17')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='knitro', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+6))

    # m=build_fermentation_convergence_testFinal2(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_18')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_19')

    # m=build_fermentation_convergence_testFinal3(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_19')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_20')

    # m=build_fermentation_convergence_testFinal4(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_20')
    # # opt1 = SolverFactory('gams')
    # # results = opt1.solve(m, solver='conopt4', tee=True)
    # # generate_initialization(m=m,model_name='validation_fermentation_21')

    # time=[]
    # vec={}
    # pH=[]

    # for t in m.t:
    #     time.append(t*m.final_time*(1/60)*(1/60))
    #     pH.append(m.pH[t].value)

    # for j in m.j:
    #     vec[j]=[]
    #     for t in m.t:
    #         vec[j].append(m.C[t,j].value)

    # # for j in m.j:

    # #     plt.plot(time,vec[j],label=j)
    # #     # original = pd.read_csv('biorefinery_models/ph_hydrolisis.csv', header=None)
    # #     # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--k')
    # #     plt.xlabel('time [-]')
    # #     plt.ylabel('Concentration [g/kg]')
    # #     plt.legend()
    # # plt.show()
            
    # colors=['b','g','m','r','k','y','c']
    # contador=-1
    # for j in m.j:
    #     if j=='G' or j=='X' or j=='Eth' or j=='Cell':
    #         contador=contador+1
    #         plt.plot(time,vec[j],colors[contador],label=j)
    #         original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
    #         plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
    #     plt.xlabel('time [h]')
    #     plt.ylabel('Concentration [g/kg]')
    #     plt.legend()
    # plt.show()

    # contador=-1
    # for j in m.j:
    #     if j=='CS' or j=='XS' or j=='E':
    #         contador=contador+1
    #         plt.plot(time,vec[j],colors[contador],label=j)
    #         original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
    #         plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
    #     plt.xlabel('time [h]')
    #     plt.ylabel('Concentration [g/kg]')
    #     plt.legend()
    # plt.show()

    # plt.plot(time,pH)
    # plt.xlabel('time [-]')
    # plt.ylabel('pH')
    # plt.show()

    # # DATA TO ADJUST PARAMETERS
    # colors=['b','g','m','r','k','y','c']
    # adjust_data={}
    # contador=-1
    # for j in m.j:
    #     if j=='G' or j=='X' or j=='Eth' or j=='Cell' or  j=='CS' or j=='XS' or j=='E':
    #         contador=contador+1
    #         original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
    #         adjust_data[j]=[]
    #         for t in m.t:
    #             adjusted_t=t*m.final_time*(1/60)*(1/60)
    #             adjust_data[j].append(max([np.interp(adjusted_t,original.iloc[:, 0],original.iloc[:, 1]),0]))
    #         # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
    #         # plt.plot(time,adjust_data[j],colors[contador],label=j)
    # # plt.xlabel('time [h]')
    # # plt.ylabel('Concentration [g/kg]')
    # # plt.legend()
    # # plt.show()




    # m=build_fermentation_convergence_testFinal5_adjust(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1,data=adjust_data)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_21')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='conopt4', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_22')

    # m.K0G.pprint()
    # m.K1G.pprint()
    # m.K2G.pprint()
    # m.K0X.pprint()
    # m.K1X.pprint()
    # m.K2X.pprint()


    m=build_fermentation_convergence_testFinal6(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_22')
    opt1 = SolverFactory('gams')
    results = opt1.solve(m, solver='conopt4', tee=True)
    generate_initialization(m=m,model_name='validation_fermentation_23')

    time=[]
    vec={}
    pH=[]

    for t in m.t:
        time.append(t*m.final_time*(1/60)*(1/60))
        pH.append(m.pH[t].value)

    for j in m.j:
        vec[j]=[]
        for t in m.t:
            vec[j].append(m.C[t,j].value)

    colors=['b','g','m','r','k','y','c']
    contador=-1
    for j in m.j:
        if j=='G' or j=='X' or j=='Eth' or j=='Cell':
            contador=contador+1
            plt.plot(time,vec[j],colors[contador],label=j)
            original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
            plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
        plt.xlabel('time [h]')
        plt.ylabel('Concentration [g/kg]')
        plt.legend()
    plt.show()

    contador=-1
    for j in m.j:
        if j=='CS' or j=='XS' or j=='E':
            contador=contador+1
            plt.plot(time,vec[j],colors[contador],label=j)
            original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
            plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
        plt.xlabel('time [h]')
        plt.ylabel('Concentration [g/kg]')
        plt.legend()
    plt.show()

    plt.plot(time,pH)
    plt.xlabel('time [-]')
    plt.ylabel('pH')
    plt.show()















    ###EXTRA------------------------------------------------------

    # m=build_fermentation_convergence_test6(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count+3))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='knitro', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+4))

    # count=16
    # step=0.1
    # ending=1
    # for conv_p in np.arange(0,ending+step,step):
    #     count=count+1
    #     print('convergence_param_is ',conv_p)
    #     m=build_fermentation_convergence_test7(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=conv_p)
    #     m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count-1))
    #     opt1 = SolverFactory('gams')
    #     results = opt1.solve(m, solver='ipopth', tee=True)
    #     generate_initialization(m=m,model_name='validation_fermentation_'+str(count))


    # count=83
    # m=build_fermentation_convergence_test7(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_'+str(count))
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='knitro', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_'+str(count+1))

    # m=build_fermentation_convergence_test7(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,conv_param=1)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_84')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='ipopth', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_85')

    # m=build_fermentation_convergence_test8(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_5')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='knitro', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_86')

    # m=build_fermentation_convergence_test8(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_86')
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='knitro', tee=True)
    # generate_initialization(m=m,model_name='validation_fermentation_87')
