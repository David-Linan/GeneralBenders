import pyomo.environ as pe
import copy
import logging
import pyomo.dae as dae
from model_serializer import StoreSpec, from_json, to_json
import os
from pyomo.opt.base.solvers import SolverFactory
import random
import pickle



def initialize_model(
    m: pe.ConcreteModel(),
    json_path=None,
    from_feasible: bool = False,
    feasible_model: str = '',
) -> pe.ConcreteModel():
    """
    Function that return an initialized model from an existing json file
    Args:
        m: Pyomo model that is to be initialized
        from_feasible: If initialization is made from an external file
        feasible_model: Feasible initialization path or example
    Returns:
        m: Initialized Pyomo model
    """

    wts = StoreSpec.value()

    if json_path is None:
        os.path.join(os.path.curdir)

        dir_path = os.path.dirname(os.path.abspath(__file__))

        if from_feasible:
            json_path = os.path.join(
                dir_path, feasible_model+'_initialization.json')
        else:
            json_path = os.path.join(
                dir_path, 'dsda_initialization.json')

    from_json(m, fname=json_path, wts=wts)
    return m


def generate_initialization(
    m: pe.ConcreteModel(),
    starting_initialization: bool = False,
    model_name: str = '',
    human_read: bool = True,
    wts=StoreSpec.value(),
):
    """
    Function that creates a json file for initialization based on a model m
    Args:
        m: Base Pyomo model for initializtion
        starting_intialization: Use to create "dsda_starting_initialization.json" file with a known feasible initialized model m
        model_name: Name of the model for the initialization
        human_read: Make the json file readable by a human
        wts: What to save, initially the values, but we might want something different. Check model_serializer tests for examples
    Returns:
        json_path: Path where json file is stored
    """

    dir_path = os.path.dirname(os.path.abspath(__file__))

    if starting_initialization:
        json_path = os.path.join(
            dir_path, model_name + '_initialization.json')
    else:
        if model_name != '':
            json_path = os.path.join(
                dir_path, model_name + '_initialization.json')
        else:
            json_path = os.path.join(
                dir_path, 'dsda_initialization.json')

    to_json(m, fname=json_path, human_read=human_read, wts=wts)

    return json_path



# For open loop optimization
def build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time: float=506160,discretization: str='differences',n_f_elements_t: int=37):
    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------

    # n_f_elements_t dictates the discretized prediction horizon (between 1 and total_f_elements)
    # total_f_elements is the fixed total number fo finite elements that defines the sambpling time with respect to total_sim_time
    m.final_time = pe.Param(initialize=total_sim_time,doc='Prediction horizon with respect to 0 seconds [s]')  
    # m.current_starting_time=pe.Param(initialize=current_start_time_sconds,doc='Current start time [s]')
    # m.current_final_time=pe.Param(initialize=m.current_starting_time+m.final_time,doc='final simulation time with respect to the current start time [s]')
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
    # m.M0_prev=pe.Param(initialize=M0_prev_input,doc='Hold-up initial condition from previous time step')
    # m.C0_prev=pe.Param(m.j,initialize=C0_prev_input,doc='Concentration initial condition from previous time step')


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

    m.K0G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=5.388758642823563,doc='Parameter for pH dependency in glucose rate of fermentation model') 
    m.K2G=pe.Param(initialize=0.009698396119741,doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0X=pe.Param(initialize=1,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=5.375237819425663,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=0.009314982725521,doc='Parameter for pH dependency in xylose rate of fermentation model')
    

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

    m.F_C5liquid=pe.Var(m.t,initialize=628*(1/60)*(1/60),within=pe.NonNegativeReals,bounds=(0,2*628*(1/60)*(1/60)),doc='C5liquid flow [kg/s]')
    m.F_liquified_fibers=pe.Var(m.t,initialize=2487*(1/60)*(1/60),within=pe.NonNegativeReals,bounds=(0,2*2487*(1/60)*(1/60)),doc='Liquified fibers flow [kg/s]')

    m.pH=pe.Var(initialize=5.39,bounds=(5.36,5.4),within=pe.NonNegativeReals,doc='pH')     #bounds=(5.36,5.4)




    # m.F_base=pe.Var(initialize=0.00139,within=pe.NonNegativeReals,bounds=(0,0.01),doc='base flow for pH control [kg/s]')
    m.F_base=pe.Param(initialize=0,doc='base flow for pH control [kg/s]')
    m.F_acid=pe.Param(initialize=0,doc='acid flow for pH control [kg/s]')
    


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
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,mutable=True,doc='C5liquid concentration [g/kg]')

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
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,mutable=True,doc='Liquified fibers concentration [g/kg]')



    _C_base={}
    _C_base['Base']=270 #based on hydrolisis model
    m.C_base=pe.Param(m.j,initialize=_C_base,default=0,doc='Base control flow concentration [g/kg]')

    _C_acid={}
    _C_acid['AC']=100 # TODO find an appropriate value
    m.C_acid=pe.Param(m.j,initialize=_C_acid,default=0,doc='Acid control flow concentration [g/kg]')


    #----- Initical conditions  ----------------------------------
    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    # m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_yeast=pe.Var(initialize=147,within=pe.NonNegativeReals,bounds=(10,1000),doc='Initial yeast hold up in the reactor [kg]')
    # m.M0_yeast.fix(147)
    m.M0_water=pe.Param(initialize=2400,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
    m.M0=pe.Var(initialize=m.M0_fibers+m.M0_water+pe.value(m.M0_yeast),within=pe.NonNegativeReals,bounds=(m.M0_fibers+m.M0_water+m.M0_yeast.lb,m.M0_fibers+m.M0_water+m.M0_yeast.ub),doc='Initial hold up in the reactor [kg]')

    def _M0_def(m):
        return m.M0==m.M0_fibers+m.M0_water+m.M0_yeast
    m.M0_def=pe.Constraint(rule=_M0_def)
    # def _C0(m,j):
    #     if j=='Cell':
    #         return (1000*m.M0_yeast)/(m.M0)
    #     else:
    #         return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    # m.C0=pe.Param(m.j,initialize=_C0,doc='Initial concentration of the components involved [g/kg]')

    def _C0_init(m,j):
        if j=='Cell':
            return (1000*pe.value(m.M0_yeast))/(m.M0)
        else:
            return (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0=pe.Var(m.j,within=pe.NonNegativeReals,bounds=(0,1000),initialize=_C0_init,doc='Initial concentration of the components involved [g/kg]')

    def _C0_def(m,j):
        if j=='Cell':
            return m.C0[j] == (1000*m.M0_yeast)/(m.M0)
        else:
            return m.C0[j] == (m.C_liquified_fibers[j]*m.M0_fibers)/(m.M0)
    m.C0_def=pe.Constraint(m.j,rule=_C0_def)
    #----- Maximum reactor hold up------------------------------------------------
    m.Mmax=pe.Param(initialize=220000,doc='Maximum hold up in the reactor [kg]') #TODO: not using it so far

    # ----- Feed parameters --------------------------------------------------
    m.Fin=pe.Var(m.t,initialize=0,within=pe.NonNegativeReals,bounds=(0,10),doc='Feed flow [kg/s]')
    m.Cin=pe.Var(m.t,m.j,initialize=0,within=pe.NonNegativeReals,bounds=(0,1000),doc='Feed composition [g/kg]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,1), doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,1), doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,1), doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,1), doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,1), doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return pe.value(m.C0[j])
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals,bounds=(0,1000), doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,m.Mmax), doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals,bounds=(-1,1), doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,bounds=(-1,1),doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t,bounds=(-10000,10000),initialize=0)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t,bounds=(-10000000,10000000),initialize=0)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Initial condition
            return m.M[t] == m.M0
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
  
        if t==m.t.first(): #Initial condition
            return m.C[t,j] == m.C0[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
        m = discretizer_t.reduce_collocation_points(m,var=m.F_C5liquid,ncp=1,contset=m.t)
        m = discretizer_t.reduce_collocation_points(m,var=m.F_liquified_fibers,ncp=1,contset=m.t)
        # m = discretizer_t.reduce_collocation_points(m,var=m.pH,ncp=1,contset=m.t)     
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')

    # # ------------------Definition of feed flow and output flow information---------------------


    # for t in m.t:
    #     if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
    #         m.F_liquified_fibers[t].value=2487*(1/60)*(1/60)
    #         m.F_C5liquid[t].value=0
    #     elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
    #         m.F_liquified_fibers[t].value=2487*(1/60)*(1/60)
    #         m.F_C5liquid[t].value=628*(1/60)*(1/60)
    #     elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
    #         m.F_liquified_fibers[t].value=0
    #         m.F_C5liquid[t].value=0

    # def _Feed_constraint(m,t):
    #     if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
    #         return m.Fin[t]==m.F_liquified_fibers[t]
    #     elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
    #         return m.Fin[t]==m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
    #     elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
    #         return m.Fin[t]==0#m.F_base+m.F_acid
    # m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    # def _Feed_concentration_constraint(m,t,j):
    #     if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
    #         return m.Cin[t,j]==m.C_liquified_fibers[j]
    #     elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
    #         return m.Cin[t,j]*(m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid)==(m.F_C5liquid[t]*m.C_C5liquid[j]+m.F_liquified_fibers[t]*m.C_liquified_fibers[j]+m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j])
    #     elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
    #         return m.Cin[t,j]*(m.F_base+m.F_acid)== 0#m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j]    
    # m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)


    def _Feed_constraint(m,t):

        return m.Fin[t]==m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid

    m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    def _Feed_concentration_constraint(m,t,j):

        return m.Cin[t,j]*(m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid)==(m.F_C5liquid[t]*m.C_C5liquid[j]+m.F_liquified_fibers[t]*m.C_liquified_fibers[j]+m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j])
  
    m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)

    m.F_C5liquid[m.t.first()].fix(0)
    m.F_liquified_fibers[m.t.first()].fix(0)




    # def _current_integral_F(m):
    #     return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    # m.current_integral_F=pe.Param(initialize=_current_integral_F)

    # def _current_integral_C5(m):    
    #     return m.final_time*sum(  ((pe.value(m.F_C5liquid[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    # m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    # def _ingegral_F(m):
    #     return m.final_time*sum(  (((m.F_liquified_fibers[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_F
    # m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    # def _ingegral_C5(m):
    #     return m.final_time*sum(  (((m.F_C5liquid[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_C5
    # m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    # def _ingegral_F_rq(m,t):
    #     if  (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) >70*60*60:
    #         return m.F_liquified_fibers[t]==0
    #     elif t!=m.t.first():
    #         if keep_constant_flows:
    #             return m.F_liquified_fibers[t]==2487*(1/60)*(1/60)
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # m.ingegral_F_rq=pe.Constraint(m.t,rule=_ingegral_F_rq)

    # def _ingegral_C5_rq(m,t):
    #     if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<= 10*60*60 or (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) >70*60*60:
    #         return  m.F_C5liquid[t]==0
    #     elif t!=m.t.first():
    #         if keep_constant_flows:
    #             return m.F_C5liquid[t]==628*(1/60)*(1/60)
    #         else:
    #             return pe.Constraint.Skip
    #     else:
    #         return pe.Constraint.Skip
    # m.ingegral_C5_rq=pe.Constraint(m.t,rule=_ingegral_C5_rq)

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
    m.eta_severity=pe.Param(initialize=1, doc='Severity factor') 
    

    # m.j_elect = pe.Set(initialize=['C2H4O2', 'H+', 'C2H3O2-', 'OH-','CO2aq','HCO3-','CO3-2','C4H6O4','C4H5O4-','C4H4O4-2','C3H6O3','C3H5O3-','NaOH','Na+'],doc='components for pH calculations')
    # m.r_elect = pe.Set(initialize=['1','2','3','4','5','6','7','8'],doc='set of reactions for pH calculations')

    # _MW_elect={}
    # _MW_elect['C2H4O2']=60.05
    # _MW_elect['H+']=1.007825032
    # _MW_elect[ 'C2H3O2-']=59.04
    # _MW_elect['OH-']=17.007 
    # _MW_elect['CO2aq']=44.009
    # _MW_elect['HCO3-']=61.017
    # _MW_elect['CO3-2']=60.009
    # _MW_elect['C4H6O4']=118.09
    # _MW_elect['C4H5O4-']=117.08
    # _MW_elect['C4H4O4-2']=116.07
    # _MW_elect['C3H6O3']=90.08
    # _MW_elect['C3H5O3-']=89.07
    # _MW_elect['NaOH']=39.997
    # _MW_elect['Na+']= 22.9897693   
    # m.MW_elect=pe.Param(m.j_elect,initialize=_MW_elect,doc='Molecular weight of electrolytes [g/mol]')

    # _Equil_lhs={}
    # _Equil_lhs['1']=1.63E-5
    # _Equil_lhs['2']=1
    # _Equil_lhs['3']=5.39E-14
    # _Equil_lhs['4']=5.14E-7
    # _Equil_lhs['5']=6.69E-11
    # _Equil_lhs['6']=6.51E-5
    # _Equil_lhs['7']=2.08E-6
    # _Equil_lhs['8']=1.27E-4

    # m.Equil_lhs=pe.Param(m.r_elect,initialize=_Equil_lhs,doc='lhs Equilibrium constants for pH calculations')
    # _Equil_rhs={}
    # _Equil_rhs['1']=1
    # _Equil_rhs['2']=0
    # _Equil_rhs['3']=1
    # _Equil_rhs['4']=1
    # _Equil_rhs['5']=1
    # _Equil_rhs['6']=1
    # _Equil_rhs['7']=1
    # _Equil_rhs['8']=1
    # m.Equil_rhs=pe.Param(m.r_elect,initialize=_Equil_rhs,doc='rhs constants for pH calculations')

    # _coef_elect={}
    
    # _coef_elect['C2H4O2','1']=-1
    # _coef_elect['C2H3O2-','1']=1
    # _coef_elect['H+','1']=1

    # _coef_elect['NaOH','2']=-1
    # _coef_elect['Na+','2']=1
    # _coef_elect['OH-','2']=1

    # _coef_elect['H+','3']=1
    # _coef_elect['OH-','3']=1

    # _coef_elect['CO2aq','4']=-1
    # _coef_elect['HCO3-','4']=1
    # _coef_elect['H+','4']=1

    # _coef_elect['HCO3-','5']=-1
    # _coef_elect['CO3-2','5']=1
    # _coef_elect['H+','5']=1

    # _coef_elect['C4H6O4','6']=-1
    # _coef_elect['C4H5O4-','6']=1
    # _coef_elect['H+','6']=1

    # _coef_elect['C4H5O4-','7']=-1
    # _coef_elect['C4H4O4-2','7']=1
    # _coef_elect['H+','7']=1

    # _coef_elect['C3H6O3','8']=-1
    # _coef_elect['C3H5O3-','8']=1
    # _coef_elect['H+','8']=1

    # m.coef_elect=pe.Param(m.j_elect,m.r_elect,initialize=_coef_elect,default=0,doc='Stoichiometry coefficient of species j in reaction r')

    # _C_elect_init_param={}

    # _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    # _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    # _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    # _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    # _C_elect_init_param['H+']=0
    # m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # # m.C_elect_init_param.pprint()

    # m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    # m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    # m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    # m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,bounds=(-10,10),initialize=0,doc='production/consumption terms in reactions for pH calculations')

    # m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,bounds=(0,10),initialize=0.001,doc='Initial concentration of electrolytes')

    # def _C_elect_init_constraint(m,t,j):
    #     if j=='C2H4O2': #Acetic acid
    #         return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
    #     elif j=='C2H3O2-': #Acetate
    #         return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
    #     elif j=='NaOH': #Base
    #         return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    #     else: #TODO: this model is not rigurous enouugh? 
    #         return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    # m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    # m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,bounds=(0,10),initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    # def _equilibrium_relationships(m,t,r):
    #     # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
    #     if r=='4':
    #         # return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    #         return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])

    #     # For the remaining reactions we only consider the normal equilibrium calculation
    #     else:
    #         # If it is only a fordward reaction
    #         if m.Equil_rhs[r]==0:
    #             return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
    #         # If the reaction is an equilibrium reaction
    #         else:
    #             return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    # m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    # def _elect_balances(m,t,j):
    #     if j=='CO2aq':
    #         return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
    #     else:
    #         return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    # m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    # def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
    #     return 5.5
    # # def _pH_bounds(m,t):
    # #     if t==m.t.first() and m.current_starting_time==0:
    # #         return (1,8)
    # #     else:
    # #         return (5.2,5.6)
    # m.pH=pe.Var(within=pe.NonNegativeReals,initialize=_pH,bounds=(5,6),doc='pH')



    # def _pH_definition(m,t):
    #     # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
    #     # if t==m.t.first():
    #     #     return m.pH[t]==m.pH[m.t.next(t)]
    #     # else:
    #     # return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    #     return m.pH[t]==5.5
    #     # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
    # m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH)-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH-5.178044612)**2)/(2*((1.088854751)**2)))
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
        # return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        return (m.Ceb[t,e])*(1+m.k_ads[e]*m.Cef[t,e]) == (m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])*m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/1)
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            # return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
            return m.CebC[t,e]*(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.Ceb[t,e]*((m.C[t,'CS'])/1)
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            # return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
            return m.CebX[t,e]*(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.Ceb[t,e]*((m.C[t,'XS'])/1)
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
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G*pe.exp(-(((m.pH-m.K1G)**2)/(2*(m.K2G**2)))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
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
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*(m.K0X*pe.exp(-(((m.pH-m.K1X)**2)/(2*(m.K2X**2))))   )  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

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

    # m.desired_conc=pe.Param(initialize=75)
    # def _purity(m):
    #     return m.C[m.t.last(),'Eth'] >= m.desired_conc
    # m.purity_const=pe.Constraint(rule=_purity)

    def _obj(m):
        # sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        # 1000*sum( (m.pH[t]-5.5)**2 for t in m.t)
        # return 1 
        # return sum((m.C[t,'Eth']-100)**2  for t in m.t)
        # return -m.C[m.t.last(),'Eth'] +100*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon

        # return -1*m.C[m.t.last(),'Eth']+1000*sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+1000*sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
        # return (50*m.M0_yeast-5*m.C[m.t.last(),'Eth']*m.M[m.t.last()])+0*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
        objective_val=objective_function(m)
        return objective_val
        # return (50*m.M0_yeast-5*m.C[m.t[19],'Eth']*m.M[m.t[19]])+0*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
        # return -m.M[m.t.last()]-1000*m.C[m.t.last(),'Eth']
        # return -sum(m.pH[t] for t in m.t)
        # return -m.C_elect_equil[m.t.last(),'H+']
        #sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        # w={}
        # w['Eth']=1
        # w['G']=1
        # w['X']=1
        # return sum(sum( w[j]*((m.C[t,j]-data[j][m.t.ord(t)-1])**2)    for j in ['G','X','Eth']) for t in m.t) 
        # return -m.M[m.t.last()]
    m.obj = pe.Objective(rule=_obj)

    return m

def objective_function(m):
    wight=0
    return (50*m.M0_yeast-5*m.C[m.t.last(),'Eth']*m.M[m.t.last()])+wight*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon

    # return (50*m.M0_yeast-5*m.C[m.t[20],'Eth']*m.M[m.t[20]])+1e+8*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
    # return (-5*m.C[m.t.last(),'Eth'])+0*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon

def three_reactors_model(nominal_flow_F,nominal_flow_C5,constant_flows,include_fed_batch_op_time,Inoc_phase_time_wrt_0,Fed_batch_time_wrt_0,Total_F,Total_C5,react_list: list=[1,2,3],total_sim_time: float=506160,discretization: str='differences',n_f_elements_t: int=37):

    m = pe.ConcreteModel(name='three reactors')

    m.react_set=pe.Set(initialize=react_list,doc='Set of reactors')
    m.reactor={}
    m.share_F={}
    m.share_C5={}

    m.updated_F_constraint={}
    m.updated_Con_constraint={}
    m.ingegral_F={}
    m.ingegral_C5={}
    m.ingegral_F_rq={}
    m.ingegral_C5_rq={}



    for r in m.react_set:
        # Fermentation model
        m_r=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization=discretization,n_f_elements_t=n_f_elements_t)
        # Initialize model
        m_r=initialize_model(m_r,from_feasible=True,feasible_model='validation_fermentation_one_reactor')
        
        # Fermentation model update
        # Deleting curent objective function
        m_r.del_component(m_r.obj)        
        m.reactor[r]=m_r
        setattr(m,'reactor_%r' %r,m.reactor[r])

        #Delete current feed constraints and add updated ones
        m.reactor[r].del_component(m.reactor[r].Feed_constraint)
        m.reactor[r].del_component(m.reactor[r].Feed_concentration_constraint)


        def _updated_F_constraint(m,t):
            m=m.reactor[r]
            if include_fed_batch_op_time:
                if t*m.final_time<=Inoc_phase_time_wrt_0:
                    return m.Fin[t]==m.F_liquified_fibers[t]
                elif t*m.final_time>Inoc_phase_time_wrt_0 and t*m.final_time<=Fed_batch_time_wrt_0:
                    return m.Fin[t]==m.F_C5liquid[t] + m.F_liquified_fibers[t] 
                else:
                    return m.Fin[t]==0
            else:
                if t*m.final_time<=Inoc_phase_time_wrt_0:
                    return m.Fin[t]==m.F_liquified_fibers[t]
                else:
                    return m.Fin[t]==m.F_C5liquid[t] + m.F_liquified_fibers[t]
        m.updated_F_constraint[r]=pe.Constraint(m.reactor[r].t,rule=_updated_F_constraint)
        setattr(m,'updated_F_constraint_%r' %r,m.updated_F_constraint[r])


        def _updated_Con_constraint(m,t,j):
            m=m.reactor[r]
            if include_fed_batch_op_time:
                if t*m.final_time<=Inoc_phase_time_wrt_0:
                    return m.Cin[t,j]==m.C_liquified_fibers[j]
                elif t*m.final_time>Inoc_phase_time_wrt_0 and t*m.final_time<=Fed_batch_time_wrt_0:
                    return m.Cin[t,j]*(m.F_C5liquid[t] + m.F_liquified_fibers[t])==(m.F_C5liquid[t]*m.C_C5liquid[j]+m.F_liquified_fibers[t]*m.C_liquified_fibers[j])
                else:
                    return m.Cin[t,j]== 0
            else:
                if t*m.final_time<=Inoc_phase_time_wrt_0:
                    return m.Cin[t,j]==m.C_liquified_fibers[j]
                else:
                    return m.Cin[t,j]*(m.F_C5liquid[t] + m.F_liquified_fibers[t])==(m.F_C5liquid[t]*m.C_C5liquid[j]+m.F_liquified_fibers[t]*m.C_liquified_fibers[j])

        m.updated_Con_constraint[r]=pe.Constraint(m.reactor[r].t,m.reactor[r].j,rule=_updated_Con_constraint)
        setattr(m,'updated_Con_constraint_%r' %r,m.updated_Con_constraint[r])


        m.share_F[r]=pe.Var(within=pe.NonNegativeReals,bounds=(0,Total_F),initialize=Total_F/len(react_list))
        setattr(m,'share_F_%r' %r,m.share_F[r])

        m.share_C5[r]=pe.Var(within=pe.NonNegativeReals,bounds=(0,Total_C5),initialize=Total_C5/len(react_list))
        setattr(m,'share_C5_%r' %r,m.share_C5[r])


        def _ingegral_F(m):
            return m.reactor[r].final_time*sum(  (((m.reactor[r].F_liquified_fibers[t])))*(t-m.reactor[r].t.prev(t))    for t in m.reactor[r].t if t!=m.reactor[r].t.first())==m.share_F[r]
        m.ingegral_F[r]=pe.Constraint(rule=_ingegral_F)
        setattr(m,'ingegral_F_%r' %r,m.ingegral_F[r])

        def _ingegral_C5(m):
            return m.reactor[r].final_time*sum(  (((m.reactor[r].F_C5liquid[t])))*(t-m.reactor[r].t.prev(t))    for t in m.reactor[r].t if t!=m.reactor[r].t.first())==m.share_C5[r]
        m.ingegral_C5[r]=pe.Constraint(rule=_ingegral_C5)
        setattr(m,'ingegral_C5_%r' %r,m.ingegral_C5[r])
        
        def _ingegral_F_rq(m,t):
            m=m.reactor[r]
            if include_fed_batch_op_time:
                if  t*m.final_time >Fed_batch_time_wrt_0:
                    return m.F_liquified_fibers[t]==0
                elif t!=m.t.first():
                    if constant_flows:
                        return m.F_liquified_fibers[t]==nominal_flow_F    
                    else:
                        return pe.Constraint.Skip
                else:
                    return pe.Constraint.Skip                
            else:
                return pe.Constraint.Skip    
        m.ingegral_F_rq[r]=pe.Constraint(m.reactor[r].t,rule=_ingegral_F_rq)
        setattr(m,'ingegral_F_rq_%r' %r,m.ingegral_F_rq[r])

        def _ingegral_C5_rq(m,t):
            m=m.reactor[r]
            if include_fed_batch_op_time:
                if t*m.final_time<= Inoc_phase_time_wrt_0 or t*m.final_time >Fed_batch_time_wrt_0:
                    return  m.F_C5liquid[t]==0
                elif t!=m.t.first():
                    if constant_flows:
                        return m.F_C5liquid[t]==nominal_flow_C5
                    else:
                        return pe.Constraint.Skip
                else:
                    return pe.Constraint.Skip
            else:
                if t*m.final_time<= Inoc_phase_time_wrt_0:
                    return  m.F_C5liquid[t]==0
                else:
                    return pe.Constraint.Skip
        m.ingegral_C5_rq[r]=pe.Constraint(m.reactor[r].t,rule=_ingegral_C5_rq)
        setattr(m,'ingegral_C5_rq_%r' %r,m.ingegral_C5_rq[r])

    # COnstraints that links operation of different reactors: shared use of resources
    def _share_F_constraint(m):
        return Total_F==sum( m.share_F[r] for r in m.react_set)
    m.share_F_constraint=pe.Constraint(rule=_share_F_constraint)
    
    def _share_C5_constraint(m):
        return Total_C5==sum(m.share_C5[r] for r in m.react_set)
    m.share_C5_constraint=pe.Constraint(rule=_share_C5_constraint)


    # Declare new objective function
    def _global_objective(m):
        return  sum(objective_function(m.reactor[r]) for r in m.react_set)
    m.obj=pe.Objective(rule=_global_objective)

    return m

# For simulation
def build_fermentation_one_time_step_new(total_sim_time: float=190*60*60,
                                         discretization: str='collocation',
                                         n_f_elements_t: int=1,
                                         current_start_time_sconds: float=0,
                                         M0_prev_input: float=0,
                                         C0_prev_input: dict={'CS':0, 'XS':0, 'LS':0,'C':0,'G':0, 'X':0, 'F':0, 'E':0,'AC':0,'Cell':0,'Eth':0,'CO2':0,'ACT':0,'HMF':0,'Base':0},
                                         pH_val: float=5.5,
                                         M_yeast: float=100,
                                         F_fibers:float=1,
                                         F_C5:float=1,
                                         glucose_disturbance_F:float=0,
                                         xylose_disturbance_F:float=0,
                                         cs_disturbance_F:float=0,
                                         xs_disturbance_F:float=0,
                                         ls_disturbance_F:float=0,
                                         f_disturbance_F:float=0,
                                         e_disturbance_F:float=0,
                                         ac_disturbance_F:float=0,
                                         act_disturbance_F:float=0,
                                         hmf_disturbance_F:float=0,
                                         base_disturbance_F:float=0,
                                         glucose_disturbance_C5:float=0,
                                         xylose_disturbance_C5:float=0,
                                         cs_disturbance_C5:float=0,
                                         xs_disturbance_C5:float=0,
                                         ls_disturbance_C5:float=0,
                                         f_disturbance_C5:float=0,
                                         ac_disturbance_C5:float=0,
                                         act_disturbance_C5:float=0,
                                         hmf_disturbance_C5:float=0,
                                         ) -> pe.ConcreteModel():

    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------
    m.final_time = pe.Param(initialize=(total_sim_time),doc='final simulation time with respect to 0 seconds [s]')  
    m.current_starting_time=pe.Param(initialize=current_start_time_sconds,doc='Current start time [s]')
    m.current_final_time=pe.Param(initialize=m.current_starting_time+m.final_time,doc='final simulation time with respect to the current start time [s]')
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
    m.M0_prev=pe.Param(initialize=M0_prev_input,doc='Hold-up initial condition from previous time step')
    m.C0_prev=pe.Param(m.j,initialize=C0_prev_input,doc='Concentration initial condition from previous time step')


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

    m.K0G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Param(initialize=5.388758642823563,doc='Parameter for pH dependency in glucose rate of fermentation model') 
    m.K2G=pe.Param(initialize=0.009698396119741,doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0X=pe.Param(initialize=1,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Param(initialize=5.375237819425663,doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Param(initialize=0.009314982725521,doc='Parameter for pH dependency in xylose rate of fermentation model')
    

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

    m.F_C5liquid=pe.Param(initialize=F_C5,doc='C5liquid flow [kg/s]')
    m.F_liquified_fibers=pe.Param(initialize=F_fibers,doc='Liquified fibers flow [kg/s]')

    # m.F_base=pe.Var(initialize=0.00139,within=pe.NonNegativeReals,bounds=(0,0.01),doc='base flow for pH control [kg/s]')
    m.F_base=pe.Param(initialize=0,doc='base flow for pH control [kg/s]')
    m.F_acid=pe.Param(initialize=0,doc='acid flow for pH control [kg/s]')
    


    _C_C5liquid={}
    _C_C5liquid['CS']=max(1.2+cs_disturbance_C5*1.2,0)
    _C_C5liquid['XS']=max(0.5+xs_disturbance_C5*0.5,0)
    _C_C5liquid['LS']=max(0.7+ls_disturbance_C5*0.7,0)
    _C_C5liquid['C']=0 #0.1   # NOT reported. Guess
    _C_C5liquid['G']=max(10+glucose_disturbance_C5*10,0)
    _C_C5liquid['X']=max(29.7+xylose_disturbance_C5*29.7,0)
    _C_C5liquid['F']=max(0.5+f_disturbance_C5*0.5,0)
    _C_C5liquid['E']=0
    _C_C5liquid['AC']=max(4.1+ac_disturbance_C5*4.1,0) # this may be the mixture of acids
    _C_C5liquid['Cell']=0 # Same as yeast (?)
    _C_C5liquid['Eth']=0
    _C_C5liquid['CO2']=0
    _C_C5liquid['ACT']=max(0.2/2+act_disturbance_C5*(0.2/2),0) # Maybe "Acetyls" in table?
    _C_C5liquid['HMF']=max(0.3+hmf_disturbance_C5*0.3,0) 
    _C_C5liquid['Base']=0
    m.C_C5liquid=pe.Param(m.j,initialize=_C_C5liquid,doc='C5liquid concentration [g/kg]')

    _C_liquified_fibers={}
    _C_liquified_fibers['CS']=max(50+cs_disturbance_F*50,0)
    _C_liquified_fibers['XS']=max(1+xs_disturbance_F*1,0)
    _C_liquified_fibers['LS']=max(78+ls_disturbance_F*78,0)
    _C_liquified_fibers['C']=0 #26.6/2   # NOT reported
    _C_liquified_fibers['G']=max(98+glucose_disturbance_F*98,0)
    _C_liquified_fibers['X']=max(59+xylose_disturbance_F*59,0)
    _C_liquified_fibers['F']=max(0.2+f_disturbance_F*0.2,0)
    _C_liquified_fibers['E']=max(4.9+e_disturbance_F*4.9,0)
    _C_liquified_fibers['AC']=max(16+ac_disturbance_F*16,0) # this may be the mixture of acids
    _C_liquified_fibers['Cell']=0 # Same as yeast (?)
    _C_liquified_fibers['Eth']=0
    _C_liquified_fibers['CO2']=0
    _C_liquified_fibers['ACT']=max(0.1+act_disturbance_F*0.1,0)
    _C_liquified_fibers['HMF']= max(0.1+hmf_disturbance_F*0.1,0)    
    _C_liquified_fibers['Base']=max(8.6+base_disturbance_F*8.6,0)  
    m.C_liquified_fibers=pe.Param(m.j,initialize=_C_liquified_fibers,doc='Liquified fibers concentration [g/kg]')



    _C_base={}
    _C_base['Base']=270 #based on hydrolisis model
    m.C_base=pe.Param(m.j,initialize=_C_base,default=0,doc='Base control flow concentration [g/kg]')

    _C_acid={}
    _C_acid['AC']=100 # TODO find an appropriate value
    m.C_acid=pe.Param(m.j,initialize=_C_acid,default=0,doc='Acid control flow concentration [g/kg]')


    #----- Initical conditions  ----------------------------------
    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=M_yeast,doc='Initial yeast hold up in the reactor [kg]')
    m.M0_water=pe.Param(initialize=2400,doc='Initial water hold up in the reactor [kg]') #TODO: Adjust to complete 220 tons, which should also agree if adjusting to guarantee initial yeast concentration in plot
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
    m.Fin=pe.Var(m.t,initialize=0,within=pe.NonNegativeReals,bounds=(0,10),doc='Feed flow [kg/s]')
    m.Cin=pe.Var(m.t,m.j,initialize=0,within=pe.NonNegativeReals,bounds=(0,1000),doc='Feed composition [g/kg]')

    #---- Variables from hydrolisis model--------------------------------------------------
    m.Ce=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Cef=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.Ceb=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m.CebC=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m.CebX=pe.Var(m.t, m.e, initialize=1,within=pe.NonNegativeReals,bounds=(0,1000), doc='Concentration of adsorbed enzymes to xylan g/kg')
    m.r1=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,100), doc='Cellulose to cellobiose rate, g/kg s')
    m.r2=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,100), doc='Cellulose to glucose rate, g/kg s')
    m.r3=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,100), doc='Cellobiose to glucose rate, g/kg s')
    m.r4=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,100), doc='Xylan to xylose rate, g/kg s')
    m.r5=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,100), doc='Xylan to acetic acid rate, g/kg s')

    #---- main variables -------------------------------------------------------------
    def _C_init(m,t,j):
        return m.C0[j]
    m.C=pe.Var(m.t, m.j, initialize=_C_init,within=pe.NonNegativeReals,bounds=(0,1000), doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,m.Mmax), doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
    m.R = pe.Var(m.t, m.j, initialize=1, within=pe.Reals,bounds=(-100,100), doc='units of g/ (kg s)')

    # ---------Reaction kinetic expresions for fermentation part -------------------------

    m.q=pe.Var(m.t,m.j,initialize=1,within=pe.Reals,bounds=(-100,100),doc='fermentation reactions kinetic expresions [g/kg s]')

    #---------derivative variables-------------------------------------------
    m.dCdt=dae.DerivativeVar(m.C,wrt=m.t,bounds=(-1000000,1000000),initialize=0)
    m.dMdt=dae.DerivativeVar(m.M,wrt=m.t,bounds=(-1000000000,1000000000),initialize=0)

    #--------constraitns----------------------------------------------------

    # Total balance differential equation
    def _Diff_mass(m,t):    
        if t==m.t.first(): #Final condition from previous time step
            return m.M[t] == m.M0_prev 
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):  
        if t==m.t.first(): # Final condition from previous step
            return m.C[t,j] == m.C0_prev[j]
        else:
            return  m.M[t]*m.dCdt[t,j]== m.final_time*(m.Fin[t]*(m.Cin[t,j]-m.C[t,j]) + m.M[t]*m.R[t,j]) 
    m.Diff_comp=pe.Constraint(m.t,m.j,rule=_Diff_comp)

    if discretization=='collocation':
        discretizer_t = pe.TransformationFactory('dae.collocation')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, ncp=3, wrt=m.t, scheme='LAGRANGE-RADAU')
    else:
        discretizer_t = pe.TransformationFactory('dae.finite_difference')
        discretizer_t.apply_to(m, nfe=n_f_elements_t, wrt=m.t, scheme='BACKWARD')


    # ------------------Definition of feed flow and output flow information---------------------
    def _Feed_constraint(m,t):

        return m.Fin[t]==m.F_C5liquid + m.F_liquified_fibers+m.F_base+m.F_acid       #(m.Mmax-m.M0)/(70*60*60-10*60*60)

    m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    def _Feed_concentration_constraint(m,t,j):

        return m.Cin[t,j]*(m.F_C5liquid + m.F_liquified_fibers+m.F_base+m.F_acid)==(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j]+m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j])
  
    m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)

    #------------------- pH modeling (from hydrolysis) ----------------------------------------------
    m.eta_T=pe.Param(initialize=0.3, doc='Temperature efficiency factor. Value between 0 and 1') #NOTE: Temperature can be assumed constant at 50 C
    m.eta_severity=pe.Param(initialize=1, doc='Severity factor') 
    

    # m.j_elect = pe.Set(initialize=['C2H4O2', 'H+', 'C2H3O2-', 'OH-','CO2aq','HCO3-','CO3-2','C4H6O4','C4H5O4-','C4H4O4-2','C3H6O3','C3H5O3-','NaOH','Na+'],doc='components for pH calculations')
    # m.r_elect = pe.Set(initialize=['1','2','3','4','5','6','7','8'],doc='set of reactions for pH calculations')

    # _MW_elect={}
    # _MW_elect['C2H4O2']=60.05
    # _MW_elect['H+']=1.007825032
    # _MW_elect[ 'C2H3O2-']=59.04
    # _MW_elect['OH-']=17.007 
    # _MW_elect['CO2aq']=44.009
    # _MW_elect['HCO3-']=61.017
    # _MW_elect['CO3-2']=60.009
    # _MW_elect['C4H6O4']=118.09
    # _MW_elect['C4H5O4-']=117.08
    # _MW_elect['C4H4O4-2']=116.07
    # _MW_elect['C3H6O3']=90.08
    # _MW_elect['C3H5O3-']=89.07
    # _MW_elect['NaOH']=39.997
    # _MW_elect['Na+']= 22.9897693   
    # m.MW_elect=pe.Param(m.j_elect,initialize=_MW_elect,doc='Molecular weight of electrolytes [g/mol]')

    # _Equil_lhs={}
    # _Equil_lhs['1']=1.63E-5
    # _Equil_lhs['2']=1
    # _Equil_lhs['3']=5.39E-14
    # _Equil_lhs['4']=5.14E-7
    # _Equil_lhs['5']=6.69E-11
    # _Equil_lhs['6']=6.51E-5
    # _Equil_lhs['7']=2.08E-6
    # _Equil_lhs['8']=1.27E-4

    # m.Equil_lhs=pe.Param(m.r_elect,initialize=_Equil_lhs,doc='lhs Equilibrium constants for pH calculations')
    # _Equil_rhs={}
    # _Equil_rhs['1']=1
    # _Equil_rhs['2']=0
    # _Equil_rhs['3']=1
    # _Equil_rhs['4']=1
    # _Equil_rhs['5']=1
    # _Equil_rhs['6']=1
    # _Equil_rhs['7']=1
    # _Equil_rhs['8']=1
    # m.Equil_rhs=pe.Param(m.r_elect,initialize=_Equil_rhs,doc='rhs constants for pH calculations')

    # _coef_elect={}
    
    # _coef_elect['C2H4O2','1']=-1
    # _coef_elect['C2H3O2-','1']=1
    # _coef_elect['H+','1']=1

    # _coef_elect['NaOH','2']=-1
    # _coef_elect['Na+','2']=1
    # _coef_elect['OH-','2']=1

    # _coef_elect['H+','3']=1
    # _coef_elect['OH-','3']=1

    # _coef_elect['CO2aq','4']=-1
    # _coef_elect['HCO3-','4']=1
    # _coef_elect['H+','4']=1

    # _coef_elect['HCO3-','5']=-1
    # _coef_elect['CO3-2','5']=1
    # _coef_elect['H+','5']=1

    # _coef_elect['C4H6O4','6']=-1
    # _coef_elect['C4H5O4-','6']=1
    # _coef_elect['H+','6']=1

    # _coef_elect['C4H5O4-','7']=-1
    # _coef_elect['C4H4O4-2','7']=1
    # _coef_elect['H+','7']=1

    # _coef_elect['C3H6O3','8']=-1
    # _coef_elect['C3H5O3-','8']=1
    # _coef_elect['H+','8']=1

    # m.coef_elect=pe.Param(m.j_elect,m.r_elect,initialize=_coef_elect,default=0,doc='Stoichiometry coefficient of species j in reaction r')

    # _C_elect_init_param={}

    # _C_elect_init_param['CO2aq']=0*m.rho_soluble_kg_L*(1/m.MW_elect['CO2aq'])
    # _C_elect_init_param['C4H6O4']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C4H6O4'])   #Succinic acid
    # _C_elect_init_param['C3H6O3']=0* m.rho_soluble_kg_L*(1/m.MW_elect['C3H6O3'])  #Lactic acid
    # _C_elect_init_param['NaOH']=0* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    # _C_elect_init_param['H+']=0
    # m.C_elect_init_param=pe.Param(m.j_elect,initialize=0,default=0,doc='Initial concentration of electrolytes [mol/L]')
    # # m.C_elect_init_param.pprint()

    # m.kCO2=pe.Param(initialize=489.6,doc='mass transfer coefficient of CO2 [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"    489.6
    # m.r_kCO2=pe.Param(initialize=2.4*(60)*(24),doc='reaction rate constant in the equilibrium CO2 reaction [   1/d    ]') #NOT given, retrieved from: "Extensions to modeling aerobic carbon degradation using combined respirometric–titrimetric measurements in view of activated sludge model calibration"
    # m.CO2_atm=pe.Param(initialize=1.71E-5,doc='Atmospheric CO2 concentration [ mol/L ]') #Given in ACC short paper

    # m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,bounds=(-10,10),initialize=0,doc='production/consumption terms in reactions for pH calculations')

    # m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,bounds=(0,10),initialize=0.001,doc='Initial concentration of electrolytes')

    # def _C_elect_init_constraint(m,t,j):
    #     if j=='C2H4O2': #Acetic acid
    #         return m.C_elect_init[t,j]==m.C[t,'AC']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H4O2'])
    #     elif j=='C2H3O2-': #Acetate
    #         return m.C_elect_init[t,j]==m.C[t,'ACT']* m.rho_soluble_kg_L*(1/m.MW_elect['C2H3O2-'])
    #     elif j=='NaOH': #Base
    #         return m.C_elect_init[t,j]==m.C[t,'Base']* m.rho_soluble_kg_L*(1/m.MW_elect['NaOH'])
    #     else: #TODO: this model is not rigurous enouugh? 
    #         return m.C_elect_init[t,j]==m.C_elect_init_param[j]

    # m.C_elect_init_constraint=pe.Constraint(m.t,m.j_elect,rule=_C_elect_init_constraint)

    # m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,bounds=(0,10),initialize=1E-5,doc='Equilibrium concentration of electrolytes')


    # def _equilibrium_relationships(m,t,r):
    #     # for the CO2 equilbrium reaction we also consider the transfer of aqueous CO2 to the gas phase
    #     if r=='4':
    #         # return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])+((m.kCO2*m.Equil_lhs[r])/m.r_kCO2)*(m.CO2_atm-m.C_elect_equil[t,'CO2aq'])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    #         return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])

    #     # For the remaining reactions we only consider the normal equilibrium calculation
    #     else:
    #         # If it is only a fordward reaction
    #         if m.Equil_rhs[r]==0:
    #             return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==0
    #         # If the reaction is an equilibrium reaction
    #         else:
    #             return m.Equil_lhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==-1])==m.Equil_rhs[r]*pe.prod([m.C_elect_equil[t,j] for j in m.j_elect if m.coef_elect[j,r]==1])
    # m.equilibrium_relationships=pe.Constraint(m.t,m.r_elect,rule=_equilibrium_relationships)

    # def _elect_balances(m,t,j):
    #     if j=='CO2aq':
    #         return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)   
    #     else:
    #         return m.C_elect_equil[t,j]==m.C_elect_init[t,j] + sum(m.coef_elect[j,r]*m.avance[t,r] for r in m.r_elect)
    # m.elect_balances=pe.Constraint(m.t,m.j_elect,rule=_elect_balances)

    # def _pH(m,t): #TODO: either leave pH constant or include electrolyte balance
    #     return 5.5
    # # def _pH_bounds(m,t):
    # #     if t==m.t.first() and m.current_starting_time==0:
    # #         return (1,8)
    # #     else:
    # #         return (5.2,5.6)
    # m.pH=pe.Var(within=pe.NonNegativeReals,initialize=_pH,bounds=(5,6),doc='pH')

    m.pH=pe.Param(initialize=pH_val,doc='pH')

    # def _pH_definition(m,t):
    #     # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
    #     # if t==m.t.first():
    #     #     return m.pH[t]==m.pH[m.t.next(t)]
    #     # else:
    #     # return 10**(-m.pH[t])==m.C_elect_equil[t,'H+']
    #     return m.pH[t]==5.5
    #     # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
    # m.pH_definition=pe.Constraint(m.t,rule=_pH_definition)


    def _eta_pH_init(m,t):  
        return pe.exp(-((pe.value(m.pH)-5.178044612)**2)/(2*((1.088854751)**2)))
    m.eta_pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_eta_pH_init, bounds=(0,1.1), doc='pH efficiency factor. Value between 0 and 1') 

    def _eq_eta_pH(m,t):
        return m.eta_pH[t]== pe.exp(-((m.pH-5.178044612)**2)/(2*((1.088854751)**2)))
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
        # return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))
        return (m.Ceb[t,e])*(1+m.k_ads[e]*m.Cef[t,e]) == (m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])*m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/1)
        # else:
        #     return pe.Constraint.Skip
    m.adsorbed_free_equilibrium=pe.Constraint(m.t,m.e,rule=_adsorbed_free_equilibrium)

    def _bounded_enzyme_concentration(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            # return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
            return m.CebC[t,e]*(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.Ceb[t,e]*((m.C[t,'CS'])/1)
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            # return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
            return m.CebX[t,e]*(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.Ceb[t,e]*((m.C[t,'XS'])/1)
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
            return m.q[t,j] == (1/m.Y_Eth_G)*(   (   m.qmax_G*(m.K0G*pe.exp(-(((m.pH-m.K1G)**2)/(2*(m.K2G**2)))))   )*m.C[t,'Cell']*(m.C[t,'G']/(m.KSP_G+m.C[t,'G']+(((m.C[t,'G'])**2)/(m.KIP_G))))   )*(   1-(m.C[t,'Eth']/m.PMP_G)**m.gamma_G   )*(   (m.KI_F_G)/(m.KI_F_G+m.C[t,'F'])   )*(   (m.KI_ACT_G)/(m.KI_ACT_G+m.C[t,'ACT'])   )*(   (m.KI_HMF_G)/(m.KI_HMF_G+m.C[t,'HMF'])   )
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
            return m.q[t,j]== (1/m.Y_Eth_X)*(  (  m.qmax_X*(m.K0X*pe.exp(-(((m.pH-m.K1X)**2)/(2*(m.K2X**2))))   )  )*m.C[t,'Cell']*((m.C[t,'X'])/(m.KSP_X+m.C[t,'X']+((m.C[t,'X']**2)/(m.KIP_X))))  )*(  1-((m.C[t,'Eth'])/(m.PMP_X))**m.gamma_X  )*(  (m.KI_F_X/(m.KI_F_X+m.C[t,'F']))  )*(  (m.KI_ACT_X/(m.KI_ACT_X+m.C[t,'ACT']))  )*(  (m.KI_HMF_X/(m.KI_HMF_X+m.C[t,'HMF']))  )

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

        return 1 

    m.obj = pe.Objective(rule=_obj)

    return m




if __name__ == '__main__':
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    # NUMBER OF CYCLES TO BE SIMULATED IN THE SOM
    Number_cycles_som=4

    step=190*60*60/50     #NOTE: We assume this is the sampling time of the system Sampling_time


    include_fed_batch_op_time=True# If fed batch operation time will be included
    constant_flows=True # If include_fed_batch_op_time is true, then we also allow the option to have constant flows (for comparison purposes)



    disturbance=True
    variation_param_sim=0.3 # parameter to define uncertainty range (for simulation)


    # Include control actions constraint
    include_control_actions_constraint=True
    control_actions_val=0.04 #kg/s 0.04

    # Available reactors
    reactors_list=[1,2,3]


    # MORE TIMES REQUIRED TO DEFINE OPERATION
    Start_new_batch_time_wrt_0=70*60*60  #NOTE: In standar operation, this is the same as fed_batch time, but we will allow fed batch time to be variable
    Fed_batch_time_wrt_0=Start_new_batch_time_wrt_0
    Reaction_end_time_wrt_0=Start_new_batch_time_wrt_0*2
    Inoc_phase_time_wrt_0=10*60*60  # SECONDS
    Cycle_time_wrt_0=Start_new_batch_time_wrt_0*3+step
    # print(step)
    # print(Start_new_batch_time_wrt_0)
    # print(Inoc_phase_time_wrt_0)
    # print(Cycle_time_wrt_0)    
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Start_new_batch_time_wrt_0 and Start_new_batch_time_wrt_0<=cont_time+step:
             Start_new_batch_time_wrt_0=cont_time+step
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Fed_batch_time_wrt_0 and Fed_batch_time_wrt_0<=cont_time+step:
             Fed_batch_time_wrt_0=cont_time
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Inoc_phase_time_wrt_0 and Inoc_phase_time_wrt_0<=cont_time+step:
             Inoc_phase_time_wrt_0=cont_time
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Reaction_end_time_wrt_0 and Reaction_end_time_wrt_0<=cont_time+step:
             Reaction_end_time_wrt_0=cont_time+step
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Cycle_time_wrt_0 and Cycle_time_wrt_0<=cont_time+step:
             Cycle_time_wrt_0=cont_time+step
             break
    # print(Start_new_batch_time_wrt_0)
    # print(Inoc_phase_time_wrt_0)
    # print(Cycle_time_wrt_0)

    # print(Start_new_batch_time_wrt_0-step)
    # print(Inoc_phase_time_wrt_0-step)
    # print(Cycle_time_wrt_0-step)

    #Start time of SOM operation
    start_time=0 # NOTE: Must always be 0
    #Final time of SOM operation
    end_time=Number_cycles_som*(Cycle_time_wrt_0-step)

    # Dictionary with reactors start-time
    r_start_times={}
    for reactors in reactors_list:
        r_start_times[reactors]=(reactors-1)*Start_new_batch_time_wrt_0

    # print(r_start_times)

    # Dictionary with reactors operation modes

    r_operation_mode={} # reactor: list with operationg mode for every point in time
    for reactors in reactors_list:
        r_operation_mode[reactors]=[]
    # 0: batch reactor turned off, 1: start the execution of a batch reactor, 2: Batch reactor in Inoculum phase, 3: Batch reactor under fed-batch execution, 4: batch reactor without feeds! (if needed) 
    print('Time','  |  Operation mode r1', '  |  Operation mode r2','  |  Operation mode r3')
    time_index=-1
    current_Start_t=copy.deepcopy(r_start_times)
    for cont_time in pe.RangeSet(start_time,end_time,step):
        time_index=time_index+1
        for reactors in reactors_list:
            if cont_time==r_start_times[reactors] or cont_time==Cycle_time_wrt_0+current_Start_t[reactors]:
                r_operation_mode[reactors].append(1)
                current_Start_t[reactors]=cont_time
            elif cont_time>current_Start_t[reactors] and cont_time<=Inoc_phase_time_wrt_0+current_Start_t[reactors]:
                r_operation_mode[reactors].append(2)
            elif cont_time>current_Start_t[reactors]+Inoc_phase_time_wrt_0 and cont_time<= Fed_batch_time_wrt_0+current_Start_t[reactors]:
                r_operation_mode[reactors].append(3)
            elif cont_time>current_Start_t[reactors]+Fed_batch_time_wrt_0 and cont_time<=Reaction_end_time_wrt_0+current_Start_t[reactors]:
                if include_fed_batch_op_time:
                    r_operation_mode[reactors].append(4)
                else:
                    r_operation_mode[reactors].append(3)

            else:
                r_operation_mode[reactors].append(0)


        print(cont_time,'  |  ',r_operation_mode[1][time_index],'  |  ',r_operation_mode[2][time_index],'  |  ',r_operation_mode[3][time_index])



    # NONLINEAR ECONOMIC MODEL PREDICTIVE CONTROL
    solver_list=['conopt','conopt4','knitro','ipopth']
    simulation_solvers=solver_list
    tee=False
    discretization_type_fer='differences'
    sim_discretization='differences'
    sim_n_finite_elements=3
    # discretization_type_fer='collocation'
    finite_elem_t_fer=37
    total_sim_time=Reaction_end_time_wrt_0


    # INITIALIZE 
    init_old=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=190*60*60,discretization=discretization_type_fer,n_f_elements_t=50) 
    init_old=initialize_model(init_old,from_feasible=True,feasible_model='validation_fermentation')

    init_new=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer) 
    time_index=init_new.t
    old_time_index=init_old.t
    for v in init_new.component_objects(ctype=pe.Var):
        for c in init_old.component_objects(ctype=pe.Var):
            if v.name==c.name:
                old_v=c

        # Check if variable has time index. If it does, initialize this variable with its final state value
        try: # If variable is defined over multiple sets
            position=[v.index_set()._sets[j].name==time_index.name for j in range(len(v.index_set()._sets))] # returns tru for the position of the index that corresponds to time
        except: # If only defined over a single set
            position=[v.index_set().name==time_index.name]
        if any(position): # If the variable has time index
            # itentify location of time index
            cuenta=0
            for i in position:
                if i==True:
                    loc=cuenta #location of time index
                    break
                cuenta=cuenta+1
  

            if len(position)==1: # variables only have time index
                prev_t=time_index.first()
                locatp=1
                for index in v.index_set().data():
                    current_t=index  
                    if current_t>prev_t:
                        locatp=locatp+1         
                        
                    partial_index_lst=index
                    partial_index_lst=old_time_index[locatp]
                    partial_index=partial_index_lst

                    prev_t=index
                    v[index].value=old_v[partial_index].value    

            else: # variables also have other indexes
                prev_t=time_index.first()
                locatp=1
                for index in v.index_set().data():
                    current_t=index[loc]   
                    if current_t>prev_t:
                        locatp=locatp+1         
                        
                    partial_index_lst=list(index)
                    partial_index_lst[loc]=old_time_index[locatp]
                    partial_index=tuple(partial_index_lst)

                    prev_t=index[loc] 
                    v[index].value=old_v[partial_index].value                  

        else:

            for index in v.index_set().data(): 
                v[index].value=old_v[index].value



    nominal_flow_F=2487*(1/60)*(1/60)
    nominal_flow_C5=628*(1/60)*(1/60)

    for t in init_new.t:
        if t*init_new.final_time<=Inoc_phase_time_wrt_0: # Inoculum phase
            init_new.F_liquified_fibers[t].fix(nominal_flow_F)
            init_new.F_C5liquid[t].fix(0)
        elif t*init_new.final_time> Inoc_phase_time_wrt_0 and t*init_new.final_time <=Fed_batch_time_wrt_0: #Fed-batch phase
            init_new.F_liquified_fibers[t].fix(nominal_flow_F)
            init_new.F_C5liquid[t].fix(nominal_flow_C5)
        else: #Batch phase
            init_new.F_liquified_fibers[t].fix(0)
            init_new.F_C5liquid[t].fix(0)






    opt1 = SolverFactory('gams') # Solve problem
    init_new.results = opt1.solve(init_new, solver='conopt4', tee=False)   
    generate_initialization(m=init_new,model_name='validation_fermentation_one_reactor')
    # init_new.M0_yeast.pprint()
    print('original step',step)
    step_updated=init_new.final_time*(init_new.t.next(init_new.t.first())-init_new.t.first())
    print('updated step',step_updated)

    # CALCULATE AVAILABLE SUBSTRATE PER CYCLE
    nominal_available_F_per_batch=init_new.final_time*sum(  ((pe.value(init_new.F_liquified_fibers[t])))*(t-init_new.t.prev(t))    for t in init_new.t if t!=init_new.t.first())
    nominal_available_C5_per_batch=init_new.final_time*sum(  ((pe.value(init_new.F_C5liquid[t])))*(t-init_new.t.prev(t))    for t in init_new.t if t!=init_new.t.first())
    
    nominal_available_F_per_cycle=nominal_available_F_per_batch*len(reactors_list)
    nominal_available_C5_per_cycle=nominal_available_C5_per_batch*len(reactors_list)



    print('Available Fibers per cycle [kg]: ',nominal_available_F_per_cycle)
    print('Available C5 per cycle [kg]:',nominal_available_C5_per_cycle)




    # THREE REACTORS MODEL TEST # NOTE: I CAN DELETE THIS
    # m3=three_reactors_model(react_list=reactors_list,total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
    # opt1 = SolverFactory('gams') # Solve problem
    # m3.results = opt1.solve(m3, solver='conopt4', tee=False)   


    time_list=[] #Simulated time points
    time_point_obj_evaluation_dict={}
    Hold_up_dict={} #Simulated hold ups
    pH_dict={} # Simulated pH
    yeast_dict={} # Simulated yeast
    C5_dict={} # Simulated C5 flow
    fiber_dict={} #Simulated fibers flow
    objective_dict={} #Simulated objective function
    Concentration_dict={}
    Concentration_disturbance_dict_C5={}
    Concentration_disturbance_dict_F={}
    for reactors in reactors_list:
        Hold_up_dict.update({reactors:[]})
        pH_dict.update({reactors:[]})
        yeast_dict.update({reactors:[]})
        C5_dict.update({reactors:[]})
        fiber_dict.update({reactors:[]})
        objective_dict.update({reactors:[]})
        time_point_obj_evaluation_dict.update({reactors:[]})
        Concentration_dict.update({('CS',reactors):[], ('XS',reactors):[], ('LS',reactors):[],('C',reactors):[],('G',reactors):[], ('X',reactors):[], ('F',reactors):[], ('E',reactors):[],('AC',reactors):[],('Cell',reactors):[],('Eth',reactors):[],('CO2',reactors):[],('ACT',reactors):[],('HMF',reactors):[],('Base',reactors):[]}) #Simulated concentrations
        Concentration_disturbance_dict_C5.update({('CS',reactors):[], ('XS',reactors):[], ('LS',reactors):[],('C',reactors):[],('G',reactors):[], ('X',reactors):[], ('F',reactors):[], ('E',reactors):[],('AC',reactors):[],('Cell',reactors):[],('Eth',reactors):[],('CO2',reactors):[],('ACT',reactors):[],('HMF',reactors):[],('Base',reactors):[]})
        Concentration_disturbance_dict_F.update({('CS',reactors):[], ('XS',reactors):[], ('LS',reactors):[],('C',reactors):[],('G',reactors):[], ('X',reactors):[], ('F',reactors):[], ('E',reactors):[],('AC',reactors):[],('Cell',reactors):[],('Eth',reactors):[],('CO2',reactors):[],('ACT',reactors):[],('HMF',reactors):[],('Base',reactors):[]})

    # MODEL PREDICTIVE CONTROL LOOP
    disc_time=-1
    current_cycle=-1
    first_shut_down_has_occured=False # Becomes True once a reactor is shut down for the first time, meaning that from now on two reactors will always be operating at a time
    New_cycle=False # Through the MPC, will take the value of one if at the current time point we are starting a new cycle
    transcurred_points_since_last_restart={} # TO KNOW FROM WHERE TO START FIXING VARIABLES
    contador={} 
    for reactors in reactors_list:
        contador[reactors]=0    
    for reactors in reactors_list:
        transcurred_points_since_last_restart[reactors]=[]

    available_Fibers=[] # Available ammount of fibers at current time point. NOTE: May change every new cycle
    available_C5=[] # Available ammount of C5. NOTE: May change every new cycle
    current_available_F=nominal_available_F_per_cycle
    current_available_C5=nominal_available_C5_per_cycle

    used_F_previous_reactor=0
    used_C5_previous_reactor=0
    used_F_previous_reactor_next=0
    used_C5_previous_reactor_next=0

    C0_prev={} 
    breaker=False 
    random.seed(10)
    for current_start_time in pe.RangeSet(start_time,end_time-step_updated,step_updated):
        disc_time=disc_time+1
        # DISTURBANCE
        if disturbance:
            if disc_time==0:
                g_disturbance_F=0
                x_disturbance_F=0
                cs_disturbance_F=0
                xs_disturbance_F=0
                ls_disturbance_F=0
                f_disturbance_F=0
                e_disturbance_F=0
                ac_disturbance_F=0
                act_disturbance_F=0
                hmf_disturbance_F=0
                base_disturbance_F=0

                
                g_disturbance_C5=0
                x_disturbance_C5=0
                cs_disturbance_C5=0
                xs_disturbance_C5=0
                ls_disturbance_C5=0
                f_disturbance_C5=0
                ac_disturbance_C5=0
                act_disturbance_C5=0
                hmf_disturbance_C5=0            
            
            elif disc_time!=0 and disc_time%25==0: 
            
                g_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                x_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                cs_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                xs_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                ls_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                f_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                e_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                ac_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                act_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                hmf_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
                base_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)

                
                g_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                x_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                cs_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                xs_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                ls_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                f_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                ac_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                act_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
                hmf_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)

        else:
            g_disturbance_F=0
            x_disturbance_F=0
            cs_disturbance_F=0
            xs_disturbance_F=0
            ls_disturbance_F=0
            f_disturbance_F=0
            e_disturbance_F=0
            ac_disturbance_F=0
            act_disturbance_F=0
            hmf_disturbance_F=0
            base_disturbance_F=0

            
            g_disturbance_C5=0
            x_disturbance_C5=0
            cs_disturbance_C5=0
            xs_disturbance_C5=0
            ls_disturbance_C5=0
            f_disturbance_C5=0
            ac_disturbance_C5=0
            act_disturbance_C5=0
            hmf_disturbance_C5=0
        
        # Chek if we are starting a new cycle based on operation mode of reactor 1. Depending on the case, update the available ammount of substrate. NOTE: We could update this more often to see what happens, but I will not do that for the moment.
        if r_operation_mode[1][disc_time]==1:
            New_cycle=True
            current_cycle=current_cycle+1
            current_available_F=nominal_available_F_per_cycle # TODO: update randomly every new cycle to see variations in yeast mass every new cycle
            current_available_C5=nominal_available_C5_per_cycle # TODO: update randomly every new cycle to see variations in yeast mass every new cycle
        # TODO: ALSO EVERY NEW CYCLE UPDATE THE PERTURBATIONS IN COMPOSITIONS!!!!
        
        else:
            New_cycle=False

        available_Fibers.append(current_available_F) # NOTE: this is the total ammount in the in the cycle and is not updated based on previous consumption
        available_C5.append(current_available_C5) # NOTE: this is the total ammount in the in the cycle and is not updated based on previous consumption

        # Define optimization model
        
        mad=three_reactors_model(nominal_flow_F,nominal_flow_C5,constant_flows,include_fed_batch_op_time,Inoc_phase_time_wrt_0,Fed_batch_time_wrt_0,available_Fibers[disc_time],available_C5[disc_time],react_list=reactors_list,total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)

        # Initialize model from previously updated model whenever discrete time is not 0

        if disc_time!=0: 
            mad=initialize_model(mad,from_feasible=True,feasible_model='prev_init') 

        # Fix unused variables and deactivate unused constraints
        for r in reactors_list:
            if r_operation_mode[r][disc_time]>1:
                contador[r]=contador[r]+1
            elif r_operation_mode[r][disc_time]<=1:
                contador[r]=0
            transcurred_points_since_last_restart[r].append(contador[r])

            # 0: do not fix anything, else: fix and deactivate equations for  contador[r] +1 points
            # NOTE: I only get feedback from what I am running right now. The remaining reactor/reactors will run in the future, hence there is no feedback for it
            if contador[r]!=0: # If this is not a re-start scheduling point-> fix pH and yeast mass, as well as flows up to contador[r] +1 points
                mad.reactor[r].pH.fix(pe.value(mad.reactor[r].pH))
                mad.reactor[r].M0_yeast.fix(pe.value(mad.reactor[r].M0_yeast))

                for t in mad.reactor[r].t:
                    if mad.reactor[r].t.ord(t)<=contador[r]+1:
                        mad.reactor[r].F_C5liquid[t].fix(pe.value(mad.reactor[r].F_C5liquid[t]))
                        mad.reactor[r].F_liquified_fibers[t].fix(pe.value(mad.reactor[r].F_liquified_fibers[t]))
                        mad.reactor[r].M[t].fix(pe.value(mad.reactor[r].M[t]))

                        mad.reactor[r].Diff_mass[t].deactivate()
                        


                        for j in mad.reactor[r].j:
                            mad.reactor[r].C[t,j].fix(pe.value(mad.reactor[r].C[t,j]))
                            mad.reactor[r].Diff_comp[t,j].deactivate()
            else:
                mad.reactor[r].F_C5liquid[mad.reactor[r].t.first()].fix(0)
                mad.reactor[r].F_liquified_fibers[mad.reactor[r].t.first()].fix(0)            

            # If I am turning off a reactor, then I must include a constraint to guarantee that I am not using more than the available substrate per cycle
            if contador[r]==finite_elem_t_fer:
                used_F_previous_reactor_next=pe.value(mad.share_F[r]) 
                used_C5_previous_reactor_next=pe.value(mad.share_C5[r])

        if r_operation_mode[1][disc_time]==0:
            first_shut_down_has_occured=True 
        if first_shut_down_has_occured and max(contador[r] for r in reactors_list)!=finite_elem_t_fer:
            def _Integral_past_F(mad):
                return sum(mad.share_F[r] for r in reactors_list if r_operation_mode[r][disc_time]>=1) + used_F_previous_reactor_next == available_Fibers[disc_time]
            mad.Integral_past_F=pe.Constraint(rule=_Integral_past_F)

            def _Integral_past_C5(mad):
                return sum(mad.share_C5[r] for r in reactors_list if r_operation_mode[r][disc_time]>=1) + used_C5_previous_reactor_next == available_C5[disc_time]
            mad.Integral_past_C5=pe.Constraint(rule=_Integral_past_C5)

            used_F_previous_reactor=used_F_previous_reactor_next
            used_C5_previous_reactor=used_C5_previous_reactor_next
        elif first_shut_down_has_occured and max(contador[r] for r in reactors_list)==finite_elem_t_fer:
            def _Integral_past_F(mad):
                return sum(mad.share_F[r] for r in reactors_list if r_operation_mode[r][disc_time]>=1) + used_F_previous_reactor == available_Fibers[disc_time]
            mad.Integral_past_F=pe.Constraint(rule=_Integral_past_F)

            def _Integral_past_C5(mad):
                return sum(mad.share_C5[r] for r in reactors_list if r_operation_mode[r][disc_time]>=1) + used_C5_previous_reactor == available_C5[disc_time]
            mad.Integral_past_C5=pe.Constraint(rule=_Integral_past_C5)

        if (not constant_flows) and (include_control_actions_constraint):
            mad.control_act_C5={}
            mad.control_act_F={}
            mad.control_act_2_C5={}
            mad.control_act_2_F={}

            for r in reactors_list:


                def _control_act_C5(mad,t):
                    if t==mad.reactor[r].t.first() or t==mad.reactor[r].t.next(mad.reactor[r].t.first())  or mad.reactor[r].t.ord(t)<=contador[r]+1:
                        return pe.Constraint.Skip
                    else:
                        return  mad.reactor[r].F_C5liquid[t]-mad.reactor[r].F_C5liquid[mad.reactor[r].t.prev(t)]  <=control_actions_val

                mad.control_act_C5[r]=pe.Constraint(mad.reactor[r].t,rule=_control_act_C5)
                setattr(mad,'control_act_C5_%r' %r,mad.control_act_C5[r])
                def _control_act_F(mad,t):
                    if t==mad.reactor[r].t.first() or t==mad.reactor[r].t.next(mad.reactor[r].t.first()) or mad.reactor[r].t.ord(t)<=contador[r]+1:
                        return pe.Constraint.Skip
                    else:
                        return      mad.reactor[r].F_liquified_fibers[t]-mad.reactor[r].F_liquified_fibers[mad.reactor[r].t.prev(t)]  <=control_actions_val

                mad.control_act_F[r]=pe.Constraint(mad.reactor[r].t,rule=_control_act_F)
                setattr(mad,'control_act_F_%r' %r,mad.control_act_F[r])   


                def _control_act_C5_2(mad,t):
                    if t==mad.reactor[r].t.first() or t==mad.reactor[r].t.next(mad.reactor[r].t.first()) or mad.reactor[r].t.ord(t)<=contador[r]+1:
                        return pe.Constraint.Skip
                    else:
                        return mad.reactor[r].F_C5liquid[t]-mad.reactor[r].F_C5liquid[mad.reactor[r].t.prev(t)] >=-control_actions_val 
                mad.control_act_2_C5[r]=pe.Constraint(mad.reactor[r].t,rule=_control_act_C5_2)
                setattr(mad,'control_act_C5_2_%r' %r,mad.control_act_2_C5[r])

                def _control_act_F_2(mad,t):
                    if t==mad.reactor[r].t.first() or t==mad.reactor[r].t.next(mad.reactor[r].t.first()) or mad.reactor[r].t.ord(t)<=contador[r]+1:
                        return pe.Constraint.Skip
                    else:
                        return  mad.reactor[r].F_liquified_fibers[t]-mad.reactor[r].F_liquified_fibers[mad.reactor[r].t.prev(t)]  >= -control_actions_val

                mad.control_act_2_F[r]=pe.Constraint(mad.reactor[r].t,rule=_control_act_F_2)
                setattr(mad,'control_act_F_2_%r' %r,mad.control_act_2_F[r])  
        # Solve open-loop optimal control problem

        opt1 = SolverFactory('gams') # Solve problem

        for solver_used in solver_list:
            mad.results = opt1.solve(mad, solver=solver_used, tee=tee)

            if mad.results.solver.termination_condition == 'infeasible' or mad.results.solver.termination_condition == 'other' or mad.results.solver.termination_condition == 'unbounded' or mad.results.solver.termination_condition == 'invalidProblem' or mad.results.solver.termination_condition == 'solverFailure' or mad.results.solver.termination_condition == 'internalSolverError' or mad.results.solver.termination_condition == 'error'  or mad.results.solver.termination_condition == 'resourceInterrupt' or mad.results.solver.termination_condition == 'licensingProblem' or mad.results.solver.termination_condition == 'noSolution' or mad.results.solver.termination_condition == 'noSolution' or mad.results.solver.termination_condition == 'intermediateNonInteger':
                mad.dsda_status = 'Evaluated_Infeasible'
                if disc_time!=0:
                    mad=initialize_model(mad,from_feasible=True,feasible_model='prev_init')  
            else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                mad.dsda_status = 'Optimal'
                break

        print('Iteration:',disc_time,'--Status:',mad.dsda_status,'--last solver used:',solver_used)
        if mad.dsda_status=='Evaluated_Infeasible':
            break
                


        # Print relevant optimization information
        print('Share F (optimization)','  |  ',        pe.value(mad.share_F[1])     ,'  |  ',        pe.value(mad.share_F[2])     ,'  |  ',        pe.value(mad.share_F[3]) )
        print('Share C5 (optimization)','  |  ',        pe.value(mad.share_C5[1])     ,'  |  ',        pe.value(mad.share_C5[2])     ,'  |  ',        pe.value(mad.share_C5[3]) )
        print('Yest Mass (optimization)','  |  ',        pe.value(mad.reactor[1].M0_yeast)     ,'  |  ',        pe.value(mad.reactor[2].M0_yeast)     ,'  |  ',        pe.value(mad.reactor[3].M0_yeast) )

        print(current_start_time,'  |  ',r_operation_mode[1][disc_time],'  |  ',r_operation_mode[2][disc_time],'  |  ',r_operation_mode[3][disc_time],'  |  ',transcurred_points_since_last_restart[1][disc_time],'  |  ',transcurred_points_since_last_restart[2][disc_time],'  |  ',transcurred_points_since_last_restart[3][disc_time])


        # Simulate optimal control action using one time step
        # --1) retrieve optimal control actions
 

        reactors_evalauted=0
        for r in reactors_list:
            # if reactor is on and will keep on for next time interval
            if disc_time+1<len(r_operation_mode[r]):
                condition=r_operation_mode[r][disc_time]>=1 and r_operation_mode[r][disc_time+1]>=1
            else:
                condition= r_operation_mode[r][disc_time]>=1
            if condition:
                reactors_evalauted=reactors_evalauted+1
                position_to_take_control_action=contador[r]+2
                position_to_take_state=contador[r]+1
                posotion_to_update_state_inCOM=position_to_take_control_action


                optimal_pH=pe.value(mad.reactor[r].pH)
                optimal_yeast=pe.value(mad.reactor[r].M0_yeast)

                optimal_F=pe.value(mad.reactor[r].F_liquified_fibers[mad.reactor[r].t[position_to_take_control_action]])
                optimal_C5=pe.value(mad.reactor[r].F_C5liquid[mad.reactor[r].t[position_to_take_control_action]])

                M0_prev=pe.value(mad.reactor[r].M[mad.reactor[r].t[position_to_take_state]])
                for j in mad.reactor[r].j:
                    C0_prev[j]=pe.value(mad.reactor[r].C[mad.reactor[r].t[position_to_take_state],j])
            
                mad_sim=build_fermentation_one_time_step_new(total_sim_time=step_updated,
                                                            discretization=sim_discretization,
                                                            n_f_elements_t=sim_n_finite_elements,
                                                            current_start_time_sconds=current_start_time,
                                                            M0_prev_input=M0_prev,
                                                            C0_prev_input=C0_prev,
                                                            pH_val=optimal_pH,
                                                            M_yeast=optimal_yeast,
                                                            F_fibers=optimal_F,
                                                            F_C5=optimal_C5,
                                                            glucose_disturbance_F=g_disturbance_F,
                                                            xylose_disturbance_F=x_disturbance_F,
                                                            cs_disturbance_F=cs_disturbance_F,
                                                            xs_disturbance_F=xs_disturbance_F,
                                                            ls_disturbance_F=ls_disturbance_F,
                                                            f_disturbance_F=f_disturbance_F,
                                                            e_disturbance_F=e_disturbance_F,
                                                            ac_disturbance_F=ac_disturbance_F,
                                                            act_disturbance_F=act_disturbance_F,
                                                            hmf_disturbance_F=hmf_disturbance_F,
                                                            base_disturbance_F=base_disturbance_F,
                                                            glucose_disturbance_C5=g_disturbance_C5,
                                                            xylose_disturbance_C5=x_disturbance_C5,
                                                            cs_disturbance_C5=cs_disturbance_C5,
                                                            xs_disturbance_C5=xs_disturbance_C5,
                                                            ls_disturbance_C5=ls_disturbance_C5,
                                                            f_disturbance_C5=f_disturbance_C5,
                                                            ac_disturbance_C5=ac_disturbance_C5,
                                                            act_disturbance_C5=act_disturbance_C5,
                                                            hmf_disturbance_C5=hmf_disturbance_C5) 

                if contador[r]!=0:
                    mad_sim=initialize_model(mad_sim,from_feasible=True,feasible_model='prev_init_sim_'+str(r))  
                
                # Solve simulation

                opt2 = SolverFactory('gams') # Solve problem

                for solver_used in simulation_solvers:
                    mad_sim.results = opt2.solve(mad_sim, solver=solver_used, tee=tee)

                    if mad_sim.results.solver.termination_condition == 'infeasible' or mad_sim.results.solver.termination_condition == 'other' or mad_sim.results.solver.termination_condition == 'unbounded' or mad_sim.results.solver.termination_condition == 'invalidProblem' or mad_sim.results.solver.termination_condition == 'solverFailure' or mad_sim.results.solver.termination_condition == 'internalSolverError' or mad_sim.results.solver.termination_condition == 'error'  or mad_sim.results.solver.termination_condition == 'resourceInterrupt' or mad_sim.results.solver.termination_condition == 'licensingProblem' or mad_sim.results.solver.termination_condition == 'noSolution' or mad_sim.results.solver.termination_condition == 'noSolution' or mad_sim.results.solver.termination_condition == 'intermediateNonInteger':
                        mad_sim.dsda_status = 'Evaluated_Infeasible'

                    else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                        mad_sim.dsda_status = 'Optimal'
                        break
                print('reactor ',r,' simulation')
                print('Iteration:',disc_time,'--Simulation Status:',mad_sim.dsda_status,'--last solver used:',solver_used)
                if mad_sim.dsda_status=='Evaluated_Infeasible':
                    breaker=True  
                
                # Update original model with new states
                mad.reactor[r].M[mad.reactor[r].t[posotion_to_update_state_inCOM]].value=pe.value(mad_sim.M[mad_sim.t.last()])
                for j in mad.reactor[r].j:
                    mad.reactor[r].C[mad.reactor[r].t[posotion_to_update_state_inCOM],j].value=pe.value(mad_sim.C[mad_sim.t.last(),j])
            
                #save reletant information
                for t in mad_sim.t:
                    if reactors_evalauted==1:
                        time_list.append((mad_sim.current_starting_time+t*(mad_sim.current_final_time-mad_sim.current_starting_time))*(1/(60*60)))
                    if contador[r]==finite_elem_t_fer-1 and t == mad_sim.t.last():
                        time_point_obj_evaluation_dict[r].append(1)
                    else:
                        time_point_obj_evaluation_dict[r].append(0)
                    Hold_up_dict[r].append(pe.value(mad_sim.M[t]))
                    pH_dict[r].append(pe.value(mad_sim.pH))
                    yeast_dict[r].append(pe.value(mad_sim.M0_yeast))
                    C5_dict[r].append(pe.value(mad_sim.F_C5liquid))
                    fiber_dict[r].append(pe.value(mad_sim.F_liquified_fibers))
                    for j in mad_sim.j:
                        Concentration_dict[(j,r)].append(pe.value(mad_sim.C[t,j]))  
                        Concentration_disturbance_dict_C5[(j,r)].append(mad_sim.C_C5liquid[j])     
                        Concentration_disturbance_dict_F[(j,r)].append(mad_sim.C_liquified_fibers[j])                

                if contador[r]==finite_elem_t_fer-1:
                    objective_dict[r].append(50*yeast_dict[r][-1]-5*Concentration_dict[('Eth',r)][-1]*Hold_up_dict[r][-1])

                generate_initialization(m=mad_sim,model_name='prev_init_sim_'+str(r))                

            # If reactor is off
            else:
                #save reletant information

                for t in mad_sim.t:
                    Hold_up_dict[r].append(0)
                    pH_dict[r].append(0)
                    yeast_dict[r].append(0)
                    C5_dict[r].append(0)
                    fiber_dict[r].append(0)
                    time_point_obj_evaluation_dict[r].append(0)
                    for j in mad_sim.j:
                        Concentration_dict[(j,r)].append(0)
                        Concentration_disturbance_dict_C5[(j,r)].append(0)     
                        Concentration_disturbance_dict_F[(j,r)].append(0) 

        if breaker:
            break


        generate_initialization(m=mad,model_name='prev_init')




    test_name='Traditional_final'
    with open('saved_'+test_name,'wb') as save_file:
        pickle.dump([time_list,Hold_up_dict,pH_dict,yeast_dict,C5_dict,fiber_dict,Concentration_dict,objective_dict,time_point_obj_evaluation_dict,Concentration_disturbance_dict_C5,Concentration_disturbance_dict_F],save_file)
        print('data saved successfully to file')
