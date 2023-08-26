from __future__ import division
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.errors import InfeasibleConstraintException
import time
import logging
from model_serializer import StoreSpec, from_json, to_json 
import math
import os

# PYTHON VERSION: cpython 3.10.11, windows 10
# PYOMO VERSION: 6.4.4
# GAMS VERSION: 41.5.0


def solve_subproblem(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 0,
) -> pe.ConcreteModel():
    """
    Function that checks feasibility and optimizes subproblem model.
    Note integer variables have to be previously fixed in the external reformulation
    Args:
        m: Fixed subproblem model that is to be solved
        subproblem_solver: MINLP or NLP solver algorithm
        timelimit: time limit in seconds for the solve statement
        gams_output: Determine keeping or not GAMS files
        tee: Display iteration output
        rel_tol: Relative optimality tolerance
    Returns:
        m: Solved subproblem model
    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        preprocess_problem(m, simple=True)

    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        return m
    end_prep=time.time()
    m.dsda_usertime =m.dsda_usertime + (end_prep-start_prep)
    output_options = {}

    # Output report
    if gams_output:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        gams_path = os.path.join(dir_path, "gamsfiles/")
        if not(os.path.exists(gams_path)):
            print('Directory for automatically generated files ' +
                  gams_path + ' does not exist. We will create it')
            os.makedirs(gams_path)
        output_options = {'keepfiles': True,
                          'tmpdir': gams_path,
                          'symbolic_solver_labels': True}

    subproblem_solver_options['add_options'] = subproblem_solver_options.get(
        'add_options', [])
    subproblem_solver_options['add_options'].append(
        'option reslim=%s;' % timelimit)
    subproblem_solver_options['add_options'].append(
        'option optcr=%s;' % rel_tol)
    # Solve
    solvername = 'gams'
    
    if subproblem_solver=='OCTERACT':
        opt = SolverFactory(solvername)
    else:
        opt = SolverFactory(solvername, solver=subproblem_solver)

    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options,
                          skip_trivial_constraints=True,
                          )

    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time

    # Assign D-SDA status
    if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
        m.dsda_status = 'Evaluated_Infeasible'
    else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
        m.dsda_status = 'Optimal'
    # if m.results.solver.termination_condition == 'locallyOptimal' or m.results.solver.termination_condition == 'optimal' or m.results.solver.termination_condition == 'globallyOptimal':
    #     m.dsda_status = 'Optimal'

    return m

def preprocess_problem(m, simple: bool = True):
    """
    Function that applies certain tranformations to the mdoel to first verify that it is not trivially 
    infeasible (via FBBT) and second, remove extra constraints to help NLP solvers
    Args:
        m: MI(N)LP model that is going to be preprocessed
        simple: Boolean variable to carry on a simple preprocessing (only FBBT) or a more complete one, prone to fail
    Returns:

    """
    if not simple:
        pe.TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        pe.TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        pe.TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        pe.TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        pe.TransformationFactory(
            'contrib.constraints_to_var_bounds').apply_to(m)
        pe.TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        pe.TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
            m, tmp=False, ignore_infeasible=True)
    # fbbt(m)

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

def solve_with_minlp(
    m: pe.ConcreteModel(),
    transformation: str = 'bigm',
    minlp: str = 'baron',
    minlp_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 0.001,
    transform_required: bool = True
) -> pe.ConcreteModel():
    """
    Function that transforms a GDP model and solves it as a mixed-integer nonlinear
    programming (MINLP) model.
    Args:
        m: Pyomo GDP model that is to be solved using MINLP
        transformation: GDP to MINLP transformation to be used
        minlp: MINLP solver algorithm
        minlp_options: MINLP solver algorithm options
        timelimit: time limit in seconds for the solve statement
        gams_output: Determine keeping or not GAMS files
        tee: Dsiplay iterations
        rel_tol: Relative optimality tolerance
    Returns:
        m: Solved MINLP model
    """

    # Transformation step
    if transform_required:
        pe.TransformationFactory('core.logical_to_linear').apply_to(m)
        transformation_string = 'gdp.' + transformation
        pe.TransformationFactory(transformation_string).apply_to(m)

    # Output report
    output_options = {}
    if gams_output:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        gams_path = os.path.join(dir_path, "gamsfiles/")
        if not(os.path.exists(gams_path)):
            print('Directory for automatically generated files ' +
                  gams_path + ' does not exist. We will create it')
            os.makedirs(gams_path)
        output_options = {'keepfiles': True,
                          'tmpdir': gams_path,
                          'symbolic_solver_labels': True}

    minlp_options['add_options'] = minlp_options.get('add_options', [])
    minlp_options['add_options'].append('option reslim=%s;' % timelimit)
    minlp_options['add_options'].append('option optcr=%s;' % rel_tol)

    # Solve
    solvername = 'gams'
    if minlp=='OCTERACT':
        opt = SolverFactory(solvername)
    else:
        opt = SolverFactory(solvername, solver=minlp)
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **minlp_options,
                          )
    # update_boolean_vars_from_binary(m)
    return m

def scheduling_and_control_gdp_N_GBD(x_initial: list=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2],last_time_hours: float=14, demand_p1_kmol: float=1,demand_p2_kmol: float=1):
    # Data
    Infty=10 
    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='scheduling_control_gdp')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=0.5,doc='lenght of time periods of discretized time grid for scheduling [units of time]') #TODO: Update as required
    m.lastT=pe.Param(initialize=math.floor(last_time_hours/m.delta),doc='last discrete time value in the scheduling time grid') #TODO: Update as required
    

    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.Q = pe.Set(initialize=['A','B','C','D','E','F'],doc='Chemical species')#TODO: Note that here I only consider species relevant for the dynamic model
    m.J=pe.Set(initialize=['Mix','R_large','R_small','Sep','Pack'],doc='Set of Units')
    m.I=pe.Set(initialize=['Mix','R1','R2','R3','Sep','Pack1','Pack2'], doc='Set of tasks')
    m.K=pe.Set(initialize=['S1','M1','M2','M3','W1','P1','P2','I1','I2','I3','I4','I5','I6'],doc='Set of states')
    #Subsets
    m.J_reactors=pe.Set(initialize=['R_large','R_small'],within=m.J)
    m.I_reactions=pe.Set(initialize=['R1','R2','R3'],within=m.I)   
    m.J_noDynamics=m.J-m.J_reactors
    m.I_noDynamics=m.I-m.I_reactions
    m.K_inputs=pe.Set(initialize=['S1','M1','M2','M3'],within=m.K)
    m.K_products=pe.Set(initialize=['P1','P2'],within=m.K)
    #----------Scalars that depend on sets
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of nntime]')
    # -----------parameters--------------------------------------------------
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
    
    m.gamma=pe.Param(m.K,initialize={'S1':Infty,'M1':Infty,'M2':Infty,'M3':Infty,'W1':Infty,'P1':Infty,'P2':Infty,'I1':2,'I2':2,'I3':2,'I4':2,'I5':5,'I6':5},default=0,doc="maximum amount of material k that can be stored [m^3]")


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
    
    
    def _demand(m,K,T):
        if K=='P1' and T==m.lastT:
            return (demand_p1_kmol)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        elif K=='P2' and T==m.lastT:
            return (demand_p2_kmol)/sum(m.C[K,Q] for Q in m.Q) #1 is the parameter in you article
        else:
            return 0
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="Minimum demand of material k at time t [m^3]")
    m.S0=pe.Param(m.K,initialize={'M1':Infty,'M2':Infty,'M3':Infty,'S1':Infty},default=0,doc="Initial amount of state k [m^3]") #You is not reporting this, so I am assuming it is infinity. This makes sense with the objective function his defines if it is assumed that raw material is available whenever we want to buy it, and that it can instantanelusly go to our production facility

    _fixed_cost={}
    _fixed_cost['Mix','Mix']=10

    _fixed_cost['R1','R_large']=30
    _fixed_cost['R1','R_small']=20

    _fixed_cost['R2','R_large']=30
    _fixed_cost['R2','R_small']=20

    _fixed_cost['R3','R_large']=30
    _fixed_cost['R3','R_small']=20

    _fixed_cost['Sep','Sep']=100

    _fixed_cost['Pack1','Pack']=50
    _fixed_cost['Pack2','Pack']=50

    m.fixed_cost=pe.Param(m.I,m.J,default=0,initialize=_fixed_cost,doc="Fixed cost to run task i in unit j [m.u./batch]")

    _variable_cost_param={}
    _variable_cost_param['Mix','Mix']=30

    _variable_cost_param['Sep','Sep']=100

    _variable_cost_param['Pack1','Pack']=50
    _variable_cost_param['Pack2','Pack']=50

    m.variable_cost=pe.Param(m.I,m.J,default=0,initialize=_variable_cost_param,doc="Variabe batch cost [m.u/m^3]") 

    def _raw_cost(m,K):
        if K=='S1':
            return 0 
        elif K=='M1': #A
            return 100*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M2': #b
            return 150*sum(m.C[K,Q] for Q in m.Q)
        elif K=='M3':#c
            return 200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.raw_cost=pe.Param(m.K,default=0,initialize=_raw_cost,doc='Unit cost of raw materials [m.u./m^3]')

    m.hot_cost=pe.Param(initialize=10,doc='Unit cost of heating fluid [m.u./m^3]')
    m.cold_cost=pe.Param(initialize=1,doc='Unit cost of cooling fluid [m.u./m^3]')

    def _revenue(m,K):
        if K=='P1':
            return 700*sum(m.C[K,Q] for Q in m.Q)
        elif K=='P2':
            return 1200*sum(m.C[K,Q] for Q in m.Q)
        else:
            return 0
    m.revenue=pe.Param(m.K,default=0,initialize=_revenue,doc='revenue from selling one unit of material k [m.u/m^3]')


    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')


    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['Sep','Sep']=3 

    _tau_p['Pack1','Pack']=1.5 
    _tau_p['Pack2','Pack']=1.5 

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,mutable=True,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(pe.value(m.tau_p[I,J])/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,mutable=True,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    # # -----------scheduling variables -----------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,bounds=(0,1),doc='1 if unit j processes task i starting at time t')   
    # help(pe.Var)
    def _B_bounds(m,I,J,T):
        return (0,m.beta_max[I,J])
    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_B_bounds,initialize=0,doc='Batch size of task i processed in unit j starting at time t')
    def _S_bounds(m,K,T):
        return (0,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,initialize=0,doc='Inventory of material k at time t')

    # Auxiliary ariables required to decrease combinatorial complexity
    m.sumX=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=(0,m.lastT+1),initialize=0,doc='Auxiliary variable 1 for disjunctive section. Appears in UNIT UTILIZATION cosntraints ')
    def _B_shift_bounds(m,I,J,T):
        return (0,m.beta_max[I,J])
    m.B_shift=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,bounds=_B_shift_bounds,initialize=0,doc='Auxiliary variable 2 for disjunctive section. Appears in MATERIAL BALANCES. This is a time-shifted version of variable B')
    # # ----------Reactor variables that do not depend on disjunctions------------------------------------------------------
    def _Vreactor_bounds(m,I,J):
        return (m.model().beta_min[I,J],m.model().beta_max[I,J])
    m.Vreactor=pe.Var(m.I_reactions,m.J_reactors,within=pe.NonNegativeReals,bounds=_Vreactor_bounds,doc='Reactive mixture volume for reaction I in reactor J [m^3]') #TODO: link this variable with batch size variables

    # # ----------Scheduling Constraints that DO NOT depend on disjunctions-----------------------------------------
    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    def _E_DEMAND_SATISFACTION(m,K):
        return m.S[K,m.lastT]>=m.demand[K,m.lastT]
    m.E_DEMAND_SATISFACTION=pe.Constraint(m.K_products,rule=_E_DEMAND_SATISFACTION,doc='INVENTORY LEVEL OF PRODUCTS NEEDS TO MEET THE ORDER DEMAND')
        
        
    def _E1_UNIT(m,J,T):
        return sum(m.sumX[I,J,T] for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _DEF_AUX1_INDEP(m,I,J,T):
        return m.sumX[I,J,T]==sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-pe.value(m.tau[I,J])+1)
    m.DEF_AUX1_INDEP=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_DEF_AUX1_INDEP,doc='Definition of auxiliary variable 1. Unit-tasks independent of disjunctions')
    
    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B_shift[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _DEF_AUX2_INDEP(m,I,J,T):
        if T==0:        
            return pe.Constraint.Skip
        elif T-pe.value(m.tau[I,J])>=0:
            return m.B_shift[I,J,T]==m.B[I,J,T-pe.value(m.tau[I,J])]
        else:
            return m.B_shift[I,J,T]==0
    m.DEF_AUX2_INDEP=pe.Constraint(m.I_noDynamics,m.J_noDynamics,m.T,rule=_DEF_AUX2_INDEP,doc='Definition of auxiliary variable 2. Unit-tasks independent of disjunctions')

    #*****DISJUNCTIVE SECTION**********************************   
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

    m.ordered_set={}
    m.YR={}
    m.oneYR={}
    m.YR_disjunct={}
    m.Disjunction1={}
    positcui=-1
    for I in m.I_reactions:
        for J in m.J_reactors:
            positcui=positcui+1
            m.ordered_set[I,J]=pe.RangeSet(m.minTau[I,J],m.maxTau[I,J],doc='Ordered set for each reaction-reactor pair') 
            setattr(m,'ordered_set_%s_%s' %(I,J),m.ordered_set[I,J])

            def _YRinit(m,ordered_set):
                if ordered_set==(x_initial[positcui]+m.minTau[I,J]-1):
                    return True 
                else:
                    return False
            m.YR[I,J]=pe.BooleanVar(m.ordered_set[I,J],initialize=_YRinit)
            setattr(m,'YR_%s_%s' %(I,J),m.YR[I,J])            

            #Constraint that allow to apply the reformulation over YR
            def _select_one(m):
                return pe.exactly(1,m.YR[I,J])
            m.oneYR[I,J]=pe.LogicalConstraint(rule=_select_one) 
            setattr(m,'oneYR_%s_%s' %(I,J),m.oneYR[I,J])  

            # Declaration of disjuncts
            def _build_disjuncts(m,indexTau):  #Disjuncts for first Boolean variable
                m.model().tau[I,J]=indexTau
                m.model().tau_p[I,J]=pe.value(m.model().tau[I,J])*m.model().delta #Both times are assumed to be discrete
                # #----------- Variable processing times----------------------------------------------------------------
                def _DEF_VAR_TIME(m):
                    return m.model().varTime[I,J]==min([pe.value(m.model().tau_p[I,J]),m.model().maxTau[I,J]*m.model().delta])
                m.DEF_VAR_TIME=pe.Constraint(rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
                # m.DEF_VAR_TIME.display()

                # # --------- Constraint for Aux variable 1-------------------------------------------------------------
                def _DEF_AUX1(m,T):
                    return m.model().sumX[I,J,T]==sum(m.model().X[I,J,TP] for TP in m.model().T if TP<=T and TP>=T-pe.value(m.model().tau[I,J])+1)
                m.DEF_AUX1=pe.Constraint(m.model().T,rule=_DEF_AUX1,doc='Definition of auxiliary variable 1')
                # # --------- Constraint for Aux variable 2-------------------------------------------------------------
                def _DEF_AUX2(m,T):
                    if T==0:        
                        return pe.Constraint.Skip
                    elif T-pe.value(m.model().tau[I,J])>=0:
                        return m.model().B_shift[I,J,T]==m.model().B[I,J,T-pe.value(m.model().tau[I,J])]
                    else:
                        return m.model().B_shift[I,J,T]==0
                m.DEF_AUX2=pe.Constraint(m.model().T,rule=_DEF_AUX2,doc='Definition of auxiliary variable 2')
                # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------    
            m.YR_disjunct[I,J]=Disjunct(m.ordered_set[I,J],rule=_build_disjuncts,doc="each disjunct defines those constraints that are activated depending on the selected tau")    
            setattr(m,'YR_Disjunct_%s_%s' %(I,J),m.YR_disjunct[I,J])
            
            #Create disjunction
            def Disjunction1(m):    #Disjunction for first Boolean variable
                return [m.YR_disjunct[I,J][dis_set] for dis_set in m.ordered_set[I,J]]
            m.Disjunction1[I,J]=Disjunction(rule=Disjunction1,xor=True)
            setattr(m,'Disjunction1_%s_%s' %(I,J),m.Disjunction1[I,J])

            # Associate disjuncts with boolean variables
            for index in m.ordered_set[I,J]:
                m.YR[I,J][index].associate_binary_var(m.YR_disjunct[I,J][index].indicator_var)

    #****END OF DISJUNCTIVE SECTION*****************************


    # # ----------Linking constraints-------------------------------------------
    #1) Reactor volumes and scheduling capacities
    def _linking1_1(m,I,J,T):
        return m.B[I,J,T]-m.Vreactor[I,J] <= (m.beta_max[I,J]-m.beta_min[I,J])*(1-m.X[I,J,T])  
    m.linking1=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_1,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 

    def _linking1_2(m,I,J,T):
        return -(m.B[I,J,T]-m.Vreactor[I,J]) <= m.beta_max[I,J]*(1-m.X[I,J,T])  
    m.linking2=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking1_2,doc='Linking constraint to fuarantee that batch sizes agree with reactor volumes') 
      
    #There is an important assumption here (discussed before): If a given task I is executed multiple times in reactor J, then it is always executed the same way, i.e., same batch size, same time 

    #-----------Reactors dynamic models--------------------------------

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
                return (295,m.T_R_max[J])  
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
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
            


    # # -------Discretization---------------------------------------------------

    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=9*(1) #Keep Fhot constant every three discretization points %TODO: what I should keep constant is the actual sampling time, not the number of discrete points
    keep_constant_Fcold=9*(1) #Keep Fcold constant every three discretization points  %TODO: what I should keep constant is the actual sampling time, not the number of discrete points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:        #TODO: Depending on selected variable time the number of discretization points must change accordingly
            discretizer.apply_to(m, nfe=30*(1), ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            
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
  
    # # -------Reformulation----------------------------------------------------
    def _I_J(m):
        return ((I,J) for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1)
    m.I_J=pe.Set(dimen=2,initialize=_I_J,doc='task-unit nodes')
    #m.I_J.display()
    def _lastN(m,I,J):
        if I in m.I_reactions and J in m.J_reactors: 
            return math.floor((m.T.__len__()-1)/m.minTau[I,J])  #TODO: Note that I am using the minimum, or I can use Tau, but I would have to incorporate this within the disjunction.
        else:
            return math.floor((m.T.__len__()-1)/pe.value(m.tau[I,J]))
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')

    def _Nref_bounds(m,I,J):
        return (0,m.lastN[I,J])
    m.Nref=pe.Var(m.I_J,within=pe.Integers,bounds=_Nref_bounds,initialize=0,doc='reformulation variables from 0 to lastN')

    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==m.Nref[I,J]
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    #-----------Objective function----------------------------------------------

    m.TCP1=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Fixed costs for all unit-tasks')
    def _C_TCP1(m):
        return  m.TCP1==sum(sum(sum(m.fixed_cost[I, J]*m.X[I, J, T]for J in m.J) for I in m.I) for T in m.T) 
    m.C_TCP1=pe.Constraint(rule=_C_TCP1)
    m.TCP2=pe.Var(within=pe.Reals,initialize=0,doc='TPC: Variable cost for unit-tasks that do not consider dynamics')
    def _C_TCP2(m):
        return m.TCP2==sum(sum(sum(m.variable_cost[I, J]*m.B[I, J, T] for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    m.C_TCP2=pe.Constraint(rule=_C_TCP2)
    m.TCP3=pe.Var(within=pe.NonNegativeReals,initialize=0,doc='TPC: Variable cost for unit-tasks that do consider dynamics')
    def _C_TCP3(m):
        return m.TCP3== sum(sum(sum(m.X[I, J, T]*(m.hot_cost*m.Integral_hot[I, J][m.N[I, J].last()] + m.cold_cost*m.Integral_cold[I, J][m.N[I, J].last()]) for T in m.T) for I in m.I_reactions)for J in m.J_reactors)
    m.C_TCP3=pe.Constraint(rule=_C_TCP3)  
    m.TMC= pe.Var(within=pe.Reals,initialize=0,doc='TMC: Total material cost')
    def _C_TMC(m):
        return m.TMC==sum(m.raw_cost[K]*(m.S0[K]-m.S[K, m.lastT]) for K in m.K_inputs) 
    m.C_TMC=pe.Constraint(rule=_C_TMC)
    m.SALES=pe.Var(within=pe.Reals,initialize=0,doc='SALES: Revenue form selling products')
    def _C_SALES(m):
        return m.SALES==sum(m.revenue[K]*m.S[K, m.lastT] for K in m.K_products)
    m.C_SALES=pe.Constraint(rule=_C_SALES)



    def _obj_scheduling(m):
        return m.TCP1+m.TCP2+m.TCP3+m.TMC-m.SALES 
    m.obj = pe.Objective(rule=_obj_scheduling, sense=pe.minimize)  

    return m



if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


    print('\n-------GENERALIZED BENDERS DECOMPOSITION TEST-------------------------------------')
    ######-------------------SOLVER AND MODEL DECLARATION -------------------------##################
    mip_solver='CPLEX'
    nlp_solver='conopt4'
    transform='bigm'
    sub_options={'add_options':['GAMS_MODEL.threads=0;']} #Subproblem solver options: use all available threads
    Infinity_aprox=100000
    kwargs={}
    kwargs2=kwargs.copy()
    model_fun=scheduling_and_control_gdp_N_GBD

    #####------------------COMPUTATIONAL EXPERIMENTS-------------------------######################


   # 1: Naive execution of GBD
   # 2: Execution of GBD with heuristic cuts: minimum processing time s.t. variable capacity
   # 3: Execution of GBD with heuristic cuts: minimum processing time s.t. fixed capacity at its maximum
   # 4: Execution of GBD with heuristic cuts: no-good cuts in master problem
   # 5: Execution of GBD with heuristic cuts: no-good cuts in master problem, only for binary variables related to variable processing times
    experiment=1
    auxiliary_cuts=True
    if auxiliary_cuts: 
        if experiment ==1:
            best_sol_name='current_best_GBD_V3_subproblem_naive__improved_Feasibility'
        elif experiment ==2:
            best_sol_name='current_best_GBD_V3_subproblem_min_t_vary_B__improved_Feasibility'
        elif experiment ==3:
            best_sol_name='current_best_GBD_V3_subproblem_min_t_fix_B__improved_Feasibility'
        elif experiment ==4:
            best_sol_name='current_best_GBD_V3_subproblem_no_good_all__improved_Feasibility'
        elif experiment ==5:
            best_sol_name='current_best_GBD_V3_subproblem_no_good_t_only__improved_Feasibility'
    else:    
        if experiment ==1:
            best_sol_name='current_best_GBD_V3_subproblem_naive'
        elif experiment ==2:
            best_sol_name='current_best_GBD_V3_subproblem_min_t_vary_B'
        elif experiment ==3:
            best_sol_name='current_best_GBD_V3_subproblem_min_t_fix_B'
        elif experiment ==4:
            best_sol_name='current_best_GBD_V3_subproblem_no_good_ally'
        elif experiment ==5:
            best_sol_name='current_best_GBD_V3_subproblem_no_good_t_only'
    ######-------------------------- MATER PROBLEM ------------------------------##################
    mas=model_fun(**kwargs2) #master problem
    mas.cuts=pe.ConstraintList() #Initialize benders cuts

    #Deactivate subproblem constraint
    for I in mas.I_reactions:
        for J in mas.J_reactors:
            mas.c_dCdtheta[I,J].deactivate()
            mas.c_dTRdtheta[I,J].deactivate()                        
            mas.c_dTJdtheta[I,J].deactivate()
            mas.c_dIntegral_hotdtheta[I,J].deactivate()
            mas.c_dIntegral_colddtheta[I,J].deactivate()
            mas.Constant_control1[I,J].deactivate()                        
            mas.Constant_control2[I,J].deactivate()
            mas.finalCon[I,J].deactivate()
            mas.finalTemp[I,J].deactivate()
    mas.C_TCP3.deactivate()

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(mas)
    transformation_string = 'gdp.' + transform
    pe.TransformationFactory(transformation_string).apply_to(mas)

    ######---------------------------- SUBPROBLEM --------------------------------##################
    sub=model_fun(**kwargs2) #subproblem

    # deactivate master problem constraints
    sub.E2_CAPACITY_LOW.deactivate()
    sub.E2_CAPACITY_UP.deactivate()
    sub.E3_BALANCE_INIT.deactivate()
    sub.E_DEMAND_SATISFACTION.deactivate()
    sub.E1_UNIT.deactivate()
    sub.DEF_AUX1_INDEP.deactivate()
    sub.E3_BALANCE.deactivate()
    sub.DEF_AUX2_INDEP.deactivate()
    sub.linking1.deactivate()
    sub.linking2.deactivate()
    for I in sub.I_reactions:
        for J in sub.J_reactors:
            for disj in sub.ordered_set[I,J]:
                disjunct=sub.YR_disjunct[I,J][disj]
                for constr in disjunct.component_objects(pe.Constraint, descend_into=True):
                    if constr.local_name=='DEF_VAR_TIME' or constr.local_name=='DEF_AUX1' or constr.local_name=='DEF_AUX2': #NOTE that I am deactivating everythin. I am just using this to show that I can compare by name to deactivate what I want
                        constr.deactivate()
            sub.YR_disjunct[I,J].deactivate()
            sub.Disjunction1[I,J].deactivate()
            sub.oneYR[I,J].deactivate()
    sub.X_Z_relation.deactivate()
    sub.obj.deactivate()

    # re-define objective function
    sub.obj.deactivate()
    def _obj_dynamic(m):
        return m.TCP3
    sub.obj_dyn = pe.Objective(rule=_obj_dynamic, sense=pe.minimize) 

    #linking variables and constraints in subproblem:
    sub.link_Vreactor=pe.Var(sub.I_reactions,sub.J_reactors,within=pe.NonNegativeReals)
    sub.link_varTime=pe.Var(sub.I_reactions,sub.J_reactors,within=pe.NonNegativeReals)
    sub.link_X=pe.Var(sub.I,sub.J,sub.T,within=pe.NonNegativeReals)

    def _const_link_Vreactor(sub,I,J):
        return sub.Vreactor[I,J]-sub.link_Vreactor[I,J]==0
    sub.const_link_Vreactor=pe.Constraint(sub.I_reactions,sub.J_reactors,rule=_const_link_Vreactor)

    def _const_link_VarTime(sub,I,J):
        return sub.varTime[I,J]-sub.link_varTime[I,J]==0
    sub.const_link_VarTime=pe.Constraint(sub.I_reactions,sub.J_reactors,rule=_const_link_VarTime)

    # We have linking constraints fixed, so we can change the domain of variables X to continuous in subproblems
    sub.X.domain=pe.NonNegativeReals
    def _const_link_X(sub,I,J,T):
        return sub.X[I,J,T]-sub.link_X[I,J,T]==0
    sub.const_link_X=pe.Constraint(sub.I,sub.J,sub.T,rule=_const_link_X)

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(sub)
    transformation_string = 'gdp.' + transform
    pe.TransformationFactory(transformation_string).apply_to(sub)

    # Dual variables (Lagrange multipliers)
    sub.dual = pe.Suffix(direction=pe.Suffix.IMPORT) #define dual variables

    #####------------------Heuristic cuts for experiments 2 and 3-------------------------######################
    mint=sub.clone()
    mint.obj_dyn.deactivate()
    def _obj_t(m):
        return sum(sum( mint.varTime[I,J] for I in mint.I_reactions) for J in mint.J_reactors)
    mint.obj_t = pe.Objective(rule=_obj_t, sense=pe.minimize)   

    for I in mint.I:
        for J in mint.J:
            if experiment ==3 and I in mint.I_reactions and J in mint.J_reactors:
                mint.Vreactor[I,J].fix(mas.Vreactor[I,J].ub) # NOTE: that the maximum capacity for this case study agrees with upper bound
            for T in mint.T:
                mint.X[I,J,T].fix(1)

    for I_J in mint.I_J:
            I=I_J[0]
            J=I_J[1]
            mint.Nref[I,J].fix(1) 
    if experiment ==2 or experiment ==3:
    # TEST MINIMUM PROCESSING TIME (add minimum processing time constraint to master)
        
        mint=solve_subproblem(mint,subproblem_solver='conopt4',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)   

        def _const_min_VarTime(mas,I,J):
            return mas.varTime[I,J]>=pe.value(mint.varTime[I,J])
        mas.const_min_VarTime=pe.Constraint(mas.I_reactions,mas.J_reactors,rule=_const_min_VarTime)
        print('\n minimum variable processing times')
        mas.const_min_VarTime.pprint()

    ######---------------------------- FEASIBILITY SUBPROBLEM --------------------------------##################
    feas=sub.clone()
    feas.elastic_vars={}

    count_elastic=0
    for constr in feas.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
        if constr.parent_component().name != 'const_link_Vreactor' and constr.parent_component().name != 'const_link_VarTime' and constr.parent_component().name != 'const_link_X': 
            if constr.equality:
                count_elastic=count_elastic+1
                feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                constr._body+=feas.elastic_vars[count_elastic]

                count_elastic=count_elastic+1
                feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                constr._body+=-feas.elastic_vars[count_elastic]
            else:
                count_elastic=count_elastic+1
                if constr.has_lb():
                    feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                    setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                    constr._body+=feas.elastic_vars[count_elastic]
                if constr.has_ub():
                    feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                    setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                    constr._body+=-feas.elastic_vars[count_elastic]

    feas.obj_dyn.deactivate()
    def _obj_feas(m):
        return sum( feas.elastic_vars[i]  for i in feas.elastic_vars.keys())
    feas.obj_feas = pe.Objective(rule=_obj_feas, sense=pe.minimize)     

    ######---------------------------- SOLUTION OF FIRST SUBPROBLEM --------------------------------##################
    mas=initialize_model(mas,from_feasible=True,feasible_model='case_1_scheduling_and_dynamics_solution_GDB_init') #Initialization from solution that is known to be feasible
    for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
        if v.name=='varTime' or v.name=='Vreactor':
            for index in v:
                if index==None:
                    v.fix(pe.value(v))
                else:
                    v[index].fix(pe.value(v[index]))
        elif v.name=='X':
            for index in v:
                if index==None:
                    v.fix(round(pe.value(v)))
                else:
                    v[index].fix(round(pe.value(v[index])))

    mas=solve_with_minlp(mas,transformation='',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
    print(pe.value(mas.obj))

    # fix master variables in subproblem 
    for I in sub.I:
        for J in sub.J:
            for T in sub.T:
                sub.B[I,J,T].fix(pe.value(mas.B[I,J,T]))
                sub.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

    for I_J in sub.I_J:
            I=I_J[0]
            J=I_J[1]
            sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

    for K in sub.K:
        for T in sub.T:
            sub.S[K,T].fix(pe.value(mas.S[K,T]))

    for I in sub.I_reactions:
        for J in sub.J_reactors:
            sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  
            sub.link_Vreactor[I,J].fix(pe.value(mas.Vreactor[I,J]))  


    sub=solve_subproblem(sub,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
  
    
    if sub.dsda_status=='Optimal':
        mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(  sub.dual[sub.const_link_Vreactor[I,J]]*( mas.Vreactor[I,J]-pe.value(sub.link_Vreactor[I,J])   )      for I in mas.I_reactions) for J in mas.J_reactors) +sum(sum(sum(  sub.dual[sub.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(sub.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
        print('Optimality cut to initialize: ')
        mas.cuts.pprint()
        
        TPC1=pe.value(sub.TCP1)
        TPC2=pe.value(sub.TCP2)
        TPC3=pe.value(sub.TCP3)
        TMC=pe.value(sub.TMC)
        SALES=pe.value(sub.SALES)
        sub.UBD=TPC1+TPC2+TPC3+TMC-SALES
        print(sub.UBD)
    else:
        sub.UBD=Infinity_aprox
        print('Initialization is ot feasible')
        exit()

    # make sure that variables in master are left unfixed
    for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
        if v.name=='varTime' or v.name=='Vreactor' or v.name=='X':
            for index in v:
                if index==None:
                    v.unfix()
                else:
                    v[index].unfix()

    ######---------------------------- GENERALIZED BENDERS DECOMPOSIION ALGORITHM --------------------------------##################

    start=time.time()
    max_iter=100000
    epsilon=0
    time_limit=86400 #seconds
    for k in range(max_iter):
        print('------------------------Iteration ',str(k),'------------------------------------')
    #1: solve the master problem
        mas=solve_with_minlp(mas,transformation='',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
        mas.LBD=pe.value(mas.obj)

        if experiment == 4:
        #1.1.: add no-good cuts (cuts that avoid repeating combination of binary variables)
            expr=0
            for v in mas.component_data_objects(ctype=pe.Var,descend_into=True,active=True):
                if v.is_binary() and int(round(pe.value(v)))==int(1):
                    expr+=v-1
                elif v.is_binary() and int(round(pe.value(v)))==int(0):
                    expr+=-v          
            
            mas.cuts.add( expr<= -1)
        elif experiment ==5:
            #1.1.: add no-good cuts (cuts that avoid repeating combination of binary variables related to variable processing times)
            expr=0
            for v in mas.component_data_objects(ctype=pe.Var,descend_into=True,active=True):
                if v.is_binary() and int(round(pe.value(v)))==int(1) and v.parent_component().name !='X':
                    expr+=v-1
                elif v.is_binary() and int(round(pe.value(v)))==int(0) and v.parent_component().name !='X':
                    expr+=-v          
            
            mas.cuts.add( expr<= -1)

    #2: verify stopping criterion
        current=time.time()
        print('Primal obj: ',str(sub.UBD),'Master obj:', str(mas.LBD),'Current time: ',str(current-start))
        if sub.UBD- mas.LBD <=epsilon:
            break
        if current-start>=time_limit:
            break

    #3: Solve primal problem
        # fix variables in subproblem 
        for I in sub.I:
            for J in sub.J:
                for T in sub.T:

                    if pe.value(mas.B[I,J,T])>=mas.B[I,J,T].ub: 
                        sub.B[I,J,T].fix(mas.B[I,J,T].ub)
                    elif pe.value(mas.B[I,J,T])<=mas.B[I,J,T].lb:
                        sub.B[I,J,T].fix(mas.B[I,J,T].lb)
                    else:
                        sub.B[I,J,T].fix(pe.value(mas.B[I,J,T]))

                    sub.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

        for I_J in sub.I_J:
                I=I_J[0]
                J=I_J[1]
                sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

        for K in sub.K:
            for T in sub.T:
                if pe.value(mas.S[K,T])>=mas.S[K,T].ub:
                    sub.S[K,T].fix(mas.S[K,T].ub)
                elif pe.value(mas.S[K,T])<=mas.S[K,T].lb:
                    sub.S[K,T].fix(mas.S[K,T].lb)
                else:
                    sub.S[K,T].fix(pe.value(mas.S[K,T]))

        for I in sub.I_reactions:
            for J in sub.J_reactors:
                if pe.value(mas.varTime[I,J])>=mas.varTime[I,J].ub:
                    sub.link_varTime[I,J].fix(mas.varTime[I,J].ub)  
                elif pe.value(mas.varTime[I,J])<=mas.varTime[I,J].lb:
                    sub.link_varTime[I,J].fix(mas.varTime[I,J].lb)  
                else:
                    sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))   
                if pe.value(mas.Vreactor[I,J])>=mas.Vreactor[I,J].ub:
                    sub.link_Vreactor[I,J].fix(mas.Vreactor[I,J].ub)  
                elif pe.value(mas.Vreactor[I,J])<=mas.Vreactor[I,J].lb:
                    sub.link_Vreactor[I,J].fix(mas.Vreactor[I,J].lb)  
                else:
                    sub.link_Vreactor[I,J].fix(pe.value(mas.Vreactor[I,J])) 

        # solve primal (subprolem)
        sub=solve_subproblem(sub,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
        generate_initialization(m=sub,model_name='GBD_subproblem') #save solution, in case I need an alternative initialization for the feasibility subproblem
        print('Subproblem status:',sub.dsda_status)

        if sub.dsda_status=='Optimal':
              
            TPC1=pe.value(sub.TCP1)
            TPC2=pe.value(sub.TCP2)
            TPC3=pe.value(sub.TCP3)
            TMC=pe.value(sub.TMC)
            SALES=pe.value(sub.SALES)

            sub.UBD_new=min([sub.UBD,TPC1+TPC2+TPC3+TMC-SALES])
            # Update the best known solution if it improved
            if sub.UBD_new<sub.UBD:
                generate_initialization(m=sub,model_name=best_sol_name)
            # Update subproblem solution with the best subproblem solution identified so far  
            sub.UBD=sub.UBD_new


            mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(  sub.dual[sub.const_link_Vreactor[I,J]]*( mas.Vreactor[I,J]-pe.value(sub.link_Vreactor[I,J])   )      for I in mas.I_reactions) for J in mas.J_reactors) +sum(sum(sum(  sub.dual[sub.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(sub.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
            print('Optimaliti cut added')
            
        else:
            if auxiliary_cuts:
                # Improved feasibility cuts
                for I in mint.I_reactions:
                    for J in mint.J_reactors:
                
                        if pe.value(mas.Vreactor[I,J])>=mas.Vreactor[I,J].ub: 
                            mint.link_Vreactor[I,J].fix(mas.Vreactor[I,J].ub)
                        elif pe.value(mas.Vreactor[I,J])<=mas.Vreactor[I,J].lb:
                            mint.link_Vreactor[I,J].fix(mas.Vreactor[I,J].lb)
                        else:
                            mint.link_Vreactor[I,J].fix(pe.value(mas.Vreactor[I,J]))
                mint=solve_subproblem(mint,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0) 
                if mint.dsda_status=='Optimal':
                    for I in mint.I_reactions:
                        for J in mint.J_reactors:
                                mas.cuts.add(pe.value(mint.varTime[I,J])+mint.dual[mint.const_link_Vreactor[I,J]]*(mas.Vreactor[I,J]-pe.value(mint.link_Vreactor[I,J]))<=mas.varTime[I,J])
                    print('Improved feasibility cuts added')
                else:
                    # Naive feasibility cuts
                    # fix variables in subproblem 
                    for I in feas.I:
                        for J in feas.J:
                            for T in feas.T:

                                if pe.value(mas.B[I,J,T])>=mas.B[I,J,T].ub: 
                                    feas.B[I,J,T].fix(mas.B[I,J,T].ub)
                                elif pe.value(mas.B[I,J,T])<=mas.B[I,J,T].lb:
                                    feas.B[I,J,T].fix(mas.B[I,J,T].lb)
                                else:
                                    feas.B[I,J,T].fix(pe.value(mas.B[I,J,T]))

                                feas.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

                    for I_J in feas.I_J:
                            I=I_J[0]
                            J=I_J[1]
                            feas.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

                    for K in feas.K:
                        for T in feas.T:
                            if pe.value(mas.S[K,T])>=mas.S[K,T].ub:
                                feas.S[K,T].fix(mas.S[K,T].ub)
                            elif pe.value(mas.S[K,T])<=mas.S[K,T].lb:
                                feas.S[K,T].fix(mas.S[K,T].lb)
                            else:
                                feas.S[K,T].fix(pe.value(mas.S[K,T]))

                    for I in feas.I_reactions:
                        for J in feas.J_reactors:
                            if pe.value(mas.varTime[I,J])>=mas.varTime[I,J].ub:
                                feas.link_varTime[I,J].fix(mas.varTime[I,J].ub)  
                            elif pe.value(mas.varTime[I,J])<=mas.varTime[I,J].lb:
                                feas.link_varTime[I,J].fix(mas.varTime[I,J].lb)  
                            else:
                                feas.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  
                            if pe.value(mas.Vreactor[I,J])>=mas.Vreactor[I,J].ub:
                                feas.link_Vreactor[I,J].fix(mas.Vreactor[I,J].ub)  
                            elif pe.value(mas.Vreactor[I,J])<=mas.Vreactor[I,J].lb:
                                feas.link_Vreactor[I,J].fix(mas.Vreactor[I,J].lb)  
                            else:
                                feas.link_Vreactor[I,J].fix(pe.value(mas.Vreactor[I,J]))  

                    feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                    print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
                    if feas.dsda_status=='Optimal':
                        sum_infeasibility=pe.value(feas.obj_feas)
                        mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(  feas.dual[feas.const_link_Vreactor[I,J]]*( mas.Vreactor[I,J]-pe.value(feas.link_Vreactor[I,J])   )      for I in mas.I_reactions) for J in mas.J_reactors) +sum(sum(sum(  feas.dual[feas.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(feas.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                        print('Feasibility cut added')
                    else:
                        print('Problem with feasibility stage. Trying a different initialization')
                        feas=initialize_model(feas,from_feasible=True,feasible_model='GBD_subproblem')
                        feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                        print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
                        if feas.dsda_status=='Optimal':
                            sum_infeasibility=pe.value(feas.obj_feas)
                            mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(  feas.dual[feas.const_link_Vreactor[I,J]]*( mas.Vreactor[I,J]-pe.value(feas.link_Vreactor[I,J])   )      for I in mas.I_reactions) for J in mas.J_reactors) +sum(sum(sum(  feas.dual[feas.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(feas.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                            print('Feasibility cut added')
                        else:
                            print('GBD solver failure: subproblem detected as infeasible, and fatal error with feasibility stage')
                            break
            else:
                # Naive feasibility cuts
                # fix variables in subproblem 
                for I in feas.I:
                    for J in feas.J:
                        for T in feas.T:

                            if pe.value(mas.B[I,J,T])>=mas.B[I,J,T].ub: 
                                feas.B[I,J,T].fix(mas.B[I,J,T].ub)
                            elif pe.value(mas.B[I,J,T])<=mas.B[I,J,T].lb:
                                feas.B[I,J,T].fix(mas.B[I,J,T].lb)
                            else:
                                feas.B[I,J,T].fix(pe.value(mas.B[I,J,T]))

                            feas.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

                for I_J in feas.I_J:
                        I=I_J[0]
                        J=I_J[1]
                        feas.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

                for K in feas.K:
                    for T in feas.T:
                        if pe.value(mas.S[K,T])>=mas.S[K,T].ub:
                            feas.S[K,T].fix(mas.S[K,T].ub)
                        elif pe.value(mas.S[K,T])<=mas.S[K,T].lb:
                            feas.S[K,T].fix(mas.S[K,T].lb)
                        else:
                            feas.S[K,T].fix(pe.value(mas.S[K,T]))

                for I in feas.I_reactions:
                    for J in feas.J_reactors:
                        if pe.value(mas.varTime[I,J])>=mas.varTime[I,J].ub:
                            feas.link_varTime[I,J].fix(mas.varTime[I,J].ub)  
                        elif pe.value(mas.varTime[I,J])<=mas.varTime[I,J].lb:
                            feas.link_varTime[I,J].fix(mas.varTime[I,J].lb)  
                        else:
                            feas.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  
                        if pe.value(mas.Vreactor[I,J])>=mas.Vreactor[I,J].ub:
                            feas.link_Vreactor[I,J].fix(mas.Vreactor[I,J].ub)  
                        elif pe.value(mas.Vreactor[I,J])<=mas.Vreactor[I,J].lb:
                            feas.link_Vreactor[I,J].fix(mas.Vreactor[I,J].lb)  
                        else:
                            feas.link_Vreactor[I,J].fix(pe.value(mas.Vreactor[I,J]))  

                feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
                if feas.dsda_status=='Optimal':
                    sum_infeasibility=pe.value(feas.obj_feas)
                    mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(  feas.dual[feas.const_link_Vreactor[I,J]]*( mas.Vreactor[I,J]-pe.value(feas.link_Vreactor[I,J])   )      for I in mas.I_reactions) for J in mas.J_reactors) +sum(sum(sum(  feas.dual[feas.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(feas.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                    print('Feasibility cut added')
                else:
                    print('Problem with feasibility stage. Trying a different initialization')
                    feas=initialize_model(feas,from_feasible=True,feasible_model='GBD_subproblem')
                    feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                    print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
                    if feas.dsda_status=='Optimal':
                        sum_infeasibility=pe.value(feas.obj_feas)
                        mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(  feas.dual[feas.const_link_Vreactor[I,J]]*( mas.Vreactor[I,J]-pe.value(feas.link_Vreactor[I,J])   )      for I in mas.I_reactions) for J in mas.J_reactors) +sum(sum(sum(  feas.dual[feas.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(feas.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                        print('Feasibility cut added')
                    else:
                        print('GBD solver failure: subproblem detected as infeasible, and fatal error with feasibility stage')
                        break                

    ######---------------------------- DISPLAY SOLUTION SUMMARY --------------------------------##################
    model_fun=scheduling_and_control_gdp_N_GBD
    kwargs3=kwargs.copy()  
    subsol=model_fun(**kwargs3)
    subsol=initialize_model(subsol,from_feasible=True,feasible_model=best_sol_name)

    Sol_found=[]
    for I in subsol.I_reactions:
        for J in subsol.J_reactors:
            if subsol.I_i_j_prod[I,J]==1:
                for K in subsol.ordered_set[I,J]:
                    if round(pe.value(subsol.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-subsol.minTau[I,J]+1)
    for I_J in subsol.I_J:
        Sol_found.append(1+round(pe.value(subsol.Nref[I_J])))
    print('EXT_VARS_FOUND',Sol_found)
    TPC1=pe.value(subsol.TCP1)
    TPC2=pe.value(subsol.TCP2)
    TPC3=pe.value(subsol.TCP3)
    TMC=pe.value(subsol.TMC)
    SALES=pe.value(subsol.SALES)
    OBJ_FOUND=TPC1+TPC2+TPC3+TMC-SALES

    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJECTIVE:',str(OBJ_FOUND))

    
    print('\n-------DICOPT TEST-------------------------------------')

    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > dicopt.opt \n','maxcycles 20000 \n','stop 2 \n','relaxed 0 \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    # init_name='case_1_scheduling_and_dynamics_solution_GDB_init'
    # minlp_solver='dicopt'
    # kwargs4=kwargs.copy() 
    # m=model_fun(**kwargs4)
    # m=initialize_model(m=m,from_feasible=True,feasible_model=init_name) 
    # m.pprint()

    # experiment=3
 
    # if experiment ==1:
    #     best_sol_name='naive'
    # elif experiment ==2:
    #     best_sol_name='min_t_vary_B'
    #     m.initcuts=pe.ConstraintList()
    #     m.initcuts.add(m.varTime[('R1', 'R_large')]>=1.625802848031681)
    #     m.initcuts.add(m.varTime[('R1', 'R_small')]>=1.649138048728084)
    #     m.initcuts.add(m.varTime[('R2', 'R_large')]>=2.54458538005377)
    #     m.initcuts.add(m.varTime[('R2', 'R_small')]>=2.557392266150617)
    #     m.initcuts.add(m.varTime[('R3', 'R_large')]>=1.084716847688982)
    #     m.initcuts.add(m.varTime[('R3', 'R_small')]>=1.109942023857737)
    
    # elif experiment ==3:
    #     best_sol_name='min_t_fix_B'
    #     m.initcuts=pe.ConstraintList()
    #     m.initcuts.add(m.varTime[('R1', 'R_large')]>=2.136622088669454)
    #     m.initcuts.add(m.varTime[('R1', 'R_small')]>=2.229415319435365)
    #     m.initcuts.add(m.varTime[('R2', 'R_large')]>=2.855769349001259)
    #     m.initcuts.add(m.varTime[('R2', 'R_small')]>=2.918142248015295)
    #     m.initcuts.add(m.varTime[('R3', 'R_large')]>=1.591711258452907)
    #     m.initcuts.add(m.varTime[('R3', 'R_small')]>=1.675857698391268)

    # start=time.time()
    # m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()    
    # solname='GBD_V3_case_1_minlp_'+minlp_solver+'_from_'+best_sol_name
    # save=generate_initialization(m=m,model_name=solname)

    # if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #     m.dicopt_status='Infeasible'
    # else:
    #     m.dicopt_status='Optimal'

    # if m.dicopt_status=='Optimal':
    #     Sol_founddicopt=[]
    #     for I in m.I_reactions:
    #         for J in m.J_reactors:
    #             if m.I_i_j_prod[I,J]==1:
    #                 for K in m.ordered_set[I,J]:
    #                     if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
    #                         Sol_founddicopt.append(K-m.minTau[I,J]+1)
    #     for I_J in m.I_J:
    #         Sol_founddicopt.append(1+round(pe.value(m.Nref[I_J])))


    #     print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_founddicopt,'cputime DICOPT=',str(end-start))
    # else:
    #     print('DICOPT infeasible','cputime DICOPT=',str(end-start))

    # TPC1=pe.value(m.TCP1)
    # TPC2=pe.value(m.TCP2)
    # TPC3=pe.value(m.TCP3)
    # TMC=pe.value(m.TMC)
    # SALES=pe.value(m.SALES)
    # OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    # print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    # print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    # print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    # print('TMC: Total material cost: ',str(TMC))
    # print('SALES: Revenue form selling products: ',str(SALES))
    # print('OBJ:',str(OBJVAL))