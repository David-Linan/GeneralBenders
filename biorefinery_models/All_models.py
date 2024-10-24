import pyomo.environ as pe
import pyomo.dae as dae
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import numpy as np
from pyomo.gdp import Disjunct, Disjunction
import copy
import csv
import itertools as it
import os
import time
from math import isnan
from model_serializer import StoreSpec, from_json, to_json
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
# from pyomo.contrib.gdpopt.data_class import MasterProblemResult
from pyomo.core.base.misc import display
from pyomo.core.plugins.transform.logical_to_linear import \
    update_boolean_vars_from_binary
from pyomo.opt import SolutionStatus, SolverResults
from pyomo.opt import TerminationCondition as tc
from copy import deepcopy
import logging



def dummy_logic(m):
    logic_expr = []
    return logic_expr
def complementary_model(m,x,dict_extvar,tee: bool = False):

    m.fer.scaled_F_C5liquid_new.fix(     ((m.fer.scaled_F_C5liquid_new.ub-m.fer.scaled_F_C5liquid_new.lb)/(dict_extvar[1]['Ext_var_upper_bound']-dict_extvar[1]['Ext_var_lower_bound']))*(x[0]-dict_extvar[1]['Ext_var_upper_bound']) +  m.fer.scaled_F_C5liquid_new.ub           )   
    m.fer.scaled_F_liquified_fibers_new.fix(((m.fer.scaled_F_liquified_fibers_new.ub-m.fer.scaled_F_liquified_fibers_new.lb)/(dict_extvar[2]['Ext_var_upper_bound']-dict_extvar[2]['Ext_var_lower_bound']))*(x[1]-dict_extvar[2]['Ext_var_upper_bound']) +  m.fer.scaled_F_liquified_fibers_new.ub   )
    m.fer.scaled_F_base_constant.fix( ((m.fer.scaled_F_base_constant.ub-m.fer.scaled_F_base_constant.lb)/(dict_extvar[3]['Ext_var_upper_bound']-dict_extvar[3]['Ext_var_lower_bound']))*(x[2]-dict_extvar[3]['Ext_var_upper_bound']) +  m.fer.scaled_F_base_constant.ub  )
    if tee:
        print('*****Current value of complicating variables: '+m.fer.scaled_F_C5liquid_new.name+'='+str(m.fer.scaled_F_C5liquid_new.value)+m.fer.scaled_F_liquified_fibers_new.name+'='+str(m.fer.scaled_F_liquified_fibers_new.value)+m.fer.scaled_F_base_constant.name+'='+str(m.fer.scaled_F_base_constant.value))

    return m

def get_external_information(
    m: pe.ConcreteModel(),
    ext_ref,
    tee: bool = False,
):
    """
    Function that obtains information from the model to perform the reformulation with external variables.
    The model must be a GDP problem with exactly one "Exactly(k_j, [Y_j1,Y_j2,Y_j3,...])" constraint for each list of variables
    [Y_j1,Y_j2,Y_j3,...] that is going to be reformulated over set j.
    Args:
        m: GDP model that is going to be reformulated
        ext_ref: Dictionary with Boolean variables to be reformulated (keys) and their corresponding ordered sets (values). Both keys and values are pyomo objects.
        tee: Display reformulation
    Returns:
        reformulation_dict: A dictionary of dictionaries that looks as follows:
            {1:{'exactly_number':Number of external variables for this type,
                'Boolean_vars_names':list with names of the ordered Boolean variables to be reformulated,
                'Boolean_vars_ordered_index': Indexes where the external reformulation is applied,
                'Ext_var_lower_bound': Lower bound for this type of external variable,
                'Ext_var_upper_bound': Upper bound for this type of external variable },
             2:{...},...}

            The first key (positive integer) represent a type of external variable identified in the model. For this type of external variable
            a dictionary is created.
        number_of_external_variables: Number of external variables
        lower_bounds: Dictionary with positive integer keys identifying the external variable, and its lower bound as value
        upper_bounds: Dictionary with positive integer keys identifying the external variable, and its upper bound as value

    """

    # If Boolean variables that are going to be reformulated are defined over multiple sets try:
    try:
        # index of the set where reformultion can be applied for a given boolean variable
        ref_index = {}
        # index of the sets where the reformulation cannot be applied for a given boolean variable
        no_ref_index = {}
        for i in ext_ref:
            ref_index[i] = []
            no_ref_index[i] = []
            for index_set in range(len(i.index_set()._sets)):
                if i.index_set()._sets[index_set].name == ext_ref[i].name:
                    ref_index[i].append(index_set)
                else:
                    no_ref_index[i].append(index_set)
    # If boolean variables that are going to be reformulated are defined over a single set except:
    except:
        # index of the set where reformultion can be applied for a given boolean variable
        ref_index = {}
        # index of the sets where the reformulation cannot be applied for a given boolean variable
        no_ref_index = {}
        for i in ext_ref:
            ref_index[i] = []
            no_ref_index[i] = []
            if i.index_set().name == ext_ref[i].name:
                ref_index[i].append(0)
            else:
                no_ref_index[i].append(0)

    # Identify the variables that can be reformulated by performing a loop over logical constraints
    count = 1
    # dict of dicts: it contains information from the exactly variables that can be reformulated into external variables.
    reformulation_dict = {}
    for c in m.component_data_objects(pe.LogicalConstraint, descend_into=True):
        if c.body.getname() == 'exactly':
            exactly_number = c.body.args[0]
            for possible_Boolean in ext_ref:

                # expected boolean variable where the reformulation is going to be applied
                expected_Boolean = possible_Boolean.name
                Boolean_name_list = []
                Boolean_name_list = Boolean_name_list + \
                    [c.body.args[1:][k]._component()._name for k in range(
                        len(c.body.args[1:]))]
                if all(x == expected_Boolean for x in Boolean_name_list):
                    # expected ordered set index where the reformulation is going to be applied
                    expected_ordered_set_index = ref_index[possible_Boolean]
                    # index of sets where the reformulation is not applied
                    index_of_other_sets = no_ref_index[possible_Boolean]
                    if len(index_of_other_sets) >= 1:  # If there are other indexes
                        Other_Sets_listOFlists = []
                        verification_Other_Sets_listOFlists = []
                        for j in index_of_other_sets:
                            Other_Sets_listOFlists.append(
                                [c.body.args[1:][k].index()[j] for k in range(len(c.body.args[1:]))])
                            if all(c.body.args[1:][x].index()[j] == c.body.args[1:][0].index()[j] for x in range(len(c.body.args[1:]))):
                                verification_Other_Sets_listOFlists.append(
                                    True)
                            else:
                                verification_Other_Sets_listOFlists.append(
                                    False)
                        # If we get to this point and it is true, it means that we can apply the reformulation for this combination of Boolean var and Exactly-type constraint
                        if all(verification_Other_Sets_listOFlists):
                            reformulation_dict[count] = {}
                            reformulation_dict[count]['exactly_number'] = exactly_number
                            # rearange boolean vars in constraint
                            sorted_args = sorted(c.body.args[1:], key=lambda x: x.index()[
                                                 expected_ordered_set_index[0]])
                            # Now work with the ordered version sorted_args instead of c.body.args[1:]
                            reformulation_dict[count]['Boolean_vars_names'] = [
                                sorted_args[k].name for k in range(len(sorted_args))]
                            reformulation_dict[count]['Boolean_vars_ordered_index'] = [sorted_args[k].index(
                            )[expected_ordered_set_index[0]] for k in range(len(sorted_args))]
                            reformulation_dict[count]['Ext_var_lower_bound'] = 1
                            reformulation_dict[count]['Ext_var_upper_bound'] = len(
                                sorted_args)

                            count = count+1
                    # If there is only one index, then we can apply the reformulation at this point
                    else:
                        reformulation_dict[count] = {}
                        reformulation_dict[count]['exactly_number'] = exactly_number
                        # rearange boolean vars in constraint
                        sorted_args = sorted(
                            c.body.args[1:], key=lambda x: x.index())
                        # Now work with the ordered version sorted_args instead of c.body.args[1:]
                        reformulation_dict[count]['Boolean_vars_names'] = [
                            sorted_args[k].name for k in range(len(sorted_args))]
                        reformulation_dict[count]['Boolean_vars_ordered_index'] = [
                            sorted_args[k].index() for k in range(len(sorted_args))]
                        reformulation_dict[count]['Ext_var_lower_bound'] = 1
                        reformulation_dict[count]['Ext_var_upper_bound'] = len(
                            sorted_args)

                        count = count+1

    number_of_external_variables = sum(
        reformulation_dict[j]['exactly_number'] for j in reformulation_dict)

    lower_bounds = {}
    upper_bounds = {}

    exvar_num = 1
    for i in reformulation_dict:
        for j in range(reformulation_dict[i]['exactly_number']):
            lower_bounds[exvar_num] = reformulation_dict[i]['Ext_var_lower_bound']
            upper_bounds[exvar_num] = reformulation_dict[i]['Ext_var_upper_bound']
        exvar_num = exvar_num+1

    if tee:
        print('\nReformulation Summary\n--------------------------------------------------------------------------')
        exvar_num = 0
        for i in reformulation_dict:
            for j in range(reformulation_dict[i]['exactly_number']):
                print('External variable x['+str(exvar_num)+'] '+' is associated to '+str(reformulation_dict[i]['Boolean_vars_names']) +
                      ' and it must be within '+str(reformulation_dict[i]['Ext_var_lower_bound'])+' and '+str(reformulation_dict[i]['Ext_var_upper_bound'])+'.')
                exvar_num = exvar_num+1

        print('\nThere are '+str(number_of_external_variables) +
              ' external variables in total')

    return reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds

def external_ref(
    m: pe.ConcreteModel(),
    x,
    extra_logic_function,
    dict_extvar: dict = {},
    mip_ref: bool = False,
    transformation: str = 'bigm',
    tee: bool = False
):
    """
    Function that
    Args:
        m: GDP model that is going to be reformulated
        x: List with current value of the external variables
        extra_logic_function: Function that returns a list of lists of the form [a,b], where a is an expressions of the reformulated Boolean variables and b is an equivalent Boolean or indicator variable (b<->a)
        dict_extvar: A dictionary of dictionaries that looks as follows:
            {1:{'exactly_number':Number of external variables for this type,
                'Boolean_vars_names':list with names of the ordered Boolean variables to be reformulated,
                'Boolean_vars_ordered_index': Indexes where the external reformulation is applied,
                'Binary_vars_names':list with names of the ordered Binary variables to be reformulated, [Potentially]
                'Binary_vars_ordered_index': Indexes where the external reformulation is applied, [Potentially]
                'Ext_var_lower_bound': Lower bound for this type of external variable,
                'Ext_var_upper_bound': Upper bound for this type of external variable },
             2:{...},...}

            The first key (positive integer) represent a type of external variable identified in the model. For this type of external variable
            a dictionary is created.
        mip_ref: whether the reformulation will consider binary variables besides Booleans coming from a GDP->MIP reformulation
        tee: Display reformulation
    Returns:
        m: A model where the independent Boolean variables that were reformulated are fixed and Boolean/indicator variables that are calculated in
        terms of the independent Boolean variables are fixed too (depending on the extra_logic_function provided by the user)

    """
    # This part of code is required due to the deep copy issue: we have to compare Boolean variables by name
    for i in dict_extvar:
        dict_extvar[i]['Boolean_vars'] = []
        for j in dict_extvar[i]['Boolean_vars_names']:
            for boolean in m.component_data_objects(pe.BooleanVar, descend_into=True):
                if(boolean.name == j):
                    dict_extvar[i]['Boolean_vars'] = dict_extvar[i]['Boolean_vars']+[boolean]
        if mip_ref:
            # This part of code is required due to the deep copy issue: we have to compare binary variables by name
            # By uncommenting in previous function extvars_gdp_to_mip we would pass directly dict_extvar[i]['Binary_vars']
            dict_extvar[i]['Binary_vars'] = []
            for j in dict_extvar[i]['Binary_vars_names']:
                for binary in m.component_data_objects(pe.Var, descend_into=True):
                    if(binary.name == j):
                        dict_extvar[i]['Binary_vars'] = dict_extvar[i]['Binary_vars']+[binary]

# The function would start here if there were no problems with deep copy.
    ext_var_position = 0
    for i in dict_extvar:
        for j in range(dict_extvar[i]['exactly_number']):
            for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                if x[ext_var_position] == k:
                    if not mip_ref:
                        # fix True variables: depending on the current value of the external variables, some Independent Boolean variables can be fixed
                        dict_extvar[i]['Boolean_vars'][k-1].fix(True)
                    else:
                        # fix 0 variables: depending on the current value of the external variables, some Independent Binary variables can be fixed
                        dict_extvar[i]['Binary_vars'][k-1].fix(1)
                        dict_extvar[i]['Boolean_vars'][k-1].set_value(True)
            ext_var_position = ext_var_position+1
        # Double loop required from fact that exactly_number >= 1. TODO Is there a better way to do this?
        for j in range(dict_extvar[i]['exactly_number']):
            for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                if not mip_ref:
                    # fix False variables: If the independent Boolean variable is not fixed at "True", then it is fixed at "False".
                    if not dict_extvar[i]['Boolean_vars'][k-1].is_fixed():
                        dict_extvar[i]['Boolean_vars'][k-1].fix(False)
                else:
                    # fix 0 variables: If the independent Boolean variable is not fixed at "1", then it is fixed at "0".
                    if not dict_extvar[i]['Binary_vars'][k-1].is_fixed():
                        dict_extvar[i]['Binary_vars'][k-1].fix(0)
                        dict_extvar[i]['Boolean_vars'][k-1].set_value(False)

    # Other Boolean and Indicator variables are fixed depending on the information provided by the user
    logic_expr = extra_logic_function(m)
    for i in logic_expr:
        if not mip_ref:
            i[1].fix(pe.value(i[0]))
        else:
            i[1].set_value(pe.value(i[0]))

    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    if mip_ref:  # Transform problem to MINLP
        transformation_string = 'gdp.' + transformation
        pe.TransformationFactory(transformation_string).apply_to(m)
    else:  # Deactivate disjunction's constraints in the case of pure GDP
        pe.TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
        m, tmp=False, ignore_infeasible=True)




    # NOTE: THIS PART IS CASE STUDY DEPENDENT

    m=complementary_model(m,x,dict_extvar,tee)

    #********************THE LINE ABOVE IS SIMPLY A TEST FOR SCHEDULING AND CONTROL PROBLEMS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if tee:
        print('\nFixed variables at current iteration:\n')
        print('\n Independent Boolean variables\n')
        for i in dict_extvar:
            for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                print(dict_extvar[i]['Boolean_vars_names'][k-1] +
                      '='+str(dict_extvar[i]['Boolean_vars'][k-1].value))

        print('\n Dependent Boolean variables and disjunctions\n')
        for i in logic_expr:
            print(i[1].name+'='+str(i[1].value))

        if mip_ref:
            print('\n Independent binary variables\n')
            for i in dict_extvar:
                for k in range(1, len(dict_extvar[i]['Binary_vars'])+1):
                    print(dict_extvar[i]['Binary_vars_names'][k-1] +
                          '='+str(dict_extvar[i]['Binary_vars'][k-1].value))

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
        m.obj.value=0
    else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
        m.dsda_status = 'Optimal'
    # if m.results.solver.termination_condition == 'locallyOptimal' or m.results.solver.termination_condition == 'optimal' or m.results.solver.termination_condition == 'globallyOptimal':
    #     m.dsda_status = 'Optimal'

    return m

def neighborhood_k_eq_all(dimension: int = 2) -> dict:
    """
    Function creates a k=all neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        directions: Dictionary contaning in each item a list with a direction within the neighborhood
    """

    num_neigh = 2
    directions={}
    directions[1]=list(1*np.ones(dimension, dtype=int))
    directions[2]=list(-1*np.ones(dimension, dtype=int))
    return directions


def neighborhood_k_eq_2(dimension: int = 2) -> dict:
    """
    Function creates a k=2 neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        directions: Dictionary contaning in each item a list with a direction within the neighborhood
    """

    num_neigh = 2*dimension
    neighbors = np.concatenate(
        (np.eye(dimension, dtype=int), -np.eye(dimension, dtype=int)), axis=1)
    directions = {}
    for i in range(num_neigh):
        direct = []
        directions[i+1] = direct
        for j in range(dimension):
            direct.append(neighbors[j, i])
    return directions


def neighborhood_k_eq_inf(dimension: int = 2) -> dict:
    """
    Function creates a k=Infinity neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        temp: Dictionary contaning in each item a list with a direction within the neighborhood
        TODO change temp name here to something more useful
    """

    neighbors = list(it.product([-1, 0, 1], repeat=dimension))
    directions = {}
    for i in range(len(neighbors)):
        directions[i+1] = list(neighbors[i])
    temp = directions.copy()
    for i in directions.keys():
        if temp[i] == [0]*dimension:
            temp.pop(i, None)
    return temp

def neighborhood_k_eq_l_natural(dimension: int = 2) -> dict:
    """
    Function creates a k=l_natural neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        directions: Dictionary contaning in each item a list with a direction within the neighborhood
    """
    if dimension==1:
        directions=neighborhood_k_eq_2(dimension)
    else:
        set_=np.arange(1,dimension+1,1)
        N_lflat=np.zeros(((2**(dimension+1))-2,dimension),dtype=int)
        k=0
        for i in range(dimension):
            sub=np.array(list(it.combinations(set_,dimension-i)),dtype=int)
            f=np.size(sub,0)
            for j in range(0,f):
                N_lflat[k,0:dimension]=np.array(np.isin(set_,sub[j,:]),dtype=int)
                k=k+1
                N_lflat[k,0:dimension]=-np.array(np.isin(set_,sub[j,:]),dtype=int)
                k=k+1

        # if dimension>=3:
        #     sort_=np.lexsort(N_lflat,axis=1)
        # else: 
        #     sort_=np.lexsort(N_lflat,axis=0)
        directions=dict(enumerate(N_lflat.tolist(),1))
    return directions


def neighborhood_k_eq_l_natural_modified(dimension: int = 2) -> dict:
    """
    Function creates a k=l_natural_modified neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        directions: Dictionary contaning in each item a list with a direction within the neighborhood
    """
    num_var_proc_time=6


    if dimension==1:
        directions=neighborhood_k_eq_2(dimension)
    else:
        set_=np.arange(1,dimension+1,1)
        N_lflat=np.zeros(((2**(dimension+1))-2,dimension),dtype=int)
        k=0
        for i in range(dimension):
            sub=np.array(list(it.combinations(set_,dimension-i)),dtype=int)
            f=np.size(sub,0)
            for j in range(0,f):
                partial=np.array(np.isin(set_,sub[j,:]),dtype=int)
                # # The following is equivalent to saying that: for procesing times k=2, for N_ij: k=l_natural. I am eliminating interactions between processing times and processing times-N_ij. Thus, size of neighborhood is: (2^(n_Nij+1)-2      +         2*n_tau)
                # if ~((sum(abs(partial[k]) for k in range(num_var_proc_time))>=2) or (sum(abs(partial[k]) for k in range(num_var_proc_time))>=1 and sum(abs(partial[k]) for k in range(num_var_proc_time,dimension))>=1)):
                #     N_lflat[k,0:dimension]=np.array(np.isin(set_,sub[j,:]),dtype=int)
                #     k=k+1
                #     N_lflat[k,0:dimension]=-np.array(np.isin(set_,sub[j,:]),dtype=int)
                #     k=k+1
                # #The following is equivalent to saying that: for procesing times k=2, for N_ij: k=l_natural. I am eliminating interactions between processing times and processing times-N_ij. Also, I am not considering interactions between pairs, trios, 4,5 or 6 or 7 or 8 (i.e., only interactions of 9 and 10 vars)
                # if ~((sum(abs(partial[k]) for k in range(num_var_proc_time))>=2) or (sum(abs(partial[k]) for k in range(num_var_proc_time))>=1 and sum(abs(partial[k]) for k in range(num_var_proc_time,dimension))>=1) or (sum(abs(partial[k]) for k in range(num_var_proc_time,dimension))<=8 and sum(abs(partial[k]) for k in range(num_var_proc_time,dimension))>=2)):
                #     N_lflat[k,0:dimension]=np.array(np.isin(set_,sub[j,:]),dtype=int)
                #     k=k+1
                #     N_lflat[k,0:dimension]=-np.array(np.isin(set_,sub[j,:]),dtype=int)
                #     k=k+1

                #Only consider second order interactions
                if sum(abs(partial[k]) for k in range(dimension))<=2:
                    N_lflat[k,0:dimension]=np.array(np.isin(set_,sub[j,:]),dtype=int)
                    k=k+1
                    N_lflat[k,0:dimension]=-np.array(np.isin(set_,sub[j,:]),dtype=int)
                    k=k+1
        N_lflat=N_lflat[~np.all(N_lflat==0, axis=1)]
        # if dimension>=3:
        #     sort_=np.lexsort(N_lflat,axis=1)
        # else: 
        #     sort_=np.lexsort(N_lflat,axis=0)
        directions=dict(enumerate(N_lflat.tolist(),1))
    return directions


def neighborhood_k_eq_m_natural(dimension: int = 2) -> dict:
    """
    Function creates a k=m_natural neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        directions: Dictionary contaning in each item a list with a direction within the neighborhood
    """
    N_Mflat=np.zeros((dimension*(dimension+1),dimension),dtype=int)
    mat1=np.eye(dimension,dimension,dtype=int)
    mat1=np.append(mat1,np.zeros((1,dimension),dtype=int),axis=0)
    f=np.size(mat1,0)
    k=0
    for i in range(f):
        for j in range(f):
            if i!=j:
                N_Mflat[k,0:dimension]=mat1[i,:]-mat1[j,:]

                k=k+1
    
    directions=dict(enumerate(N_Mflat.tolist(),1))
    return directions


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


def find_actual_neighbors(
    start: list,
    neighborhood: dict,
    min_allowed: dict = {},
    max_allowed: dict = {}
) -> dict:
    """
    Function that creates all neighbors of a given point. Neighbor 0 is the starting point
    Args:
        start: Point of which neighbors want to be created
        neighborhood: Neighborhood (output of a k-Neighborhood function)
        min_allowed: In keys contains external variables and in items their respective lower bounds
        max_allowed: In keys contains external variables and in items their respective upper bounds
    Returns:
        new_neighbors: Contains neighbors of the actual point
    """

    neighbors = {0: start}
    for i in neighborhood.keys():   # Calculate neighbors
        neighbors[i] = list(map(sum, zip(start, list(neighborhood[i]))))

    new_neighbors = {}
    num_vars = len(neighbors[0])
    for i in neighbors.keys():
        checked = 0
        for j in range(num_vars):  # Check if within bounds
            if neighbors[i][j] >= min_allowed[j+1] and neighbors[i][j] <= max_allowed[j+1]:
                checked += 1
        if checked == num_vars:  # Add neighbor if all variables are within bounds
            new_neighbors[i] = neighbors[i]

    return new_neighbors


def evaluate_neighbors(
    stop_when_improvement_found: bool,
    ext_vars: dict,
    fmin: float,
    model_function,
    model_args: dict,
    ext_dict: dict,
    ext_logic,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    iter_timelimit: float = 10,
    current_time: float = 0,
    timelimit: float = 3600,
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 1e-3,
    global_evaluated: list = [],
    init_path=None,
):
    """
    Function that evaluates a group of given points and returns the best
    Args:
        ext_vars: dict with neighbors where neighbor 0 is actual point
        fmin: Objective at actual point
        model_function: function that returns GDP model to be solved
        model_args: Contains the argument values needed for model_function
        ext_dict: Dictionary with Boolean variables to be reformulated (keys) and their corresponding ordered sets (values)
        ext_logic: Function that returns a list of lists of the form [a,b], where a is an expressions of the reformulated Boolean variables and b is an equivalent Boolean or indicator variable (b<->a)
        mip_transformation: Whether to solve the enumeration using the external variables applied to the MIP problem insed of the GDP
        transformation: Which transformation to apply to the GDP 
        subproblem_solver: MINLP or NLP solver algorithm
        subproblem_solver_options: MINLP or NLP solver algorithm options
        iter_timelimit: time limit in seconds for the solve statement for each iteration
        current_time: Current time in global algorithm
        timelimit: time limit in seconds for the algorithm
        gams_output: Determine keeping or not GAMS files
        tee: Display iteration output
        global_tee: display D-SDA iteration output
        rel_tol: Relative optimality tolerance
        global_evaluated: list with points already evaluated
        init_path: path to initialization file
    Returns:
        fmin: Type int and gives the best neighbor's objective
        best_var: Type list and gives the best neighbor
        best_dir: Type int and is the steepest direction (key in neighborhood)
        improve: Type bool and shows if an improvement was made while looking for neighbors
        evaluation_time: Total solver-statement time only
        ns_evaluated: evaluations in neighbor search
        best_path: path to json with best solution found

    """
    # Global Tolerance parameters
    epsilon = 1e-10
    abs_tol = 1e-5
    min_improve = 1e-5
    min_improve_rel = 1e-3

    # Initialize
    ns_evaluated = []
    evaluation_time = 0
    improve = False
    best_var = ext_vars[0]
    here = ext_vars[0]
    best_dir = 0  # Position in dictionary
    best_dist = 0
    best_path = init_path
    temp = ext_vars  # TODO change name to something more saying. Points? Combinations?
    temp.pop(0, None)

    if global_tee:
        print()
        print('Neighbor search around:', best_var)

    for i in temp.keys():   # Solve all models
        if temp[i] not in global_evaluated:
            m = model_function(**model_args)
            m_init = initialize_model(m, json_path=init_path)
            if mip_transformation:  # If you want a MIP reformulation, go ahead and use it'
                m_init, ext_dict = extvars_gdp_to_mip(
                    m=m,
                    gdp_dict_extvar=ext_dict,
                    transformation=transformation,
                )

            m_fixed = external_ref(
                m=m_init,
                x=temp[i],
                extra_logic_function=ext_logic,
                dict_extvar=ext_dict,
                mip_ref=mip_transformation,
                tee=False,
            )
            t_remaining = min(iter_timelimit, timelimit -
                              (time.perf_counter() - current_time))
            if t_remaining < 0:  # No time reamining for optimization
                break
            m_solved = solve_subproblem(
                m=m_fixed,
                subproblem_solver=subproblem_solver,
                subproblem_solver_options=subproblem_solver_options,
                timelimit=t_remaining,
                gams_output=gams_output,
                tee=tee,
                rel_tol=rel_tol)
            evaluation_time += m_solved.dsda_usertime
            ns_evaluated.append(temp[i])
            t_end = time.perf_counter()

            if m_solved.dsda_status == 'Optimal':   # Check if D-SDA status is optimal
                if global_tee:
                    print('Evaluated:', temp[i], '   |   Objective:', round(pe.value(
                        m_solved.obj), 5), '   |   Global Time:', round(t_end - current_time, 2))
                dist = sum((x-y)**2 for x, y in zip(temp[i], here))
                act_obj = pe.value(m_solved.obj)

                # Assuming minimization problem
                # Implements heuristic of largest move
                # if not improve:
                    # We want a minimum improvement in the first found solution
                if (fmin - act_obj) > min_improve or (fmin - act_obj)/(abs(fmin)+epsilon) > min_improve_rel:
                    fmin = act_obj
                    best_var = temp[i]
                    best_dir = i
                    best_dist = dist
                    improve = True
                    best_path = generate_initialization(
                        m_solved, starting_initialization=False, model_name='best')
                    if stop_when_improvement_found:
                        break
                # else:
                #     # We want slightly worse solutions if the distance is larger
                #     if (((act_obj - fmin) < abs_tol) or ((act_obj - fmin)/(abs(fmin)+epsilon) < rel_tol)) and dist >= best_dist:
                #         fmin = act_obj
                #         best_var = temp[i]
                #         best_dir = i
                #         best_dist = dist
                #         improve = True
                #         best_path = generate_initialization(
                #             m_solved, starting_initialization=False, model_name='best')

            if time.perf_counter() - current_time > timelimit:  # current
                break

    if global_tee:
        print()
        print('New best neighbor:', best_var)
    return fmin, best_var, best_dir, improve, evaluation_time, ns_evaluated, best_path

def do_line_search(
    start: list,
    fmin: float,
    direction: list,
    model_function,
    model_args: dict,
    ext_dict: dict,
    ext_logic,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    min_allowed: dict = {},
    max_allowed: dict = {},
    iter_timelimit: float = 10,
    timelimit: float = 3600,
    current_time: float = 0,
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = False,
    rel_tol: float = 1e-3,
    global_evaluated: list = [],
    init_path=None
):
    """
    Function that moves in a given "best direction" and evaluates the new moved point
    Args:
        start: Point of that is to be moved
        fmin: Objective at actual point
        direction: moving direction
        model_function: function that returns GDP model to be solved
        model_args: Contains the argument values needed for model_function
        ext_dict: Dictionary with Boolean variables to be reformulated (keys) and their corresponding ordered sets (values)
        ext_logic: Function that returns a list of lists of the form [a,b], where a is an expressions of the reformulated Boolean variables and b is an equivalent Boolean or indicator variable (b<->a)
        mip_transformation: Whether to solve the enumeration using the external variables applied to the MIP problem insed of the GDP
        transformation: Which transformation to apply to the GDP 
        subproblem_solver: MINLP or NLP solver algorithm
        subproblem_solver_options: MINLP or NLP solver algorithm options
        min_allowed: In keys contains external variables and in items their respective lower bounds
        max_allowed: In keys contains external variables and in items their respective upper bounds
        iter_timelimit: time limit in seconds for the solve statement for each iteration
        current_time: Current time in global algorithm
        gams_output: Determine keeping or not GAMS files
        tee: Display iteration output
        global_tee: display D-SDA iteration output
        rel_tol: Relative optimality tolerance
        global_evaluated: list with points already evaluated
        init_path: path to initialization file
    Returns:
        fmin: Type int and gives the moved point objective
        best_var: Type list and gives the moved point
        moved: Type bool and shows if an improvement was made while line searching
        ls_time: Total solver-statement time only
        ls_evaluated: evaluations in line search
        new_path: path of best json file
    """
    # Global Tolerance parameters
    epsilon = 1e-10
    min_improve = 1e-5
    min_improve_rel = 1e-3

    # Initialize
    ls_evaluated = []
    ls_time = 0
    best_var = start
    moved = False
    new_path = init_path

    # Line search in given direction
    moved_point = list(map(sum, zip(list(start), list(direction))))
    checked = 0
    for j in range(len(moved_point)):   # Check if within bounds
        if moved_point[j] >= min_allowed[j+1] and moved_point[j] <= max_allowed[j+1]:
            checked += 1

    if checked == len(moved_point):     # Solve model
        if moved_point not in global_evaluated:
            m = model_function(**model_args)
            m_init = initialize_model(m, json_path=init_path)
            if mip_transformation:  # If you want a MIP reformulation, go ahead and use it'
                m_init, ext_dict = extvars_gdp_to_mip(
                    m=m,
                    gdp_dict_extvar=ext_dict,
                    transformation=transformation,
                )
            m_fixed = external_ref(
                m=m_init,
                x=moved_point,
                extra_logic_function=ext_logic,
                dict_extvar=ext_dict,
                mip_ref=mip_transformation,
                tee=False,
            )

            t_remaining = min(iter_timelimit, timelimit -
                              (time.perf_counter() - current_time))
            if t_remaining < 0:
                return fmin, best_var, moved, ls_time, ls_evaluated, new_path
            m_solved = solve_subproblem(
                m=m_fixed,
                subproblem_solver=subproblem_solver,
                subproblem_solver_options=subproblem_solver_options,
                timelimit=t_remaining,
                gams_output=gams_output,
                tee=tee,
                rel_tol=rel_tol
            )
            ls_time += m_solved.dsda_usertime
            ls_evaluated.append(moved_point)

            if m_solved.dsda_status == 'Optimal':   # Check status
                if global_tee:
                    print('Evaluated:', moved_point, '   |   Objective:', round(pe.value(
                        m_solved.obj), 5), '   |   Global Time:', round(time.perf_counter() - current_time, 2))
                act_obj = pe.value(m_solved.obj)
                # Return moved point
                if (fmin - act_obj) > min_improve or (fmin - act_obj)/(abs(fmin)+epsilon) > min_improve_rel:
                    fmin = act_obj
                    best_var = moved_point
                    moved = True
                    new_path = generate_initialization(
                        m_solved, starting_initialization=False, model_name='best')

    return fmin, best_var, moved, ls_time, ls_evaluated, new_path

def solve_with_dsda(
    model_function,
    model_args: dict,
    starting_point: list,
    ext_dict,
    ext_logic,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    k: str = 'Infinity',
    provide_starting_initialization: bool = True,
    feasible_model: str = '',
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    iter_timelimit: float = 1000,
    timelimit: float = 3600,
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 1e-3,
    scaling: bool=False,
    scale_factor: float = 1,
    stop_neigh_verif_when_improv: bool=False,
    route_initial: list=[],
    obj_route_initial: list=[]
):
    """
    Function that computes Discrete-Steepest Descend Algorithm
    Args:
        k: Type of neighborhood
        model_function: function that returns GDP model to be solved
        model_args: Contains the argument values needed for model_function
        starting_point: Feasible external variable initial point
        ext_dict: Dictionary with Boolean variables to be reformulated (keys) and their corresponding ordered sets (values). Both keys and values are pyomo objects.
        ext_logic: Function that returns a list of lists of the form [a,b], where a is an expressions of the reformulated Boolean variables and b is an equivalent Boolean or indicator variable (b<->a).
        mip_transformation: Whether to solve the enumeration using the external variables applied to the MIP problem insed of the GDP
        transformation: Which transformation to apply to the GDP 
        provide_intialization: If an existing json file is provided with a feasible initialization of starting_point
        subproblem_solver: MINLP or NLP solver algorithm
        subproblem_solver_options: MINLP or NLP solver algorithm options
        iter_timelimit: time limit in seconds for the solve statement for each iteration
        timelimit: time limit in seconds for the algorithm
        gams_output: Determine keeping or not GAMS files
        tee: Display iteration output
        global_tee: Display D-SDA output
        rel_tol: Relative optimality tolerance
    Returns:
        m2_solved: Solved Pyomo Model
        route: List containing points evaluated in throughout iteration
        obj_route: List containing objectives evaluated in throughout iteration

    """

    if global_tee:
        print('\nStarting D-SDA with k =', k)
        print('--------------------------------------------------------------------------')

    # Initialize
    route =route_initial
    obj_route =obj_route_initial
    global_evaluated =route_initial
    ext_var = starting_point

    # Check if  feasible initialization is provided
    m = model_function(**model_args)
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict)
    if len(starting_point) != num_ext_var:
        print("The size of the initialization vector must be equal to "+str(num_ext_var))

    t_start = time.perf_counter()
    dsda_usertime = 0
    if provide_starting_initialization:
        m_init = initialize_model(
            m, from_feasible=True, feasible_model=feasible_model, json_path=None)
    else:
        m_init = m

    if mip_transformation:  # If you want a MIP reformulation, go ahead and use it'
        m_init, dict_extvar = extvars_gdp_to_mip(
            m=m,
            gdp_dict_extvar=dict_extvar,
            transformation=transformation,
        )

    m_fixed = external_ref(
        m=m_init,
        x=ext_var,
        extra_logic_function=ext_logic,
        dict_extvar=dict_extvar,
        mip_ref=mip_transformation,
        tee=False
    )

    # Solve for initialization
    m_solved = solve_subproblem(
        m=m_fixed,
        subproblem_solver=subproblem_solver,
        subproblem_solver_options=subproblem_solver_options,
        timelimit=iter_timelimit,
        gams_output=gams_output,
        tee=tee,
        rel_tol=rel_tol,
    )
    dsda_usertime += m_solved.dsda_usertime
    fmin = pe.value(m_solved.obj)
    if global_tee:
        print('Initializing...')
        print('Evaluated:', ext_var, '   |   Objective:', round(fmin, 5),
              '   |   Global Time:', round(time.perf_counter() - t_start, 2))
        if m_solved.dsda_status=='FBBT_Infeasible' or m_solved.dsda_status=='Evaluated_Infeasible':
            print('WARNING: Initialization is infeasible. Neighborhood verification will be performed to check if there is a feasible point nearby.') 

    # m_solved.pprint()
    best_path = generate_initialization(m_solved)

    route.append(ext_var)
    obj_route.append(fmin)
    global_evaluated.append(ext_var)

    # Define neighborhood
    if k == '2':
        neighborhood = neighborhood_k_eq_2(len(ext_var))
    elif k == 'Infinity':
        neighborhood = neighborhood_k_eq_inf(len(ext_var))
    elif k == 'L_natural':
        neighborhood = neighborhood_k_eq_l_natural(len(ext_var))   
    elif k == 'L_natural_modified':
        neighborhood = neighborhood_k_eq_l_natural_modified(len(ext_var)) 
    elif k == 'M_natural':
        neighborhood = neighborhood_k_eq_m_natural(len(ext_var))   
    elif k == 'all_interactions':
        neighborhood = neighborhood_k_eq_all(len(ext_var))  
    else:
        return "Enter a valid neighborhood"
    
    if scaling:
        for i in neighborhood.keys():
            neighborhood[i]=list(scale_factor*np.asarray(neighborhood[i]))

    looking_in_neighbors = True

    # Look in neighbors (outer cycle)
    while looking_in_neighbors:

        if time.perf_counter() - t_start > timelimit:
            break

        # Find neighbors of the actual point
        neighbors = find_actual_neighbors(ext_var, neighborhood,
                                          min_allowed=min_allowed, max_allowed=max_allowed)

        if time.perf_counter() - t_start > timelimit:
            break

        fmin, best_var, best_dir, improve, eval_time, ns_evaluated, best_path = evaluate_neighbors(
            stop_when_improvement_found=stop_neigh_verif_when_improv,
            ext_vars=neighbors,
            fmin=fmin,
            model_function=model_function,
            model_args=model_args,
            ext_dict=dict_extvar,
            ext_logic=ext_logic,
            mip_transformation=mip_transformation,
            transformation=transformation,
            subproblem_solver=subproblem_solver,
            subproblem_solver_options=subproblem_solver_options,
            iter_timelimit=iter_timelimit,
            timelimit=timelimit,
            current_time=t_start,
            gams_output=gams_output,
            tee=tee,
            global_tee=global_tee,
            rel_tol=rel_tol,
            global_evaluated=global_evaluated,
            init_path=best_path,
        )

        dsda_usertime += eval_time
        global_evaluated = global_evaluated + ns_evaluated

        # Stopping condition in case there is no improvement amongst neighbors
        if improve:
            line_searching = True
            route.append(best_var)
            obj_route.append(fmin)
            if global_tee and time.perf_counter() - t_start < timelimit:
                print()
                print('Line search in direction:', neighborhood[best_dir])

            # If improvement was made start line search (inner cycle)
            while line_searching:

                if time.perf_counter() - t_start > timelimit:
                    break

                fmin, best_var, moved, ls_time, ls_evaluated, best_path = do_line_search(
                    start=best_var,
                    fmin=fmin,
                    direction=neighborhood[best_dir],
                    model_function=model_function,
                    model_args=model_args,
                    ext_dict=dict_extvar,
                    ext_logic=ext_logic,
                    mip_transformation=mip_transformation,
                    transformation=transformation,
                    subproblem_solver=subproblem_solver,
                    min_allowed=min_allowed,
                    max_allowed=max_allowed,
                    iter_timelimit=iter_timelimit,
                    timelimit=timelimit,
                    current_time=t_start,
                    gams_output=gams_output,
                    tee=tee,
                    global_tee=global_tee,
                    rel_tol=rel_tol,
                    global_evaluated=global_evaluated,
                    init_path=best_path,
                )
                global_evaluated = global_evaluated + ls_evaluated
                dsda_usertime += ls_time

                if time.perf_counter() - t_start > timelimit:
                    break

                # Stopping condition in case no movement was done
                if moved:
                    route.append(best_var)
                    obj_route.append(fmin)
                else:
                    ext_var = best_var
                    line_searching = False
                    if global_tee:
                        print()
                        print('New best point:', best_var)

        else:
            looking_in_neighbors = False

    t_end = round(time.perf_counter() - t_start, 2)

    # Generate final solved model
    m2 = model_function(**model_args)
    m2_solved = initialize_model(m2, json_path=best_path)
    m2_solved.dsda_time = t_end
    m2_solved.dsda_usertime = dsda_usertime
    if t_end > timelimit:
        m2_solved.dsda_status = 'maxTimeLimit'
    else:
        m2_solved.dsda_status = 'optimal'

    # Print results
    if global_tee:
        print('--------------------------------------------------------------------------')
        print('Objective:', round(fmin, 5))
        print('External variables:', route[-1])
        print('Execution time [s]:', t_end)
        print('User time [s]:', round(dsda_usertime, 5))

    return m2_solved, route, obj_route


def build_pretreatment():
    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='reactor_model')
    # ------------scalars    ------------------------------------------------
    m.final_time = pe.Param(initialize=3600)  # [sec]
    m.delta_z = pe.Param(initialize=1.2)  # [m]
    m.L_r = pe.Param(initialize=12)  # [m]
    m.speed = pe.Param(initialize=0.013)  # [m/s]
    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = dae.ContinuousSet(bounds=(0, m.final_time))
    # spatial coordinate index (k)
    m.k = pe.RangeSet(0, 10, 1)
    # chemical species
    m.j = pe.Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS',
              'G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O'])
    # Kinetic parameters
    m.Ts = pe.Param(initialize=185)
    m.EA = pe.Param(initialize=61229)
    m.kA = pe.Param(initialize=106225)
    m.kL = pe.Param(initialize=1.03E+33)
    m.EL = pe.Param(initialize=325629)
    m.kXo = pe.Param(initialize=2.78E+31)
    m.EXo = pe.Param(initialize=298011)
    m.kX = pe.Param(initialize=1.31E+34)
    m.EX = pe.Param(initialize=304680)
    m.kG = pe.Param(initialize=1.11E+35)
    m.EG = pe.Param(initialize=335614)
    m.kPL = pe.Param(initialize=1.03E+33)
    m.EPL = pe.Param(initialize=325629)
    m.kF = pe.Param(initialize=5.09E+33)
    m.EF = pe.Param(initialize=327253)
    m.kAc = pe.Param(initialize=4.88E+24)
    m.EAc = pe.Param(initialize=242687)
    m.kH = pe.Param(initialize=1E+31)
    m.EH = pe.Param(initialize=300000)
    m.alpha = pe.Param(initialize=0.1019)
    m.Fin = pe.Param(initialize=6)  # [kg/s]
    m.h0 = pe.Param(initialize=117)  # [kJ/kg]
    m.cb = pe.Param(initialize=3.8)  # [kJ/kg K]
    m.Rg = pe.Param(initialize=8.3145)  # [J/mol K]
    m.C0 = pe.Param(m.j, initialize={'CS': 125, 'XS': 70, 'AS': 8, 'LS': 80, 'ACS': 16,
                 'G': 0, 'XO': 0, 'X': 0, 'A': 0, 'AC': 0, 'F': 0, 'H': 0, 'W': 600, 'O': 101})
    # -----------variables ---------------------------------------------------
    m.c = pe.Var(m.t, m.k, m.j, initialize=500,
              within=pe.NonNegativeReals, bounds=(0, 10000))
    m.dcdt = dae.DerivativeVar(m.c, withrespectto=m.t)
    m.h = pe.Var(m.t, m.k, initialize=100,
              within=pe.NonNegativeReals, bounds=(0, 10000))
    m.dhdt = dae.DerivativeVar(m.h, withrespectto=m.t)
    m.R = pe.Var(m.t, m.k, m.j, within=pe.Reals)
    # ----------Balance equations (concentration and energy)-------------------

    def _dcdt(m, t, k, j):
        if k == m.k.first() or t == m.t.first():
            return pe.Constraint.Skip
        return m.dcdt[t, k, j] == (m.speed/m.delta_z)*(m.c[t, k-1, j]-m.c[t, k, j])+m.R[t, k, j]
    m.c_dcdt = pe.Constraint(m.t, m.k, m.j, rule=_dcdt)

    def _T(m, t, k):
        return m.h[t, k]/m.cb + 273
    m.T = pe.Expression(m.t, m.k, rule=_T)

    def _dhdt(m, t, k):
        if k == m.k.first() or t == m.t.first():
            return pe.Constraint.Skip
        return m.dhdt[t, k] == (m.speed/m.delta_z)*(m.h[t, k-1]-m.h[t, k])-0.000192*(m.T[t, k]-298)

    m.c_dhdt = pe.Constraint(m.t, m.k, rule=_dhdt)

    def _rG(m, t, k):
        return m.kG*pe.exp(-m.EG/(m.Rg*m.T[t, k]))*m.c[t, k, 'CS']
    m.rG = pe.Expression(m.t, m.k, rule=_rG)

    def _rH(m, t, k):
        return m.kH*pe.exp(-m.EH/(m.Rg*m.T[t, k]))*m.c[t, k, 'G']
    m.rH = pe.Expression(m.t, m.k, rule=_rH)

    def _rA(m, t, k):
        return m.kA*pe.exp(-m.EA/(m.Rg*m.T[t, k]))*m.c[t, k, 'AS']
    m.rA = pe.Expression(m.t, m.k, rule=_rA)

    def _rXo(m, t, k):
        return m.kXo*pe.exp(-m.EXo/(m.Rg*m.T[t, k]))*m.c[t, k, 'XS']
    m.rXo = pe.Expression(m.t, m.k, rule=_rXo)

    def _rX(m, t, k):
        return m.kX*pe.exp(-m.EX/(m.Rg*m.T[t, k]))*m.c[t, k, 'XO']
    m.rX = pe.Expression(m.t, m.k, rule=_rX)

    def _rFx(m, t, k):
        return m.kF*pe.exp(-m.EF/(m.Rg*m.T[t, k]))*(m.c[t, k, 'X'])
    m.rFx = pe.Expression(m.t, m.k, rule=_rFx)

    def _rFa(m, t, k):
        return m.kF*pe.exp(-m.EF/(m.Rg*m.T[t, k]))*(m.c[t, k, 'A'])
    m.rFa = pe.Expression(m.t, m.k, rule=_rFa)

    def _rF(m, t, k):
        return m.rFa[t, k]+m.rFx[t, k]
    m.rF = pe.Expression(m.t, m.k, rule=_rF)

    def _rLXo(m, t, k):
        return m.kL*pe.exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'XO'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLXo = pe.Expression(m.t, m.k, rule=_rLXo)

    def _rLX(m, t, k):
        return m.kL*pe.exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'X'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLX = pe.Expression(m.t, m.k, rule=_rLX)

    def _rLA(m, t, k):
        return m.kL*pe.exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'A'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLA = pe.Expression(m.t, m.k, rule=_rLA)

    def _rLG(m, t, k):
        return m.kL*pe.exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'G'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLG = pe.Expression(m.t, m.k, rule=_rLG)

    def _rL(m, t, k):
        return m.rLXo[t, k]+m.rLX[t, k]+m.rLA[t, k]+m.rLG[t, k]
    m.rL = pe.Expression(m.t, m.k, rule=_rL)

    def _rLF(m, t, k):
        return m.kL*pe.exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'XO']+m.c[t, k, 'X']+m.c[t, k, 'A']+m.c[t, k, 'G'])*(m.c[t, k, 'F'])
    m.rLF = pe.Expression(m.t, m.k, rule=_rLF)

    def _rLH(m, t, k):
        return m.kL*pe.exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'XO']+m.c[t, k, 'X']+m.c[t, k, 'A']+m.c[t, k, 'G'])*(m.c[t, k, 'H'])
    m.rLH = pe.Expression(m.t, m.k, rule=_rLH)

    def _rAc(m, t, k):
        return m.kAc*pe.exp(-m.EAc/(m.Rg*m.T[t, k]))*(m.c[t, k, 'ACS'])
    m.rAc = pe.Expression(m.t, m.k, rule=_rAc)

    def _R(m, t, k, j):
        if j == 'CS':
            return m.R[t, k, j] == -m.rG[t, k]
        elif j == 'XS':
            return m.R[t, k, j] == -m.rXo[t, k]
        elif j == 'AS':
            return m.R[t, k, j] == -m.rA[t, k]
        elif j == 'LS':
            return m.R[t, k, j] == m.rL[t, k]
        elif j == 'ACS':
            return m.R[t, k, j] == -m.rAc[t, k]
        elif j == 'G':
            return m.R[t, k, j] == m.rG[t, k]-(1-m.alpha)*m.rLG[t, k]
        elif j == 'XO':
            return m.R[t, k, j] == m.rXo[t, k]-m.rX[t, k]-(1-m.alpha)*m.rLXo[t, k]
        elif j == 'X':
            return m.R[t, k, j] == m.rX[t, k]-m.rFx[t, k]-(1-m.alpha)*m.rLX[t, k]
        elif j == 'A':
            return m.R[t, k, j] == m.rA[t, k]-m.rFa[t, k]-(1-m.alpha)*m.rLA[t, k]
        elif j == 'AC':
            return m.R[t, k, j] == m.rAc[t, k]
        elif j == 'F':
            return m.R[t, k, j] == m.rF[t, k] - m.alpha*m.rLF[t, k]
        elif j == 'H':
            return m.R[t, k, j] == m.rH[t, k] - m.alpha*m.rLH[t, k]
        elif j == 'W':
            return m.R[t, k, j] == 0
        else:
            return m.R[t, k, j] == 0

    m.c_R = pe.Constraint(m.t, m.k, m.j, rule=_R)


    # INITIAL CONDITIONS
    def _IC1(m, k, j):
        if k == 0:
            return pe.Constraint.Skip
        return m.c[0, k, j] == m.C0[j]
    m.IC1 = pe.Constraint(m.k, m.j, rule=_IC1)

    def _IC2(m, k):
        if k == 0:
            return pe.Constraint.Skip
        return m.h[0, k] == m.h0
    m.IC2 = pe.Constraint(m.k, rule=_IC2)
    
    # BOUNDARY CONDITIONS
    def _IC3(m, t, j):

        return m.c[t, 0, j] == m.C0[j]
    m.IC3 = pe.Constraint(m.t, m.j, rule=_IC3)

    def _IC4(m, t):
        return m.h[t, 0] == m.cb*(m.Ts)
    m.IC4 = pe.Constraint(m.t, rule=_IC4)

    m.obj = pe.Objective(expr=(-m.c[3600, 10, 'G']))

    discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # discretizer = TransformationFactory('dae.collocation')
    # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    return m

def build_hydrolisis(time: float=170*(3600),discretization: str='collocation',n_f_elements_x: int=10,n_f_elements_t: int=10) -> pe.ConcreteModel(): #TODO: MODIFY INPUTS
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
            elif x==m.x.first(): #boundary condition 1 
                return m.C[t,x,j] == m.CFEED[j]
            elif x==m.x.last():  #boundary condition 2 
                return m.dCdx[t,x,j] == 0
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] + m.D[t,x]*m.dC2dx2[t,x,j]+m.dCdx[t,x,j]*m.dDdx[t,x]+m.R[t,x,j]
        else:
            if t==m.t.first() and x>m.x.first(): #Initial condition
                return m.C[t,x,j] == m.C0[j]
            elif x==m.x.first(): #boundary condition 1 
                return m.C[t,x,j] == m.CFEED[j]
            else:  # Partial differential equation
                return  m.dCdt[t,x,j]== -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.partialDiff=pe.Constraint(m.t,m.x,m.j,rule=_partialDiff)

    # INITIAL CONDITION
    # def _partialDiff_init(m,t,x,j):
    #     if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): # NOTE: According to prunescu model, diffusivity effects are only considered in the liquid fraction of the slurry
    #         if t==m.t.first() and x>m.x.first() and x<m.x.last(): #Initial condition
    #             return m.C[t,x,j] == m.C0[j]
    #         else: 
    #             return  pe.Constraint.Skip
    #     else:
    #         if t==m.t.first() and x>m.x.first(): #Initial condition
    #             return m.C[t,x,j] == m.C0[j]
    #         else: 
    #             return  pe.Constraint.Skip            
    # m.partialDiff_init=pe.Constraint(m.t,m.x,m.j,rule=_partialDiff_init)   


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

def build_hydrolisis_cstr(m):

    # Generating CSTR equations to model what happens at the exit of the first hydrolisis reactor
    m_cstr=pe.ConcreteModel(name='CSTR hydrolisis reactor')
    m_cstr.tau=pe.Param(initialize=140*3600-m.hyd.tR,doc='residence time of the CSTR reactor in seconds')

    m_cstr.t=pe.Set(initialize=m.hyd.t)
    m_cstr.j=pe.Set(initialize=m.hyd.j)
    m_cstr.e=pe.Set(initialize=m.hyd.e)

    m_cstr.alpha_enzymes=pe.Param(m_cstr.e,initialize=m.hyd.alpha_enzymes,doc='Fraction of each enzyme type (between 0 and 1)')
    m_cstr.max_ads_enz=pe.Param(m_cstr.e,initialize=m.hyd.max_ads_enz,doc='Maximum adsorbed enzymes [-]')
    m_cstr.k_ads=pe.Param(m_cstr.e,initialize=m.hyd.k_ads,doc='Adsorption constant [-]')


    m_cstr.C=pe.Var(m_cstr.t, m_cstr.j, initialize=1,within=pe.NonNegativeReals, doc='Concentrations, units of g/kg') #bounds=(0, 10000))
    m_cstr.Ce=pe.Var(m_cstr.t, m_cstr.e, initialize=1,within=pe.NonNegativeReals, doc='Enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m_cstr.Cef=pe.Var(m_cstr.t, m_cstr.e, initialize=1,within=pe.NonNegativeReals, doc='Free enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m_cstr.Ceb=pe.Var(m_cstr.t,m_cstr.e, initialize=1,within=pe.NonNegativeReals, doc='Bounded enzyme types concentrations, units of g/kg') #bounds=(0, 10000))
    m_cstr.CebC=pe.Var(m_cstr.t,  m_cstr.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to cellulose g/kg')
    m_cstr.CebX=pe.Var(m_cstr.t, m_cstr.e, initialize=1,within=pe.NonNegativeReals, doc='Concentration of adsorbed enzymes to xylan g/kg')
    m_cstr.r1=pe.Var(m_cstr.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to cellobiose rate, g/kg s')
    m_cstr.r2=pe.Var(m_cstr.t,initialize=1,within=pe.NonNegativeReals, doc='Cellulose to glucose rate, g/kg s')
    m_cstr.r3=pe.Var(m_cstr.t,initialize=1,within=pe.NonNegativeReals, doc='Cellobiose to glucose rate, g/kg s')
    m_cstr.r4=pe.Var(m_cstr.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to xylose rate, g/kg s')
    m_cstr.r5=pe.Var(m_cstr.t,initialize=1,within=pe.NonNegativeReals, doc='Xylan to acetic acid rate, g/kg s')
    m_cstr.R = pe.Var(m_cstr.t, m_cstr.j, initialize=1, within=pe.Reals, doc='units of g/ (kg s)')

    m_cstr.eta=pe.Var(m_cstr.t,within=pe.NonNegativeReals,initialize=1,doc='temperature and pH dependence of reaction rates')
    m_cstr.Cfeed=pe.Var(m_cstr.t, m_cstr.j, initialize=1,within=pe.NonNegativeReals, doc='Feed concentrations, units of g/kg')
     
    # enzyme balances
    def _enzyme_fractions_cstr(m,t,e):
        return m.Ce[t,e] == m.alpha_enzymes[e]*m.C[t,'E']
    m_cstr.enzyme_fractions=pe.Constraint(m_cstr.t,m_cstr.e,rule=_enzyme_fractions_cstr)

    def _bounded_free_equilibrium_cstr(m,t,e):
        return m.Ce[t,e] == m.Ceb[t,e]  +    m.Cef[t,e]
    m_cstr.bounded_free_equilibrium=pe.Constraint(m_cstr.t,m_cstr.e,rule=_bounded_free_equilibrium_cstr)

    def _adsorbed_free_equilibrium_cstr(m,t,e): #NOTE: I am assuming that the concentration solids does not include enzymes. #TODO: check the effect of including them +sum(m_cstr.Ceb[t,x,e] for e in m_cstr.e)
    
        return (m.Ceb[t,e])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']) == m.max_ads_enz[e]*((m.k_ads[e]*m.Cef[t,e])/(1+m.k_ads[e]*m.Cef[t,e]))

    m_cstr.adsorbed_free_equilibrium=pe.Constraint(m_cstr.t,m_cstr.e,rule=_adsorbed_free_equilibrium_cstr)

    def _bounded_enzyme_concentration_cstr(m,t,e):
        if e=='1' or e=='2':                            # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebC[t,e] == m.Ceb[t,e]*((m.C[t,'CS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS'])) 
        else:                                           # NOTE: that denominator is Solid concentration. modify if needed
            return m.CebX[t,e] == m.Ceb[t,e]*((m.C[t,'XS'])/(m.C[t,'CS']+ m.C[t,'XS']+m.C[t,'LS']))
    m_cstr.bounded_enzyme_concentration=pe.Constraint(m_cstr.t,m_cstr.e,rule=_bounded_enzyme_concentration_cstr)


    # MODELING OF REACTIONS
    def _r1_definition_cstr(m,t):
        K1_r1=0.00034       # reaction rate constant, kg/(g*s)
        IC1_r1=0.0014       # Inhibition of r1 by cellobiose, g/kg
        IX1_r1=0.1007       # Inhibition of r1 by xylose, g/kg
        IG1_r1=0.073        # Inhibition of r1 by glucose, g/kg
        IF1_r1=10           #  Inhibition of r1 by furfural, g/kg
        
        return m.r1[t] == (K1_r1*m.eta[t]*m.CebC[t,'1']*m.C[t,'CS'])/(1+(m.C[t,'C']/IC1_r1)+(m.C[t,'X']/IX1_r1)+(m.C[t,'G']/IG1_r1)+(m.C[t,'F']/IF1_r1))
    m_cstr.r1_definition=pe.Constraint(m_cstr.t,rule=_r1_definition_cstr)

    def _r2_definition_cstr(m,t):
        K2_r2=0.0023 #changed         # reaction rate constant, kg/(g*s)
        IC2_r2=132          # Inhibition of r2 by cellobiose, g/kg
        IX2_r2=0.029           # Inhibition of r2 by xylose, g/kg
        IG2_r2=0.34          # Inhibition of r2 by glucose, g/kg
        IF2_r2=10          #  Inhibition of r2 by furfural, g/kg
        return m.r2[t] == (K2_r2*m.eta[t]*(m.CebC[t,'1']+m.CebC[t,'2'])*m.C[t,'CS'])/(1+(m.C[t,'C']/IC2_r2)+(m.C[t,'X']/IX2_r2)+(m.C[t,'G']/IG2_r2)+(m.C[t,'F']/IF2_r2))
    m_cstr.r2_definition=pe.Constraint(m_cstr.t,rule=_r2_definition_cstr)

    def _r3_definition_cstr(m,t):
        K3_r3=0.07                # reaction rate constant, kg/(g*s)
        I3_r3=24.3               #overall inhibition term for r3, g/kg
        IX3_r3= 201              # Inhibition of r3 by xylose, g/kg
        IG3_r3= 3.9             # Inhibition of r3 by glucose, g/kg
        IF3_r3=10               #  Inhibition of r3 by furfural, g/kg
        return m.r3[t] == (K3_r3*m.eta[t]* m.Cef[t,'2']*m.C[t,'C'])/(I3_r3*(1+(m.C[t,'X']/IX3_r3)+(m.C[t,'G']/IG3_r3)+(m.C[t,'F']/IF3_r3))+m.C[t,'C'])
    m_cstr.r3_definition=pe.Constraint(m_cstr.t,rule=_r3_definition_cstr)

    def _r4_definition_cstr(m,t):
        K4_r4=0.0087#0.0027     # reaction rate constant, kg/(g*s)
        IC4_r4= 24.3         # Inhibition of r4 by cellobiose, g/kg
        IX4_r4= 201         # Inhibition of r4 by xylose, g/kg 
        IG4_r4= 2.39         # Inhibition of r4 by glucose, g/kg
        IF4_r4= 10         #  Inhibition of r4 by furfural, g/kg
        return m.r4[t] == (K4_r4*m.eta[t]*m.CebX[t,'3']*m.C[t,'XS'])/(1+(m.C[t,'C']/IC4_r4)+(m.C[t,'X']/IX4_r4)+(m.C[t,'G']/IG4_r4)+(m.C[t,'F']/IF4_r4))
    m_cstr.r4_definition=pe.Constraint(m_cstr.t,rule=_r4_definition_cstr)

    def _r5_definition_cstr(m,t):
        Beta_r5=0.5     # acetic acid to xylose ratio
        return m.r5[t] ==Beta_r5*m.r4[t] 
    m_cstr.r5_definition=pe.Constraint(m_cstr.t,rule=_r5_definition_cstr)
    # ['CS', 'XS', 'LS',              'C','G', 'X', 'F', 'E','AC']
    def _R_definition_cstr(m,t,j):
        if j=='CS':              
            return m.R[t,j] == -m.r1[t]-m.r2[t] #Cellulose->Cellobiose (r1), #Cellulose->Glucose (r2) 
        elif j=='XS':
            return m.R[t,j] == -m.r4[t]-m.r5[t] #Xylan->Xylose (r4), #Xylan->Acetic Acid (r5)
        elif j=='LS':
            return m.R[t,j] == 0 
        elif j=='C':
            return m.R[t,j] == m.r1[t]-m.r3[t]     #Cellulose->Cellobiose (r1),  #Cellobiose->Glucose (r3)
        elif j=='G':
            return m.R[t,j] == m.r2[t]+m.r3[t]      #Cellulose->Glucose (r2), #Cellobiose->Glucose (r3)
        elif j=='X':
            return m.R[t,j] == m.r4[t] #Xylan->Xylose (r4)
        elif j=='F':
            return m.R[t,j] == 0
        elif j=='E':
            return m.R[t,j] == 0 #NOTE: Deactivation of enzymes is not considered in Prunescu work
        elif j=='AC':
            return m.R[t,j] == m.r5[t] #Xylan->Acetic Acid (r5)         
    m_cstr.R_definition=pe.Constraint(m_cstr.t,m_cstr.j, rule=_R_definition_cstr)


    # CSTR EQUATION
    def _cstr_eq(m,t,j):
        return m.C[t,j]==m.Cfeed[t,j]+ m.tau*m.R[t,j]
    m_cstr.cstr_eq=pe.Constraint(m_cstr.t,m_cstr.j, rule=_cstr_eq)

    return m_cstr

def build_fermentation_old(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

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

def build_fermentation_pH_control_adjustment2(discretization: str='collocation',n_f_elements_t: int=10,data: dict={}) -> pe.ConcreteModel():

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

    # m.K0G=pe.Param(initialize=1,doc='Parameter for pH dependency in glucose rate of fermentation model')
    # m.K1G=pe.Param(initialize=4.846165818555403,doc='Parameter for pH dependency in glucose rate of fermentation model') 
    # m.K2G=pe.Param(initialize=0.220507978244498,doc='Parameter for pH dependency in glucose rate of fermentation model')


    # m.K0X=pe.Param(initialize=1.0,doc='Parameter for pH dependency in xylose rate of fermentation model')
    # m.K1X=pe.Param(initialize=5.263611373537922,doc='Parameter for pH dependency in xylose rate of fermentation model')
    # m.K2X=pe.Param(initialize=0.010727558852723,doc='Parameter for pH dependency in xylose rate of fermentation model')
    
    m.K0G=pe.Var(initialize=1,within=pe.NonNegativeReals,bounds=(0.01,1),doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K1G=pe.Var(initialize=4.846165818555403,within=pe.NonNegativeReals,bounds=(4,6),doc='Parameter for pH dependency in glucose rate of fermentation model')
    m.K2G=pe.Var(initialize=0.220507978244498,within=pe.NonNegativeReals,bounds=(1e-9,10),doc='Parameter for pH dependency in glucose rate of fermentation model')


    m.K0X=pe.Var(initialize=1,within=pe.NonNegativeReals,bounds=(0.01,1),doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K1X=pe.Var(initialize=5.263611373537922,within=pe.NonNegativeReals,bounds=(4,6),doc='Parameter for pH dependency in xylose rate of fermentation model')
    m.K2X=pe.Var(initialize=0.010727558852723,within=pe.NonNegativeReals,bounds=(1e-9,10),doc='Parameter for pH dependency in xylose rate of fermentation model')
   
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

    m.F_base=pe.Var(m.t,initialize=0, within=pe.NonNegativeReals,doc='base flow for pH control [kg/s]')
    m.F_acid=pe.Var(m.t,initialize=0, within=pe.NonNegativeReals,doc='acid flow for pH control [kg/s]')
    


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



    _C_base={}
    _C_base['Base']=270 #based on hydrolisis model
    m.C_base=pe.Param(m.j,initialize=_C_base,default=0,doc='Base control flow concentration [g/kg]')

    _C_acid={}
    _C_acid['AC']=100 # TODO find an appropriate value
    m.C_acid=pe.Param(m.j,initialize=_C_acid,default=0,doc='Acid control flow concentration [g/kg]')


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
    m.Fin=pe.Var(m.t,initialize=0,within=pe.NonNegativeReals,doc='Feed flow [kg/s]')
    m.Cin=pe.Var(m.t,m.j,initialize=0,within=pe.NonNegativeReals,doc='Feed composition [g/kg]')
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]') # NOTE we are not considering the unload phase, hence we fix output flow as 0 

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
    m.M=pe.Var(m.t,initialize=1,within=pe.NonNegativeReals,bounds=(0,m.Mmax), doc='Fermenter hold-up in kg') #MAXIMUM HOLD UP IN m^3 is 250   The fermentation tank is filled up to 220 t with a constant feed rate calculated as the sum between the enzymatic hydrolysis outflow rate and the C5 liquid from the pretreatment process
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


    # ------------------Definition of feed flow and output flow information---------------------
    for t in m.t:
        m.Fout[t]=0
        m.F_acid[t].fix(0)

    def _Feed_constraint(m,t):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Fin[t]==m.F_liquified_fibers
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Fin[t]==m.F_C5liquid + m.F_liquified_fibers+m.F_base[t]+m.F_acid[t]       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Fin[t]==m.F_base[t]+m.F_acid[t]
    m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    def _Feed_concentration_constraint(m,t,j):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Cin[t,j]==m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Cin[t,j]*(m.F_C5liquid + m.F_liquified_fibers+m.F_base[t]+m.F_acid[t])==(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j]+m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j])
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Cin[t,j]*(m.F_base[t]+m.F_acid[t])== m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j]    
    m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)

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
        # if t==m.t.first():
        #     return m.pH[t]==m.pH[m.t.next(t)]
        # else:
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
        # sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        # 1000*sum( (m.pH[t]-5.5)**2 for t in m.t)
        return sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        # w={}
        # w['Eth']=1
        # w['G']=1
        # w['X']=1
        # return sum(sum( w[j]*((m.C[t,j]-data[j][m.t.ord(t)-1])**2)    for j in ['G','X','Eth']) for t in m.t) 
    m.obj = pe.Objective(rule=_obj)

    return m

def build_fermentation(discretization: str='collocation',n_f_elements_t: int=10) -> pe.ConcreteModel():

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

    m.F_C5liquid=pe.Param(initialize=628*(1/60)*(1/60),doc='C5liquid flow [kg/s]')
    m.F_liquified_fibers=pe.Param(initialize=2487*(1/60)*(1/60),doc='Liquified fibers flow [kg/s]')

    m.F_base=pe.Var(m.t,initialize=0.01, within=pe.NonNegativeReals,bounds=(0,0.01),doc='base flow for pH control [kg/s]')
    m.F_acid=pe.Param(m.t,initialize=0,mutable=True,doc='acid flow for pH control [kg/s]')
    


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



    _C_base={}
    _C_base['Base']=270 #based on hydrolisis model
    m.C_base=pe.Param(m.j,initialize=_C_base,default=0,doc='Base control flow concentration [g/kg]')

    _C_acid={}
    _C_acid['AC']=100 # TODO find an appropriate value
    m.C_acid=pe.Param(m.j,initialize=_C_acid,default=0,doc='Acid control flow concentration [g/kg]')


    #----- Initical conditions  ----------------------------------
    m.M0_fibers=pe.Param(initialize=1e-8,doc='Initial liquified fibers hold up in the reactor [kg]')
    m.M0_yeast=pe.Param(initialize=147,doc='Initial yeast hold up in the reactor [kg]')
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
    m.Fout=pe.Param(m.t,initialize=0,mutable=True,doc='Output flow [kg/s]') # NOTE we are not considering the unload phase, hence we fix output flow as 0 

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
        return m.C0[j]
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


    # ------------------Definition of feed flow and output flow information---------------------
    for t in m.t:
        m.Fout[t]=0
        m.F_acid[t]=0

    def _Feed_constraint(m,t):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Fin[t]==m.F_liquified_fibers
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Fin[t]==m.F_C5liquid + m.F_liquified_fibers+m.F_base[t]+m.F_acid[t]       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Fin[t]==m.F_base[t]+m.F_acid[t]
    m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    def _Feed_concentration_constraint(m,t,j):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Cin[t,j]==m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Cin[t,j]*(m.F_C5liquid + m.F_liquified_fibers+m.F_base[t]+m.F_acid[t])==(m.F_C5liquid*m.C_C5liquid[j]+m.F_liquified_fibers*m.C_liquified_fibers[j]+m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j])
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Cin[t,j]*(m.F_base[t]+m.F_acid[t])== m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j]    
    m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)

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

    m.avance=pe.Var(m.t,m.r_elect,within=pe.Reals,bounds=(-10,10),initialize=0,doc='production/consumption terms in reactions for pH calculations')

    m.C_elect_init=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,bounds=(0,10),initialize=0.001,doc='Initial concentration of electrolytes')

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

    m.C_elect_equil=pe.Var(m.t,m.j_elect,within=pe.NonNegativeReals,bounds=(0,10),initialize=1E-5,doc='Equilibrium concentration of electrolytes')


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
    m.pH=pe.Var(m.t,within=pe.NonNegativeReals,initialize=_pH,bounds=(1,8),doc='pH profile for model validation')


    def _pH_definition(m,t):
        # return m.pH[t]==-pe.log10(m.C_elect_equil[t,'H+'])
        # if t==m.t.first():
        #     return m.pH[t]==m.pH[m.t.next(t)]
        # else:
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
        # sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        # 1000*sum( (m.pH[t]-5.5)**2 for t in m.t)
        return 1 
        #sum(sum( (m.C[t,j]-data[j][m.t.ord(t)-1])**2    for j in ['G','X','Eth','Cell','CS','XS','E']) for t in m.t)
        # w={}
        # w['Eth']=1
        # w['G']=1
        # w['X']=1
        # return sum(sum( w[j]*((m.C[t,j]-data[j][m.t.ord(t)-1])**2)    for j in ['G','X','Eth']) for t in m.t) 
    m.obj = pe.Objective(rule=_obj)

    return m

def fermentation_optimal_control(include_parametrization: bool=True,simple_parametrization: bool=True,fix_base_flow: bool=True, include_flow_integrals: bool=True, fed_batch_and_batch_phase: bool=True, param_val: int=2)-> pe.ConcreteModel():
    
    m=pe.ConcreteModel(name='Fermentation_control')


    m_fer=build_fermentation(discretization='differences',n_f_elements_t=50)
    m.fer=initialize_model(m_fer,from_feasible=True,feasible_model='validation_fermentation')


    if fix_base_flow:
        # Keeping base dosification fixed for the moment
        for t in m.fer.t:
            m.fer.F_base[t].fix(m.fer.F_base[t].value) 
    m.fer.del_component(m.fer.obj)


    # Remove constant feed parameters
    m.fer.del_component(m.fer.F_C5liquid)
    m.fer.del_component(m.fer.F_liquified_fibers)

    # Creating new variables for time
    def _F_C5liquid_new(m,t):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return 0
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return 628*(1/60)*(1/60)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return 0
    m.fer.F_C5liquid_new=pe.Var(m.fer.t,initialize=_F_C5liquid_new,within=pe.NonNegativeReals,bounds=(0,628*(1/60)*(1/60)*2),doc='C5liquid flow [kg/s]')


    def _F_liquified_fibers_new(m,t):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return 2487*(1/60)*(1/60)
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return 2487*(1/60)*(1/60)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return 0
    m.fer.F_liquified_fibers_new=pe.Var(m.fer.t,initialize=_F_liquified_fibers_new,within=pe.NonNegativeReals,bounds=(0,2487*(1/60)*(1/60)*2),doc='C5liquid flow [kg/s]')

    # Updating input flow constraints
    m.fer.del_component(m.fer.Feed_constraint)
    m.fer.del_component(m.fer.Feed_concentration_constraint)

    def _Feed_constraint_new(m,t):
        if fed_batch_and_batch_phase:
            if t*m.final_time<=10*60*60: # Inoculum phase
                return m.Fin[t]==m.F_liquified_fibers_new[t]
            elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
                return m.Fin[t]==m.F_C5liquid_new[t] + m.F_liquified_fibers_new[t]+m.F_base[t]+m.F_acid[t]       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
            elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
                return m.Fin[t]==m.F_base[t]+m.F_acid[t]
        else:
            if t*m.final_time<=10*60*60: # Inoculum phase
                return m.Fin[t]==m.F_liquified_fibers_new[t]
            else:
                return m.Fin[t]==m.F_C5liquid_new[t] + m.F_liquified_fibers_new[t]+m.F_base[t]+m.F_acid[t]       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
    m.fer.Feed_constraint_new=pe.Constraint(m.fer.t,rule=_Feed_constraint_new)

    def _Feed_concentration_constraint_new(m,t,j):
        if fed_batch_and_batch_phase:
            if t*m.final_time<=10*60*60: # Inoculum phase
                return m.Cin[t,j]==m.C_liquified_fibers[j]
            elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
                return m.Cin[t,j]*(m.F_C5liquid_new[t] + m.F_liquified_fibers_new[t]+m.F_base[t]+m.F_acid[t])==(m.F_C5liquid_new[t]*m.C_C5liquid[j]+m.F_liquified_fibers_new[t]*m.C_liquified_fibers[j]+m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j])
            elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
                return m.Cin[t,j]*(m.F_base[t]+m.F_acid[t])== m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j]    
        else:
            if t*m.final_time<=10*60*60: # Inoculum phase
                return m.Cin[t,j]==m.C_liquified_fibers[j]
            else: 
                return m.Cin[t,j]*(m.F_C5liquid_new[t] + m.F_liquified_fibers_new[t]+m.F_base[t]+m.F_acid[t])==(m.F_C5liquid_new[t]*m.C_C5liquid[j]+m.F_liquified_fibers_new[t]*m.C_liquified_fibers[j]+m.F_base[t]*m.C_base[j]+m.F_acid[t]*m.C_acid[j])          
    m.fer.Feed_concentration_constraint_new=pe.Constraint(m.fer.t,m.fer.j,rule=_Feed_concentration_constraint_new)

    # FLOW integrals and constraints
    if include_flow_integrals:
        def _ingegral_F(m):
            return m.final_time*sum(  (((m.F_liquified_fibers_new[m.t.prev(t)])+(m.F_liquified_fibers_new[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())<=2500*(70-0)#2487*(70-0)
        m.fer.ingegral_F=pe.Constraint(rule=_ingegral_F)

        def _ingegral_C5(m):
            return m.final_time*sum(  (((m.F_C5liquid_new[m.t.prev(t)])+(m.F_C5liquid_new[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())<=650*(70-10)#628*(70-10)
        m.fer.ingegral_C5=pe.Constraint(rule=_ingegral_C5)
    if include_parametrization:
        # Creating parametrization of flows
        if simple_parametrization:
            # Parametrization coefficients
            m.fer.a0_C5=pe.Var(initialize=628*(1/60)*(1/60),within=pe.Reals)
            m.fer.a1_C5=pe.Var(initialize=0,within=pe.Reals)
            m.fer.a2_C5=pe.Var(initialize=0,within=pe.Reals)
            m.fer.a3_C5=pe.Var(initialize=0,within=pe.Reals)

            # m.fer.a_inoc_F=pe.Var(initialize=2487*(1/60)*(1/60),within=pe.NonNegativeReals)
            m.fer.a0_F=pe.Var(initialize=2487*(1/60)*(1/60),within=pe.Reals)
            m.fer.a1_F=pe.Var(initialize=0,within=pe.Reals)
            m.fer.a2_F=pe.Var(initialize=0,within=pe.Reals)
            m.fer.a3_F=pe.Var(initialize=0,within=pe.Reals)

            # Relationship between flow and coefficients
            def _parametrization_C5(m,t):
                if fed_batch_and_batch_phase:
                    if t*m.final_time<=10*60*60: # Inoculum phase  
                        return m.F_C5liquid_new[t]==0
                    elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
                        return m.F_C5liquid_new[t]==m.a0_C5+m.a1_C5*pe.cos(m.a2_C5*t+m.a3_C5)
                    elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
                        return m.F_C5liquid_new[t]==0
                else:
                    if t*m.final_time<=10*60*60: # Inoculum phase  
                        return m.F_C5liquid_new[t]==0
                    else:
                        return m.F_C5liquid_new[t]==m.a0_C5+m.a1_C5*pe.cos(m.a2_C5*t+m.a3_C5)            
            m.fer.parametrization_C5=pe.Constraint(m.fer.t,rule=_parametrization_C5)

            def _parametrization_F(m,t):
                if fed_batch_and_batch_phase:
                    if t*m.final_time<=10*60*60: # Inoculum phase  
                        return m.F_liquified_fibers_new[t]== m.a0_F+m.a1_F*pe.cos(m.a2_F*t+m.a3_F)
                    elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
                        return  m.F_liquified_fibers_new[t]==m.a0_F+m.a1_F*pe.cos(m.a2_F*t+m.a3_F)
                    elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
                        return m.F_liquified_fibers_new[t]==0
                else:
                    return  m.F_liquified_fibers_new[t]==m.a0_F+m.a1_F*pe.cos(m.a2_F*t+m.a3_F)
            m.fer.parametrization_F=pe.Constraint(m.fer.t,rule=_parametrization_F)

            # def _additional_con_F(m):
            #     return m.a0_F>=m.a1_F
            # m.fer.additional_constraint_F=pe.Constraint(rule=_additional_con_F)

            # def _additional_con_C5(m):
            #     return m.a0_C5>=m.a1_C5
            # m.fer.additional_constraint_C5=pe.Constraint(rule=_additional_con_C5)
        else:
            m.fer.P=pe.RangeSet(1,param_val,1)
            # Parametrization coefficients
            m.fer.a0_C5=pe.Var(initialize=628*(1/60)*(1/60),within=pe.Reals)
            m.fer.a1_C5=pe.Var(m.fer.P,initialize=0,within=pe.Reals)
            m.fer.a2_C5=pe.Var(m.fer.P,initialize=0,within=pe.Reals)
            m.fer.a3_C5=pe.Var(m.fer.P,initialize=0,within=pe.Reals)

            # m.fer.a_inoc_F=pe.Var(initialize=2487*(1/60)*(1/60),within=pe.NonNegativeReals)
            m.fer.a0_F=pe.Var(initialize=2487*(1/60)*(1/60),within=pe.Reals)
            m.fer.a1_F=pe.Var(m.fer.P,initialize=0,within=pe.Reals)
            m.fer.a2_F=pe.Var(m.fer.P,initialize=0,within=pe.Reals)
            m.fer.a3_F=pe.Var(m.fer.P,initialize=0,within=pe.Reals)

            # Relationship between flow and coefficients
            def _parametrization_C5(m,t):
                if fed_batch_and_batch_phase:
                    if t*m.final_time<=10*60*60: # Inoculum phase  
                        return m.F_C5liquid_new[t]==0
                    elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
                        return m.F_C5liquid_new[t]==m.a0_C5+sum(m.a1_C5[p]*pe.cos(m.a2_C5[p]*t+m.a3_C5[p]) for p in m.P)
                    elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
                        return m.F_C5liquid_new[t]==0
                else:
                    if t*m.final_time<=10*60*60: # Inoculum phase  
                        return m.F_C5liquid_new[t]==0
                    else:
                        return m.F_C5liquid_new[t]==m.a0_C5+sum(m.a1_C5[p]*pe.cos(m.a2_C5[p]*t+m.a3_C5[p]) for p in m.P)
            m.fer.parametrization_C5=pe.Constraint(m.fer.t,rule=_parametrization_C5)

            def _parametrization_F(m,t):
                if fed_batch_and_batch_phase:
                    if t*m.final_time<=10*60*60: # Inoculum phase  
                        return m.F_liquified_fibers_new[t]== m.a0_F+sum(m.a1_F[p]*pe.cos(m.a2_F[p]*t+m.a3_F[p]) for p in m.P)
                    elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
                        return  m.F_liquified_fibers_new[t]==m.a0_F+sum(m.a1_F[p]*pe.cos(m.a2_F[p]*t+m.a3_F[p]) for p in m.P)
                    elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
                        return m.F_liquified_fibers_new[t]==0
                else:
                    return  m.F_liquified_fibers_new[t]==m.a0_F+sum(m.a1_F[p]*pe.cos(m.a2_F[p]*t+m.a3_F[p]) for p in m.P)
            m.fer.parametrization_F=pe.Constraint(m.fer.t,rule=_parametrization_F)

            # def _additional_con_F(m):
            #     return m.a0_F>=sum(m.a1_F[p] for p in m.P)
            # m.fer.additional_constraint_F=pe.Constraint(rule=_additional_con_F)

            # def _additional_con_C5(m):
            #     return m.a0_C5>=sum(m.a1_C5[p] for p in m.P)
            # m.fer.additional_constraint_C5=pe.Constraint(rule=_additional_con_C5)
    # Objective function
    def _obj_rule(m):
        # return -m.C[m.t.last(),'Eth']*m.M[m.t.last()]
        return -m.C[m.t.last(),'Eth']
    m.fer.obj=pe.Objective(rule=_obj_rule)
    return m

#SIMPLIFIED VERSION, GIVEN THAT THE PROBLEM IN fermentation_optimal_control BECAME TOO COMPLEX
def fermentation_optimal_control_2()-> pe.ConcreteModel():
    
    m=pe.ConcreteModel(name='Fermentation_control_simplified')


    m_fer=build_fermentation(discretization='differences',n_f_elements_t=50)
    m.fer=initialize_model(m_fer,from_feasible=True,feasible_model='validation_fermentation')

    m.fer.del_component(m.fer.obj)
    m.fer.del_component(m.fer.F_base)
    m.fer.F_base_constant=pe.Var(initialize=0.01,within=pe.NonNegativeReals,bounds=(0,1),doc='Constant base flow during fed-batch phase')

    # Remove constant feed parameters
    # val1=pe.value(m.fer.F_C5liquid)
    # val2=pe.value(m.fer.F_liquified_fibers)
    m.fer.del_component(m.fer.F_C5liquid)
    m.fer.del_component(m.fer.F_liquified_fibers)

    # Creating new variables for time
    def _F_C5liquid_new(m):
        return 628*(1/60)*(1/60)
    m.fer.F_C5liquid_new=pe.Var(initialize=_F_C5liquid_new,within=pe.NonNegativeReals,bounds=(0,628*(1/60)*(1/60)*2),doc='C5liquid flow [kg/s]')

    # m.fer.F_C5liquid_new.fix(val1)


    def _F_liquified_fibers_new(m):
        return 2487*(1/60)*(1/60)
    m.fer.F_liquified_fibers_new=pe.Var(initialize=_F_liquified_fibers_new,within=pe.NonNegativeReals,bounds=(0,2487*(1/60)*(1/60)*2),doc='C5liquid flow [kg/s]')
    
    # m.fer.F_liquified_fibers_new.fix(val2)

    # Updating input flow constraints
    m.fer.del_component(m.fer.Feed_constraint)
    m.fer.del_component(m.fer.Feed_concentration_constraint)

    def _Feed_constraint_new(m,t):

        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Fin[t]==m.F_liquified_fibers_new
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Fin[t]==m.F_C5liquid_new + m.F_liquified_fibers_new+m.F_base_constant+m.F_acid[t]       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Fin[t]==0

    m.fer.Feed_constraint_new=pe.Constraint(m.fer.t,rule=_Feed_constraint_new)

    def _Feed_concentration_constraint_new(m,t,j):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Cin[t,j]==m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Cin[t,j]*(m.F_C5liquid_new + m.F_liquified_fibers_new+m.F_base_constant+m.F_acid[t])==(m.F_C5liquid_new*m.C_C5liquid[j]+m.F_liquified_fibers_new*m.C_liquified_fibers[j]+m.F_base_constant*m.C_base[j]+m.F_acid[t]*m.C_acid[j])
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Cin[t,j]== 0  
    m.fer.Feed_concentration_constraint_new=pe.Constraint(m.fer.t,m.fer.j,rule=_Feed_concentration_constraint_new)

    # Objective function
    def _obj_rule(m):
        # return -m.C[m.t.last(),'Eth']*m.M[m.t.last()]
        return -m.C[m.t.last(),'Eth']
    m.fer.obj=pe.Objective(rule=_obj_rule)
    return m

def dsda_model(x_up: list=[100,100,100])-> pe.ConcreteModel():

    #-----Model
    m=pe.ConcreteModel(name='gdp_dsda_model')
   #-------Ordered sets 
    m.set1=pe.RangeSet(1,x_up[0],doc= "set of first group of Boolean variables") #C5
    m.set2=pe.RangeSet(1,x_up[1],doc= "set of second group of Boolean variables") #FIBERS
    m.set3=pe.RangeSet(1,x_up[2],doc= "set of third group of Boolean variables") #BASE

    #-----Variables
    m.Y1=pe.BooleanVar(m.set1,doc="Boolean variable associated to set 1")
    m.Y2=pe.BooleanVar(m.set2,doc="Boolean variable associated to set 2")
    m.Y3=pe.BooleanVar(m.set3,doc="Boolean variable associated to set 3")
    #-----Logical constraints

    #Constraint that allow to apply the reformulation over Y1
    def select_one_Y1(m):
        return pe.exactly(1,m.Y1)
    m.oneY1=pe.LogicalConstraint(rule=select_one_Y1)

    #Constraint that allow to apply the reformulation over Y2
    def select_one_Y2(m):
        return pe.exactly(1,m.Y2)
    m.oneY2=pe.LogicalConstraint(rule=select_one_Y2)

    #Constraint that allow to apply the reformulation over Y3
    def select_one_Y3(m):
        return pe.exactly(1,m.Y3)
    m.oneY3=pe.LogicalConstraint(rule=select_one_Y3)

    #Fermentation model
    m_fer=build_fermentation(discretization='differences',n_f_elements_t=50)
    m.fer=initialize_model(m_fer,from_feasible=True,feasible_model='validation_fermentation')

    m.fer.del_component(m.fer.obj)
    m.fer.del_component(m.fer.F_base)
    m.fer.F_base_constant=pe.Var(initialize=0.0001,within=pe.NonNegativeReals,bounds=(0,0.01),doc='Constant base flow during fed-batch phase')

    # Remove constant feed parameters
    # val1=pe.value(m.fer.F_C5liquid)
    # val2=pe.value(m.fer.F_liquified_fibers)
    m.fer.del_component(m.fer.F_C5liquid)
    m.fer.del_component(m.fer.F_liquified_fibers)

    # Creating new variables for time
    def _F_C5liquid_new(m):
        return 628*(1/60)*(1/60)
    m.fer.F_C5liquid_new=pe.Var(initialize=_F_C5liquid_new,within=pe.NonNegativeReals,bounds=(0,628*(1/60)*(1/60)*2),doc='C5liquid flow [kg/s]')

    # m.fer.F_C5liquid_new.fix(val1)


    def _F_liquified_fibers_new(m):
        return 2487*(1/60)*(1/60)
    m.fer.F_liquified_fibers_new=pe.Var(initialize=_F_liquified_fibers_new,within=pe.NonNegativeReals,bounds=(0,2487*(1/60)*(1/60)*2),doc='C5liquid flow [kg/s]')
    
    # m.fer.F_liquified_fibers_new.fix(val2)

    # Updating input flow constraints
    m.fer.del_component(m.fer.Feed_constraint)
    m.fer.del_component(m.fer.Feed_concentration_constraint)

    def _Feed_constraint_new(m,t):

        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Fin[t]==m.F_liquified_fibers_new
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Fin[t]==m.F_C5liquid_new + m.F_liquified_fibers_new+m.F_base_constant+m.F_acid[t]       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Fin[t]==0

    m.fer.Feed_constraint_new=pe.Constraint(m.fer.t,rule=_Feed_constraint_new)

    def _Feed_concentration_constraint_new(m,t,j):
        if t*m.final_time<=10*60*60: # Inoculum phase
            return m.Cin[t,j]==m.C_liquified_fibers[j]
        elif t*m.final_time> 10*60*60 and t*m.final_time <=70*60*60: #Fed-batch phase
            return m.Cin[t,j]*(m.F_C5liquid_new + m.F_liquified_fibers_new+m.F_base_constant+m.F_acid[t])==(m.F_C5liquid_new*m.C_C5liquid[j]+m.F_liquified_fibers_new*m.C_liquified_fibers[j]+m.F_base_constant*m.C_base[j]+m.F_acid[t]*m.C_acid[j])
        elif t*m.final_time>70*60*60 and t*m.final_time<=190*60*60: #Batch phase
            return m.Cin[t,j]== 0  
    m.fer.Feed_concentration_constraint_new=pe.Constraint(m.fer.t,m.fer.j,rule=_Feed_concentration_constraint_new)


    
        
    #OBJECTIVE FUNCTION EQUAL TO CONSTANT TO INDICATE ITS A SIMULATION

    def _obj_rule2(m):
        return 1
    m.fer.obj2=pe.Objective(rule=_obj_rule2)



    #SCALE MODEL TRANSFORMATION
    m=scale_model(m)

    #DUMMY OBJECTIVE
    m.obj=pe.Var(initialize=0) #Store objective function


    def _obj_cons_rule(m):
        return  m.model().obj==-m.scaled_C[m.t.last(),'Eth']
    m.fer.obj_cons=pe.Constraint(rule=_obj_cons_rule)
    return m


def scale_model(m):

    m.scaling_factor = pe.Suffix(direction=pe.Suffix.EXPORT)
    # m.scaling_factor[m.fer.obj2] = 1e-6 # scale the objective
    m.scaling_factor[m.fer.F_base_constant]=1e+3
    m.scaling_factor[m.fer.Fin]=1e+1
    m.scaling_factor[m.fer.Cin]=1e-2
    m.scaling_factor[m.fer.Ce]=1
    m.scaling_factor[m.fer.Cef]=1
    m.scaling_factor[m.fer.Ceb]=1e+1
    m.scaling_factor[m.fer.CebC]=1e+3
    m.scaling_factor[m.fer.r1]=1e+7
    m.scaling_factor[m.fer.r2]=1e+6
    m.scaling_factor[m.fer.r3]=1e+6
    m.scaling_factor[m.fer.r4]=1e+7
    m.scaling_factor[m.fer.r5]=1e+7
    m.scaling_factor[m.fer.C]=1
    m.scaling_factor[m.fer.M]=1e-5
    m.scaling_factor[m.fer.R]=1e+5
    m.scaling_factor[m.fer.q]=1e+5
    m.scaling_factor[m.fer.avance]=1
    m.scaling_factor[m.fer.C_elect_init]=1e+1
    m.scaling_factor[m.fer.C_elect_equil]=1e+1
    m.scaling_factor[m.fer.dCdt]=1e-3
    m.scaling_factor[m.fer.dMdt]=1e-5

    m = pe.TransformationFactory('core.scale_model').create_using(m)

    return m


def global_model(initialize_vars_at_steady_state: bool=True)-> pe.ConcreteModel():
    #-------------------------------------------------------------------------
    #-------------pyomo model-------------------------------------------------
    #-------------------------------------------------------------------------
    m = pe.ConcreteModel(name='global_model')

    #-------------------------------------------------------------------------
    #-------------pretreatment part of the model------------------------------
    #-------------------------------------------------------------------------

    m_pre=build_pretreatment()
    m.pre=initialize_model(m_pre,from_feasible=True,feasible_model='validation_pretreatment')

    #------------creating new initial condition (at t=0), which will guarantee that the reactor is initially at steady state
    
    # Creating new parameterS with steady state concentration profiles 
    def _C_SS_init_pre(m_pre,k_pre,j_pre):
        return pe.value(m_pre.c[m.pre.t.last(),k_pre,j_pre])
    m.pre.C_SS=pe.Param(m.pre.k,m.pre.j,initialize=_C_SS_init_pre,doc='Steady state concentration profiles')

    def _h_SS_init_pre(m_pre,k_pre):
        return pe.value(m_pre.h[m_pre.t.last(),k_pre])
    m.pre.h_SS=pe.Param(m.pre.k,initialize=_h_SS_init_pre,doc='Steady state enthalpy profile')

    # Deleting initial condition constraint from the original model
    m.pre.del_component(m.pre.IC1)
    m.pre.del_component(m.pre.IC2)

    # Creating new initial condition constraints
    def _IC1(m_pre, k_pre, j_pre):
        if k_pre == 0:
            return pe.Constraint.Skip
        return m_pre.c[0, k_pre, j_pre] == m_pre.C_SS[k_pre,j_pre]
    m.pre.IC1_new = pe.Constraint(m.pre.k, m.pre.j, rule=_IC1)

    def _IC2(m_pre, k_pre):
        if k_pre == 0:
            return pe.Constraint.Skip
        return m_pre.h[0, k_pre] == m_pre.h_SS[k_pre]
    m.pre.IC2_new = pe.Constraint(m.pre.k, rule=_IC2)

    if initialize_vars_at_steady_state:
        # Reinitializing variable values at steady state
        time_index=m.pre.t # Time index. NOTE: depends on the case study
        for v in m.pre.component_objects(ctype=pe.Var):
            # Check if variable has time index. If it does, initialize this variable with its steady state value
            position=[v.index_set()._sets[j].name==time_index.name for j in range(len(v.index_set()._sets))] # returns tru for the position of the index that corresponds to time
            if any(position):
                # itentify location of time index
                cuenta=0
                for i in position:
                    if i==True:
                        loc=cuenta #location of time index
                        break
                    cuenta=cuenta+1
                # Assign steady state value for time-indexed variables, except final time
                for index in v.index_set().data():
                    if index[loc]==time_index.last():
                        continue
                    partial_index_lst=list(index)
                    partial_index_lst[loc]=time_index.last()
                    partial_index=tuple(partial_index_lst)
                    if v[partial_index].value!=None:
                        v[index].value=pe.value(v[partial_index])


    #-------------------------------------------------------------------------
    #-------------hydrolisis part of the model--------------------------------
    #-------------------------------------------------------------------------

    m_hyd=build_hydrolisis(time=72000,discretization='collocation',n_f_elements_x=6,n_f_elements_t=5)
    m.hyd=initialize_model(m_hyd,from_feasible=True,feasible_model='validation_hydrolisis')
    m.hyd.del_component(m.hyd.obj)


    #------------creating new initial condition (at t=0), which will guarantee that the reactor is initially at steady state
    # Creating new parameterS with steady state concentration profiles 
    def _C_SS_init_hyd(m_hyd,x_hyd,j_hyd):
        return pe.value(m_hyd.C[m.hyd.t.last(),x_hyd,j_hyd])
    m.hyd.C_SS=pe.Param(m.hyd.x,m.hyd.j,initialize=_C_SS_init_hyd,doc='Steady state concentration profiles')

    #Updating partial differential equation in the original model with steady state initial condition
    
    for t in m.hyd.t:
        for x in m.hyd.x:
            for j in m.hyd.j:
                if any(j == jp for jp in ['C','G', 'X', 'F', 'E','AC']): 
                    if t==m.hyd.t.first() and x>m.hyd.x.first() and x<m.hyd.x.last(): 
                        m.hyd.partialDiff[t,x,j].set_value(m.hyd.partialDiff[t,x,j].body==m.hyd.C_SS[x,j])
                else:
                    if t==m.hyd.t.first() and x>m.hyd.x.first(): 
                        m.hyd.partialDiff[t,x,j].set_value(m.hyd.partialDiff[t,x,j].body==m.hyd.C_SS[x,j])

    if initialize_vars_at_steady_state:
        # Reinitializing variable values at steady state
        time_index=m.hyd.t # Time index. NOTE: depends on the case study
        for v in m.hyd.component_objects(ctype=pe.Var):
            # Check if variable has time index. If it does, initialize this variable with its steady state value
            position=[v.index_set()._sets[j].name==time_index.name for j in range(len(v.index_set()._sets))] # returns tru for the position of the index that corresponds to time
            if any(position):
                # itentify location of time index
                cuenta=0
                for i in position:
                    if i==True:
                        loc=cuenta #location of time index
                        break
                    cuenta=cuenta+1
                # Assign steady state value for time-indexed variables, except final time
                for index in v.index_set().data():
                    if index[loc]==time_index.last():
                        continue
                    partial_index_lst=list(index)
                    partial_index_lst[loc]=time_index.last()
                    partial_index=tuple(partial_index_lst)
                    if v[partial_index].value!=None:
                        v[index].value=pe.value(v[partial_index])

    #-----------CSTR model to complete desired retention time
    m_cstr=build_hydrolisis_cstr(m) 
    m.cstr=initialize_model(m_cstr,from_feasible=True,feasible_model='validation_cstr')

    # Equations that conect cstr with hydrolisis model
    def _eta_hyd_cstr_definition(m,t):
        return m.cstr.eta[t]==m.hyd.eta[m.hyd.x.last(),t]
    m.eta_hyd_cstr=pe.Constraint(m.cstr.t,rule=_eta_hyd_cstr_definition)

    def  _concentration_hyd_cstr(m,t,j):
        return m.cstr.Cfeed[t,j]==m.hyd.C[t,m.hyd.x.last(),j]   
    m.concentration_hyd_cstr=pe.Constraint(m.cstr.t,m.cstr.j,rule=_concentration_hyd_cstr)
    #-------------------------------------------------------------------------
    #-------------fermentation part of the model------------------------------
    #-------------------------------------------------------------------------

    m_fer=build_fermentation(discretization='differences',n_f_elements_t=50)
    m.fer=initialize_model(m_fer,from_feasible=True,feasible_model='validation_fermentation')
    for t in m.fer.t:
        m.fer.F_base[t].fix(m.fer.F_base[t].value) 
    m.fer.del_component(m.fer.obj)

    return m



if __name__ == '__main__':
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    ###-----------------------------------------------------------------
    ###------------------------VALIDATION-------------------------------
    ###-----------------------------------------------------------------
    v1='PRETREATMENT--'
    v2='HYDROLISIS--'
    v3='FERMENTATION--'
    v4='GLOBAL--'
    v5='1_FERMENTATION_CONTROL--' # FIRST TESTS TO IDENTIFY FEASIBLE SOLUTION WITHOUT PARAMETRIZATION
    v6='2_FERMENTATION_CONTROL--' # TRYING TO INCLUDE FLOW PARAMETRIZATION
    v7='FERMENTATION_DSDA_TEST'
    solver='conopt'

    ### PRETREATMENT SIMULATION
    if v1=='PRETREATMENT':
        m = build_pretreatment()
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)
        solved=generate_initialization(m=m,model_name='validation_pretreatment')
        

        s0 = []
        T1 = []
        t = []
        TT = []
        s1 = []
        s2 = []
        s3 = []
        s4 = []
        s5 = []
        s6 = []
        s7 = []
        s8 = []
        s9 = []

        for i in m.k:
            t.append(i*1.2)

            s0.append(m.c[3600, i, 'CS'].value)
            s1.append(m.c[3600, i, 'XS'].value)
            s2.append(m.c[3600, i, 'AS'].value)
            s3.append(m.c[3600, i, 'G'].value)
            s4.append(m.c[3600, i, 'XO'].value)
            s5.append(m.c[3600, i, 'X'].value)
            s6.append(m.c[3600, i, 'A'].value)
            s7.append(m.c[3600, i, 'AC'].value)
            s8.append(m.c[3600, i, 'F'].value)
            T1.append(pe.value(m.T[3600, i]))
        #
        for i in T1:
            a = i-273
            TT.append(a)

        original = pd.read_csv('biorefinery_models/Cellulose.csv', header=None)
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--g')
        plt.plot(t, s0, 'g', label='Cellulose ')
        original = pd.read_csv('biorefinery_models/Xylan.csv', header=None)
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--m')
        plt.plot(t, s1, 'm', label='Xylan')
        original = pd.read_csv('biorefinery_models/Arabinan.csv', header=None)
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--r')
        plt.plot(t, s2, 'r', label='Arabinan')

        plt.xlabel('length,m')
        plt.ylabel('Concentration, g/kg')
        plt.legend()
        plt.show()

        original = pd.read_csv('biorefinery_models/Glucose.csv', header=None)
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--g')
        plt.plot(t, s3, 'g', label='Glucose')
        original = pd.read_csv('biorefinery_models/Xylose.csv', header=None)
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--m')
        plt.plot(t, s5, 'm', label='Xylose')
        original = pd.read_csv('biorefinery_models/Arabinose.csv', header=None)
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--r')
        plt.plot(t, s6, 'r', label='Arabinose')

        plt.xlabel('length,m')
        plt.ylabel('Concentration, g/kg')
        plt.legend()
        plt.show()

        original = pd.read_csv('biorefinery_models/Temperature.csv', header=None)
        plt.plot(t, TT, 'k')
        plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--k')
        plt.xlabel('length,m')
        plt.ylabel('Temperature, °C')
        plt.show()
    ### HYDROLISIS SIMULATION
    if v2=='HYDROLISIS':
        sim_time=72000 #seconds
        discretization_type='collocation'
        finite_elem_x=6 
        finite_elem_t=5

        # PFR
        m=build_hydrolisis(time=sim_time,discretization=discretization_type,n_f_elements_x=finite_elem_x,n_f_elements_t=finite_elem_t)
        m=initialize_model(m,from_feasible=True,feasible_model='validation_hydrolisis_init')
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)
        generate_initialization(m=m,model_name='validation_hydrolisis')


        # CSTR
        m_global=pe.ConcreteModel()
        m_global.hyd=m
        m_cstr=build_hydrolisis_cstr(m_global)

        # Equations that conect cstr with hydrolisis model
        def _eta_hyd_cstr_definition(m_cstr,t):
            return m_cstr.eta[t]==pe.value(m.eta[m.x.last(),t])
        m_cstr.eta_hyd_cstr=pe.Constraint(m_cstr.t,rule=_eta_hyd_cstr_definition)

        def  _concentration_hyd_cstr(m_cstr,t,j):
            return m_cstr.Cfeed[t,j]==pe.value(m.C[t,m.x.last(),j])   
        m_cstr.concentration_hyd_cstr=pe.Constraint(m_cstr.t,m_cstr.j,rule=_concentration_hyd_cstr)
        m_cstr.obj=pe.Objective(expr=1)

        opt1 = SolverFactory('gams')
        results = opt1.solve(m_cstr, solver=solver, tee=True)
        generate_initialization(m=m_cstr,model_name='validation_cstr')

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
    ### FERMENTATION ADJUSTMENT AND SIMULATION
    if v3=='FERMENTATION':
        discretization_type_fer='differences'
        finite_elem_t_fer=50

        m=build_fermentation_old(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
        m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_init')
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)
        generate_initialization(m=m,model_name='validation_fermentation_init2')

        # DATA TO ADJUST PARAMETERS
        colors=['b','g','m','r','k','y','c']
        adjust_data={}
        contador=-1
        for j in m.j:
            if j=='G' or j=='X' or j=='Eth' or j=='Cell' or  j=='CS' or j=='XS' or j=='E':
                contador=contador+1
                original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
                adjust_data[j]=[]
                for t in m.t:
                    adjusted_t=t*m.final_time*(1/60)*(1/60)
                    adjust_data[j].append(max([np.interp(adjusted_t,original.iloc[:, 0],original.iloc[:, 1]),0]))
                # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
                # plt.plot(time,adjust_data[j],colors[contador],label=j)
        # plt.xlabel('time [h]')
        # plt.ylabel('Concentration [g/kg]')
        # plt.legend()
        # plt.show()

        mad=build_fermentation_pH_control_adjustment2(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer,data=adjust_data)
        mad=initialize_model(mad,from_feasible=True,feasible_model='validation_fermentation_init2')        
        opt1 = SolverFactory('gams')
        results = opt1.solve(mad, solver=solver, tee=True)
        generate_initialization(m=mad,model_name='validation_fermentation_readjusted2')

        mad.K0G.pprint()
        mad.K1G.pprint()
        mad.K2G.pprint()
        mad.K0X.pprint()
        mad.K1X.pprint()
        mad.K2X.pprint()

        mad=build_fermentation(discretization=discretization_type_fer,n_f_elements_t=finite_elem_t_fer)
        mad=initialize_model(mad,from_feasible=True,feasible_model='validation_fermentation_readjusted2')    
        for t in mad.t:
            mad.F_base[t].fix(mad.F_base[t].value)   
        opt1 = SolverFactory('gams')
        results = opt1.solve(mad, solver=solver, tee=True)
        generate_initialization(m=mad,model_name='validation_fermentation')

        time=[]
        vec={}
        pH=[]
        Hold_up=[]
        Flow_base=[]
        Flow_acid=[]

        for t in mad.t:
            time.append(t*mad.final_time*(1/60)*(1/60))
            pH.append(mad.pH[t].value)
            Hold_up.append(mad.M[t].value)
            Flow_base.append(mad.F_base[t].value)
            Flow_acid.append(mad.F_acid[t].value)

        for j in mad.j:
            vec[j]=[]
            for t in mad.t:
                vec[j].append(mad.C[t,j].value)

        colors=['b','g','m','r','k','y','c']
        contador=-1
        for j in mad.j:
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
        for j in mad.j:
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

        plt.plot(time,Hold_up)
        plt.xlabel('time [-]')
        plt.ylabel('Hold-up [kg]')
        plt.show()

        plt.plot(time,Flow_acid)
        plt.xlabel('time [-]')
        plt.ylabel('Acid flow')
        plt.show()

        plt.plot(time,Flow_base)
        plt.xlabel('time [-]')
        plt.ylabel('Base flow')
        plt.show()
    ### GLOBAL MODEL
    if v4=='GLOBAL':
        
        # 1: Feed to pretreatment is soaked until approx. 40% dry matter before entering the thermal reactor, resulting in a total outlet flow of 2316 kg/h
        m=global_model()
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)
    ### FERMENTATION OPTIMIZATION TESTS
    if v5=='1_FERMENTATION_CONTROL':
    
        m=fermentation_optimal_control(include_parametrization=False,simple_parametrization=False,fix_base_flow=True,include_flow_integrals=True,fed_batch_and_batch_phase=False)
        # Note that "fermentation_optimal_control" has an initialization from the simulation inside, hence I do not need to initialize here!       
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)  
        generate_initialization(m=m,model_name='fermentation_control2')   


        m=fermentation_optimal_control(include_parametrization=False,simple_parametrization=False,fix_base_flow=False,include_flow_integrals=True,fed_batch_and_batch_phase=False)
        m=initialize_model(m,from_feasible=True,feasible_model='fermentation_control2')         
        for t in m.fer.t:
            m.fer.F_base[t].fix(0) 
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)  
        generate_initialization(m=m,model_name='fermentation_control3')   



        m=fermentation_optimal_control(include_parametrization=False,simple_parametrization=False,fix_base_flow=False,include_flow_integrals=True,fed_batch_and_batch_phase=False)
        m=initialize_model(m,from_feasible=True,feasible_model='fermentation_control3') 
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)  
        generate_initialization(m=m,model_name='fermentation_control4') 

        m=fermentation_optimal_control(include_parametrization=False,simple_parametrization=False,fix_base_flow=False,include_flow_integrals=True,fed_batch_and_batch_phase=False)
        m=initialize_model(m,from_feasible=True,feasible_model='fermentation_control3')

        # # # for j in range(1,100,1):
        # # #     start=tim.time()
        # # #     print('current test',j)
        # # #     m=fermentation_optimal_control(include_parametrization=True,simple_parametrization=False,fix_base_flow=True,include_flow_integrals=True,fed_batch_and_batch_phase=False, param_val=j)
        # # #     # m=initialize_model(m,from_feasible=True,feasible_model='')         
        # # #     opt1 = SolverFactory('gams')
        # # #     m.results = opt1.solve(m, solver='conopt4', tee=False)
        # # #     end=tim.time()
        # # #     print('CPU TIME=',end-start)
        # # #     if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
        # # #         continue
        # # #     else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
        # # #         print('Found_feas_sol_for j=',j)  
        # # #         generate_initialization(m=m,model_name='fermentation_control_j'+str(j))  

 


        mad=m.fer

        # mad.a0_C5.pprint()
        # mad.a1_C5.pprint()
        # mad.a2_C5.pprint()
        # mad.a3_C5.pprint()

        # mad.a0_F.pprint()
        # mad.a1_F.pprint()
        # mad.a2_F.pprint()
        # mad.a3_F.pprint()


        time=[]
        vec={}
        pH=[]
        Hold_up=[]
        Flow_base=[]
        Flow_acid=[]
        Flow_C5=[]
        Flow_F=[]

        for t in mad.t:
            time.append(t*mad.final_time*(1/60)*(1/60))
            pH.append(mad.pH[t].value)
            Hold_up.append(mad.M[t].value)
            Flow_base.append(mad.F_base[t].value)
            Flow_acid.append(mad.F_acid[t].value)
            Flow_C5.append(mad.F_C5liquid_new[t].value)
            Flow_F.append(mad.F_liquified_fibers_new[t].value)

        for j in mad.j:
            vec[j]=[]
            for t in mad.t:
                vec[j].append(mad.C[t,j].value)

        colors=['b','g','m','r','k','y','c']
        contador=-1
        for j in mad.j:
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
        for j in mad.j:
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

        plt.plot(time,Hold_up)
        plt.xlabel('time [-]')
        plt.ylabel('Hold-up [kg]')
        plt.show()

        plt.plot(time,Flow_acid)
        plt.xlabel('time [-]')
        plt.ylabel('Acid flow')
        plt.show()

        plt.plot(time,Flow_base)
        plt.xlabel('time [-]')
        plt.ylabel('Base flow')
        plt.show()

        plt.plot(time,Flow_C5)
        plt.xlabel('time [-]')
        plt.ylabel('C5 flow')
        plt.show()

        plt.plot(time,Flow_F)
        plt.xlabel('time [-]')
        plt.ylabel('Liquified fibers flow')
        plt.show()


    if v6=='2_FERMENTATION_CONTROL':
        m=fermentation_optimal_control_2()



        m.fer.obj.deactivate()

        def _obj_rule2(m):
            return 1#-m.C[m.t.last(),'Eth']#-m.M[m.t.last()]   #-m.C[m.t.last(),'Eth']
        m.fer.obj2=pe.Objective(rule=_obj_rule2)


        m.scaling_factor = pe.Suffix(direction=pe.Suffix.EXPORT)
        # m.scaling_factor[m.fer.obj2] = 1e-6 # scale the objective
        m.scaling_factor[m.fer.F_base_constant]=1e+3
        m.scaling_factor[m.fer.Fin]=1e+1
        m.scaling_factor[m.fer.Cin]=1e-2
        m.scaling_factor[m.fer.Ce]=1
        m.scaling_factor[m.fer.Cef]=1
        m.scaling_factor[m.fer.Ceb]=1e+1
        m.scaling_factor[m.fer.CebC]=1e+3
        m.scaling_factor[m.fer.r1]=1e+7
        m.scaling_factor[m.fer.r2]=1e+6
        m.scaling_factor[m.fer.r3]=1e+6
        m.scaling_factor[m.fer.r4]=1e+7
        m.scaling_factor[m.fer.r5]=1e+7
        m.scaling_factor[m.fer.C]=1
        m.scaling_factor[m.fer.M]=1e-5
        m.scaling_factor[m.fer.R]=1e+5
        m.scaling_factor[m.fer.q]=1e+5
        m.scaling_factor[m.fer.avance]=1
        m.scaling_factor[m.fer.C_elect_init]=1e+1
        m.scaling_factor[m.fer.C_elect_equil]=1e+1
        m.scaling_factor[m.fer.dCdt]=1e-3
        m.scaling_factor[m.fer.dMdt]=1e-5

        m = pe.TransformationFactory('core.scale_model').create_using(m)

        # m.fer.F_liquified_fibers_new.pprint()

        # var_val_comparison={}
        # for v in m.component_data_objects(ctype=pe.Var):
        #     try:
        #         var_val_comparison[v.name]=[pe.value(v),0]
        #     except:
        #         var_val_comparison[v.name]=[0,0]




        # def _CEth_concentration(m):
        #     return m.C[m.t.last(),'Eth']<=60
        # m.fer.CEth_concentration=pe.Constraint(rule=_CEth_concentration)


        # m=initialize_model(m,from_feasible=True,feasible_model='fermentation_control4') 
        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True)  


        # m = pe.TransformationFactory('core.scale_model').propagate_solution(scaled_model,m)

        # generate_initialization(m=m,model_name='fermentation_control_new') 
        
        m.fer.scaled_obj2.deactivate()

        def _obj_rule3(m):
            return 1#-m.scaled_M[m.t.last()]#-m.C[m.t.last(),'Eth']#-m.C[m.t.last(),'Eth']#-m.M[m.t.last()]   #-m.C[m.t.last(),'Eth']
        m.fer.obj3=pe.Objective(rule=_obj_rule3)

        m.fer.scaled_F_C5liquid_new.fix(pe.value(m.fer.scaled_F_C5liquid_new))   
        m.fer.scaled_F_liquified_fibers_new.fix(pe.value(m.fer.scaled_F_liquified_fibers_new))
        m.fer.scaled_F_base_constant.fix(pe.value(m.fer.scaled_F_base_constant))


        m.fer.scaled_F_C5liquid_new.pprint()
        m.fer.scaled_F_liquified_fibers_new.pprint()
        m.fer.scaled_F_base_constant.pprint()

        opt1 = SolverFactory('gams')
        results = opt1.solve(m, solver=solver, tee=True) 


        # generate_initialization(m=m,model_name='fermentation_control_new2') 
        # for v in m.component_data_objects(ctype=pe.Var):
        #     try:
        #         var_val_comparison[v.name][1]=pe.value(v)
        #     except:
        #         var_val_comparison[v.name][1]=0


        # magnitude_dif={}
        # for v in m.component_data_objects(ctype=pe.Var):
        #     magnitude_dif[v.name]=abs(var_val_comparison[v.name][1]-var_val_comparison[v.name][0])
        # magnitude_dif_sorted=sorted(magnitude_dif.items(), key=lambda x: x[1])
        # print(magnitude_dif_sorted)



        mad=m.fer



        time=[]
        vec={}
        pH=[]
        Hold_up=[]
        Flow_base=[]
        Flow_acid=[]
        Flow_C5=[]
        Flow_F=[]

        for t in mad.t:
            time.append(t*mad.final_time*(1/60)*(1/60))
            pH.append(mad.scaled_pH[t].value)
            Hold_up.append(mad.scaled_M[t].value)
            Flow_base.append(mad.scaled_F_base_constant.value)
            Flow_acid.append(mad.F_acid[t].value)
            Flow_C5.append(mad.scaled_F_C5liquid_new.value)
            Flow_F.append(mad.scaled_F_liquified_fibers_new.value)

        for j in mad.j:
            vec[j]=[]
            for t in mad.t:
                vec[j].append(mad.scaled_C[t,j].value)

        colors=['b','g','m','r','k','y','c']
        contador=-1
        for j in mad.j:
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
        for j in mad.j:
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

        plt.plot(time,Hold_up)
        plt.xlabel('time [-]')
        plt.ylabel('Hold-up [kg]')
        plt.show()

        plt.plot(time,Flow_acid)
        plt.xlabel('time [-]')
        plt.ylabel('Acid flow')
        plt.show()

        plt.plot(time,Flow_base)
        plt.xlabel('time [-]')
        plt.ylabel('Base flow')
        plt.show()

        plt.plot(time,Flow_C5)
        plt.xlabel('time [-]')
        plt.ylabel('C5 flow')
        plt.show()

        plt.plot(time,Flow_F)
        plt.xlabel('time [-]')
        plt.ylabel('Liquified fibers flow')
        plt.show()

    if v7=='FERMENTATION_DSDA_TEST':

        # RETRIEVE ORDERED VARIABLES
        x_up=[100,100,100]   #Upper bound for discrete search range. Must be greater or equal to 1
        mdsda=dsda_model(x_up=x_up)
        ext_ref={mdsda.Y1:mdsda.set1,mdsda.Y2:mdsda.set2,mdsda.Y3:mdsda.set3}
        [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(mdsda,ext_ref,tee=True)
        




# scaled_F_C5liquid_new : C5liquid flow [kg/s]
#     Size=1, Index=None
#     Key  : Lower : Value             : Upper              : Fixed : Stale : Domain
#     None :   0.0 : 0.174441527677707 : 0.3488888888888889 :  True : False : NonNegativeReals
# scaled_F_liquified_fibers_new : C5liquid flow [kg/s]
#     Size=1, Index=None
#     Key  : Lower : Value             : Upper              : Fixed : Stale : Domain
#     None :   0.0 : 0.690843130837456 : 1.3816666666666668 :  True : False : NonNegativeReals
# scaled_F_base_constant : Constant base flow during fed-batch phase
#     Size=1, Index=None
#     Key  : Lower : Value            : Upper  : Fixed : Stale : Domain
#     None :   0.0 : 1.41509777957679 : 1000.0 :  True : False : NonNegativeReals



        x_init=[round((0.174441527677707-mdsda.fer.scaled_F_C5liquid_new.ub)*((upper_bounds[1]-lower_bounds[1])/(mdsda.fer.scaled_F_C5liquid_new.ub-mdsda.fer.scaled_F_C5liquid_new.lb))+upper_bounds[1])
           ,round((0.690843130837456 -mdsda.fer.scaled_F_liquified_fibers_new.ub)*((upper_bounds[2]-lower_bounds[2])/(mdsda.fer.scaled_F_liquified_fibers_new.ub-mdsda.fer.scaled_F_liquified_fibers_new.lb))+upper_bounds[2])
           ,round((1.41509777957679 -mdsda.fer.scaled_F_base_constant.ub)*((upper_bounds[3]-lower_bounds[3])/(mdsda.fer.scaled_F_base_constant.ub-mdsda.fer.scaled_F_base_constant.lb))+upper_bounds[3])]
        print(x_init)
        # mdsda.fer.scaled_F_C5liquid_new.pprint()
        # mdsda.fer.scaled_F_liquified_fibers_new.pprint()
        # mdsda.fer.scaled_F_base_constant.pprint()

        # mdsda=external_ref(mdsda,x_init,dummy_logic,reformulation_dict,tee =True)

        # opt1 = SolverFactory('gams')
        # results = opt1.solve(mdsda, solver=solver, tee=True)  

        # DSDA test
        
        start=time.time()
        neighdef='Infinity'
        D_SDAsol,routeDSDA,obj_route=solve_with_dsda(dsda_model,{'x_up':x_up},x_init,ext_ref,dummy_logic,k = neighdef,provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = solver,subproblem_solver_options={},iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = True,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=True)
        end=time.time()
        print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  

