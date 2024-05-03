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
import random




def dummy_logic(m):
    logic_expr = []
    for n in m.set1:
        logic_expr.append([m.Y1[n],m.Y1_disjunct[n].indicator_var])
    for n in m.set2:
        logic_expr.append([m.Y2[n],m.Y2_disjunct[n].indicator_var])
    return logic_expr
def dummy_logic_v2(m):
    logic_expr = []
    for r in m.react_set:
        for n in m.set1[r]:
            logic_expr.append([m.Y1[r][n],m.Y1_disjunct[r][n].indicator_var])
        for n in m.set2[r]:
            logic_expr.append([m.Y2[r][n],m.Y2_disjunct[r][n].indicator_var])
    return logic_expr
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
    # m=scale_model(m)
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
    # print(m.results.solver.termination_condition)
    if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
        m.dsda_status = 'Evaluated_Infeasible'
        m.obj.value=1000000
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

def initialize_variables_at_final_state(m)-> pe.ConcreteModel():

    # Reinitializing variable values at steady state
    time_index=m.t # Time index. NOTE: depends on the case study
    for v in m.component_objects(ctype=pe.Var):
        # Check if variable has time index. If it does, initialize this variable with its final state value
        try: # If variable is defined over multiple sets
            position=[v.index_set()._sets[j].name==time_index.name for j in range(len(v.index_set()._sets))] # returns tru for the position of the index that corresponds to time
        except: # If only defined over a single set
            position=[v.index_set().name==time_index.name]
        if any(position):
            # itentify location of time index
            cuenta=0
            for i in position:
                if i==True:
                    loc=cuenta #location of time index
                    break
                cuenta=cuenta+1
            # Assign final value for time-indexed variables, except final time
            for index in v.index_set().data():
                try:
                    if index[loc]==time_index.last():
                        continue
                    partial_index_lst=list(index)
                    partial_index_lst[loc]=time_index.last()
                    partial_index=tuple(partial_index_lst)
                except:
                    if index==time_index.last():
                        continue
                    partial_index_lst=index
                    partial_index_lst=time_index.last()
                    partial_index=partial_index_lst
                
                if v[partial_index].value!=None:
                    v[index].value=pe.value(v[partial_index])
    return m



# For open loop pessimization
def build_fermentation_one_time_step_optimizing_flows_pH_open_loop_pessimization(total_sim_time: float=190*60*60,discretization: str='collocation',n_f_elements_t: int=1,total_f_elements_t:int=50,current_start_time_sconds: float=0,M0_prev_input: float=0,C0_prev_input: dict={'CS':0, 'XS':0, 'LS':0,'C':0,'G':0, 'X':0, 'F':0, 'E':0,'AC':0,'Cell':0,'Eth':0,'CO2':0,'ACT':0,'HMF':0,'Base':0},keep_constant_flows: bool=False):
    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------

    # n_f_elements_t dictates the discretized prediction horizon (between 1 and total_f_elements)
    # total_f_elements is the fixed total number fo finite elements that defines the sambpling time with respect to total_sim_time
    m.final_time = pe.Param(initialize=(n_f_elements_t*total_sim_time)/total_f_elements_t,doc='Prediction horizon with respect to 0 seconds [s]')  
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

    m.F_C5liquid=pe.Var(m.t,initialize=628*(1/60)*(1/60),within=pe.NonNegativeReals,bounds=(0,2*628*(1/60)*(1/60)),doc='C5liquid flow [kg/s]')
    m.F_liquified_fibers=pe.Var(m.t,initialize=2487*(1/60)*(1/60),within=pe.NonNegativeReals,bounds=(0,2*2487*(1/60)*(1/60)),doc='Liquified fibers flow [kg/s]')

    m.pH=pe.Var(initialize=5.39,bounds=(5.36,5.4),within=pe.NonNegativeReals,doc='pH')     #bounds=(5.36,5.4)
    # m.pH.fix(5.41)


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

    def _C_C5liquid_bounds(m,j):
        if j=='X':
            return (25,35)
        elif j=='G':
            return (5,15)
        else:
            return (_C_C5liquid[j],_C_C5liquid[j])
    m.C_C5liquid=pe.Var(m.j,initialize=_C_C5liquid,within=pe.NonNegativeReals,bounds=_C_C5liquid_bounds,doc='C5liquid concentration [g/kg]')

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

    def _C_liquified_fibers_bouds(m,j):
        if j=='X':
            return (50,70)
        elif j=='G':
            return (90,110)
        else:
            return (_C_liquified_fibers[j],_C_liquified_fibers[j])
    m.C_liquified_fibers=pe.Var(m.j,initialize=_C_liquified_fibers,within=pe.NonNegativeReals,bounds=_C_liquified_fibers_bouds,doc='Liquified fibers concentration [g/kg]')

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
        if t==m.t.first() and m.current_starting_time==0: #Initial condition
            return m.M[t] == m.M0
        elif t==m.t.first(): #Final condition from previous time step
            return m.M[t] == m.M0_prev #TODO: remove, no longer needed. Now, we have to consider the case where initial conditio of the batch changes!
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
  
        if t==m.t.first() and m.current_starting_time==0: #Initial condition
            return m.C[t,j] == m.C0[j]
        elif t==m.t.first(): # Final condition from previous step
            return m.C[t,j] == m.C0_prev[j] #TODO: remove, no longer needed. Now, we have to consider the case where initial conditio of the batch changes!
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

    # def _initF_C5(m,t):
    #     return 628*(1/60)*(1/60)
    # m.F_C5liquid=pe.Param(m.t,initialize=_initF_C5,doc='C5liquid flow [kg/s]')
    # def _initF_Fibers(m,t):
    #     return 2487*(1/60)*(1/60)
    # m.F_liquified_fibers=pe.Param(m.t,initialize=_initF_Fibers,doc='Liquified fibers flow [kg/s]')


    # m.pH=pe.Param(m.t,initialize=5.3969605,doc='pH')

    for t in m.t:
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
            m.F_liquified_fibers[t].value=2487*(1/60)*(1/60)
            m.F_C5liquid[t].value=0
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
            m.F_liquified_fibers[t].value=2487*(1/60)*(1/60)
            m.F_C5liquid[t].value=628*(1/60)*(1/60)
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
            m.F_liquified_fibers[t].value=0
            m.F_C5liquid[t].value=0

    def _Feed_constraint(m,t):
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
            return m.Fin[t]==m.F_liquified_fibers[t]
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
            return m.Fin[t]==m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
            return m.Fin[t]==0#m.F_base+m.F_acid
    m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    def _Feed_concentration_constraint(m,t,j):
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
            return m.Cin[t,j]==m.C_liquified_fibers[j]
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
            return m.Cin[t,j]*(m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid)==(m.F_C5liquid[t]*m.C_C5liquid[j]+m.F_liquified_fibers[t]*m.C_liquified_fibers[j]+m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j])
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
            return m.Cin[t,j]*(m.F_base+m.F_acid)== 0#m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j]    
    m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)


    m.F_C5liquid[m.t.first()].fix(0)
    m.F_liquified_fibers[m.t.first()].fix(0)


    # def _current_integral_F(m):
    #     return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[m.t.prev(t)])+pe.value(m.F_liquified_fibers[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    # m.current_integral_F=pe.Param(initialize=_current_integral_F)

    # def _current_integral_C5(m):    
    #     return m.final_time*sum(  ((pe.value(m.F_C5liquid[m.t.prev(t)])+pe.value(m.F_C5liquid[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    # m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    # def _ingegral_F(m):
    #     return m.final_time*sum(  (((m.F_liquified_fibers[m.t.prev(t)])+(m.F_liquified_fibers[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_F
    # m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    # def _ingegral_C5(m):
    #     return m.final_time*sum(  (((m.F_C5liquid[m.t.prev(t)])+(m.F_C5liquid[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_C5
    # m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    # def _current_integral_F(m):
    #     return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())
    # m.current_integral_F=pe.Param(initialize=_current_integral_F)

    # def _current_integral_C5(m):    
    #     return m.final_time*sum(  ((pe.value(m.F_C5liquid[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())
    # m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    # def _ingegral_F(m):
    #     return m.final_time*sum(  (((m.F_liquified_fibers[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())==m.current_integral_F
    # m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    # def _ingegral_C5(m):
    #     return m.final_time*sum(  (((m.F_C5liquid[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())==m.current_integral_C5
    # m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    def _current_integral_F(m):
        return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    m.current_integral_F=pe.Param(initialize=_current_integral_F)

    def _current_integral_C5(m):    
        return m.final_time*sum(  ((pe.value(m.F_C5liquid[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    def _ingegral_F(m):
        return m.final_time*sum(  (((m.F_liquified_fibers[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_F
    m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    def _ingegral_C5(m):
        return m.final_time*sum(  (((m.F_C5liquid[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_C5
    m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    def _ingegral_F_rq(m,t):
        if  (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) >70*60*60:
            return m.F_liquified_fibers[t]==0
        elif t!=m.t.first():
            if keep_constant_flows:
                return m.F_liquified_fibers[t]==2487*(1/60)*(1/60)
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    m.ingegral_F_rq=pe.Constraint(m.t,rule=_ingegral_F_rq)

    def _ingegral_C5_rq(m,t):
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<= 10*60*60 or (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) >70*60*60:
            return  m.F_C5liquid[t]==0
        elif t!=m.t.first():
            if keep_constant_flows:
                return m.F_C5liquid[t]==628*(1/60)*(1/60)
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    m.ingegral_C5_rq=pe.Constraint(m.t,rule=_ingegral_C5_rq)

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
        objective_val=objective_function(m)
        return -objective_val
        # return (50*m.M0_yeast-5*m.C[m.t.last(),'Eth']*m.M[m.t.last()])+0*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
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

# For open loop optimization
def build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time: float=190*60*60,discretization: str='collocation',n_f_elements_t: int=1,total_f_elements_t:int=50,current_start_time_sconds: float=0,M0_prev_input: float=0,C0_prev_input: dict={'CS':0, 'XS':0, 'LS':0,'C':0,'G':0, 'X':0, 'F':0, 'E':0,'AC':0,'Cell':0,'Eth':0,'CO2':0,'ACT':0,'HMF':0,'Base':0},keep_constant_flows: bool=False):
    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='fermentation_model')
    # ------------shared scalars with hydrolisis model ----------------------

    # n_f_elements_t dictates the discretized prediction horizon (between 1 and total_f_elements)
    # total_f_elements is the fixed total number fo finite elements that defines the sambpling time with respect to total_sim_time
    m.final_time = pe.Param(initialize=(n_f_elements_t*total_sim_time)/total_f_elements_t,doc='Prediction horizon with respect to 0 seconds [s]')  
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
        if t==m.t.first() and m.current_starting_time==0: #Initial condition
            return m.M[t] == m.M0
        elif t==m.t.first(): #Final condition from previous time step
            return m.M[t] == m.M0_prev #TODO: remove, no longer needed. Now, we have to consider the case where initial conditio of the batch changes!
        else:
            return  m.dMdt[t] == m.final_time*(m.Fin[t]) 
        -m.vx*m.dCdx[t,x,j] +m.R[t,x,j]            
    m.Diff_mass=pe.Constraint(m.t,rule=_Diff_mass)

    # Balance per component equation
    def _Diff_comp(m,t,j):
  
        if t==m.t.first() and m.current_starting_time==0: #Initial condition
            return m.C[t,j] == m.C0[j]
        elif t==m.t.first(): # Final condition from previous step
            return m.C[t,j] == m.C0_prev[j] #TODO: remove, no longer needed. Now, we have to consider the case where initial conditio of the batch changes!
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

    # def _initF_C5(m,t):
    #     return 628*(1/60)*(1/60)
    # m.F_C5liquid=pe.Param(m.t,initialize=_initF_C5,doc='C5liquid flow [kg/s]')
    # def _initF_Fibers(m,t):
    #     return 2487*(1/60)*(1/60)
    # m.F_liquified_fibers=pe.Param(m.t,initialize=_initF_Fibers,doc='Liquified fibers flow [kg/s]')


    # m.pH=pe.Param(m.t,initialize=5.3969605,doc='pH')

    for t in m.t:
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
            m.F_liquified_fibers[t].value=2487*(1/60)*(1/60)
            m.F_C5liquid[t].value=0
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
            m.F_liquified_fibers[t].value=2487*(1/60)*(1/60)
            m.F_C5liquid[t].value=628*(1/60)*(1/60)
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
            m.F_liquified_fibers[t].value=0
            m.F_C5liquid[t].value=0

    def _Feed_constraint(m,t):
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
            return m.Fin[t]==m.F_liquified_fibers[t]
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
            return m.Fin[t]==m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid       #(m.Mmax-m.M0)/(70*60*60-10*60*60)
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
            return m.Fin[t]==0#m.F_base+m.F_acid
    m.Feed_constraint=pe.Constraint(m.t,rule=_Feed_constraint)

    def _Feed_concentration_constraint(m,t,j):
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=10*60*60: # Inoculum phase
            return m.Cin[t,j]==m.C_liquified_fibers[j]
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) <=70*60*60: #Fed-batch phase
            return m.Cin[t,j]*(m.F_C5liquid[t] + m.F_liquified_fibers[t]+m.F_base+m.F_acid)==(m.F_C5liquid[t]*m.C_C5liquid[j]+m.F_liquified_fibers[t]*m.C_liquified_fibers[j]+m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j])
        elif (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))>70*60*60 and (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<=190*60*60: #Batch phase
            return m.Cin[t,j]*(m.F_base+m.F_acid)== 0#m.F_base*m.C_base[j]+m.F_acid*m.C_acid[j]    
    m.Feed_concentration_constraint=pe.Constraint(m.t,m.j,rule=_Feed_concentration_constraint)


    m.F_C5liquid[m.t.first()].fix(0)
    m.F_liquified_fibers[m.t.first()].fix(0)


    # def _current_integral_F(m):
    #     return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[m.t.prev(t)])+pe.value(m.F_liquified_fibers[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    # m.current_integral_F=pe.Param(initialize=_current_integral_F)

    # def _current_integral_C5(m):    
    #     return m.final_time*sum(  ((pe.value(m.F_C5liquid[m.t.prev(t)])+pe.value(m.F_C5liquid[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    # m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    # def _ingegral_F(m):
    #     return m.final_time*sum(  (((m.F_liquified_fibers[m.t.prev(t)])+(m.F_liquified_fibers[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_F
    # m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    # def _ingegral_C5(m):
    #     return m.final_time*sum(  (((m.F_C5liquid[m.t.prev(t)])+(m.F_C5liquid[t]))/2)*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_C5
    # m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    # def _current_integral_F(m):
    #     return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())
    # m.current_integral_F=pe.Param(initialize=_current_integral_F)

    # def _current_integral_C5(m):    
    #     return m.final_time*sum(  ((pe.value(m.F_C5liquid[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())
    # m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    # def _ingegral_F(m):
    #     return m.final_time*sum(  (((m.F_liquified_fibers[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())==m.current_integral_F
    # m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    # def _ingegral_C5(m):
    #     return m.final_time*sum(  (((m.F_C5liquid[t])))*(m.t.next(t)-t)    for t in m.t if t!=m.t.last())==m.current_integral_C5
    # m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    def _current_integral_F(m):
        return  m.final_time*sum(  ((pe.value(m.F_liquified_fibers[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    m.current_integral_F=pe.Param(initialize=_current_integral_F)

    def _current_integral_C5(m):    
        return m.final_time*sum(  ((pe.value(m.F_C5liquid[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())
    m.current_integral_C5=pe.Param(initialize=_current_integral_C5)

    def _ingegral_F(m):
        return m.final_time*sum(  (((m.F_liquified_fibers[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_F
    m.ingegral_F=pe.Constraint(rule=_ingegral_F)

    def _ingegral_C5(m):
        return m.final_time*sum(  (((m.F_C5liquid[t])))*(t-m.t.prev(t))    for t in m.t if t!=m.t.first())==m.current_integral_C5
    m.ingegral_C5=pe.Constraint(rule=_ingegral_C5)


    def _ingegral_F_rq(m,t):
        if  (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) >70*60*60:
            return m.F_liquified_fibers[t]==0
        elif t!=m.t.first():
            if keep_constant_flows:
                return m.F_liquified_fibers[t]==2487*(1/60)*(1/60)
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    m.ingegral_F_rq=pe.Constraint(m.t,rule=_ingegral_F_rq)

    def _ingegral_C5_rq(m,t):
        if (m.current_starting_time+t*(m.current_final_time-m.current_starting_time))<= 10*60*60 or (m.current_starting_time+t*(m.current_final_time-m.current_starting_time)) >70*60*60:
            return  m.F_C5liquid[t]==0
        elif t!=m.t.first():
            if keep_constant_flows:
                return m.F_C5liquid[t]==628*(1/60)*(1/60)
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip
    m.ingegral_C5_rq=pe.Constraint(m.t,rule=_ingegral_C5_rq)

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

# FOR ONE STEP SIMULATION
def build_fermentation_one_time_step_new(total_sim_time: float=190*60*60,
                                         discretization: str='collocation',
                                         n_f_elements_t: int=1,total_f_elements_t:int=50,
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
    m.final_time = pe.Param(initialize=(total_sim_time)/(total_f_elements_t),doc='final simulation time with respect to 0 seconds [s]')  
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


def dsda_model(m_fer,keep_constant_flows: bool=False)-> pe.ConcreteModel():
    '''
    m_fer: fermentation model
    keep_constant_flows: True if flows are going to remain fixed
    '''
    #-----Model
    m=pe.ConcreteModel(name='gdp_dsda_model')
    #Fermentation model
    m.fer=m_fer
   #-------Ordered sets 
    # print(m.fer.t.__len__())
    # m.set1=pe.RangeSet(1,m.fer.t.__len__(),doc= "set of first group of Boolean variables") #Fed batch time
    m.set1=pe.Set(initialize=m.fer.t)
    m.set2=pe.Set(initialize=m.fer.t)#pe.RangeSet(1,x_up[0]+1,doc= "set of second group of Boolean variables") #Processing time
    #-----Variables
    m.Y1=pe.BooleanVar(m.set1,doc="Boolean variable associated to set 1")
    m.Y2=pe.BooleanVar(m.set2,doc="Boolean variable associated to set 2")
    #-----Logical constraints

    #Constraint that allow to apply the reformulation over Y1
    def select_one_Y1(m):
        return pe.exactly(1,m.Y1)
    m.oneY1=pe.LogicalConstraint(rule=select_one_Y1)

    #Constraint that allow to apply the reformulation over Y2
    def select_one_Y2(m):
        return pe.exactly(1,m.Y2)
    m.oneY2=pe.LogicalConstraint(rule=select_one_Y2)

    #Constraint that indicates that fed batch time is less than total processing time
    def _logic1(m,set2):
        return m.Y2[set2].implies(pe.lor([m.Y1[set1] for set1 in m.set1 if set1<=set2]))
    m.logic1=pe.LogicalConstraint(m.set2,rule=_logic1)

    # Fermentation model update
    # Deleting curent objective function
    m.fer.del_component(m.fer.obj)
    # new variable that will incorporate objective function
    m.obj=pe.Var(within=pe.Reals,initialize=0)
    # Updating input flow constraints
    m.fer.del_component(m.fer.Feed_constraint)
    m.fer.del_component(m.fer.Feed_concentration_constraint)
    m.fer.del_component(m.fer.ingegral_F_rq)
    m.fer.del_component(m.fer.ingegral_C5_rq)


    # Disjunctive section
    # m.delta=pe.Param(initialize=m.fer.final_time /(m.fer.t.__len__()-1),doc='lenght of time periods of discretized time grid for dynamcis [seconds]')    
    m.tau_p=pe.Param(initialize=70*60*60,mutable=True,doc='Time required for the fed batch phase [seconds]')
    # m.tau_p_batch=pe.Param(initialize=m.fer.final_time,mutable=True,doc='Time required for batch operation [seconds]')
    #-----First disjunction
    def build_disjuncts1(m,set1):  #Disjuncts for first Boolean variable

        # m.model().tau_p.value=(set1-1)*m.model().delta
        m.model().tau_p.value=m.model().fer.current_starting_time+set1*(m.model().fer.current_final_time-m.model().fer.current_starting_time)

        def _Feed_constraint_new(m,t):

            if (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))<=10*60*60: # Inoculum phase
                return m.model().fer.Fin[t]==m.model().fer.F_liquified_fibers[t]
            elif (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))> 10*60*60 and (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time)) <=pe.value(m.model().tau_p) : #Fed-batch phase
                return m.model().fer.Fin[t]==m.model().fer.F_C5liquid[t] + m.model().fer.F_liquified_fibers[t]       #(m.model().Mmax-m.model().M0)/(70*60*60-10*60*60)
            elif (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))>pe.value(m.model().tau_p)  and t*m.model().fer.final_time<=190*60*60: #Batch phase
                return m.model().fer.Fin[t]==0

        m.Feed_constraint_new=pe.Constraint(m.model().fer.t,rule=_Feed_constraint_new)

        def _Feed_concentration_constraint_new(m,t,j):
            if (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))<=10*60*60: # Inoculum phase
                return m.model().fer.Cin[t,j]==m.model().fer.C_liquified_fibers[j]
            elif (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))> 10*60*60 and (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time)) <=pe.value(m.model().tau_p) : #Fed-batch phase
                return m.model().fer.Cin[t,j]*(m.model().fer.F_C5liquid[t] + m.model().fer.F_liquified_fibers[t])==(m.model().fer.F_C5liquid[t]*m.model().fer.C_C5liquid[j]+m.model().fer.F_liquified_fibers[t]*m.model().fer.C_liquified_fibers[j])
            elif (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))>pe.value(m.model().tau_p)  and (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))<=190*60*60: #Batch phase
                return m.model().fer.Cin[t,j]== 0  
        m.Feed_concentration_constraint_new=pe.Constraint(m.model().fer.t,m.model().fer.j,rule=_Feed_concentration_constraint_new)


        def _ingegral_F_rq_new(m,t):
            if  (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time)) >pe.value(m.model().tau_p):
                return m.model().fer.F_liquified_fibers[t]==0
            elif t!=m.model().fer.t.first():
                if keep_constant_flows:
                    return m.model().fer.F_liquified_fibers[t]==2487*(1/60)*(1/60)
                else:
                    return pe.Constraint.Skip
            else:
                return pe.Constraint.Skip
        m.ingegral_F_rq_new=pe.Constraint(m.model().fer.t,rule=_ingegral_F_rq_new)

        def _ingegral_C5_rq_new(m,t):
            if (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time))<= 10*60*60 or (m.model().fer.current_starting_time+t*(m.model().fer.current_final_time-m.model().fer.current_starting_time)) >pe.value(m.model().tau_p):
                return  m.model().fer.F_C5liquid[t]==0
            elif t!=m.model().fer.t.first():
                if keep_constant_flows:
                    return m.model().fer.F_C5liquid[t]==628*(1/60)*(1/60)
                else:
                    return pe.Constraint.Skip
            else:
                return pe.Constraint.Skip
        m.ingegral_C5_rq_new=pe.Constraint(m.model().fer.t,rule=_ingegral_C5_rq_new)

        # def _deff_F(m,t):
        #     if t*m.model().fer.final_time<=10*60*60: # Inoculum phase
        #         return m.model().fer.F_liquified_fibers[t]==2487*(1/60)*(1/60)
        #     elif t*m.model().fer.final_time> 10*60*60 and t*m.model().fer.final_time <=pe.value(m.model().tau_p) : #Fed-batch phase
        #         return m.model().fer.F_liquified_fibers[t]==2487*(1/60)*(1/60)
        #     elif t*m.model().fer.final_time>pe.value(m.model().tau_p) and t*m.model().fer.final_time<=190*60*60: #Batch phase
        #         return m.model().fer.F_liquified_fibers[t]==0#pe.Constraint.Skip#m.model().fer.F_liquified_fibers_new[t]==0
        # m.deff_F=pe.Constraint(m.model().fer.t,rule=_deff_F)

        # def _deff_C5(m,t):
        #     if t*m.model().fer.final_time<=10*60*60: # Inoculum phase
        #         return m.model().fer.F_C5liquid[t]==0#pe.Constraint.Skip#m.model().fer.F_C5liquid_new[t]==0
        #     elif t*m.model().fer.final_time> 10*60*60 and t*m.model().fer.final_time <=pe.value(m.model().tau_p) : #Fed-batch phase
        #         return m.model().fer.F_C5liquid[t]==628*(1/60)*(1/60)
        #     elif t*m.model().fer.final_time>pe.value(m.model().tau_p) and t*m.model().fer.final_time<=190*60*60: #Batch phase
        #         return m.model().fer.F_C5liquid[t]==0#pe.Constraint.Skip#m.model().fer.F_C5liquid_new[t]==0
        # m.deff_C5=pe.Constraint(m.model().fer.t,rule=_deff_C5)
    m.Y1_disjunct=Disjunct(m.set1,rule=build_disjuncts1,doc="each disjunct is defined over set 1")
    # m.Y1_disjunct.pprint()


    def build_disjuncts2(m,set2):  #Disjuncts for first Boolean variable
        # def _ethanol_concentration_Requirement(m):
        #     return m.model().fer.C[set2,'Eth']>=0
        # m.ethanol_concentration_Requirement=pe.Constraint(rule=_ethanol_concentration_Requirement)

        def _batch_time(m):
            return m.model().obj==objective_function(m.model().fer,final_time=set2)
        m.batch_time=pe.Constraint(rule=_batch_time)
    m.Y2_disjunct=Disjunct(m.set2,rule=build_disjuncts2,doc="each disjunct is defined over set 2")

    def Disjunction1(m):    #Disjunction for first Boolean variable
        return [m.Y1_disjunct[j] for j in m.set1]
    m.Disjunction1=Disjunction(rule=Disjunction1,xor=True)


    def Disjunction2(m):    #Disjunction for second Boolean variable
        return [m.Y2_disjunct[j] for j in m.set2]
    m.Disjunction2=Disjunction(rule=Disjunction2,xor=True)

    #Associate boolean variables to disjuncts
    for n1 in m.set1:
        m.Y1[n1].associate_binary_var(m.Y1_disjunct[n1].indicator_var)

    for n2 in m.set2:
        m.Y2[n2].associate_binary_var(m.Y2_disjunct[n2].indicator_var)



    # m.dumob=pe.Var(within=pe.Reals,initialize=1)
    # def _dumob_c(m):
    #     return -m.fer.C[m.fer.t.last(),'Eth']<=m.dumob
    # m.dumob_c=pe.Constraint(rule=_dumob_c)
    def _obj_rule_new(m):
        # return -m.fer.C[m.fer.t.last(),'Eth']
        # return -m.fer.C[m.fer.t.last(),'Eth']*m.fer.M[m.fer.t.last()]
        # return -m.obj_dummy_time-1000*m.fer.C[m.fer.t.last(),'Eth']
        # return -m.fer.C[m.fer.t.last(),'Eth']
        return  m.obj
        # return m.dumob
        # return -m.fer.M[m.fer.t.last()]
    m.obj_lower_level=pe.Objective(rule=_obj_rule_new)
    return m

def complete_dsda_model_optimization(total_sim_time: float=190*60*60,discretization: str='collocation',n_f_elements_t: int=1,total_f_elements_t:int=50,current_start_time_sconds: float=0,keep_constant_flows: bool=False):

    m=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization=discretization,n_f_elements_t=n_f_elements_t,total_f_elements_t=total_f_elements_t,current_start_time_sconds=current_start_time_sconds,keep_constant_flows=keep_constant_flows)
    # m.pH.fix(5.4)
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_updated')
    mdsda=dsda_model(m,keep_constant_flows=keep_constant_flows)

    return mdsda



def reactors_5_complete_dsda_model_optimization(total_sim_time: float=190*60*60,discretization: str='collocation',n_f_elements_t: int=1,total_f_elements_t:int=50,current_start_time_sconds: float=0,keep_constant_flows: bool=False):



    # ------------pyomo model------------------------------------------------
    m = pe.ConcreteModel(name='5_reactor_model')
    m.react_set=pe.Set(initialize=[1,2,3,4,5],doc='Set of reactors')
    m.reactor={}
    m.set1={}
    m.set2={}
    m.Y1={}
    m.Y2={}
    m.oneY1={}
    m.oneY2={}
    m.logic1={}
    m.obj_term={}
    m.tau_p={}
    m.Y1_disjunct={}
    m.Y2_disjunct={}
    m.Disjunction1={}
    m.Disjunction2={}

    for r in m.react_set:
        # Fermentation model
        m_r=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization=discretization,n_f_elements_t=n_f_elements_t,total_f_elements_t=total_f_elements_t,current_start_time_sconds=current_start_time_sconds,keep_constant_flows=keep_constant_flows)
        # Initialize model
        if keep_constant_flows:
            m_r=initialize_model(m_r,from_feasible=True,feasible_model='validation_fermentation_updated')
        else:
            m_r=initialize_model(m_r,from_feasible=True,feasible_model='validation_fermentation_updated')
        
        # Fermentation model update
        # Deleting curent objective function
        m_r.del_component(m_r.obj)
        # Updating input flow constraints
        m_r.del_component(m_r.Feed_constraint)
        m_r.del_component(m_r.Feed_concentration_constraint)
        m_r.del_component(m_r.ingegral_F_rq)
        m_r.del_component(m_r.ingegral_C5_rq)
        
        m.reactor[r]=m_r
        setattr(m,'reactor_%r' %r,m.reactor[r])
    #-------Ordered sets 
        m.set1[r]=pe.Set(initialize=m_r.t)
        setattr(m,'set1_%r' %r,m.set1[r])
        m.set2[r]=pe.Set(initialize=m_r.t)#pe.RangeSet(1,x_up[0]+1,doc= "set of second group of Boolean variables") #Processing time
        setattr(m,'set2_%r' %r,m.set2[r])
        #-----Variables
        m.Y1[r]=pe.BooleanVar(m.set1[r],doc="Boolean variable associated to set 1")
        setattr(m,'Y1_%r' %r,m.Y1[r])
        m.Y2[r]=pe.BooleanVar(m.set2[r],doc="Boolean variable associated to set 2")
        setattr(m,'Y2_%r' %r,m.Y2[r])
        #-----Logical constraints

        #Constraint that allow to apply the reformulation over Y1
        def select_one_Y1(m):
            return pe.exactly(1,m.Y1[r])
        m.oneY1[r]=pe.LogicalConstraint(rule=select_one_Y1)
        setattr(m,'oneY1_%r' %r,m.oneY1[r])

        #Constraint that allow to apply the reformulation over Y2
        def select_one_Y2(m):
            return pe.exactly(1,m.Y2[r])
        m.oneY2[r]=pe.LogicalConstraint(rule=select_one_Y2)
        setattr(m,'oneY2_%r' %r,m.oneY2[r])
        #Constraint that indicates that fed batch time is less than total processing time
        def _logic1(m,set2):
            return m.Y2[r][set2].implies(pe.lor([m.Y1[r][set1] for set1 in m.set1[r] if set1<=set2]))
        m.logic1[r]=pe.LogicalConstraint(m.set2[r],rule=_logic1)
        setattr(m,'logic1_%r' %r,m.logic1[r])

        # new variable that will incorporate objective function
        m.obj_term[r]=pe.Var(within=pe.Reals,initialize=0)
        setattr(m,'obj_term_%r' %r,m.obj_term[r])


        # Disjunctive section
        # m.delta=pe.Param(initialize=m.fer.final_time /(m.fer.t.__len__()-1),doc='lenght of time periods of discretized time grid for dynamcis [seconds]')    
        m.tau_p[r]=pe.Param(initialize=70*60*60,mutable=True,doc='Time required for the fed batch phase [seconds]')
        setattr(m,'tau_p_%r' %r,m.tau_p[r])
        # m.tau_p_batch=pe.Param(initialize=m.fer.final_time,mutable=True,doc='Time required for batch operation [seconds]')
        #-----First disjunction
        def build_disjuncts1(m,set1):  #Disjuncts for first Boolean variable

            # m.model().reactor[r].tau_p.value=(set1-1)*m.model().reactor[r].delta
            m.model().tau_p[r].value=m.model().reactor[r].current_starting_time+set1*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time)

            def _Feed_constraint_new(m,t):

                if (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))<=10*60*60: # Inoculum phase
                    return m.model().reactor[r].Fin[t]==m.model().reactor[r].F_liquified_fibers[t]
                elif (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))> 10*60*60 and (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time)) <=pe.value(m.model().tau_p[r]) : #Fed-batch phase
                    return m.model().reactor[r].Fin[t]==m.model().reactor[r].F_C5liquid[t] + m.model().reactor[r].F_liquified_fibers[t]       #(m.model().reactor[r].Mmax-m.model().reactor[r].M0)/(70*60*60-10*60*60)
                elif (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))>pe.value(m.model().tau_p[r])  and t*m.model().reactor[r].final_time<=190*60*60: #Batch phase
                    return m.model().reactor[r].Fin[t]==0

            m.Feed_constraint_new=pe.Constraint(m.model().reactor[r].t,rule=_Feed_constraint_new)

            def _Feed_concentration_constraint_new(m,t,j):
                if (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))<=10*60*60: # Inoculum phase
                    return m.model().reactor[r].Cin[t,j]==m.model().reactor[r].C_liquified_fibers[j]
                elif (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))> 10*60*60 and (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time)) <=pe.value(m.model().tau_p[r]) : #Fed-batch phase
                    return m.model().reactor[r].Cin[t,j]*(m.model().reactor[r].F_C5liquid[t] + m.model().reactor[r].F_liquified_fibers[t])==(m.model().reactor[r].F_C5liquid[t]*m.model().reactor[r].C_C5liquid[j]+m.model().reactor[r].F_liquified_fibers[t]*m.model().reactor[r].C_liquified_fibers[j])
                elif (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))>pe.value(m.model().tau_p[r])  and (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))<=190*60*60: #Batch phase
                    return m.model().reactor[r].Cin[t,j]== 0  
            m.Feed_concentration_constraint_new=pe.Constraint(m.model().reactor[r].t,m.model().reactor[r].j,rule=_Feed_concentration_constraint_new)


            def _ingegral_F_rq_new(m,t):
                if  (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time)) >pe.value(m.model().tau_p[r]):
                    return m.model().reactor[r].F_liquified_fibers[t]==0
                elif t!=m.model().reactor[r].t.first():
                    if keep_constant_flows:
                        return m.model().reactor[r].F_liquified_fibers[t]==2487*(1/60)*(1/60)
                    else:
                        return pe.Constraint.Skip
                else:
                    return pe.Constraint.Skip
            m.ingegral_F_rq_new=pe.Constraint(m.model().reactor[r].t,rule=_ingegral_F_rq_new)

            def _ingegral_C5_rq_new(m,t):
                if (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time))<= 10*60*60 or (m.model().reactor[r].current_starting_time+t*(m.model().reactor[r].current_final_time-m.model().reactor[r].current_starting_time)) >pe.value(m.model().tau_p[r]):
                    return  m.model().reactor[r].F_C5liquid[t]==0
                elif t!=m.model().reactor[r].t.first():
                    if keep_constant_flows:
                        return m.model().reactor[r].F_C5liquid[t]==628*(1/60)*(1/60)
                    else:
                        return pe.Constraint.Skip
                else:
                    return pe.Constraint.Skip
            m.ingegral_C5_rq_new=pe.Constraint(m.model().reactor[r].t,rule=_ingegral_C5_rq_new)

            # def _deff_F(m,t):
            #     if t*m.model().reactor[r].final_time<=10*60*60: # Inoculum phase
            #         return m.model().reactor[r].F_liquified_fibers[t]==2487*(1/60)*(1/60)
            #     elif t*m.model().reactor[r].final_time> 10*60*60 and t*m.model().reactor[r].final_time <=pe.value(m.model().reactor[r].tau_p) : #Fed-batch phase
            #         return m.model().reactor[r].F_liquified_fibers[t]==2487*(1/60)*(1/60)
            #     elif t*m.model().reactor[r].final_time>pe.value(m.model().reactor[r].tau_p) and t*m.model().reactor[r].final_time<=190*60*60: #Batch phase
            #         return m.model().reactor[r].F_liquified_fibers[t]==0#pe.Constraint.Skip#m.model().reactor[r].F_liquified_fibers_new[t]==0
            # m.deff_F=pe.Constraint(m.model().reactor[r].t,rule=_deff_F)

            # def _deff_C5(m,t):
            #     if t*m.model().reactor[r].final_time<=10*60*60: # Inoculum phase
            #         return m.model().reactor[r].F_C5liquid[t]==0#pe.Constraint.Skip#m.model().reactor[r].F_C5liquid_new[t]==0
            #     elif t*m.model().reactor[r].final_time> 10*60*60 and t*m.model().reactor[r].final_time <=pe.value(m.model().reactor[r].tau_p) : #Fed-batch phase
            #         return m.model().reactor[r].F_C5liquid[t]==628*(1/60)*(1/60)
            #     elif t*m.model().reactor[r].final_time>pe.value(m.model().reactor[r].tau_p) and t*m.model().reactor[r].final_time<=190*60*60: #Batch phase
            #         return m.model().reactor[r].F_C5liquid[t]==0#pe.Constraint.Skip#m.model().reactor[r].F_C5liquid_new[t]==0
            # m.deff_C5=pe.Constraint(m.model().reactor[r].t,rule=_deff_C5)
        m.Y1_disjunct[r]=Disjunct(m.set1[r],rule=build_disjuncts1,doc="each disjunct is defined over set 1")
        setattr(m,'Y1_disjunct_%r' %r,m.Y1_disjunct[r])
        # m.Y1_disjunct.pprint()


        def build_disjuncts2(m,set2):  #Disjuncts for first Boolean variable
            # def _ethanol_concentration_Requirement(m):
            #     return m.model().reactor[r].C[set2,'Eth']>=0
            # m.ethanol_concentration_Requirement=pe.Constraint(rule=_ethanol_concentration_Requirement)

            def _batch_time(m):
                return m.model().obj_term[r]==objective_function(m.model().reactor[r],final_time=set2)
            m.batch_time=pe.Constraint(rule=_batch_time)
        m.Y2_disjunct[r]=Disjunct(m.set2[r],rule=build_disjuncts2,doc="each disjunct is defined over set 2")
        setattr(m,'Y2_disjunct_%r' %r,m.Y2_disjunct[r])

        def Disjunction1(m):    #Disjunction for first Boolean variable
            return [m.Y1_disjunct[r][j] for j in m.set1[r]]
        m.Disjunction1[r]=Disjunction(rule=Disjunction1,xor=True)
        setattr(m,'Disjunction1_%r' %r,m.Disjunction1[r])

        def Disjunction2(m):    #Disjunction for second Boolean variable
            return [m.Y2_disjunct[r][j] for j in m.set2[r]]
        m.Disjunction2[r]=Disjunction(rule=Disjunction2,xor=True)
        setattr(m,'Disjunction2_%r' %r,m.Disjunction2[r])

        #Associate boolean variables to disjuncts
        for n1 in m.set1[r]:
            m.Y1[r][n1].associate_binary_var(m.Y1_disjunct[r][n1].indicator_var)

        for n2 in m.set2[r]:
            m.Y2[r][n2].associate_binary_var(m.Y2_disjunct[r][n2].indicator_var)
 

    # Declare new objective function
    def _global_objective(m):
        return  sum(m.obj_term[r] for r in m.react_set)
    m.obj=pe.Objective(rule=_global_objective)

    return m



# ROBUST OPTIMIZATION
def main_robust_model():
    m=pe.ConcreteModel(name='Robust optimization main problem')

    m.objvar=pe.Var(within=pe.Reals)
    
    def _obj(m):
        return m.objvar
    m.obj=pe.Objective(rule=_obj)
    return m

def objective_function(m,final_time: float=-1):
    wight=0
    if final_time==-1: # Means that final time is not being assigned, which indicates that final time is not being optimized
        return (50*m.M0_yeast-5*m.C[m.t.last(),'Eth']*m.M[m.t.last()])+wight*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
    else:
        return ((50*m.M0_yeast-5*m.C[final_time,'Eth']*m.M[final_time])
                +5*final_time*(m.current_final_time-m.current_starting_time)
                +wight*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first() and t<=final_time)+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first() and t<=final_time))
                )
    # return (50*m.M0_yeast-5*m.C[m.t[20],'Eth']*m.M[m.t[20]])+1e+8*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon
    # return (-5*m.C[m.t.last(),'Eth'])+0*(sum( (m.F_C5liquid[t]-m.F_C5liquid[m.t.prev(t)])**2 for t in m.t if t !=m.t.first())+sum((m.F_liquified_fibers[t]-m.F_liquified_fibers[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()))#+0*sum((m.pH[t]-m.pH[m.t.prev(t)])**2 for t in m.t if t !=m.t.first()) #maximize concentration of ethanol at the end of the prediction horizon

def solve_robust(mad,aux,vtol,variation_param,solver_list,tee):
    '''
    Returns model m, optimized as a robust optimization problem considering uncertainties in flow compositions
    '''
    # Optimizer: Define optimization model for optimizer in robust optimization
    mad.obj.deactivate()
    # Optimizer: robust optimization
    main=main_robust_model()
    main.cuts=pe.ConstraintList()

    # pessimizer: Define optimization model for pessimizer in robust optimization        
    # Flows, pH and Myeast are fixed in pessimization, hence there are some constraints that we do not need to avoid numerical issues
    aux.ingegral_F.deactivate()
    aux.ingegral_C5.deactivate()
    aux.ingegral_F_rq.deactivate()
    aux.ingegral_C5_rq.deactivate()

    u_prev=[]  #Initialization of uncertainty
    u_lower=[]
    u_upper=[]

    # percentage variation
    variation=variation_param

    for j in aux.j:
        aux.C_C5liquid[j].ub=pe.value(aux.C_C5liquid[j])+pe.value(aux.C_C5liquid[j])*variation
        aux.C_C5liquid[j].lb=max(pe.value(aux.C_C5liquid[j])-pe.value(aux.C_C5liquid[j])*variation,0)
        
        u_prev.append(pe.value(aux.C_C5liquid[j]))
        u_lower.append(aux.C_C5liquid[j].lb)
        u_upper.append(aux.C_C5liquid[j].ub)

        
        aux.C_liquified_fibers[j].ub=pe.value(aux.C_liquified_fibers[j])+pe.value(aux.C_liquified_fibers[j])*variation
        aux.C_liquified_fibers[j].lb=max(pe.value(aux.C_liquified_fibers[j])-pe.value(aux.C_liquified_fibers[j])*variation,0)
        
        u_prev.append(pe.value(aux.C_liquified_fibers[j]))
        u_lower.append(aux.C_liquified_fibers[j].lb)
        u_upper.append(aux.C_liquified_fibers[j].ub)

    print('^^^^ Nominal parameters: ',u_prev)
    print('^^^^ Lower bounds: ',u_lower)
    print('^^^^ Upper bounds:',u_upper)
    
    iterat=5# maximum number of iterations
    main.scenarios={}
    for k in range(iterat):
        print('^^^^ -----ROBUST OPT ITERATION',k)
        # 1) OPTIMIZATION
        main.scenarios[k]=mad.clone()
        locat=0
        for j in main.scenarios[k].j: 
            main.scenarios[k].C_C5liquid[j]=u_prev[locat]
            locat=locat+1
            main.scenarios[k].C_liquified_fibers[j]=u_prev[locat]
            locat=locat+1
        setattr(main,'scenarios_%s' %k,main.scenarios[k])

        main.cuts.add(objective_function(main.scenarios[k])<=main.objvar) 

        # Guarantee that variables are the same for evey scenario
        if k>=1:
            for kprime in range(k):
                kc=kprime+1
                main.cuts.add(main.scenarios[kc].pH==main.scenarios[0].pH)
                main.cuts.add(main.scenarios[kc].M0_yeast==main.scenarios[0].M0_yeast)
                for t in main.scenarios[0].t:
                    main.cuts.add(main.scenarios[kc].F_C5liquid[t]==main.scenarios[0].F_C5liquid[t])
                    main.cuts.add(main.scenarios[kc].F_liquified_fibers[t]==main.scenarios[0].F_liquified_fibers[t])
        
            # Initialize model (previous uncertain scenarions)
            main=initialize_model(main,from_feasible=True,feasible_model='main_optimized')
            # Initialize model (current uncertain scenario)
            for v in aux.component_objects(ctype=pe.Var):
                for c in main.scenarios[k].component_objects(ctype=pe.Var):
                    if 'scenarios_'+str(k)+'.'+v.name==c.name:
                        # print(v.name)
                        # print(c.name)
                        for index in v.index_set().data():
                            # print(dir(v[index]))
                            c[index].value=v[index].value

                        break            
        generate_initialization(m=main,model_name='current_init')
        opt1 = SolverFactory('gams') # Solve problem

        for solver_used in solver_list:
            main.results = opt1.solve(main, solver=solver_used, tee=tee)

            if main.results.solver.termination_condition == 'infeasible' or main.results.solver.termination_condition == 'other' or main.results.solver.termination_condition == 'unbounded' or main.results.solver.termination_condition == 'invalidProblem' or main.results.solver.termination_condition == 'solverFailure' or main.results.solver.termination_condition == 'internalSolverError' or main.results.solver.termination_condition == 'error'  or main.results.solver.termination_condition == 'resourceInterrupt' or main.results.solver.termination_condition == 'licensingProblem' or main.results.solver.termination_condition == 'noSolution' or main.results.solver.termination_condition == 'noSolution' or main.results.solver.termination_condition == 'intermediateNonInteger':
                main.dsda_status = 'Evaluated_Infeasible'
                main=initialize_model(main,from_feasible=True,feasible_model='current_init')  
                
            else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                main.dsda_status = 'Optimal'
                break
        
        print('^^^^ Iteration:',k,'--Optimization status:',main.dsda_status,'--last solver used:',solver_used)
        # print('Objective from OPTIMIZATION: ',pe.value(main.objvar))
        # print('pH from OPTIMIZATION: ',pe.value(main.scenarios[0].pH))
        # print('Myeast from OPTIMIZATION: ',pe.value(main.scenarios[0].M0_yeast))
        # main.scenarios[0].F_C5liquid.pprint()
        # main.scenarios[0].F_liquified_fibers.pprint()

        # GENERATE INITIALIZATION
        generate_initialization(m=main,model_name='main_optimized')        

        # 2) PESSIMIZATION
        aux.pH.fix(pe.value(main.scenarios[0].pH))
        aux.M0_yeast.fix(pe.value(main.scenarios[0].M0_yeast))
        for t in aux.t:
            aux.F_C5liquid[t].fix(pe.value(main.scenarios[0].F_C5liquid[t]))
            aux.F_liquified_fibers[t].fix(pe.value(main.scenarios[0].F_liquified_fibers[t]))
    
        
        # initialize pessimization
        for c in main.scenarios[k].component_objects(ctype=pe.Var):
            for v in aux.component_objects(ctype=pe.Var):
                if 'scenarios_'+str(k)+'.'+v.name==c.name:
                    # print(v.name)
                    # print(c.name)
                    for index in v.index_set().data():
                        # print(dir(v[index]))
                        v[index].value=c[index].value

                    break
                    # for comp_v in v:
                    #     comp_v.value=pe.value(c)
                # print(v.name)
                # print(c.name)

        generate_initialization(m=aux,model_name='current_init_pes')
        opt1 = SolverFactory('gams') # Solve problem

        for solver_used in solver_list:
            aux.results = opt1.solve(aux, solver=solver_used, tee=tee)

            if aux.results.solver.termination_condition == 'infeasible' or aux.results.solver.termination_condition == 'other' or aux.results.solver.termination_condition == 'unbounded' or aux.results.solver.termination_condition == 'invalidProblem' or aux.results.solver.termination_condition == 'solverFailure' or aux.results.solver.termination_condition == 'internalSolverError' or aux.results.solver.termination_condition == 'error'  or aux.results.solver.termination_condition == 'resourceInterrupt' or aux.results.solver.termination_condition == 'licensingProblem' or aux.results.solver.termination_condition == 'noSolution' or aux.results.solver.termination_condition == 'noSolution' or aux.results.solver.termination_condition == 'intermediateNonInteger':
                aux.dsda_status = 'Evaluated_Infeasible'
                aux=initialize_model(aux,from_feasible=True,feasible_model='current_init_pes')  
                
            else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                aux.dsda_status = 'Optimal'
                break
        generate_initialization(m=aux,model_name='aux_pessimized')
        print('^^^^ Iteration:',k,'--Pessimization status:',aux.dsda_status,'--last solver used:',solver_used)        
        #3) UPDATE UNCERTAINTY SAMPLED POINTS
        const_violation=-pe.value(aux.obj)-pe.value(main.objvar)
        print('^^^^ Constraint violation from PESSIMIZATION: ',const_violation)
        if const_violation>0:
            locat = 0
            for j in aux.j:
                u_prev[locat]=pe.value(aux.C_C5liquid[j])
                locat=locat+1
                u_prev[locat]=pe.value(aux.C_liquified_fibers[j])
                locat=locat+1
            print('^^^^ New worst case parameter from PESSIMIZATION: u=',u_prev)
        # 4) VERIFY STOPPING CRITERION
        if const_violation<=vtol:
            break        

    # FINAL OPTIMIZATION MODEL WITH OPTIMIZED PARAMETERS AND VARIABLES UNDER WORST CASE  
    mad=initialize_model(mad,from_feasible=True,feasible_model='aux_pessimized')
    # locat=0
    # for j in mad.j: 
    #     mad.C_C5liquid[j]=u_prev[locat]
    #     locat=locat+1
    #     mad.C_liquified_fibers[j]=u_prev[locat]
    #     locat=locat+1
    mad.dsda_status=aux.dsda_status
    return mad

if __name__ == '__main__':
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


    keep_Ph_FIXED=False
    pHval=5.37#5.3805

    keep_Yeast_FIXED=False
    Yeastval=80


    solver_list=['conopt','conopt4','knitro','baron','ipopth']
    tee=True
    discretization_type_fer='differences'
    # discretization_type_fer='collocation'
    finite_elem_t_fer=50 
    total_elements=finite_elem_t_fer #prediction horizon, which is constnt, i.e., the total batch duration
    total_sim_time=190*(60)*(60) #Total batch time in seconds 
    step=total_sim_time/total_elements      #Sampling_time
    start_time=0
    disturbance=True





    control_horizon=19   # 19 is actually the last time I perform control actions, hence, the control horizon should be at most this or less

    # Simulation parameters
    sim_discretization='collocation'
    sim_n_finite_elements=3
    simulation_solvers=['conopt','conopt4','knitro','baron','ipopth']


    # ROBUST OPTIMIZATION parameters
    robust=True# True if optimization problems solved using robust optimization
    vtol=1e-5
    variation_param_opt=0.3 # parameter to define uncertainty range (for optimization)
    variation_param_sim=0.3 # parameter to define uncertainty range (for simulation)

    # CLOSED LOOP
    constant_flows=True
    random.seed(10)
    time_list=[] #Simulated time points
    Hold_up_list=[] #Simulated hold ups
    pH_list=[] # Simulated pH
    yeast_list=[] # Simulated yeast
    C5_list=[] # Simulated C5 flow
    fiber_list=[] #Simulated fibers flow
    objective_list=[]
    Concentration_dict={'CS':[], 'XS':[], 'LS':[],'C':[],'G':[], 'X':[], 'F':[], 'E':[],'AC':[],'Cell':[],'Eth':[],'CO2':[],'ACT':[],'HMF':[],'Base':[]} #Simulated concentrations


    # ################################  INITIALIZE D-SDA version
   
    # m=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization='differences',n_f_elements_t=total_elements,total_f_elements_t=total_elements,current_start_time_sconds=start_time,keep_constant_flows=constant_flows)    
    # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_updated')

    # # m=initialize_model(m,from_feasible=True,feasible_model='validation_fermentation_updated')
    # # m.pH.fix(5.4) # NOTE: IF FIXING VARIABLES, MUST BE DONE AFTER INITIALIZING MODEL, OTHERWHISE THEY WILL BE AUTOMATICALLY FIXED AT INITIALZIATION VALUE
    # mdsda=dsda_model(m,keep_constant_flows=constant_flows)
    # ext_ref={mdsda.Y1:mdsda.set1,mdsda.Y2:mdsda.set2}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(mdsda,ext_ref,tee=True)


    # x_init=[round((70*60*60-mdsda.fer.final_time)*((upper_bounds[1]-lower_bounds[1])/(mdsda.fer.final_time))+upper_bounds[1]),51]
    # # x_init=[18,32]
    # mdsda=external_ref(mdsda,x_init,dummy_logic,reformulation_dict,tee =False)
    # sub_options={}
    # mdsda=solve_subproblem(mdsda,subproblem_solver='conopt4',subproblem_solver_options=sub_options,tee=True)
    # if not constant_flows:
    #     generate_initialization(m=mdsda,model_name='validation_fermentation_updated_dsda_variable_flows')
    # else:
    #     generate_initialization(m=mdsda,model_name='validation_fermentation_updated_dsda_constant_flows')


    # mdsda.fer.M0_yeast.pprint()
    # mdsda.fer.pH.pprint()
    # mdsda.fer.F_C5liquid.pprint()
    # mdsda.fer.F_liquified_fibers.pprint()

    # yeast_used=pe.value(mdsda.fer.M0_yeast)
    # pH_used=pe.value(mdsda.fer.pH)
    # ethanol_concentration=pe.value(mdsda.fer.C[mdsda.fer.t[x_init[1]],'Eth'])
    # Final_hold_up=pe.value(mdsda.fer.M[mdsda.fer.t[x_init[1]]])
    # optimized_time=mdsda.fer.t[x_init[1]]*(mdsda.fer.current_final_time-mdsda.fer.current_starting_time)


    # print('\n ****RELEVANT VARIABLES:')
    # print('Optimal pH [kg]:',pH_used)
    # print('Yeast [kg]:',yeast_used)
    # print('Ethanol concentration [g/kg]:',ethanol_concentration)
    # print('Final hold-up [kg]',Final_hold_up)
    # print('Optimized processing time [s]',optimized_time)


    # print('\n ****COST BREAKDOWN:')
    # print('Yeast cost $:',yeast_used*50)
    # print('Ethanol revenue $:',5*ethanol_concentration*Final_hold_up)
    # print('Operation cost $:',5*optimized_time)
    # print('Profit $:',5*ethanol_concentration*Final_hold_up-yeast_used*50-5*optimized_time)



    ############################### INITIAL DSDA TEST
    # start=time.time()
    # neighdef='2'
    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > conopt4.opt \n','$offecho \n']}
    # arguments_dict={'total_sim_time':total_sim_time,
    #                 'discretization':discretization_type_fer,
    #                 'n_f_elements_t':total_elements,
    #                 'total_f_elements_t':total_elements,
    #                 'current_start_time_sconds':start_time,
    #                 'keep_constant_flows':constant_flows}
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(complete_dsda_model_optimization,arguments_dict,x_init,ext_ref,dummy_logic,k = neighdef,provide_starting_initialization= True,feasible_model='validation_fermentation_updated_dsda',subproblem_solver = 'conopt4',subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = tee,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=False)

    # end=time.time()
    # print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  
    # D_SDAsol.fer.M0_yeast.pprint()
    # D_SDAsol.fer.pH.pprint()
    # yeast_used=pe.value(D_SDAsol.fer.M0_yeast)
    # pH_used=pe.value(D_SDAsol.fer.pH)
    # ethanol_concentration=pe.value(D_SDAsol.fer.C[D_SDAsol.fer.t[routeDSDA[-1][1]],'Eth'])
    # Final_hold_up=pe.value(D_SDAsol.fer.M[D_SDAsol.fer.t[routeDSDA[-1][1]]])
    # optimized_time=D_SDAsol.fer.t[routeDSDA[-1][1]]*(D_SDAsol.fer.current_final_time-D_SDAsol.fer.current_starting_time)


    # print('\n ****RELEVANT VARIABLES:')
    # print('Optimal pH [kg]:',pH_used)
    # print('Yeast [kg]:',yeast_used)
    # print('Ethanol concentration [g/kg]:',ethanol_concentration)
    # print('Final hold-up [kg]',Final_hold_up)
    # print('Optimized processing time [s]',optimized_time)


    # print('\n ****COST BREAKDOWN:')
    # print('Yeast cost $:',yeast_used*50)
    # print('Ethanol revenue $:',5*ethanol_concentration*Final_hold_up)
    # print('Operation cost $:',5*optimized_time)
    # print('Profit $:',5*ethanol_concentration*Final_hold_up-yeast_used*50-5*optimized_time)



    ######## 5 REACTORS MODEL##################
    m5=reactors_5_complete_dsda_model_optimization(total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=total_elements,total_f_elements_t=total_elements,current_start_time_sconds=start_time,keep_constant_flows=constant_flows)
    ext_ref={}

    for r in m5.react_set:
        ext_ref[m5.Y1[r]]=m5.set1[r]
        ext_ref[m5.Y2[r]]=m5.set2[r]

    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m5,ext_ref,tee=True)

    x_init=[]
    for r in m5.react_set:
        x_init.append(round((70*60*60-m5.reactor[r].final_time)*((upper_bounds[1]-lower_bounds[1])/(m5.reactor[r].final_time))+upper_bounds[1]))
        x_init.append(51)

    mdsda=external_ref(m5,x_init,dummy_logic_v2,reformulation_dict,tee =False)
    sub_options={}
    mdsda=solve_subproblem(mdsda,subproblem_solver='conopt4',subproblem_solver_options=sub_options,tee=True)


    start=time.time()
    neighdef='2'
    sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=0;','$onecho > conopt4.opt \n','$offecho \n']}
    arguments_dict={'total_sim_time':total_sim_time,
                    'discretization':discretization_type_fer,
                    'n_f_elements_t':total_elements,
                    'total_f_elements_t':total_elements,
                    'current_start_time_sconds':start_time,
                    'keep_constant_flows':constant_flows}
    D_SDAsol,routeDSDA,obj_route=solve_with_dsda(reactors_5_complete_dsda_model_optimization,arguments_dict,x_init,ext_ref,dummy_logic_v2,k = neighdef,provide_starting_initialization= True,feasible_model='validation_fermentation_updated_dsda',subproblem_solver = 'conopt4',subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = tee,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=True)

    end=time.time()
    print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  
    for r in m5.react_set:
        print('/n /n !!!!!!!!!!!!!! REACTOR',r,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        D_SDAsol.reactor[r].M0_yeast.pprint()
        D_SDAsol.reactor[r].pH.pprint()
        yeast_used=pe.value(D_SDAsol.reactor[r].M0_yeast)
        pH_used=pe.value(D_SDAsol.reactor[r].pH)
        ethanol_concentration=pe.value(D_SDAsol.reactor[r].C[D_SDAsol.reactor[r].t[routeDSDA[-1][1]],'Eth'])
        Final_hold_up=pe.value(D_SDAsol.reactor[r].M[D_SDAsol.reactor[r].t[routeDSDA[-1][1]]])
        optimized_time=D_SDAsol.reactor[r].t[routeDSDA[-1][1]]*(D_SDAsol.reactor[r].current_final_time-D_SDAsol.reactor[r].current_starting_time)


        print('\n ****RELEVANT VARIABLES:')
        print('Optimal pH [kg]:',pH_used)
        print('Yeast [kg]:',yeast_used)
        print('Ethanol concentration [g/kg]:',ethanol_concentration)
        print('Final hold-up [kg]',Final_hold_up)
        print('Optimized processing time [s]',optimized_time)


        print('\n ****COST BREAKDOWN:')
        print('Yeast cost $:',yeast_used*50)
        print('Ethanol revenue $:',5*ethanol_concentration*Final_hold_up)
        print('Operation cost $:',5*optimized_time)
        print('Profit $:',5*ethanol_concentration*Final_hold_up-yeast_used*50-5*optimized_time)






    # C0_prev={}
    # count_last_elements=0
    # for disc_time in range(total_elements):
    # # for disc_time in [0]:
    #     current_start_time=disc_time*step #current start time

    #     # Define optimization model
    #     mad=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=total_elements,total_f_elements_t=total_elements,current_start_time_sconds=start_time,keep_constant_flows=constant_flows)  

    #     #Decrease number of finite elements in control horizon once we are approaching the end of the batch
    #     if disc_time+control_horizon>=total_elements+1:
    #         count_last_elements=count_last_elements+1
    #         control_horizon_updated=control_horizon-count_last_elements
    #     else:
    #         control_horizon_updated=control_horizon 

    #         # Keep control actions constant after the end of the control horizon                
    #         non_controlled_horizon=total_elements-control_horizon_updated
            
    #         # TODO: assuming finite differences
    #         def _constant_control_C5(m,t):
    #             if m.t.ord(m.t.last())-m.t.ord(t)<=non_controlled_horizon-1 and all([(m.current_starting_time+tt*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+tt*(m.current_final_time-m.current_starting_time)) <=70*60*60 for tt in [m.t.prev(t),t]]):
    #                 return m.F_C5liquid[t]==m.F_C5liquid[m.t.prev(t)]
    #             else:
    #                 return pe.Constraint.Skip
    #         mad.constant_control_C5=pe.Constraint(mad.t,rule=_constant_control_C5)
    #         def _constant_control_F(m,t):
    #             if m.t.ord(m.t.last())-m.t.ord(t)<=non_controlled_horizon-1 and all([(m.current_starting_time+tt*(m.current_final_time-m.current_starting_time)) <=70*60*60 for tt in [m.t.prev(t),t]]):
    #                 return m.F_liquified_fibers[t]==m.F_liquified_fibers[m.t.prev(t)]
    #             else:
    #                 return pe.Constraint.Skip
    #         mad.constant_control_F=pe.Constraint(mad.t,rule=_constant_control_F)


    #     if disc_time!=0: 
    #         mad=initialize_model(mad,from_feasible=True,feasible_model='prev_init') 

    #         # Fix first stage desicions
    #         mad.pH.fix(pe.value(mad.pH))
    #         mad.M0_yeast.fix(pe.value(mad.M0_yeast))       

    #         # Fix previous steps desicions @TODO: what I should do here is a simulation of one time step in the Simulation oriented Model (SOM), including potential disturbances; then fix previous control actions , update states and remove differential equations to avoid infeasibilities
    #         # TODO: assuming finite differences, actually, to consider mistmatch, I can assume orthogonal collocation for simulation to consider mistmatch


    #         for t in mad.t:
    #             if mad.t.ord(t)<=disc_time+1:
    #                 mad.F_C5liquid[t].fix(pe.value(mad.F_C5liquid[t]))
    #                 mad.F_liquified_fibers[t].fix(pe.value(mad.F_liquified_fibers[t]))
    #                 mad.M[t].fix(pe.value(mad.M[t]))
    #                 for j in mad.j:
    #                     mad.C[t,j].fix(pe.value(mad.C[t,j]))
                    
    #                 mad.Diff_mass[t].deactivate()
    #                 for j in mad.j:
    #                     mad.Diff_comp[t,j].deactivate()


    #             #     # TODO: also fix states up to this point+1 (those measured from simulation) and remove constraints up to this point. Do not remove integral constraints or other constraints needed during the whole horizon
    #             # elif mad.t.ord(t)<=disc_time+1:
    #             #     mad.M[t].fix(pe.value(mad.M[t]))
    #             #     for j in mad.j:
    #             #         mad.C[t,j].fix(pe.value(mad.C[t,j]))
    #     else: 
    #         mad=initialize_model(mad,from_feasible=True,feasible_model='validation_fermentation_updated')

    #         if keep_Ph_FIXED:
    #             mad.pH.fix(pHval)
    #         if keep_Yeast_FIXED:
    #             mad.M0_yeast.fix(Yeastval)
    #     if current_start_time<=70*60*60:

    #         if robust:
    #             aux=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_pessimization(total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=total_elements,total_f_elements_t=total_elements,current_start_time_sconds=start_time,keep_constant_flows=constant_flows)
    #             if disc_time==0:
    #                 aux=initialize_model(aux,from_feasible=True,feasible_model='validation_fermentation_updated')
    #             else:
    #                 aux=initialize_model(aux,from_feasible=True,feasible_model='prev_init')
    #                 for t in aux.t:
    #                     if aux.t.ord(t)<=disc_time+1:
    #                         aux.F_C5liquid[t].fix(pe.value(aux.F_C5liquid[t]))
    #                         aux.F_liquified_fibers[t].fix(pe.value(aux.F_liquified_fibers[t]))
    #                         aux.M[t].fix(pe.value(aux.M[t]))
    #                         for j in aux.j:
    #                             aux.C[t,j].fix(pe.value(aux.C[t,j]))
                            
    #                         aux.Diff_mass[t].deactivate()
    #                         for j in aux.j:
    #                             aux.Diff_comp[t,j].deactivate()
    #             mad=solve_robust(mad,aux,vtol,variation_param_opt,solver_list,tee)
    #             print('Iteration:',disc_time,'--Status:',mad.dsda_status)
    #             if mad.dsda_status=='Evaluated_Infeasible':
    #                 break
    #         else:
    #             opt1 = SolverFactory('gams') # Solve problem

    #             for solver_used in solver_list:
    #                 mad.results = opt1.solve(mad, solver=solver_used, tee=tee)

    #                 if mad.results.solver.termination_condition == 'infeasible' or mad.results.solver.termination_condition == 'other' or mad.results.solver.termination_condition == 'unbounded' or mad.results.solver.termination_condition == 'invalidProblem' or mad.results.solver.termination_condition == 'solverFailure' or mad.results.solver.termination_condition == 'internalSolverError' or mad.results.solver.termination_condition == 'error'  or mad.results.solver.termination_condition == 'resourceInterrupt' or mad.results.solver.termination_condition == 'licensingProblem' or mad.results.solver.termination_condition == 'noSolution' or mad.results.solver.termination_condition == 'noSolution' or mad.results.solver.termination_condition == 'intermediateNonInteger':
    #                     mad.dsda_status = 'Evaluated_Infeasible'
    #                     if disc_time!=0:
    #                         mad=initialize_model(mad,from_feasible=True,feasible_model='prev_init')  
    #                 else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
    #                     mad.dsda_status = 'Optimal'
    #                     break

    #             print('Iteration:',disc_time,'--Status:',mad.dsda_status,'--last solver used:',solver_used)
    #             if mad.dsda_status=='Evaluated_Infeasible':
    #                 break
        
    #     # Simulate optimal control action using one time step
    #     # --1) retrieve optimal control action 
    #     optimal_F=pe.value(mad.F_liquified_fibers[round((current_start_time+step)/total_sim_time,6)])
    #     # print(optimal_F)
    #     optimal_C5=pe.value(mad.F_C5liquid[round((current_start_time+step)/total_sim_time,6)])
    #     # print(optimal_C5)
    #     print('calculated added mass',(optimal_F+optimal_C5)*step)
    #     optimal_pH=pe.value(mad.pH)
    #     optimal_yeast=pe.value(mad.M0_yeast)
    #     # --2) retrieve current initial state
    #     M0_prev=pe.value(mad.M[round(current_start_time/total_sim_time,6)]) 
    #     print('Plant initial state',M0_prev)
    #     for j in mad.j:
    #         C0_prev[j]=pe.value(mad.C[round(current_start_time/total_sim_time,6),j]) 
    #     # --3) perform one time step simulation
    #     if disturbance:
    #         # u_disturbance=[-0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0, 0, -0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0, -0.3, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0, 0.0]

    #         # # #0,1,2,3,4,5,8,9,10,11,12,13,15,16,17,24,25,26,27,29
    #         # cs_disturbance_C5=u_disturbance[0]
    #         # xs_disturbance_C5=u_disturbance[2]
    #         # ls_disturbance_C5=u_disturbance[4]
    #         # g_disturbance_C5=u_disturbance[8]
    #         # x_disturbance_C5=u_disturbance[10]
    #         # f_disturbance_C5=u_disturbance[12]
    #         # ac_disturbance_C5=u_disturbance[16]
    #         # act_disturbance_C5=u_disturbance[24]
    #         # hmf_disturbance_C5=u_disturbance[26]


    #         # cs_disturbance_F=u_disturbance[1]
    #         # xs_disturbance_F=u_disturbance[3]
    #         # ls_disturbance_F=u_disturbance[5]
    #         # g_disturbance_F=u_disturbance[9]
    #         # x_disturbance_F=u_disturbance[11]
    #         # f_disturbance_F=u_disturbance[13]
    #         # e_disturbance_F=u_disturbance[15]
    #         # ac_disturbance_F=u_disturbance[17]
    #         # act_disturbance_F=u_disturbance[25]
    #         # hmf_disturbance_F=u_disturbance[27]
    #         # base_disturbance_F=u_disturbance[29]

            



    #         g_disturbance_F=-variation_param_sim
    #         x_disturbance_F=-variation_param_sim
    #         cs_disturbance_F=-variation_param_sim
    #         xs_disturbance_F=-variation_param_sim
    #         ls_disturbance_F=-variation_param_sim
    #         f_disturbance_F=-variation_param_sim
    #         e_disturbance_F=-variation_param_sim
    #         ac_disturbance_F=-variation_param_sim
    #         act_disturbance_F=-variation_param_sim
    #         hmf_disturbance_F=-variation_param_sim
    #         base_disturbance_F=-variation_param_sim

            
    #         g_disturbance_C5=-variation_param_sim
    #         x_disturbance_C5=-variation_param_sim
    #         cs_disturbance_C5=-variation_param_sim
    #         xs_disturbance_C5=-variation_param_sim
    #         ls_disturbance_C5=-variation_param_sim
    #         f_disturbance_C5=-variation_param_sim
    #         ac_disturbance_C5=-variation_param_sim
    #         act_disturbance_C5=-variation_param_sim
    #         hmf_disturbance_C5=-variation_param_sim

    #         # g_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # x_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # cs_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # xs_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ls_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # f_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # e_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ac_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # act_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # hmf_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # base_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)

            
    #         # g_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # x_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # cs_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # xs_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ls_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # f_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ac_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # act_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # hmf_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)

    #         # g_disturbance_F=-0.5#random.uniform(-0.5,0.5)
    #         # x_disturbance_F=-0.5#random.uniform(-0.5,0.5)
    #         # cs_disturbance_F=0
    #         # xs_disturbance_F=0
    #         # ls_disturbance_F=0
    #         # f_disturbance_F=0
    #         # e_disturbance_F=0
    #         # ac_disturbance_F=0
    #         # act_disturbance_F=0
    #         # hmf_disturbance_F=0
    #         # base_disturbance_F=0

            
    #         # g_disturbance_C5=-0.5#random.uniform(-0.5,0.5)
    #         # x_disturbance_C5=-0.5#random.uniform(-0.5,0.5)
    #         # cs_disturbance_C5=0
    #         # xs_disturbance_C5=0
    #         # ls_disturbance_C5=0
    #         # f_disturbance_C5=0
    #         # ac_disturbance_C5=0
    #         # act_disturbance_C5=0
    #         # hmf_disturbance_C5=0
    #     else:
    #         g_disturbance_F=0
    #         x_disturbance_F=0
    #         cs_disturbance_F=0
    #         xs_disturbance_F=0
    #         ls_disturbance_F=0
    #         f_disturbance_F=0
    #         e_disturbance_F=0
    #         ac_disturbance_F=0
    #         act_disturbance_F=0
    #         hmf_disturbance_F=0
    #         base_disturbance_F=0

            
    #         g_disturbance_C5=0
    #         x_disturbance_C5=0
    #         cs_disturbance_C5=0
    #         xs_disturbance_C5=0
    #         ls_disturbance_C5=0
    #         f_disturbance_C5=0
    #         ac_disturbance_C5=0
    #         act_disturbance_C5=0
    #         hmf_disturbance_C5=0              
    #     mad_sim=build_fermentation_one_time_step_new(total_sim_time=total_sim_time,
    #                                                     discretization=sim_discretization,
    #                                                     n_f_elements_t=sim_n_finite_elements,
    #                                                     total_f_elements_t=total_elements,
    #                                                     current_start_time_sconds=current_start_time,
    #                                                     M0_prev_input=M0_prev,
    #                                                     C0_prev_input=C0_prev,
    #                                                     pH_val=optimal_pH,
    #                                                     M_yeast=optimal_yeast,
    #                                                     F_fibers=optimal_F,
    #                                                     F_C5=optimal_C5,
    #                                                     glucose_disturbance_F=g_disturbance_F,
    #                                                     xylose_disturbance_F=x_disturbance_F,
    #                                                     cs_disturbance_F=cs_disturbance_F,
    #                                                     xs_disturbance_F=xs_disturbance_F,
    #                                                     ls_disturbance_F=ls_disturbance_F,
    #                                                     f_disturbance_F=f_disturbance_F,
    #                                                     e_disturbance_F=e_disturbance_F,
    #                                                     ac_disturbance_F=ac_disturbance_F,
    #                                                     act_disturbance_F=act_disturbance_F,
    #                                                     hmf_disturbance_F=hmf_disturbance_F,
    #                                                     base_disturbance_F=base_disturbance_F,
    #                                                     glucose_disturbance_C5=g_disturbance_C5,
    #                                                     xylose_disturbance_C5=x_disturbance_C5,
    #                                                     cs_disturbance_C5=cs_disturbance_C5,
    #                                                     xs_disturbance_C5=xs_disturbance_C5,
    #                                                     ls_disturbance_C5=ls_disturbance_C5,
    #                                                     f_disturbance_C5=f_disturbance_C5,
    #                                                     ac_disturbance_C5=ac_disturbance_C5,
    #                                                     act_disturbance_C5=act_disturbance_C5,
    #                                                     hmf_disturbance_C5=hmf_disturbance_C5) 
    #     if disc_time!=0:
    #         mad_sim=initialize_model(mad_sim,from_feasible=True,feasible_model='prev_init_sim')  
    #     # --4) Solve simulation
    #     opt2 = SolverFactory('gams') # Solve problem

    #     for solver_used in simulation_solvers:
    #         mad_sim.results = opt2.solve(mad_sim, solver=solver_used, tee=tee)

    #         if mad_sim.results.solver.termination_condition == 'infeasible' or mad_sim.results.solver.termination_condition == 'other' or mad_sim.results.solver.termination_condition == 'unbounded' or mad_sim.results.solver.termination_condition == 'invalidProblem' or mad_sim.results.solver.termination_condition == 'solverFailure' or mad_sim.results.solver.termination_condition == 'internalSolverError' or mad_sim.results.solver.termination_condition == 'error'  or mad_sim.results.solver.termination_condition == 'resourceInterrupt' or mad_sim.results.solver.termination_condition == 'licensingProblem' or mad_sim.results.solver.termination_condition == 'noSolution' or mad_sim.results.solver.termination_condition == 'noSolution' or mad_sim.results.solver.termination_condition == 'intermediateNonInteger':
    #             mad_sim.dsda_status = 'Evaluated_Infeasible'

    #         else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
    #             mad_sim.dsda_status = 'Optimal'
    #             break
    #     print('Iteration:',disc_time,'--Simulation Status:',mad_sim.dsda_status,'--last solver used:',solver_used)
    #     if mad_sim.dsda_status=='Evaluated_Infeasible':
    #         break            
    #     # --5): Update original model miwth new states
    #     # print(mad_sim.final_time.value,round((current_start_time+step),6))
    #     # print(mad.M[round((current_start_time+step)/total_sim_time,6)].value)
    #     print('Plant final state',pe.value(mad_sim.M[mad_sim.t.last()]))
    #     print('Plant added mass',pe.value(mad_sim.M[mad_sim.t.last()])-M0_prev)
    #     mad.M[round((current_start_time+step)/total_sim_time,6)].value=pe.value(mad_sim.M[mad_sim.t.last()])
    #     # print(mad.M[round((current_start_time+step)/total_sim_time,6)].value)
    #     for j in mad.j:
    #         mad.C[round((current_start_time+step)/total_sim_time,6),j].value=pe.value(mad_sim.C[mad_sim.t.last(),j])


    #     generate_initialization(m=mad_sim,model_name='prev_init_sim')
    #     generate_initialization(m=mad,model_name='prev_init')

        
    #     # for t in [k for k in mad.t if mad.t.ord(k)==disc_time+1]:
    #     #     time_list.append((mad.current_starting_time+t*(mad.current_final_time-mad.current_starting_time))*(1/(60*60)))
    #     #     Hold_up_list.append(pe.value(mad.M[t]))
    #     #     pH_list.append(pe.value(mad.pH))
    #     #     yeast_list.append(pe.value(mad.M0_yeast))
    #     #     C5_list.append(pe.value(mad.F_C5liquid[t]))
    #     #     fiber_list.append(pe.value(mad.F_liquified_fibers[t]))
    #     #     for j in mad.j:
    #     #         Concentration_dict[j].append(pe.value(mad.C[t,j]))
    #     # if disc_time==total_elements-1:
    #     #     time_list.append(total_sim_time*(1/(60*60)))
    #     #     Hold_up_list.append(pe.value(mad.M[mad.t.last()]))
    #     #     pH_list.append(pe.value(mad.pH))
    #     #     yeast_list.append(pe.value(mad.M0_yeast))
    #     #     C5_list.append(pe.value(mad.F_C5liquid[mad.t.last()]))
    #     #     fiber_list.append(pe.value(mad.F_liquified_fibers[mad.t.last()]))
    #     #     for j in mad.j:
    #     #         Concentration_dict[j].append(pe.value(mad.C[mad.t.last(),j]))     


    #     for t in mad_sim.t:
    #         time_list.append((mad_sim.current_starting_time+t*(mad_sim.current_final_time-mad_sim.current_starting_time))*(1/(60*60)))
    #         Hold_up_list.append(pe.value(mad_sim.M[t]))
    #         pH_list.append(pe.value(mad_sim.pH))
    #         yeast_list.append(pe.value(mad_sim.M0_yeast))
    #         C5_list.append(pe.value(mad_sim.F_C5liquid))
    #         fiber_list.append(pe.value(mad_sim.F_liquified_fibers))
    #         for j in mad_sim.j:
    #             Concentration_dict[j].append(pe.value(mad_sim.C[t,j])) 
    # final_objective=(50*yeast_list[0]-5*Concentration_dict['Eth'][-1]*Hold_up_list[-1])
    # final_hold_up=Hold_up_list[-1]
    # final_ethanol_concentration=Concentration_dict['Eth'][-1]
    # yeast_mass=yeast_list[0]
    # print('Evaluated economic objective function: ',final_objective) 
    # print('Final hold up [kg]', final_hold_up)
    # print('Final ethanol concentration [g/kg]',final_ethanol_concentration)
    # print('Yeast mass [kg]',yeast_mass)
    # print('*************************************\n\n')          

    #     # save objective function
    #     # objective_list.append()




    # # OPEN LOOP
    # constant_flows=True
    # random.seed(10)
    # time_list_open=[] #Simulated time points
    # Hold_up_list_open=[] #Simulated hold ups
    # pH_list_open=[] # Simulated pH
    # yeast_list_open=[] # Simulated yeast
    # C5_list_open=[] # Simulated C5 flow
    # fiber_list_open=[] #Simulated fibers flow
    # objective_list_open=[]
    # Concentration_dict_open={'CS':[], 'XS':[], 'LS':[],'C':[],'G':[], 'X':[], 'F':[], 'E':[],'AC':[],'Cell':[],'Eth':[],'CO2':[],'ACT':[],'HMF':[],'Base':[]} #Simulated concentrations

    # C0_prev={}
    # count_last_elements=0
    # for disc_time in range(total_elements):
    # # for disc_time in [0]:
    #     current_start_time=disc_time*step #current start time

    #     # Define optimization model
    #     mad=build_fermentation_one_time_step_optimizing_flows_pH_open_loop_optimization(total_sim_time=total_sim_time,discretization=discretization_type_fer,n_f_elements_t=total_elements,total_f_elements_t=total_elements,current_start_time_sconds=start_time,keep_constant_flows=constant_flows)  

    #     #Decrease number of finite elements in control horizon once we are approaching the end of the batch
    #     if disc_time+control_horizon>=total_elements+1:
    #         count_last_elements=count_last_elements+1
    #         control_horizon_updated=control_horizon-count_last_elements
    #     else:
    #         control_horizon_updated=control_horizon 

    #         # Keep control actions constant after the end of the control horizon                
    #         non_controlled_horizon=total_elements-control_horizon_updated
            
    #         # TODO: assuming finite differences
    #         def _constant_control_C5(m,t):
    #             if m.t.ord(m.t.last())-m.t.ord(t)<=non_controlled_horizon-1 and all([(m.current_starting_time+tt*(m.current_final_time-m.current_starting_time))> 10*60*60 and (m.current_starting_time+tt*(m.current_final_time-m.current_starting_time)) <=70*60*60 for tt in [m.t.prev(t),t]]):
    #                 return m.F_C5liquid[t]==m.F_C5liquid[m.t.prev(t)]
    #             else:
    #                 return pe.Constraint.Skip
    #         mad.constant_control_C5=pe.Constraint(mad.t,rule=_constant_control_C5)
    #         def _constant_control_F(m,t):
    #             if m.t.ord(m.t.last())-m.t.ord(t)<=non_controlled_horizon-1 and all([(m.current_starting_time+tt*(m.current_final_time-m.current_starting_time)) <=70*60*60 for tt in [m.t.prev(t),t]]):
    #                 return m.F_liquified_fibers[t]==m.F_liquified_fibers[m.t.prev(t)]
    #             else:
    #                 return pe.Constraint.Skip
    #         mad.constant_control_F=pe.Constraint(mad.t,rule=_constant_control_F)


    #     if disc_time!=0: 
    #         mad=initialize_model(mad,from_feasible=True,feasible_model='prev_init') 

    #         # Fix first stage desicions
    #         mad.pH.fix(pe.value(mad.pH))
    #         mad.M0_yeast.fix(pe.value(mad.M0_yeast))       

    #         # Fix previous steps desicions @TODO: what I should do here is a simulation of one time step in the Simulation oriented Model (SOM), including potential disturbances; then fix previous control actions , update states and remove differential equations to avoid infeasibilities
    #         # TODO: assuming finite differences, actually, to consider mistmatch, I can assume orthogonal collocation for simulation to consider mistmatch

    #         for t in mad.t:
    #             mad.F_C5liquid[t].fix(pe.value(mad.F_C5liquid[t]))
    #             mad.F_liquified_fibers[t].fix(pe.value(mad.F_liquified_fibers[t]))



    #             #     # TODO: also fix states up to this point+1 (those measured from simulation) and remove constraints up to this point. Do not remove integral constraints or other constraints needed during the whole horizon
    #             # elif mad.t.ord(t)<=disc_time+1:
    #             #     mad.M[t].fix(pe.value(mad.M[t]))
    #             #     for j in mad.j:
    #             #         mad.C[t,j].fix(pe.value(mad.C[t,j]))
    #     else: 
    #         mad=initialize_model(mad,from_feasible=True,feasible_model='validation_fermentation_updated')

    #         if keep_Ph_FIXED:
    #             mad.pH.fix(pHval)
    #         if keep_Yeast_FIXED:
    #             mad.M0_yeast.fix(Yeastval)
    #     if current_start_time==0:
    #         opt1 = SolverFactory('gams') # Solve problem

    #         for solver_used in solver_list:
    #             mad.results = opt1.solve(mad, solver=solver_used, tee=tee)

    #             if mad.results.solver.termination_condition == 'infeasible' or mad.results.solver.termination_condition == 'other' or mad.results.solver.termination_condition == 'unbounded' or mad.results.solver.termination_condition == 'invalidProblem' or mad.results.solver.termination_condition == 'solverFailure' or mad.results.solver.termination_condition == 'internalSolverError' or mad.results.solver.termination_condition == 'error'  or mad.results.solver.termination_condition == 'resourceInterrupt' or mad.results.solver.termination_condition == 'licensingProblem' or mad.results.solver.termination_condition == 'noSolution' or mad.results.solver.termination_condition == 'noSolution' or mad.results.solver.termination_condition == 'intermediateNonInteger':
    #                 mad.dsda_status = 'Evaluated_Infeasible'
    #                 if disc_time!=0:
    #                     mad=initialize_model(mad,from_feasible=True,feasible_model='prev_init')  
    #             else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
    #                 mad.dsda_status = 'Optimal'
    #                 break
            

            


    #         print('Iteration:',disc_time,'--Status:',mad.dsda_status,'--last solver used:',solver_used)
    #         if mad.dsda_status=='Evaluated_Infeasible':
    #             break
        
    #     # Simulate optimal control action using one time step
    #     # --1) retrieve optimal control action 
    #     optimal_F=pe.value(mad.F_liquified_fibers[round((current_start_time+step)/total_sim_time,6)])
    #     # print(optimal_F)
    #     optimal_C5=pe.value(mad.F_C5liquid[round((current_start_time+step)/total_sim_time,6)])
    #     # print(optimal_C5)
    #     print('calculated added mass',(optimal_F+optimal_C5)*step)
    #     optimal_pH=pe.value(mad.pH)
    #     optimal_yeast=pe.value(mad.M0_yeast)
    #     # --2) retrieve current initial state
    #     M0_prev=pe.value(mad.M[round(current_start_time/total_sim_time,6)]) 
    #     print('Plant initial state',M0_prev)
    #     for j in mad.j:
    #         C0_prev[j]=pe.value(mad.C[round(current_start_time/total_sim_time,6),j]) 
    #     # --3) perform one time step simulation
    #     if disturbance:
    #         # u_disturbance=[-0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0, 0, -0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0, -0.3, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0, 0.0]

    #         # # #0,1,2,3,4,5,8,9,10,11,12,13,15,16,17,24,25,26,27,29
    #         # cs_disturbance_C5=u_disturbance[0]
    #         # xs_disturbance_C5=u_disturbance[2]
    #         # ls_disturbance_C5=u_disturbance[4]
    #         # g_disturbance_C5=u_disturbance[8]
    #         # x_disturbance_C5=u_disturbance[10]
    #         # f_disturbance_C5=u_disturbance[12]
    #         # ac_disturbance_C5=u_disturbance[16]
    #         # act_disturbance_C5=u_disturbance[24]
    #         # hmf_disturbance_C5=u_disturbance[26]


    #         # cs_disturbance_F=u_disturbance[1]
    #         # xs_disturbance_F=u_disturbance[3]
    #         # ls_disturbance_F=u_disturbance[5]
    #         # g_disturbance_F=u_disturbance[9]
    #         # x_disturbance_F=u_disturbance[11]
    #         # f_disturbance_F=u_disturbance[13]
    #         # e_disturbance_F=u_disturbance[15]
    #         # ac_disturbance_F=u_disturbance[17]
    #         # act_disturbance_F=u_disturbance[25]
    #         # hmf_disturbance_F=u_disturbance[27]
    #         # base_disturbance_F=u_disturbance[29]

    #         g_disturbance_F=-variation_param_sim
    #         x_disturbance_F=-variation_param_sim
    #         cs_disturbance_F=-variation_param_sim
    #         xs_disturbance_F=-variation_param_sim
    #         ls_disturbance_F=-variation_param_sim
    #         f_disturbance_F=-variation_param_sim
    #         e_disturbance_F=-variation_param_sim
    #         ac_disturbance_F=-variation_param_sim
    #         act_disturbance_F=-variation_param_sim
    #         hmf_disturbance_F=-variation_param_sim
    #         base_disturbance_F=-variation_param_sim

            
    #         g_disturbance_C5=-variation_param_sim
    #         x_disturbance_C5=-variation_param_sim
    #         cs_disturbance_C5=-variation_param_sim
    #         xs_disturbance_C5=-variation_param_sim
    #         ls_disturbance_C5=-variation_param_sim
    #         f_disturbance_C5=-variation_param_sim
    #         ac_disturbance_C5=-variation_param_sim
    #         act_disturbance_C5=-variation_param_sim
    #         hmf_disturbance_C5=-variation_param_sim

    #         # g_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # x_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # cs_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # xs_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ls_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # f_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # e_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ac_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # act_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # hmf_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)
    #         # base_disturbance_F=random.uniform(-variation_param_sim,variation_param_sim)

            
    #         # g_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # x_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # cs_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # xs_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ls_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # f_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # ac_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # act_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)
    #         # hmf_disturbance_C5=random.uniform(-variation_param_sim,variation_param_sim)

    #         # g_disturbance_F=-0.5#random.uniform(-0.5,0.5)
    #         # x_disturbance_F=-0.5#random.uniform(-0.5,0.5)
    #         # cs_disturbance_F=0
    #         # xs_disturbance_F=0
    #         # ls_disturbance_F=0
    #         # f_disturbance_F=0
    #         # e_disturbance_F=0
    #         # ac_disturbance_F=0
    #         # act_disturbance_F=0
    #         # hmf_disturbance_F=0
    #         # base_disturbance_F=0

            
    #         # g_disturbance_C5=-0.5#random.uniform(-0.5,0.5)
    #         # x_disturbance_C5=-0.5#random.uniform(-0.5,0.5)
    #         # cs_disturbance_C5=0
    #         # xs_disturbance_C5=0
    #         # ls_disturbance_C5=0
    #         # f_disturbance_C5=0
    #         # ac_disturbance_C5=0
    #         # act_disturbance_C5=0
    #         # hmf_disturbance_C5=0
    #     else:
    #         g_disturbance_F=0
    #         x_disturbance_F=0
    #         cs_disturbance_F=0
    #         xs_disturbance_F=0
    #         ls_disturbance_F=0
    #         f_disturbance_F=0
    #         e_disturbance_F=0
    #         ac_disturbance_F=0
    #         act_disturbance_F=0
    #         hmf_disturbance_F=0
    #         base_disturbance_F=0

            
    #         g_disturbance_C5=0
    #         x_disturbance_C5=0
    #         cs_disturbance_C5=0
    #         xs_disturbance_C5=0
    #         ls_disturbance_C5=0
    #         f_disturbance_C5=0
    #         ac_disturbance_C5=0
    #         act_disturbance_C5=0
    #         hmf_disturbance_C5=0               
    #     mad_sim=build_fermentation_one_time_step_new(total_sim_time=total_sim_time,
    #                                                     discretization=sim_discretization,
    #                                                     n_f_elements_t=sim_n_finite_elements,
    #                                                     total_f_elements_t=total_elements,
    #                                                     current_start_time_sconds=current_start_time,
    #                                                     M0_prev_input=M0_prev,
    #                                                     C0_prev_input=C0_prev,
    #                                                     pH_val=optimal_pH,
    #                                                     M_yeast=optimal_yeast,
    #                                                     F_fibers=optimal_F,
    #                                                     F_C5=optimal_C5,
    #                                                     glucose_disturbance_F=g_disturbance_F,
    #                                                     xylose_disturbance_F=x_disturbance_F,
    #                                                     cs_disturbance_F=cs_disturbance_F,
    #                                                     xs_disturbance_F=xs_disturbance_F,
    #                                                     ls_disturbance_F=ls_disturbance_F,
    #                                                     f_disturbance_F=f_disturbance_F,
    #                                                     e_disturbance_F=e_disturbance_F,
    #                                                     ac_disturbance_F=ac_disturbance_F,
    #                                                     act_disturbance_F=act_disturbance_F,
    #                                                     hmf_disturbance_F=hmf_disturbance_F,
    #                                                     base_disturbance_F=base_disturbance_F,
    #                                                     glucose_disturbance_C5=g_disturbance_C5,
    #                                                     xylose_disturbance_C5=x_disturbance_C5,
    #                                                     cs_disturbance_C5=cs_disturbance_C5,
    #                                                     xs_disturbance_C5=xs_disturbance_C5,
    #                                                     ls_disturbance_C5=ls_disturbance_C5,
    #                                                     f_disturbance_C5=f_disturbance_C5,
    #                                                     ac_disturbance_C5=ac_disturbance_C5,
    #                                                     act_disturbance_C5=act_disturbance_C5,
    #                                                     hmf_disturbance_C5=hmf_disturbance_C5) 
    #     if disc_time!=0:
    #         mad_sim=initialize_model(mad_sim,from_feasible=True,feasible_model='prev_init_sim')  
    #     # --4) Solve simulation
    #     opt2 = SolverFactory('gams') # Solve problem

    #     for solver_used in simulation_solvers:
    #         mad_sim.results = opt2.solve(mad_sim, solver=solver_used, tee=tee)

    #         if mad_sim.results.solver.termination_condition == 'infeasible' or mad_sim.results.solver.termination_condition == 'other' or mad_sim.results.solver.termination_condition == 'unbounded' or mad_sim.results.solver.termination_condition == 'invalidProblem' or mad_sim.results.solver.termination_condition == 'solverFailure' or mad_sim.results.solver.termination_condition == 'internalSolverError' or mad_sim.results.solver.termination_condition == 'error'  or mad_sim.results.solver.termination_condition == 'resourceInterrupt' or mad_sim.results.solver.termination_condition == 'licensingProblem' or mad_sim.results.solver.termination_condition == 'noSolution' or mad_sim.results.solver.termination_condition == 'noSolution' or mad_sim.results.solver.termination_condition == 'intermediateNonInteger':
    #             mad_sim.dsda_status = 'Evaluated_Infeasible'

    #         else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
    #             mad_sim.dsda_status = 'Optimal'
    #             break
    #     print('Iteration:',disc_time,'--Simulation Status:',mad_sim.dsda_status,'--last solver used:',solver_used)
    #     if mad_sim.dsda_status=='Evaluated_Infeasible':
    #         break            
    #     # --5): Update original model miwth new states
    #     # print(mad_sim.final_time.value,round((current_start_time+step),6))
    #     # print(mad.M[round((current_start_time+step)/total_sim_time,6)].value)
    #     mad.M[round((current_start_time+step)/total_sim_time,6)].value=pe.value(mad_sim.M[mad_sim.t.last()])
    #     print('Plant final state',pe.value(mad_sim.M[mad_sim.t.last()]))
    #     print('Plant added mass',pe.value(mad_sim.M[mad_sim.t.last()])-M0_prev)
    #     # print(mad.M[round((current_start_time+step)/total_sim_time,6)].value)
    #     for j in mad.j:
    #         mad.C[round((current_start_time+step)/total_sim_time,6),j].value=pe.value(mad_sim.C[mad_sim.t.last(),j])


    #     generate_initialization(m=mad_sim,model_name='prev_init_sim')
    #     generate_initialization(m=mad,model_name='prev_init')

        
    #     # for t in [k for k in mad.t if mad.t.ord(k)==disc_time+1]:
    #     #     time_list.append((mad.current_starting_time+t*(mad.current_final_time-mad.current_starting_time))*(1/(60*60)))
    #     #     Hold_up_list.append(pe.value(mad.M[t]))
    #     #     pH_list.append(pe.value(mad.pH))
    #     #     yeast_list.append(pe.value(mad.M0_yeast))
    #     #     C5_list.append(pe.value(mad.F_C5liquid[t]))
    #     #     fiber_list.append(pe.value(mad.F_liquified_fibers[t]))
    #     #     for j in mad.j:
    #     #         Concentration_dict[j].append(pe.value(mad.C[t,j]))
    #     # if disc_time==total_elements-1:
    #     #     time_list.append(total_sim_time*(1/(60*60)))
    #     #     Hold_up_list.append(pe.value(mad.M[mad.t.last()]))
    #     #     pH_list.append(pe.value(mad.pH))
    #     #     yeast_list.append(pe.value(mad.M0_yeast))
    #     #     C5_list.append(pe.value(mad.F_C5liquid[mad.t.last()]))
    #     #     fiber_list.append(pe.value(mad.F_liquified_fibers[mad.t.last()]))
    #     #     for j in mad.j:
    #     #         Concentration_dict[j].append(pe.value(mad.C[mad.t.last(),j]))     


    #     for t in mad_sim.t:
    #         time_list_open.append((mad_sim.current_starting_time+t*(mad_sim.current_final_time-mad_sim.current_starting_time))*(1/(60*60)))
    #         Hold_up_list_open.append(pe.value(mad_sim.M[t]))
    #         pH_list_open.append(pe.value(mad_sim.pH))
    #         yeast_list_open.append(pe.value(mad_sim.M0_yeast))
    #         C5_list_open.append(pe.value(mad_sim.F_C5liquid))
    #         fiber_list_open.append(pe.value(mad_sim.F_liquified_fibers))
    #         for j in mad_sim.j:
    #             Concentration_dict_open[j].append(pe.value(mad_sim.C[t,j]))            

    #     # save objective function
    #     # objective_list_open.append()

    # final_objective_open=(50*yeast_list_open[0]-5*Concentration_dict_open['Eth'][-1]*Hold_up_list_open[-1])
    # final_hold_up_open=Hold_up_list_open[-1]
    # final_ethanol_concentration_open=Concentration_dict_open['Eth'][-1]
    # yeast_mass_open=yeast_list_open[0]
    # print('Evaluated economic objective function: ',final_objective_open) 
    # print('Final hold up [kg]', final_hold_up_open)
    # print('Final ethanol concentration [g/kg]',final_ethanol_concentration_open)
    # print('Yeast mass [kg]',yeast_mass_open)
    # print('*************************************\n\n')    

    # colors=['b','g','m','r','k','y','c']
    # contador=-1
    # for j in mad.j:
    #     if j=='G' or j=='X' or j=='Eth' or j=='Cell':
    #         contador=contador+1
    #         plt.plot(time_list,Concentration_dict[j],colors[contador],label=j+' (ENMPC)')
    #         plt.plot(time_list_open,Concentration_dict_open[j],'--'+colors[contador],label=j+' (Constant flows)')
    #         # original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
    #         # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
    #     plt.xlabel('time [h]')
    #     plt.ylabel('Concentration [g/kg]')
    #     plt.legend()
    # plt.show()

    # contador=-1
    # for j in mad.j:
    #     if j=='CS' or j=='XS' or j=='E':
    #         contador=contador+1
    #         plt.plot(time_list,Concentration_dict[j],colors[contador],label=j+' (ENMPC)')
    #         plt.plot(time_list_open,Concentration_dict_open[j],'--'+colors[contador],label=j+' (Constant flows)')
    #         # original = pd.read_csv('biorefinery_models/'+j+'_ferm.csv', header=None)
    #         # plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values,'--'+colors[contador])
    #     plt.xlabel('time [h]')
    #     plt.ylabel('Concentration [g/kg]')
    #     plt.legend()
    # plt.show()

    # plt.plot(time_list,pH_list,label='ENMPC')
    # plt.plot(time_list_open,pH_list_open,label='Constant flows')
    # plt.xlabel('time [h]')
    # plt.ylabel('pH')
    # plt.legend()
    # plt.show()

    # plt.plot(time_list,C5_list,label='ENMPC')
    # plt.plot(time_list_open,C5_list_open,label='Constant flows')
    # plt.xlabel('time [h]')
    # plt.ylabel('C5 flow [kg/s]')
    # plt.legend()
    # plt.show()

    # plt.plot(time_list,fiber_list,label='ENMPC')
    # plt.plot(time_list_open,fiber_list_open,label='Constant flows')
    # plt.xlabel('time [h]')
    # plt.ylabel('Liquified fibers flow [kg/s]')
    # plt.legend()
    # plt.show()

    # plt.plot(time_list,Hold_up_list,label='ENMPC')
    # plt.plot(time_list_open,Hold_up_list_open,label='Constant flows')
    # plt.xlabel('time [h]')
    # plt.ylabel('Hold-up [kg]')
    # plt.legend()
    # plt.show()

    # plt.plot(time_list,yeast_list,label='ENMPC')
    # plt.plot(time_list_open,yeast_list_open,label='Constant flows')
    # plt.xlabel('time [h]')
    # plt.ylabel('yeast [kg]')
    # plt.legend()
    # plt.show()
