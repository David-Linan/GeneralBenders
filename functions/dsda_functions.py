import copy
import csv
import itertools as it
import os
import time
from math import isnan
import math
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pe
from functions.model_serializer import StoreSpec, from_json, to_json
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
# from pyomo.contrib.gdpopt.data_class import MasterProblemResult
from pyomo.core.base.misc import display
from pyomo.core.plugins.transform.logical_to_linear import \
    update_boolean_vars_from_binary
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolutionStatus, SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.opt.base.solvers import SolverFactory
import pyomo.dae as dae
from copy import deepcopy

#TODO: this depends on each specific problem
def complementary_model(m,x):
    ' model that defines only required disjunctions. This can be only used with purely disjunctive methods such as DSDA or LDBD. This complementeary model must be included in external_ref function'
    current=-1
    for I in m.I_reactions:
        for J in m.J_reactors:
            current=current+1
            m.tau[I,J]=x[current]+m.minTau[I,J]-1
            m.tau_p[I,J]=pe.value(m.tau[I,J])*m.delta #Both times are assumed to be discrete
            # print(pe.value(m.tau_p[I,J]))
    # #----------- Variable processing times----------------------------------------------------------------
    def _DEF_VAR_TIME(m,I,J):
        return m.varTime[I,J]<=pe.value(m.tau_p[I,J])
    m.DEF_VAR_TIME=pe.Constraint(m.I_reactions,m.J_reactors,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
    # m.DEF_VAR_TIME.display()
    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------
    # TODO: The following equations make the disjunction require a lot of time to generate and therefore the model requires a lot of time to construct
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-pe.value(m.tau[I,J])+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-pe.value(m.tau[I,J])] for J in m.J if m.I_i_j_prod[I,J]==1 and T-pe.value(m.tau[I,J])>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')   




    m.DEF_Nref={}
    m.finalCon={}
    m.finalTemp={}
    for I_J in m.I_J:
        current=current+1
        I=I_J[0]
        J=I_J[1]  
        indexN= x[current]-1     
        def _DEF_Nref(m):
            return m.Nref[I,J]==indexN
        m.DEF_Nref[I,J]=pe.Constraint(rule=_DEF_Nref)
        setattr(m,'DEF_Nref_%s_%s' %(I,J),m.DEF_Nref[I,J])

        if I in m.I_reactions and J in m.J_reactors:
            # Final concentration constraint
            def _finalCon(m,N,Q):
                if N==m.N[I,J].last() and indexN!=0:
                    return m.Cvar[I,J][N,Q] == m.C_final[I,Q]
                else:
                    return pe.Constraint.Skip
            m.finalCon[I,J]=pe.Constraint(m.N[I,J],m.Q_balance[I],rule=_finalCon)
            setattr(m,'finalCon_%s_%s' %(I,J),m.finalCon[I,J])

            #Final temperature constraints                
            def _finalTemp(m,N):
                if N==m.N[I,J].last() and indexN!=0:
                    return m.TRvar[I,J][N]<= m.T_R_final[I]
                else:
                    return pe.Constraint.Skip
            m.finalTemp[I,J]=pe.Constraint(m.N[I,J],rule=_finalTemp)
            setattr(m,'finalTemp_%s_%s' %(I,J),m.finalTemp[I,J])
    return m

def complementary_model_for_sequential(m,x):
    ' model that defines only required disjunctions. This can be only used with purely disjunctive methods such as DSDA or LDBD. This complementeary model must be included in external_ref function'
    current=-1
    for I in m.I_reactions:
        for J in m.J_reactors:
            current=current+1
            m.tau[I,J]=x[current]+m.minTau[I,J]-1
            m.tau_p[I,J]=pe.value(m.tau[I,J])*m.delta #Both times are assumed to be discrete
            # print(pe.value(m.tau_p[I,J]))
    # #----------- Variable processing times----------------------------------------------------------------
    def _DEF_VAR_TIME(m,I,J):
        return m.varTime[I,J]==pe.value(m.tau_p[I,J])
    m.DEF_VAR_TIME=pe.Constraint(m.I_reactions,m.J_reactors,rule=_DEF_VAR_TIME,doc='Assignment of variable time value')
    # m.DEF_VAR_TIME.display()
    # # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------
    # TODO: The following equations make the disjunction require a lot of time to generate and therefore the model requires a lot of time to construct
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-pe.value(m.tau[I,J])+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1           
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-pe.value(m.tau[I,J])] for J in m.J if m.I_i_j_prod[I,J]==1 and T-pe.value(m.tau[I,J])>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)#-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')   

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

# ------------------------THIS VERSION IS VALID FOR SCHEDULING ONLY---------------------------------------------------------
# def external_ref(
#     m: pe.ConcreteModel(),
#     x,
#     extra_logic_function,
#     dict_extvar: dict = {},
#     mip_ref: bool = False,
#     transformation: str = 'bigm',
#     tee: bool = False
# ):
#     """
#     Function that
#     Args:
#         m: GDP model that is going to be reformulated
#         x: List with current value of the external variables
#         extra_logic_function: Function that returns a list of lists of the form [a,b], where a is an expressions of the reformulated Boolean variables and b is an equivalent Boolean or indicator variable (b<->a)
#         dict_extvar: A dictionary of dictionaries that looks as follows:
#             {1:{'exactly_number':Number of external variables for this type,
#                 'Boolean_vars_names':list with names of the ordered Boolean variables to be reformulated,
#                 'Boolean_vars_ordered_index': Indexes where the external reformulation is applied,
#                 'Binary_vars_names':list with names of the ordered Binary variables to be reformulated, [Potentially]
#                 'Binary_vars_ordered_index': Indexes where the external reformulation is applied, [Potentially]
#                 'Ext_var_lower_bound': Lower bound for this type of external variable,
#                 'Ext_var_upper_bound': Upper bound for this type of external variable },
#              2:{...},...}

#             The first key (positive integer) represent a type of external variable identified in the model. For this type of external variable
#             a dictionary is created.
#         mip_ref: whether the reformulation will consider binary variables besides Booleans coming from a GDP->MIP reformulation
#         tee: Display reformulation
#     Returns:
#         m: A model where the independent Boolean variables that were reformulated are fixed and Boolean/indicator variables that are calculated in
#         terms of the independent Boolean variables are fixed too (depending on the extra_logic_function provided by the user)

#     """
#     countt=-1
#     for I_J in m.I_J:
#         countt=countt+1
#         for N in m.Nref:
#             if N<=m.lastN[I_J]:
#                 m.Z[N,I_J].fix(False)
#                 if x[countt]-1 == N:
#                     m.Z[N,I_J].fix(True)
#     # Other Boolean and Indicator variables are fixed depending on the information provided by the user
#     logic_expr = extra_logic_function(m)
#     for i in logic_expr:
#         if not mip_ref:
#             i[1].fix(pe.value(i[0]))
#         else:
#             i[1].set_value(pe.value(i[0]))

#     pe.TransformationFactory('core.logical_to_linear').apply_to(m)
#     if mip_ref:  # Transform problem to MINLP
#         transformation_string = 'gdp.' + transformation
#         pe.TransformationFactory(transformation_string).apply_to(m)
#     else:  # Deactivate disjunction's constraints in the case of pure GDP
#         pe.TransformationFactory('gdp.fix_disjuncts').apply_to(m)

#     pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
#         m, tmp=False, ignore_infeasible=True)

#     if tee:
#         m.Z.pprint()
#     return m

#TODO: Define parameter to decide if a complementary_model(m,x) will be used
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

    #TODO: Generalize this, I am updating the portion of the model that depends on tau for scheduling. This is to avoid using very large models
    # m=complementary_model(m,x)
    #TODO
    #TODO
    #TODO
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


#this only keeps disjunctions and boolean variables within a neighborhood of the current point active. This version is not general
def external_ref_neighborhood(
    m: pe.ConcreteModel(),
    x,
    extra_logic_function,
    dict_extvar: dict = {},
    mip_ref: bool = False,
    transformation: str = 'bigm',
    tee: bool = False,
    feasibility_cuts: list=[],
    dynamic_vars: bool=True, # if True: solve approximate problem in a vecinity of dynamic problem (processing times) If false: solve approximate problem in a vecinity of scheduling problem (batching variables)
    neigh_size: int=1, # 1 means infinity neighborhood, 2 means second-nearest neighbors, etc
    interactions: int=1, #1 means neighbors with no interaction, 2 means neighbors with at most double interactions, etc. a very large value means all interactions.
    remove_cons: bool=False,
    eval_inner: bool=True
):
    """
    Function that
    Args:
        m: GDP model that is going to be reformulated
        x: List with current value of the external variables
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
        m: A model that restricts boolean variables within a neighborhood of x

    """

    num_ext=len(x) #number of external variables
    param_interaction=num_ext-interactions
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

    #First, fix everything to false
    for i in dict_extvar:
            for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                if not mip_ref:
                    dict_extvar[i]['Boolean_vars'][k-1].fix(False)
                else:
                    dict_extvar[i]['Binary_vars'][k-1].fix(0)
                    dict_extvar[i]['Boolean_vars'][k-1].set_value(False) #TODO: this line has not been tested


    # Now unfixt those variables that are within a neighborhood of external variables.
    ext_var_position = 0
    for i in dict_extvar:
        for j in range(dict_extvar[i]['exactly_number']):
            for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                if k>=x[ext_var_position]-neigh_size and k<=x[ext_var_position]+neigh_size: #If Boolean var is within a neighborhood of the current value of external variables           
                    if not mip_ref:
                        if dict_extvar[i]['Boolean_vars'][k-1].is_fixed():
                            dict_extvar[i]['Boolean_vars'][k-1].unfix()
                    else:
                        if dict_extvar[i]['Binary_vars'][k-1].is_fixed():
                            dict_extvar[i]['Binary_vars'][k-1].unfix()
                            dict_extvar[i]['Boolean_vars'][k-1].unfix() #TODO: this line has not been tested
            ext_var_position = ext_var_position+1


    if feasibility_cuts: # if there are feasibility cuts, add them
        m.feas_cut_con={}
        posit=0
        for avoid in feasibility_cuts:
            posit=posit+1
            avoid_list=[]
            ext_var_position = 0
            for i in dict_extvar:
                for j in range(dict_extvar[i]['exactly_number']):
                    avoid_list.append(dict_extvar[i]['Boolean_vars'][avoid[ext_var_position]-1])
                    ext_var_position = ext_var_position+1

            def feas_cut_rule(m):
                return pe.lnot(pe.land(avoid_list))
            m.feas_cut_con[posit]=pe.LogicalConstraint(rule=feas_cut_rule)   
            setattr(m,'feas_cut_con_%s' %str(posit),m.feas_cut_con[posit]) 
                
    
    #Constraint to only accept interactions between 2, 3 ,4... variables
    if param_interaction>=1:
        ext_var_position = 0
        exactly_list=[]
        for i in dict_extvar:
            for j in range(dict_extvar[i]['exactly_number']):
                for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                    if x[ext_var_position] == k:
                        exactly_list.append(dict_extvar[i]['Boolean_vars'][k-1])
                ext_var_position = ext_var_position+1

        def exactly_cont_rule(m):
            return pe.exactly(param_interaction,exactly_list)
        m.exactly_cont=pe.LogicalConstraint(rule=exactly_cont_rule)


    if eval_inner:
        if dynamic_vars:
            #Constraint to fix external variables related to scheduling desicions (N_i,j)
            ext_var_position = 0
            exactly_list=[]
            for i in dict_extvar:
                for j in range(dict_extvar[i]['exactly_number']):
                    for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                        if x[ext_var_position] == k and ext_var_position+1>=7:#TODO: GENERALIZE
                            # exactly_list.append(dict_extvar[i]['Boolean_vars'][k-1])
                            dict_extvar[i]['Boolean_vars'][k-1].fix(True)
                    ext_var_position = ext_var_position+1

            # def fix_sched_rule(m):
            #     return pe.exactly(10,exactly_list)
            # m.fix_sched=pe.LogicalConstraint(rule=fix_sched_rule)



            # DEACTIVATE SCHEDULING CONSTRAINTS
            # m.E2_CAPACITY_LOW.deactivate()
            # m.E2_CAPACITY_UP.deactivate()
            # m.E3_BALANCE_INIT.deactivate()
            # m.E_DEMAND_SATISFACTION.deactivate()
            # m.linking1.deactivate()
            # m.linking2.deactivate()
            # m.E1_UNIT.deactivate()
            # m.E3_BALANCE.deactivate()
            # m.X_Z_relation.deactivate()
            # m.DEF_AUX1_INDEP.deactivate()
            # m.DEF_AUX2_INDEP.deactivate()
        else:
            #Constraint to fix external variables related to processing times (tau_i,j)
            ext_var_position = 0
            exactly_list=[]
            for i in dict_extvar:
                for j in range(dict_extvar[i]['exactly_number']):
                    for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                        if x[ext_var_position] == k and ext_var_position+1<=6:#TODO: GENERALIZE
                            # exactly_list.append(dict_extvar[i]['Boolean_vars'][k-1])
                            dict_extvar[i]['Boolean_vars'][k-1].fix(True)
                    ext_var_position = ext_var_position+1

            # def fix_ttimes_rule(m):
            #     return pe.exactly(6,exactly_list)
            # m.fix_ttimes=pe.LogicalConstraint(rule=fix_ttimes_rule)
            
            # DEACTIVATE DYNAMIC CONSTRAINTS
            # for I in m.I_reactions:
            #     for J in m.J_reactors:
            #         m.c_dCdtheta[I,J].deactivate()
            #         m.c_dTRdtheta[I,J].deactivate()                        
            #         m.c_dTJdtheta[I,J].deactivate()
            #         m.c_dIntegral_hotdtheta[I,J].deactivate()
            #         m.c_dIntegral_colddtheta[I,J].deactivate()
            #         m.Constant_control1[I,J].deactivate()                        
            #         m.Constant_control2[I,J].deactivate()


    if remove_cons:
        # # DEACTIVATE DYNAMIC CONSTRAINTS
        # for I in m.I_reactions:
        #     for J in m.J_reactors:
        #         m.c_dCdtheta[I,J].deactivate()
        #         m.c_dTRdtheta[I,J].deactivate()                        
        #         m.c_dTJdtheta[I,J].deactivate()
        #         m.c_dIntegral_hotdtheta[I,J].deactivate()
        #         m.c_dIntegral_colddtheta[I,J].deactivate()
        #         m.Constant_control1[I,J].deactivate()                        
        #         m.Constant_control2[I,J].deactivate()       

        # # DEACTIVATE SCHEDULING CONSTRAINTS
        # m.E2_CAPACITY_LOW.deactivate()
        # m.E2_CAPACITY_UP.deactivate()
        # m.E3_BALANCE_INIT.deactivate()
        # m.E_DEMAND_SATISFACTION.deactivate()
        # m.linking1.deactivate()
        # m.linking2.deactivate()
        # m.E1_UNIT.deactivate()
        # m.E3_BALANCE.deactivate()
        # m.X_Z_relation.deactivate()
        # m.DEF_AUX1_INDEP.deactivate()
        # m.DEF_AUX2_INDEP.deactivate()

        #DEACTIVATE ALL CONSTRAINTS
        for consts in m.component_objects(pe.Constraint,descend_into=True):
            # print(consts.name)
            if consts.name =='C_TCP1' or consts.name =='C_TCP2' or consts.name =='C_TCP3' or consts.name =='C_TMC' or consts.name =='C_SALES':
                continue
            else:
                consts.deactivate()
        #

    # # Other Boolean and Indicator variables are fixed depending on the information provided by the user
    logic_expr = extra_logic_function(m)
    for i in logic_expr:
        if i[0].is_fixed():
            if not mip_ref:
                i[1].fix(pe.value(i[0]))
            else:
                i[1].set_value(pe.value(i[0]))#TODO: this line has not been tested
                
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    if mip_ref:  # Transform problem to MINLP
        transformation_string = 'gdp.' + transformation
        pe.TransformationFactory(transformation_string).apply_to(m)
    # else:  # Deactivate disjunction's constraints in the case of pure GDP
    #     pe.TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
        m, tmp=False, ignore_infeasible=True)


    # if tee:
    #     print('\nFixed variables at current iteration:\n')
    #     print('\n Independent Boolean variables\n')
    #     for i in dict_extvar:
    #         for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
    #             print(dict_extvar[i]['Boolean_vars_names'][k-1] +
    #                   '='+str(dict_extvar[i]['Boolean_vars'][k-1].value))

    #     print('\n Dependent Boolean variables and disjunctions\n')
    #     for i in logic_expr:
    #         print(i[1].name+'='+str(i[1].value))

    #     if mip_ref:
    #         print('\n Independent binary variables\n')
    #         for i in dict_extvar:
    #             for k in range(1, len(dict_extvar[i]['Binary_vars'])+1):
    #                 print(dict_extvar[i]['Binary_vars_names'][k-1] +
    #                       '='+str(dict_extvar[i]['Binary_vars'][k-1].value))

    return m
#this only keeps disjunctions and boolean variables within a neighborhood of the current point active. This version is more general
def external_ref_neighborhood_general(
    m: pe.ConcreteModel(),
    x,
    extra_logic_function,
    dict_extvar: dict = {},
    feasibility_cuts: list=[],
    neigh_size: int=1, # 1 means infinity neighborhood, 2 means second-nearest neighbors, etc
    limit_interactions=False, #True if the optimized value of Boolean variables within a neighborhood will be limited to consider only some interactions. See "interactions"
    interactions: int=1, #1 means neighbors with no interaction, 2 means neighbors with at most double interactions, etc. Must be greater or equal to 1
):
    """
    Function that
    Args:
        m: GDP model that is going to be reformulated
        x: List with current value of the external variables
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
        m: A model that restricts boolean variables within a neighborhood of x

    """

    num_ext=len(x) #number of external variables
    param_interaction=num_ext-interactions
    # This part of code is required due to the deep copy issue: we have to compare Boolean variables by name
    for i in dict_extvar:
        dict_extvar[i]['Boolean_vars'] = []
        for j in dict_extvar[i]['Boolean_vars_names']:
            for boolean in m.component_data_objects(pe.BooleanVar, descend_into=True):
                if(boolean.name == j):
                    dict_extvar[i]['Boolean_vars'] = dict_extvar[i]['Boolean_vars']+[boolean]


# The function would start here if there were no problems with deep copy.

    ext_var_position = 0
    for i in dict_extvar:
        for j in range(dict_extvar[i]['exactly_number']):
            for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                if k<x[ext_var_position]-neigh_size or k>x[ext_var_position]+neigh_size:            
                    if not dict_extvar[i]['Boolean_vars'][k-1].is_fixed():
                        dict_extvar[i]['Boolean_vars'][k-1].fix(False) #fix all variables outside current neighborhood to false
                else:
                    if dict_extvar[i]['Boolean_vars'][k-1].is_fixed():
                        dict_extvar[i]['Boolean_vars'][k-1].unfix()                   
            ext_var_position = ext_var_position+1



    if feasibility_cuts: # if there are feasibility cuts, add them
        m.feas_cut_con={}
        posit=0
        for avoid in feasibility_cuts:
            posit=posit+1
            avoid_list=[]
            ext_var_position = 0
            for i in dict_extvar:
                for j in range(dict_extvar[i]['exactly_number']):
                    avoid_list.append(dict_extvar[i]['Boolean_vars'][avoid[ext_var_position]-1])
                    ext_var_position = ext_var_position+1

            def feas_cut_rule(m):
                return pe.lnot(pe.land(avoid_list))
            m.feas_cut_con[posit]=pe.LogicalConstraint(rule=feas_cut_rule)   
            setattr(m,'feas_cut_con_%s' %str(posit),m.feas_cut_con[posit]) 
                
    
    #Constraint to only accept interactions between 2, 3 ,4... variables
    if limit_interactions==True and param_interaction>=1:
        ext_var_position = 0
        exactly_list=[]
        for i in dict_extvar:
            for j in range(dict_extvar[i]['exactly_number']):
                for k in range(1, len(dict_extvar[i]['Boolean_vars'])+1):
                    if x[ext_var_position] == k:
                        exactly_list.append(dict_extvar[i]['Boolean_vars'][k-1])
                ext_var_position = ext_var_position+1

        def exactly_cont_rule(m):
            return pe.exactly(param_interaction,exactly_list)
        m.exactly_cont=pe.LogicalConstraint(rule=exactly_cont_rule)


    # # Other Boolean and Indicator variables are fixed depending on the information provided by the user
    logic_expr = extra_logic_function(m)
    for i in logic_expr:
        if i[0].is_fixed():
            i[1].fix(pe.value(i[0]))

                
    # pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    # pe.TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)
    # pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m, tmp=False, ignore_infeasible=True)

    return m



#for sequential scheduling and control
def external_ref_sequential(
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

    #TODO: Generalize this, I am updating the portion of the model that depends on tau for scheduling. This is to avoid using very large models
    # m=complementary_model(m,x)
    #for sequential:
    m=complementary_model_for_sequential(m,x)
    #TODO
    #TODO
    #TODO
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

#for sequential scheduling and control
def external_ref_sequential_case_2(
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

def extvars_gdp_to_mip(
    m: pe.ConcreteModel(),
    gdp_dict_extvar: dict = {},
    transformation: str = 'bigm',
):
    """
    Function that
    Args:
        m: GDP model that is going to be reformulated
        gdp_dict_extvar: A dictionary of dictionaries that looks as follows:
            {1:{'exactly_number':Number of external variables for this type,
                'Boolean_vars_names':list with names of the ordered Boolean variables to be reformulated,
                'Boolean_vars_ordered_index': Indexes where the external reformulation is applied,
                'Ext_var_lower_bound': Lower bound for this type of external variable,
                'Ext_var_upper_bound': Upper bound for this type of external variable },
             2:{...},...}
        transformation: GDP to MINLP transformation to be used

            The first key (positive integer) represent a type of external variable identified in the model. For this type of external variable
            a dictionary is created.
        tee: Display reformulation
    Returns:
        m: A MIP model transformed from the original GDP m model via the 'transformation' argument
        mip_dict_extvar: A dictionary of dictionaries that looks as follows:
            {1:{'exactly_number':Number of external variables for this type,
                'Boolean_vars_names':list with names of the ordered Boolean variables to be reformulated,
                'Boolean_vars_ordered_index': Indexes where the external reformulation is applied,
                'Binary_vars_names':list with names of the ordered Binary variables to be reformulated,
                'Binary_vars_ordered_index': Indexes where the external reformulation is applied,
                'Ext_var_lower_bound': Lower bound for this type of external variable,
                'Ext_var_upper_bound': Upper bound for this type of external variable },
             2:{...},...}

    """

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    transformation_string = 'gdp.' + transformation
    pe.TransformationFactory(transformation_string).apply_to(m)

    mip_dict_extvar = copy.deepcopy(gdp_dict_extvar)

    # This part of code is required due to the deep copy issue: we have to compare Boolean variables by name
    for i in mip_dict_extvar.keys():
        mip_dict_extvar[i]['Boolean_vars'] = []
        for j in mip_dict_extvar[i]['Boolean_vars_names']:
            for boolean in m.component_data_objects(pe.BooleanVar, descend_into=True):
                if(boolean.name == j):
                    mip_dict_extvar[i]['Boolean_vars'] = mip_dict_extvar[i]['Boolean_vars']+[boolean]
        # Add extra terms to the dictionary to be relevant for binary variables
        mip_dict_extvar[i]['Binary_vars_names'] = [
            boolean.get_associated_binary().name for boolean in mip_dict_extvar[i]['Boolean_vars']]
        # Uncomment the next line in case that deepcopy works
        # mip_dict_extvar[key]['Binary_vars'] = [
        #     boolean.get_associated_binary() for boolean in gdp_dict_extvar[key]['Boolean_vars']]
        mip_dict_extvar[i]['Binary_vars_ordered_index'] = mip_dict_extvar[i]['Boolean_vars_ordered_index']

    return m, mip_dict_extvar

#TODO: fbbt seems to be making the algorithm slower! I have commented this line.
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
    else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
        m.dsda_status = 'Optimal'
    # if m.results.solver.termination_condition == 'locallyOptimal' or m.results.solver.termination_condition == 'optimal' or m.results.solver.termination_condition == 'globallyOptimal':
    #     m.dsda_status = 'Optimal'

    return m

#TODO: GENERALIZE. APPROXIMATE SOLUTIO OF SUBPROBLEMS.
def solve_subproblem_aprox(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 0,
    best_sol: float= 1e+8,
    new_case: bool=False, # If algorithm will be used for new case study involving kondili STN
    with_distillation: bool=False  #If modified model with distillation dynamics will be considered  
) -> pe.ConcreteModel():
    """
    Function that checks feasibility and optimizes subproblem model.
    Note integer variables have to be previously fixed in the external reformulation.
    This function is for problems that can be decoupled into a steady-state (or scheduling) and a dynamic part
    
    Args:

    Returns:

    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        # start=time.time()
        preprocess_problem(m, simple=True)
        # end=time.time()
        # print('preprocess_time: ',str(end-start))
    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        m.pruned_Status = 'FBBT_Infeasible'
        m.best_sol=1e+8
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

    approximate_solution=False# If true, after solving lower bounding scheduling problem, then scheduling variables are fixed and NLP control problem is solved
                                # If false, then lower bounding scheduling is solved first, and then original minlp subproblem is solved, i.e., this is actually what we have called the enhanced dsda
    scheduling_only=False #True: only perform scheduling subproblems
    
    add_cutoff=False # If true, cplex cutoff option is activated when solving subproblems approximately. This means that solutions with objective function worst than current incumbent solution are considered as infeasible
    #### MODIFICATIONS FROM HERE WITH RESPECT TO ORIGINAL FUNCTION ################################    
    #DEACTIVATE DYNAMIC CONSTRAINTS

    #CHECK
    # for consts in m.component_data_objects(pe.Constraint,descend_into=True):
    #     if consts.body.polynomial_degree()>=2: #Polynomial degree greater or equal to 2 means nonlinear constraint
    #         consts.deactivate() 

    if new_case:
        for I in m.I_dynamics:
            for J in m.J_dynamics:
                for T in m.T:
                    m.c_defCT0[I,J,T].deactivate()
                    m.c_dCAdtheta[I,J,T].deactivate() 
                    m.c_dCBdtheta[I,J,T].deactivate() 
                    m.c_dCCdtheta[I,J,T].deactivate() 
                    m.c_dVdtheta[I,J,T].deactivate() 
                    m.c_dTRdtheta[I,J,T].deactivate() 
                    m.c_dTJdtheta[I,J,T].deactivate() 
                    m.c_dIntegral_hotdtheta[I,J,T].deactivate() 
                    m.c_dIntegral_colddtheta[I,J,T].deactivate() 
                    m.Constant_control1[I,J,T].deactivate() 
                    m.Constant_control2[I,J,T].deactivate() 
                    m.Constant_control3[I,J,T].deactivate() 
        m.C_TCP3.deactivate()              
        m.obj.deactivate()
        m.obj_dummy.deactivate()
        m.obj_scheduling.activate()  
        if with_distillation:
                for I in m.I_distil:
                    for J in m.J_distil:
                        for T in m.T:   
                            for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                cons.deactivate()
    else:
        for I in m.I_reactions:
            for J in m.J_reactors:
                m.c_dCdtheta[I,J].deactivate()
                m.c_dTRdtheta[I,J].deactivate()                        
                m.c_dTJdtheta[I,J].deactivate()
                m.c_dIntegral_hotdtheta[I,J].deactivate()
                m.c_dIntegral_colddtheta[I,J].deactivate()
                m.Constant_control1[I,J].deactivate()                        
                m.Constant_control2[I,J].deactivate()
        m.C_TCP3.deactivate()
        m.obj.deactivate()
        m.obj_scheduling.activate()
        m.obj_dummy.deactivate()


    subproblem_solver_options_2=deepcopy(subproblem_solver_options)

    if approximate_solution and add_cutoff and best_sol!=1e+8:
        subproblem_solver_options_2['add_options'].append('$onecho > cplex.opt \ncutup %s \n$offecho \n' % best_sol )
    #SOLVE SCHEDULING ONLY PROBLEM
    opt = SolverFactory(solvername, solver='cplex')

    # start=time.time()
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options_2,
                          skip_trivial_constraints=True,
                          )
    # end=time.time()
    # print('actual sol time part 1:',str(end-start))



    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time


    if scheduling_only:
        m.obj.activate()
        m.obj.value=pe.value(m.obj_scheduling)
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.pruned_Status = 'Evaluated_Infeasible_NotPruned'
            m.best_sol=1e+8
        else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
            m.dsda_status = 'Optimal'
            m.pruned_Status = 'Optimal_NotPruned'
            m.best_sol=pe.value(m.obj)        
    else: 
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.pruned_Status = 'Pruned_SchedulingInfeasible'
            m.best_sol=1e+8
            m.obj.activate()
            m.obj.value=1e+8
        else:
            if pe.value(m.obj_scheduling)>=best_sol:
                m.dsda_status = 'Evaluated_Infeasible'
                m.pruned_Status = 'Pruned_NoImprovementExpected'
                m.best_sol=1e+8
                m.obj.activate()
                m.obj.value=pe.value(m.obj_scheduling)
            else:  
                # ACTIVATE DYNAMIC CONSTRAINTS
                if new_case:
                    for I in m.I_dynamics:
                        for J in m.J_dynamics:
                            for T in m.T:
                                m.c_defCT0[I,J,T].activate()
                                m.c_dCAdtheta[I,J,T].activate() 
                                m.c_dCBdtheta[I,J,T].activate() 
                                m.c_dCCdtheta[I,J,T].activate() 
                                m.c_dVdtheta[I,J,T].activate() 
                                m.c_dTRdtheta[I,J,T].activate() 
                                m.c_dTJdtheta[I,J,T].activate() 
                                m.c_dIntegral_hotdtheta[I,J,T].activate() 
                                m.c_dIntegral_colddtheta[I,J,T].activate() 
                                m.Constant_control1[I,J,T].activate() 
                                m.Constant_control2[I,J,T].activate() 
                                m.Constant_control3[I,J,T].activate() 
                    m.C_TCP3.activate()              
                    m.obj.activate()
                    m.obj_dummy.deactivate()
                    m.obj_scheduling.deactivate()  

                    if with_distillation:
                        for I in m.I_distil:
                            for J in m.J_distil:
                                for T in m.T:   
                                    for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                        cons.activate()                     
                else:
                    for I in m.I_reactions:
                        for J in m.J_reactors:
                            m.c_dCdtheta[I,J].activate()
                            m.c_dTRdtheta[I,J].activate()                        
                            m.c_dTJdtheta[I,J].activate()
                            m.c_dIntegral_hotdtheta[I,J].activate()
                            m.c_dIntegral_colddtheta[I,J].activate()
                            m.Constant_control1[I,J].activate()                        
                            m.Constant_control2[I,J].activate()
                    m.C_TCP3.activate()
                    m.obj.activate()
                    m.obj_scheduling.deactivate() 
                    m.obj_dummy.deactivate()



                if approximate_solution:
                    # FIX SCHEDULING VARIABLES
                    for v in m.component_objects(pe.Var, descend_into=True):
                        if v.name=='X' or v.name=='Nref':
                            for index in v:
                                if index==None:
                                    v.fix(round(pe.value(v)))
                                else:
                                    v[index].fix(round(pe.value(v[index])))

                        # elif v.name=='Vreactor' or v.name=='B' or v.name=='S' or v.name=='varTime':
                        #     for index in v:
                        #         if index==None:
                        #             v.fix(pe.value(v))
                        #         else:
                        #             v[index].fix(pe.value(v[index]))


                opt = SolverFactory(solvername, solver=subproblem_solver)
                # start=time.time()
                m.results = opt.solve(m, tee=tee,
                                    **output_options,
                                    **subproblem_solver_options,
                                    skip_trivial_constraints=True,
                                    )
                # end=time.time()
                # print('actual sol time part 2=',str(end-start))

                m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time

                #### END OF MODIFICATIONS #######################################################################


                # Assign D-SDA status
                if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                    m.dsda_status = 'Evaluated_Infeasible'
                    m.pruned_Status = 'Evaluated_Infeasible_NotPruned'
                    m.best_sol=1e+8
                else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                    m.dsda_status = 'Optimal'
                    m.pruned_Status = 'Optimal_NotPruned'
                    m.best_sol=pe.value(m.obj)
    return m

def solve_subproblem_aprox_fix_all_scheduling(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 0,
    best_sol: float= 1e+8,
    new_case: bool=False, # If algorithm will be used for new case study involving kondili STN
    with_distillation: bool=False  #If modified model with distillation dynamics will be considered  
) -> pe.ConcreteModel():
    """
    Function that checks feasibility and optimizes subproblem model.
    Note integer variables have to be previously fixed in the external reformulation.
    This function is for problems that can be decoupled into a steady-state (or scheduling) and a dynamic part
    
    Args:

    Returns:

    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        # start=time.time()
        preprocess_problem(m, simple=True)
        # end=time.time()
        # print('preprocess_time: ',str(end-start))
    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        m.pruned_Status = 'FBBT_Infeasible'
        m.best_sol=1e+8
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

    approximate_solution=True# If true, after solving lower bounding scheduling problem, then scheduling variables are fixed and NLP control problem is solved
                                # If false, then lower bounding scheduling is solved first, and then original minlp subproblem is solved, i.e., this is actually what we have called the enhanced dsda
    scheduling_only=False #True: only perform scheduling subproblems
    
    add_cutoff=True # If true, cplex cutoff option is activated when solving subproblems approximately. This means that solutions with objective function worst than current incumbent solution are considered as infeasible
    #### MODIFICATIONS FROM HERE WITH RESPECT TO ORIGINAL FUNCTION ################################    
    #DEACTIVATE DYNAMIC CONSTRAINTS

    #CHECK
    # for consts in m.component_data_objects(pe.Constraint,descend_into=True):
    #     if consts.body.polynomial_degree()>=2: #Polynomial degree greater or equal to 2 means nonlinear constraint
    #         consts.deactivate() 

    if new_case:
        for I in m.I_dynamics:
            for J in m.J_dynamics:
                for T in m.T:
                    m.c_defCT0[I,J,T].deactivate()
                    m.c_dCAdtheta[I,J,T].deactivate() 
                    m.c_dCBdtheta[I,J,T].deactivate() 
                    m.c_dCCdtheta[I,J,T].deactivate() 
                    m.c_dVdtheta[I,J,T].deactivate() 
                    m.c_dTRdtheta[I,J,T].deactivate() 
                    m.c_dTJdtheta[I,J,T].deactivate() 
                    m.c_dIntegral_hotdtheta[I,J,T].deactivate() 
                    m.c_dIntegral_colddtheta[I,J,T].deactivate() 
                    m.Constant_control1[I,J,T].deactivate() 
                    m.Constant_control2[I,J,T].deactivate() 
                    m.Constant_control3[I,J,T].deactivate() 
        m.C_TCP3.deactivate()              
        m.obj.deactivate()
        m.obj_dummy.deactivate()
        m.obj_scheduling.activate()  
        if with_distillation:
                for I in m.I_distil:
                    for J in m.J_distil:
                        for T in m.T:   
                            for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                cons.deactivate()
    else:
        for I in m.I_reactions:
            for J in m.J_reactors:
                m.c_dCdtheta[I,J].deactivate()
                m.c_dTRdtheta[I,J].deactivate()                        
                m.c_dTJdtheta[I,J].deactivate()
                m.c_dIntegral_hotdtheta[I,J].deactivate()
                m.c_dIntegral_colddtheta[I,J].deactivate()
                m.Constant_control1[I,J].deactivate()                        
                m.Constant_control2[I,J].deactivate()
        m.C_TCP3.deactivate()
        m.obj.deactivate()
        m.obj_scheduling.activate()
        m.obj_dummy.deactivate()


    subproblem_solver_options_2=deepcopy(subproblem_solver_options)

    if approximate_solution and add_cutoff and best_sol!=1e+8:
        subproblem_solver_options_2['add_options'].append('$onecho > cplex.opt \ncutup %s \n$offecho \n' % best_sol )
    #SOLVE SCHEDULING ONLY PROBLEM
    opt = SolverFactory(solvername, solver='cplex')

    # start=time.time()
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options_2,
                          skip_trivial_constraints=True,
                          )
    # end=time.time()
    # print('actual sol time part 1:',str(end-start))



    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time


    if scheduling_only:
        m.obj.activate()
        m.obj.value=pe.value(m.obj_scheduling)
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.pruned_Status = 'Evaluated_Infeasible_NotPruned'
            m.best_sol=1e+8
        else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
            m.dsda_status = 'Optimal'
            m.pruned_Status = 'Optimal_NotPruned'
            m.best_sol=pe.value(m.obj)        
    else: 
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.pruned_Status = 'Pruned_SchedulingInfeasible'
            m.best_sol=1e+8
            m.obj.activate()
            m.obj.value=1e+8
        else:
            if pe.value(m.obj_scheduling)>=best_sol:
                m.dsda_status = 'Evaluated_Infeasible'
                m.pruned_Status = 'Pruned_NoImprovementExpected'
                m.best_sol=1e+8
                m.obj.activate()
                m.obj.value=pe.value(m.obj_scheduling)
            else:  
                # ACTIVATE DYNAMIC CONSTRAINTS
                if new_case:
                    for I in m.I_dynamics:
                        for J in m.J_dynamics:
                            for T in m.T:
                                m.c_defCT0[I,J,T].activate()
                                m.c_dCAdtheta[I,J,T].activate() 
                                m.c_dCBdtheta[I,J,T].activate() 
                                m.c_dCCdtheta[I,J,T].activate() 
                                m.c_dVdtheta[I,J,T].activate() 
                                m.c_dTRdtheta[I,J,T].activate() 
                                m.c_dTJdtheta[I,J,T].activate() 
                                m.c_dIntegral_hotdtheta[I,J,T].activate() 
                                m.c_dIntegral_colddtheta[I,J,T].activate() 
                                m.Constant_control1[I,J,T].activate() 
                                m.Constant_control2[I,J,T].activate() 
                                m.Constant_control3[I,J,T].activate() 
                    m.C_TCP3.activate()              
                    m.obj.activate()
                    m.obj_dummy.deactivate()
                    m.obj_scheduling.deactivate()  

                    if with_distillation:
                        for I in m.I_distil:
                            for J in m.J_distil:
                                for T in m.T:   
                                    for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                        cons.activate()                     
                else:
                    for I in m.I_reactions:
                        for J in m.J_reactors:
                            m.c_dCdtheta[I,J].activate()
                            m.c_dTRdtheta[I,J].activate()                        
                            m.c_dTJdtheta[I,J].activate()
                            m.c_dIntegral_hotdtheta[I,J].activate()
                            m.c_dIntegral_colddtheta[I,J].activate()
                            m.Constant_control1[I,J].activate()                        
                            m.Constant_control2[I,J].activate()
                    m.C_TCP3.activate()
                    m.obj.activate()
                    m.obj_scheduling.deactivate() 
                    m.obj_dummy.deactivate()



                if approximate_solution:
                    # FIX SCHEDULING VARIABLES
                    for v in m.component_objects(pe.Var, descend_into=True):
                        if v.name=='X' or v.name=='Nref':
                            for index in v:
                                if index==None:
                                    v.fix(round(pe.value(v)))
                                else:
                                    v[index].fix(round(pe.value(v[index])))

                        elif v.name=='Vreactor' or v.name=='B' or v.name=='S' or v.name=='varTime':
                            for index in v:
                                if index==None:
                                    v.fix(pe.value(v))
                                else:
                                    v[index].fix(pe.value(v[index]))


                opt = SolverFactory(solvername, solver=subproblem_solver)
                # start=time.time()
                m.results = opt.solve(m, tee=tee,
                                    **output_options,
                                    **subproblem_solver_options,
                                    skip_trivial_constraints=True,
                                    )
                # end=time.time()
                # print('actual sol time part 2=',str(end-start))

                m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time

                #### END OF MODIFICATIONS #######################################################################


                # Assign D-SDA status
                if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                    m.dsda_status = 'Evaluated_Infeasible'
                    m.pruned_Status = 'Evaluated_Infeasible_NotPruned'
                    m.best_sol=1e+8
                else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                    m.dsda_status = 'Optimal'
                    m.pruned_Status = 'Optimal_NotPruned'
                    m.best_sol=pe.value(m.obj)
    return m


def solve_subproblem_aprox_tau_only(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 0,
    best_sol: float= 1e+8
) -> pe.ConcreteModel():
    """
    Function that checks feasibility and optimizes subproblem model.
    Note integer variables have to be previously fixed in the external reformulation.
    This function is for problems that can be decoupled into a steady-state (or scheduling) and a dynamic part
    
    Args:

    Returns:

    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        # start=time.time()
        preprocess_problem(m, simple=True)
        # end=time.time()
        # print('preprocess_time: ',str(end-start))
    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        m.pruned_Status = 'FBBT_Infeasible'
        m.best_sol=1e+8
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

    approximate_solution=True# If true, after solving lower bounding scheduling problem, then scheduling variables are fixed and NLP control problem is solved
                                # If false, then lower bounding scheduling is solved first, and then original minlp subproblem is solved, i.e., this is actually what we have called the enhanced dsda
    scheduling_only=False #True: only perform scheduling subproblems
    #### MODIFICATIONS FROM HERE WITH RESPECT TO ORIGINAL FUNCTION ################################    
    #DEACTIVATE DYNAMIC CONSTRAINTS
    for I in m.I_reactions:
        for J in m.J_reactors:
            m.c_dCdtheta[I,J].deactivate()
            m.c_dTRdtheta[I,J].deactivate()                        
            m.c_dTJdtheta[I,J].deactivate()
            m.c_dIntegral_hotdtheta[I,J].deactivate()
            m.c_dIntegral_colddtheta[I,J].deactivate()
            m.Constant_control1[I,J].deactivate()                        
            m.Constant_control2[I,J].deactivate()
            m.finalCon[I,J].deactivate()
            m.finalTemp[I,J].deactivate()
    m.C_TCP3.deactivate()
    m.obj.deactivate()
    m.obj_scheduling.activate()


    #SOLVE SCHEDULING ONLY PROBLEM
    opt = SolverFactory(solvername, solver='cplex')

    # start=time.time()
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options,
                          skip_trivial_constraints=True,
                          )
    # end=time.time()
    # print('actual sol time part 1:',str(end-start))

    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time


    if scheduling_only:
        m.obj.activate()
        m.obj.value=pe.value(m.obj_scheduling)
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.pruned_Status = 'Evaluated_Infeasible_NotPruned'
            m.best_sol=1e+8
        else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
            m.dsda_status = 'Optimal'
            m.pruned_Status = 'Optimal_NotPruned'
            m.best_sol=pe.value(m.obj)        
    else: 
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.pruned_Status = 'Pruned_SchedulingInfeasible'
            m.best_sol=1e+8
        else:
            if pe.value(m.obj_scheduling)>=best_sol:
                m.dsda_status = 'Evaluated_Infeasible'
                m.pruned_Status = 'Pruned_NoImprovementExpected'
                m.best_sol=1e+8
            else:  
                # ACTIVATE DYNAMIC CONSTRAINTS
                for I in m.I_reactions:
                    for J in m.J_reactors:
                        m.c_dCdtheta[I,J].activate()
                        m.c_dTRdtheta[I,J].activate()                        
                        m.c_dTJdtheta[I,J].activate()
                        m.c_dIntegral_hotdtheta[I,J].activate()
                        m.c_dIntegral_colddtheta[I,J].activate()
                        m.Constant_control1[I,J].activate()                        
                        m.Constant_control2[I,J].activate()
                        if round(pe.value(m.Nref[I,J]))>=1:
                            m.finalCon[I,J].activate()
                            m.finalTemp[I,J].activate() 
                m.C_TCP3.activate()
                m.obj.activate()
                m.obj_scheduling.deactivate() 

                if approximate_solution:
                    # FIX SCHEDULING VARIABLES
                    for v in m.component_objects(pe.Var, descend_into=True):
                        # if v.name=='X' or v.name=='B' or v.name=='S' or v.name=='Nref':
                        if v.name=='X' or v.name=='Nref':
                            for index in v:
                                if index==None:
                                    v.fix(round(pe.value(v)))
                                else:
                                    v[index].fix(round(pe.value(v[index])))

                opt = SolverFactory(solvername, solver=subproblem_solver)
                # start=time.time()
                m.results = opt.solve(m, tee=tee,
                                    **output_options,
                                    **subproblem_solver_options,
                                    skip_trivial_constraints=True,
                                    )
                # end=time.time()
                # print('actual sol time part 2=',str(end-start))

                m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time

                #### END OF MODIFICATIONS #######################################################################


                # Assign D-SDA status
                if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                    m.dsda_status = 'Evaluated_Infeasible'
                    m.pruned_Status = 'Evaluated_Infeasible_NotPruned'
                    m.best_sol=1e+8
                else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
                    m.dsda_status = 'Optimal'
                    m.pruned_Status = 'Optimal_NotPruned'
                    m.best_sol=pe.value(m.obj)
    return m


def solve_subproblem_aprox_sequential(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 1e-3
) -> pe.ConcreteModel():
    """
    Function that solves scheduling and then solves control. In scheduling is infeasible control is not solved.
    Note integer variables have to be previously fixed in the external reformulation.
    This function is for problems that can be decoupled into a steady-state (or scheduling) and a dynamic part
    
    Args:

    Returns:

    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        # start=time.time()
        preprocess_problem(m, simple=True)
        # end=time.time()
        # print('preprocess_time: ',str(end-start))
    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        m.best_sol=1e+8
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

    approximate_solution=True# If true, after solving lower bounding scheduling problem, then scheduling variables are fixed and NLP control problem is solved
                                # If false, then lower bounding scheduling is solved first, and then original minlp subproblem is solved, i.e., this is actually what we have called the enhanced dsda
    scheduling_only=False
    #### MODIFICATIONS FROM HERE WITH RESPECT TO ORIGINAL FUNCTION ################################    
    #DEACTIVATE DYNAMIC CONSTRAINTS
    for I in m.I_reactions:
        for J in m.J_reactors:
            m.c_dCdtheta[I,J].deactivate()
            m.c_dTRdtheta[I,J].deactivate()                        
            m.c_dTJdtheta[I,J].deactivate()
            m.c_dIntegral_hotdtheta[I,J].deactivate()
            m.c_dIntegral_colddtheta[I,J].deactivate()
            m.Constant_control1[I,J].deactivate()                        
            m.Constant_control2[I,J].deactivate()
            m.finalCon[I,J].deactivate()
            m.finalTemp[I,J].deactivate()
    m.C_TCP3.deactivate()
    m.obj.deactivate()
    m.obj_dummy.deactivate()
    m.obj_scheduling.activate()


    #SOLVE SCHEDULING ONLY PROBLEM
    opt = SolverFactory(solvername, solver='cplex')

    # start=time.time()
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options,
                          skip_trivial_constraints=True,
                          )
    # end=time.time()
    # print('actual sol time part 1:',str(end-start))

    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time


    if scheduling_only:
        m.obj.activate()
        m.obj.value=pe.value(m.obj_scheduling)
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.best_sol=1e+8
            m.sched_Status="Scheduling_Infeasible"
        else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
            m.dsda_status = 'Optimal'
            m.best_sol=pe.value(m.obj)
            m.sched_Status="Scheduling_Feasible" 
            m.cont_Status="Not_Evaluated"       
    else: 
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.best_sol=1e+8
            m.sched_Status="Scheduling_Infeasible"
            m.cont_Status="Not_evaluated"
            m.source={}
            for I in m.I_reactions:
                for J in m.J_reactors:
                    m.source[I,J]="Not_evaluated"
        else:
             
            # ACTIVATE DYNAMIC CONSTRAINTS
            for I in m.I_reactions:
                for J in m.J_reactors:
                    m.c_dCdtheta[I,J].activate()
                    m.c_dTRdtheta[I,J].activate()                        
                    m.c_dTJdtheta[I,J].activate()
                    m.c_dIntegral_hotdtheta[I,J].activate()
                    m.c_dIntegral_colddtheta[I,J].activate()
                    m.Constant_control1[I,J].activate()                        
                    m.Constant_control2[I,J].activate()
                    if round(pe.value(m.Nref[I,J]))>=1:
                        m.finalCon[I,J].activate()
                        m.finalTemp[I,J].activate()                        
            m.C_TCP3.activate()
            m.obj.activate()
            m.obj_dummy.deactivate()
            m.obj_scheduling.deactivate() 

            # DEACTIVATE SCHEDULING CONSTRAINTS
            m.E2_CAPACITY_LOW.deactivate()
            m.E2_CAPACITY_UP.deactivate()
            m.E3_BALANCE_INIT.deactivate()
            m.E_DEMAND_SATISFACTION.deactivate()
            m.linking1.deactivate()
            m.linking2.deactivate()
            m.E1_UNIT.deactivate()
            m.E3_BALANCE.deactivate()
            m.X_Z_relation.deactivate()
            m.DEF_VAR_TIME.deactivate()            




            if approximate_solution:
                # FIX SCHEDULING VARIABLES
                for v in m.component_objects(pe.Var, descend_into=True):
                    if v.name=='X' or v.name=='Nref':
                        for index in v:
                            if index==None:
                                v.fix(round(pe.value(v)))
                            else:
                                v[index].fix(round(pe.value(v[index])))
                    elif v.name=='Vreactor' or v.name=='B' or v.name=='S' or v.name=='varTime':
                        for index in v:
                            if index==None:
                                v.fix(pe.value(v))
                            else:
                                v[index].fix(pe.value(v[index]))

            opt = SolverFactory(solvername, solver=subproblem_solver)
            # start=time.time()
            m.results = opt.solve(m, tee=tee,
                                **output_options,
                                **subproblem_solver_options,
                                skip_trivial_constraints=True,
                                )
            # end=time.time()
            # print('actual sol time part 2=',str(end-start))

            m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time
            m.sched_Status="Scheduling_Feasible"
            #### END OF MODIFICATIONS #######################################################################


            # Assign D-SDA status
            if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                m.dsda_status = 'Evaluated_Infeasible'
                m.best_sol=1e+8
                m.cont_Status="Dynamcis_Infeasible"
                # If dynamics are infeasible, we identify the source of infeasibility
                m.source={}
                for II in m.I_reactions:
                    for JJ in m.J_reactors:     
                        if round(pe.value(m.Nref[II,JJ]))>=1:
                            for I in m.I_reactions:
                                for J in m.J_reactors:
                                    m.c_dCdtheta[I,J].deactivate()
                                    m.c_dTRdtheta[I,J].deactivate()                        
                                    m.c_dTJdtheta[I,J].deactivate()
                                    m.c_dIntegral_hotdtheta[I,J].deactivate()
                                    m.c_dIntegral_colddtheta[I,J].deactivate()
                                    m.Constant_control1[I,J].deactivate()                        
                                    m.Constant_control2[I,J].deactivate()
                                    m.finalCon[I,J].deactivate()
                                    m.finalTemp[I,J].deactivate()
                            m.C_TCP3.deactivate()
                            m.obj.deactivate()
                            m.obj_scheduling.deactivate() 

                            m.c_dCdtheta[II,JJ].activate()
                            m.c_dTRdtheta[II,JJ].activate()                        
                            m.c_dTJdtheta[II,JJ].activate()
                            m.c_dIntegral_hotdtheta[II,JJ].activate()
                            m.c_dIntegral_colddtheta[II,JJ].activate()
                            m.Constant_control1[II,JJ].activate()                        
                            m.Constant_control2[II,JJ].activate()
                            m.finalCon[II,JJ].activate()
                            m.finalTemp[II,JJ].activate() 
                            m.obj_dummy.activate()

                            if approximate_solution:
                                # FIX SCHEDULING VARIABLES
                                for v in m.component_objects(pe.Var, descend_into=True):
                                    if v.name=='X' or v.name=='Nref':
                                        for index in v:
                                            if index==None:
                                                v.fix(round(pe.value(v)))
                                            else:
                                                v[index].fix(round(pe.value(v[index])))
                                    elif v.name=='Vreactor' or v.name=='B' or v.name=='S' or v.name=='varTime':
                                        for index in v:
                                            if index==None:
                                                v.fix(pe.value(v))
                                            else:
                                                v[index].fix(pe.value(v[index]))                           

                            opt = SolverFactory(solvername, solver=subproblem_solver)
                            # start=time.time()
                            m.results = opt.solve(m, tee=tee,
                                                **output_options,
                                                **subproblem_solver_options,
                                                skip_trivial_constraints=True,
                                                )
                            
                            if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                                m.source[II,JJ]="Infeasible"
                            else:
                                m.source[II,JJ]="Feasible"
                        else:
                            m.source[II,JJ]="Not_scheduled"
               
            else:  
                m.cont_Status="Dynamcis_feasible"
                m.dsda_status = 'Optimal'
                m.best_sol=pe.value(m.obj)
                m.source={}
                for I in m.I_reactions:
                    for J in m.J_reactors:
                        if round(pe.value(m.Nref[I,J]))>=1:
                            m.source[I,J]="Feasible"
                        else:
                            m.source[I,J]="Not_scheduled"

    return m

def solve_subproblem_aprox_sequential_case_2(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 1e-3,
    dynamic_dist_model: bool = False
) -> pe.ConcreteModel():
    """
    Function that solves scheduling and then solves control. In scheduling is infeasible control is not solved.
    Note integer variables have to be previously fixed in the external reformulation.
    This function is for problems that can be decoupled into a steady-state (or scheduling) and a dynamic part
    
    Args:

    Returns:

    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        # start=time.time()
        preprocess_problem(m, simple=True)
        # end=time.time()
        # print('preprocess_time: ',str(end-start))
    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        m.best_sol=1e+8
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

    approximate_solution=True# If true, after solving lower bounding scheduling problem, then scheduling variables are fixed and NLP control problem is solved
                                # If false, then lower bounding scheduling is solved first, and then original minlp subproblem is solved, i.e., this is actually what we have called the enhanced dsda
    scheduling_only=False

    model_V='V2' #specify model version to avoid confusion (go to case_study_2_model.py to see available models) #TODO: GENERALIZE FOR DIFFERENT MODELS
    #### MODIFICATIONS FROM HERE WITH RESPECT TO ORIGINAL FUNCTION ################################    
    #DEACTIVATE DYNAMIC CONSTRAINTS AND NONLINEAR CONSTRAINTS. More generaly, deactivate nonlinear constraints

    #CHECK
    # for consts in m.component_data_objects(pe.Constraint,descend_into=True):
    #     if consts.body.polynomial_degree()>=2: #Polynomial degree greater or equal to 2 means nonlinear constraint
    #         consts.deactivate()   

    if model_V=='V2':

        for I in m.I_dynamics:
            for J in m.J_dynamics:
                for T in m.T:
                    m.c_defCT0[I,J,T].deactivate()
                    m.c_dCAdtheta[I,J,T].deactivate() 
                    m.c_dCBdtheta[I,J,T].deactivate() 
                    m.c_dCCdtheta[I,J,T].deactivate() 
                    m.c_dVdtheta[I,J,T].deactivate() 
                    m.c_dTRdtheta[I,J,T].deactivate() 
                    m.c_dTJdtheta[I,J,T].deactivate() 
                    m.c_dIntegral_hotdtheta[I,J,T].deactivate() 
                    m.c_dIntegral_colddtheta[I,J,T].deactivate() 
                    m.Constant_control1[I,J,T].deactivate() 
                    m.Constant_control2[I,J,T].deactivate() 
                    m.Constant_control3[I,J,T].deactivate() 
        m.C_TCP3.deactivate()              
        m.obj.deactivate()
        m.obj_dummy.deactivate()
        m.obj_scheduling.activate()  


        if dynamic_dist_model:
                for I in m.I_distil:
                    for J in m.J_distil:
                        for T in m.T:   
                            for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                cons.deactivate()

    #SOLVE SCHEDULING ONLY PROBLEM
    opt = SolverFactory(solvername, solver='cplex')

    # start=time.time()
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options,
                          skip_trivial_constraints=True,
                          )
    # end=time.time()
    # print('actual sol time part 1:',str(end-start))

    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time


    if scheduling_only:
        m.obj.activate()
        m.obj.value=pe.value(m.obj_scheduling)
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.best_sol=1e+8
            m.sched_Status="Scheduling_Infeasible"
        else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
            m.dsda_status = 'Optimal'
            m.best_sol=pe.value(m.obj)
            m.sched_Status="Scheduling_Feasible" 
            m.cont_Status="Not_Evaluated"       
    else: 
        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
            m.dsda_status = 'Evaluated_Infeasible'
            m.best_sol=1e+8
            m.sched_Status="Scheduling_Infeasible"
            m.cont_Status="Not_evaluated"
            m.source={}
            for I in m.I_dynamics:
                for J in m.J_dynamics:
                    m.source[I,J]="Not_evaluated"
            if dynamic_dist_model:
                for I in m.I_distil:
                    for J in m.J_distil:
                        m.source[I,J]="Not_evaluated"
        else:
             
            # ACTIVATE DYNAMIC CONSTRAINTS
            if model_V=='V2':

                for I in m.I_dynamics:
                    for J in m.J_dynamics:
                        for T in m.T:
                            m.c_defCT0[I,J,T].activate()
                            m.c_dCAdtheta[I,J,T].activate() 
                            m.c_dCBdtheta[I,J,T].activate() 
                            m.c_dCCdtheta[I,J,T].activate() 
                            m.c_dVdtheta[I,J,T].activate() 
                            m.c_dTRdtheta[I,J,T].activate() 
                            m.c_dTJdtheta[I,J,T].activate() 
                            m.c_dIntegral_hotdtheta[I,J,T].activate() 
                            m.c_dIntegral_colddtheta[I,J,T].activate() 
                            m.Constant_control1[I,J,T].activate() 
                            m.Constant_control2[I,J,T].activate() 
                            m.Constant_control3[I,J,T].activate() 
                m.C_TCP3.activate()              
                m.obj.activate()
                m.obj_dummy.deactivate()
                m.obj_scheduling.deactivate()  

                if dynamic_dist_model:
                    for I in m.I_distil:
                        for J in m.J_distil:
                            for T in m.T:   
                                for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                    cons.activate()                    

            # DEACTIVATE SCHEDULING CONSTRAINTS
            #TODO: I THINK THIS SHOULD BE OPTIONAL, BECAUSE I WILL FIX SCHEDULING CONSTRAINTS IN NEXT STEP 
        

            if approximate_solution:
                # FIX SCHEDULING VARIABLES
                for v in m.component_objects(pe.Var, descend_into=True):
                    if v.name=='X' or v.name=='Nref':
                        for index in v:
                            if index==None:
                                v.fix(round(pe.value(v)))
                            else:
                                v[index].fix(round(pe.value(v[index])))
                    elif v.name=='sumX' or v.name=='B' or v.name=='S' or v.name=='B_shift':# or v.name=='VarTime': #NOTE that in this case I do not include VarTime, becase un this case I am considering it using the inequality constraint
                        for index in v:
                            if index==None:
                                v.fix(pe.value(v))
                            else:
                                v[index].fix(pe.value(v[index]))

            opt = SolverFactory(solvername, solver=subproblem_solver)
            # start=time.time()
            m.results = opt.solve(m, tee=tee,
                                **output_options,
                                **subproblem_solver_options,
                                skip_trivial_constraints=True,
                                )
            # m=initialize_model(m,from_feasible=True,feasible_model='in_borrar_partial') 
            # end=time.time()
            # print('actual sol time part 2=',str(end-start))

            m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time
            m.sched_Status="Scheduling_Feasible"
            #### END OF MODIFICATIONS #######################################################################


            # Assign D-SDA status
            if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                m.dsda_status = 'Evaluated_Infeasible'
                m.best_sol=1e+8
                m.cont_Status="Dynamcis_Infeasible"
                # If dynamics are infeasible, we identify the source of infeasibility
                m.source={}


                m.C_TCP3.deactivate()
                m.obj.deactivate()
                m.obj_scheduling.deactivate() 
                m.obj_dummy.activate()




                # Identify if source is in reactors

                if dynamic_dist_model:
                    for I in m.I_distil:
                        for J in m.J_distil:
                            for T in m.T:   
                                for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                    cons.deactivate()  

                for II in m.I_dynamics:
                    for JJ in m.J_dynamics:     
                        # if round(pe.value(m.Nref[II,JJ]))>=1:
                        for I in m.I_dynamics:
                            for J in m.J_dynamics:
                                for T in m.T:
                                    m.c_defCT0[I,J,T].deactivate()
                                    m.c_dCAdtheta[I,J,T].deactivate() 
                                    m.c_dCBdtheta[I,J,T].deactivate() 
                                    m.c_dCCdtheta[I,J,T].deactivate() 
                                    m.c_dVdtheta[I,J,T].deactivate() 
                                    m.c_dTRdtheta[I,J,T].deactivate() 
                                    m.c_dTJdtheta[I,J,T].deactivate() 
                                    m.c_dIntegral_hotdtheta[I,J,T].deactivate() 
                                    m.c_dIntegral_colddtheta[I,J,T].deactivate() 
                                    m.Constant_control1[I,J,T].deactivate() 
                                    m.Constant_control2[I,J,T].deactivate() 
                                    m.Constant_control3[I,J,T].deactivate() 

                        for TT in m.T:
                            m.c_defCT0[II,JJ,TT].activate()
                            m.c_dCAdtheta[II,JJ,TT].activate() 
                            m.c_dCBdtheta[II,JJ,TT].activate() 
                            m.c_dCCdtheta[II,JJ,TT].activate() 
                            m.c_dVdtheta[II,JJ,TT].activate() 
                            m.c_dTRdtheta[II,JJ,TT].activate() 
                            m.c_dTJdtheta[II,JJ,TT].activate() 
                            m.c_dIntegral_hotdtheta[II,JJ,TT].activate() 
                            m.c_dIntegral_colddtheta[II,JJ,TT].activate() 
                            m.Constant_control1[II,JJ,TT].activate() 
                            m.Constant_control2[II,JJ,TT].activate() 
                            m.Constant_control3[II,JJ,TT].activate()


                        if approximate_solution:
                            # FIX SCHEDULING VARIABLES
                            for v in m.component_objects(pe.Var, descend_into=True):
                                if v.name=='X' or v.name=='Nref':
                                    for index in v:
                                        if index==None:
                                            v.fix(round(pe.value(v)))
                                        else:
                                            v[index].fix(round(pe.value(v[index])))
                                elif v.name=='sumX' or v.name=='B' or v.name=='S' or v.name=='B_shift': #or v.name=='VarTime':
                                    for index in v:
                                        if index==None:
                                            v.fix(pe.value(v))
                                        else:
                                            v[index].fix(pe.value(v[index]))                           

                        opt = SolverFactory(solvername, solver=subproblem_solver)
                        # start=time.time()
                        m.results = opt.solve(m, tee=tee,
                                            **output_options,
                                            **subproblem_solver_options,
                                            skip_trivial_constraints=True,
                                            )
                        # save=generate_initialization(m=m,model_name='in_borrar_partial')
                        
                        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                            m.source[II,JJ]="Infeasible"
                        else:
                            m.source[II,JJ]="Feasible"


                # Identify if source is in distillation

                if dynamic_dist_model:
                    for I in m.I_dynamics:
                        for J in m.J_dynamics:
                            for T in m.T:
                                m.c_defCT0[I,J,T].deactivate()
                                m.c_dCAdtheta[I,J,T].deactivate() 
                                m.c_dCBdtheta[I,J,T].deactivate() 
                                m.c_dCCdtheta[I,J,T].deactivate() 
                                m.c_dVdtheta[I,J,T].deactivate() 
                                m.c_dTRdtheta[I,J,T].deactivate() 
                                m.c_dTJdtheta[I,J,T].deactivate() 
                                m.c_dIntegral_hotdtheta[I,J,T].deactivate() 
                                m.c_dIntegral_colddtheta[I,J,T].deactivate() 
                                m.Constant_control1[I,J,T].deactivate() 
                                m.Constant_control2[I,J,T].deactivate() 
                                m.Constant_control3[I,J,T].deactivate()  

                    for II in m.I_distil:
                        for JJ in m.J_distil:     
                            # if round(pe.value(m.Nref[II,JJ]))>=1:
                            for I in m.I_distil:
                                for J in m.J_distil:
                                    for T in m.T:
                                        for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                                            cons.deactivate()  

                            for TT in m.T:
                                for cons in m.dist_models[II,JJ,TT].component_data_objects(pe.Constraint,descend_into=True):
                                    cons.activate() 


                            if approximate_solution:
                                # FIX SCHEDULING VARIABLES
                                for v in m.component_objects(pe.Var, descend_into=True):
                                    if v.name=='X' or v.name=='Nref':
                                        for index in v:
                                            if index==None:
                                                v.fix(round(pe.value(v)))
                                            else:
                                                v[index].fix(round(pe.value(v[index])))
                                    elif v.name=='sumX' or v.name=='B' or v.name=='S' or v.name=='B_shift':# or v.name=='VarTime':
                                        for index in v:
                                            if index==None:
                                                v.fix(pe.value(v))
                                            else:
                                                v[index].fix(pe.value(v[index]))                           

                            opt = SolverFactory(solvername, solver=subproblem_solver)
                            # start=time.time()
                            m.results = opt.solve(m, tee=tee,
                                                **output_options,
                                                **subproblem_solver_options,
                                                skip_trivial_constraints=True,
                                                )
                            # save=generate_initialization(m=m,model_name='in_borrar_partial')
                            
                            if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                                m.source[II,JJ]="Infeasible"
                            else:
                                m.source[II,JJ]="Feasible"



            else:  
                m.cont_Status="Dynamcis_feasible"
                m.dsda_status = 'Optimal'
                m.best_sol=pe.value(m.obj)
                m.source={}
                for I in m.I_dynamics:
                    for J in m.J_dynamics:
                        m.source[I,J]="Feasible"
                if dynamic_dist_model:
                    for I in m.I_distil:
                        for J in m.J_distil:
                            m.source[I,J]="Feasible"                

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


def solve_with_gdpopt(
    m: pe.ConcreteModel(),
    mip: str = 'cplex',
    mip_options: dict = {},
    nlp: str = 'knitro',
    nlp_options: dict = {},
    minlp: str = 'baron',
    minlp_options: dict = {},
    timelimit: float = 1000,
    strategy: str = 'LOA',
    mip_output: bool = False,
    nlp_output: bool = False,
    minlp_output: bool = False,
    tee: bool = False,
    rel_tol: float = 1e-3,
) -> pe.ConcreteModel():
    """
    Function that solves GDP model using GDPopt
    Args:
        m: GDP model that is to be solved
        mip: MIP solver algorithm
        nlp: NLP solver algorithm
        timelimit: time limit in seconds for the solve statement
        strategy: GDPopt strategy
        mip_output: Determine keeping or not GAMS files of the MIP model
        nlp_output: Determine keeping or not GAMS files of the NLP model
        tee: Display iterations
        rel_tol: Relative optimality tolerance for subproblems and GDPOpt itself
    Returns:
        m: Solved GDP model
    """

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)

    # Output report

    mip_options['add_options'] = mip_options.get('add_options', [])
    mip_options['add_options'].append('option optcr=0.0;')

    nlp_options['add_options'] = nlp_options.get('add_options', [])
    nlp_options['add_options'].append('option optcr=%s;' % rel_tol)

    minlp_options['add_options'] = minlp_options.get('add_options', [])
    minlp_options['add_options'].append('option optcr=%s;' % rel_tol)

    if mip_output:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        gams_path = os.path.join(dir_path, "gamsfiles/")
        if not(os.path.exists(gams_path)):
            print('Directory for automatically generated files ' +
                  gams_path + ' does not exist. We will create it')
            os.makedirs(gams_path)
        mip_options['keepfiles'] = True
        mip_options['tmpdir'] = gams_path
        mip_options['symbolic_solver_labels'] = True

    if nlp_output:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        gams_path = os.path.join(dir_path, "gamsfiles/")
        if not(os.path.exists(gams_path)):
            print('Directory for automatically generated files ' +
                  gams_path + ' does not exist. We will create it')
            os.makedirs(gams_path)
        nlp_options['keepfiles'] = True
        nlp_options['tmpdir'] = gams_path
        nlp_options['symbolic_solver_labels'] = True

    if minlp_output:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        gams_path = os.path.join(dir_path, "gamsfiles/")
        if not(os.path.exists(gams_path)):
            print('Directory for automatically generated files ' +
                  gams_path + ' does not exist. We will create it')
            os.makedirs(gams_path)
        minlp_options['keepfiles'] = True
        minlp_options['tmpdir'] = gams_path
        minlp_options['symbolic_solver_labels'] = True

    # Solve
    solvername = 'gdpopt'
    opt = SolverFactory(solvername)
    m.results = opt.solve(m, tee=tee,
                          strategy=strategy,
                          time_limit=timelimit,
                          mip_solver='gams',
                          mip_solver_args=dict(
                              solver=mip, warmstart=True, **mip_options),
                          nlp_solver='gams',
                          nlp_solver_args=dict(
                              solver=nlp, warmstart=True, tee=tee, **nlp_options),
                          minlp_solver='gams',
                          minlp_solver_args=dict(
                              solver=minlp, warmstart=True, tee=tee, **minlp_options), local_minlp_solver='gams', local_minlp_solver_args=dict(
                              solver=minlp, warmstart=True, tee=tee, **minlp_options)
                        # #   mip_presolve=True, #True is the default
                        #   init_strategy='fix_disjuncts',#'fix_disjuncts'##'set_covering'#
                        #   init_algorithm='fix_disjuncts',
                        #   #   set_cover_iterlim=0,
                        #   #iterlim#=1000,
                        # #   force_subproblem_nlp=False,
                        #   subproblem_presolve=False,
                        #   #   bound_tolerance=rel_tol
                        #   #   calc_disjunctive_bounds=True
                          )
    # update_boolean_vars_from_binary(m)
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


def evaluate_neighbors_aprox(
    best_sol: float,
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
            m_solved = solve_subproblem_aprox(
                m=m_fixed,
                subproblem_solver=subproblem_solver,
                subproblem_solver_options=subproblem_solver_options,
                timelimit=t_remaining,
                gams_output=gams_output,
                tee=tee,
                rel_tol=rel_tol,
                best_sol=best_sol)
            if m_solved.best_sol<=best_sol:
                best_sol=m_solved.best_sol
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
            elif global_tee:
                if m_solved.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('Pruned:', temp[i], '   |   Lower bound problem infeasible   |   Global Time:', round(t_end - current_time, 2))                    
                elif m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                    print('Pruned:', temp[i], '   |   No improvement expected   |   Global Time:', round(t_end - current_time, 2))          
            if time.perf_counter() - current_time > timelimit:  # current
                break
    if global_tee:
        print()
        print('New best neighbor:', best_var)
    return fmin, best_var, best_dir, improve, evaluation_time, ns_evaluated, best_path

def evaluate_neighbors_aprox_tau_only(
    best_sol: float,
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

            m_fixed = external_ref_sequential(
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
            m_solved = solve_subproblem_aprox_tau_only(
                m=m_fixed,
                subproblem_solver=subproblem_solver,
                subproblem_solver_options=subproblem_solver_options,
                timelimit=t_remaining,
                gams_output=gams_output,
                tee=tee,
                rel_tol=rel_tol,
                best_sol=best_sol)
            if m_solved.best_sol<=best_sol:
                best_sol=m_solved.best_sol
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
            elif global_tee:
                if m_solved.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('Pruned:', temp[i], '   |   Lower bound problem infeasible   |   Global Time:', round(t_end - current_time, 2))                    
                elif m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                    print('Pruned:', temp[i], '   |   No improvement expected   |   Global Time:', round(t_end - current_time, 2))          
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

def do_line_search_aprox(
    start: list,
    fmin: float,
    direction: list,
    model_function,
    model_args: dict,
    ext_dict: dict,
    ext_logic,
    best_sol: float,
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
            m_solved = solve_subproblem_aprox(
                m=m_fixed,
                subproblem_solver=subproblem_solver,
                subproblem_solver_options=subproblem_solver_options,
                timelimit=t_remaining,
                gams_output=gams_output,
                tee=tee,
                rel_tol=rel_tol,
                best_sol=best_sol
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
            elif global_tee:
                if m_solved.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('Pruned:', moved_point, '   |   Lower bound problem infeasible   |   Global Time:', round(time.perf_counter() - current_time, 2))                    
                elif m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                    print('Pruned:', moved_point, '   |   No improvement expected   |   Global Time:', round(time.perf_counter() - current_time, 2))                                    
    return fmin, best_var, moved, ls_time, ls_evaluated, new_path

def do_line_search_aprox_tau_only(
    start: list,
    fmin: float,
    direction: list,
    model_function,
    model_args: dict,
    ext_dict: dict,
    ext_logic,
    best_sol: float,
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
            m_fixed = external_ref_sequential(
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
            m_solved = solve_subproblem_aprox_tau_only(
                m=m_fixed,
                subproblem_solver=subproblem_solver,
                subproblem_solver_options=subproblem_solver_options,
                timelimit=t_remaining,
                gams_output=gams_output,
                tee=tee,
                rel_tol=rel_tol,
                best_sol=best_sol
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
            elif global_tee:
                if m_solved.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('Pruned:', moved_point, '   |   Lower bound problem infeasible   |   Global Time:', round(time.perf_counter() - current_time, 2))                    
                elif m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                    print('Pruned:', moved_point, '   |   No improvement expected   |   Global Time:', round(time.perf_counter() - current_time, 2))                                    
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

def solve_with_dsda_aprox(
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
        print('\nStarting enhanced D-SDA with k =', k)
        print('--------------------------------------------------------------------------')

    # Initialize
    route = []
    obj_route = []
    global_evaluated = []
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
    m_solved = solve_subproblem_aprox(
        m=m_fixed,
        subproblem_solver=subproblem_solver,
        subproblem_solver_options=subproblem_solver_options,
        timelimit=iter_timelimit,
        gams_output=gams_output,
        tee=tee,
        rel_tol=rel_tol,
    )
    best_sol=m_solved.best_sol
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

        fmin, best_var, best_dir, improve, eval_time, ns_evaluated, best_path  = evaluate_neighbors_aprox(
            best_sol=best_sol,
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

        best_sol=fmin
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

                fmin, best_var, moved, ls_time, ls_evaluated, best_path = do_line_search_aprox(
                    start=best_var,
                    fmin=fmin,
                    direction=neighborhood[best_dir],
                    model_function=model_function,
                    model_args=model_args,
                    ext_dict=dict_extvar,
                    ext_logic=ext_logic,
                    best_sol=best_sol,
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
                best_sol=fmin
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

def solve_with_dsda_aprox_tau_only(
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
        print('\nStarting enhanced D-SDA with k =', k)
        print('--------------------------------------------------------------------------')

    # Initialize
    route = []
    obj_route = []
    global_evaluated = []
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

    m_fixed = external_ref_sequential(
        m=m_init,
        x=ext_var,
        extra_logic_function=ext_logic,
        dict_extvar=dict_extvar,
        mip_ref=mip_transformation,
        tee=False
    )

    # Solve for initialization
    m_solved = solve_subproblem_aprox_tau_only(
        m=m_fixed,
        subproblem_solver=subproblem_solver,
        subproblem_solver_options=subproblem_solver_options,
        timelimit=iter_timelimit,
        gams_output=gams_output,
        tee=tee,
        rel_tol=rel_tol,
    )
    best_sol=m_solved.best_sol
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

        fmin, best_var, best_dir, improve, eval_time, ns_evaluated, best_path  = evaluate_neighbors_aprox_tau_only(
            best_sol=best_sol,
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

        best_sol=fmin
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

                fmin, best_var, moved, ls_time, ls_evaluated, best_path = do_line_search_aprox_tau_only(
                    start=best_var,
                    fmin=fmin,
                    direction=neighborhood[best_dir],
                    model_function=model_function,
                    model_args=model_args,
                    ext_dict=dict_extvar,
                    ext_logic=ext_logic,
                    best_sol=best_sol,
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
                best_sol=fmin
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




def sequential_iterative_1(
    ext_logic,
    starting_point: list,
    model_function,
    ext_dict,
    rate_tau: int=1,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    provide_starting_initialization: bool = True,
    feasible_model: str = '',
    subproblem_solver: str = 'knitro',
    iter_timelimit: float = 1000000,
    subproblem_solver_options: dict = {},
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 1e-3):

    """
    rate_beta_ub: Real, rate to decrease upper bound of beta
    rate_tau: integer, rate to increase external variables related to variable processing time

    """
    if global_tee:
        print('\nStarting Sequential Scheduling and control')
        print('--------------------------------------------------------------------------')



    m = model_function()
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict)
    if len(starting_point) != num_ext_var:
        print("The size of the initialization vector must be equal to "+str(num_ext_var))

    t_start = time.perf_counter()


    if provide_starting_initialization:
        m = initialize_model(m, from_feasible=True, feasible_model=feasible_model, json_path=None)


    #START LOOP
    count_1=0
    ext_var = starting_point #external variables updated iteratively
    while any([ext_var[pos]!=max_allowed[pos+1] for pos in range(len(ext_var))]):
        count_1=count_1+1
        if global_tee:
            print("--------------------------------------------------------------------------")
            print("\n --Loop iteration "+str(count_1))
        if count_1 >=2:
            for posit in range(len(ext_var)):
                ext_var[posit]=min([ext_var[posit]+rate_tau,max_allowed[posit+1]])

        m = model_function()
        if mip_transformation:
            m, dict_extvar = extvars_gdp_to_mip(m=m,gdp_dict_extvar=dict_extvar,transformation=transformation,)
        m = external_ref_sequential(m=m,x=ext_var,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)    
        if global_tee:
            print("-------Current value of Ext vars: ",ext_var)      
        m = solve_subproblem_aprox_sequential(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=iter_timelimit,gams_output=gams_output,tee=tee,rel_tol=rel_tol)           
        if global_tee:
            print("-------Status: ",m.sched_Status," ",m.cont_Status,"  |  Current CPU time [s]:",time.perf_counter()-t_start)
        if m.dsda_status == 'Optimal':
            print('--------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------')
            print('Feasible solution with: ')
            print("Current value of Ext vars (related to processing times): ",ext_var)
            print("Objective function:",m.best_sol) 
            print("CPU time[s]: ",time.perf_counter()-t_start)
            return m
    return m

# IMPROVED VERSION OF SEQUENTIAL ITERATIVE STRATEGY. This is the one that ggoes to the scheduling and dynamics article
def sequential_iterative_2(
    ext_logic,
    starting_point: list,
    model_function,
    kwargs,
    ext_dict,
    rate_tau: int=1,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    provide_starting_initialization: bool = True,
    feasible_model: str = '',
    subproblem_solver: str = 'knitro',
    iter_timelimit: float = 1000000,
    subproblem_solver_options: dict = {},
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 1e-3):

    """
    rate_beta_ub: Real, rate to decrease upper bound of beta
    rate_tau: integer, rate to increase external variables related to variable processing time

    """
    if global_tee:
        print('\nStarting Sequential Scheduling and control')
        print('--------------------------------------------------------------------------')



    m = model_function(**kwargs)
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict)
    if len(starting_point) != num_ext_var:
        print("The size of the initialization vector must be equal to "+str(num_ext_var))

    t_start = time.perf_counter()


    if provide_starting_initialization:
        m = initialize_model(m, from_feasible=True, feasible_model=feasible_model, json_path=None)


    #START LOOP
    count_1=0
    ext_var = starting_point #external variables updated iteratively
    while any([ext_var[pos]!=max_allowed[pos+1] for pos in range(len(ext_var))]):
        count_1=count_1+1
        if global_tee:
            print("--------------------------------------------------------------------------")
            print("\n --Loop iteration "+str(count_1))
        if count_1 >=2:

            posit=-1
            for I in m.I_reactions:
                for J in m.J_reactors:
                    posit=posit+1
                    if m.source[I,J]=='Infeasible' or m.source[I,J]=='Not_evaluated':
                        ext_var[posit]=min([ext_var[posit]+rate_tau,max_allowed[posit+1]])

        m = model_function(**kwargs)
        if mip_transformation:
            m, dict_extvar = extvars_gdp_to_mip(m=m,gdp_dict_extvar=dict_extvar,transformation=transformation,)
        m = external_ref_sequential(m=m,x=ext_var,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)    
        if global_tee:
            print("-------Current value of Ext vars: ",ext_var)      
        m = solve_subproblem_aprox_sequential(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=iter_timelimit,gams_output=gams_output,tee=tee,rel_tol=rel_tol)           
        if global_tee:
            print("-------Status: ",m.sched_Status," ",m.cont_Status,"  |  Current CPU time [s]:",time.perf_counter()-t_start)
            print("-------Dynamic models: ",m.source)
        if m.dsda_status == 'Optimal':
            print('--------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------')
            print('Feasible solution with: ')
            print("Current value of Ext vars (related to processing times): ",ext_var)
            print("Objective function:",m.best_sol) 
            print("CPU time[s]: ",time.perf_counter()-t_start)
            return m,ext_var
    return m,ext_var


# IMPROVED VERSION OF SEQUENTIAL ITERATIVE STRATEGY. This is the one that ggoes to the scheduling and dynamics article. Applied to case study 2
def sequential_iterative_2_case2(
    ext_logic,
    starting_point: list,
    model_function,
    kwargs,
    ext_dict,
    rate_tau: int=1,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    provide_starting_initialization: bool = True,
    feasible_model: str = '',
    subproblem_solver: str = 'knitro',
    iter_timelimit: float = 1000000,
    subproblem_solver_options: dict = {},
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 1e-3,
    dynamic_dist_model: bool=False):

    """
    rate_beta_ub: Real, rate to decrease upper bound of beta
    rate_tau: integer, rate to increase external variables related to variable processing time

    """
    if global_tee:
        print('\nStarting Sequential Scheduling and control')
        print('--------------------------------------------------------------------------')



    m = model_function(**kwargs)
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict)
    if len(starting_point) != num_ext_var:
        print("The size of the initialization vector must be equal to "+str(num_ext_var))

    t_start = time.perf_counter()


    if provide_starting_initialization:
        m = initialize_model(m, from_feasible=True, feasible_model=feasible_model, json_path=None)


    #START LOOP
    count_1=0
    ext_var = starting_point #external variables updated iteratively
    while any([ext_var[pos]!=max_allowed[pos+1] for pos in range(len(ext_var))]):
        count_1=count_1+1
        if global_tee:
            print("--------------------------------------------------------------------------")
            print("\n --Loop iteration "+str(count_1))
        if count_1 >=2:

            posit=0 #TODO, GENERALIZE THIS COUNTER
            for I in m.I_dynamics:
                for J in m.J_dynamics:
                    print(m.source[I,J])
                    posit=posit+1
                    if m.source[I,J]=='Infeasible' or m.source[I,J]=='Not_evaluated':
                        print(min([ext_var[posit]+rate_tau,max_allowed[posit+1]]))
                        ext_var[posit]=min([ext_var[posit]+rate_tau,max_allowed[posit+1]])
                        print(ext_var)
            if dynamic_dist_model: 
                posit=6
                for I in m.I_distil:
                    for J in m.J_distil:
                        print(m.source[I,J])
                        posit=posit+1
                        if m.source[I,J]=='Infeasible' or m.source[I,J]=='Not_evaluated':
                            print(min([ext_var[posit]+rate_tau,max_allowed[posit+1]]))
                            ext_var[posit]=min([ext_var[posit]+rate_tau,max_allowed[posit+1]])
                            print(ext_var)
        m = model_function(**kwargs)
        if provide_starting_initialization:
            m = initialize_model(m, from_feasible=True, feasible_model=feasible_model, json_path=None)
        if mip_transformation:
            m, dict_extvar = extvars_gdp_to_mip(m=m,gdp_dict_extvar=dict_extvar,transformation=transformation,)
        m = external_ref_sequential_case_2(m=m,x=ext_var,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)    
        if global_tee:
            print("-------Current value of Ext vars: ",ext_var)      
        m = solve_subproblem_aprox_sequential_case_2(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=iter_timelimit,gams_output=gams_output,tee=tee,rel_tol=rel_tol, dynamic_dist_model= dynamic_dist_model)           
        if global_tee:
            print("-------Status: ",m.sched_Status," ",m.cont_Status,"  |  Current CPU time [s]:",time.perf_counter()-t_start)
            print("-------Dynamic models: ",m.source)
            # if not dynamic_dist_model:

            #     #--------------------------------- Gantt plot--------------------------------------------
            #     fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
            #     # Setting Y-axis limits
            #     gnt.set_ylim(8, 52) #TODO: change depending case study
                
            #     # Setting X-axis limits
            #     gnt.set_xlim(0, m.lastT.value*m.delta.value)
                
            #     # Setting labels for x-axis and y-axis
            #     gnt.set_xlabel('Time [h]')
            #     gnt.set_ylabel('Units')
                
            #     # Setting ticks on y-axis
            #     gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
            #     # Labelling tickes of y-axis
            #     gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
                
                
            #     # Setting graph attribute
            #     gnt.grid(False)
                
            #     # Declaring bars in schedule
            #     height=9
            #     already_used=[]
            #     for j in m.J:

            #         if j=='U1':
            #             lower_y_position=40    
            #         elif j=='U2':
            #             lower_y_position=30    
            #         elif j=='U3':
            #             lower_y_position=20
            #         elif j=='U4':
            #             lower_y_position=10
            #         for i in m.I:
            #             if i=='T1':
            #                 bar_color='tab:red'
            #             elif i=='T2':
            #                 bar_color='tab:green'    
            #             elif i=='T3':
            #                 bar_color='tab:blue'    
            #             elif i=='T4':
            #                 bar_color='tab:orange' 
            #             elif i=='T5':
            #                 bar_color='tab:olive'
            #             for t in m.T:
            #                 try:
            #                     if i in m.I_dynamics and j in m.J_dynamics:
            #                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
            #                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
            #                             already_used.append(i)
            #                         elif round(pe.value(m.X[i,j,t]))==1:
            #                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
            #                     else:
            #                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
            #                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
            #                             already_used.append(i)
            #                         elif round(pe.value(m.X[i,j,t]))==1:
            #                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        

            #                 except:
            #                     pass 
            #     gnt.tick_params(axis='both', which='major', labelsize=15)
            #     gnt.tick_params(axis='both', which='minor', labelsize=15) 
            #     gnt.yaxis.label.set_size(15)
            #     gnt.xaxis.label.set_size(15)
            #     plt.legend()
            #     plt.show()
            #     plt.clf()
            #     plt.cla()
            #     plt.close()

            # else:

            #     #--------------------------------- Gantt plot--------------------------------------------
            #     fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
            #     # Setting Y-axis limits
            #     gnt.set_ylim(8, 52) #TODO: change depending case study
                
            #     # Setting X-axis limits
            #     gnt.set_xlim(0, m.lastT.value*m.delta.value)
                
            #     # Setting labels for x-axis and y-axis
            #     gnt.set_xlabel('Time [h]')
            #     gnt.set_ylabel('Units')
                
            #     # Setting ticks on y-axis
            #     gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
            #     # Labelling tickes of y-axis
            #     gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
                
                
            #     # Setting graph attribute
            #     gnt.grid(False)
                
            #     # Declaring bars in schedule
            #     height=9
            #     already_used=[]
            #     for j in m.J:

            #         if j=='U1':
            #             lower_y_position=40    
            #         elif j=='U2':
            #             lower_y_position=30    
            #         elif j=='U3':
            #             lower_y_position=20
            #         elif j=='U4':
            #             lower_y_position=10
            #         for i in m.I:
            #             if i=='T1':
            #                 bar_color='tab:red'
            #             elif i=='T2':
            #                 bar_color='tab:green'    
            #             elif i=='T3':
            #                 bar_color='tab:blue'    
            #             elif i=='T4':
            #                 bar_color='tab:orange' 
            #             elif i=='T5':
            #                 bar_color='tab:olive'
            #             for t in m.T:
            #                 try:
            #                     if i in m.I_dynamics and j in m.J_dynamics:
            #                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
            #                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
            #                             already_used.append(i)
            #                         elif round(pe.value(m.X[i,j,t]))==1:
            #                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              
            #                     elif i in m.I_distil and j in m.J_distil:
            #                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
            #                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
            #                             already_used.append(i)
            #                         elif round(pe.value(m.X[i,j,t]))==1:
            #                             gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                                                          
            #                     else:
            #                         if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
            #                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
            #                             already_used.append(i)
            #                         elif round(pe.value(m.X[i,j,t]))==1:
            #                             gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
            #                             gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        

            #                 except:
            #                     pass 
            #     gnt.tick_params(axis='both', which='major', labelsize=15)
            #     gnt.tick_params(axis='both', which='minor', labelsize=15) 
            #     gnt.yaxis.label.set_size(15)
            #     gnt.xaxis.label.set_size(15)
            #     plt.legend()
            #     plt.show()
            #     plt.clf()
            #     plt.cla()
            #     plt.close()


        if m.dsda_status == 'Optimal':
            print('--------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------')
            print('Feasible solution with: ')
            print("Current value of Ext vars (related to processing times): ",ext_var)
            print("Objective function:",m.best_sol) 
            print("CPU time[s]: ",time.perf_counter()-t_start)
            return m,ext_var
    return m,ext_var

def sequential_non_iterative_2_case2(
    ext_logic,
    starting_point: list,
    model_function,
    kwargs,
    ext_dict,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    provide_starting_initialization: bool = False,
    feasible_model: str = '',
    subproblem_solver: str = 'dicopt',
    subproblem_solver_options: dict = {},
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 0,
    with_distillation: bool = False,
    fixedB: bool=True, # DECIDES IF THE PROBLEM WILL BE SOLVED WITH A FIXED MAXIMUM CAPACITY HERUSTICA AT EVERY STAGE
    last_stage_dynamic_cost: bool=False): # DECIDES IF THE THIRD BLOCK WILL MINIMIZE COSTS. IF False, then the last stage simply uses dynamics found in processing time minimization stage):

    """
    rate_beta_ub: Real, rate to decrease upper bound of beta
    rate_tau: integer, rate to increase external variables related to variable processing time

    """
    if global_tee:
        print('\nStarting Sequential Scheduling and control (non_iterative)')
        print('--------------------------------------------------------------------------')

    ### FIRST SOLVE MINIMIZATION OF PROCESSING TIMES!!!!
    kwargs['sequential']=True
    kwargs['max_capacity']=True
    m = model_function(**kwargs)
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict)
    if len(starting_point) != num_ext_var:
        print("The size of the initialization vector must be equal to "+str(num_ext_var))

    t_start = time.perf_counter()


    if provide_starting_initialization:
        m = initialize_model(m, from_feasible=True, feasible_model=feasible_model, json_path=None)


    m = external_ref_sequential_case_2(m=m,x=starting_point,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)

    # DEACTIVATE SCHEDULING CONSTRAINTS
    m.E2_CAPACITY_LOW.deactivate()
    m.E2_CAPACITY_UP.deactivate()
    m.E3_BALANCE_INIT.deactivate()
    m.E1_UNIT.deactivate()
    m.E3_BALANCE.deactivate()
    m.X_Z_relation.deactivate()

    m.C_TCP1.deactivate()
    m.C_TCP2.deactivate()
    m.C_TCP3.deactivate()
    m.C_TMC.deactivate()
    m.C_SALES.deactivate()

    m.obj_scheduling.deactivate()
    m.obj_dummy.deactivate()


    # Only consider one execution of dynamic tasks
    for Index in m.I_J:
            m.Nref[Index].fix(1)
            I=Index[0]
            J=Index[1]
            for T in m.T:           
                m.X[I,J,T].fix(0)
                if T==m.T.first() and I in m.I_dynamics and J in m.J_dynamics:
                    m.X[I,J,T].fix(1)
                elif with_distillation:
                    if T==m.T.first() and I in m.I_distil and J in m.J_distil:
                        m.X[I,J,T].fix(1)

    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee) 
    save=generate_initialization(m=m,model_name='partial_borrar')

    # retrieve maximum capacity
    max_Capa={}
    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                if T==m.T.first():
                    max_Capa[I, J, T]=pe.value(m.B[I, J, T])   
    if with_distillation:
        for I in m.I_distil:
            for J in m.J_distil:
                for T in m.T:
                    if T==m.T.first():
                        max_Capa[I, J, T]=pe.value(m.B[I, J, T])

    kwargs['max_capacity']=False
    m = model_function(**kwargs)
    m = external_ref_sequential_case_2(m=m,x=starting_point,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)

    # DEACTIVATE SCHEDULING CONSTRAINTS
    m.E2_CAPACITY_LOW.deactivate()
    m.E2_CAPACITY_UP.deactivate()
    m.E3_BALANCE_INIT.deactivate()
    m.E1_UNIT.deactivate()
    m.E3_BALANCE.deactivate()
    m.X_Z_relation.deactivate()

    m.C_TCP1.deactivate()
    m.C_TCP2.deactivate()
    m.C_TCP3.deactivate()
    m.C_TMC.deactivate()
    m.C_SALES.deactivate()

    m.obj_scheduling.deactivate()
    m.obj_dummy.deactivate()

    # Only consider one execution of dynamic tasks
    for Index in m.I_J:
            m.Nref[Index].fix(1)
            I=Index[0]
            J=Index[1]
            for T in m.T:           
                m.X[I,J,T].fix(0)
                if T==m.T.first() and I in m.I_dynamics and J in m.J_dynamics:
                    m.X[I,J,T].fix(1)
                elif with_distillation:
                    if T==m.T.first() and I in m.I_distil and J in m.J_distil:
                        m.X[I,J,T].fix(1)

    #fix operation at maximum capacity
    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                if T==m.T.first():
                    m.B[I, J, T].fix(max_Capa[I, J, T]) 
    if with_distillation:
        for I in m.I_distil:
            for J in m.J_distil:
                for T in m.T:
                    if T==m.T.first():
                        m.B[I, J, T].fix(max_Capa[I, J, T]) 

    m=initialize_model(m,from_feasible=True,feasible_model='partial_borrar')  
    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee)    
    if with_distillation:
        save=generate_initialization(m=m,model_name='case_2_min_proc_time_solution_with_distillation')
    else:
        save=generate_initialization(m=m,model_name='case_2_min_proc_time_solution')
    min_proc_time={}
    for I in m.I_dynamics:
        for J in m.J_dynamics:
            min_proc_time[I,J]=pe.value(m.varTime[I,J,0])
    if with_distillation:
        for I in m.I_distil:
            for J in m.J_distil:
                min_proc_time[I,J]=pe.value(m.varTime[I,J,0])

    # Solve scheduling
    m = model_function(**kwargs)
    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                m.c_defCT0[I,J,T].deactivate()
                m.c_dCAdtheta[I,J,T].deactivate() 
                m.c_dCBdtheta[I,J,T].deactivate() 
                m.c_dCCdtheta[I,J,T].deactivate() 
                m.c_dVdtheta[I,J,T].deactivate() 
                m.c_dTRdtheta[I,J,T].deactivate() 
                m.c_dTJdtheta[I,J,T].deactivate() 
                m.c_dIntegral_hotdtheta[I,J,T].deactivate() 
                m.c_dIntegral_colddtheta[I,J,T].deactivate() 
                m.Constant_control1[I,J,T].deactivate() 
                m.Constant_control2[I,J,T].deactivate() 
                m.Constant_control3[I,J,T].deactivate() 

    if with_distillation:
        for I in m.I_distil:
            for J in m.J_distil:
                for T in m.T:
                    for cons in m.dist_models[I,J,T].component_data_objects(pe.Constraint,descend_into=True):
                        cons.deactivate()                

    m.C_TCP3.deactivate()

    m.obj_scheduling.activate()
    m.obj.deactivate()
    m.obj_dummy.deactivate()
    if with_distillation:
        def _linking1_11(m,I,J,T):
            if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
                return m.varTime[I,J,T]-min_proc_time[I,J] <= (kwargs['upper_t_h'][I,J]-min_proc_time[I,J])*(1-m.X[I,J,T])  
            return pe.Constraint.Skip
        m.linking111=pe.Constraint(m.I,m.J,m.T,rule=_linking1_11,doc='Linking constraint to guarantee operation at minimum processing time') 

        def _linking1_22(m,I,J,T):
            if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
                return -(m.varTime[I,J,T]-min_proc_time[I,J]) <= min_proc_time[I,J]*(1-m.X[I,J,T]) 
            return pe.Constraint.Skip 
        m.linking122=pe.Constraint(m.I,m.J,m.T,rule=_linking1_22,doc='Linking constraint to guarantee operation at minimum processing time') 
        if fixedB:
            def _linking2_11(m,I,J,T):
                if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
                    return m.B[I,J,T]-max_Capa[I, J, 0] <= (m.beta_max[I,J]-max_Capa[I, J, 0])*(1-m.X[I,J,T]) 
                return pe.Constraint.Skip 
            m.linking211=pe.Constraint(m.I,m.J,m.T,rule=_linking2_11,doc='Linking constraint to guarantee operation at maximum capacity') 

            def _linking2_22(m,I,J,T):
                if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
                    return -(m.B[I,J,T]-max_Capa[I, J, 0] )<= max_Capa[I, J, 0]*(1-m.X[I,J,T])  
                return pe.Constraint.Skip
            m.linking222=pe.Constraint(m.I,m.J,m.T,rule=_linking2_22,doc='Linking constraint to guarantee operation at maximum capacity') 
    else:
        def _linking1_11(m,I,J,T):
            return m.varTime[I,J,T]-min_proc_time[I,J] <= (kwargs['upper_t_h'][I,J]-min_proc_time[I,J])*(1-m.X[I,J,T])  
        m.linking111=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking1_11,doc='Linking constraint to guarantee operation at minimum processing time') 

        def _linking1_22(m,I,J,T):
            return -(m.varTime[I,J,T]-min_proc_time[I,J]) <= min_proc_time[I,J]*(1-m.X[I,J,T])  
        m.linking122=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking1_22,doc='Linking constraint to guarantee operation at minimum processing time') 
        if fixedB:
            def _linking2_11(m,I,J,T):
                return m.B[I,J,T]-max_Capa[I, J, 0] <= (m.beta_max[I,J]-max_Capa[I, J, 0])*(1-m.X[I,J,T])  
            m.linking211=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking2_11,doc='Linking constraint to guarantee operation at maximum capacity') 

            def _linking2_22(m,I,J,T):
                return -(m.B[I,J,T]-max_Capa[I, J, 0] )<= max_Capa[I, J, 0]*(1-m.X[I,J,T])  
            m.linking222=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking2_22,doc='Linking constraint to guarantee operation at maximum capacity')         

    m=solve_with_minlp(m,transformation='bigm',minlp='cplex',timelimit=86400,gams_output=False,tee=tee,rel_tol=0)
    if with_distillation:
        save=generate_initialization(m=m,model_name='case_2_scheduling_solution_with_distillation')
    else:
        save=generate_initialization(m=m,model_name='case_2_scheduling_solution')

    # retrieve ext vars
    Sol_found=[]
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m.minTau[I,J]+1)
    for I_J in m.I_J:
        Sol_found.append(1+round(pe.value(m.Nref[I_J])))

    m = model_function(**kwargs)
    m = external_ref_sequential_case_2(m=m,x=Sol_found,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)
    m.obj_scheduling.activate()
    m.obj.deactivate()
    m.obj_dummy.deactivate()
    if with_distillation:
        m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_solution_with_distillation')  
    else:
        m=initialize_model(m,from_feasible=True,feasible_model='case_2_scheduling_solution') 
    for v in m.component_objects(pe.Var, descend_into=True):
        if v.name=='X' or v.name=='Nref':
            for index in v:
                if index==None:
                    v.fix(round(pe.value(v)))
                else:
                    v[index].fix(round(pe.value(v[index])))
        elif v.name=='sumX' or v.name=='B' or v.name=='S' or v.name=='B_shift':
            for index in v:
                if index==None:
                    v.fix(pe.value(v))
                else:
                    v[index].fix(pe.value(v[index]))

    if with_distillation:
        def _linking1_11(m,I,J,T):
            if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
                return m.varTime[I,J,T]-min_proc_time[I,J] <= (kwargs['upper_t_h'][I,J]-min_proc_time[I,J])*(1-m.X[I,J,T])  
            return pe.Constraint.Skip
        m.linking111=pe.Constraint(m.I,m.J,m.T,rule=_linking1_11,doc='Linking constraint to guarantee operation at minimum processing time') 

        def _linking1_22(m,I,J,T):
            if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
                return -(m.varTime[I,J,T]-min_proc_time[I,J]) <= min_proc_time[I,J]*(1-m.X[I,J,T]) 
            return pe.Constraint.Skip 
        m.linking122=pe.Constraint(m.I,m.J,m.T,rule=_linking1_22,doc='Linking constraint to guarantee operation at minimum processing time') 
        # NOTE: In this case I already fixed B. I do not have to do it again
        # if fixedB:
        #     def _linking2_11(m,I,J,T):
        #         if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
        #             return m.B[I,J,T]-max_Capa[I, J, 0] <= (m.beta_max[I,J]-max_Capa[I, J, 0])*(1-m.X[I,J,T]) 
        #         return pe.Constraint.Skip 
        #     m.linking211=pe.Constraint(m.I,m.J,m.T,rule=_linking2_11,doc='Linking constraint to guarantee operation at maximum capacity') 

        #     def _linking2_22(m,I,J,T):
        #         if (I in m.I_dynamics and J in m.J_dynamics) or (I in m.I_distil and J in m.J_distil): 
        #             return -(m.B[I,J,T]-max_Capa[I, J, 0] )<= max_Capa[I, J, 0]*(1-m.X[I,J,T])  
        #         return pe.Constraint.Skip
        #     m.linking222=pe.Constraint(m.I,m.J,m.T,rule=_linking2_22,doc='Linking constraint to guarantee operation at maximum capacity') 
    else:
        def _linking1_11(m,I,J,T):
            return m.varTime[I,J,T]-min_proc_time[I,J] <= (kwargs['upper_t_h'][I,J]-min_proc_time[I,J])*(1-m.X[I,J,T])  
        m.linking111=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking1_11,doc='Linking constraint to guarantee operation at minimum processing time') 

        def _linking1_22(m,I,J,T):
            return -(m.varTime[I,J,T]-min_proc_time[I,J]) <= min_proc_time[I,J]*(1-m.X[I,J,T])  
        m.linking122=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking1_22,doc='Linking constraint to guarantee operation at minimum processing time') 
        # NOTE: In this case I already fixed B. I do not have to do it again
        # if fixedB:
        #     def _linking2_11(m,I,J,T):
        #         return m.B[I,J,T]-max_Capa[I, J, 0] <= (m.beta_max[I,J]-max_Capa[I, J, 0])*(1-m.X[I,J,T])  
        #     m.linking211=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking2_11,doc='Linking constraint to guarantee operation at maximum capacity') 

        #     def _linking2_22(m,I,J,T):
        #         return -(m.B[I,J,T]-max_Capa[I, J, 0] )<= max_Capa[I, J, 0]*(1-m.X[I,J,T])  
        #     m.linking222=pe.Constraint(m.I_dynamics,m.J_dynamics,m.T,rule=_linking2_22,doc='Linking constraint to guarantee operation at maximum capacity')         


    # DEACTIVATE SCHEDULING CONSTRAINTS
    m.E2_CAPACITY_LOW.deactivate()
    m.E2_CAPACITY_UP.deactivate()
    m.E3_BALANCE_INIT.deactivate()
    m.E1_UNIT.deactivate()
    m.E3_BALANCE.deactivate()
    m.X_Z_relation.deactivate()


    if with_distillation:
        if not last_stage_dynamic_cost:
            m.obj_scheduling.deactivate()
            def _min_t(m):
                return sum(sum(sum( m.X[I, J, T]*m.varTime[I,J,T] for J in m.J_dynamics)  for I in m.I_dynamics) for T in m.T)+sum(sum(sum( m.X[I, J, T]*m.varTime[I,J,T] for J in m.J_distil)  for I in m.I_distil)for T in m.T)
            m.min_t=pe.Objective(rule=_min_t,sense=pe.minimize) 
    else:
        if fixedB:
            if not last_stage_dynamic_cost:
                m.obj_scheduling.deactivate()
                def _min_t(m):
                    return sum(sum(sum( m.X[I, J, T]*m.varTime[I,J,T] for J in m.J_dynamics)  for I in m.I_dynamics) for T in m.T)
                m.min_t=pe.Objective(rule=_min_t,sense=pe.minimize) 
        else:
            # NOTE : This is atest to show that operation becomes infeasible due to the processing time wrongly selected by sequential approach
            m.linking3_1['T2','U3',5].deactivate()
            m.linking3_2['T2','U3',5].deactivate()
            m.obj_scheduling.deactivate()

            def _min_square(m):
                return (m.CC['T2','U3',5][m.N['T2','U3',5].last()]-m.CCDESIRED)**2
            m.min_squareobj=pe.Objective(rule=_min_square,sense=pe.minimize)

    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee) 

    # ACTIVATE SCHEDULING CONSTRAINTS
    m.E2_CAPACITY_LOW.activate()
    m.E2_CAPACITY_UP.activate()
    m.E3_BALANCE_INIT.activate()
    m.E1_UNIT.activate()
    m.E3_BALANCE.activate()
    m.X_Z_relation.activate()


    if with_distillation:
        save=generate_initialization(m=m,model_name='case_2_scheduling_and_dynamics_solution_with_distillation')
    else:
        save=generate_initialization(m=m,model_name='case_2_scheduling_and_dynamics_solution')

    if global_tee:
        print(" CPU time [s]:",time.perf_counter()-t_start)

    return m



def sequential_non_iterative_2(
    ext_logic,
    starting_point: list,
    model_function,
    kwargs,
    ext_dict,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    provide_starting_initialization: bool = False,
    feasible_model: str = '',
    subproblem_solver: str = 'dicopt',
    subproblem_solver_options: dict = {},
    tee: bool = False,
    global_tee: bool = True,
    rel_tol: float = 0,
    fixedB: bool=True, # DECIDES IF THE PROBLEM WILL BE SOLVED WITH A FIXED MAXIMUM CAPACITY HERUSTICA AT EVERY STAGE
    last_stage_dynamic_cost: bool=False): # DECIDES IF THE THIRD BLOCK WILL MINIMIZE COSTS. IF False, then the last stage simply uses dynamics found in processing time minimization stage

    """
    rate_beta_ub: Real, rate to decrease upper bound of beta
    rate_tau: integer, rate to increase external variables related to variable processing time

    """
    if global_tee:
        print('\nStarting Sequential Scheduling and control (non_iterative)')
        print('--------------------------------------------------------------------------')

    ### FIRST SOLVE MINIMIZATION OF PROCESSING TIMES!!!!
    kwargs['sequential']=True
    kwargs['max_capacity']=True
    m = model_function(**kwargs)
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict)
    if len(starting_point) != num_ext_var:
        print("The size of the initialization vector must be equal to "+str(num_ext_var))

    t_start = time.perf_counter()


    if provide_starting_initialization:
        m = initialize_model(m, from_feasible=True, feasible_model=feasible_model, json_path=None)


    m = external_ref_sequential_case_2(m=m,x=starting_point,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)

    # DEACTIVATE SCHEDULING CONSTRAINTS
    m.E2_CAPACITY_LOW.deactivate()
    m.E2_CAPACITY_UP.deactivate()
    m.E3_BALANCE_INIT.deactivate()
    m.E1_UNIT.deactivate()
    m.E3_BALANCE.deactivate()
    m.X_Z_relation.deactivate()

    m.C_TCP1.deactivate()
    m.C_TCP2.deactivate()
    m.C_TCP3.deactivate()
    m.C_TMC.deactivate()
    m.C_SALES.deactivate()

    m.obj_scheduling.deactivate()
    m.obj_dummy.deactivate()


    # Only consider one execution of dynamic tasks
    for Index in m.I_J:
            m.Nref[Index].fix(1)
            I=Index[0]
            J=Index[1]
            for T in m.T:           
                if T==m.T.first() and I in m.I_reactions and J in m.J_reactors:
                    m.X[I,J,T].fix(1)
                else:
                    m.X[I,J,T].fix(0)


    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee) 
    save=generate_initialization(m=m,model_name='partial_borrar')

    # retrieve maximum capacity
    max_Capa={}
    for I in m.I_reactions:
        for J in m.J_reactors:
            for T in m.T:
                if T==m.T.first():
                    max_Capa[I, J, T]=pe.value(m.B[I, J, T])   

    kwargs['max_capacity']=False
    m = model_function(**kwargs)
    m = external_ref_sequential_case_2(m=m,x=starting_point,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)

    # DEACTIVATE SCHEDULING CONSTRAINTS
    m.E2_CAPACITY_LOW.deactivate()
    m.E2_CAPACITY_UP.deactivate()
    m.E3_BALANCE_INIT.deactivate()
    m.E1_UNIT.deactivate()
    m.E3_BALANCE.deactivate()
    m.X_Z_relation.deactivate()

    m.C_TCP1.deactivate()
    m.C_TCP2.deactivate()
    m.C_TCP3.deactivate()
    m.C_TMC.deactivate()
    m.C_SALES.deactivate()

    m.obj_scheduling.deactivate()
    m.obj_dummy.deactivate()

    # Only consider one execution of dynamic tasks
    for Index in m.I_J:
            m.Nref[Index].fix(1)
            I=Index[0]
            J=Index[1]
            for T in m.T:           
                if T==m.T.first() and I in m.I_reactions and J in m.J_reactors:
                    m.X[I,J,T].fix(1)
                else:
                    m.X[I,J,T].fix(0)
    #fix operation at maximum capacity
    for I in m.I_reactions:
        for J in m.J_reactors:
            for T in m.T:
                if T==m.T.first():
                    m.B[I, J, T].fix(max_Capa[I, J, T]) 

    m=initialize_model(m,from_feasible=True,feasible_model='partial_borrar')  
    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee)    
    try:
        if kwargs['last_time_hours']==28:
            save=generate_initialization(m=m,model_name='case_1_28h_min_proc_time_solution')
        else:
            save=generate_initialization(m=m,model_name='case_1_min_proc_time_solution')
    except:
        save=generate_initialization(m=m,model_name='case_1_min_proc_time_solution')
    
    min_proc_time={}
    for I in m.I_reactions:
        for J in m.J_reactors:
            min_proc_time[I,J]=math.ceil(pe.value(m.varTime[I,J])/m.delta)*m.delta

    # Solve scheduling
    m = model_function(**kwargs)
    for I in m.I_reactions:
        for J in m.J_reactors:
            m.c_dCdtheta[I,J].deactivate()
            m.c_dTRdtheta[I,J].deactivate()                        
            m.c_dTJdtheta[I,J].deactivate()
            m.c_dIntegral_hotdtheta[I,J].deactivate()
            m.c_dIntegral_colddtheta[I,J].deactivate()
            m.Constant_control1[I,J].deactivate()                        
            m.Constant_control2[I,J].deactivate()
            # m.finalCon[I,J].deactivate()
            # m.finalTemp[I,J].deactivate()

    m.C_TCP3.deactivate()

    m.obj_scheduling.activate()
    m.obj.deactivate()
    m.obj_dummy.deactivate()

    def _linking1_11(m,I,J):
        return m.varTime[I,J]-min_proc_time[I,J] <= 0 
    m.linking111=pe.Constraint(m.I_reactions,m.J_reactors,rule=_linking1_11,doc='Linking constraint to guarantee operation at minimum processing time') 

    def _linking1_22(m,I,J):
        return -(m.varTime[I,J]-min_proc_time[I,J]) <=0
    m.linking122=pe.Constraint(m.I_reactions,m.J_reactors,rule=_linking1_22,doc='Linking constraint to guarantee operation at minimum processing time') 
    if fixedB:
        def _linking2_11(m,I,J,T):
            return m.B[I,J,T]-max_Capa[I, J, 0] <= (m.beta_max[I,J]-max_Capa[I, J, 0])*(1-m.X[I,J,T])  
        m.linking211=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking2_11,doc='Linking constraint to guarantee operation at maximum capacity') 

        def _linking2_22(m,I,J,T):
            return -(m.B[I,J,T]-max_Capa[I, J, 0] )<= max_Capa[I, J, 0]*(1-m.X[I,J,T])  
        m.linking222=pe.Constraint(m.I_reactions,m.J_reactors,m.T,rule=_linking2_22,doc='Linking constraint to guarantee operation at maximum capacity') 

    m=solve_with_minlp(m,transformation='bigm',minlp='cplex',timelimit=86400,gams_output=False,tee=tee,rel_tol=0)
    try:
        if kwargs['last_time_hours']==28:
            save=generate_initialization(m=m,model_name='case_1_28h_scheduling_solution')
        else:
            save=generate_initialization(m=m,model_name='case_1_scheduling_solution')
    except:
        save=generate_initialization(m=m,model_name='case_1_scheduling_solution')


    # retrieve capacity
    Capa={}
    for I in m.I:
        for J in m.J:
            for T in m.T:
                if round(pe.value(m.X[I,J,T]))==1:   
                    Capa[I, J, T]=min([pe.value(m.B[I, J, T]),m.beta_max[I,J]])  
                else:
                    Capa[I, J, T]=0 
    # retrieve ext vars
    Sol_found=[]
    for I in m.I_reactions:
        for J in m.J_reactors:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m.minTau[I,J]+1)
    for I_J in m.I_J:
        Sol_found.append(1+round(pe.value(m.Nref[I_J])))

    m = model_function(**kwargs)
    m = external_ref_sequential_case_2(m=m,x=Sol_found,extra_logic_function=ext_logic,dict_extvar=dict_extvar,mip_ref=mip_transformation,tee=False)

    m.obj_scheduling.deactivate()
    m.obj.deactivate()
    m.obj_dummy.deactivate()

    # TO AVOID THE ISSUE THAT CONOPT4 SOLVER ERRONEOUSLY DETECTS CPLEX SOLUTION AS INFEASIBLE, we fix capacities and processing times using an optimization problems
    def _min_square(m):
        return sum(sum(sum( ((m.B[I,J,T]-Capa[I, J, T])**2) for T in m.T) for J in m.J) for I in m.I)+sum(sum( (m.varTime[I,J]-min_proc_time[I,J])**2 for I in m.I_reactions)for J in m.J_reactors)
        # return sum(sum(sum( ((m.B[I,J,T]-Capa[I, J, T])**2) for T in m.T) for J in m.J) for I in m.I)-sum(sum( (m.varTime[I,J]) for I in m.I_reactions)for J in m.J_reactors)
    m.min_squareobj=pe.Objective(rule=_min_square,sense=pe.minimize)

    try:
        if kwargs['last_time_hours']==28:
            m=initialize_model(m,from_feasible=True,feasible_model='case_1_28h_scheduling_solution') 
        else:   
            m=initialize_model(m,from_feasible=True,feasible_model='case_1_scheduling_solution')  
    except:
        m=initialize_model(m,from_feasible=True,feasible_model='case_1_scheduling_solution') 
    # FIX DISCRETE SCHEDULING VARIABLES
    for v in m.component_objects(pe.Var, descend_into=True):
        if v.name=='X' or v.name=='Nref':
            for index in v:
                if index==None:
                    v.fix(round(pe.value(v)))
                else:
                    v[index].fix(round(pe.value(v[index])))

    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee) 

    # APPLY ECONOMIC OBJECTIVE FUNCTION CONSIDERING DYNAMIC TERM
    m.min_squareobj.deactivate()
    if last_stage_dynamic_cost:
        m.obj_scheduling.activate()
    else:
        def _min_t(m):
            return sum(sum( m.varTime[I,J] for I in m.I_reactions)for J in m.J_reactors)
        m.min_t=pe.Objective(rule=_min_t,sense=pe.minimize)       

    #NOW WE CAN FIX THE REMAINING SCHEDULING VARIABLES
    for v in m.component_objects(pe.Var, descend_into=True):
        if v.name=='Vreactor' or v.name=='B' or v.name=='S' or v.name=='varTime':
            for index in v:
                if index==None:
                    v.fix(pe.value(v))
                else:
                    v[index].fix(pe.value(v[index]))

    #SOLVE AGAIN
    m = solve_subproblem(m=m,subproblem_solver=subproblem_solver,subproblem_solver_options=subproblem_solver_options,timelimit=1000000000,gams_output=False,tee=tee) 


    if global_tee:
        print(" CPU time [s]:",time.perf_counter()-t_start)

    return m




def visualize_dsda(
    route: list = [],
    feas_x: list = [],
    feas_y: list = [],
    objs: list = [],
    k: str = '?',
    ext1_name: str = 'External variable 1',
    ext2_name: str = 'External variable 2'
):
    """
    Function that plots Discrete-Steepest Descend Algorithm for two external variables
    Args:
        route: List containing points evaluated in throughout iteration
        feas_x: List containing x-axis position of feasible points
        feas_y: List containing y-axis position of feasible points
        objs: List containing objective function of feasible points
        k: Type of neighborhood
        ext1_name: External variable 1 name
        ext2_name: External variable 2 name

    Returns:

    """

    X1, X2 = feas_x, feas_y
    cm = plt.cm.get_cmap('viridis_r')

    def drawArrow(A, B):
        plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], width=0.00005,
                  head_width=0.15, head_length=0.08, color='black', shape='full')

    for i in range(len(route)-1):
        drawArrow(route[i], route[i+1])

    sc = plt.scatter(X1, X2, s=80, c=objs, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.set_label('Objective function', rotation=90)
    title_string = 'D-SDA with k = '+k
    plt.title(title_string)
    plt.xlabel(ext1_name)
    plt.ylabel(ext2_name)
    plt.show()


def solve_complete_external_enumeration(
    model_function,
    model_args: dict,
    ext_dict: dict,
    ext_logic,
    mip_transformation: bool = False,
    transformation: str = 'bigm',
    feasible_model: str = '',
    points: list = [],
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    iter_timelimit: float = 10,
    timelimit: float = None,
    gams_output: bool = False,
    tee: bool = False,
    global_tee: bool = True,
    export_csv: bool = False
):
    """
    Function that computes complete enumeration using the external variable reformulation
    Args:
        model_function: function that returns GDP model to be solved
        model_args: Contains the argument values needed for model_function
        ext_dict: Dictionary with Boolean variables to be reformulated (keys) and their corresponding ordered sets (values). Both keys and values are pyomo objects.
        ext_logic: Function that returns a list of lists of the form [a,b], where a is an expressions of the reformulated Boolean variables and b is an equivalent Boolean or indicator variable (b<->a).
        mip_transformation: Whether to solve the enumeration using the external variables applied to the MIP problem insed of the GDP
        transformation: Which transformation to apply to the GDP 
        feasible_model: feasible model name to initialize
        points: list of points to carry on enumeration
        subproblem_solver: MINLP or NLP solver algorithm
        subproblem_solver_options: MINLP or NLP solver algorithm options
        iter_timelimit: time limit in seconds for the solve statement for each iteration
        timelimit: time limit in seconds for the algorithm
        gams_output: Determine keeping or not GAMS files
        tee: Display iteration output
        global_tee: Display D-SDA output
        export_csv: Export answer to a csv file
    Returns:
        m2_solved: Solved Pyomo Model

    """
    results = {}
    feasibles = {}
    t_start = time.perf_counter()
    csv_columns = ['Point', 'x', 'y', 'Objective',
                   'Status', 'Time', 'Global_Time']
    dict_data = []
    csv_file = 'compl_enum_'+str(feasible_model) + \
        '_'+str(subproblem_solver)+'.csv'

    m = model_function(**model_args)
    dict_extvar, num_ext_var, min_allowed, max_allowed = get_external_information(
        m, ext_dict, tee=global_tee)

    bounds = []
    for i in range(1, num_ext_var+1):
        bounds.append(list(range(min_allowed[i], max_allowed[i]+1)))

    if len(points) == 0:
        points = list(it.product(*bounds))

    if timelimit is None:
        timelimit = 1.5*iter_timelimit*len(points)

    if global_tee:
        print('\nStarting Complete Enumeration of External Variables')
        print('----------------------------------------------------------------------------------------------')

    for i in points:
        new_result = {}
        m = model_function(**model_args)
        m_init = initialize_model(
            m=m,
            from_feasible=True,
            feasible_model=feasible_model,
            json_path=None,
        )
        if mip_transformation:  # If you want a MIP reformulation, go ahead and use it
            csv_file = 'compl_enum_'+str(feasible_model) + \
                '_'+str(subproblem_solver)+'_' + transformation + '.csv'
            m_init, dict_extvar = extvars_gdp_to_mip(
                m=m,
                gdp_dict_extvar=dict_extvar,
                transformation=transformation,
            )
        m_fixed = external_ref(
            m=m_init,
            x=list(i),
            extra_logic_function=ext_logic,
            dict_extvar=dict_extvar,
            mip_ref=mip_transformation,
            tee=False,
        )
        t_remaining = min(iter_timelimit, timelimit -
                          (time.perf_counter() - t_start))
        if t_remaining < 0:  # No time remaining for optimization
            break
        m_solved = solve_subproblem(
            m=m_fixed,
            subproblem_solver=subproblem_solver,
            subproblem_solver_options=subproblem_solver_options,
            timelimit=t_remaining,
            gams_output=gams_output,
            tee=tee,
        )

        results[i] = (m_solved.dsda_status, pe.value(m_solved.obj))

        if global_tee:
            print('Evaluated:', list(i), ' |   Objective:', round(pe.value(m_solved.obj), 5), ' |   Global Time:', round(time.perf_counter()-t_start, 2),
                  ' |   Status:', m_solved.dsda_status)

        if m_solved.dsda_status == 'Optimal':
            feasibles[i] = float(pe.value(m_solved.obj))

        if export_csv:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            csv_file = os.path.join(dir_path, "../../results", csv_file)
            if m_solved.dsda_status != 'FBBT_Infeasible':
                new_result = {'Point': list(i), 'x': i[0], 'y': i[1], 'Objective': pe.value(
                    m_solved.obj), 'Status': m_solved.dsda_status, 'Time': m_solved.results.solver.user_time, 'Global_Time': time.perf_counter()-t_start}
                dict_data.append(new_result)
            try:
                with open(csv_file, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in dict_data:
                        writer.writerow(data)
            except IOError:
                print("I/O error")

        if time.perf_counter() - t_start > timelimit:
            break

    int_feasibles = {}
    for i in feasibles:
        if not isnan(feasibles[i]):
            int_feasibles[i] = feasibles[i]

    # If there are feasible integer combinations resolve with the best found
    if int_feasibles:
        minimum = min(int_feasibles, key=int_feasibles.get)
        m2 = model_function(**model_args)
        m2_init = initialize_model(
            m=m2,
            from_feasible=True,
            feasible_model=feasible_model,
            json_path=None,
        )
        if mip_transformation:  # If you want a MIP reformulation, go ahead and use it
            m2_init, dict_extvar = extvars_gdp_to_mip(
                m=m2_init,
                gdp_dict_extvar=dict_extvar,
                transformation=transformation,
            )
        m2_fixed = external_ref(
            m=m2_init,
            x=list(minimum),
            extra_logic_function=ext_logic,
            dict_extvar=dict_extvar,
            mip_ref=mip_transformation,
            tee=False,
        )
        m2_solved = solve_subproblem(
            m=m2_fixed,
            subproblem_solver=subproblem_solver,
            subproblem_solver_options=subproblem_solver_options,
            timelimit=iter_timelimit,
            gams_output=gams_output,
            tee=tee,
        )
        if not mip_transformation:  # Error generating json file with MINLP fixed problems
            _ = generate_initialization(m_solved)

        t_end = time.perf_counter()-t_start
        m2_solved.total_time = t_end

        print(m2_solved.results)
        if export_csv:
            final = {'Point': list(minimum), 'x': minimum[0], 'y': minimum[1], 'Objective': pe.value(
                m2_solved.obj), 'Status': 'Final', 'Time': m2_solved.results.solver.user_time, 'Global_Time': t_end}
            dict_data.append(final)
            try:
                with open(csv_file, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in dict_data:
                        writer.writerow(data)
            except IOError:
                print("I/O error")

        if global_tee:
            print(
                '----------------------------------------------------------------------------------------------')
            print('Objective:', round(pe.value(m2_solved.obj), 5))
            print('External variables:', list(minimum))
            print('Execution time [s]:', round(t_end, 2))

        return m2_solved

    return None
