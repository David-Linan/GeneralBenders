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
from pyomo.contrib.gdpopt.data_class import MasterProblemResult
from pyomo.core.base.misc import display
from pyomo.core.plugins.transform.logical_to_linear import \
    update_boolean_vars_from_binary
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolutionStatus, SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.opt.base.solvers import SolverFactory
import pyomo.dae as dae



def complementary_model(m,x):
    _tau_p={}

    _tau_p['Mix','Mix']=1.5

    _tau_p['R1','R_large']=x[0]+m.minTau['R1','R_large']-1 
    _tau_p['R1','R_small']=x[1]+m.minTau['R1','R_small']-1

    _tau_p['R2','R_large']=x[2]+m.minTau['R2','R_large']-1
    _tau_p['R2','R_small']=x[3]+m.minTau['R2','R_small']-1 

    _tau_p['R3','R_large']=x[4]+m.minTau['R3','R_large']-1 
    _tau_p['R3','R_small']=x[5]+m.minTau['R3','R_small']-1 

    _tau_p['Sep','Sep']=3 

    _tau_p['Pack1','Pack']=1.5 
    _tau_p['Pack2','Pack']=1.5 

    #TODO: the input info I am declaring here is in HOURS. Check that it makes sense with respect to the time discretization in reactors balances!!!!!!!
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,mutable=True,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(pe.value(m.tau_p[I,J])/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,mutable=True,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    #-----------Reactors dynamic models--------------------------------
    # !!! Assumption. Here I will create 6 continuous time grids, assuming that e.g., when R1 occurs in R_large, the task is always executed the same way (i.e., same tau)
    # !!! This means that initial conditions do not change and disturbances are the same whenever a task is executed multiple times in the same unit
    # !!! The six time grids stand for:
    # R_large-R1,R_large-R2,R_large-R3,R_small-R1,R_small-R2,R_small-R3
    # TODO: Energy balance has a volume term, hence energy balance is affected by batch size. This means that I must enforce that batch size is the same along time for every reactor-reaction pair. In this way my assumption will make sense    
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
    m.dCdt={} # Composition derivatives
    m.dTRdt={} #Reactor temperature derivatives
    m.dTJdt={} #Jacket temperature derivatives

    #Differential equations
    m.c_dCdt={}
    m.c_dTRdt={}
    m.c_dTJdt={}
    
    #Final constraint
    m.finalCon={}
    m.finalTemp={}      

    for I in m.I_reactions:
        m.Q_balance[I]=pe.Set(initialize=[Q for Q in m.Q if m.coef[I,Q]!=0],within=m.Q,doc='Species of interest for reaction I')
        setattr(m,'Q_balance_[%s]' %I,m.Q_balance[I])
        for J in m.J_reactors:
            m.N[I,J]=dae.ContinuousSet(bounds=(0,pe.value(m.tau_p[I,J])),doc='Continuous time set for reaction I in reactor J [h]') #TODO: chek units of time, are they consistent? should I use hours? 
            setattr(m,'N_[%s,%s]' %(I,J),m.N[I,J]) # TODO: I think the name of the pyomo object do not affect, because I can access these sets through dictionary m.N. Check if this is correct


            def _Cvar_bounds(m,N,Q):
                return (min([m.C_initial[I,Q],m.C_final[I,Q]]),max([m.C_initial[I,Q],m.C_final[I,Q]])) #TODO: Check bounds 
            m.Cvar[I,J]=pe.Var(m.N[I,J],m.Q_balance[I],within=pe.NonNegativeReals,bounds=_Cvar_bounds, doc='Component composition profile [kmol/m^3]') 
            setattr(m,'Cvar_(%s,%s)' %(I,J),m.Cvar[I,J]) 

            def _TRvar_bounds(m,N):
                return (m.T_R_initial[I],m.T_R_max[J]) #TODO: Check bounds 
            m.TRvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TRvar_bounds,doc='Reactor temperatrue profile [K]')
            setattr(m,'TRvar_(%s,%s)' %(I,J),m.TRvar[I,J])

            def _TJvar_bounds(m,N):
                return (m.T_J_initial[I],m.T_J_max[J]) #TODO: Check bounds 
            m.TJvar[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=_TJvar_bounds,doc='Jacket temperature profile [K]')
            setattr(m,'TJvar_(%s,%s)' %(I,J),m.TJvar[I,J])

            m.Fhot[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of heating fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fhot_(%s,%s)' %(I,J),m.Fhot[I,J])

            m.Fcold[I,J]=pe.Var(m.N[I,J],within=pe.NonNegativeReals,bounds=(0,m.F_max[J]),doc='Flow of cooling fluid [m^3/h]') #TODO: Check bounds 
            setattr(m,'Fcold_(%s,%s)' %(I,J),m.Fcold[I,J])

            m.dCdt[I,J] = dae.DerivativeVar(m.Cvar[I,J], withrespectto=m.N[I,J], doc='Derivative of composition')
            setattr(m,'dCdt_(%s,%s)' %(I,J),m.dCdt[I,J])

            m.dTRdt[I,J]=dae.DerivativeVar(m.TRvar[I,J], withrespectto=m.N[I,J], doc='Derivative of reactor temperature')
            setattr(m,'dTRdt_(%s,%s)' %(I,J),m.dTRdt[I,J])

            m.dTJdt[I,J]=dae.DerivativeVar(m.TJvar[I,J], withrespectto=m.N[I,J], doc='Derivative of jacket temperature')
            setattr(m,'dTJdt_(%s,%s)' %(I,J),m.dTJdt[I,J])

            def _dCdt(m,N,Q):
                if N == m.N[I,J].first(): 
                    return m.Cvar[I,J][N,Q] == m.C_initial[I,Q] # Initial condition
                else:                                         #This is what the author calls Rb
                    return m.dCdt[I,J][N,Q] == m.coef[I,Q]*   m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]) 
            m.c_dCdt[I,J] = pe.Constraint(m.N[I,J],m.Q_balance[I], rule=_dCdt)
            setattr(m,'c_dCdt_(%s,%s)' %(I,J),m.c_dCdt[I,J])


            def _dTRdt(m,N):
                if N == m.N[I,J].first():
                    return m.TRvar[I,J][N] == m.T_R_initial[I] #Initial condition
                else:
                    return m.dTRdt[I,J][N] == (((m.z[I]*pe.exp(-m.er[I]/m.TRvar[I,J][N])*pe.prod([m.Cvar[I,J][N,Q_2] for Q_2 in m.Q_balance[I] if m.coef[I,Q_2]<=-1]))*(-m.delta_h[I]))/(m.rho_R[I]*m.c_R[I]))+((m.ua[J]*( m.TJvar[I,J][N]- m.TRvar[I,J][N]))/(m.Vreactor[I,J]*m.rho_R[I]*m.c_R[I]))  
            m.c_dTRdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTRdt)
            setattr(m,'c_dTRdt_(%s,%s)' %(I,J),m.c_dTRdt[I,J])
            # m.c_dTRdt[I,J].pprint()

            def _dTJdt(m,N):
                if N == m.N[I,J].first():
                    return m.TJvar[I,J][N] == m.T_J_initial[I] #Initial condition
                else:
                    return m.dTJdt[I,J][N] == (((m.Fhot[I,J][N]*(m.T_H[J]-m.TJvar[I,J][N]))+(m.Fcold[I,J][N]*(m.T_C[J]-m.TJvar[I,J][N])))/(m.v_J[J]))+((m.ua[J]*(m.TRvar[I,J][N]-m.TJvar[I,J][N]))/(m.v_J[J]*m.rho_J[J]*m.c_J[J]))  
            m.c_dTJdt[I,J]=pe.Constraint(m.N[I,J],rule=_dTJdt)
            setattr(m,'c_dTJdt_(%s,%s)' %(I,J),m.c_dTJdt[I,J])
            
            
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

            # m.c_dCdt['R3','R_large'].display()
            # m.Cvar['R3','R_large'].display()  
            # m.Q_balance['R1'].pprint()
            # m.Q_balance['R2'].pprint()
            # m.Q_balance['R3'].pprint()

# # ----------Scheduling Constraints that depend on disjunctions-----------------------------------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-pe.value(m.tau[I,J])+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')
    #m.E1_UNIT.display()

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-pe.value(m.tau[I,J])] for J in m.J if m.I_i_j_prod[I,J]==1 and T-pe.value(m.tau[I,J])>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')


    # # -------Discretization---------------------------------------------------
    # discretizer = pe.TransformationFactory('dae.finite_difference')
    # discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # # discretizer = TransformationFactory('dae.collocation')
    # # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    #Constant control actions
    m.Constant_control1={}
    m.Constant_control2={}
    keep_constant_Fhot=6 #Keep Fhot constant every three discretization points
    keep_constant_Fcold=6 #Keep Fcold constant every three discretization points 


    discretizer = pe.TransformationFactory('dae.collocation') #dae.finite_difference is also possible

    for I in m.I_reactions:
        for J in m.J_reactors:        #TODO: Depending on selected variable time the number of discretization points must change accordingly
            discretizer.apply_to(m, nfe=5, ncp=3, wrt=m.N[I,J], scheme='LAGRANGE-RADAU') #if using finite differences, I can use FORWARD, BACKWARD, ETC
            # m=discretizer.reduce_collocation_points(m,var=m.Fcold[I,J],ncp=1,contset=m.N[I,J]) %TODO: NOT WORKING, HELP !!
            
            
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
        return math.floor((m.T.__len__()-1)/pe.value(m.tau[I,J]))  #TODO: CHANGE THIS IF I USE MY OWN FORMULATION
    m.lastN=pe.Param(m.I_J,initialize=_lastN,doc='last element for subsets of ordered set')
    # m.lastN.display()
    def _Nref_bounds(m,I,J):
        return (0,m.lastN[I,J])
    m.Nref=pe.Var(m.I_J,within=pe.Integers,bounds=_Nref_bounds,doc='reformulation variables from 0 to lastN')
    
    def _X_Z_relation(m,I,J):
        return sum(m.X[I,J,T] for T in m.T)==m.Nref[I,J]
    m.X_Z_relation=pe.Constraint(m.I_J,rule=_X_Z_relation,doc='constraint that specifies the relationship between Integer and binary variables')   

    def _obj(m): #TODO: CONSIDER OTHER TERMS
        # return sum(sum(sum(  m.fixed_cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
        return sum(sum( pe.value(m.tau[I,J]) for J in m.J_reactors) for I in m.I_reactions)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)    
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
    m=complementary_model(m,x)
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
    fbbt(m)


def solve_subproblem(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 1e-3,
) -> pe.ConcreteModel():
    """
    Function that checks feasibility and subproblem model.
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

    try:
        # Feasibility and preprocessing checks
        preprocess_problem(m, simple=True)

    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        return m

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
    opt = SolverFactory(solvername, solver=subproblem_solver)
    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options,
                          skip_trivial_constraints=True,
                          )

    m.dsda_usertime = m.results.solver.user_time

    # Assign D-SDA status
    if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
        m.dsda_status = 'Evaluated_Infeasible'
    else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
        m.dsda_status = 'Optimal'
    # if m.results.solver.termination_condition == 'locallyOptimal' or m.results.solver.termination_condition == 'optimal' or m.results.solver.termination_condition == 'globallyOptimal':
    #     m.dsda_status = 'Optimal'

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
                              solver=minlp, warmstart=True, tee=tee, **minlp_options),
                          mip_presolve=True, #True is the default
                          init_strategy='set_covering',#'fix_disjuncts'##'set_covering'#
                          #   set_cover_iterlim=0,
                          #iterlim#=1000,
                          force_subproblem_nlp=False,
                          #subproblem_presolve=False
                          #   bound_tolerance=rel_tol
                          #   calc_disjunctive_bounds=True
                          )
    # update_boolean_vars_from_binary(m)
    return m


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
                tee=tee)
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
                if not improve:
                    # We want a minimum improvement in the first found solution
                    if (fmin - act_obj) > min_improve or (fmin - act_obj)/(abs(fmin)+epsilon) > min_improve_rel:
                        fmin = act_obj
                        best_var = temp[i]
                        best_dir = i
                        best_dist = dist
                        improve = True
                        best_path = generate_initialization(
                            m_solved, starting_initialization=False, model_name='best')
                else:
                    # We want slightly worse solutions if the distance is larger
                    if (((act_obj - fmin) < abs_tol) or ((act_obj - fmin)/(abs(fmin)+epsilon) < rel_tol)) and dist >= best_dist:
                        fmin = act_obj
                        best_var = temp[i]
                        best_dir = i
                        best_dist = dist
                        improve = True
                        best_path = generate_initialization(
                            m_solved, starting_initialization=False, model_name='best')

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
    m_solved = solve_subproblem(
        m=m_fixed,
        subproblem_solver=subproblem_solver,
        subproblem_solver_options=subproblem_solver_options,
        timelimit=iter_timelimit,
        gams_output=gams_output,
        tee=tee,
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
    else:
        return "Enter a valid neighborhood ('Infinity' or '2')"

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
