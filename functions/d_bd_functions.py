from math import fabs
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
from functions.cuts_functions import convex_clousure,initialization_sampling_naive
from functions.dsda_functions import get_external_information,external_ref,initialize_model,generate_initialization, solve_subproblem, solve_subproblem_aprox,solve_with_minlp
from functions.feasibility_functions import feasibility_1,feasibility_2,feasibility_2_modified,feasibility_1_aprox,feasibility_2_aprox
import time
import random
from itertools import product
import warnings

def solve_subproblem_and_neighborhood_FEAS1(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,model_fun,kwargs):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors

    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        m_fixed = external_ref(m=model,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved,_=feasibility_1(m_fixed)
        #print(m_solved.dsda_status)
        #if m_solved.dsda_status=='Optimal':
        status[0]=1
        generated_dict[tuple(x)]=m_solved
        #else:
        #    status[0]=0
        #    generated_dict[tuple(x)]=infinity_val
    if generated_dict[tuple(x)]!=0:
        #solve neighborhood (only if central point was INfeasible)
        if status[0]==1:
            count=0 #count to add elements to status
            for j in neigh:    #TODO TRY TO IMPROVE THIS FOR USING UPPER AND LOWER BOUNDS FOR EXTERNAL VARIABLES!!!!!!!!!!!!!!!!!!!! SO FAR THIS IS BEING EVALUATED WITH FBBT
                count=count+1
                current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
                #print(current_value)
                if tuple(current_value) in Internaldata: #If subproblem was already solved
                    if Internaldata[tuple(current_value)]!=infinity_val:
                        status[count]=1
                    else:
                        status[count]=0
                    generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
                else: #If subproblem has not been solved yet
                    #1: fix external variables
                    #print(current_value)
                    model = model_fun(**kwargs)
                    m_fixed2 = external_ref(m=model,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                    m_solved2,_=feasibility_1(m_fixed2)


                    #print(m_solved2.dsda_status)
    #                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=m_solved2
                    # else:
                    #     status[count]=0
                    #     generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                    #print(pe.value(m_solved2.obj))
                    if m_solved2==0: #stop if a feasible solution is found
                        break        
    return generated_dict


def solve_subproblem_and_neighborhood_FEAS2(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,first_path,model_fun,kwargs):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldata: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors
    init_path=first_path
    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        model=initialize_model(m=model,json_path=first_path)
        m_fixed = external_ref(m=model,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved,_,_=feasibility_2(m_fixed,sub_solver,infinity_val)
        #print(m_solved.dsda_status)
        #if m_solved.dsda_status=='Optimal':
        status[0]=1
        generated_dict[tuple(x)]=m_solved
        # else:
        #     status[0]=0
        #     generated_dict[tuple(x)]=infinity_val
    if generated_dict[tuple(x)]!=0:
        #solve neighborhood (only if central point was infeasible)
        if status[0]==1:
            count=0 #count to add elements to status
            for j in neigh:    #TODO TRY TO IMPROVE THIS FOR USING UPPER AND LOWER BOUNDS FOR EXTERNAL VARIABLES!!!!!!!!!!!!!!!!!!!! SO FAR THIS IS BEING EVALUATED WITH FBBT
                count=count+1
                current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
                #print(current_value)
                if tuple(current_value) in Internaldata: #If subproblem was already solved
                    if Internaldata[tuple(current_value)]!=infinity_val:
                        status[count]=1
                    else:
                        status[count]=0
                    generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
                else: #If subproblem has not been solved yet
                    #1: fix external variables
                    #print(current_value)
                    model2 = model_fun(**kwargs)
                    model2=initialize_model(m=model2,json_path=first_path)
                    m_fixed2 = external_ref(m=model2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                    m_solved2,_,_=feasibility_2(m_fixed2,sub_solver,infinity_val)
                    #print(m_solved2.dsda_status)
                    # if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=m_solved2
                    # else:
                    #     status[count]=0
                    #     generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                    #print(pe.value(m_solved2.obj))
                
                    if m_solved2==0: #stop if a feasible solution is found
                        break            

    return generated_dict,init_path

def solve_subproblem_and_neighborhood_FEAS1_aprox(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,model_fun,kwargs,tee: bool=True):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors

    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        m_fixed = external_ref(m=model,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved,_=feasibility_1_aprox(m_fixed)
        #print(m_solved.dsda_status)
        #if m_solved.dsda_status=='Optimal':
        status[0]=1
        generated_dict[tuple(x)]=m_solved
        if tee:
            print('Evaluated:', x, '   |   Objective:', round(m_solved, 5))
        #else:
        #    status[0]=0
        #    generated_dict[tuple(x)]=infinity_val
    if generated_dict[tuple(x)]!=0:
        #solve neighborhood (only if central point was INfeasible)
        if status[0]==1:
            count=0 #count to add elements to status
            for j in neigh:    #TODO TRY TO IMPROVE THIS FOR USING UPPER AND LOWER BOUNDS FOR EXTERNAL VARIABLES!!!!!!!!!!!!!!!!!!!! SO FAR THIS IS BEING EVALUATED WITH FBBT
                count=count+1
                current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
                #print(current_value)
                if tuple(current_value) in Internaldata: #If subproblem was already solved
                    if Internaldata[tuple(current_value)]!=infinity_val:
                        status[count]=1
                    else:
                        status[count]=0
                    generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
                else: #If subproblem has not been solved yet
                    #1: fix external variables
                    #print(current_value)
                    model = model_fun(**kwargs)
                    m_fixed2 = external_ref(m=model,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                    m_solved2,_=feasibility_1_aprox(m_fixed2)


                    #print(m_solved2.dsda_status)
    #                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=m_solved2
                    if tee:
                        print('Evaluated:', current_value, '   |   Objective:', round(m_solved2, 5))
                    # else:
                    #     status[count]=0
                    #     generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                    #print(pe.value(m_solved2.obj))
                    if m_solved2==0: #stop if a feasible solution is found
                        break        
    return generated_dict

def solve_subproblem_and_neighborhood_FEAS2_aprox(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,first_path,model_fun,kwargs,tee:bool=True,new_case: bool=False, with_distillation: bool=False,lower_bounds: dict={},upper_bounds: dict={}):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldata: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors
    init_path=first_path
    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
        source={}
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        model=initialize_model(m=model,json_path=first_path)
        m_fixed = external_ref(m=model,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved,_,_,source=feasibility_2_aprox(m_fixed,sub_solver,infinity_val,new_case=new_case,with_distillation=with_distillation)
        #print(m_solved.dsda_status)
        #if m_solved.dsda_status=='Optimal':
        if tee:
            print('Evaluated:', x, '   |   Objective:', round(m_solved, 5),'   |   Infeasibility:',source)

        if len(source)==0:
            status[0]=0
            generated_dict[tuple(x)]=m_solved
        else:
            status[0]=1
            generated_dict[tuple(x)]=m_solved
            if 'Infeasible' in source.values():  #TODO: this is not general!!!! I am assuming that I have not evaluated all tau values and that feasibility will be eventually attained
                return generated_dict,init_path,source
        # else:
        #     status[0]=0
        #     generated_dict[tuple(x)]=infinity_val
    if generated_dict[tuple(x)]!=0:
        #solve neighborhood (only if central point was infeasible)
        if status[0]==1:
            count=0 #count to add elements to status
            for j in (jj for jj in neigh if np.all(np.array(x)+np.array(neigh[jj])>=np.array([lower_bounds[k] for k in lower_bounds.keys()]))  and np.all(np.array(x)+np.array(neigh[jj])<=np.array([upper_bounds[k] for k in lower_bounds.keys()]))): 
                count=count+1
                current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
                #print(current_value)
                if tuple(current_value) in Internaldata: #If subproblem was already solved
                    if Internaldata[tuple(current_value)]!=infinity_val:
                        status[count]=1
                    else:
                        status[count]=0
                    generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
                else: #If subproblem has not been solved yet
                    #1: fix external variables
                    #print(current_value)
                    model2 = model_fun(**kwargs)
                    model2=initialize_model(m=model2,json_path=first_path)
                    m_fixed2 = external_ref(m=model2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                    m_solved2,_,_,source2=feasibility_2_aprox(m_fixed2,sub_solver,infinity_val)
                    #print(m_solved2.dsda_status)
                    # if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=m_solved2
                    if tee:
                        print('Evaluated:', current_value, '   |   Objective:', round(m_solved2, 5),'   |   Infeasibility:',source2)
                    # else:
                    #     status[count]=0
                    #     generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                    #print(pe.value(m_solved2.obj))
                
                    if m_solved2==0: #stop if a feasible solution is found
                        break            

    return generated_dict,init_path,source

def solve_subproblem_and_neighborhood(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,init_path,model_fun,kwargs,sub_solver_opt: dict={},tee:bool=False, rel_tol: float=0):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors

    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        #2: Initialize model
        m_initialized=initialize_model(m=model,json_path=init_path)
        m_fixed = external_ref(m=m_initialized,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved=solve_subproblem(m=m_fixed, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False,rel_tol=rel_tol)
        #print(m_solved.dsda_status)
        if m_solved.dsda_status=='Optimal':
            status[0]=1
            generated_dict[tuple(x)]=pe.value(m_solved.obj)
            init_path = generate_initialization(m=m_solved)
            if tee:
                print('Evaluated:', x, '   |   Objective:', round(pe.value(m_solved.obj), 5))
        else:
            status[0]=0
            generated_dict[tuple(x)]=infinity_val
    #solve neighborhood (only if central point was feasible)
    if status[0]==1:
        if tee:
            print()
            print('Neighbor search around:', x)
        count=0 #count to add elements to status
        for j in neigh:    #TODO TRY TO IMPROVE THIS FOR USING UPPER AND LOWER BOUNDS FOR EXTERNAL VARIABLES!!!!!!!!!!!!!!!!!!!! SO FAR THIS IS BEING EVALUATED WITH FBBT, BUT THIS IS PROBLEMATIC BECAUSE I MUST DELETE DISJUNCTIONS FROM THE CODE (JUST LEAVE DISJUNCTS)
            count=count+1
            current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
            #print(current_value)
            if tuple(current_value) in Internaldata: #If subproblem was already solved
                if Internaldata[tuple(current_value)]!=infinity_val:
                    status[count]=1
                else:
                    status[count]=0
                generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
            else: #If subproblem has not been solved yet
                #1: fix external variables
                #print(current_value)
                model = model_fun(**kwargs)
                m_initialized2=initialize_model(m=model,json_path=init_path)
                m_fixed2 = external_ref(m=m_initialized2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved2=solve_subproblem(m=m_fixed2, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False,rel_tol=rel_tol)
                #print(m_solved2.dsda_status)
                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=pe.value(m_solved2.obj)
                    if tee:
                        print('Evaluated:', current_value, '   |   Objective:', round(pe.value(m_solved2.obj), 5))
                else:
                    status[count]=0
                    generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                #print(pe.value(m_solved2.obj))
    return generated_dict,init_path,m_solved

def solve_subproblem_and_neighborhood_except(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,init_path,model_fun,kwargs,sub_solver_opt: dict={},tee:bool=False,rel_tol: float=0,new_case: bool=False, with_distillation: bool=False):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors

    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        #2: Initialize model
        m_initialized=initialize_model(m=model,json_path=init_path)
        m_fixed = external_ref(m=m_initialized,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved=solve_subproblem(m=m_fixed, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False,rel_tol=rel_tol)
        #print(m_solved.dsda_status)
        if m_solved.dsda_status=='Optimal':
            status[0]=1
            generated_dict[tuple(x)]=pe.value(m_solved.obj)
            init_path = generate_initialization(m=m_solved)
            if tee:
                print('Evaluated:', x, '   |   Objective:', round(pe.value(m_solved.obj), 5))
        else:
            status[0]=0
            generated_dict[tuple(x)]=infinity_val
    #solve neighborhood (only if central point was feasible)
    if status[0]==1:
        if tee:
            print()
            print('Neighbor search around:', x)
        count=0 #count to add elements to status
        for j in neigh:    #TODO TRY TO IMPROVE THIS FOR USING UPPER AND LOWER BOUNDS FOR EXTERNAL VARIABLES!!!!!!!!!!!!!!!!!!!! SO FAR THIS IS BEING EVALUATED WITH FBBT, BUT THIS IS PROBLEMATIC BECAUSE I MUST DELETE DISJUNCTIONS FROM THE CODE (JUST LEAVE DISJUNCTS)
            count=count+1
            current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
            #print(current_value)
            if tuple(current_value) in Internaldata: #If subproblem was already solved
                if Internaldata[tuple(current_value)]!=infinity_val:
                    status[count]=1
                else:
                    status[count]=0
                generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
            else: #If subproblem has not been solved yet
                #1: fix external variables
                #print(current_value)
                model = model_fun(**kwargs)
                m_initialized2=initialize_model(m=model,json_path=init_path)
                m_fixed2 = external_ref(m=m_initialized2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved2=solve_subproblem(m=m_fixed2, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False, rel_tol=rel_tol)
                #print(m_solved2.dsda_status)
                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=pe.value(m_solved2.obj)
                    if tee:
                        print('Evaluated:', current_value, '   |   Objective:', round(pe.value(m_solved2.obj), 5))
                else:
                    status[count]=0
                    generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                #print(pe.value(m_solved2.obj))
    return generated_dict,init_path

def solve_subproblem_and_neighborhood_aprox(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,init_path,model_fun,kwargs,sub_solver_opt: dict={},tee:bool=False,best_sol: float=1e+8, rel_tol: float=0, new_case: bool=False, with_distillation: bool=False,lower_bounds: dict={},upper_bounds: dict={}):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors

    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        #2: Initialize model
        m_initialized=initialize_model(m=model,json_path=init_path)
        m_fixed = external_ref(m=m_initialized,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved=solve_subproblem_aprox(m=m_fixed, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False,rel_tol=rel_tol,best_sol=best_sol,new_case=new_case, with_distillation=with_distillation)
        if m_solved.best_sol<=best_sol:
            best_sol=m_solved.best_sol
        #print(m_solved.dsda_status)
        if m_solved.dsda_status=='Optimal':
            status[0]=1
            generated_dict[tuple(x)]=pe.value(m_solved.obj)
            init_path = generate_initialization(m=m_solved)
            if tee:
                print('Evaluated:', x, '   |   Objective:', round(pe.value(m_solved.obj), 5))
        else:

            if m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                status[0]=1
                generated_dict[tuple(x)]=pe.value(m_solved.obj_scheduling)
            else:
                status[0]=0
                generated_dict[tuple(x)]=infinity_val

            if tee:
                if m_solved.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('Pruned:', x, '   |   Lower bound problem infeasible')                    
                elif m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                    print('Pruned:', x, '   |   No improvement expected   |   Objective:', round(pe.value(m_solved.obj_scheduling), 5)) 
    #solve neighborhood (only if central point was feasible)
    if status[0]==1:
        if tee:
            print()
            print('Neighbor search around:', x)
        count=0 #count to add elements to status


        for j in (jj for jj in neigh if np.all(np.array(x)+np.array(neigh[jj])>=np.array([lower_bounds[k] for k in lower_bounds.keys()]))  and np.all(np.array(x)+np.array(neigh[jj])<=np.array([upper_bounds[k] for k in lower_bounds.keys()]))): 
            count=count+1
            current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
            #print(current_value)
            if tuple(current_value) in Internaldata: #If subproblem was already solved
                if Internaldata[tuple(current_value)]!=infinity_val:
                    status[count]=1
                else:
                    status[count]=0
                generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
            else: #If subproblem has not been solved yet
                #1: fix external variables
                #print(current_value)
                model = model_fun(**kwargs)
                m_initialized2=initialize_model(m=model,json_path=init_path)
                m_fixed2 = external_ref(m=m_initialized2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved2=solve_subproblem_aprox(m=m_fixed2, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False,rel_tol=rel_tol,best_sol=best_sol,new_case=new_case, with_distillation=with_distillation)
                if m_solved2.best_sol<=best_sol:
                    best_sol=m_solved2.best_sol
                    m_solved.best_sol=best_sol
                #print(m_solved2.dsda_status)
                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=pe.value(m_solved2.obj)
                    if tee:
                        print('Evaluated:', current_value, '   |   Objective:', round(pe.value(m_solved2.obj), 5))
                else:
                    
                    if m_solved2.pruned_Status=='Pruned_NoImprovementExpected':
                        status[count]=1
                        generated_dict[tuple(current_value)]=pe.value(m_solved2.obj_scheduling)
                    else:
                        status[count]=0
                        generated_dict[tuple(current_value)]=infinity_val   
                  
                    if tee:
                        if m_solved2.pruned_Status=='Pruned_SchedulingInfeasible':
                            print('Pruned:', current_value, '   |   Lower bound problem infeasible')                    
                        elif m_solved2.pruned_Status=='Pruned_NoImprovementExpected':
                            print('Pruned:', current_value, '   |   No improvement expected   |   Objective:', round(pe.value(m_solved2.obj_scheduling), 5)) 
                #print(pe.value(m_solved2.obj))
    return generated_dict,init_path,m_solved
#TODO: use the neighborhoo verification in other solve_subproblem and neighborhood functions!!!
def solve_subproblem_and_neighborhood_aprox_except(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,init_path,model_fun,kwargs,sub_solver_opt: dict={},tee:bool=False,best_sol: float=1e+8, rel_tol: float=0, new_case: bool=False, with_distillation: bool=False,lower_bounds: dict={},upper_bounds: dict={}):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors

    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = model_fun(**kwargs)
        #2: Initialize model
        m_initialized=initialize_model(m=model,json_path=init_path)
        m_fixed = external_ref(m=m_initialized,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved=solve_subproblem_aprox(m=m_fixed, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False, rel_tol=rel_tol,best_sol=best_sol,new_case=new_case, with_distillation=with_distillation)
        if m_solved.best_sol<=best_sol:
            best_sol=m_solved.best_sol
        #print(m_solved.dsda_status)
        if m_solved.dsda_status=='Optimal':
            status[0]=1
            generated_dict[tuple(x)]=pe.value(m_solved.obj)
            init_path = generate_initialization(m=m_solved)
            if tee:
                print('Evaluated:', x, '   |   Objective:', round(pe.value(m_solved.obj), 5))
        else:

            if m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                status[0]=1
                generated_dict[tuple(x)]=pe.value(m_solved.obj_scheduling)
            else:
                status[0]=0
                generated_dict[tuple(x)]=infinity_val

            if tee:
                if m_solved.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('Pruned:', x, '   |   Lower bound problem infeasible')                    
                elif m_solved.pruned_Status=='Pruned_NoImprovementExpected':
                    print('Pruned:', x, '   |   No improvement expected   |   Objective:', round(pe.value(m_solved.obj_scheduling), 5)) 
    #solve neighborhood (only if central point was feasible)
    if status[0]==1:
        if tee:
            print()
            print('Neighbor search around:', x)
        count=0 #count to add elements to status
        for j in (jj for jj in neigh if np.all(np.array(x)+np.array(neigh[jj])>=np.array([lower_bounds[k] for k in lower_bounds.keys()]))  and np.all(np.array(x)+np.array(neigh[jj])<=np.array([upper_bounds[k] for k in lower_bounds.keys()]))):   
            count=count+1
            current_value=np.array(x)+np.array(neigh[j])    #value of external variables for current neighbor
            #print(current_value)
            if tuple(current_value) in Internaldata: #If subproblem was already solved
                if Internaldata[tuple(current_value)]!=infinity_val:
                    status[count]=1
                else:
                    status[count]=0
                generated_dict[tuple(current_value)]=Internaldata[tuple(current_value)]
            else: #If subproblem has not been solved yet
                #1: fix external variables
                #print(current_value)
                model = model_fun(**kwargs)
                m_initialized2=initialize_model(m=model,json_path=init_path)
                m_fixed2 = external_ref(m=m_initialized2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved2=solve_subproblem_aprox(m=m_fixed2, subproblem_solver=sub_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False, rel_tol=rel_tol,best_sol=best_sol,new_case=new_case, with_distillation=with_distillation)
                if m_solved2.best_sol<=best_sol:
                    best_sol=m_solved2.best_sol
                #print(m_solved2.dsda_status)
                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=pe.value(m_solved2.obj)
                    if tee:
                        print('Evaluated:', current_value, '   |   Objective:', round(pe.value(m_solved2.obj), 5))
                else:
                    
                    if m_solved2.pruned_Status=='Pruned_NoImprovementExpected':
                        status[count]=1
                        generated_dict[tuple(current_value)]=pe.value(m_solved2.obj_scheduling)
                    else:
                        status[count]=0
                        generated_dict[tuple(current_value)]=infinity_val   
                  
                    if tee:
                        if m_solved2.pruned_Status=='Pruned_SchedulingInfeasible':
                            print('Pruned:', current_value, '   |   Lower bound problem infeasible')                    
                        elif m_solved2.pruned_Status=='Pruned_NoImprovementExpected':
                            print('Pruned:', current_value, '   |   No improvement expected   |   Objective:', round(pe.value(m_solved2.obj_scheduling), 5)) 
                #print(pe.value(m_solved2.obj))
    return generated_dict,init_path

def build_master(num_ext,lower_b,upper_b,current,stage,D,use_random: bool=False):
    """
    Function that builds the master problem

    use_random: True if a random point will be generated for initializations when required. False if you want to use the deterministric strategy
        
    """
    initial={}
    randomp=[]
    if stage==1 or stage==2: #generate random number different from current value and points evaluated so far
        initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value
        # if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_b[n_e] for   n_e in lower_b.keys()) and all(el[n_e-1]<=upper_b[n_e] for   n_e in lower_b.keys()) ]))-float(math.prod(upper_b[n_e]-lower_b[n_e]+1 for n_e in lower_b)))<=0.01: #if every point has been evaluated
        #     initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value
        # else:
        #     if use_random:
        #         #Generate random numbers
        #         while True:
        #             randomp=[random.randint(lower_b[n_e],upper_b[n_e]) for n_e in lower_b.keys()]  #This allows to consider difficult problems where, e.g., there is a big infeasible region with the same objective function values.
        #             if all([np.linalg.norm(np.array(randomp)-np.array(list(i)))>=0.1 for i in list(D.keys())]):
        #                 initial={n_e:randomp[n_e-1] for n_e in lower_b.keys()}
        #                 break
        #     else:
        #         #generate nonrandom numbers. It is better to go to the closest point that has not been evaluated (using e.g. a lexicographical ordering).  
        #         arrays=[range(lower_b[n_e],upper_b[n_e]+1) for n_e in lower_b.keys()]

        #         cart_prduct=list(product(*arrays)) #cartesian product, this also requires a lot of memory

        #         #TODO: after the cartesian product, I am organizing this with respect to the current value using a distance metric. Note that I can aslo explore points that are far away in the future.
        #         cart_prduct_sorted=cart_prduct#sorted(cart_prduct, key=lambda x: np.linalg.norm(np.array(list(x))-np.array(current)      )      ) #I am sorting to evaluate the closests point. I can also sort to evaluate the one that is far away (exploration!!!!!)
        #         for j in cart_prduct_sorted:
        #             non_randomp=list(j)
        #             if all([np.linalg.norm(np.array(non_randomp)-np.array(list(i)))>=0.1 for i in list(D.keys())]):
        #                 initial={n_e:non_randomp[n_e-1] for n_e in lower_b.keys()}
        #                 break
    else:
        initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value

    #print(initial)
    #Model
    m=pe.ConcreteModel(name='Master_problem')
    m.extset=pe.RangeSet(1,num_ext,1,doc='Set to organize external variables')
    #External variables
    def _boundsRule(m,extset):
        return (lower_b[extset],upper_b[extset])
    def _initialRule(m,extset):
        return initial[extset]    
    m.x=pe.Var(m.extset,within=pe.Integers,bounds=_boundsRule,initialize=_initialRule)

    # m.x1=pe.Var(within=pe.Integers, bounds=(lower_b[1],upper_b[1]),initialize=initial[1])
    # m.x2=pe.Var(within=pe.Integers, bounds=(lower_b[2],upper_b[2]),initialize=initial[2])

    #Known constraints (assumption!!! I know constraints a priori)
    #m.known=pe.Constraint(expr=m.x1-m.x2>=7)
    #m.known2=pe.Constraint(expr=m.x1>=9)
    #m.known3=pe.Constraint(expr=m.x2<=9)

    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()

    def obj_rule(m):
        return m.zobj 
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)
    notevaluated=[round(k) for k in initial.values()]
    return m,notevaluated

def run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):
# IMPORTANT!!!!: IF INCLUDING known_solutions, MAKE SURE THAT THE INITIALIZATION IS FEASIBLE 
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=1 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 

    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #Initialize with user provided initialization
    model = model_fun(**kwargs)
    init_path = generate_initialization(m=model) #user provided initialization (TODO: CHECK, BUT I THINK THIS IS THE USER PROVIDED, TODO: change this line if another strategy to generate initialization will be considered)
    
    if use_multi_start==False:
        #test initialization 
        model=initialize_model(m=model,json_path=init_path)
        m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        #original line
        m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False)
        #m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver,subproblem_solver_options= {'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver conopt4\n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}, timelimit=10000, tee=False)
        #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: uncomment last line for nonlinear scheduling problem      
        
        if m_init_solved.dsda_status=='FBBT_Infeasible':
            initial_Stage=1   
        if m_init_solved.dsda_status=='Evaluated_Infeasible':
            initial_Stage=2        
        if m_init_solved.dsda_status=='Optimal':
            init_path = generate_initialization(m=m_init_solved)
            initial_Stage=3
        if tee==True:
            print('Problem will be initialized at stage '+str(initial_Stage))    
    else:
        if tee==True:
            print('Executing multi-start...')
        #execute multistart (if required)
        D_feasibility1={} #optimal infeasibility 1 measure, infinity otherwise
        D_feasibility2={} #optimal infeasibility 2 measure, infinity otherwise
        D_optimality={}# optimal objective for feasible, infinity otherwise
        best_solution_value=infinity_val  #best solution updated through the multistart
        random_points_number=n_points_multstart

        if random_points_number >= 2:
            sampled_points=initialization_sampling_naive(random_points_number,lower_bounds,upper_bounds)
            #solve random points
            for i in sampled_points:
                model = model_fun(**kwargs)
                m_init=initialize_model(m=model,json_path=init_path)
                m_fix = external_ref(m=m_init,x=i,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)          
                m_sol_feas1,_=feasibility_1(m_fix)
                if m_sol_feas1!=0:
                    D_feasibility1[tuple(i)]=m_sol_feas1
                    D_feasibility2[tuple(i)]=infinity_val
                    D_optimality[tuple(i)]=infinity_val
                else:
                    m_sol_feas2,_=feasibility_2_modified(m_fix,nlp_solver,infinity_val)
                    if m_sol_feas2!=0:
                        D_feasibility2[tuple(i)]=m_sol_feas2
                        D_optimality[tuple(i)]=infinity_val
                    else:                   
                        m_sol=solve_subproblem(m=m_fix, subproblem_solver=nlp_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False)
                        if m_sol.dsda_status=='Optimal':
                            D_optimality[tuple(i)]=pe.value(m_sol.obj)
                            if pe.value(m_sol.obj)<best_solution_value:
                                best_solution_value=pe.value(m_sol.obj)
                                best_initialization=i
                                #print(best_initialization)
                                init_path=generate_initialization(m=m_sol)  #initialization updated according to the best solution obtained so far
                        else:
                            D_optimality[tuple(i)]=infinity_val
        else:
            warnings.warn('Use more than one random_points_number')
        if len([k for k in D_optimality if D_optimality[k]!=infinity_val])==0:
            if len([k for k in D_feasibility2 if D_feasibility2[k]!=infinity_val])==0:
                D_random=D_feasibility1
                initial_Stage=1
                initialization=list(min(D_random,key=D_random.get)) 
            else:
                D_random=D_feasibility2
                initial_Stage=2
                initialization=list(min(D_random,key=D_random.get))          
        else:
            D_random=D_optimality
            initial_Stage=3
            initialization=best_initialization

    end=time.time()
    pre_processing_time=end-start#time required to test initialization and to perform multi_start
    important_info_preprocessing=[D_random,pre_processing_time,initial_Stage]
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    #-----------STAGE 1
    if initial_Stage==1:
        if tee==True:
            print('stage 1...')
        #NOW SOLVE  
        x_actual=initialization
        D={}
        D=D_random.copy()

        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,model_fun,kwargs)
            fobj_actual=list(new_values.values())[0]
            if tee==True:
                print('S1--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,1,D,use_random)            
            #print(not_eval)
	        
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #  # Multiple cuts per iteration: cuts for all points in D
            for i in D:
                cuts=convex_clousure(D,list(i))
                m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)             #Solve master problem       
            
            SolverFactory('gams', solver='cplex').solve(m, tee=False) #TODO: generalize this
            if tee==True:
                print('S1--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5: 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
            #if 0 in D.values():
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_bounds[n_e] for   n_e in lower_bounds.keys()) and all(el[n_e-1]<=upper_bounds[n_e] for   n_e in lower_bounds.keys()) ]))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        # D.update({tuple([round(pe.value(m.x[posita])) for posita in m.extset]):infinity_val})
                        D.update({tuple(x_actual):infinity_val})
                        x_actual=list(min(D, key=D.get))

            else:
                x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

        end = time.time()
        #print('stage 1: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict)
        important_info['m3_s1']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
        #rewrite the dictionary for stages 2: infinity value for infieasible values and remove the feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val
    #-----------STAGE 2
    if initial_Stage==1 or initial_Stage==2:
        if tee==True:
            print('stage 2...')
        if initial_Stage==2:       
            x_actual=initialization
            D={}
            D=D_random.copy()
        #use x_actual and D from previous stages otherwise
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,init_path=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs)
            #print(new_values)
            fobj_actual=list(new_values.values())[0]
            if tee==True:
                print('S2--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model  
            #define model
            m,not_eval=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,2,D,use_random)          
            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #  # Multiple cuts per iteration: cuts for all points in D
            for i in D:
                cuts=convex_clousure(D,list(i))
                m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S2--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5: #TODO: use a general tolerance
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
            #if 0 in D.values():
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_bounds[n_e] for   n_e in lower_bounds.keys()) and all(el[n_e-1]<=upper_bounds[n_e] for   n_e in lower_bounds.keys()) ]))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
	                    # D.update({tuple([round(pe.value(m.x[posita])) for posita in m.extset]):infinity_val})
                        D.update({tuple(x_actual):infinity_val})
                        x_actual=list(min(D, key=D.get))
            else:
                x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

        end = time.time()
        #print('stage 2: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict)
        important_info['m3_s2']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
        #rewrite the dictionary for stages 3: infinity value for infieasible values and remove the feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val
        #-----------STAGE 3
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            if tee==True:
                print('--------------Iteration ',str(k),'--------------------------------------------')
                print('ext_var e_',str(k),'=',str(x_actual))
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if tuple(x_actual) not in D:
                new_values,init_path,m_solved=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs,sub_solver_opt=sub_solver_opt,tee=tee)

            else:
                new_values,init_path=solve_subproblem_and_neighborhood_except(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs,sub_solver_opt=sub_solver_opt,tee=tee)
                model = model_fun(**kwargs)
                m_initialized=initialize_model(m=model,json_path=init_path)
                m_fixed = external_ref(m=m_initialized,x=x_actual,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved=solve_subproblem(m=m_fixed, subproblem_solver=nlp_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False)

            fobj_actual=list(new_values.values())[0]
            if tee==True:
                # print('S3--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                print('Subproblem objective','f*(e_',str(k),')= ',str(fobj_actual))
            #Add points to D
            D.update(new_values)
            #print(D)
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)           
	        ## A single cut per iteration
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #  # Multiple cuts per iteration: cuts for all points in D
            for i in D:
                cuts=convex_clousure(D,list(i))
                m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)

            # Multiple cuts per iteration: points within a neighborhood of central points evaluated, at a distance of 0.5
            # for i in x_dict:
               
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #     for j in neigh: 
            #         if tuple(list(np.array(x_dict[i])+np.array(neigh[j]))) in D:   
            #             current_value=list(np.array(x_dict[i])+np.array(neigh[j])*0.5)  
            #             cuts=convex_clousure(D,current_value)
            #             m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)



            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('Master objective','z_',str(k+1),'= ',str(pe.value(m.zobj)))
                # print('S3--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5 or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
                break
            else:
                x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
    return important_info,important_info_preprocessing,D,x_actual,m_solved

def run_function_dbd_aprox(initialization,
                           infinity_val,
                           nlp_solver,
                           neigh,
                           maxiter,
                           ext_ref,logic_fun,
                           model_fun,
                           model_fun_feasibility_scheduling,
                           kwargs,
                           use_random: bool=False,
                           use_multi_start: bool=False,
                           n_points_multstart: int=10,
                           sub_solver_opt: dict={}, 
                           tee: bool=False, 
                           known_solutions: dict={},
                           rel_tol: float = 0,
                           new_case: bool=False, # If algorithm will be used for new case study involving kondili STN
                           with_distillation: bool=False,  #If modified model with distillation dynamics will be considered  
                           provide_starting_initialization: bool = True,
                           feasible_model: str = ''):
# IMPORTANT!!!!: IF INCLUDING known_solutions, MAKE SURE THAT THE INITIALIZATION IS FEASIBLE 
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=1 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 

    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #Initialize with user provided initialization
    model = model_fun(**kwargs)
    init_path = generate_initialization(m=model) #user provided initialization (TODO: CHECK, BUT I THINK THIS IS THE USER PROVIDED, TODO: change this line if another strategy to generate initialization will be considered)
    
    if use_multi_start==False:
        #test initialization 

        if provide_starting_initialization:
            model = initialize_model(model, from_feasible=True, feasible_model=feasible_model, json_path=None)
        else:
            model=initialize_model(m=model,json_path=init_path)
        m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        #original line
        m_init_solved=solve_subproblem_aprox(m=m_init_fixed, subproblem_solver=nlp_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False, rel_tol=rel_tol,new_case=new_case,with_distillation=with_distillation)
        best_sol=m_init_solved.best_sol
        #m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver,subproblem_solver_options= {'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver conopt4\n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}, timelimit=10000, tee=False)
        #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: uncomment last line for nonlinear scheduling problem      
        
        if m_init_solved.dsda_status=='FBBT_Infeasible':
            initial_Stage=1   
        if m_init_solved.dsda_status=='Evaluated_Infeasible':
            initial_Stage=2        
        if m_init_solved.dsda_status=='Optimal':
            init_path = generate_initialization(m=m_init_solved)
            initial_Stage=3
        if tee==True:
            print('Starting enhanced LD-BD at stage '+str(initial_Stage))    
    else:
        if tee==True:
            print('Executing multi-start...')
        #execute multistart (if required)
        D_feasibility1={} #optimal infeasibility 1 measure, infinity otherwise
        D_feasibility2={} #optimal infeasibility 2 measure, infinity otherwise
        D_optimality={}# optimal objective for feasible, infinity otherwise
        best_solution_value=infinity_val  #best solution updated through the multistart
        random_points_number=n_points_multstart

        if random_points_number >= 2:
            sampled_points=initialization_sampling_naive(random_points_number,lower_bounds,upper_bounds)
            #solve random points
            for i in sampled_points:
                model = model_fun(**kwargs)
                m_init=initialize_model(m=model,json_path=init_path)
                m_fix = external_ref(m=m_init,x=i,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)          
                m_sol_feas1,_=feasibility_1(m_fix)
                if m_sol_feas1!=0:
                    D_feasibility1[tuple(i)]=m_sol_feas1
                    D_feasibility2[tuple(i)]=infinity_val
                    D_optimality[tuple(i)]=infinity_val
                else:
                    m_sol_feas2,_=feasibility_2_modified(m_fix,nlp_solver,infinity_val)
                    if m_sol_feas2!=0:
                        D_feasibility2[tuple(i)]=m_sol_feas2
                        D_optimality[tuple(i)]=infinity_val
                    else:                   
                        m_sol=solve_subproblem(m=m_fix, subproblem_solver=nlp_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False)
                        if m_sol.dsda_status=='Optimal':
                            D_optimality[tuple(i)]=pe.value(m_sol.obj)
                            if pe.value(m_sol.obj)<best_solution_value:
                                best_solution_value=pe.value(m_sol.obj)
                                best_initialization=i
                                #print(best_initialization)
                                init_path=generate_initialization(m=m_sol)  #initialization updated according to the best solution obtained so far
                        else:
                            D_optimality[tuple(i)]=infinity_val
        else:
            warnings.warn('Use more than one random_points_number')
        if len([k for k in D_optimality if D_optimality[k]!=infinity_val])==0:
            if len([k for k in D_feasibility2 if D_feasibility2[k]!=infinity_val])==0:
                D_random=D_feasibility1
                initial_Stage=1
                initialization=list(min(D_random,key=D_random.get)) 
            else:
                D_random=D_feasibility2
                initial_Stage=2
                initialization=list(min(D_random,key=D_random.get))          
        else:
            D_random=D_optimality
            initial_Stage=3
            initialization=best_initialization

    end=time.time()
    pre_processing_time=end-start#time required to test initialization and to perform multi_start
    important_info_preprocessing=[D_random,pre_processing_time,initial_Stage]
    cpupre=end-start
    if tee==True:
        print('-------------------------------------------')
        print('CPU time preprocessing [s]= '+str(cpupre))
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    #-----------STAGE 1
    if initial_Stage==1:
        if tee==True:
            print('stage 1...')
        #NOW SOLVE  
        x_actual=initialization
        D={}
        D=D_random.copy()

        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values=solve_subproblem_and_neighborhood_FEAS1_aprox(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,model_fun,kwargs)
            fobj_actual=list(new_values.values())[0]
            if tee==True:
                print('S1--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,1,D,use_random)            



            #print(not_eval)
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)

            #  # Multiple cuts per iteration: cuts for all points in D
            for i in D:
                cuts=convex_clousure(D,list(i))
                m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False) #TODO: generalize this
            if tee==True:
                print('S1--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5: 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
            #if 0 in D.values():
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_bounds[n_e] for   n_e in lower_bounds.keys()) and all(el[n_e-1]<=upper_bounds[n_e] for   n_e in lower_bounds.keys()) ]))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                                                
                        # D.update({tuple([round(pe.value(m.x[posita])) for posita in m.extset]):infinity_val})
                        D.update({tuple(x_actual):infinity_val})
                        x_actual=list(min(D, key=D.get))
            else:
                x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

        end = time.time()
        #print('stage 1: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict)
        important_info['m3_s1']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
        cpu1=end-start
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time stage 1 [s]= '+str(cpu1)+'   |   ext. vars='+str(x_actual))
        #rewrite the dictionary for stages 2: infinity value for infieasible values and remove the feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val
    #-----------STAGE 2
    if initial_Stage==1 or initial_Stage==2:
        if tee==True:
            print('stage 2...')
        if initial_Stage==2:       
            x_actual=initialization
            D={}
            D=D_random.copy()
        #use x_actual and D from previous stages otherwise
        if new_case:
            current_tau=x_actual[:8] #TODO: GENERALIZE
        else:
            current_tau=x_actual[:6] #TODO: GENERALIZE
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,init_path,source=solve_subproblem_and_neighborhood_FEAS2_aprox(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs,tee=tee,new_case=new_case,with_distillation=with_distillation,lower_bounds=lower_bounds,upper_bounds=upper_bounds)
            #print(new_values)
            fobj_actual=list(new_values.values())[0]
            if tee==True:
                print('S2--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
 
             #CASE SPECIFIC CUTS: TODO: GENERALIZE
            if new_case:
                previous_tau=current_tau.copy()
                if len(source)!=0:
                    for element in source:
                        if source[element]=='Infeasible' and element==('T2', 'U2'):
                            current_tau[1]=x_actual[1]+1
                        elif source[element]=='Infeasible' and element==('T2', 'U3'):
                            current_tau[2]=x_actual[2]+1
                        elif source[element]=='Infeasible' and element==('T5', 'U4'):
                            current_tau[7]=x_actual[7]+1
            else:
                previous_tau=current_tau.copy()
                if len(source)!=0:
                    cuentass=-1
                    for element in source:
                        cuentass=cuentass+1
                        if source[element]=='Infeasible':
                            current_tau[cuentass]=x_actual[cuentass]+1
            #if cut for times is different, update D, and find new optimal point
            if previous_tau!=current_tau:
                for j in D.keys():
                    list_j=list(j)
                    if new_case:
                        list_j_tau=list_j[:8] #TODO: GENERALIZE
                    else:
                        list_j_tau=list_j[:6]
                    if any([list_j_tau[posit]<current_tau[posit] for posit in range(len(current_tau))]):
                        D.update({j:infinity_val})
                
                output_ext_vars=current_tau.copy()     
                kwargs2=kwargs.copy()
                kwargs2.update({'x_initial':output_ext_vars})
                m_scheduling_only=model_fun_feasibility_scheduling(**kwargs2)
                # sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','varsel -1 \n','intsollim 1 \n','$offecho \n']}
                sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','$offecho \n']} 
                m_scheduling_only = solve_with_minlp(m_scheduling_only,transformation='bigm',minlp='cplex',minlp_options=sub_options_cplex_Feas,timelimit=360000000,gams_output=False,tee=False,rel_tol=rel_tol)

                for I_J in m_scheduling_only.I_J:
                    output_ext_vars.append(1+round(pe.value(m_scheduling_only.Nref[I_J])))

                if m_scheduling_only.results.solver.termination_condition == 'infeasible' or m_scheduling_only.results.solver.termination_condition == 'other' or m_scheduling_only.results.solver.termination_condition == 'unbounded' or m_scheduling_only.results.solver.termination_condition == 'invalidProblem' or m_scheduling_only.results.solver.termination_condition == 'solverFailure' or m_scheduling_only.results.solver.termination_condition == 'internalSolverError' or m_scheduling_only.results.solver.termination_condition == 'error'  or m_scheduling_only.results.solver.termination_condition == 'resourceInterrupt' or m_scheduling_only.results.solver.termination_condition == 'licensingProblem' or m_scheduling_only.results.solver.termination_condition == 'noSolution' or m_scheduling_only.results.solver.termination_condition == 'noSolution' or m_scheduling_only.results.solver.termination_condition == 'intermediateNonInteger':
                    print('No feasible solution for the scheduling problem for the current cut of processing times. The problem is infeasible')
                    break
                else:
                    if tee:
                        print('New point found that satisfies scheduling constraints: ', output_ext_vars)
                    x_actual=output_ext_vars
            else:
                #Calculate new convex hull and dd cuts to the current model  
                #define model
                m,not_eval=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,2,D,use_random)          
                # for i in x_dict:
                #     cuts=convex_clousure(D,x_dict[i])
                #     #print(cuts)
                #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)

                #  # Multiple cuts per iteration: cuts for all points in D
                for i in D:
                    cuts=convex_clousure(D,list(i))
                    m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)

                if tee==True:
                    print('Current cut for tau: ',current_tau)
                secondcuent=-1
                for posit in m.extset:
                    secondcuent=secondcuent+1
                    if secondcuent<=5:
                        j=current_tau[secondcuent]
                        m.cuts.add(m.x[posit]>=j)

                #Solve master problem       
                SolverFactory('gams', solver='cplex').solve(m, tee=False)
                if tee==True:
                    print('S2--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
                #Stop?
                #print([pe.value(m.x1),pe.value(m.x2)])
                #print(new_values)

                # print(list(min(D, key=D.get)))
                # print(D)
                # D.update({tuple([round(pe.value(m.x[posita])) for posita in m.extset]):infinity_val})
                # print(D)
                # print(list(min(D, key=D.get)))

                if fabs(fobj_actual-pe.value(m.zobj))<=1e-5: #TODO: use a general tolerance
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if 
                #if 0 in D.values():
                    if 0 in D.values():
                        break
                    else:
                        if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_bounds[n_e] for   n_e in lower_bounds.keys()) and all(el[n_e-1]<=upper_bounds[n_e] for   n_e in lower_bounds.keys()) ]))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                            break
                        else:


                            #TODO: ADDITIONAL VERIFICATIONS IMPOSED DOE TO THE PROBLEM-SPECIFIC CUTS!!!!. Since I evaluate a single point when starting, the solution of the first master dictated by the first cut
                            # may return a soluton that closes the gap (because the cutting plane may be constant). SINCE THE GAP CLOSED, THE ALGORITHM AUTOMATICALLY SEARCH WITHIN D TO FIND A REINITIALIZATION POINT.
                            # Since there was a single point evaluated. It goes to taht point in a loop. To fix this, I check first if the solution of the master is not in D. If that is the case, then it is worth trying that
                            # soluton first!!!!  
                            # D.update({tuple([round(pe.value(m.x[posita])) for posita in m.extset]):infinity_val})

                            if tuple([round(pe.value(m.x[posita])) for posita in m.extset]) in D.keys():
                                D.update({tuple(x_actual):infinity_val})
                                x_actual=list(min(D, key=D.get))
                                if D[min(D, key=D.get)]==infinity_val:

                                    # input_ext_vars=current_tau.copy()
                                    # output_ext_vars=current_tau.copy()
                                    # kay=len(x_actual)-len(current_tau)
                                    # kay2=len(current_tau)
                                    # for i in range(kay): 
                                    #     input_ext_vars.append(x_actual[i+kay2])        

                                    # m_scheduling_only=model_fun(**kwargs)
                                    # m_scheduling_only = external_ref(m=m_scheduling_only,x=input_ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
                                    # #REMOVE CONTROL CONSTRAINTS
                                    # for I in m_scheduling_only.I_reactions:
                                    #     for J in m_scheduling_only.J_reactors:
                                    #         m_scheduling_only.c_dCdtheta[I,J].deactivate()
                                    #         m_scheduling_only.c_dTRdtheta[I,J].deactivate()                        
                                    #         m_scheduling_only.c_dTJdtheta[I,J].deactivate()
                                    #         m_scheduling_only.c_dIntegral_hotdtheta[I,J].deactivate()
                                    #         m_scheduling_only.c_dIntegral_colddtheta[I,J].deactivate()
                                    #         m_scheduling_only.Constant_control1[I,J].deactivate()                        
                                    #         m_scheduling_only.Constant_control2[I,J].deactivate()
                                    #         m_scheduling_only.finalCon[I,J].deactivate()
                                    #         m_scheduling_only.finalTemp[I,J].deactivate()
                                    # for I_J in m_scheduling_only.I_J:
                                    #     I=I_J[0]
                                    #     J=I_J[1]  
                                    #     m_scheduling_only.DEF_Nref[I,J].deactivate()
                                    #     if I in m_scheduling_only.I_reactions and J in m_scheduling_only.J_reactors:
                                    #         m_scheduling_only.finalCon[I,J].deactivate()
                                    #         m_scheduling_only.finalTemp[I,J].deactivate()
                                    # m_scheduling_only.C_TCP3.deactivate()
                                    # m_scheduling_only.obj.deactivate()
                                    # m_scheduling_only.obj_dummy.deactivate()
                                    # sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','varsel -1 \n','intsollim 1 \n','$offecho \n']}
                                    # # sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','$offecho \n']}
                                    # m_scheduling_only = solve_subproblem(m=m_scheduling_only,subproblem_solver='cplex',subproblem_solver_options=sub_options_cplex_Feas,timelimit=100000000,gams_output=False,tee=False,rel_tol=rel_tol)

                                    # for I_J in m_scheduling_only.I_J:
                                    #     output_ext_vars.append(1+round(pe.value(m_scheduling_only.Nref[I_J])))



                                    # if m_scheduling_only.dsda_status=='Optimal':
                                    #     x_actual=output_ext_vars
                                    # else:
                                    print('Infeasible: There are no additional candidate solutions in D to explore')
                                    break
                            else:
                                x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

                else:
                    if [round(pe.value(m.x[posita])) for posita in m.extset]==x_actual:

                        # input_ext_vars=current_tau.copy()
                        # output_ext_vars=current_tau.copy()
                        # kay=len(x_actual)-len(current_tau)
                        # kay2=len(current_tau)
                        # for i in range(kay): 
                        #     input_ext_vars.append(x_actual[i+kay2])        

                        # m_scheduling_only=model_fun(**kwargs)
                        # m_scheduling_only = external_ref(m=m_scheduling_only,x=input_ext_vars,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
                        # #REMOVE CONTROL CONSTRAINTS
                        # for I in m_scheduling_only.I_reactions:
                        #     for J in m_scheduling_only.J_reactors:
                        #         m_scheduling_only.c_dCdtheta[I,J].deactivate()
                        #         m_scheduling_only.c_dTRdtheta[I,J].deactivate()                        
                        #         m_scheduling_only.c_dTJdtheta[I,J].deactivate()
                        #         m_scheduling_only.c_dIntegral_hotdtheta[I,J].deactivate()
                        #         m_scheduling_only.c_dIntegral_colddtheta[I,J].deactivate()
                        #         m_scheduling_only.Constant_control1[I,J].deactivate()                        
                        #         m_scheduling_only.Constant_control2[I,J].deactivate()
                        #         m_scheduling_only.finalCon[I,J].deactivate()
                        #         m_scheduling_only.finalTemp[I,J].deactivate()
                        # for I_J in m_scheduling_only.I_J:
                        #     I=I_J[0]
                        #     J=I_J[1]  
                        #     m_scheduling_only.DEF_Nref[I,J].deactivate()
                        #     if I in m_scheduling_only.I_reactions and J in m_scheduling_only.J_reactors:
                        #         m_scheduling_only.finalCon[I,J].deactivate()
                        #         m_scheduling_only.finalTemp[I,J].deactivate()
                        # m_scheduling_only.C_TCP3.deactivate()
                        # m_scheduling_only.obj.deactivate()
                        # m_scheduling_only.obj_dummy.deactivate()
                        # sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','varsel -1 \n','intsollim 1 \n','$offecho \n']}
                        # # sub_options_cplex_Feas={'add_options':['GAMS_MODEL.optfile = 1;','$onecho > cplex.opt \n','$offecho \n']} 
                        # m_scheduling_only = solve_subproblem(m=m_scheduling_only,subproblem_solver='cplex',subproblem_solver_options=sub_options_cplex_Feas,timelimit=100000000,gams_output=False,tee=False,rel_tol=rel_tol)

                        # for I_J in m_scheduling_only.I_J:
                        #     output_ext_vars.append(1+round(pe.value(m_scheduling_only.Nref[I_J])))



                        # if m_scheduling_only.dsda_status=='Optimal':
                        #     x_actual=output_ext_vars
                        # else:
                        print('Same value for external vars in two consecutive iteratons: the problem seems to be infeasible')
                        break
                    else:
                        x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

        end = time.time()
        #print('stage 2: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict)
        important_info['m3_s2']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
        cpu2=end-start
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time stage 2 [s]= '+str(cpu2)+'   |   ext. vars='+str(x_actual))
        #rewrite the dictionary for stages 3: infinity value for infieasible values and remove the feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val
        #-----------STAGE 3
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            if tee==True:
                print('--------------Iteration ',str(k),'--------------------------------------------')
                print('ext_var e_',str(k),'=',str(x_actual))
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if tuple(x_actual) not in D:
                new_values,init_path,m_solved=solve_subproblem_and_neighborhood_aprox(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs,sub_solver_opt=sub_solver_opt,tee=tee,best_sol=best_sol,rel_tol=rel_tol,new_case=new_case,with_distillation=with_distillation,lower_bounds=lower_bounds,upper_bounds=upper_bounds)

            else:
                new_values,init_path=solve_subproblem_and_neighborhood_aprox_except(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs,sub_solver_opt=sub_solver_opt,tee=tee,best_sol=best_sol,rel_tol=rel_tol,new_case=new_case,with_distillation=with_distillation,lower_bounds=lower_bounds,upper_bounds=upper_bounds)
                model = model_fun(**kwargs)
                m_initialized=initialize_model(m=model,json_path=init_path)
                m_fixed = external_ref(m=m_initialized,x=x_actual,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved=solve_subproblem_aprox(m=m_fixed, subproblem_solver=nlp_solver,subproblem_solver_options= sub_solver_opt, timelimit=10000, tee=False,best_sol=best_sol,rel_tol=rel_tol,new_case=new_case,with_distillation=with_distillation)
                        
            if min(list(new_values.values()))<=best_sol:
                best_sol=min(list(new_values.values()))
            if tee:
                print('current best objective:',best_sol)
            fobj_actual=list(new_values.values())[0]
            if tee==True:
                # print('S3--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                print('Subproblem objective','f*(e_',str(k),')= ',str(fobj_actual))
            #Add points to D
            D.update(new_values)
            #print(D)
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)           
            
            # A single cut per iteration
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)

            #  # Multiple cuts per iteration: cuts for all points in D
            for i in D:
                cuts=convex_clousure(D,list(i))
                m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)

             # Multiple cuts per iteration: points within a neighborhood of central points evaluated, at a distance of 0.5
            # for i in x_dict:
               
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
            #     #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)

            #     for j in neigh: 
            #         if tuple(list(np.array(x_dict[i])+np.array(neigh[j]))) in D:   
            #             current_value=list(np.array(x_dict[i])+np.array(neigh[j])*0.5)  
            #             cuts=convex_clousure(D,current_value)
            #             m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)



            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('Master objective','z_',str(k+1),'= ',str(pe.value(m.zobj)))
                # print('S3--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5 or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
                break
            else:
                x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]

        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        cpu3=end-start
        # if tee==True:
        print('-------------------------------------------')
        print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time stage 3 [s]= '+str(cpu3)+'   |   ext. vars='+str(x_actual))
        try:
            print('CPU time preprocessing [s]='+str(cpupre))
        except:
            pass
        try:
            print('CPU time stage 1 [s]='+str(cpu1))
        except:
            pass
        try:
            print('CPU time stage 2 [s]='+str(cpu2))
        except:
            pass
        print('CPU time stage 3 [s]='+str(cpu3))
    return important_info,important_info_preprocessing,D,x_actual,m_solved








#



def run_function_dbd_scheduling_cost_min(model_fun_feas,minimum_obj,epsilon,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):
# IMPORTANT!!!!: IF INCLUDING known_solutions, MAKE SURE THAT THE INITIALIZATION IS FEASIBLE 
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=3 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    _, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 
    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            if tee==True and k==1:
                print('S3---- User provided lower bound= '+str(minimum_obj))
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if k!=1:
                kwargs_Feas={'objective':minimum_obj,'epsilon':0.01}# TODO: use this epsilon as input
                m_feas=model_fun_feas(**kwargs_Feas)
                sub_options_feasibility={}
                sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > cplex.opt \n','intsollim 1\n','mipemphasis 4\n','$offecho \n']}
                pe.TransformationFactory('core.logical_to_linear').apply_to(m_feas)
                m_solution=solve_subproblem(m_feas,subproblem_solver = nlp_solver,subproblem_solver_options= sub_options_feasibility,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)
                if m_solution.dsda_status=='Optimal':
                    fobj_actual=pe.value(m_solution.obj_dummy)#minimum_obj #I should extract the solution of the subproblem, but I know that this is going to be its solution
                else:
                    fobj_actual=infinity_val
                if tee==True:
                    print('S3--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                #Add points to D
                D.update({tuple(x_actual):fobj_actual})
                #print(D)
                #Calculate new convex hull and dd cuts to the current model
                #define model
                if fabs(fobj_actual-minimum_obj)<=1e-5: #or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if
                    final_sol=[]
                    for I_J in m_solution.I_J:
                        for N in m_solution.N:
                            if pe.value(m_solution.Z_binary[N,I_J])==1:
                                final_sol.append(N+1)    
                    x_actual=final_sol
                    D.update({tuple(final_sol):fobj_actual})
                    break
            m,_=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            _cost={}# TODO: GENERALIZE THIS!!!!
            _cost[1]=10

            _cost[2]=15
            _cost[3]=30

            _cost[4]=5
            _cost[5]=25

            _cost[6]=5
            _cost[7]=20

            _cost[8]=20
            if k==1:
                m.cuts.add(minimum_obj<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset))  
            else:
                m.cuts.add(minimum_obj+epsilon<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)) #TODO: epsilon must be the minimum coefficient in the objective function
            m.cuts.add(sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S3--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)

            x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]
            minimum_obj=pe.value(m.zobj)
        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
    return important_info,D,x_actual

def run_function_dbd_scheduling_cost_min_nonlinear(model_fun_feas,minimum_obj,epsilon,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):
# IMPORTANT!!!!: IF INCLUDING known_solutions, MAKE SURE THAT THE INITIALIZATION IS FEASIBLE 
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=3 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    _, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 
    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            if tee==True and k==1:
                print('S3---- User provided lower bound= '+str(minimum_obj))
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if k!=1:
                kwargs_Feas={'objective':minimum_obj,'epsilon':0.01}# TODO: use this epsilon as input
                m_feas=model_fun_feas(**kwargs_Feas)
                sub_options_feasibility={}
                # sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','optcr 1\n','optca 1000000000\n','feaspump 2\n','MAXCYCLES 20\n','fp_stalllimit 0\n','$offecho \n']}
                if nlp_solver=='dicopt':
                    sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}
                elif nlp_solver=='baron':
                    sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > baron.opt \n','FirstFeas 1\n',' NumSol 1\n',' NumLoc 0\n','$offecho \n']}
                pe.TransformationFactory('core.logical_to_linear').apply_to(m_feas)
                m_solution=solve_subproblem(m_feas,subproblem_solver = nlp_solver,subproblem_solver_options= sub_options_feasibility,timelimit= 1000000,gams_output = False,tee= True,rel_tol = 0)
                if m_solution.dsda_status=='Optimal':
                    fobj_actual=pe.value(m_solution.obj_dummy) #minimum_obj #I should extract the solution of the subproblem, but I know that this is going to be its solution
                else:
                    fobj_actual=infinity_val
                if tee==True:
                    print('S3--'+'--iter '+str(k)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                #Add points to D
                D.update({tuple(x_actual):fobj_actual})
                #print(D)
                #Calculate new convex hull and dd cuts to the current model
                #define model
                if fabs(fobj_actual-minimum_obj)<=1e-5: #or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if
                    final_sol=[]
                    for I_J in m_solution.I_J:
                        for N in m_solution.N:
                            if pe.value(m_solution.Z_binary[N,I_J])==1:
                                final_sol.append(N+1)    
                    x_actual=final_sol
                    D.update({tuple(final_sol):fobj_actual})
                    break
            m,_=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            _cost={}# TODO: GENERALIZE THIS!!!!
            _cost[1]=10

            _cost[2]=15
            _cost[3]=30

            _cost[4]=5
            _cost[5]=25

            _cost[6]=5
            _cost[7]=20

            _cost[8]=20
            if k==1:
                m.cuts.add(minimum_obj<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset))  
            else:
                m.cuts.add(minimum_obj+epsilon<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)) #TODO: epsilon must be the minimum coefficient in the objective function
            m.cuts.add(sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S3--'+'--iter '+str(k)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)

            x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]
            minimum_obj=pe.value(m.zobj)
        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
    return important_info,D,x_actual

def run_function_dbd_scheduling_cost_min_ref_2(model_fun_feas,minimum_obj,epsilon,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):
# IMPORTANT!!!!: IF INCLUDING known_solutions, MAKE SURE THAT THE INITIALIZATION IS FEASIBLE 
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=3 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    _, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 
    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            if tee==True and k==1:
                print('S3---- User provided lower bound= '+str(minimum_obj))
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if k!=1:
                kwargs_Feas={'objective':minimum_obj,'epsilon':0.01}# TODO: use this epsilon as input
                m_feas=model_fun_feas(**kwargs_Feas)
                sub_options_feasibility={}
                sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > cplex.opt \n','intsollim 1\n','mipemphasis 1\n','$offecho \n']}
                pe.TransformationFactory('core.logical_to_linear').apply_to(m_feas)
                m_solution=solve_subproblem(m_feas,subproblem_solver = nlp_solver,subproblem_solver_options= sub_options_feasibility,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)
                if m_solution.dsda_status=='Optimal':
                    fobj_actual=pe.value(m_solution.obj_dummy)#minimum_obj #I should extract the solution of the subproblem, but I know that this is going to be its solution
                else:
                    fobj_actual=infinity_val
                if tee==True:
                    print('S3--'+'--iter '+str(k-1)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                #Add points to D
                D.update({tuple(x_actual):fobj_actual})
                #print(D)
                #Calculate new convex hull and dd cuts to the current model
                #define model
                if fabs(fobj_actual-minimum_obj)<=1e-5: #or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if
                    final_sol=[]
                    for I_J in m_solution.I_J:
                            final_sol.append(pe.value(m_solution.Nref[I_J])+1)    
                    x_actual=final_sol
                    D.update({tuple(final_sol):fobj_actual})
                    m_return= m_solution                     
                    break
            m,_=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            _cost={}# TODO: GENERALIZE THIS!!!!
            _cost[1]=10

            _cost[2]=15
            _cost[3]=30

            _cost[4]=5
            _cost[5]=25

            _cost[6]=5
            _cost[7]=20

            _cost[8]=20
            if k==1:
                m.cuts.add(minimum_obj<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset))  
            else:
                m.cuts.add(minimum_obj+epsilon<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)) #TODO: epsilon must be the minimum coefficient in the objective function
            m.cuts.add(sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S3--'+'--iter '+str(k-1)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)

            x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]
            minimum_obj=pe.value(m.zobj)
        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
    return important_info,D,x_actual,m_return

def run_function_dbd_scheduling_cost_min_nonlinear_ref_2(model_fun_feas,minimum_obj,absolute_gap,epsilon,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):
# IMPORTANT!!!!: IF INCLUDING known_solutions, MAKE SURE THAT THE INITIALIZATION IS FEASIBLE 
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=3 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    _, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=True) 
    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()

        sub_options_feasibility={}
        if nlp_solver=='dicopt': #TODO: CONOPT 4 used for a specific case study
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}
        elif nlp_solver=='baron':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > baron.opt \n','FirstFeas 1\n',' NumSol 1\n','$offecho \n']}
        elif nlp_solver=='lindoglobal':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > lindoglobal.opt \n',' GOP_OPT_MODE 0\n','$offecho \n']}
        elif nlp_solver=='antigone':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > antigone.opt \n','abs_opt_tol 100\n','rel_opt_tol 1\n','$offecho \n']}
        elif nlp_solver=='sbb':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > sbb.opt \n','intsollim 1\n','$offecho \n']}                
        elif nlp_solver=='bonmin':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > bonmin.opt \n','bonmin.pump_for_minlp yes\n','pump_for_minlp.solution_limit 1\n','solution_limit 1\n','$offecho \n']}   
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            if tee==True and k==1:
                print('S3---- User provided lower bound= '+str(minimum_obj))
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if k!=1:
                kwargs_Feas={'objective':minimum_obj,'epsilon':absolute_gap}# TODO: use this epsilon as input
                m_feas=model_fun_feas(**kwargs_Feas)
                pe.TransformationFactory('core.logical_to_linear').apply_to(m_feas)
                m_solution=solve_subproblem(m_feas,subproblem_solver = nlp_solver,subproblem_solver_options= sub_options_feasibility,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)
                
                if m_solution.dsda_status=='Optimal':
                    fobj_actual=pe.value(m_solution.obj_dummy) #minimum_obj #I should extract the solution of the subproblem, but I know that this is going to be its solution
                else:
                    fobj_actual=infinity_val
                if tee==True:
                    print('S3--'+'--iter '+str(k-1)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                #Add points to D
                D.update({tuple(x_actual):fobj_actual})
                #print(D)
                #Calculate new convex hull and dd cuts to the current model
                #define model
                if fabs(fobj_actual-minimum_obj)<=absolute_gap: #or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if
                
                    final_sol=[]
                    for I_J in m_solution.I_J:
                            final_sol.append(pe.value(m_solution.Nref[I_J])+1)     
                    x_actual=final_sol
                    D.update({tuple(final_sol):fobj_actual})   
                    m_return= m_solution 
                    actual_absolute_gap= fabs(fobj_actual-minimum_obj) 
                    actual_relative_gap=(actual_absolute_gap/fabs(minimum_obj))*100              
                    break
            m,_=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            _cost={}# TODO: GENERALIZE THIS!!!!
            # _cost[1]=10

            # _cost[2]=15
            # _cost[3]=30

            # _cost[4]=5
            # _cost[5]=25

            # _cost[6]=5
            # _cost[7]=20

            # _cost[8]=20

            #scheduling and control
            _cost[1]=10
            _cost[2]=30
            _cost[3]=20
            _cost[4]=30
            _cost[5]=20
            _cost[6]=30
            _cost[7]=20
            _cost[8]=100
            _cost[9]=50
            _cost[10]=50
            if k==1:
                m.cuts.add(minimum_obj<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset))  
            else:
                m.cuts.add(minimum_obj+epsilon<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)) #TODO: epsilon must be the minimum coefficient in the objective function
            m.cuts.add(sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S3--'+'--iter '+str(k-1)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)

            x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]
            minimum_obj=pe.value(m.zobj)
        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
            print('optca=',str(actual_absolute_gap),'| optcr=',str(actual_relative_gap),'%')
    return important_info,D,x_actual,m_return