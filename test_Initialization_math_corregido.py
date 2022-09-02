from cmath import inf
from scipy import optimize
import itertools as it
from math import fabs
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
from gdp_mathprob_easy import build_math
from functions.cuts_functions import convex_clousure,initialization_sampling
from functions.dsda_functions import neighborhood_k_eq_inf, get_external_information,external_ref,initialize_model,generate_initialization, solve_subproblem, solve_with_dsda,solve_with_gdpopt,solve_with_minlp
from functions.dsda_functions import preprocess_problem,solve_with_gdpopt,solve_with_minlp,solve_with_dsda,neighborhood_k_eq_inf, get_external_information,external_ref,initialize_model,generate_initialization, solve_subproblem
from functions.feasibility_functions import feasibility_1,feasibility_2,feasibility_2_modified
import copy
import time
import random
import pickle
import logging
from itertools import product
import warnings

#Random seed for master problem initializations (if required)
random.seed(30)

def problem_logic_math(m):
    logic_expr = []
    for n in m.set1:
        logic_expr.append([m.Y1[n],m.Y1_disjunct[n].indicator_var])
    for n in m.set2:
        logic_expr.append([m.Y2[n],m.Y2_disjunct[n].indicator_var])
    return logic_expr

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

def solve_subproblem_and_neighborhood(x,neigh,Internaldata,infinity_val,reformulation_dict,logic_fun,sub_solver,init_path,model_fun,kwargs):
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
        m_solved=solve_subproblem(m=m_fixed, subproblem_solver=sub_solver, timelimit=10000, tee=False)
        #print(m_solved.dsda_status)
        if m_solved.dsda_status=='Optimal':
            status[0]=1
            generated_dict[tuple(x)]=pe.value(m_solved.obj)
            init_path = generate_initialization(m=m_solved)
        else:
            status[0]=0
            generated_dict[tuple(x)]=infinity_val
    #solve neighborhood (only if central point was feasible)
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
                m_initialized2=initialize_model(m=model,json_path=init_path)
                m_fixed2 = external_ref(m=m_initialized2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                m_solved2=solve_subproblem(m=m_fixed2, subproblem_solver=sub_solver, timelimit=10000, tee=False)
                #print(m_solved2.dsda_status)
                if m_solved2.dsda_status=='Optimal':
                    status[count]=1
                    generated_dict[tuple(current_value)]=pe.value(m_solved2.obj)
                else:
                    status[count]=0
                    generated_dict[tuple(current_value)]=infinity_val   #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
                #print(pe.value(m_solved2.obj))
    return generated_dict,init_path


def build_master(lower_b,upper_b,current,stage,D,use_random: bool=False):
    """
    Function that builds the master problem

    use_random: True if a random point will be generated for initializations when required. False if you want to use the deterministric strategy
        
    """
    initial={}
    randomp=[]
    if stage==1 or stage==2: #generate random number different from current value and points evaluated so far
        if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_b[n_e] for   n_e in lower_b.keys()) and all(el[n_e-1]<=upper_b[n_e] for   n_e in lower_b.keys()) ]))-float(math.prod(upper_b[n_e]-lower_b[n_e]+1 for n_e in lower_b)))<=0.01: #if every point has been evaluated
            initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value
        else:
            if use_random:
                #Generate random numbers
                while True:
                    randomp=[random.randint(lower_b[n_e],upper_b[n_e]) for n_e in lower_b.keys()]  #This allows to consider difficult problems where, e.g., there is a big infeasible region with the same objective function values.
                    if all([np.linalg.norm(np.array(randomp)-np.array(list(i)))>=0.1 for i in list(D.keys())]):
                        initial={n_e:randomp[n_e-1] for n_e in lower_b.keys()}
                        break
            else:
                #generate nonrandom numbers. It is better to go to the closest point that has not been evaluated (using e.g. a lexicographical ordering).  
                arrays=[range(lower_b[n_e],upper_b[n_e]+1) for n_e in lower_b.keys()]
                cart_prduct=list(product(*arrays)) #cartesian product
                #TODO: after the cartesian product, I am organizing this with respect to the current value using a distance metric. Note that I can aslo explore points that are far away in the future.
                cart_prduct_sorted=sorted(cart_prduct, key=lambda x: np.linalg.norm(np.array(list(x))-np.array(current)      )      ) #I am sorting to evaluate the closests point. I can also sort to evaluate the one that is far away (exploration!!!!!)
                for j in cart_prduct_sorted:
                    non_randomp=list(j)
                    if all([np.linalg.norm(np.array(non_randomp)-np.array(list(i)))>=0.1 for i in list(D.keys())]):
                        initial={n_e:non_randomp[n_e-1] for n_e in lower_b.keys()}
                        break
    else:
        initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value

    #print(initial)
    #Model
    m=pe.ConcreteModel(name='Master_problem')
    #External variables (#TODO: THIS SHOULD BE AUTOMATIC)
    m.x1=pe.Var(within=pe.Integers, bounds=(lower_b[1],upper_b[1]),initialize=initial[1])
    m.x2=pe.Var(within=pe.Integers, bounds=(lower_b[2],upper_b[2]),initialize=initial[2])

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



def run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10, tee: bool=False):
    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=1 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) #TODO: use the number of external variables in functions that requrie it, e.g., master problem function

    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #Initialize with user provided initialization
    model = model_fun(**kwargs)
    init_path = generate_initialization(m=model) #user provided initialization (TODO: CHECK, BUT I THINK THIS IS THE USER PROVIDED, TODO: change this line if another strategy to generate initialization will be considered)
    
    if use_multi_start==False:
        #test initialization 
        model=initialize_model(m=model,json_path=init_path)
        m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
        if m_init_solved.dsda_status=='FBBT_Infeasible':
            initial_Stage=1   
        if m_init_solved.dsda_status=='Evaluated_Infeasible':
            initial_Stage=2        
        if m_init_solved.dsda_status=='Optimal':
            init_path = generate_initialization(m=m_init_solved)
            initial_Stage=3
    else:
        #execute multistart (if required)
        D_feasibility1={} #optimal infeasibility 1 measure, infinity otherwise
        D_feasibility2={} #optimal infeasibility 2 measure, infinity otherwise
        D_optimality={}# optimal objective for feasible, infinity otherwise
        best_solution_value=infinity_val  #best solution updated through the multistart
        random_points_number=n_points_multstart

        if random_points_number >= 2:
            sampled_points=initialization_sampling(random_points_number,lower_bounds,upper_bounds)
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
                        m_sol=solve_subproblem(m=m_fix, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
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
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,1,D,use_random)            
            for i in x_dict:
                cuts=convex_clousure(D,x_dict[i])
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj) #TODO: generalize this
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False) #TODO: generalize this

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
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        #print('stage 1: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print('evaluated stage 1',x_dict,'\n')
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
        if initial_Stage==2:
            x_actual=initialization
            D={}
            D=D_random.copy()
        #use x_actual and D from previous stages otherwise
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:

            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,init_path=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs)
            fobj_actual=list(new_values.values())[0]
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model  
            #define model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,2,D,use_random)          
            for i in x_dict:
                cuts=convex_clousure(D,x_dict[i])
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

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
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        #print('stage 2: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print('evaluated stage 2',x_dict,'\n')
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
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            if tee==True:
                print('--------------Iteration ',str(k),'--------------------------------------------')
                print('ext_var e_',str(k),'=',str(x_actual))
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,reformulation_dict,logic_fun,nlp_solver,init_path,model_fun,kwargs)
            fobj_actual=list(new_values.values())[0]
            if tee==True:
                print('Subproblem objective','f*(e_',str(k),')= ',str(fobj_actual))
            #Add points to D
            D.update(new_values)
            #print(D)
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,3,D)  
            if tee==True:
                print('List of updated cuts:')
            contador=0                          
            for i in x_dict:
                contador=contador+1
                cuts=convex_clousure(D,x_dict[i])
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
                if tee==True:
                    print('Cut ',str(contador),': ',str(float(cuts[0])),'*x1  +  ',str(float(cuts[1])),'*x2  +',str(float(cuts[2])),'  <=  z')
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if tee==True:
                print('Master objective','z_',str(k+1),'= ',str(pe.value(m.zobj)))
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5: 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print('evaluated stage 3',x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
    return important_info,important_info_preprocessing,D,x_dict


if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


    #--------------------one run----------------------------------------------------------------
    #INIT_VALUES
    kwargs={}
    model_fun =build_math
    logic_fun=problem_logic_math
    model=model_fun(**kwargs)
    ext_ref = {model.Y1: model.set1, model.Y2: model.set2} #reformulation sets and variables
    initialization=[5,1] 
    infinity_val=1e+5
    nlp_solver='msnlp'
    neigh=neighborhood_k_eq_inf(2)
    maxiter=100
    points=5




    solve_with_dsda(model_fun,{},initialization,ext_ref,logic_fun,provide_starting_initialization=True,feasible_model='dsda',subproblem_solver = nlp_solver,global_tee=True,rel_tol = 0)
    info_solver,info_preprocess,evaluated,_=run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,use_multi_start=False,n_points_multstart=points,tee=True)
    #print(evaluated)
    print(info_solver)
    print(info_preprocess)


    last={}
    last_dsda={}
    req_time_dbd={}
    req_time_dsda={}
    for i in range(1,6):
        for j in range(1,6):
            initalization=[i,j]
            in_time=time.time()
            _,_,_,dictt=run_function_dbd(initalization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,use_multi_start=False,n_points_multstart=points,tee=False)
            end_time=time.time()
            last[tuple(initalization)]=dictt[list(dictt)[-1]]
            req_time_dbd[tuple(initalization)]=end_time-in_time

            in_time=time.time()
            _,route_dsda,_=solve_with_dsda(model_fun,{},initalization,ext_ref,logic_fun,provide_starting_initialization=True,feasible_model='dsda',subproblem_solver = nlp_solver,global_tee=False,rel_tol = 0)
            end_time=time.time()
            last_dsda[tuple(initalization)]=route_dsda[-1]
            req_time_dsda[tuple(initalization)]=end_time-in_time

    print(last)
    print(last_dsda)
    print(req_time_dbd)
    print(req_time_dsda)
    print('average time LDBD=',(sum(i for i in req_time_dbd.values()))/(len(req_time_dbd)))
    print("average time DSDA=",(sum(i for i in req_time_dsda.values()))/(len(req_time_dsda)))










