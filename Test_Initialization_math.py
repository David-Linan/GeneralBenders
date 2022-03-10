from __future__ import division
from scipy import optimize
import itertools as it
from math import fabs
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
from gdp_mathprob import build_math
from cuts_functions import convex_clousure,initialization_sampling
from dsda_functions import neighborhood_k_eq_inf, get_external_information,external_ref,initialize_model,generate_initialization, solve_subproblem, solve_with_dsda,solve_with_gdpopt,solve_with_minlp
import copy
import time
from feasibility_functions import feasibility_1,feasibility_2
import random
import logging

#Random seed for master problem initializations
random.seed(30)

def problem_logic_math(m):
    logic_expr = []
    for n in m.set1:
        logic_expr.append([m.Y1[n],m.Y1_disjunct[n].indicator_var])
    for n in m.set2:
        logic_expr.append([m.Y2[n],m.Y2_disjunct[n].indicator_var])
    return logic_expr

def solve_subproblem_and_neighborhood_FEAS1(x,neigh,Internaldata,infinity_val,Adjustable_val,reformulation_dict,logic_fun):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        Adjustable_val: distance from central point at which the points used to calculated convex hull will be located (usually 0.5)
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
        model = build_math()
        m_fixed = external_ref(m=model,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved,_=feasibility_1(m_fixed)
        #print(m_solved.dsda_status)
        #if m_solved.dsda_status=='Optimal':
        status[0]=1
        generated_dict[tuple(x)]=m_solved
        #else:
        #    status[0]=0
        #    generated_dict[tuple(x)]=infinity_val


    generated_list_feasible=generated_list_feasible+[x] 
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
                    model = build_math()
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
                if status[count]==1:
                    generated_list_feasible=generated_list_feasible+[list((np.array(x)+Adjustable_val*np.array(neigh[j])))]

    return generated_dict,generated_list_feasible

def solve_subproblem_and_neighborhood_FEAS2(x,neigh,Internaldata,infinity_val,Adjustable_val,reformulation_dict,logic_fun,sub_solver,first_path):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldata: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        Adjustable_val: distance from central point at which the points used to calculated convex hull will be located (usually 0.5)
        reformulation_dict: directory with reformualtion info
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
    """
    generated_dict={}
    generated_list_feasible=[] #if required
    status=[None]*(len(neigh.keys())+1)   #status of the solution obtained: 1 if feasible, 0 if infeasible. Status[0] correspond to x and the subsequent positions to its neighbors
    init_path=''
    #solve subproblem
    if tuple(x) in Internaldata: #If subproblem was already solved
        if Internaldata[tuple(x)]!=infinity_val:
            status[0]=1 #feasible
        else:
            status[0]=0 #infeasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)] #THIS IS A TEST
    else: #If subproblem has not been solved yet
        #1: fix external variables
        model = build_math()
        model=initialize_model(m=model,json_path=first_path)
        m_fixed = external_ref(m=model,x=x,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
        m_solved,_,return_path=feasibility_2(m_fixed,sub_solver,infinity_val)
        if return_path!='':
            init_path=return_path 
        #print(m_solved.dsda_status)
        #if m_solved.dsda_status=='Optimal':
        status[0]=1
        generated_dict[tuple(x)]=m_solved
        # else:
        #     status[0]=0
        #     generated_dict[tuple(x)]=infinity_val


    generated_list_feasible=generated_list_feasible+[x] 
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
                    model2 = build_math()
                    model2=initialize_model(m=model2,json_path=first_path)
                    m_fixed2 = external_ref(m=model2,x=current_value,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,tee=False)
                    m_solved2,_,return_path2=feasibility_2(m_fixed2,sub_solver,infinity_val)
                    if return_path2!='':
                        init_path=return_path2
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
                
                if status[count]==1:
                    generated_list_feasible=generated_list_feasible+[list((np.array(x)+Adjustable_val*np.array(neigh[j])))]

    return generated_dict,generated_list_feasible,init_path
def solve_subproblem_and_neighborhood(x,neigh,Internaldata,infinity_val,Adjustable_val,reformulation_dict,logic_fun,sub_solver,init_path):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
        model: GDP model to be solved
        Internaldat: Contains the objective function information of those subproblems that were already solved (It is the same as D during the solution procedure)
        infinity_val: value of infinity
        Adjustable_val: distance from central point at which the points used to calculated convex hull will be located (usually 0.5)
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
        model = build_math()
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


    generated_list_feasible=generated_list_feasible+[x] 

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
                model = build_math()
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
            
           
            
            if status[count]==1:
                generated_list_feasible=generated_list_feasible+[list((np.array(x)+Adjustable_val*np.array(neigh[j])))]

    return generated_dict,generated_list_feasible,init_path



def build_master(lower_b,upper_b,current,stage):

    """
    Function that builds the master problem
        args:
        - stage: stage 1,2 or 3
    """
    initial={}
    if stage==1 or stage==2:
        for n_e in lower_b:
            initial[n_e]=random.randint(lower_b[n_e],upper_b[n_e])  #This allows to consider difficult problems where, e.g., there is a big infeasible region with the same objective function values.
            #initial[n_e]=current[n_e-1] #this initialization method is not that good for finding feasible solutions with "plane" type infeasible regions
    else:
        for n_e in lower_b:
            initial[n_e]=current[n_e-1]


    #print(initial)
    #Model
    m=pe.ConcreteModel(name='Master_problem')
    #External variables (THIS SHOULD BE AUTOMATIC)
    m.x1=pe.Var(within=pe.Integers, bounds=(lower_b[1],upper_b[1]),initialize=initial[1])
    m.x2=pe.Var(within=pe.Integers, bounds=(lower_b[2],upper_b[2]),initialize=initial[2])

    #Known constraints (assumption!!! I know constraints a priori)
    #m.known

    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()



    def obj_rule(m):
        return m.zobj
    
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)

    return m




if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    
    ###REFORMUALTION EXTERNAL VARIABLES
    model =build_math()
    ext_ref = {model.Y1: model.set1, model.Y2: model.set2} #reformulation sets and variables
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=True) 
    print('-------------------------------------------------------------------------- \n \n')
    #INIT_VALUES
    initialization=[14,5] 
    infinity_val=1e+8
    Adjustable_val=0.5
    tol=1E-6
    fobj_actual=infinity_val
    nlp_solver='msnlp'
    neigh=neighborhood_k_eq_inf(2)
    maxiter=100
    iterations=range(1,maxiter+1)

    #No random sampling here: TODO WE HAVE TO THINK HOW CAN WE INTEGRATE THIS
    D_random={}


    #TEST 5: Solve using cuts: first idea
    #GENERATE INITIALIZATION
    model = build_math()
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_math,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE  

        D={}
        D=D_random.copy()


        x_dict={}  #value of x at each iteration
        fval_dict={}   #objective function value at each iteration
        lower_bound_dict={}    #lower bound for the objective function. (is it)

        start = time.time()
        for k in iterations:

            #if first iteration, initialize
            if k==1:
                x_actual=initialization
            m=build_master(lower_bounds,upper_bounds,x_actual,1)
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,_=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math)
            fobj_actual=list(new_values.values())[0]
            fval_dict[k]=fobj_actual  
            #Add points to D
            D.update(new_values)
            #print(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #Calculate new convex hull and dd cuts to the current model
            for i in D: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,list(i))
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            lower_bound_dict[k]=pe.value(m.zobj)
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            #if pe.value(m.zobj)==fobj_actual:
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        print('stage 1: method_1 time:',end - start,'method_1 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
        #print(D)
        print(x_dict)

        #Stage 2
        #First rewrite the dictionary: infinity value for infieasible values and remove the only feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #Now we can start stage 2
        #model = build_math()
        #init_path = generate_initialization(m=model)
        #m_first_values = external_ref(m=model,x=x_actual,extra_logic_function=problem_logic_math,dict_extvar=reformulation_dict,tee=False)
        
        x_dict={}  #value of x at each iteration
        fval_dict={}   #objective function value at each iteration
        lower_bound_dict={}    #lower bound for the objective function. (is it)

        start = time.time()
        for k in iterations:

            #define model
            m=build_master(lower_bounds,upper_bounds,x_actual,2)

            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,_,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)
            fobj_actual=list(new_values.values())[0]
            fval_dict[k]=fobj_actual  
            #Add points to D
            D.update(new_values)
            #print(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #Calculate new convex hull and dd cuts to the current model
            for i in D: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,list(i))
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            lower_bound_dict[k]=pe.value(m.zobj)
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            #if pe.value(m.zobj)==fobj_actual:
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        print('stage 2: method_1 time:',end - start,'method_1 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
        print(x_dict)

        #print(D)
        #Stage 3: Optimization

        #First D must be updated
        ##OPTION 1: EVERYTHING THAT WAS DECLARED INFEASIBLE IN STAGES 1 AND 2 REMAINS INFEASIBLE
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #print(D)
        ##OPTION 2: ONLY THOSE POINTS THAT WHERE DELCARED INFEASIBLE IN STAGE 1 REMAIN INFEASIBLE (probably some points declared infeasible in stage 2 are not really infeasible)
        #D={k: v for k, v in D.items() if v == infinity_val}

        #Bring initialization from previous stages
        init_path=path_result
    else:
        x_actual=initialization
        D={}
        D=D_random.copy()

    x_dict={}  #value of x at each iteration
    fval_dict={}   #objective function value at each iteration
    lower_bound_dict={}    #lower bound for the objective function. (is it)

    start = time.time()
    for k in iterations:

        #define model
        m=build_master(lower_bounds,upper_bounds,x_actual,3)

        #if first iteration, initialize
        #if k==1:
        #    x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,_,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)
        fobj_actual=list(new_values.values())[0]
        fval_dict[k]=fobj_actual  
        #Add points to D
        D.update(new_values)
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in D: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,list(i))
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        lower_bound_dict[k]=pe.value(m.zobj)
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
        
        #if pe.value(m.zobj)==fobj_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    end = time.time()
    print('stage 3: method_1 time:',end - start,'method_1 obj:',D[tuple(x_actual)])
    #print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
    #print(D)
    print(x_dict,'\n')



    #SECOND
    #GENERATE INITIALIZATION
    model = build_math()
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_math,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE 
        D={}
        D=D_random.copy()

        only_feasible_bag=[]
        x_dict={}  #value of x at each iteration
        start = time.time()

        for k in iterations:

            #if first iteration, initialize
            if k==1:
                x_actual=initialization
                only_feasible_bag.extend(list(x) for x in D)
            m=build_master(lower_bounds,upper_bounds,x_actual,1)
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math)
            #print(new_values)
            #print(feasible_n)
            #print(new_values)
            #print(feasible_n)
            #print(all_n)
            only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
            #print(only_feasible_bag)
            #print(only_feasible_bag)
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            
            #print(new_values)

            #Calculate new convex hull and dd cuts to the current model
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            #if 0 in D.values():
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        print('stage1: method_2 time:',end - start,'method_2 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        print(x_dict)


        #Stage 2
        #First rewrite the dictionary: infinity value for infieasible values and remove the only feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #Now we can start stage 2
        only_feasible_bag=[]
        x_dict={}  #value of x at each iteration
        start = time.time()

        for k in iterations:

            #define model
            m=build_master(lower_bounds,upper_bounds,x_actual,2)

            #if first iteration, initialize
            if k==1:
            #    x_actual=initialization
                only_feasible_bag.extend(list(x) for x in D)
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)
            #print(new_values)
            #print(feasible_n)
            #print(new_values)
            #print(feasible_n)
            #print(all_n)
            only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
            #print(only_feasible_bag)
            #print(only_feasible_bag)
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            
            #print(new_values)

            #Calculate new convex hull and dd cuts to the current model
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            #if 0 in D.values():
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        print('stage2: method_2 time:',end - start,'method_2 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        print(x_dict)

        #Stage 3: Optimization

        #First D must be updated
        ##OPTION 1: EVERYTHING THAT WAS DECLARED INFEASIBLE IN STAGES 1 AND 2 REMAINS INFEASIBLE
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #print(D)
        ##OPTION 2: ONLY THOSE POINTS THAT WHERE DELCARED INFEASIBLE IN STAGE 1 REMAIN INFEASIBLE (probably some points declared infeasible in stage 2 are not really infeasible)
        #D={k: v for k, v in D.items() if v == infinity_val}

        #Bring initialization from previous stages
        init_path=path_result
    else:
        x_actual=initialization
        D={}
        D=D_random.copy()

    only_feasible_bag=[]
    x_dict={}  #value of x at each iteration
    start = time.time()

    for k in iterations:

        #define model
        m=build_master(lower_bounds,upper_bounds,x_actual,3)

        #if first iteration, initialize
        if k==1:
        #    x_actual=initialization
            only_feasible_bag.extend(list(x) for x in D)
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)
        #print(new_values)
        #print(feasible_n)
        #print(new_values)
        #print(feasible_n)
        #print(all_n)
        only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
        #print(only_feasible_bag)
        #print(only_feasible_bag)
        #Add points to D
        D.update(new_values)
        
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,i)
            #print(cuts)
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    end = time.time()
    print('stage 3: method_2 time:',end - start,'method_2 obj:',D[tuple(x_actual)])
    #print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    print(x_dict,'\n')





    #TEST 7: Solve using cuts: third idea
    #GENERATE INITIALIZATION
    model = build_math()
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_math,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE  
        D={}
        D=D_random.copy()


        x_dict={}  #value of x at each iteration
        start = time.time()
        for k in iterations:


            #if first iteration, initialize
            if k==1:
                x_actual=initialization
            m=build_master(lower_bounds,upper_bounds,x_actual,1)
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,_=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math)

            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model            
            for i in x_dict:
                cuts=convex_clousure(D,x_dict[i])
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual: 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
            #if 0 in D.values():
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        print('stage 1: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        print(x_dict)


        #Stage 2
        #First rewrite the dictionary: infinity value for infieasible values and remove the only feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        x_dict={}  #value of x at each iteration
        start = time.time()
        for k in iterations:

            #define model
            m=build_master(lower_bounds,upper_bounds,x_actual,2)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,_,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)

            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #print(D)
            #Calculate new convex hull and dd cuts to the current model            
            for i in x_dict:
                cuts=convex_clousure(D,x_dict[i])
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual: 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
            #if 0 in D.values():
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        print('stage 2: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        print(x_dict)

        #Stage 3: Optimization

        #First D must be updated
        ##OPTION 1: EVERYTHING THAT WAS DECLARED INFEASIBLE IN STAGES 1 AND 2 REMAINS INFEASIBLE
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #print(D)
        ##OPTION 2: ONLY THOSE POINTS THAT WHERE DELCARED INFEASIBLE IN STAGE 1 REMAIN INFEASIBLE (probably some points declared infeasible in stage 2 are not really infeasible)
        #D={k: v for k, v in D.items() if v == infinity_val}

        #Bring initialization from previous stages
        init_path=path_result
    else:
        x_actual=initialization
        D={}
        D=D_random.copy()

    x_dict={}  #value of x at each iteration
    start = time.time()
    for k in iterations:

        #define model
        m=build_master(lower_bounds,upper_bounds,x_actual,3)

        #if first iteration, initialize
        #if k==1:
        #    x_actual=initialization
        #print(x_actual)

        #update current value of x in the dictionary
        x_dict[k]=x_actual
        #calculate objective function for current point and its neighborhood (subproblem)
        new_values,_,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)

        #Add points to D
        D.update(new_values)
        #print(D)
        #Calculate new convex hull and dd cuts to the current model            
        for i in x_dict:
            cuts=convex_clousure(D,x_dict[i])
            #print(cuts)
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        #print(new_values)
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual: 
        #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
        #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
        #if 
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

    end = time.time()
    print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
    #print('Cuts calculated from the central points evaluated so far.')
    print(x_dict,'\n')



    #MEJOR VERSION HASTA AHORA, PERO CON EL ERROR DE QUE NO ESTA COSIDERANDO LA INFORMACION DEL RANDOM SAMPLING EN LA PRIMERA ITERACION
#    model = build_math()
    #init_path = generate_initialization(m=model)
#    m_first_values = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_math,dict_extvar=reformulation_dict,tee=False)
    #GENERATE INITIALIZATION
    model = build_math()
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_math,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE   
        D={}
        D=D_random.copy()


        only_feasible_bag=[]


        x_dict={}  #value of x at each iteration

        start = time.time()

        for k in iterations:

            #if first iteration, initialize
            if k==1:
                x_actual=initialization
            m=build_master(lower_bounds,upper_bounds,x_actual,1)
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math)
            #print(new_values)
            #print(feasible_n)
            #print(new_values)
            #print(feasible_n)
            #print(all_n)
            only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
            #print(only_feasible_bag)
            #print(only_feasible_bag)
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            
            #print(new_values)

            #Calculate new convex hull and dd cuts to the current model
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            #if 0 in D.values():
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        print('stage 1: method_4 time:',end - start,'method_4 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        print(x_dict)


        #Stage 2
        #First rewrite the dictionary: infinity value for infieasible values and remove the only feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val



        only_feasible_bag=[]
        x_dict={}  #value of x at each iteration
        start = time.time()

        for k in iterations:

            #define model
            m=build_master(lower_bounds,upper_bounds,x_actual,2)

            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)

            #print(new_values)
            #print(feasible_n)
            #print(new_values)
            #print(feasible_n)
            #print(all_n)
            only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
            #print(only_feasible_bag)
            #print(only_feasible_bag)
            #Add points to D
            D.update(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            
            #print(new_values)

            #Calculate new convex hull and dd cuts to the current model
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            #if 0 in D.values():
                break
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        print('stage 2: method_4 time:',end - start,'method_4 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        print(x_dict)

        #Stage 3: Optimization

        #First D must be updated
        ##OPTION 1: EVERYTHING THAT WAS DECLARED INFEASIBLE IN STAGES 1 AND 2 REMAINS INFEASIBLE
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #print(D)
        ##OPTION 2: ONLY THOSE POINTS THAT WHERE DELCARED INFEASIBLE IN STAGE 1 REMAIN INFEASIBLE (probably some points declared infeasible in stage 2 are not really infeasible)
        #D={k: v for k, v in D.items() if v == infinity_val}

        #Bring initialization from previous stages
        init_path=path_result
    else:
        x_actual=initialization
        D={}
        D=D_random.copy()

    only_feasible_bag=[]
    x_dict={}  #value of x at each iteration
    start = time.time()

    for k in iterations:

        #define model
        m=build_master(lower_bounds,upper_bounds,x_actual,3)

        #if first iteration, initialize
        #if k==1:
        #    x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_math,nlp_solver,init_path)
        #print(new_values)
        #print(feasible_n)
        #print(new_values)
        #print(feasible_n)
        #print(all_n)
        only_feasible_bag.extend(x for x in feasible_n if x not in only_feasible_bag)
        #print(only_feasible_bag)
        #print(only_feasible_bag)
        #Add points to D
        D.update(new_values)
        
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,i)
            #print(cuts)
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    end = time.time()
    print('stage 3: method_4 time:',end - start,'method_4 obj:',D[tuple(x_actual)])
    #print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    print(x_dict,'\n')


