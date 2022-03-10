from pyomo.core.base.set import RangeSet
from scipy import optimize
import itertools as it
from math import fabs
import math
import numpy as np
import pyomo.environ as pe
from math import fabs
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
from gdp_column import build_column,build_column_minlp_gdp
from cuts_functions import convex_clousure,initialization_sampling
from dsda_functions import preprocess_problem,solve_with_gdpopt,solve_with_minlp,solve_with_dsda,neighborhood_k_eq_inf, get_external_information,external_ref,initialize_model,generate_initialization, solve_subproblem
import copy
import time
from pyomo.util.infeasible import log_infeasible_constraints,log_infeasible_bounds,log_close_to_bounds,log_active_constraints
from pyomo.util.blockutil import log_model_constraints
from feasibility_functions import feasibility_1,feasibility_2
import random
import pickle
import logging
from itertools import product

#Random seed for master problem initializations
random.seed(30)

def problem_logic_column(m):
    logic_expr = []
    for n in m.intTrays:
        logic_expr.append([pe.land(~m.YR[n] for n in range(
            m.reboil_tray+1, m.feed_tray)), m.YR_is_down])
        logic_expr.append([pe.land(~m.YB[n]
                                   for n in range(m.feed_tray+1, m.max_trays)), m.YB_is_up])
    for n in m.conditional_trays:
        logic_expr.append([pe.land(pe.lor(m.YR[j] for j in range(n, m.max_trays)), pe.lor(
            pe.land(~m.YB[j] for j in range(n, m.max_trays)), m.YB[n])), m.tray[n].indicator_var])
        logic_expr.append([~pe.land(pe.lor(m.YR[j] for j in range(n, m.max_trays)), pe.lor(
            pe.land(~m.YB[j] for j in range(n, m.max_trays)), m.YB[n])), m.no_tray[n].indicator_var])
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
        model = build_column(8, 17, 0.95, 0.95)
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
                    model = build_column(8, 17, 0.95, 0.95)
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
        model = build_column(8, 17, 0.95, 0.95)
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
                    model2 = build_column(8, 17, 0.95, 0.95)
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
        model = build_column(8, 17, 0.95, 0.95)
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
                model = build_column(8, 17, 0.95, 0.95)
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


def build_master(lower_b,upper_b,current,stage,D,use_random: bool=False):
    """
    Function that builds the master problem

    use_random: True if a random point will be generated for initializations when required. False if you want to use the deterministric strategy
        
    """
    initial={}
    randomp=[]
    if stage==1 or stage==2: #generate random number different from current value and points evaluated so far
        if fabs(float(len(D.keys()))-float(math.prod(upper_b[n_e]-lower_b[n_e]+1 for n_e in lower_b)))<=0.01: #if every point has been evaluated
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
    #External variables (THIS SHOULD BE AUTOMATIC)
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

def run_function(initialization,infinity_val,Adjustable_val,nlp_solver,neigh,maxiter,use_random: bool=False):
    important_info={}
    iterations=range(1,maxiter+1)

    #No random sampling here: TODO WE HAVE TO THINK HOW CAN WE INTEGRATE THIS
    D_random={}


    #TEST 5: Solve using cuts: first idea
    #GENERATE INITIALIZATION
    start=time.time()
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)
    end=time.time()
    #print(end-start)
    #important_info["cuts_first_iter"]=end-start

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE

        D={}
        D=D_random.copy()


        x_dict={}  #value of x at each iteration
        #fval_dict={}   #objective function value at each iteration
        #lower_bound_dict={}    #lower bound for the objective function. (is it)
        fobj_actual=infinity_val

        start = time.time()
        for k in iterations:
            #if first iteration, initialize
            if k==1:
                x_actual=initialization
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,_=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column)
            fobj_actual=list(new_values.values())[0]
            #fval_dict[k]=fobj_actual  
            #Add points to D
            D.update(new_values)
            #print(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #Calculate new convex hull and dd cuts to the current model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,1,D,use_random)
            for i in D: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,list(i))
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #lower_bound_dict[k]=pe.value(m.zobj)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval

                        #TODO: I also have to try: what if I update all values so far in D equal to infinity (Not only the point m.x1,m.x2 where I got stuck??????? )
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        #print('stage 1: method_1 time:',end - start,'method_1 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
        #print(x_dict)
        important_info['m1_s1']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']

        #Stage 2
        #First rewrite the dictionary: infinity value for infieasible values and remove the only feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

        #Now we can start stage 2
        #model = build_column(8, 17, 0.95, 0.95)
        #init_path = generate_initialization(m=model)
        #m_first_values = external_ref(m=model,x=x_actual,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
        
        x_dict={}  #value of x at each iteration
        #fval_dict={}   #objective function value at each iteration
        #lower_bound_dict={}    #lower bound for the objective function. (is it)
        fobj_actual=infinity_val

        start = time.time()
        for k in iterations:
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,_,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
            fobj_actual=list(new_values.values())[0]
            #fval_dict[k]=fobj_actual  
            #Add points to D
            D.update(new_values)
            #print(new_values)
            if 0 in D.values():
                x_actual=list(next(reversed(D.keys())))
                x_dict[str(k)+', neighborhood']=x_actual
                break
            #Calculate new convex hull and dd cuts to the current model
            #define model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,2,D,use_random)
            for i in D: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,list(i))
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #lower_bound_dict[k]=pe.value(m.zobj)
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        #print('stage 2: method_1 time:',end - start,'method_1 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
        #print(x_dict)
        important_info['m1_s2']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']


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
    #fval_dict={}   #objective function value at each iteration
    #lower_bound_dict={}    #lower bound for the objective function. (is it)
    fobj_actual=infinity_val

    start = time.time()
    for k in iterations:
        #if first iteration, initialize
        #if k==1:
        #    x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,_,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
        fobj_actual=list(new_values.values())[0]
        #fval_dict[k]=fobj_actual  
        #Add points to D
        D.update(new_values)
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        #define model
        m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,3,D)
        for i in D: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,list(i))
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        #lower_bound_dict[k]=pe.value(m.zobj)
        if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
        
        #if pe.value(m.zobj)==fobj_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    end = time.time()
    #print('stage 3: method_1 time:',end - start,'method_1 obj:',D[tuple(x_actual)])
    #print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
    #print(x_dict,'\n')
    important_info['m1_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']



    #SECOND
    #GENERATE INITIALIZATION
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE 
        D={}
        D=D_random.copy()

        only_feasible_bag=[]
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val

        start = time.time()

        for k in iterations:



            #if first iteration, initialize
            if k==1:
                x_actual=initialization
                only_feasible_bag.extend(list(x) for x in D)
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column)
            fobj_actual=list(new_values.values())[0]
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
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,1,D,use_random)
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        #print('stage1: method_2 time:',end - start,'method_2 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        #print(x_dict)
        #print(D)
        important_info['m2_s1']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']

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
        fobj_actual=infinity_val

        start = time.time()

        for k in iterations:
            #if first iteration, initialize
            if k==1:
            #    x_actual=initialization
                only_feasible_bag.extend(list(x) for x in D)
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
            fobj_actual=list(new_values.values())[0]
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
            #define model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,2,D,use_random)
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        #print('stage2: method_2 time:',end - start,'method_2 obj:',D[tuple(x_actual)])
        #print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        #print(x_dict)
        important_info['m2_s2']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
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
    fobj_actual=infinity_val

    start = time.time()

    for k in iterations:
        #if first iteration, initialize
        if k==1:
        #    x_actual=initialization
            only_feasible_bag.extend(list(x) for x in D)
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
        fobj_actual=list(new_values.values())[0]
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
        #define model
        m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,3,D)
        for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,i)
            #print(cuts)
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    end = time.time()
    #print('stage 3: method_2 time:',end - start,'method_2 obj:',D[tuple(x_actual)])
    #print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    #print(x_dict,'\n')
    important_info['m2_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']


    #TEST 7: Solve using cuts: third idea
    #GENERATE INITIALIZATION
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE  
        D={}
        D=D_random.copy()


        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:

            #if first iteration, initialize
            if k==1:
                x_actual=initialization
            #print(x_actual)

            #update current value of x in the dictionary
            x_dict[k]=x_actual
            #calculate objective function for current point and its neighborhood (subproblem)
            new_values,_=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column)
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
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

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
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        #print('stage 1: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict)
        important_info['m3_s1']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']

        #Stage 2
        #First rewrite the dictionary: infinity value for infieasible values and remove the only feasible entry
        for j in D:
            if D[j]==0:
                del D[j]
                break
            else:
                D[j]=infinity_val

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
            new_values,_,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
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
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5: 
            #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
            #if 
            #if 0 in D.values():
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]

        end = time.time()
        #print('stage 2: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict)
        important_info['m3_s2']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
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
        new_values,_,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
        fobj_actual=list(new_values.values())[0]
        #Add points to D
        D.update(new_values)
        #print(D)
        #Calculate new convex hull and dd cuts to the current model
        #define model
        m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,3,D)            
        for i in x_dict:
            cuts=convex_clousure(D,x_dict[i])
            #print(cuts)
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        #print(new_values)
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
    #print(x_dict,'\n')
    important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']


    #MEJOR VERSION HASTA AHORA, PERO CON EL ERROR DE QUE NO ESTA COSIDERANDO LA INFORMACION DEL RANDOM SAMPLING EN LA PRIMERA ITERACION
   #model = build_column(8, 17, 0.95, 0.95)
    #init_path = generate_initialization(m=model)
   #m_first_values = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    #GENERATE INITIALIZATION
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)

    if m_init_solved.dsda_status!='Optimal':
        #NOW SOLVE   
        D={}
        D=D_random.copy()


        only_feasible_bag=[]


        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()

        for k in iterations:


            #if first iteration, initialize
            if k==1:
                x_actual=initialization

            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n=solve_subproblem_and_neighborhood_FEAS1(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column)
            fobj_actual=list(new_values.values())[0]
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
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,1,D,use_random)
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        # print('stage 1: method_4 time:',end - start,'method_4 obj:',D[tuple(x_actual)])
        # print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        # print(x_dict)
        important_info['m4_s1']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']

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
        fobj_actual=infinity_val
        start = time.time()

        for k in iterations:

            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            new_values,feasible_n,path_result=solve_subproblem_and_neighborhood_FEAS2(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
            fobj_actual=list(new_values.values())[0]
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
            #define model
            m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,2,D,use_random)
            for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
                cuts=convex_clousure(D,i)
                #print(cuts)
                m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)

            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
                if 0 in D.values():
                    break
                else:
                    if fabs(float(len(D.keys()))-float(math.prod(upper_bounds[n_e]-lower_bounds[n_e]+1 for n_e in lower_bounds)))<=0.01: #if every point has been evaluated
                        break
                    else:
                        x_actual=not_eval
                        D.update({tuple([round(pe.value(m.x1)),round(pe.value(m.x2))]):infinity_val})
            else:
                x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
        end = time.time()
        # print('stage 2: method_4 time:',end - start,'method_4 obj:',D[tuple(x_actual)])
        # print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
        # print(x_dict)
        important_info['m4_s2']=[D[tuple(x_actual)],end - start,'if objective=0-> status is optimal']
                                         
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
    fobj_actual=infinity_val
    start = time.time()

    for k in iterations:
        #if first iteration, initialize
        #if k==1:
        #    x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_column,nlp_solver,init_path)
        fobj_actual=list(new_values.values())[0]
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
        #define model
        m,not_eval=build_master(lower_bounds,upper_bounds,x_actual,3,D)
        for i in only_feasible_bag: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,i)
            #print(cuts)
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if fabs(fobj_actual-pe.value(m.zobj))<=1e-5:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    end = time.time()
    # print('stage 3: method_4 time:',end - start,'method_4 obj:',D[tuple(x_actual)])
    # print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    # print(x_dict,'\n')
    important_info['m4_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']



    #TEST A SOLUTION
    model=build_column(8, 17, 0.95, 0.95)
    m_initialized=initialize_model(m=model,json_path=path_result)
    m_fixed = external_ref(m=m_initialized,x=x_actual,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_solved=solve_subproblem(m=m_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=True)

    #####USE OTHER SOLVERS-----------

    ### SOLVE WITH DSDA  
    #GENERATE INITIALIZATION
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)
    #NOW SOLVE
    try:  
        start = time.time()
        D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_function=build_column,model_args={'min_trays': 8, 'max_trays': 17, 'xD': 0.95, 'xB': 0.95},starting_point=initialization,ext_dict=ext_ref,ext_logic=problem_logic_column, k='Infinity',provide_starting_initialization=True,feasible_model='dsda', subproblem_solver=nlp_solver, iter_timelimit=1000,tee=False,global_tee=True)
        end = time.time()
        print('dsda time:',end - start,'dsda obj:',pe.value(D_SDAsol.obj))
        print(routeDSDA,obj_route)
        #TEST DSDA solution. D_SDAsol is already initialized from a good initialization 
        mDSDA_fixed = external_ref(m=D_SDAsol,x=routeDSDA[-1],extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
        m_DSDA_solved=solve_subproblem(m=mDSDA_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
        try:
            print('Status from DSDA solution: ',m_DSDA_solved.results.solver.termination_condition,'\n')
            important_info['DSDA_'+nlp_solver]=[pe.value(D_SDAsol.obj),end - start,m_DSDA_solved.results.solver.termination_condition]
        except:
            print('Status from DSDA solution: ',m_DSDA_solved.dsda_status,'\n')
            important_info['DSDA_'+nlp_solver]=[pe.value(D_SDAsol.obj),end - start,m_DSDA_solved.dsda_status]
    except:
        print('DSDA FBBT infeasible')
        important_info['DSDA_'+nlp_solver]=[infinity_val,end - start,'DSDA FBBT infeasible']
    #SOLVE WITH MINLP
    #GENERATE INITIALIZATION
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)
    #NOW SOLVE  
    model = build_column_minlp_gdp(initialization,8, 17, 0.95, 0.95)
    m_init=initialize_model(m=model,json_path=init_path)

    start = time.time()
    sub_opt={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > sbb.opt \n','rootsolver '+nlp_solver+'\n','subsolver '+nlp_solver+'\n','$offecho \n']}
    m_solved = solve_with_minlp(m_init, transformation='bigm', minlp='sbb', minlp_options=sub_opt,gams_output=False,tee=False,rel_tol=0)
    end = time.time()
    print('minlp time:',end - start,'minlp obj:',pe.value(m_solved.obj))
    print('Status from MINLP solution: ',m_solved.results.solver.termination_condition,'\n')
    important_info['sbb_'+nlp_solver]=[pe.value(m_solved.obj),end - start,m_solved.results.solver.termination_condition]


    #SOLVE WITH GDP. Boolean fixed at first iteration  
    #GENERATE INITIALIZATION
    model = build_column(8, 17, 0.95, 0.95)
    m_init_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_column,dict_extvar=reformulation_dict,tee=False)
    m_init_solved=solve_subproblem(m=m_init_fixed, subproblem_solver=nlp_solver, timelimit=10000, tee=False)
    init_path = generate_initialization(m=m_init_solved)
    #NOW SOLVE  
    model = build_column_minlp_gdp(initialization,8, 17, 0.95, 0.95) #TODO THIS CODE INITIALIZE YF AND YR ONLY, BUT NOT YP 
    m_init=initialize_model(m=model,json_path=init_path)

    start = time.time()
    m_solved = solve_with_gdpopt(m_init, mip='cplex',nlp=nlp_solver, timelimit=1000,strategy='GLOA', mip_output=False, nlp_output=False,rel_tol=0,tee=False)
    end = time.time()
    #print('gdp time:',end - start,'gdp obj:',pe.value(m_solved.obj))
    #print('Status from GDP-OPT solution: ',m_solved.results.solver.termination_condition,'\n')
    important_info['GLOA_'+nlp_solver]=[pe.value(m_solved.obj),end - start,m_solved.results.solver.termination_condition]

    return important_info 

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    
    ###REFORMUALTION EXTERNAL VARIABLES
    model =build_column(8, 17, 0.95, 0.95)
    ext_ref = {model.YB: model.intTrays, model.YR: model.intTrays} #reformulation sets and variables
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=True) 
    print('-------------------------------------------------------------------------- \n \n')

    # #### SOLUTION WITH DIFFERENT METHODS
    # initialization=[15,9] 
    # infinity_val=1e+5
    # Adjustable_val=0.5
    # nlp_solver='knitro'
    # neigh=neighborhood_k_eq_inf(2)
    # maxiter=100
    # info=run_function(initialization,infinity_val,Adjustable_val,nlp_solver,neigh,maxiter)
    # print(info)


    XX_1,XX_2=np.meshgrid(np.linspace(lower_bounds[1],upper_bounds[1],upper_bounds[1]-lower_bounds[1]+1),np.linspace(lower_bounds[2],upper_bounds[2],upper_bounds[2]-lower_bounds[2]+1))
    X_1=list(np.reshape(XX_1,XX_1.size))
    X_2=list(np.reshape(XX_2,XX_2.size))
    init_tup=list(zip(X_1,X_2))

    initializations=[]
    for i in init_tup:
        if all([i[0]>=9,i[1]<=9]):
            initializations.append(list(i))
    print(initializations)

    #m.known=pe.Constraint(expr=m.x1-m.x2>=7)
    #m.known2=pe.Constraint(expr=m.x1>=9)
    #m.known3=pe.Constraint(expr=m.x2<=9)


    infinity_val=1e+5
    Adjustable_val=0.5
    nlp_solver='knitro'
    neigh=neighborhood_k_eq_inf(2)
    maxiter=100

    dict_of_dicts={}
    count=1
    for i in initializations:
        dict_of_dicts[tuple(i)]=run_function(i,infinity_val,Adjustable_val,nlp_solver,neigh,maxiter)
        print("solved case ",count,"of ",len(initializations),"\n")
        count=count+1

    dictionary_data = dict_of_dicts

    a_file = open("data_distillation_gloa.pkl", "wb")
    pickle.dump(dictionary_data, a_file)
    a_file.close()