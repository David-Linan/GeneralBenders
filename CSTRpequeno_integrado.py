from scipy import optimize
import itertools as it
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
from gdp_reactor import build_cstrs
from cuts_functions import convex_clousure
from dsda_functions import neighborhood_k_eq_inf, get_external_information,external_ref,initialize_model,generate_initialization, solve_subproblem
import copy
def problem_logic_cstr(m):
    logic_expr = []
    for n in m.N:
        logic_expr.append([m.YR[n], m.YR_is_recycle[n].indicator_var])
        logic_expr.append([~m.YR[n], m.YR_is_not_recycle[n].indicator_var])
        logic_expr.append([pe.lor(pe.land(~m.YF[n2] for n2 in range(
            1, n)), m.YF[n]), m.YP_is_cstr[n].indicator_var])
        logic_expr.append([~pe.lor(pe.land(~m.YF[n2] for n2 in range(
            1, n)), m.YF[n]), m.YP_is_bypass[n].indicator_var])
        logic_expr.append([pe.lor(pe.land(~m.YF[n2] for n2 in range(
            1, n)), m.YF[n]), m.YP[n]])
    return logic_expr

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
        model = build_cstrs(5)
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
                model = build_cstrs(5)
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



def build_master():

    """
    Function that builds the master problem
        
    """
    #Model
    m=pe.ConcreteModel(name='Master_problem')

    #External variables (THIS SHOULD BE AUTOMATIC)
    m.x1=pe.Var(within=pe.Integers, bounds=(1,5),initialize=1)
    m.x2=pe.Var(within=pe.Integers, bounds=(1,5),initialize=1)

    #Known constraints (assumption!!! I know constraints a priori)
    m.known=pe.Constraint(expr=m.x2-m.x1<=0)

    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()

    def obj_rule(m):
        return m.zobj
    
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)

    return m





if __name__ == "__main__":


    model = build_cstrs(5)
    ext_ref = {model.YF: model.N, model.YR: model.N} #reformulation sets and variables
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False)  
     
    #TEST: SOLVE THE FIXED SUBPROBLEM FOR REFERENCE ONLY

    # m_fixed = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_cstr,dict_extvar=reformulation_dict,tee=False)
    # m_solved = solve_subproblem(m=m_fixed, subproblem_solver='conopt4', timelimit=10000, tee=True)
    # #print(m_solved.dsda_status)
    # init_path = generate_initialization(m=m_solved)
    # m_initialized=initialize_model(m=m_fixed,json_path=init_path)
    # m_solved_from_feasible = solve_subproblem(m=m_initialized, subproblem_solver='conopt4', timelimit=100, tee=True)
    
    
    initialization=[1,1] 
    infinity_val=1e+9
    Adjustable_val=0.5
    fobj_actual=infinity_val
    nlp_solver='knitro'
    neigh=neighborhood_k_eq_inf(2)
        


    #TEST 5: Solve using cuts: first idea
    model = build_cstrs(5)
    init_path = generate_initialization(m=model)
    m_first_values = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_cstr,dict_extvar=reformulation_dict,tee=False)
    
    D={
    # (1,1):	9.894736842,
    # (2,1):	4.061881685,
    # (3,1):	3.314845403,
    # (4,1):	3.134337036,
    # (5,1):	3.130198133,
    # (2,2):	4.061881685,
    # (3,2):	3.314845403,
    # (4,2):	3.134337036,
    # (5,2):	3.130198133,
    # (3,3):	3.314845403,
    # (4,3):	3.134337036,
    # (5,3):	3.130198133,
    # (4,4):	3.133783898,
    # (5,4):	3.130198133,
    # (5,5):	3.062014577
    # 
    }

    maxiter=100
    iterations=range(1,maxiter+1)


    x_dict={}  #value of x at each iteration
    fval_dict={}   #objective function value at each iteration
    lower_bound_dict={}    #lower bound for the objective function. (is it)

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,_,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_cstr,nlp_solver,init_path)
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
    print('Cuts are the convex hull of every point in D. This is actually similar to a D-SDA without line search')
    print(x_dict)

#SECOND
    model = build_cstrs(5)
    init_path = generate_initialization(m=model)
    m_first_values = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_cstr,dict_extvar=reformulation_dict,tee=False)

    D={
    # (1,1):	9.894736842,
    # (2,1):	4.061881685,
    # (3,1):	3.314845403,
    # (4,1):	3.134337036,
    # (5,1):	3.130198133,
    # (2,2):	4.061881685,
    # (3,2):	3.314845403,
    # (4,2):	3.134337036,
    # (5,2):	3.130198133,
    # (3,3):	3.314845403,
    # (4,3):	3.134337036,
    # (5,3):	3.130198133,
    #(4,4):	3.133783898,
    # (5,4):	3.130198133,
    # (5,5):	3.062014577
    }



    only_feasible_bag=[]



    maxiter=100
    iterations=range(1,maxiter+1)


    x_dict={}  #value of x at each iteration

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
            only_feasible_bag.extend(list(x) for x in D)
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_cstr,nlp_solver,init_path)
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
    print('Cuts are the convex hull of every point in D, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    print(x_dict)


    #TEST 7: Solve using cuts: third idea
    model = build_cstrs(5)
    init_path = generate_initialization(m=model)
    m_first_values = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_cstr,dict_extvar=reformulation_dict,tee=False)
    
    D={
    # (1,1):	9.894736842,
    # (2,1):	4.061881685,
    # (3,1):	3.314845403,
    # (4,1):	3.134337036,
    # (5,1):	3.130198133,
    # (2,2):	4.061881685,
    # (3,2):	3.314845403,
    # (4,2):	3.134337036,
    # (5,2):	3.130198133,
    # (3,3):	3.314845403,
    # (4,3):	3.134337036,
    # (5,3):	3.130198133,
    # (4,4):	3.133783898,
    # (5,4):	3.130198133,
    # (5,5):	3.062014577

    }

    maxiter=100
    iterations=range(1,maxiter+1)

    x_dict={}  #value of x at each iteration

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
        #print(x_actual)

        #update current value of x in the dictionary
        x_dict[k]=x_actual
        #calculate objective function for current point and its neighborhood (subproblem)
        new_values,_,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_cstr,nlp_solver,init_path)

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
#        if 
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('Cuts calculated from the central points evaluated so far.')
    print(x_dict)



    #     #TEST 8: Solve using cuts: four idea
    # neigh=neighborhood_k_eq_inf(2)
    # D={}
    # maxiter=100
    # iterations=range(1,maxiter+1)

    # x_dict={}  #value of x at each iteration

    # for k in iterations:

    #     #define model
    #     m=build_master()

    #     #if first iteration, initialize
    #     if k==1:
    #         x_actual=initialization
    #     #print(x_actual)

    #     #update current value of x in the dictionary
    #     x_dict[k]=x_actual
    #     #calculate objective function for current point and its neighborhood (subproblem)
    #     new_values=solve_subproblem_and_neighborhood(x_actual,neigh)

    #     #Add points to D
    #     D.update(new_values)
    #     #print(D)

    #     #Calculate new convex hull and dd cuts to the current model
    #     #for i in D: #calculate cuts only for current discrete variables in D
    #     if k==1:
    #         cuts=convex_clousure(D,list(x_actual))
    #             #print(cuts)
    #         m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
    #     else:
    #         for i in D: #calculate cuts only for current discrete variables in D
    #             cuts=convex_clousure(D,list(i))
    #             #print(cuts)
    #             m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)

    #     #Solve master problem       
    #     SolverFactory('gams', solver='baron').solve(m, tee=False)

    #     #Stop?
    #     #print([pe.value(m.x1),pe.value(m.x2)])
    #     if [pe.value(m.x1),pe.value(m.x2)]==x_actual:
    #         break
    #     else:
    #         x_actual=[pe.value(m.x1),pe.value(m.x2)]
    # print('At first iteration, only initialization for cuts. Then, every point in D is used')
    # print(x_dict)


    #MEJOR VERSION HASTA AHORA, PERO CON EL ERROR DE QUE NO ESTA COSIDERANDO LA INFORMACION DEL RANDOM SAMPLING EN LA PRIMERA ITERACION
    model = build_cstrs(5)
    init_path = generate_initialization(m=model)
    m_first_values = external_ref(m=model,x=initialization,extra_logic_function=problem_logic_cstr,dict_extvar=reformulation_dict,tee=False)
    
    D={
    # (1,1):	9.894736842,
    # (2,1):	4.061881685,
    # (3,1):	3.314845403,
    # (4,1):	3.134337036,
    # (5,1):	3.130198133,
    # (2,2):	4.061881685,
    # (3,2):	3.314845403,
    # (4,2):	3.134337036,
    # (5,2):	3.130198133,
    # (3,3):	3.314845403,
    # (4,3):	3.134337036,
    # (5,3):	3.130198133,
    # (4,4):	3.133783898,
    # (5,4):	3.130198133,
    # (5,5):	3.062014577
    }



    only_feasible_bag=[]



    maxiter=100
    iterations=range(1,maxiter+1)


    x_dict={}  #value of x at each iteration

    for k in iterations:

        #define model
        m=build_master()

        #if first iteration, initialize
        if k==1:
            x_actual=initialization
        #print(x_actual)
        #calculate objective function for current point and its neighborhood (subproblem)
        #update current value of x in the dictionary
        x_dict[k]=x_actual
        new_values,feasible_n,init_path=solve_subproblem_and_neighborhood(x_actual,neigh,D,infinity_val,Adjustable_val,reformulation_dict,problem_logic_cstr,nlp_solver,init_path)
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
    print('Cuts calculated from the central points evaluated so far, and its corresponding feasible negihborhod (but at a distance of 0.5)')
    print(x_dict)
