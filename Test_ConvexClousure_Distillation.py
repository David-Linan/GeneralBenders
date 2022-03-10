from scipy import optimize
import itertools as it
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
import pyDOE2 as DOE


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

def convex_clousure(data,xinitial):
    """
    Data: is a dictionary with tuples for the variables and the objective function value. The convex hull is calculated 
    with respect to this information

    xinitial: Is the point (list) where the convex hull is calculated
    """
    
    A_ub=[]
    b_ub=[]
    x=xinitial.copy()
    x.append(1)
    c=[-i for i in x]
    for i in data:
        
        w=list(i)
        w.append(1)
        A_ub.append(w)
        b_ub.append(data[i])

    minus_solution=optimize.linprog(c=c,A_ub=A_ub,b_ub=b_ub,method='revised simplex',bounds=(None,None))

    #objval=-minus_solution.fun
    #status=minus_solution.message
    variables=minus_solution.x
    return variables

def solve_subproblem_and_neighborhood(x,neigh):
    """
    Function that solves the NLP subproblem for a point and its neighborhood. 
    Args:
        x: central point (list) where the subproblem and the neighborhood solutions are going to be calcualted
        neigh: dictionary with directions.
    Returns:
        generated_dict: A dictionary with the points evaluated and their objective function value (central point and neighborhood).
        generated_list_feasible: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (only feasible).
        generated_list_all: A list with lists: central point and neighborhood but at a infinity (i think) distance of 0.5 (both feasible and infeasible).
    """
    infinity_val=1e+9
    Adjustable_val=0.5
    generated_dict={}
    generated_list_feasible=[] #if required
    Internaldata={(10,1):	22973.10462,
    (10,2):	22959.80455,
    (11,1):	20881.93717,
    (11,2):	20392.21947,
    (11,3):	20355.05845,
    (11,4):	21360.45657,
    (12,1):	20515.8895,
    (12,2):	19853.73216,
    (12,3):	19449.85053,
    (12,4):	19565.86054,
    (12,5):	20916.39147,
    (13,1):	20837.07923,
    (13,2):	20100.29338,
    (13,3):	19548.71845,
    (13,4):	19346.12473,
    (13,5):	19880.1769,
    (13,6):	22283.53684,
    (14,1):	21474.04306,
    (14,2):	20701.2928,
    (14,3):	20081.39547,
    (14,4):	19740.33791,
    (14,5):	19956.97464,
    (14,6):	21453.28822,
    (15,1):	22271.54705,
    (15,2):	21480.40702,
    (15,3):	20826.43131,
    (15,4):	20418.81086,
    (15,5):	20491.91223,
    (15,6):	21621.63046}


    if tuple(x) in Internaldata:
        #assign value to x if feasible
        generated_dict[tuple(x)]=Internaldata[tuple(x)]
        generated_list_feasible=generated_list_feasible+[x] #if required
        #assign value to neighbors
        for j in neigh:
            if tuple(np.array(x)+np.array(neigh[j])) in Internaldata:
                
                generated_dict[tuple(np.array(x)+np.array(neigh[j]))]=Internaldata[tuple(np.array(x)+np.array(neigh[j]))]
                generated_list_feasible=generated_list_feasible+[list((np.array(x)+Adjustable_val*np.array(neigh[j])))]
            else:
                generated_dict[tuple(np.array(x)+np.array(neigh[j]))]=infinity_val #THIS LAINE ACTUALLY HELPS A LOT TO FIND LOCAL SOLUTIONS FASTER!!!!!
    else:
        #Value of infinity if infeasible
        generated_dict[tuple(x)]=infinity_val
        generated_list_feasible=generated_list_feasible+[x]


    return generated_dict,generated_list_feasible



def build_master():

    """
    Function that builds the master problem
        
    """
    #Model
    m=pe.ConcreteModel(name='Master_problem')

    #External variables
    m.x1=pe.Var(within=pe.Integers, bounds=(1,15),initialize=1)
    m.x2=pe.Var(within=pe.Integers, bounds=(1,15),initialize=1)

    #Known constraints (assumption!!! I know constraints a priori)
    m.known=pe.Constraint(expr=m.x1-m.x2>=7)
    m.known2=pe.Constraint(expr=m.x1>=9)
    m.known3=pe.Constraint(expr=m.x2<=9)
    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()

    def obj_rule(m):
        return m.zobj
    
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)

    return m

if __name__ == "__main__":
    # # TEST 1: Inequalities from discrete points

    # #Data to compute convex hull
    # data={(1,1):	9.894736842,
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
    # (5,5):	3.062014577}


    # #Data used to find inequalities (disctrede grid)
    # x1_values=range(1,6)
    # x2_values=range(1,6)
    # grid=list(it.product(x1_values,x2_values))

    # #Initialization of vectors that contain inequailties values
    # fconvex=[]
    # variable_value=[]
    # for i in grid:
    #     solution=convex_clousure(data,list(i))
    #     fconvex.append(solution[0])
    #     variable_value.append(solution[1])



    # #Print inequalities generated
    # for i in range(0,len(grid)):
    #     print('x1*'+str(variable_value[i][0])+'+x2*'+str(variable_value[i][1])+'+'+str(variable_value[i][2])+'<='+'z')

    # TEST 2: calculate the infinity neighborhood (REQUIRED FOR TEST 3)
    #neigh=neighborhood_k_eq_inf(2)
    #print(neigh)


    # # TEST 3: calculate new points to add to convex clousure calculation
    # partial_neigh=solve_subproblem_and_neighborhood([5,5],neigh)
    # print(partial_neigh)



    #TET 4: Build master problem
#    m=build_master()

    initialization=[10,6]
    fobj_actual=1e+9


    #TEST 5: Solve using cuts: first idea
    neigh=neighborhood_k_eq_inf(2)
    D={
    # (11,2):	22973.10462,
    # (11,3):	22959.80455,
    # (12,2):	20881.93717,
    # (12,3):	20392.21947,
    # (12,4):	20355.05845,
    # (12,5):	21360.45657,
    # (13,2):	20515.8895,
    # (13,3):	19853.73216,
    # (13,4):	19449.85053,
    # (13,5):	19565.86054,
    # (13,6):	20916.39147,
    # (14,2):	20837.07923,
    # (14,3):	20100.29338,
    # (14,4):	19548.71845,
    # (14,5):	19346.12473,
    # (14,6):	19880.1769,
    # (14,7):	22283.53684,
    # (15,2):	21474.04306,
    # (15,3):	20701.2928,
    # (15,4):	20081.39547,
    # (15,5):	19740.33791,
    # (15,6):	19956.97464,
    # (15,7):	21453.28822,
    #(16,2):	22271.54705,
    # (16,3):	21480.40702,
    # (16,4):	20826.43131,
    # (16,5):	20418.81086,
    # (16,6):	20491.91223,
    # (16,7):	21621.63046


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
        new_values,_=solve_subproblem_and_neighborhood(x_actual,neigh)
        fobj_actual=list(new_values.values())[0]
        fval_dict[k]=fobj_actual  
        #Add points to D
        D.update(new_values)
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in D: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,list(i))
            #print(cuts)
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
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
    print('method_1')
    print(x_dict)


    neigh=neighborhood_k_eq_inf(2)
    D={
    # (11,2):	22973.10462,
    # (11,3):	22959.80455,
    # (12,2):	20881.93717,
    # (12,3):	20392.21947,
    # (12,4):	20355.05845,
    # (12,5):	21360.45657,
    # (13,2):	20515.8895,
    # (13,3):	19853.73216,
    # (13,4):	19449.85053,
    # (13,5):	19565.86054,
    # (13,6):	20916.39147,
    # (14,2):	20837.07923,
    # (14,3):	20100.29338,
    # (14,4):	19548.71845,
    # (14,5):	19346.12473,
    # (14,6):	19880.1769,
    # (14,7):	22283.53684,
    # (15,2):	21474.04306,
    # (15,3):	20701.2928,
    # (15,4):	20081.39547,
    # (15,5):	19740.33791,
    # (15,6):	19956.97464,
    # (15,7):	21453.28822,
    # (16,2):	22271.54705,
    # (16,3):	21480.40702,
    # (16,4):	20826.43131,
    # (16,5):	20418.81086,
    # (16,6):	20491.91223,
    # (16,7):	21621.63046
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
        new_values,feasible_n=solve_subproblem_and_neighborhood(x_actual,neigh)
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
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('method_2')
    print(x_dict)




    #     #TEST 6: Solve using cuts: second idea
    # neigh=neighborhood_k_eq_inf(2)
    # D={}
    # maxiter=10
    # iterations=range(1,maxiter+1)

    # x_dict={}  #value of x at each iteration

    # for k in iterations:

    #     #define model
    #     m=build_master()

    #     #if first iteration, initialize
    #     if k==1:
    #         x_actual=initialization
    #     #update current value of x in the dictionary
    #     x_dict[k]=x_actual

    #     #print(x_actual)
    #     #calculate objective function for current point and its neighborhood (subproblem)
    #     new_values=solve_subproblem_and_neighborhood(x_actual,neigh)

    #     #Add points to D
    #     D.update(new_values)
    #     #print(D)

    #     #Calculate new convex hull and dd cuts to the current model
    #     #for i in D: #calculate cuts only for current discrete variables in D
    #     cuts=convex_clousure(D,list(x_actual))
    #         #print(cuts)
    #     m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
    #     #Solve master problem       
    #     SolverFactory('gams', solver='cplex').solve(m, tee=False)

    #     #Stop?
    #     #print([pe.value(m.x1),pe.value(m.x2)])
    #     if [pe.value(m.x1),pe.value(m.x2)]==x_actual:
    #         break
    #     else:
    #         x_actual=[pe.value(m.x1),pe.value(m.x2)]
    # print('Cuts are the convex hull calculated at current point only')
    # print(x_dict)


    #TEST 7: Solve using cuts: third idea
    neigh=neighborhood_k_eq_inf(2)
    D={
    # (11,2):	22973.10462,
    # (11,3):	22959.80455,
    # (12,2):	20881.93717,
    # (12,3):	20392.21947,
    # (12,4):	20355.05845,
    # (12,5):	21360.45657,
    # (13,2):	20515.8895,
    # (13,3):	19853.73216,
    # (13,4):	19449.85053,
    # (13,5):	19565.86054,
    # (13,6):	20916.39147,
    # (14,2):	20837.07923,
    # (14,3):	20100.29338,
    # (14,4):	19548.71845,
    # (14,5):	19346.12473,
    # (14,6):	19880.1769,
    # (14,7):	22283.53684,
    # (15,2):	21474.04306,
    # (15,3):	20701.2928,
    # (15,4):	20081.39547,
    # (15,5):	19740.33791,
    # (15,6):	19956.97464,
    # (15,7):	21453.28822,
    # (16,2):	22271.54705,
    # (16,3):	21480.40702,
    # (16,4):	20826.43131,
    # (16,5):	20418.81086,
    # (16,6):	20491.91223,
    # (16,7):	21621.63046


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
        new_values,_=solve_subproblem_and_neighborhood(x_actual,neigh)

        #Add points to D
        D.update(new_values)
        #print(D)
        #Calculate new convex hull and dd cuts to the current model            
        for i in x_dict:
            cuts=convex_clousure(D,x_dict[i])
            #print(cuts)
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
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
    print('method_3')
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
    neigh=neighborhood_k_eq_inf(2)
    D={
    # (11,2):	22973.10462,
    # (11,3):	22959.80455,
    # (12,2):	20881.93717,
    # (12,3):	20392.21947,
    # (12,4):	20355.05845,
    # (12,5):	21360.45657,
    # (13,2):	20515.8895,
    # (13,3):	19853.73216,
    # (13,4):	19449.85053,
    # (13,5):	19565.86054,
    # (13,6):	20916.39147,
    # (14,2):	20837.07923,
    # (14,3):	20100.29338,
    # (14,4):	19548.71845,
    # (14,5):	19346.12473,
    # (14,6):	19880.1769,
    # (14,7):	22283.53684,
    # (15,2):	21474.04306,
    # (15,3):	20701.2928,
    # (15,4):	20081.39547,
    # (15,5):	19740.33791,
    # (15,6):	19956.97464,
    # (15,7):	21453.28822,
    # (16,2):	22271.54705,
    # (16,3):	21480.40702,
    # (16,4):	20826.43131,
    # (16,5):	20418.81086,
    # (16,6):	20491.91223,
    # (16,7):	21621.63046
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
        new_values,feasible_n=solve_subproblem_and_neighborhood(x_actual,neigh)
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
            m.cuts.add(m.x1*cuts[0]+m.x2*cuts[1]+cuts[2]<=m.zobj)
        
        #Solve master problem       
        SolverFactory('gams', solver='cplex').solve(m, tee=False)

        #Stop?
        #print([pe.value(m.x1),pe.value(m.x2)])
        if [round(pe.value(m.x1)),round(pe.value(m.x2))]==x_actual:
            break
        else:
            x_actual=[round(pe.value(m.x1)),round(pe.value(m.x2))]
    print('method_4')
    print(x_dict)


