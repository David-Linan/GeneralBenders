from scipy import optimize
import itertools as it
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
from cuts_functions import initialization_sampling


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
    """
    infinity_val=1e+10
    Adjustable_val=0.5
    generated_dict={}
    generated_list_feasible=[] #if required
    Internaldata={(1,1):3.62538493844036,
(2,1):5.42213171779951,
(3,1):7.21185361676206,
(4,1):8.83663891535067,
(5,1):10.2757572664113,
(1,2):5.42213171779951,
(2,2):6.59359907928721,
(3,2):7.9889108105187,
(4,2):9.37428781734065,
(5,2):10.6614129349276,
(1,3):7.21185361676206,
(2,3):7.9889108105187,
(3,3):9.02376727811947,
(4,3):10.1386261720952,
(5,3):11.2319787969793,
(1,4):8.83663891535067,
(2,4):9.37428781734065,
(3,4):10.1386261720952,
(4,4):11.0134207176556,
(5,4):11.9135181528576,
(1,5):10.2757572664113,
(2,5):10.6614129349276,
(3,5):11.2319787969793,
(4,5):11.9135181528576,
(-2,-1):5.42213171779951,
(-3,-1):7.21185361676206,
(-4,-1):8.83663891535067,
(-5,-1):10.2757572664113,
(-1,-2):5.42213171779951,
(-2,-2):6.59359907928721,
(-3,-2):7.9889108105187,
(-4,-2):9.37428781734065,
(-5,-2):10.6614129349276,
(-1,-3):7.21185361676206,
(-2,-3):7.9889108105187,
(-3,-3):9.02376727811947,
(-4,-3):10.1386261720952,
(-5,-3):11.2319787969793,
(-1,-4):8.83663891535067,
(-2,-4):9.37428781734065,
(-3,-4):10.1386261720952,
(-4,-4):11.0134207176556,
(-5,-4):11.9135181528576,
(-1,-5):10.2757572664113,
(-2,-5):10.6614129349276,
(-3,-5):11.2319787969793,
(-4,-5):11.9135181528576,
(-5,-5):12.6424111765712}


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

    #External variables (THIS SHOULD BE AUTOMATIC)
    m.x1=pe.Var(within=pe.Integers, bounds=(-5,5),initialize=1)
    m.x2=pe.Var(within=pe.Integers, bounds=(-5,5),initialize=1)

    #Known constraints (assumption!!! I know constraints a priori)
    #m.known=pe.Constraint(expr=m.x2-m.x1<=0)

    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()

    def obj_rule(m):
        return m.zobj
    
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)

    return m

if __name__ == "__main__":
    D_random={}
    random_points_number=10
    lower_bounds={1:-5,2:-5}
    upper_bounds={1:5,2:5} 

    if random_points_number >= 1:
        sampled_points=initialization_sampling(random_points_number,lower_bounds,upper_bounds)
        #solve random points
        Internaldata2={(1,1):3.62538493844036,
        (2,1):5.42213171779951,
        (3,1):7.21185361676206,
        (4,1):8.83663891535067,
        (5,1):10.2757572664113,
        (1,2):5.42213171779951,
        (2,2):6.59359907928721,
        (3,2):7.9889108105187,
        (4,2):9.37428781734065,
        (5,2):10.6614129349276,
        (1,3):7.21185361676206,
        (2,3):7.9889108105187,
        (3,3):9.02376727811947,
        (4,3):10.1386261720952,
        (5,3):11.2319787969793,
        (1,4):8.83663891535067,
        (2,4):9.37428781734065,
        (3,4):10.1386261720952,
        (4,4):11.0134207176556,
        (5,4):11.9135181528576,
        (1,5):10.2757572664113,
        (2,5):10.6614129349276,
        (3,5):11.2319787969793,
        (4,5):11.9135181528576,
        (-2,-1):5.42213171779951,
        (-3,-1):7.21185361676206,
        (-4,-1):8.83663891535067,
        (-5,-1):10.2757572664113,
        (-1,-2):5.42213171779951,
        (-2,-2):6.59359907928721,
        (-3,-2):7.9889108105187,
        (-4,-2):9.37428781734065,
        (-5,-2):10.6614129349276,
        (-1,-3):7.21185361676206,
        (-2,-3):7.9889108105187,
        (-3,-3):9.02376727811947,
        (-4,-3):10.1386261720952,
        (-5,-3):11.2319787969793,
        (-1,-4):8.83663891535067,
        (-2,-4):9.37428781734065,
        (-3,-4):10.1386261720952,
        (-4,-4):11.0134207176556,
        (-5,-4):11.9135181528576,
        (-1,-5):10.2757572664113,
        (-2,-5):10.6614129349276,
        (-3,-5):11.2319787969793,
        (-4,-5):11.9135181528576,
        (-5,-5):12.6424111765712}
        for i in sampled_points:
            
            if tuple(i) in Internaldata2:
                D_random[tuple(i)]=Internaldata2[tuple(i)]
            else:
                D_random[tuple(i)]=1e+10
    print('Random points generated:',D_random)
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

    initialization=[-5,-5]
    fobj_actual=1e+13


    #TEST 5: Solve using cuts: first idea
    neigh=neighborhood_k_eq_inf(2)
    D={}
    D=D_random.copy()
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
        #print(D)
        #print(new_values)

        #Calculate new convex hull and dd cuts to the current model
        for i in D: #calculate cuts only for current discrete variables in D
            cuts=convex_clousure(D,list(i))
            #print(cuts)
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
    print('method_1')
    print(x_dict)


    neigh=neighborhood_k_eq_inf(2)
    D={}
    D=D_random.copy()



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
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
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
    D={}
    D=D_random.copy()

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
    D={}
    D=D_random.copy()



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
            m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
        
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




