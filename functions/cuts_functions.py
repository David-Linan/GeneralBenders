from scipy import optimize 
import itertools as it
import math
import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import SolverFactory
import os
from decimal import Decimal
from smt.sampling_methods import LHS
import random


def convex_clousure(data,xinitial):
    """
    Args:
        Data: is a dictionary with tuples for the variables and the objective function value. The convex hull is calculated 
        with respect to this information

        xinitial: Is the point (list) where the convex hull is calculated

    returns:
        variables: value of the variables that optimize the problem
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

    minus_solution=optimize.linprog(c=c,A_ub=A_ub,b_ub=b_ub,method= 'highs-ds',bounds=(None,None))

    #objval=-minus_solution.fun
    #status=minus_solution.message
    #print(minus_solution.status)
    #print(minus_solution.message)
    variables=minus_solution.x
    return variables


def initialization_sampling(number_points,lower_bounds,upper_bounds,input_criterion: str='ese'):

    """
    Function that returns the points to be evaluated in the random sampling step
    """
    if number_points<=0:
        number_points=1
    list_of_limits=[]

    for i in lower_bounds:
        list_of_limits.append([lower_bounds[i],upper_bounds[i]])

    xlimits = np.array(list_of_limits)
    sampling = LHS(xlimits=xlimits,criterion=input_criterion)#,random_state=1)
    num = number_points
    x = sampling(num)
      
    newx=[] #discrete values
    for i in x.tolist():
        partial=[round(j) for j in i]
        if partial not in newx: #without repeated values
            newx.append(partial)
    newx.sort(reverse=True)
    return newx   #list with discrete randomly sampled values (without repetition)

def initialization_sampling_naive(number_points,lower_bounds,upper_bounds):

    """
    Function that returns the points to be evaluated in the random sampling step
    """
    if number_points<=0:
        number_points=1
    dimension=len(lower_bounds.keys())
    multip=1
    for i in lower_bounds.keys():
        partial_lower=lower_bounds[i]
        partial_upper=upper_bounds[i]
        multip=multip*(partial_upper-partial_lower+1)
    if number_points>=multip:
        number_points=multip
    rng=np.random.default_rng()
    newx=[]
    contador=0
    while True:
        current_random=[]
        for j in range(1,dimension+1):
            current_random_value=rng.integers(lower_bounds[j],high=upper_bounds[j],size=None,endpoint=True)
            current_random.append(current_random_value)
        #print(current_random)
        if number_points==1:
            newx.append(current_random)
            break
        else:
            if contador==0:
                contador=contador+1
                newx.append(current_random)
            else:                
                #print([current_random!=newx[k] for k in range(len(newx))])
                if all([current_random!=newx[k] for k in range(len(newx))]):
                    contador=contador+1
                    newx.append(current_random)
                    #print(contador)
                    #print(number_points)
                    
                    if contador==number_points:
                        break
     #   print(contador)
    newx.sort(reverse=True)

    return newx   #list with discrete randomly sampled values (without repetition)
###TESTS




# respuesta=initialization_sampling(2,{1: 1, 2: 1, 3: 1},{1: 5, 2: 5, 3:5})
# print('LHS',respuesta)
# #print(len(respuesta))


# respuesta=initialization_sampling_naive(2,{1: 1, 2: 1, 3: 1},{1: 5, 2: 5, 3:5})
# print('Naive',respuesta)
# #print(len(respuesta))


# respuesta=initialization_sampling(25,{1: 1, 2: 1},{1: 5, 2: 5})
# print('LHS',respuesta)
# #print(len(respuesta))


# respuesta=initialization_sampling_naive(24,{1: 1, 2: 1},{1: 5, 2: 5})
# print('Naive',respuesta)
# #print(len(respuesta))








