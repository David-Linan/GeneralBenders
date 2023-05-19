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

if __name__ == "__main__":
    a=neighborhood_k_eq_all(dimension=10)
    print(a)