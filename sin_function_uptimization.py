from __future__ import division
import pyomo.environ as pe
import pyomo.dae as dae
import math
import os
import io
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory
from pyomo.gdp import Disjunct, Disjunction
import itertools

def sin_func():
    m = pe.ConcreteModel()
    m.x=pe.Var(within=pe.Integers,bounds=(1,5),initialize=3)

    def _obj(m):
        return -m.x*pe.sin(3.5*math.pi*m.x)
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)   

    return m





if __name__ == "__main__":
    m=sin_func()

    opt = SolverFactory('gams', solver='shot')

    results = opt.solve(m, tee=True)   
    print('objective=',pe.value(m.obj),'x=',pe.value(m.x))





