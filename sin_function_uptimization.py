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
    x=pe.Var(within=pe.Integers,bounds=(1,5),initialize=25)

    def _obj(m):
        return -x*pe.sin(3.5*math.pi*x)
    m.obj = pe.Objective(rule=_obj, sense=pe.minimize)   





if __name__ == "__main__":
    m=sin_func()

    opt = SolverFactory('gams', solver='sbb')

    m.results = opt.solve(m, tee=True)   





