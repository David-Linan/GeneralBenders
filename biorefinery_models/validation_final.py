from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from dsda_functions import initialize_model, generate_initialization


def reactor_model():
    # ------------pyomo model------------------------------------------------
    m = ConcreteModel(name='reactor_model')
    # ------------scalars    ------------------------------------------------
    m.final_time = Param(initialize=3600)  # [sec]
    m.delta_z = Param(initialize=1.2)  # [m]
    m.L_r = Param(initialize=12)  # [m]
    m.speed = Param(initialize=0.013)  # [m/s]
    # -----------sets--------------------------------------------------------
    # Continuous time set
    m.t = ContinuousSet(bounds=(0, m.final_time))
    # spatial coordinate index (k)
    m.k = RangeSet(0, 10, 1)
    # chemical species
    m.j = Set(initialize=['CS', 'XS', 'AS', 'LS', 'ACS',
              'G', 'XO', 'X', 'A', 'AC', 'F', 'H', 'W', 'O'])
    # Kinetic parameters
    m.Ts = Param(initialize=185)
    m.EA = Param(initialize=61229)
    m.kA = Param(initialize=106225)
    m.kL = Param(initialize=1.03E+33)
    m.EL = Param(initialize=325629)
    m.kXo = Param(initialize=2.78E+31)
    m.EXo = Param(initialize=298011)
    m.kX = Param(initialize=1.31E+34)
    m.EX = Param(initialize=304680)
    m.kG = Param(initialize=1.11E+35)
    m.EG = Param(initialize=335614)
    m.kPL = Param(initialize=1.03E+33)
    m.EPL = Param(initialize=325629)
    m.kF = Param(initialize=5.09E+33)
    m.EF = Param(initialize=327253)
    m.kAc = Param(initialize=4.88E+24)
    m.EAc = Param(initialize=242687)
    m.kH = Param(initialize=1E+31)
    m.EH = Param(initialize=300000)
    m.alpha = Param(initialize=0.1019)
    m.Fin = Param(initialize=6)  # [kg/s]
    m.h0 = Param(initialize=117)  # [kJ/kg]
    m.cb = Param(initialize=3.8)  # [kJ/kg K]
    m.Rg = Param(initialize=8.3145)  # [J/mol K]
    m.C0 = Param(m.j, initialize={'CS': 125, 'XS': 70, 'AS': 8, 'LS': 80, 'ACS': 16,
                 'G': 0, 'XO': 0, 'X': 0, 'A': 0, 'AC': 0, 'F': 0, 'H': 0, 'W': 600, 'O': 101})
    # -----------variables ---------------------------------------------------
    m.c = Var(m.t, m.k, m.j, initialize=500,
              within=NonNegativeReals, bounds=(0, 10000))
    m.dcdt = DerivativeVar(m.c, withrespectto=m.t)
    m.h = Var(m.t, m.k, initialize=100,
              within=NonNegativeReals, bounds=(0, 10000))
    m.dhdt = DerivativeVar(m.h, withrespectto=m.t)
    m.R = Var(m.t, m.k, m.j, within=Reals)
    # ----------Balance equations (concentration and energy)-------------------

    def _dcdt(m, t, k, j):
        if k == m.k.first() or t == m.t.first():
            return Constraint.Skip
        return m.dcdt[t, k, j] == (m.speed/m.delta_z)*(m.c[t, k-1, j]-m.c[t, k, j])+m.R[t, k, j]
    m.c_dcdt = Constraint(m.t, m.k, m.j, rule=_dcdt)

    def _T(m, t, k):
        return m.h[t, k]/m.cb + 273
    m.T = Expression(m.t, m.k, rule=_T)

    def _dhdt(m, t, k):
        if k == m.k.first() or t == m.t.first():
            return Constraint.Skip
        return m.dhdt[t, k] == (m.speed/m.delta_z)*(m.h[t, k-1]-m.h[t, k])-0.000192*(m.T[t, k]-298)

    m.c_dhdt = Constraint(m.t, m.k, rule=_dhdt)

    def _rG(m, t, k):
        return m.kG*exp(-m.EG/(m.Rg*m.T[t, k]))*m.c[t, k, 'CS']
    m.rG = Expression(m.t, m.k, rule=_rG)

    def _rH(m, t, k):
        return m.kH*exp(-m.EH/(m.Rg*m.T[t, k]))*m.c[t, k, 'G']
    m.rH = Expression(m.t, m.k, rule=_rH)

    def _rA(m, t, k):
        return m.kA*exp(-m.EA/(m.Rg*m.T[t, k]))*m.c[t, k, 'AS']
    m.rA = Expression(m.t, m.k, rule=_rA)

    def _rXo(m, t, k):
        return m.kXo*exp(-m.EXo/(m.Rg*m.T[t, k]))*m.c[t, k, 'XS']
    m.rXo = Expression(m.t, m.k, rule=_rXo)

    def _rX(m, t, k):
        return m.kX*exp(-m.EX/(m.Rg*m.T[t, k]))*m.c[t, k, 'XO']
    m.rX = Expression(m.t, m.k, rule=_rX)

    def _rFx(m, t, k):
        return m.kF*exp(-m.EF/(m.Rg*m.T[t, k]))*(m.c[t, k, 'X'])
    m.rFx = Expression(m.t, m.k, rule=_rFx)

    def _rFa(m, t, k):
        return m.kF*exp(-m.EF/(m.Rg*m.T[t, k]))*(m.c[t, k, 'A'])
    m.rFa = Expression(m.t, m.k, rule=_rFa)

    def _rF(m, t, k):
        return m.rFa[t, k]+m.rFx[t, k]
    m.rF = Expression(m.t, m.k, rule=_rF)

    def _rLXo(m, t, k):
        return m.kL*exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'XO'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLXo = Expression(m.t, m.k, rule=_rLXo)

    def _rLX(m, t, k):
        return m.kL*exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'X'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLX = Expression(m.t, m.k, rule=_rLX)

    def _rLA(m, t, k):
        return m.kL*exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'A'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLA = Expression(m.t, m.k, rule=_rLA)

    def _rLG(m, t, k):
        return m.kL*exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'G'])*(m.c[t, k, 'F'] + m.c[t, k, 'H'])
    m.rLG = Expression(m.t, m.k, rule=_rLG)

    def _rL(m, t, k):
        return m.rLXo[t, k]+m.rLX[t, k]+m.rLA[t, k]+m.rLG[t, k]
    m.rL = Expression(m.t, m.k, rule=_rL)

    def _rLF(m, t, k):
        return m.kL*exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'XO']+m.c[t, k, 'X']+m.c[t, k, 'A']+m.c[t, k, 'G'])*(m.c[t, k, 'F'])
    m.rLF = Expression(m.t, m.k, rule=_rLF)

    def _rLH(m, t, k):
        return m.kL*exp(-m.EL/(m.Rg*m.T[t, k]))*(m.c[t, k, 'XO']+m.c[t, k, 'X']+m.c[t, k, 'A']+m.c[t, k, 'G'])*(m.c[t, k, 'H'])
    m.rLH = Expression(m.t, m.k, rule=_rLH)

    def _rAc(m, t, k):
        return m.kAc*exp(-m.EAc/(m.Rg*m.T[t, k]))*(m.c[t, k, 'ACS'])
    m.rAc = Expression(m.t, m.k, rule=_rAc)

    def _R(m, t, k, j):
        if j == 'CS':
            return m.R[t, k, j] == -m.rG[t, k]
        elif j == 'XS':
            return m.R[t, k, j] == -m.rXo[t, k]
        elif j == 'AS':
            return m.R[t, k, j] == -m.rA[t, k]
        elif j == 'LS':
            return m.R[t, k, j] == m.rL[t, k]
        elif j == 'ACS':
            return m.R[t, k, j] == -m.rAc[t, k]
        elif j == 'G':
            return m.R[t, k, j] == m.rG[t, k]-(1-m.alpha)*m.rLG[t, k]
        elif j == 'XO':
            return m.R[t, k, j] == m.rXo[t, k]-m.rX[t, k]-(1-m.alpha)*m.rLXo[t, k]
        elif j == 'X':
            return m.R[t, k, j] == m.rX[t, k]-m.rFx[t, k]-(1-m.alpha)*m.rLX[t, k]
        elif j == 'A':
            return m.R[t, k, j] == m.rA[t, k]-m.rFa[t, k]-(1-m.alpha)*m.rLA[t, k]
        elif j == 'AC':
            return m.R[t, k, j] == m.rAc[t, k]
        elif j == 'F':
            return m.R[t, k, j] == m.rF[t, k] - m.alpha*m.rLF[t, k]
        elif j == 'H':
            return m.R[t, k, j] == m.rH[t, k] - m.alpha*m.rLH[t, k]
        elif j == 'W':
            return m.R[t, k, j] == 0
        else:
            return m.R[t, k, j] == 0

    m.c_R = Constraint(m.t, m.k, m.j, rule=_R)

    def _IC1(m, k, j):
        if k == 0:
            return Constraint.Skip
        return m.c[0, k, j] == m.C0[j]
    m.IC1 = Constraint(m.k, m.j, rule=_IC1)

    def _IC2(m, k):
        if k == 0:
            return Constraint.Skip
        return m.h[0, k] == m.h0
    m.IC2 = Constraint(m.k, rule=_IC2)
    #

    def _IC3(m, t, j):

        return m.c[t, 0, j] == m.C0[j]
    m.IC3 = Constraint(m.t, m.j, rule=_IC3)

    def _IC4(m, t):
        return m.h[t, 0] == m.cb*(m.Ts)
    m.IC4 = Constraint(m.t, rule=_IC4)

    m.obj = Objective(expr=(-m.c[3600, 10, 'G']))

    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')
    # discretizer = TransformationFactory('dae.collocation')
    # discretizer.apply_to(m,nfe=60,ncp=3,wrt=m.t,scheme='LAGRANGE-RADAU')
    return m


if __name__ == "__main__":
    m = reactor_model()
    # opt1 = SolverFactory('gams')
    # results = opt1.solve(m, solver='ipopt', tee=True)
    # solved=generate_initialization(m=m,model_name='validation')
    m=initialize_model(m,from_feasible=True,feasible_model='validation')

    s0 = []
    T1 = []
    t = []
    TT = []
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 = []
    s7 = []
    s8 = []
    s9 = []

    for i in m.k:
        t.append(i*1.2)

        s0.append(m.c[3600, i, 'CS'].value)
        s1.append(m.c[3600, i, 'XS'].value)
        s2.append(m.c[3600, i, 'AS'].value)
        s3.append(m.c[3600, i, 'G'].value)
        s4.append(m.c[3600, i, 'XO'].value)
        s5.append(m.c[3600, i, 'X'].value)
        s6.append(m.c[3600, i, 'A'].value)
        s7.append(m.c[3600, i, 'AC'].value)
        s8.append(m.c[3600, i, 'F'].value)
        T1.append(value(m.T[3600, i]))
    #
    for i in T1:
        a = i-273
        TT.append(a)

    original = pd.read_csv('Cellulose.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--g')
    plt.plot(t, s0, 'g', label='Cellulose ')
    original = pd.read_csv('Xylan.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--m')
    plt.plot(t, s1, 'm', label='Xylan')
    original = pd.read_csv('Arabinan.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--r')
    plt.plot(t, s2, 'r', label='Arabinan')

    plt.xlabel('length,m')
    plt.ylabel('Concentration, g/kg')
    plt.legend()
    plt.show()

    original = pd.read_csv('Glucose.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--g')
    plt.plot(t, s3, 'g', label='Glucose')
    original = pd.read_csv('Xylose.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--m')
    plt.plot(t, s5, 'm', label='Xylose')
    original = pd.read_csv('Arabinose.csv', header=None)
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--r')
    plt.plot(t, s6, 'r', label='Arabinose')

    plt.xlabel('length,m')
    plt.ylabel('Concentration, g/kg')
    plt.legend()
    plt.show()

    original = pd.read_csv('Temperature.csv', header=None)
    plt.plot(t, TT, 'k')
    plt.plot(original.iloc[:, 0].values, original.iloc[:, 1].values, '--k')
    plt.xlabel('length,m')
    plt.ylabel('Temperature, °C')
    plt.show()

    p1 = list(m.t)[1]
    p2 = list(m.t)[5]
    p3 = list(m.t)[10]
    poss = [p1, p2, p3]
    for pos in poss:
        Ttime = []
        TTime = []
        for i in m.k:
            Ttime.append(value(m.T[pos, i]))
        #
        for i in Ttime:
            a = i-273
            TTime.append(a)
        plt.plot(t, TTime)

    plt.legend(['60 seconds', '300 seconds', '600 seconds'])
    plt.xlabel('length,m')
    plt.ylabel('Temperature, °C')
    plt.show()

    p1 = list(m.t)[-40]
    p2 = list(m.t)[-25]
    p3 = list(m.t)[-1]
    poss = [p1, p2, p3]
    for pos in poss:
        Ttime = []
        TTime = []
        for i in m.k:
            Ttime.append(value(m.T[pos, i]))
        #
        for i in Ttime:
            a = i-273
            TTime.append(a)
        plt.plot(t, TTime)

    plt.legend(['1260 seconds', '2160 seconds', '3600 seconds'])
    plt.xlabel('length,m')
    plt.ylabel('Temperature, °C')
    plt.show()



    poss = list(m.t)
    for pos in poss:
        Ttime = []
        TTime = []
        for i in m.k:
            Ttime.append(value(m.T[pos, i]))
        #
        for i in Ttime:
            a = i-273
            TTime.append(a)
        plt.plot(t, TTime)
    #plt.legend(poss)
    plt.ylim(170, 185)
    plt.xlabel('length,m')
    plt.ylabel('Temperature, °C')
    plt.colorbar(poss)
    plt.show()
