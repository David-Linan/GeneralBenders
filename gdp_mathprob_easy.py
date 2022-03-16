from __future__ import division
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math


def build_math():

    #-----Model
    m=pe.ConcreteModel(name='gdp_math')

    #-----Parameters
    m.pi=pe.Param(initialize=math.pi)

    #-----Sets
    m.set1=pe.RangeSet(1,5,doc= "set of first group of Boolean variables")
    m.set2=pe.RangeSet(1,5,doc= "set of first group of Boolean variables")

    m.sub1 = pe.Set(initialize=[3],within=m.set1)

    #-----Variables
    m.Y1=pe.BooleanVar(m.set1,doc="Boolean variable associated to set 1")
    m.Y2=pe.BooleanVar(m.set2,doc="Boolean variable associated to set 2")
    #Continuous variables calcualted depending on disjunctions
    m.alpha=pe.Var(within=pe.Reals)
    m.beta=pe.Var(within=pe.Reals)


    #-----Constraints independent of the disjunctions
    #def constraint0(m):
#   #     return -pe.sin(4*m.pi*m.alpha)+2*(pe.sin(2*m.pi*m.beta)**2)>=1.5
    #    return -pe.sin(4*m.pi*m.alpha)+2*(pow(pe.sin(2*m.pi*m.beta),2))>=1.5
    #m.constraint0=pe.Constraint(rule=constraint0)

    #-----Objective function
    def obj_fun(m):
#        return 4*(m.alpha**2)-2.1*(m.alpha**4)+(1/3)*(m.alpha**6)+m.alpha*m.beta-4*(m.beta**2)+4*(m.beta**4)
        return 4*(pow(m.alpha,2))-2.1*(pow(m.alpha,4))+(1/3)*(pow(m.alpha,6))+m.alpha*m.beta-4*(pow(m.beta,2))+4*(pow(m.beta,4))
    m.obj=pe.Objective(rule=obj_fun,sense=pe.minimize)

    #-----First disjunction
    def build_disjuncts1(m,set1):  #Disjuncts for first Boolean variable

        def constraint1(m):
            return m.model().alpha==-0.1+0.1*(set1-1) #.model() is required when writing constraints inside disjuncts
        m.constraint1=pe.Constraint(rule=constraint1)
    
    m.Y1_disjunct=Disjunct(m.set1,rule=build_disjuncts1,doc="each disjunct is defined over set 1")


    # def Disjunction1(m):    #Disjunction for first Boolean variable
    #     return [m.Y1_disjunct[j] for j in m.set1]
    # m.Disjunction1=Disjunction(rule=Disjunction1,xor=False)

    #Associate boolean variables to disjuncts
    for n1 in m.set1:
        m.Y1[n1].associate_binary_var(m.Y1_disjunct[n1].indicator_var)

    #-----Second disjunction
    def build_disjuncts2(m,set2):  #Disjuncts for second Boolean variable

        def constraint2(m):
            return m.model().beta==-0.9+0.1*(set2-1) #.model() is required when writing constraints inside disjuncts
        m.constraint2=pe.Constraint(rule=constraint2)
    
    m.Y2_disjunct=Disjunct(m.set2,rule=build_disjuncts2,doc="each disjunct is defined over set 2")


    # def Disjunction2(m):    #Disjunction for first Boolean variable
    #     return [m.Y2_disjunct[j] for j in m.set2]
    # m.Disjunction2=Disjunction(rule=Disjunction2,xor=False)


    #Associate boolean variables to disjuncts
    for n2 in m.set2:
        m.Y2[n2].associate_binary_var(m.Y2_disjunct[n2].indicator_var)


    #-----Logical constraints

    #Constraint that allow to apply the reformulation over Y1
    def select_one_Y1(m):
        return pe.exactly(1,m.Y1)
    m.oneY1=pe.LogicalConstraint(rule=select_one_Y1)

    #Constraint that allow to apply the reformulation over Y2
    def select_one_Y2(m):
        return pe.exactly(1,m.Y2)
    m.oneY2=pe.LogicalConstraint(rule=select_one_Y2)

    #Constraint that define an infeasible region with respect to Boolean variables

    def infeasR_rule(m):
        return pe.land([pe.lnot(m.Y1[j]) for j in m.sub1])
    m.infeasR=pe.LogicalConstraint(rule=infeasR_rule)


    #-----Return model
    return m






