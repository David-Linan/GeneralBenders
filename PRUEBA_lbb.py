from pyomo.environ import *
from pyomo.gdp import Disjunct, Disjunction

#Create a simple model
m = ConcreteModel()
m.x1 = Var(bounds = (0,8))
m.x2 = Var(bounds = (0,8))
m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
m.y1 = Disjunct()
m.y2 = Disjunct()
m.y1.c1 = Constraint(expr=m.x1 >= 2)
m.y1.c2 = Constraint(expr=m.x2 >= 2)
m.y2.c1 = Constraint(expr=m.x1 >= 3)
m.y2.c2 = Constraint(expr=m.x2 >= 3)
m.djn = Disjunction(expr=[m.y1, m.y2])

#Invoke the GDPopt-LBB solver

print(dir(SolverFactory('gdpopt').CONFIG))
print((SolverFactory('gdpopt').CONFIG.init_strategy))

results = SolverFactory('gdpopt').solve(m, strategy='LBB',init_strategy='fix_disjuncts',
minlp_solver='gams',minlp_solver_args=dict(solver='baron', warmstart=True, tee=False),#Only use solvers that can handle LP, MIP (and MINLP if there are nonlinearities in equations) 
mip_solver='gams',mip_solver_args=dict(solver='cplex', warmstart=True, tee=False),
nlp_solver='gams',nlp_solver_args=dict(solver='knitro', warmstart=True, tee=False))

print(results)  
print(results.solver.status)
#ok
print(results.solver.termination_condition)
#optimal

print([value(m.y1.indicator_var), value(m.y2.indicator_var)])
[True, False]