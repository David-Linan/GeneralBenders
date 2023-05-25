# Biorefinery Distillation Column Model Pt. 2

import pyomo.environ as pyo
import pyomo.dae as dae
import matplotlib.pyplot as plt

# Assumptions
#   - distillate drum and tray holdups are constant
#   - binary system with constant volatility
#   - equimolar flow (energy balance not required)
#   - VLE is attained in each tray (trays are ideal)
#   - vapour holdup is negligible compared with liquid holdup


def create_distillation_model(N, tf, x0, IE0, a, V, HT, HC, HB, CV, ZS, xdset):
    # Create model
    model = pyo.ConcreteModel()

    # Define model index sets
    model.N = pyo.RangeSet(1, N+2)
    model.T = dae.ContinuousSet(within=pyo.NonNegativeReals, bounds=(0, tf))

    # Define model parameters
    model.a = pyo.Param(initialize=a)
    model.V = pyo.Param(initialize=V)
    model.HT = pyo.Param(initialize=HT)
    model.HC = pyo.Param(initialize=HC)
    model.HB = pyo.Param(initialize=HB)
    model.CV = pyo.Param(initialize=CV)
    model.ZS = pyo.Param(initialize=ZS)
    model.xdset = pyo.Param(initialize=xdset)

    # Define model variables
    model.x = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.y = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))

    model.E = pyo.Var(model.T)
    model.E_sq = pyo.Var(model.T)

    model.D = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.R = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.dx = dae.DerivativeVar(model.x, wrt=model.T)

    # Define model constraints
    def equilibrium_rule(model, N, T):
        return(model.y[N, T] == (model.a*model.x[N, T])/(1 + (model.a - 1)*model.x[N, T]))
    model.equilibrium_constraint = pyo.Constraint(model.N, model.T, rule = equilibrium_rule)

    def error_rule(model, T):
        return(model.E[T] == model.x[model.N.last(), T] - model.xdset)
    model.error_constraint = pyo.Constraint(model.T, rule = error_rule)
    
    def squared_error_rule(model, T):
        return(model.E_sq[T] == model.E[T]**2)
    model.E_squared_constraint = pyo.Constraint(model.T, rule = squared_error_rule)

    def reflux_rule(model, T):
        return(model.R[T] == model.V - model.D[T])
    model.reflux_constraint = pyo.Constraint(model.T, rule = reflux_rule)

    def x_derivative_rule(model, N, T):
        if T == model.T.first():
            return(model.x[N, T] == x0)
        else:
            if N == model.N.first():
                return(model.HB*model.dx[N, T] == (model.R[T]*model.x[N+1, T] - model.V*model.y[N, T]))
            elif N == model.N.last():
                return(model.HC*model.dx[N, T] == model.V*model.y[N-1, T] - model.R[T]*model.x[N, T] - model.D[T]*model.x[N, T])
            else:
                return(model.HT*model.dx[N, T] == model.R[T]*model.x[N+1, T] + model.V*model.y[N-1, T] - model.R[T]*model.x[N, T] - model.V*model.y[N, T])
    model.x_derivative_constraint = pyo.Constraint(model.N, model.T, rule = x_derivative_rule)



    #Define discretization
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    
    #Define model objective
    def objective_rule(model):
        return(sum(model.E_sq[t] for t in model.T))
    model.objective = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
    model.objective.pprint()

    #Return model
    return(model)


# USER INPUTS
N = 10
tf = 50

x0 = 0.5
IE0 = 0

a = 2.5
V = 5
HT = 0.5
HC = 3
HB = 100
CV = 2.309
ZS = 0.5
xdset = 1

# CREATE AND SOLVE DISTILLATION MODEL
distillation_model = create_distillation_model(N, tf, x0, IE0, a, V, HT, HC, HB, CV, ZS, xdset)

solver = pyo.SolverFactory('gams', solver='conopt4')
res = solver.solve(distillation_model, tee=True)

# PLOT RESULTS
xd_obtained = []
for t in distillation_model.T:
    xd_obtained.append(
        pyo.value(distillation_model.x[distillation_model.N.last(), t]))

D_obtained = []
for t in distillation_model.T:
    D_obtained.append(pyo.value(distillation_model.D[t]))

# blank_obtained = []
# for t in distillation_model.T:
#     blank_obtained.append(pyo.value(distillation_model.blank[t]))

t_vals = []
for t in distillation_model.T:
    t_vals.append(t)

plt.plot(t_vals, xd_obtained)
plt.show()
