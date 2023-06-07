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

def create_distillation_model_simplified(N, tf, x0, a, V, HT, HC, HB, CV, ZS, xdset):
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
    model.I1 = pyo.Var(model.T, within = pyo.NonNegativeReals)
    model.I2 = pyo.Var(model.T, within = pyo.NonNegativeReals)
    model.xd_average = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, 1), initialize = x0)
    model.D = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds = (0.01, V))
    model.R = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.dx = dae.DerivativeVar(model.x, wrt=model.T)
    model.dI1 = dae.DerivativeVar(model.I1, wrt=model.T)
    model.dI2 = dae.DerivativeVar(model.I2, wrt=model.T)

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
    
    def xd_average_rule(model, T):
        if T == model.T.first():
            return(model.xd_average[T] == x0)
        else:
            return(model.I2[T]*model.xd_average[T] == model.I1[T])
    model.xd_average_constraint = pyo.Constraint(model.T, rule = xd_average_rule)
    
    def xd_average_final_rule(model, T):
        if T == model.T.last():
            return(model.xd_average[T] >= model.xdset)
        else:
            return(pyo.Constraint.Skip)
    model.xd_average_final_constraint = pyo.Constraint(model.T, rule = xd_average_final_rule)

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

    def I1_derivative_rule(model, T):
        return(model.dI1[T] == model.D[T]*model.x[model.N.last(), T])
    model.I1_derivative_constraint = pyo.Constraint(model.T, rule = I1_derivative_rule)
    
    def I2_derivative_rule(model, T):
        return(model.dI2[T] == model.D[T])
    model.I2_derivative_constraint = pyo.Constraint(model.T, rule = I2_derivative_rule)
    
    #Define discretization
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
    
    #Define model objective
    def objective_rule(model):
        return(sum(model.E_sq[t] for t in model.T))
    model.objective = pyo.Objective(rule = objective_rule)

    #Return model
    return(model)

def create_distillation_model_FixedVaporFlow_ConstantReflux(N, tf, x0, a, V, HT, HC, HB, CV, ZS, xdset):
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
    model.CV = pyo.Param(initialize=CV)
    model.ZS = pyo.Param(initialize=ZS)
    model.xdset = pyo.Param(initialize=xdset)

    # Define model variables
    model.x = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.y = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.E = pyo.Var(model.T)
    model.E_sq = pyo.Var(model.T)
    model.I1 = pyo.Var(model.T, within = pyo.NonNegativeReals)
    model.I2 = pyo.Var(model.T, within = pyo.NonNegativeReals)
    model.xd_average = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, 1), initialize = x0)
    model.D = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds = (0.0001, V))
    model.R = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.dx = dae.DerivativeVar(model.x, wrt=model.T)
    model.dI1 = dae.DerivativeVar(model.I1, wrt=model.T)
    model.dI2 = dae.DerivativeVar(model.I2, wrt=model.T)

    # New variables do consider variable reboiler hold-up NOTE: NEW
    model.HB = pyo.Var(model.T,within=pyo.NonNegativeReals,bounds=(0,HB),initialize=HB)
    model.HBx= pyo.Var(model.N,model.T,within=pyo.NonNegativeReals,bounds=(0,HB),initialize=HB)
    model.dHBX=dae.DerivativeVar(model.HBx, wrt=model.T)
    model.dHB=dae.DerivativeVar(model.HB,wrt=model.T)

    # New constraint to define the multiplication of hold-up and concentration in the reboiler NOTE: NEW
    def def_HBx_rule(model,N,T):
        if N == model.N.first():
            if T==model.T.first():
                return model.HBx[N,T] == HB*x0
            else:
                return model.HBx[N,T] == model.HB[T]*model.x[N,T]
        else:
            return pyo.Constraint.Skip
    model.def_HBx_constraint=pyo.Constraint(model.N,model.T,rule=def_HBx_rule)

    # New constraint to consider total molar balance for the kettle NOTE: NEW
    def x_derivative_total_rule(model, N, T):
        if N == model.N.first():
            if T==model.T.first():
                return model.HB[T] == HB
            else:
                return model.dHB[T] == model.R[T] - model.V
        else:
            return pyo.Constraint.Skip
    model.x_derivative_total_constraint = pyo.Constraint(model.N, model.T, rule = x_derivative_total_rule)

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
    
    def xd_average_rule(model, T):
        if T == model.T.first():
            return(model.xd_average[T] == x0)
        else:
            return(model.I2[T]*model.xd_average[T] == model.I1[T])
    model.xd_average_constraint = pyo.Constraint(model.T, rule = xd_average_rule)
    
    def xd_average_final_rule(model, T):
        if T == model.T.last():
            return(model.xd_average[T] >= model.xdset)
        else:
            return(pyo.Constraint.Skip)
    model.xd_average_final_constraint = pyo.Constraint(model.T, rule = xd_average_final_rule)

    def x_derivative_rule(model, N, T):
        if T == model.T.first():
            return(model.x[N, T] == x0)
        else:
            if N == model.N.first():
                return(model.dHBX[N, T] == (model.R[T]*model.x[N+1, T] - model.V*model.y[N, T])) #NOTE: modified constraint to consider variable holdup
            elif N == model.N.last():
                return(model.HC*model.dx[N, T] == model.V*model.y[N-1, T] - model.R[T]*model.x[N, T] - model.D[T]*model.x[N, T])
            else:
                return(model.HT*model.dx[N, T] == model.R[T]*model.x[N+1, T] + model.V*model.y[N-1, T] - model.R[T]*model.x[N, T] - model.V*model.y[N, T])
    model.x_derivative_constraint = pyo.Constraint(model.N, model.T, rule = x_derivative_rule)

    def I1_derivative_rule(model, T):
        if T==model.T.first():
            return model.I1[T] == 0
        else:
            return(model.dI1[T] == model.D[T]*model.x[model.N.last(), T])
    model.I1_derivative_constraint = pyo.Constraint(model.T, rule = I1_derivative_rule)
    
    def I2_derivative_rule(model, T):
        if T == model.T.first():
            return model.I2[T] == 0
        else:
            return(model.dI2[T] == model.D[T])
    model.I2_derivative_constraint = pyo.Constraint(model.T, rule = I2_derivative_rule)
    
    #Define discretization
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
    
    #Define model objective
    def objective_rule(model):
        # return(sum(model.E_sq[t] for t in model.T))
        return -(0*model.xd_average[model.T.last()]+model.I1[model.T.last()])
    model.objective = pyo.Objective(rule = objective_rule)

    #Return model
    def rule_cons(model,T):
        if T > model.T.first():
            return model.R[T]/model.D[T] == model.R[model.T.prev(T)]/model.D[model.T.prev(T)]
        else:
            return pyo.Constraint.Skip

    model.ConstantReflux=pyo.Constraint(model.T,rule=rule_cons) 

    return(model)

def create_distillation_model(N, tf, x0, a, V, HT, HC, HB, CV, ZS, xdset):
    # Create model
    model = pyo.ConcreteModel()

    # Define model index sets
    model.N = pyo.RangeSet(1, N+2)
    model.T = dae.ContinuousSet(within=pyo.NonNegativeReals, bounds=(0, 1))

    # Define model parameters
    model.a = pyo.Param(initialize=a)
    model.HT = pyo.Param(initialize=HT)
    model.HC = pyo.Param(initialize=HC)
    model.CV = pyo.Param(initialize=CV)
    model.ZS = pyo.Param(initialize=ZS)
    model.xdset = pyo.Param(initialize=xdset)
    model.ratio=pyo.Param(initialize=0.01, doc='ratio liquid/vapor density, and it is constant. Assumption: every component with same molecular weihgt.')


    # Define model variables
    model.variableTime=pyo.Var(within=pyo.NonNegativeReals,bounds=(0, tf))
    model.HB0var=pyo.Var(within=pyo.NonNegativeReals,bounds=(0,HB))
    # model.variableTime.fix(2) #TODO: unfix
    # model.HB0var.fix(3.75) #TODO: unfix
    model.x = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, bounds=(0, 1), doc='kmol/kmol')
    model.y = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, bounds=(0, 1), doc='kmol/kmol')
    model.E = pyo.Var(model.T)
    model.E_sq = pyo.Var(model.T)
    model.I1 = pyo.Var(model.T, within = pyo.NonNegativeReals,doc='m^3')
    model.I2 = pyo.Var(model.T, within = pyo.NonNegativeReals,doc='m^3')
    model.xd_average = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, 1), initialize = x0)
    model.D = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds = (0.0001, V),doc='m^3/h') #TODO: verify if this bounds are apropriate
    model.V = pyo.Var(model.T,within=pyo.NonNegativeReals, bounds=(0,V),doc='m^3/h')
    model.R = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0,V),doc='m^3/h')
    model.dx = dae.DerivativeVar(model.x, wrt=model.T)
    model.dI1 = dae.DerivativeVar(model.I1, wrt=model.T)
    model.dI2 = dae.DerivativeVar(model.I2, wrt=model.T)
    model.I_V=pyo.Var(model.T,within=pyo.NonNegativeReals,doc='Integral of boil-up rate' )
    model.dI_V=dae.DerivativeVar(model.I_V, wrt=model.T)

    # New variables do consider variable reboiler hold-up NOTE: NEW
    model.HB = pyo.Var(model.T,within=pyo.NonNegativeReals,bounds=(0,HB),initialize=3, doc ='m^3')
    model.HBx= pyo.Var(model.N,model.T,within=pyo.NonNegativeReals,bounds=(0,HB),initialize=3)
    model.dHBX=dae.DerivativeVar(model.HBx, wrt=model.T)
    model.dHB=dae.DerivativeVar(model.HB,wrt=model.T,doc='m^3/h')

    # New constraint to define the multiplication of hold-up and concentration in the reboiler NOTE: NEW
    def def_HBx_rule(model,N,T):
        if N == model.N.first():
            if T==model.T.first():
                return model.HBx[N,T] == model.HB0var*x0
            else:
                return model.HBx[N,T] == model.HB[T]*model.x[N,T]
        else:
            return pyo.Constraint.Skip
    model.def_HBx_constraint=pyo.Constraint(model.N,model.T,rule=def_HBx_rule)

    # New constraint to consider total molar balance for the kettle NOTE: NEW
    def x_derivative_total_rule(model, N, T):
        if N == model.N.first():
            if T==model.T.first():
                return model.HB[T] == model.HB0var
            else:
                return model.dHB[T] == model.variableTime*(model.R[T] - model.ratio*model.V[T])
        else:
            return pyo.Constraint.Skip
    model.x_derivative_total_constraint = pyo.Constraint(model.N, model.T, rule = x_derivative_total_rule)

    # Define model constraints
    def equilibrium_rule(model, N, T):
        return(model.y[N, T] == (model.a*model.x[N, T])/(1 + (model.a - 1)*model.x[N, T]))
    model.equilibrium_constraint = pyo.Constraint(model.N, model.T, rule = equilibrium_rule)

    def reflux_rule(model, T):
        return(model.R[T] == model.ratio*model.V[T] - model.D[T])
    model.reflux_constraint = pyo.Constraint(model.T, rule = reflux_rule)
    
    def xd_average_rule(model, T):
        if T == model.T.first():
            return(model.xd_average[T] == x0)
        else:
            return(model.I2[T]*model.xd_average[T] == model.I1[T])
    model.xd_average_constraint = pyo.Constraint(model.T, rule = xd_average_rule)
    


    def x_derivative_rule(model, N, T):
        if T == model.T.first():
            return(model.x[N, T] == x0)
        else:
            if N == model.N.first():
                return(model.dHBX[N, T] == model.variableTime*(model.R[T]*model.x[N+1, T] - model.ratio*model.V[T]*model.y[N, T])) #NOTE: modified constraint to consider variable holdup
            elif N == model.N.last():
                return(model.HC*model.dx[N, T] == model.variableTime*(model.ratio*model.V[T]*model.y[N-1, T] - model.R[T]*model.x[N, T] - model.D[T]*model.x[N, T]))
            else:
                return(model.HT*model.dx[N, T] == model.variableTime*(model.R[T]*model.x[N+1, T] + model.ratio*model.V[T]*model.y[N-1, T] - model.R[T]*model.x[N, T] - model.ratio*model.V[T]*model.y[N, T]))
    model.x_derivative_constraint = pyo.Constraint(model.N, model.T, rule = x_derivative_rule)

    def I1_derivative_rule(model, T):
        if T==model.T.first():
            return model.I1[T] == 0
        else:
            return(model.dI1[T] == model.variableTime*model.D[T]*model.x[model.N.last(), T])
    model.I1_derivative_constraint = pyo.Constraint(model.T, rule = I1_derivative_rule)
    
    def I2_derivative_rule(model, T):
        if T == model.T.first():
            return model.I2[T] == 0
        else:
            return(model.dI2[T] == model.variableTime*model.D[T])
    model.I2_derivative_constraint = pyo.Constraint(model.T, rule = I2_derivative_rule)

    def I_V_derivative_rule(model, T):
        if T == model.T.first():
            return model.I_V[T] == 0
        else:
            return(model.dI_V[T] == model.variableTime*model.V[T])
    model.I_V_derivative_constraint = pyo.Constraint(model.T, rule = I_V_derivative_rule)   
    #Define discretization
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, nfe=60, ncp=3, scheme='LAGRANGE-RADAU')
    
    #Define model objective
    def objective_rule(model):
        # return(sum(model.E_sq[t] for t in model.T))
        # return -(0*model.xd_average[model.T.last()]+1*model.I1[model.T.last()])
        # return -model.I2[model.T.last()]/(model.HB[model.T.first()]+N*model.HT+model.HC)
        return model.I_V[model.T.last()]
        # return model.variableTime
    model.objective = pyo.Objective(rule = objective_rule)


    # LINKING CONSTRAINTS

    def xd_average_final_rule(model, T):
        if T == model.T.last():
            return(model.xd_average[T] >= model.xdset)
        else:
            return(pyo.Constraint.Skip)
    model.xd_average_final_constraint = pyo.Constraint(model.T, rule = xd_average_final_rule)

    def product_fraction_Rquirement_rule(model):
        return(model.I2[model.T.last()] == 0.8*(model.HB[model.T.first()]+N*model.HT+model.HC))
    model.product_fraction_Rquirement = pyo.Constraint(rule = product_fraction_Rquirement_rule)   

    # TEST CONSTRAINTS------
    # def rule_cons(model,T):
    #     if T > model.T.first():
    #         return model.R[T]/model.D[T] == model.R[model.T.prev(T)]/model.D[model.T.prev(T)]
    #     else:
    #         return pyo.Constraint.Skip

    # model.ConstantReflux=pyo.Constraint(model.T,rule=rule_cons) 

    # def rule_cons2(model,T):
    #     if T > model.T.first():
    #         return model.V[T] == model.V[model.T.prev(T)]
    #     else:
    #         return pyo.Constraint.Skip
    # model.ConstantBoilUp=pyo.Constraint(model.T,rule=rule_cons2)


    # CONSTANT CONTROL CONSTRAINTS

    keep_constant_R=5*3
    keep_constant_V=5*3

    def _Constant_controlR(m,N):
        if (N!=m.T.first() and (m.T.ord(N)-1)%keep_constant_R!=0) or (N==m.T.last()):
            return m.R[N] == m.R[m.T.prev(N)]
        else:
            return pyo.Constraint.Skip
    model.Constant_controlR=pyo.Constraint(model.T,rule=_Constant_controlR,doc='Constant control action every keep_constant_u discrete points and the last one')


    def _Constant_controlV(m,N):
        if (N!=m.T.first() and (m.T.ord(N)-1)%keep_constant_V!=0) or (N==m.T.last()):
            return m.V[N] == m.V[m.T.prev(N)]
        else:
            return pyo.Constraint.Skip
    model.Constant_controlV=pyo.Constraint(model.T,rule=_Constant_controlV,doc='Constant control action every keep_constant_temp discrete points and the last one')
   
    return(model)
# USER INPUTS
if __name__ == "__main__":
    N = 10
    tf = 20 #upper bound

    x0 = 0.8
    IE0 = 0

    a = 2.5
    V = 1000 #upper bound
    HT = 0.01
    HC = 0.1
    HB = 10   #upper bound #TODO: this should be left as variable
    CV = 2.309
    ZS = 0.5
    xdset = 0.95

    # CREATE AND SOLVE DISTILLATION MODEL
    distillation_model = create_distillation_model(N, tf, x0, a, V, HT, HC, HB, CV, ZS, xdset)

    solver = pyo.SolverFactory('gams', solver='conopt4')
    res = solver.solve(distillation_model, tee=True)

    # PLOT RESULTS
    xd_obtained = []
    for t in distillation_model.T:
        xd_obtained.append(pyo.value(distillation_model.x[distillation_model.N.last(), t]))

    D_obtained = []
    for t in distillation_model.T:
        D_obtained.append(pyo.value(distillation_model.D[t]))

    V_obtained = []
    for t in distillation_model.T:
        V_obtained.append(pyo.value(distillation_model.V[t]))

    R_obtained = []
    for t in distillation_model.T:
        R_obtained.append(pyo.value(distillation_model.R[t]))

    xd_average_obtained = []
    for t in distillation_model.T:
        xd_average_obtained.append(pyo.value(distillation_model.xd_average[t]))

    reboiler_hold_up=[]
    for t in distillation_model.T:
        reboiler_hold_up.append(pyo.value(distillation_model.HB[t]))


    t_vals = []
    for t in distillation_model.T:
        t_vals.append(t*pyo.value(distillation_model.variableTime))

    plt.plot(t_vals, xd_obtained)
    plt.show()
    plt.plot(t_vals, xd_average_obtained)
    plt.show()
    plt.plot(t_vals, D_obtained)
    plt.show()

    plt.plot(t_vals,  V_obtained)
    plt.show()

    plt.plot(t_vals,  R_obtained)
    plt.show()
    plt.plot(t_vals, reboiler_hold_up)
    plt.show()
