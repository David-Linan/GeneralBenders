import pyomo.environ as pe
import math
import matplotlib.pyplot as plt

def build_STN_cost_minimization(discrete_scheduling_horizon: int=120):
    '''
    Problem statement: cost minimization, with demand constraints at the end of the scheduling horizon
    '''
    m = pe.ConcreteModel(name='STN_scheduling_model')
    #---------------------SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=discrete_scheduling_horizon,doc='last discrete time value')
    #---------------------SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #-----------------------PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')

    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')

    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")

    def _demand(m,K,T):
        if K=='S8' and T==m.lastT:
            return 1400
        elif K=='S9' and T==m.lastT:
            return 1500
        else:
            return 0 
    m.demand=pe.Param(m.K,m.T,initialize=_demand,default=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #----------------------------VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,doc='1 if unit j processes task i starting at time t')   

    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')

    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')



    #----------------------------CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #---------------------------OBJECTIVE----------------
    # cost minimization
    def _obj(m):
        return sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.minimize)
    
    return m

def build_STN_profit_maximization(discrete_scheduling_horizon: int=120):
    '''
    Problem statement: Profit maximization, without demand constraints
    '''
    m = pe.ConcreteModel(name='STN_scheduling_model')
    #---------------------SCALARS---------------------
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid [units of time]')
    m.lastT=pe.Param(initialize=discrete_scheduling_horizon,doc='last discrete time value')
    #---------------------SETS------------------------
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['U1','U2','U3','U4'],doc='Set of Units')
    m.I=pe.Set(initialize=['T1','T2','T3','T4','T5'], doc='Set of tasks') 
    m.K=pe.Set(initialize=['S1','S2','S3','S4','S5','S6','S7','S8','S9'],doc='Set of states')

    #-----------------------PARAMETERS------------------
    m.eta=pe.Param(initialize=(m.T.__len__()-1)*m.delta, doc='scheduling horizon [units of time]')

    m.t_p=pe.Param(m.T,initialize=[m.delta*j for j in m.T],doc='physical time [units of time]')

    _I_i_k_minus={}
    _I_i_k_minus['T1','S1']=1

    _I_i_k_minus['T2','S3']=1
    _I_i_k_minus['T2','S2']=1

    _I_i_k_minus['T3','S4']=1
    _I_i_k_minus['T3','S5']=1

    _I_i_k_minus['T4','S6']=1
    _I_i_k_minus['T4','S3']=1

    _I_i_k_minus['T5','S7']=1
    m.I_i_k_minus=pe.Param(m.I,m.K,initialize=_I_i_k_minus,default=0,doc='State-task mapping: outputs from states')

    _I_i_k_plus={}
    _I_i_k_plus['T1','S4']=1

    _I_i_k_plus['T2','S5']=1

    _I_i_k_plus['T3','S6']=1
    _I_i_k_plus['T3','S8']=1

    _I_i_k_plus['T4','S7']=1

    _I_i_k_plus['T5','S6']=1
    _I_i_k_plus['T5','S9']=1

    m.I_i_k_plus=pe.Param(m.I,m.K,initialize=_I_i_k_plus,default=0,doc="Task-state mapping: inputs to states")


    _rho_minus={}
    _rho_minus['T1','S1']=1

    _rho_minus['T2','S3']=0.5
    _rho_minus['T2','S2']=0.5

    _rho_minus['T3','S4']=0.4
    _rho_minus['T3','S5']=0.6

    _rho_minus['T4','S6']=0.8
    _rho_minus['T4','S3']=0.2

    _rho_minus['T5','S7']=1  
    m.rho_minus=pe.Param(m.I,m.K,initialize=_rho_minus,default=0,doc="Fraction of material in state k consumed by task i ")


    _rho_plus={}
    _rho_plus['T1','S4']=1

    _rho_plus['T2','S5']=1

    _rho_plus['T3','S6']=0.6
    _rho_plus['T3','S8']=0.4

    _rho_plus['T4','S7']=1

    _rho_plus['T5','S6']=0.1
    _rho_plus['T5','S9']=0.9
    m.rho_plus=pe.Param(m.I,m.K,initialize=_rho_plus,default=0,doc="Fraction of material in state k produced by task i ")



    _I_i_j_prod={}
    _I_i_j_prod['T1','U1']=1

    _I_i_j_prod['T2','U2']=1
    _I_i_j_prod['T2','U3']=1

    _I_i_j_prod['T3','U2']=1
    _I_i_j_prod['T3','U3']=1

    _I_i_j_prod['T4','U2']=1
    _I_i_j_prod['T4','U3']=1

    _I_i_j_prod['T5','U4']=1

    m.I_i_j_prod=pe.Param(m.I,m.J,initialize=_I_i_j_prod,default=0,doc="Unit-task mapping (Definition of units that are allowed to perform a given task")

    _tau_p={}

    _tau_p['T1','U1']=0.5

    _tau_p['T2','U2']=0.5
    _tau_p['T2','U3']=1.5

    _tau_p['T3','U2']=1
    _tau_p['T3','U3']=2.5

    _tau_p['T4','U2']=1
    _tau_p['T4','U3']=5

    _tau_p['T5','U4']=1.5
    m.tau_p=pe.Param(m.I,m.J,initialize=_tau_p,default=0,doc="Physical processing time for tasks [units of time]")
    
    def _tau(m,I,J):
        return math.ceil(m.tau_p[I,J]/m.delta) 
    m.tau=pe.Param(m.I,m.J,initialize=_tau,default=0,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")

    _beta_min={}
    _beta_min['T1','U1']=10

    _beta_min['T2','U2']=10
    _beta_min['T2','U3']=10

    _beta_min['T3','U2']=10
    _beta_min['T3','U3']=10

    _beta_min['T4','U2']=10
    _beta_min['T4','U3']=10

    _beta_min['T5','U4']=10
    m.beta_min=pe.Param(m.I,m.J,initialize=_beta_min,default=0,doc="minimum capacity of unit j for task i")

    _beta_max={}
    _beta_max['T1','U1']=100

    _beta_max['T2','U2']=50
    _beta_max['T2','U3']=80

    _beta_max['T3','U2']=50
    _beta_max['T3','U3']=80

    _beta_max['T4','U2']=50
    _beta_max['T4','U3']=80

    _beta_max['T5','U4']=200
    m.beta_max=pe.Param(m.I,m.J,initialize=_beta_max,default=0,doc="maximum capacity of unit j for task i")


    m.gamma=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000,'S4':1000,'S5':150,'S6':500,'S7':1000,'S8':4000,'S9':4000},default=0,doc="maximum amount of material k that can be stored")
    m.demand=pe.Param(m.K,m.T,initialize=0,doc="demand of material k at time t")
    m.S0=pe.Param(m.K,initialize={'S1':4000,'S2':4000,'S3':4000},default=0,doc="Initial amount of state k")

    _cost={}
    _cost['T1','U1']=10

    _cost['T2','U2']=15
    _cost['T2','U3']=30

    _cost['T3','U2']=5
    _cost['T3','U3']=25

    _cost['T4','U2']=5
    _cost['T4','U3']=20

    _cost['T5','U4']=20
    m.cost=pe.Param(m.I,m.J,default=0,initialize=_cost,doc="cost to run task i in unit j")
    m.revenue=pe.Param(m.K,default=0,initialize={'S8':3,'S9':4},doc='revenue from selling one unit of material k')

    #----------------------------VARIABLES------------------ 
    m.X=pe.Var(m.I,m.J,m.T,within=pe.Binary,initialize=0,doc='1 if unit j processes task i starting at time t')   

    m.B=pe.Var(m.I,m.J,m.T,within=pe.NonNegativeReals,doc='Batch size of task i processed in unit j starting at time t')

    def _S_bounds(m,K,T):
        return (None,m.gamma[K])
    m.S=pe.Var(m.K,m.T,within=pe.NonNegativeReals,bounds=_S_bounds,doc='Inventory of material k at time t')


    #----------------------------CONSTRAINTS----------------
    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I if  m.I_i_j_prod[I,J]==1) <=  1
        
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _E2_CAPACITY_LOW(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.beta_min[I,J]*m.X[I,J,T]<=m.B[I,J,T]

    m.E2_CAPACITY_LOW=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_LOW,doc='UNIT CAPACITY LOWER BOUND')

    def _E2_CAPACITY_UP(m,I,J,T):
        if  m.I_i_j_prod[I,J]!=1:
            return pe.Constraint.Skip
        else:
            return m.B[I,J,T]<=m.beta_max[I,J]*m.X[I,J,T]

    m.E2_CAPACITY_UP=pe.Constraint(m.I,m.J,m.T,rule=_E2_CAPACITY_UP,doc='UNIT CAPACITY UPPER BOUND')

    def _E3_BALANCE(m,K,T):
        if T==0:
            return pe.Constraint.Skip
        else:
            return m.S[K,T]==m.S[K,T-1]+sum(m.rho_plus[I,K]*sum(m.B[I,J,T-m.tau[I,J]] for J in m.J if m.I_i_j_prod[I,J]==1 and T-m.tau[I,J]>=0) for I in m.I if m.I_i_k_plus[I,K]==1) - sum(m.rho_minus[I,K]*sum(m.B[I,J,T] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,T]    
    m.E3_BALANCE=pe.Constraint(m.K,m.T,rule=_E3_BALANCE,doc='MATERIAL BALANCES')

    def _E3_BALANCE_INIT(m,K):
        return m.S[K,0]==m.S0[K]-sum(m.rho_minus[I,K]*sum(m.B[I,J,0] for J in m.J if m.I_i_j_prod[I,J]==1) for I in m.I if m.I_i_k_minus[I,K]==1)-m.demand[K,0]
    m.E3_BALANCE_INIT=pe.Constraint(m.K,rule=_E3_BALANCE_INIT,doc='MATERIAL BALANCES INITIAL CONDITION')

    #---------------------------OBJECTIVE----------------

    #profit maximization
    def _obj(m):
        return sum(m.revenue[K]*m.S[K,m.lastT] for K in m.K)-sum(sum(sum(  m.cost[I,J]*m.X[I,J,T] for J in m.J)for I in m.I)for T in m.T)
    m.obj=pe.Objective(rule=_obj,sense=pe.maximize)      

    return m



if __name__ == "__main__":
    #------------------SOLUTION OF THE PROBLEM-------------------
    # Model initialization
    discrete_scheduling_horizon=5 #you can change this to make the problem larger, e.g., try 120 and it will take about 50 sec to solve with 0.05 gap
    # m=build_STN_cost_minimization(discrete_scheduling_horizon=discrete_scheduling_horizon)  
    m=build_STN_profit_maximization(discrete_scheduling_horizon=discrete_scheduling_horizon)# NOTE: this one is better, because for cost minimization there may be infeasibilities if scheduling horizon is not long enough.
    # Solution
    gap=0.05 # GLPK optimality gap
    solver = pe.SolverFactory('glpk')
    solver.options['mipgap']=gap
    solver.solve(m, tee=True)

    # #--------------------GANTT PLOT-----------------------------------

    fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 52) 
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Time [h]')
    gnt.set_ylabel('Units')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45]) 
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) 
    
    # Setting graph attribute
    gnt.grid(False)
    
    # Declaring bars in schedule
    height=9
    already_used=[]
    for j in m.J:

        if j=='U1':
            lower_y_position=40    
        elif j=='U2':
            lower_y_position=30    
        elif j=='U3':
            lower_y_position=20
        elif j=='U4':
            lower_y_position=10
        for i in m.I:
            if i=='T1':
                bar_color='tab:red'
            elif i=='T2':
                bar_color='tab:green'    
            elif i=='T3':
                bar_color='tab:blue'    
            elif i=='T4':
                bar_color='tab:orange' 
            elif i=='T5':
                bar_color='tab:olive'
            for t in m.T:

                if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kk_index] for kk_index in range(len(already_used))):
                    gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)

                    already_used.append(i)
                elif pe.value(m.X[i,j,t])==1:
                    gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                    

    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    plt.show()


    fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 52) 
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Time [h]')
    gnt.set_ylabel('Units')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45]) 
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) 
    
    # Setting graph attribute
    gnt.grid(False)
    
    # Declaring bars in schedule
    height=9
    already_used=[]
    for j in m.J:

        if j=='U1':
            lower_y_position=40    
        elif j=='U2':
            lower_y_position=30    
        elif j=='U3':
            lower_y_position=20
        elif j=='U4':
            lower_y_position=10
        for i in m.I:
            if i=='T1':
                bar_color='tab:red'
            elif i=='T2':
                bar_color='tab:green'    
            elif i=='T3':
                bar_color='tab:blue'    
            elif i=='T4':
                bar_color='tab:orange' 
            elif i=='T5':
                bar_color='tab:olive'
            for t in m.T:

                if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kk_index] for kk_index in range(len(already_used))):
                    gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                    already_used.append(i)
                elif pe.value(m.X[i,j,t])==1:
                    gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                    gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        

    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    plt.show()
