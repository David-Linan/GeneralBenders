import pyomo.environ as pe
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from pyomo.gdp import Disjunct, Disjunction
import logging

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/')
from functions.dsda_functions import get_external_information, solve_with_minlp,solve_with_dsda,external_ref_neighborhood_general,solve_with_gdpopt

def problem_logic(m):
    logic_expr = []
    for index in m.ordered:
        for i in m.i:
            logic_expr.append([m.YR[index,i],m.YR_Disjunct[index,i].indicator_var])              

    return logic_expr
def planning_model(reformulation: bool=False):
    m=pe.ConcreteModel(name='Production planning by mixed integer programming, Section 1.2')

    # SETS

    m.NI=pe.Param(initialize=12,doc='Number of products')
    m.NK=pe.Param(initialize=3,doc='Number of machines')
    m.NT=pe.Param(initialize=15,doc='Number of time periods')

    m.i=pe.RangeSet(1,m.NI,1,doc='Set of finished products')
    m.t=pe.RangeSet(1,m.NT,1,doc='Set of time periods')
    m.k=pe.Set(initialize=[1,2,3], doc='Machines. 1: mixing, 2: Cereal packaging, 3: Fruit packaging')
    m.F2=pe.Set(initialize=[i for i in m.i if i<=6],within=m.i)
    m.F3=pe.Set(initialize=[i for i in m.i if i>6],within=m.i)

    # PARAMETERS
    
    D_data=np.array([[0, 95, 110, 96, 86,124, 83,108, 114,121, 110,124, 104, 86, 87],
     [98, 96, 96, 98, 103,104, 122,101, 89,108, 101,109, 106,108, 76],
     [106, 0, 89,123, 96,105, 83, 82, 112,109, 119, 85, 99, 80, 123],
     [98,121, 0,105, 98, 96, 101, 81, 117, 76, 103, 81, 95,105, 102],
     [0,124, 113,123, 123, 79, 111, 98, 97, 80, 98,124, 78,108, 109],
     [103,102, 0, 95, 107,105, 107,105, 75, 93, 115,113, 111,105, 85],
     [110, 93, 0,112, 84,124, 98,101, 83, 87, 105,118, 115,106, 78],
     [85, 92, 101,110, 93, 96, 120,109, 121, 87, 92, 85, 91, 93, 109],
     [122,116, 109, 0, 105,108, 88, 98, 77, 90, 110,102, 107, 99, 96],
     [120,124, 94,105, 92, 86, 101,106, 75,109, 83, 95, 79,108, 100],
     [117, 96, 78, 0, 108, 87, 114,107, 110, 94, 104,101, 108,110, 80],
     [125,112, 75, 0, 116,103, 122, 88, 85, 84, 76,102, 84, 88, 82]])
    def _D(m,i,t):
        return D_data[i-1,t-1]
    m.D=pe.Param(m.i,m.t,initialize=_D,doc='Forecast demand for item i in period t')


    def _SS(m,i,t):
        if i in m.F2:
            return 10
        else:
            return 20
    m.SS=pe.Param(m.i,m.t,initialize=_SS,doc='Safety stock of item i needed at the end of period t')
    _SS0={}
    _SS0[1]=83
    _SS0[2]=31
    _SS0[3]=11
    _SS0[4]=93
    _SS0[5]=82
    _SS0[6]=72
    _SS0[7]=23
    _SS0[8]=91
    _SS0[9]=83
    _SS0[10]=34
    _SS0[11]=61
    _SS0[12]=82
    m.SS0=pe.Param(m.i,initialize=_SS0,doc='Initial stock of item i')

    m.alpha=pe.Param(m.i,m.k, initialize=1,doc='capacity consumed on machine k to produce one unit of product i')
    _beta={}
    _beta[1]=30
    _beta[2]=20
    _beta[3]=30
    _beta[4]=40
    _beta[5]=40
    _beta[6]=10
    _beta[7]=30
    _beta[8]=20
    _beta[9]=10
    _beta[10]=50
    _beta[11]=30
    _beta[12]=20
    m.beta=pe.Param(m.i,initialize=_beta,doc='Mixing capacity consumed per cleaning operation at the end of a batch of product i')

    m.L=pe.Param(m.k,initialize={1:1400,2:700,3:700},doc='Capacity available on machine k in each time period')
    def _M(m,i,t):
        if i in m.F2:
            return min([sum(m.D[i,l] for l in m.t if l>=t)+m.SS[i,t],(m.L[1]-m.beta[i])/(m.alpha[i,1]),m.L[2]/m.alpha[i,2]])
        else:
            return min([sum(m.D[i,l] for l in m.t if l>=t)+m.SS[i,t],(m.L[1]-m.beta[i])/(m.alpha[i,1]),m.L[3]/m.alpha[i,3]])
    m.M=pe.Param(m.i,m.t,initialize=_M,doc='Big-M')

    # VARIABLES

    m.x=pe.Var(m.i,m.t,within=pe.NonNegativeReals,initialize=0,doc='amount of product i produced during time period t')
    m.s=pe.Var(m.i,m.t,within=pe.NonNegativeReals,initialize=0,doc='inventory level of product i at the end of time period t')
    m.y=pe.Var(m.i,m.t,within=pe.Binary,initialize=0,doc='1 if there is cleaning operation because production of a batch of product i')

    # CONSTRAINTS
    def _dem_sat1(m,i,t):
        if t!=m.t.first():
            return m.s[i,t]>=m.SS[i,t]
        else:
            return pe.Constraint.Skip
    m.dem_sat1=pe.Constraint(m.i,m.t,rule=_dem_sat1)

    def _dem_sat2(m,i,t):
        if t==m.t.first():
            return m.SS0[i]+m.x[i,t]==m.D[i,t]+m.s[i,t] 
        else:
            return m.s[i,m.t.prev(t)]+m.x[i,t]==m.D[i,t]+m.s[i,t] 
    m.dem_sat2=pe.Constraint(m.i,m.t,rule=_dem_sat2)

    def _vub(m,i,t):
        return m.x[i,t]<=m.M[i,t]*m.y[i,t] 
    m.vub=pe.Constraint(m.i,m.t,rule=_vub)

    def _mix_cap(m,t):
        return sum(m.alpha[i,1]*m.x[i,t]  for i in m.i)+sum( m.beta[i]*m.y[i,t]    for i in m.i)<=m.L[1]
    m.mix_cap=pe.Constraint(m.t,rule=_mix_cap)
    def _pack_cap(m,t,k):
        if k==2:
            return sum(  m.alpha[i,k]*m.x[i,t] for i in m.F2)<=m.L[k]
        elif k==3:
            return sum(  m.alpha[i,k]*m.x[i,t] for i in m.F3)<=m.L[k]
        else:
            return pe.Constraint.Skip 
    m.pack_cap=pe.Constraint(m.t,m.k,rule=_pack_cap)


    def _obj(m):
        return sum(   sum(     m.s[i,t]       for t in m.t)         for i in m.i)
    m.obj=pe.Objective(sense=pe.minimize,rule=_obj)


    # REFORMULATION
    if reformulation:
        # REFORMULATION VARIABLES
        m.tp=pe.RangeSet(1,m.NT,1,doc='Set of time periods (clone)')
        m.N=pe.Var(m.i,within=pe.Integers,bounds=(0,m.NT))
        m.P=pe.Var(m.i,m.tp,within=pe.Integers,bounds=(0,m.NT))
        m.yp=pe.Var(m.i,m.t,m.tp,within=pe.Binary)
        # LINKING CONSTRAINTS
        def _DefN(m,i):
            return m.N[i]==sum(m.y[i,t] for t in m.t)
        m.def_N=pe.Constraint(m.i,rule=_DefN)

        def _DefP(m,i,tprime):
            return m.P[i,tprime]==sum(t*m.yp[i,t,tprime] for t in m.t)
        m.def_P=pe.Constraint(m.i,m.tp,rule=_DefP)

        def _Defyp(m,i,t):
            return m.y[i,t]==sum(  m.yp[i,t,tp]     for tp in m.tp)
        m.def_yp=pe.Constraint(m.i,m.t,rule=_Defyp)

        # DISJUNCTIONS
        m.ordered=pe.RangeSet(0,m.NT)
        m.YR=pe.BooleanVar(m.ordered,m.i,initialize=False)

        def _select_one(m,i):
            return pe.exactly(1,[m.YR[index,i] for index in m.ordered])
        m.oneYR=pe.LogicalConstraint(m.i,rule=_select_one) 
        

        def _build_YR_Disjunct(m,index,i):
            def _DEF_Nref(m):
                return m.model().N[i]==index
            m.DEF_Nref=pe.Constraint(rule=_DEF_Nref)
            if index==0:
                def _C1_P(m,tprime):
                    return m.model().P[i,tprime]==0
                m.C_1P=pe.Constraint(m.model().t,rule=_C1_P)

            elif index==1:
                def _C1_P(m,tprime):
                    if tprime>=index+1:
                        return m.model().P[i,tprime]==0
                    elif tprime==1:
                        return m.model().P[i,tprime]>=1
                m.C_1P=pe.Constraint(m.model().t,rule=_C1_P)
            else:
                def _C1_P(m,tprime):
                    if tprime>=index+1:
                        return m.model().P[i,tprime]==0
                    elif tprime==1:
                        return m.model().P[i,tprime]>=1
                    else:
                        return m.model().P[i,tprime]>=m.model().P[i,m.model().t.prev(tprime)]+1
                m.C_1P=pe.Constraint(m.model().t,rule=_C1_P)

        m.YR_Disjunct=Disjunct(m.ordered,m.i,rule=_build_YR_Disjunct)


        # Create disjunction
        def _Disjunction(m,i):   
            return [m.YR_Disjunct[index,i] for index in m.ordered]
        m.Disjunction=Disjunction(m.i,rule=_Disjunction,xor=True)


        # Associate disjuncts with boolean variables
        for index in m.ordered:
            for i in m.i:
                m.YR[index,i].associate_binary_var(m.YR_Disjunct[index,i].indicator_var)


    return m

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    mip_solver='cplex'
    sub_options={'add_options':['option optcr=0;','option threads=0;','GAMS_MODEL.optfile = 1;','$onecho > cplex.opt', 'intsollim large', '$offecho']}

    m=planning_model(reformulation=True)
    # opt1 = pe.SolverFactory('gams')
    # results = opt1.solve(m, solver=mip_solver, add_options = sub_options['add_options'] , tee=True)

    m =solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0) 




    # ext_Var_init=[]
    # for i in m.i:
    #     ext_Var_init.append(int(sum(pe.value(m.y[i,t]) for t in m.t)+1))

    # # REFORMULATION

    # m=planning_model(reformulation=True)
    # ext_ref={m.YR:m.ordered}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)


    # #SOLUTION
    # sub_options={'add_options':['option threads=0;','GAMS_MODEL.optfile = 1;','$onecho > cplex.opt', 'intsollim large', '$offecho']}

    # model_fun=planning_model
    # logic_fun=problem_logic
    # kwargs={'reformulation':True}
    # current_central=ext_Var_init #initialization of external variables
    # original_m=model_fun(**kwargs) # declaration of original GDP model
    # max_iter_out=1000000 # Maximum number of iterations
    # upper_evaluated={} #evaluated points
    # old_obj=1e+10
    # teed=True


    # start=time.time()
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,ext_Var_init,ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = mip_solver,subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = True,rel_tol = 0.02,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=False)
    # end=time.time()
    # print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  


    # # CG_DSDA algorithm
    # start=time.time()
    # for out in range(max_iter_out):
    #     if teed:
    #         print('------Outer iteration= ',out+1)
    #     upper_evaluated[out+1]=[current_central] #Update evaluated points with current external variable being evaluated
        
    #     # perform the reformulation
    #     for i in original_m.i:
    #         for ind in original_m.ordered:
    #             disjunct=original_m.YR_Disjunct[ind,i]
    #             disjunct.activate()
    #     original_m = external_ref_neighborhood_general(m=original_m,x=current_central,extra_logic_function=logic_fun,dict_extvar=reformulation_dict)
    #     # TODO: had to do this outside reformulation function. Generalize!!
    #     for i in original_m.i:
    #         for ind in original_m.ordered:
    #             disjunct=original_m.YR_Disjunct[ind,i]
    #             if original_m.YR[ind,i].is_fixed() and original_m.YR[ind,i].value==False:
    #                 disjunct.deactivate()
    #                 # disjunct.pprint()
    #                 # print(disjunct.name,' deactivated') 

    #     # solve the problem
    #     m=original_m.clone() #initialize mip problem
    #     # m=solve_with_gdpopt(m,mip=mip_solver,minlp='cplex',mip_options=sub_options,timelimit=86400,rel_tol=0,strategy='LBB',tee=True)
    #     m =solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0.03) 
        
    #     # Extract solution
    #     Sol_found=[]

    #     # TODO: Generalize this part
    #     for i in original_m.i:
    #         for ind in original_m.ordered:
    #             if(round(pe.value(m.YR_Disjunct[ind,i].binary_indicator_var))==1):
    #                 Sol_found.append(ind+1) #NOTE, because my set starts at 0, but ext vars at 1
    #                 break
    #     if teed:
    #         print(Sol_found)

    #     # for v in m.component_data_objects(ctype=pe.Var):
    #     #     if v.parent_component().name!='X' and v.is_binary()==True:
    #     #         print(v.parent_component().name,'=',pe.value(v))
    #     #         # v.pprint()
    #     #         # for vv in v.items():
    #     #         #     vv.pprint()
    

    #     # Generate search direction
    #     direction=[]
    #     for i in range(len(Sol_found)):
    #         direction.append(Sol_found[i]-current_central[i])              

    #     if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
    #         m.mip_status='Infeasible'
    #         #TODO: IN CASE OF INFEASIBILITIES, I should declare infinity objective, however, it is not possible because new neighborhood always have the previous feasible solution in it.
    #     else:
    #         m.mip_status='Optimal'
    #     if teed:
    #         if m.mip_status == 'Optimal':  

    #             print('   Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                            
    #         else:
    #             print('   Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))        


    #         print('   SEARCH DIRECTION: ', direction)                
        
    #     if round(sum(abs(j) for j in direction))==0 or old_obj<=round(pe.value(m.obj)):    
    #         end=time.time()
    #         print('Objective CG-DSDA='+str(pe.value(m.obj))+', best D-SDSA='+str(Sol_found),'cputime LG-DSDA= '+str(end-start))  
    #         break
    #     else:
    #         # update objective and variables
    #         old_obj=round(pe.value(m.obj))
    #         current_central=Sol_found
    #         # delete most recently solved mip
    #         del m
    #         #update

