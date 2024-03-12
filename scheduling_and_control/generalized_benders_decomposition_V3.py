from __future__ import division
import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #NOTE: CHANGE AS NEEDED
from functions.dsda_functions import solve_subproblem,generate_initialization,initialize_model,solve_with_minlp
import pyomo.environ as pe
import time
from functions.dsda_functions import solve_subproblem,generate_initialization,initialize_model
import logging
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_GBD

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


    print('\n-------GENERALIZED BENDERS DECOMPOSITION TEST-------------------------------------')
    ######-------------------SOLVER AND MODEL DECLARATION -------------------------##################
    mip_solver='CPLEX'
    nlp_solver='conopt4'
    transform='bigm'
    sub_options={'add_options':['GAMS_MODEL.threads=0;']} #Subproblem solver options: use all available threads
    Infinity_aprox=100000
    kwargs={}
    kwargs2=kwargs.copy()
    model_fun=scheduling_and_control_gdp_N_GBD

    #####------------------COMPUTATIONAL EXPERIMENTS-------------------------######################


   # 1: Naive execution of GBD
   # 2: Execution of GBD with heuristic cuts: minimum processing time s.t. variable capacity
   # 3: Execution of GBD with heuristic cuts: minimum processing time s.t. fixed capacity at its maximum
   # 4: Execution of GBD with heuristic cuts: no-good cuts in master problem
   # 5: Execution of GBD with heuristic cuts: no-good cuts in master problem, only for binary variables related to variable processing times
    experiment=6
 
    if experiment ==1:
        best_sol_name='current_best_GBD_V3_subproblem_naive'
    elif experiment ==2:
        best_sol_name='current_best_GBD_V3_subproblem_min_t_vary_B'
    elif experiment ==3:
        best_sol_name='current_best_GBD_V3_subproblem_min_t_fix_B'
    elif experiment ==4:
        best_sol_name='current_best_GBD_V3_subproblem_no_good_all'
    elif experiment ==5:
        best_sol_name='current_best_GBD_V3_subproblem_no_good_t_only'
    elif experiment ==6:
        best_sol_name='current_best_GBD_V3_naive_from_sequential_naive'
    elif experiment ==7:
        best_sol_name='current_best_GBD_V3_naive_from_sequential_naive_min_t_vary_B'
 
    ######-------------------------- MATER PROBLEM ------------------------------##################
    mas=model_fun(**kwargs2) #master problem
    mas.cuts=pe.ConstraintList() #Initialize benders cuts

    #Deactivate subproblem constraint
    for I in mas.I_reactions:
        for J in mas.J_reactors:
            mas.c_dCdtheta[I,J].deactivate()
            mas.c_dTRdtheta[I,J].deactivate()                        
            mas.c_dTJdtheta[I,J].deactivate()
            mas.c_dIntegral_hotdtheta[I,J].deactivate()
            mas.c_dIntegral_colddtheta[I,J].deactivate()
            mas.Constant_control1[I,J].deactivate()                        
            mas.Constant_control2[I,J].deactivate()
            mas.finalCon[I,J].deactivate()
            mas.finalTemp[I,J].deactivate()
    mas.C_TCP3.deactivate()

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(mas)
    transformation_string = 'gdp.' + transform
    pe.TransformationFactory(transformation_string).apply_to(mas)

    ######---------------------------- SUBPROBLEM --------------------------------##################
    sub=model_fun(**kwargs2) #subproblem

    # deactivate master problem constraints
    sub.E2_CAPACITY_LOW.deactivate()
    sub.E2_CAPACITY_UP.deactivate()
    sub.E3_BALANCE_INIT.deactivate()
    sub.E_DEMAND_SATISFACTION.deactivate()
    sub.E1_UNIT.deactivate()
    sub.DEF_AUX1_INDEP.deactivate()
    sub.E3_BALANCE.deactivate()
    sub.DEF_AUX2_INDEP.deactivate()
    for I in sub.I_reactions:
        for J in sub.J_reactors:
            for disj in sub.ordered_set[I,J]:
                disjunct=sub.YR_disjunct[I,J][disj]
                for constr in disjunct.component_objects(pe.Constraint, descend_into=True):
                    if constr.local_name=='DEF_VAR_TIME' or constr.local_name=='DEF_AUX1' or constr.local_name=='DEF_AUX2': #NOTE that I am deactivating everythin. I am just using this to show that I can compare by name to deactivate what I want
                        constr.deactivate()
            sub.YR_disjunct[I,J].deactivate()
            sub.Disjunction1[I,J].deactivate()
            sub.oneYR[I,J].deactivate()
    sub.X_Z_relation.deactivate()
    sub.obj.deactivate()

    # re-define objective function
    sub.obj.deactivate()
    def _obj_dynamic(m):
        return m.TCP3
    sub.obj_dyn = pe.Objective(rule=_obj_dynamic, sense=pe.minimize) 

    #linking variables and constraints in subproblem:
    sub.link_B=pe.Var(sub.I,sub.J,sub.T,within=pe.NonNegativeReals)
    sub.link_varTime=pe.Var(sub.I_reactions,sub.J_reactors,within=pe.NonNegativeReals)
    sub.link_X=pe.Var(sub.I,sub.J,sub.T,within=pe.NonNegativeReals)

    def _const_link_B(sub,I,J,T):
        return sub.B[I,J,T]-sub.link_B[I,J,T]==0
    sub.const_link_B=pe.Constraint(sub.I,sub.J,sub.T,rule=_const_link_B)

    def _const_link_VarTime(sub,I,J):
        return sub.varTime[I,J]-sub.link_varTime[I,J]==0
    sub.const_link_VarTime=pe.Constraint(sub.I_reactions,sub.J_reactors,rule=_const_link_VarTime)

    # We have linking constraints fixed, so we can change the domain of variables X to continuous in subproblems
    sub.X.domain=pe.NonNegativeReals
    def _const_link_X(sub,I,J,T):
        return sub.X[I,J,T]-sub.link_X[I,J,T]==0
    sub.const_link_X=pe.Constraint(sub.I,sub.J,sub.T,rule=_const_link_X)

    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(sub)
    transformation_string = 'gdp.' + transform
    pe.TransformationFactory(transformation_string).apply_to(sub)

    # Dual variables (Lagrange multipliers)
    sub.dual = pe.Suffix(direction=pe.Suffix.IMPORT) #define dual variables

    #####------------------Heuristic cuts for experiments 2 and 3-------------------------######################
    if experiment ==2 or experiment ==3 or experiment ==7:
    # TEST MINIMUM PROCESSING TIME (add minimum processing time constraint to master)
        mint=sub.clone()
        mint.obj_dyn.deactivate()
        def _obj_t(m):
            return sum(sum( mint.varTime[I,J] for I in mint.I_reactions) for J in mint.J_reactors)
        mint.obj_t = pe.Objective(rule=_obj_t, sense=pe.minimize)   


        for I in mint.I:
            for J in mint.J:
                for T in mint.T:
                    if experiment ==3:
                        mint.B[I,J,T].fix(mas.B[I,J,T].ub) # NOTE: that the maximum capacity for this case study agrees with upper bound
                    mint.X[I,J,T].fix(1)

        for I_J in mint.I_J:
                I=I_J[0]
                J=I_J[1]
                mint.Nref[I,J].fix(1)  
        mint=solve_subproblem(mint,subproblem_solver='conopt4',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)   

        def _const_min_VarTime(mas,I,J):
            return mas.varTime[I,J]>=pe.value(mint.varTime[I,J])
        mas.const_min_VarTime=pe.Constraint(mas.I_reactions,mas.J_reactors,rule=_const_min_VarTime)
        print('\n minimum variable processing times')
        mas.const_min_VarTime.pprint()

    ######---------------------------- FEASIBILITY SUBPROBLEM --------------------------------##################
    feas=sub.clone()
    feas.elastic_vars={}

    count_elastic=0
    for constr in feas.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
        if constr.parent_component().name != 'const_link_B' and constr.parent_component().name != 'const_link_VarTime' and constr.parent_component().name != 'const_link_X': 
            if constr.equality:
                count_elastic=count_elastic+1
                feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                constr._body+=feas.elastic_vars[count_elastic]

                count_elastic=count_elastic+1
                feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                constr._body+=-feas.elastic_vars[count_elastic]
            else:
                count_elastic=count_elastic+1
                if constr.has_lb():
                    feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                    setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                    constr._body+=feas.elastic_vars[count_elastic]
                if constr.has_ub():
                    feas.elastic_vars[count_elastic]=pe.Var(within=pe.NonNegativeReals,initialize=0)
                    setattr(feas,'elastic_vars_%s' %str(count_elastic),feas.elastic_vars[count_elastic])
                    constr._body+=-feas.elastic_vars[count_elastic]

    feas.obj_dyn.deactivate()
    def _obj_feas(m):
        return sum( feas.elastic_vars[i]  for i in feas.elastic_vars.keys())
    feas.obj_feas = pe.Objective(rule=_obj_feas, sense=pe.minimize)     

    ######---------------------------- SOLUTION OF FIRST SUBPROBLEM --------------------------------##################
    if experiment ==6 or experiment ==7:
        mas=initialize_model(mas,from_feasible=True,feasible_model='case_1_scheduling_and_dynamics_solution') #Initialization from solution that is known to be feasible
        for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
            if v.name=='varTime' or v.name=='B':
                for index in v:
                    if index==None:
                        v.fix(pe.value(v))
                    else:
                        v[index].fix(pe.value(v[index]))
            elif v.name=='X':
                for index in v:
                    if index==None:
                        v.fix(round(pe.value(v)))
                    else:
                        v[index].fix(round(pe.value(v[index])))

        mas=solve_with_minlp(mas,transformation='',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
        print(pe.value(mas.obj))

        # fix master variables in subproblem 
        for I in sub.I:
            for J in sub.J:
                for T in sub.T:
                    sub.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))
                    sub.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

        for I_J in sub.I_J:
                I=I_J[0]
                J=I_J[1]
                sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

        for K in sub.K:
            for T in sub.T:
                sub.S[K,T].fix(pe.value(mas.S[K,T]))

        for I in sub.I_reactions:
            for J in sub.J_reactors:
                sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  


        sub=solve_subproblem(sub,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
    
        
        if sub.dsda_status=='Optimal':
            mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  sub.dual[sub.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(sub.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum(sum(sum(  sub.dual[sub.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(sub.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
            print('Optimality cut to initialize: ')
            mas.cuts.pprint()
            
            TPC1=pe.value(sub.TCP1)
            TPC2=pe.value(sub.TCP2)
            TPC3=pe.value(sub.TCP3)
            TMC=pe.value(sub.TMC)
            SALES=pe.value(sub.SALES)
            sub.UBD=TPC1+TPC2+TPC3+TMC-SALES
            print(sub.UBD)
            generate_initialization(m=sub,model_name=best_sol_name)
        else:
            sub.UBD=Infinity_aprox
            print('Initialization is not feasible')
            exit()

        # make sure that variables in master are left unfixed
        for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
            if v.name=='varTime' or v.name=='B' or v.name=='X':
                for index in v:
                    if index==None:
                        v.unfix()
                    else:
                        v[index].unfix()
    else:
        mas=initialize_model(mas,from_feasible=True,feasible_model='case_1_scheduling_and_dynamics_solution_GDB_init') #Initialization from solution that is known to be feasible
        for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
            if v.name=='varTime' or v.name=='B':
                for index in v:
                    if index==None:
                        v.fix(pe.value(v))
                    else:
                        v[index].fix(pe.value(v[index]))
            elif v.name=='X':
                for index in v:
                    if index==None:
                        v.fix(round(pe.value(v)))
                    else:
                        v[index].fix(round(pe.value(v[index])))

        mas=solve_with_minlp(mas,transformation='',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
        print(pe.value(mas.obj))

        # fix master variables in subproblem 
        for I in sub.I:
            for J in sub.J:
                for T in sub.T:
                    sub.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))
                    sub.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

        for I_J in sub.I_J:
                I=I_J[0]
                J=I_J[1]
                sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

        for K in sub.K:
            for T in sub.T:
                sub.S[K,T].fix(pe.value(mas.S[K,T]))

        for I in sub.I_reactions:
            for J in sub.J_reactors:
                sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  


        sub=solve_subproblem(sub,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)
    
        
        if sub.dsda_status=='Optimal':
            mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  sub.dual[sub.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(sub.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum(sum(sum(  sub.dual[sub.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(sub.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
            print('Optimality cut to initialize: ')
            mas.cuts.pprint()
            
            TPC1=pe.value(sub.TCP1)
            TPC2=pe.value(sub.TCP2)
            TPC3=pe.value(sub.TCP3)
            TMC=pe.value(sub.TMC)
            SALES=pe.value(sub.SALES)
            sub.UBD=TPC1+TPC2+TPC3+TMC-SALES
            print(sub.UBD)
        else:
            sub.UBD=Infinity_aprox
            print('Initialization is ot feasible')
            exit()

        # make sure that variables in master are left unfixed
        for v in mas.component_objects(pe.Var, descend_into=True): #test: master problem initialized with fixed linking variabels that guaranteee feasibility (NOTE: this is jsut a test to see what happens!!!)
            if v.name=='varTime' or v.name=='B' or v.name=='X':
                for index in v:
                    if index==None:
                        v.unfix()
                    else:
                        v[index].unfix()

    ######---------------------------- GENERALIZED BENDERS DECOMPOSIION ALGORITHM --------------------------------##################

    start=time.time()
    max_iter=100000
    epsilon=0
    time_limit=86400 #seconds
    for k in range(max_iter):
        print('------------------------Iteration ',str(k),'------------------------------------')
    #1: solve the master problem
        mas=solve_with_minlp(mas,transformation='',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0,transform_required=False)
        mas.LBD=pe.value(mas.obj)

        if experiment == 4:
        #1.1.: add no-good cuts (cuts that avoid repeating combination of binary variables)
            expr=0
            for v in mas.component_data_objects(ctype=pe.Var,descend_into=True,active=True):
                if v.is_binary() and int(round(pe.value(v)))==int(1):
                    expr+=v-1
                elif v.is_binary() and int(round(pe.value(v)))==int(0):
                    expr+=-v          
            
            mas.cuts.add( expr<= -1)
        elif experiment ==5:
            #1.1.: add no-good cuts (cuts that avoid repeating combination of binary variables related to variable processing times)
            expr=0
            for v in mas.component_data_objects(ctype=pe.Var,descend_into=True,active=True):
                if v.is_binary() and int(round(pe.value(v)))==int(1) and v.parent_component().name !='X':
                    expr+=v-1
                elif v.is_binary() and int(round(pe.value(v)))==int(0) and v.parent_component().name !='X':
                    expr+=-v          
            
            mas.cuts.add( expr<= -1)

    #2: verify stopping criterion
        current=time.time()
        print('Primal obj: ',str(sub.UBD),'Master obj:', str(mas.LBD),'Current time: ',str(current-start))
        if sub.UBD- mas.LBD <=epsilon:
            break
        if current-start>=time_limit:
            break

    #3: Solve primal problem
        # fix variables in subproblem 
        for I in sub.I:
            for J in sub.J:
                for T in sub.T:

                    if pe.value(mas.B[I,J,T])>=mas.B[I,J,T].ub: 
                        sub.link_B[I,J,T].fix(mas.B[I,J,T].ub)
                    elif pe.value(mas.B[I,J,T])<=mas.B[I,J,T].lb:
                        sub.link_B[I,J,T].fix(mas.B[I,J,T].lb)
                    else:
                        sub.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))

                    sub.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

        for I_J in sub.I_J:
                I=I_J[0]
                J=I_J[1]
                sub.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

        for K in sub.K:
            for T in sub.T:
                if pe.value(mas.S[K,T])>=mas.S[K,T].ub:
                    sub.S[K,T].fix(mas.S[K,T].ub)
                elif pe.value(mas.S[K,T])<=mas.S[K,T].lb:
                    sub.S[K,T].fix(mas.S[K,T].lb)
                else:
                    sub.S[K,T].fix(pe.value(mas.S[K,T]))

        for I in sub.I_reactions:
            for J in sub.J_reactors:
                if pe.value(mas.varTime[I,J])>=mas.varTime[I,J].ub:
                    sub.link_varTime[I,J].fix(mas.varTime[I,J].ub)  
                elif pe.value(mas.varTime[I,J])<=mas.varTime[I,J].lb:
                    sub.link_varTime[I,J].fix(mas.varTime[I,J].lb)  
                else:
                    sub.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))   

        # solve primal (subprolem)
        sub=solve_subproblem(sub,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
        generate_initialization(m=sub,model_name='GBD_subproblem') #save solution, in case I need an alternative initialization for the feasibility subproblem
        print('Subproblem status:',sub.dsda_status)

        if sub.dsda_status=='Optimal':
              
            TPC1=pe.value(sub.TCP1)
            TPC2=pe.value(sub.TCP2)
            TPC3=pe.value(sub.TCP3)
            TMC=pe.value(sub.TMC)
            SALES=pe.value(sub.SALES)

            sub.UBD_new=min([sub.UBD,TPC1+TPC2+TPC3+TMC-SALES])
            # Update the best known solution if it improved
            if sub.UBD_new<sub.UBD:
                generate_initialization(m=sub,model_name=best_sol_name)
            # Update subproblem solution with the best subproblem solution identified so far  
            sub.UBD=sub.UBD_new


            mas.cuts.add(sum(sum( sub.dual[sub.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(sub.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  sub.dual[sub.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(sub.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum(sum(sum(  sub.dual[sub.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(sub.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+pe.value(sub.TCP3)<=mas.TCP3)
            print('Optimaliti cut added')
            
        else:

            # fix variables in subproblem 
            for I in feas.I:
                for J in feas.J:
                    for T in feas.T:

                        if pe.value(mas.B[I,J,T])>=mas.B[I,J,T].ub: 
                            feas.link_B[I,J,T].fix(mas.B[I,J,T].ub)
                        elif pe.value(mas.B[I,J,T])<=mas.B[I,J,T].lb:
                            feas.link_B[I,J,T].fix(mas.B[I,J,T].lb)
                        else:
                            feas.link_B[I,J,T].fix(pe.value(mas.B[I,J,T]))

                        feas.link_X[I,J,T].fix(round(pe.value(mas.X[I,J,T])))

            for I_J in feas.I_J:
                    I=I_J[0]
                    J=I_J[1]
                    feas.Nref[I,J].fix(round(pe.value(mas.Nref[I,J])))

            for K in feas.K:
                for T in feas.T:
                    if pe.value(mas.S[K,T])>=mas.S[K,T].ub:
                        feas.S[K,T].fix(mas.S[K,T].ub)
                    elif pe.value(mas.S[K,T])<=mas.S[K,T].lb:
                        feas.S[K,T].fix(mas.S[K,T].lb)
                    else:
                        feas.S[K,T].fix(pe.value(mas.S[K,T]))

            for I in feas.I_reactions:
                for J in feas.J_reactors:
                    if pe.value(mas.varTime[I,J])>=mas.varTime[I,J].ub:
                        feas.link_varTime[I,J].fix(mas.varTime[I,J].ub)  
                    elif pe.value(mas.varTime[I,J])<=mas.varTime[I,J].lb:
                        feas.link_varTime[I,J].fix(mas.varTime[I,J].lb)  
                    else:
                        feas.link_varTime[I,J].fix(pe.value(mas.varTime[I,J]))  


            feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
            print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
            if feas.dsda_status=='Optimal':
                sum_infeasibility=pe.value(feas.obj_feas)
                mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  feas.dual[feas.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(feas.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum(sum(sum(  feas.dual[feas.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(feas.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                print('Feasibility cut added')
            else:
                print('Problem with feasibility stage. Trying a different initialization')
                feas=initialize_model(feas,from_feasible=True,feasible_model='GBD_subproblem')
                feas=solve_subproblem(feas,subproblem_solver=nlp_solver,subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = False,rel_tol = 0)     
                print('feasproblem status:',feas.dsda_status, feas.results.solver.termination_condition)
                if feas.dsda_status=='Optimal':
                    sum_infeasibility=pe.value(feas.obj_feas)
                    mas.cuts.add(sum(sum( feas.dual[feas.const_link_VarTime[I,J]]*(mas.varTime[I,J]-pe.value(feas.link_varTime[I,J])) for I in mas.I_reactions)  for J in mas.J_reactors)+sum(sum(sum(  feas.dual[feas.const_link_B[I,J,T]]*( mas.B[I,J,T]-pe.value(feas.link_B[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum(sum(sum(  feas.dual[feas.const_link_X[I,J,T]]*( mas.X[I,J,T]-pe.value(feas.link_X[I,J,T])   )      for I in mas.I) for J in mas.J) for T in mas.T)+sum_infeasibility<=0)
                    print('Feasibility cut added')
                else:
                    print('GBD solver failure: subproblem detected as infeasible, and fatal error with feasibility stage')
                    break



    ######---------------------------- DISPLAY SOLUTION SUMMARY --------------------------------##################
    model_fun=scheduling_and_control_gdp_N_GBD
    kwargs3=kwargs.copy()  
    subsol=model_fun(**kwargs3)
    subsol=initialize_model(subsol,from_feasible=True,feasible_model=best_sol_name)

    Sol_found=[]
    for I in subsol.I_reactions:
        for J in subsol.J_reactors:
            if subsol.I_i_j_prod[I,J]==1:
                for K in subsol.ordered_set[I,J]:
                    if round(pe.value(subsol.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-subsol.minTau[I,J]+1)
    for I_J in subsol.I_J:
        Sol_found.append(1+round(pe.value(subsol.Nref[I_J])))
    print('EXT_VARS_FOUND',Sol_found)
    TPC1=pe.value(subsol.TCP1)
    TPC2=pe.value(subsol.TCP2)
    TPC3=pe.value(subsol.TCP3)
    TMC=pe.value(subsol.TMC)
    SALES=pe.value(subsol.SALES)
    OBJ_FOUND=TPC1+TPC2+TPC3+TMC-SALES

    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJECTIVE:',str(OBJ_FOUND))


    # ######---------------------------- INFEASIBILITY VERIFICATION EXPERIMENT 3 --------------------------------##################
    # #PROOF that the infeasibility detected by one of the subproblems (the first one) in experiment 3 is due to a numerical issue, rahter than the 
    # # problem is actually infeasible. To prove this, we solve again this infeasible subproblem, but with a different solver and find feasibility
    # model_fun=scheduling_and_control_gdp_N_GBD
    # kwargs3=kwargs.copy()  
    # subsol=model_fun(**kwargs3)
    # subsol=initialize_model(subsol,from_feasible=True,feasible_model='feasibility_subproblem')

    # # Infeasibility test
    # feas2=sub.clone()
    # # fix variables in subproblem 
    # for I in feas2.I:
    #     for J in feas2.J:
    #         for T in feas2.T:

    #             if pe.value(subsol.B[I,J,T])>=subsol.B[I,J,T].ub: 
    #                 feas2.link_B[I,J,T].fix(subsol.B[I,J,T].ub)
    #             elif pe.value(subsol.B[I,J,T])<=subsol.B[I,J,T].lb:
    #                 feas2.link_B[I,J,T].fix(subsol.B[I,J,T].lb)
    #             else:
    #                 feas2.link_B[I,J,T].fix(pe.value(subsol.B[I,J,T]))

    #             feas2.link_X[I,J,T].fix(round(pe.value(subsol.X[I,J,T])))

    # for I_J in feas2.I_J:
    #         I=I_J[0]
    #         J=I_J[1]
    #         feas2.Nref[I,J].fix(round(pe.value(subsol.Nref[I,J])))

    # for K in feas2.K:
    #     for T in feas2.T:
    #         if pe.value(subsol.S[K,T])>=subsol.S[K,T].ub:
    #             feas2.S[K,T].fix(subsol.S[K,T].ub)
    #         elif pe.value(subsol.S[K,T])<=subsol.S[K,T].lb:
    #             feas2.S[K,T].fix(subsol.S[K,T].lb)
    #         else:
    #             feas2.S[K,T].fix(pe.value(subsol.S[K,T]))

    # for I in feas2.I_reactions:
    #     for J in feas2.J_reactors:
    #         if pe.value(subsol.varTime[I,J])>=subsol.varTime[I,J].ub:
    #             feas2.link_varTime[I,J].fix(subsol.varTime[I,J].ub)  
    #         elif pe.value(subsol.varTime[I,J])<=subsol.varTime[I,J].lb:
    #             feas2.link_varTime[I,J].fix(subsol.varTime[I,J].lb)  
    #         else:
    #             feas2.link_varTime[I,J].fix(pe.value(subsol.varTime[I,J]))  

    # feas2=solve_subproblem(feas2,subproblem_solver='knitro',subproblem_solver_options = sub_options,timelimit = 86400, gams_output = False,tee = True,rel_tol = 0) 
 