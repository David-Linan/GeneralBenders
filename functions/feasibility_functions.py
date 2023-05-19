import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt,_FBBTVisitorLeafToRoot
import pyomo.contrib.fbbt.interval as interval
from pyomo.common.collections import ComponentMap
from math import fabs
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.opt.base.solvers import SolverFactory
from functions.dsda_functions import generate_initialization



def feasibility_1(m):
    """
    This function calculates the sum of infeasibility with respect to those constraints that 
    remain fixed after the external variables reformulation. 

    This was adapted from pyomo.contrib.fbbt.fbbt import _fbbt_con

    """
    tol=1e-6
    sum_infeasibility=0  #sum of infeasibility
    infeasible_const=[]  #infeasible constraints
    for constr in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        bnds_dict = ComponentMap()
        visitorA = _FBBTVisitorLeafToRoot(bnds_dict, feasibility_tol=tol)
        visitorA.dfs_postorder_stack(constr.body) 
        _lb = pe.value(constr.lower)
        _ub = pe.value(constr.upper)
        if _lb is None:
            _lb = -interval.inf
        if _ub is None:
            _ub = interval.inf

        lb, ub = bnds_dict[constr.body]

        # check if the constraint is infeasible (original code)
        #if lb > _ub + tol or ub < _lb - tol:
        #    print(lb,_ub)
        #    print(ub,_lb)
        #    raise InfeasibleConstraintException('Detected an infeasible constraint during FBBT: {0}'.format(str(constr)))
        if lb > _ub + tol:
            sum_infeasibility=sum_infeasibility+fabs(lb-_ub)
            #sum_infeasibility=sum_infeasibility+1
        elif ub < _lb - tol:
            sum_infeasibility=sum_infeasibility+fabs(ub-_lb)
            #sum_infeasibility=sum_infeasibility+1
        else:
            continue
        output_dict = dict(name=constr.name)
        infeasible_const.append(output_dict)
        #constr.pprint()
    #Return total sum of infeasibility
    #print([sum_infeasibility,infeasible_const])
    return sum_infeasibility,infeasible_const


def feasibility_2(m,solver,infty_val, use_multistart: bool=False, tee: bool=False):
    """
    This function calculates the minimum sum of infeasibility with respect to those constraints
    that make the subproblem infeasible. 

    This was adapted from pyomo.util.infeasible import log_infeasible_constraints

    infty_val: Value to approximate infinity
    """

    init_path=''
    check_feas1,check_infeas1=feasibility_1(m)
    if check_feas1==0: #If it passes the first feasibility check
        tol=1e-6
        sum_infeasibility=0  #sum of infeasibility
        infeasible_const=[]  #infeasible constraints


###############    Init problem solve
        #Options depending on the solver
        if solver=='conopt' or solver=='conopt4' or solver=='knitro' or solver=='ipopt' or solver=='ipopth' or solver=='cplex':
            if solver=='cplex':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','varsel -1 \n','intsollim 1 \n','$offecho \n'] #feasopt tries to suggest the least change that would achieve feasibility. I limit the number of integer solutions because I do not want to spend too much ime here, varsel -1 leads more quickly to feasible integer solutions
            if solver=='conopt' or solver=='conopt4':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','$offecho \n'] #Do nothing
            if solver=='knitro':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','act_qpalg 1 \n','algorithm 1 \n','bar_feasible 3 \n','$offecho \n'] #Use Interior/direct algorithm and place enphasis on geting feasible first
            if solver=='ipopt' or solver=='ipopth':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','report_mininfeas_solution yes \n','$offecho \n'] #Report most feasible solution if infeasibility is detected
            if use_multistart:
                sub_options=sub_options+['$onecho > msnlp.opt \n','nlpsolver '+solver+'.1 \n','$offecho \n']   #Use the specified NLP solver and its options for multistart
        else:
            sub_options=[]
        #Solve
        if use_multistart:#VALID FOR NLP SUBPROBLEMS ONLY
            SolverFactory('gams', solver='msnlp').solve(m, tee=tee, add_options = sub_options)
        else:
            SolverFactory('gams', solver=solver).solve(m, tee=tee, add_options = sub_options) #IF MULTISTART IS NOT USED AND MSNLP is selected, no problem; msnlp will use conopt as sub_solver

###########end problem solve


        for constr in m.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
            constr_body_value = pe.value(constr.body, exception=False)
            constr_lb_value = pe.value(constr.lower, exception=False)
            constr_ub_value = pe.value(constr.upper, exception=False)

            constr_undefined = False
            equality_violated = False
            lb_violated = False
            ub_violated = False

            if constr_body_value is None:
                # Undefined constraint body value due to missing variable value
                constr_undefined = True
                pass
            else:
                # Check for infeasibilities
                if constr.equality:
                    if fabs(constr_lb_value - constr_body_value) >= tol:
                        equality_violated = True
                        sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)
                else:
                    if constr.has_lb() and constr_lb_value - constr_body_value >= tol:
                        lb_violated = True
                        sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)
                    if constr.has_ub() and constr_body_value - constr_ub_value >= tol:
                        ub_violated = True
                        sum_infeasibility=sum_infeasibility+fabs(constr_body_value - constr_ub_value)
            if not any((constr_undefined, equality_violated, lb_violated, ub_violated)):
                # constraint is fine. skip to next constraint
                continue

            output_dict = dict(name=constr.name)
            infeasible_const.append(output_dict)
    else:
        sum_infeasibility=infty_val
        infeasible_const=check_infeas1
        infeasible_const.append('There were problems with stage 1 feasibility verification. Infeasible constraints shown for stage 1')

    #Generate first feasible initialization
    if sum_infeasibility==0:
        init_path = generate_initialization(m=m)
    #Return total sum of infeasibility
    return sum_infeasibility,infeasible_const,init_path


def feasibility_1_aprox(m):
    """
    This function calculates the sum of infeasibility with respect to those constraints that 
    remain fixed after the external variables reformulation. 

    This was adapted from pyomo.contrib.fbbt.fbbt import _fbbt_con

    """
    m.obj_scheduling.deactivate()
    tol=1e-6
    sum_infeasibility=0  #sum of infeasibility
    infeasible_const=[]  #infeasible constraints
    for constr in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        bnds_dict = ComponentMap()
        visitorA = _FBBTVisitorLeafToRoot(bnds_dict, feasibility_tol=tol)
        visitorA.dfs_postorder_stack(constr.body) 
        _lb = pe.value(constr.lower)
        _ub = pe.value(constr.upper)
        if _lb is None:
            _lb = -interval.inf
        if _ub is None:
            _ub = interval.inf

        lb, ub = bnds_dict[constr.body]

        # check if the constraint is infeasible (original code)
        #if lb > _ub + tol or ub < _lb - tol:
        #    print(lb,_ub)
        #    print(ub,_lb)
        #    raise InfeasibleConstraintException('Detected an infeasible constraint during FBBT: {0}'.format(str(constr)))
        if lb > _ub + tol:
            sum_infeasibility=sum_infeasibility+fabs(lb-_ub)
            #sum_infeasibility=sum_infeasibility+1
        elif ub < _lb - tol:
            sum_infeasibility=sum_infeasibility+fabs(ub-_lb)
            #sum_infeasibility=sum_infeasibility+1
        else:
            continue
        output_dict = dict(name=constr.name)
        infeasible_const.append(output_dict)
        #constr.pprint()
    #Return total sum of infeasibility
    #print([sum_infeasibility,infeasible_const])
    return sum_infeasibility,infeasible_const


def feasibility_2_aprox(m,solver,infty_val, use_multistart: bool=False, tee: bool=False):
    """
    This function calculates the minimum sum of infeasibility with respect to those constraints
    that make the subproblem infeasible. 

    This was adapted from pyomo.util.infeasible import log_infeasible_constraints

    infty_val: Value to approximate infinity
    """
    mip_solver='cplex'
    nlp_solver='conopt4'
    solver='conopt4'

    approximate_solution=True
   

    init_path=''
    check_feas1,check_infeas1=feasibility_1_aprox(m)
    if check_feas1==0: #If it passes the first feasibility check
        tol=1e-6
        sum_infeasibility=0  #sum of infeasibility
        infeasible_const=[]  #infeasible constraints



        #DEACTIVATE DYNAMIC CONSTRAINTS
        for I in m.I_reactions:
            for J in m.J_reactors:
                m.c_dCdtheta[I,J].deactivate()
                m.c_dTRdtheta[I,J].deactivate()                        
                m.c_dTJdtheta[I,J].deactivate()
                m.c_dIntegral_hotdtheta[I,J].deactivate()
                m.c_dIntegral_colddtheta[I,J].deactivate()
                m.Constant_control1[I,J].deactivate()                        
                m.Constant_control2[I,J].deactivate()
        m.C_TCP3.deactivate()
        m.obj.deactivate()
        m.obj_dummy.deactivate()
        m.obj_scheduling.activate()


        #SOLVE SCHEDULING ONLY PROBLEM
        opt = SolverFactory('gams', solver=mip_solver)

        # start=time.time()
        m.prelim_res = opt.solve(m, tee=tee,add_options =['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +mip_solver+ '.opt \n','varsel -1 \n','intsollim 1 \n','$offecho \n'])


        if m.prelim_res.solver.termination_condition == 'infeasible' or m.prelim_res.solver.termination_condition == 'other' or m.prelim_res.solver.termination_condition == 'unbounded' or m.prelim_res.solver.termination_condition == 'invalidProblem' or m.prelim_res.solver.termination_condition == 'solverFailure' or m.prelim_res.solver.termination_condition == 'internalSolverError' or m.prelim_res.solver.termination_condition == 'error'  or m.prelim_res.solver.termination_condition == 'resourceInterrupt' or m.prelim_res.solver.termination_condition == 'licensingProblem' or m.prelim_res.solver.termination_condition == 'noSolution' or m.prelim_res.solver.termination_condition == 'noSolution' or m.prelim_res.solver.termination_condition == 'intermediateNonInteger':
            sum_infeasibility=infty_val
            infeasible_const=check_infeas1
            infeasible_const.append('Scheduling_infeasible')      
        else:

            # ACTIVATE DYNAMIC CONSTRAINTS
            for I in m.I_reactions:
                for J in m.J_reactors:
                    m.c_dCdtheta[I,J].activate()
                    m.c_dTRdtheta[I,J].activate()                        
                    m.c_dTJdtheta[I,J].activate()
                    m.c_dIntegral_hotdtheta[I,J].activate()
                    m.c_dIntegral_colddtheta[I,J].activate()
                    m.Constant_control1[I,J].activate()                        
                    m.Constant_control2[I,J].activate()
            m.C_TCP3.activate()
            m.obj.activate()
            m.obj_scheduling.deactivate() 
            m.obj_dummy.deactivate()

            if approximate_solution:
                # FIX SCHEDULING VARIABLES
                for v in m.component_objects(pe.Var, descend_into=True):
                    # if v.name=='X' or v.name=='B' or v.name=='S' or v.name=='Nref':
                    if v.name=='X' or v.name=='Nref':
                        for index in v:
                            if index==None:
                                v.fix(round(pe.value(v)))
                            else:
                                v[index].fix(round(pe.value(v[index])))
                    # elif v.name=='B' or v.name=='S':
                    #     for index in v:
                    #         if index==None:
                    #             v.fix(pe.value(v))
                    #         else:
                    #             v[index].fix(pe.value(v[index]))  
    ###############    Init problem solve
            #Options depending on the solver
            if solver=='conopt' or solver=='conopt4' or solver=='knitro' or solver=='ipopt' or solver=='ipopth' or solver=='cplex' or solver=="dicopt":
                if solver=='cplex':
                    sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','varsel -1 \n','intsollim 1 \n','$offecho \n'] #feasopt tries to suggest the least change that would achieve feasibility. I limit the number of integer solutions because I do not want to spend too much ime here, varsel -1 leads more quickly to feasible integer solutions
                if solver=='conopt' or solver=='conopt4':
                    sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','$offecho \n'] #Do nothing
                if solver=='knitro':
                    sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','act_qpalg 1 \n','algorithm 1 \n','bar_feasible 3 \n','$offecho \n'] #Use Interior/direct algorithm and place enphasis on geting feasible first
                if solver=='ipopt' or solver=='ipopth':
                    sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','report_mininfeas_solution yes \n','$offecho \n'] #Report most feasible solution if infeasibility is detected
                if solver=='dicopt': #TODO: CONOPT 4 used for a specific case study
                    sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','nlpsolver '+nlp_solver,'\n','$offecho \n']
                # if solver=='baron':
                #     sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > baron.opt \n','FirstFeas 1\n',' NumSol 1\n','$offecho \n']
                # if solver=='lindoglobal':
                #     sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > lindoglobal.opt \n',' GOP_OPT_MODE 0\n','$offecho \n']
                # if solver=='antigone':
                #     sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > antigone.opt \n','abs_opt_tol 100\n','rel_opt_tol 1\n','$offecho \n']
                # if solver=='sbb':
                #     sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > sbb.opt \n','intsollim 1\n','$offecho \n']              
                # if solver=='bonmin':
                #     sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > bonmin.opt \n','bonmin.pump_for_minlp yes\n','pump_for_minlp.solution_limit 1\n','solution_limit 1\n','$offecho \n']  

                if use_multistart:
                    sub_options=sub_options+['$onecho > msnlp.opt \n','nlpsolver '+solver+'.1 \n','$offecho \n']   #Use the specified NLP solver and its options for multistart
            else:
                sub_options=[]
            #Solve
            if use_multistart:#VALID FOR NLP SUBPROBLEMS ONLY
                m.prelim_res2=SolverFactory('gams', solver='msnlp').solve(m, tee=tee, add_options = sub_options)
            else:
                m.prelim_res2=SolverFactory('gams', solver=solver).solve(m, tee=tee, add_options = sub_options) #IF MULTISTART IS NOT USED AND MSNLP is selected, no problem; msnlp will use conopt as sub_solver

    ###########end problem solve


            for constr in m.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
                constr_body_value = pe.value(constr.body, exception=False)
                constr_lb_value = pe.value(constr.lower, exception=False)
                constr_ub_value = pe.value(constr.upper, exception=False)

                constr_undefined = False
                equality_violated = False
                lb_violated = False
                ub_violated = False

                if constr_body_value is None:
                    # Undefined constraint body value due to missing variable value
                    constr_undefined = True
                    pass
                else:
                    # Check for infeasibilities
                    if constr.equality:
                        if fabs(constr_lb_value - constr_body_value)/(1+fabs(constr_lb_value)) >= tol:
                            equality_violated = True
                            sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)/(1+fabs(constr_lb_value))
                    else:
                        if constr.has_lb() and (constr_lb_value - constr_body_value)/(1+fabs(constr_lb_value)) >= tol:
                            lb_violated = True
                            sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)/(1+fabs(constr_lb_value))
                        if constr.has_ub() and (constr_body_value - constr_ub_value)/(1+fabs(constr_ub_value)) >= tol:
                            ub_violated = True
                            sum_infeasibility=sum_infeasibility+fabs(constr_body_value - constr_ub_value)/(1+fabs(constr_ub_value))
                if not any((constr_undefined, equality_violated, lb_violated, ub_violated)):
                    # constraint is fine. skip to next constraint
                    continue

                output_dict = dict(name=constr.name)
                infeasible_const.append(output_dict)

            if not(m.prelim_res2.solver.termination_condition == 'infeasible' or m.prelim_res2.solver.termination_condition == 'other' or m.prelim_res2.solver.termination_condition == 'unbounded' or m.prelim_res2.solver.termination_condition == 'invalidProblem' or m.prelim_res2.solver.termination_condition == 'solverFailure' or m.prelim_res2.solver.termination_condition == 'internalSolverError' or m.prelim_res2.solver.termination_condition == 'error'  or m.prelim_res2.solver.termination_condition == 'resourceInterrupt' or m.prelim_res2.solver.termination_condition == 'licensingProblem' or m.prelim_res2.solver.termination_condition == 'noSolution' or m.prelim_res2.solver.termination_condition == 'noSolution' or m.prelim_res2.solver.termination_condition == 'intermediateNonInteger'):
                sum_infeasibility=0
            if sum_infeasibility!=0: 
                sum_infeasibility=sum_infeasibility/10000

            
    else:
        sum_infeasibility=infty_val
        infeasible_const=check_infeas1
        infeasible_const.append('Stage_1_infeasible')

    #Initialize source of infeasibility
    source={}
    #Generate first feasible initialization
    if sum_infeasibility==0:
        init_path = generate_initialization(m=m)
    #Identify source of infeasibility, when the scheduling was feasible, but there is infeasibility in dynamics
    elif sum_infeasibility!=infty_val:

        m.E2_CAPACITY_LOW.deactivate()
        m.E2_CAPACITY_UP.deactivate()
        m.E3_BALANCE_INIT.deactivate()
        m.E_DEMAND_SATISFACTION.deactivate()
        m.E1_UNIT.deactivate()
        m.E3_BALANCE.deactivate()
        for II in m.I_reactions:
            for JJ in m.J_reactors:     
                if round(pe.value(m.Nref[II,JJ]))>=1:
                    for I in m.I_reactions:
                        for J in m.J_reactors:
                            m.c_dCdtheta[I,J].deactivate()
                            m.c_dTRdtheta[I,J].deactivate()                        
                            m.c_dTJdtheta[I,J].deactivate()
                            m.c_dIntegral_hotdtheta[I,J].deactivate()
                            m.c_dIntegral_colddtheta[I,J].deactivate()
                            m.Constant_control1[I,J].deactivate()                        
                            m.Constant_control2[I,J].deactivate()
                            m.finalCon[I,J].deactivate()
                            m.finalTemp[I,J].deactivate()
                    m.C_TCP3.deactivate()
                    m.obj_scheduling.deactivate() 

                    m.c_dCdtheta[II,JJ].activate()
                    m.c_dTRdtheta[II,JJ].activate()                        
                    m.c_dTJdtheta[II,JJ].activate()
                    m.c_dIntegral_hotdtheta[II,JJ].activate()
                    m.c_dIntegral_colddtheta[II,JJ].activate()
                    m.Constant_control1[II,JJ].activate()                        
                    m.Constant_control2[II,JJ].activate()
                    m.finalCon[II,JJ].activate()
                    m.finalTemp[II,JJ].activate() 
                    m.obj.deactivate()
                    m.obj_dummy.activate()

                    if approximate_solution:
                        # FIX SCHEDULING VARIABLES
                        for v in m.component_objects(pe.Var, descend_into=True):
                            if v.name=='X' or v.name=='Nref':
                                for index in v:
                                    if index==None:
                                        v.fix(round(pe.value(v)))
                                    else:
                                        v[index].fix(round(pe.value(v[index])))
                            # elif v.name=='B' or v.name=='S':
                            #     for index in v:
                            #         if index==None:
                            #             v.fix(pe.value(v))
                            #         else:
                            #             v[index].fix(pe.value(v[index]))                           

                    opt = SolverFactory('gams', solver=nlp_solver)
                    # start=time.time()
                    m.results = opt.solve(m, tee=tee,skip_trivial_constraints=True)
                    #print(m.results.solver.termination_condition)
                    if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
                        source[II,JJ]="Infeasible"
                    else:
                        source[II,JJ]="Feasible"
                else:
                    source[II,JJ]="Not_scheduled"
    #Return total sum of infeasibility
    return sum_infeasibility,infeasible_const,init_path,source




def feasibility_2_modified(m,solver,infty_val, use_multistart: bool=False, tee: bool=False):
    """
    Same as feasibility 2, but no initialization is generated
    """

    check_feas1,check_infeas1=feasibility_1(m)
    if check_feas1==0: #If it passes the first feasibility check
        tol=1e-6
        sum_infeasibility=0  #sum of infeasibility
        infeasible_const=[]  #infeasible constraints


###############    Init problem solve
        #Options depending on the solver
        if solver=='conopt' or solver=='conopt4' or solver=='knitro' or solver=='ipopt' or solver=='ipopth':
            if solver=='conopt' or solver=='conopt4':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','$offecho \n'] #Do nothing
            if solver=='knitro':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','act_qpalg 1 \n','algorithm 1 \n','bar_feasible 3 \n','$offecho \n'] #Use Interior/direct algorithm and place enphasis on geting feasible first
            if solver=='ipopt' or solver=='ipopth':
                sub_options=['GAMS_MODEL.optfile = 1;','\n','$onecho > ' +solver+ '.opt \n','report_mininfeas_solution yes \n','$offecho \n'] #Report most feasible solution if infeasibility is detected
            if use_multistart:
                sub_options=sub_options+['$onecho > msnlp.opt \n','nlpsolver '+solver+'.1 \n','$offecho \n']   #Use the specified NLP solver and its options for multistart
        else:
            sub_options=[]
        #Solve
        if use_multistart:#VALID FOR NLP SUBPROBLEMS ONLY
            SolverFactory('gams', solver='msnlp').solve(m, tee=tee, add_options = sub_options)
        else:
            SolverFactory('gams', solver=solver).solve(m, tee=tee, add_options = sub_options)

###########end problem solve


        for constr in m.component_data_objects(ctype=pe.Constraint, active=True, descend_into=True):
            constr_body_value = pe.value(constr.body, exception=False)
            constr_lb_value = pe.value(constr.lower, exception=False)
            constr_ub_value = pe.value(constr.upper, exception=False)

            constr_undefined = False
            equality_violated = False
            lb_violated = False
            ub_violated = False

            if constr_body_value is None:
                # Undefined constraint body value due to missing variable value
                constr_undefined = True
                pass
            else:
                # Check for infeasibilities
                if constr.equality:
                    if fabs(constr_lb_value - constr_body_value) >= tol:
                        equality_violated = True
                        sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)
                else:
                    if constr.has_lb() and constr_lb_value - constr_body_value >= tol:
                        lb_violated = True
                        sum_infeasibility=sum_infeasibility+fabs(constr_lb_value - constr_body_value)
                    if constr.has_ub() and constr_body_value - constr_ub_value >= tol:
                        ub_violated = True
                        sum_infeasibility=sum_infeasibility+fabs(constr_body_value - constr_ub_value)
            if not any((constr_undefined, equality_violated, lb_violated, ub_violated)):
                # constraint is fine. skip to next constraint
                continue

            output_dict = dict(name=constr.name)
            infeasible_const.append(output_dict)
        sum_infeasibility=sum_infeasibility/10000
    else:
        sum_infeasibility=infty_val
        infeasible_const=check_infeas1
        infeasible_const.append('There were problems with stage 1 feasibility verification. Infeasible constraints shown for stage 1')

    #Return total sum of infeasibility
    return sum_infeasibility,infeasible_const

