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
    else:
        sum_infeasibility=infty_val
        infeasible_const=check_infeas1
        infeasible_const.append('There were problems with stage 1 feasibility verification. Infeasible constraints shown for stage 1')

    #Return total sum of infeasibility
    return sum_infeasibility,infeasible_const

