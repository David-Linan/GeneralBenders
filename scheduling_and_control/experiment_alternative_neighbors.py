from __future__ import division
import sys
# sys.path.insert(0, '/home/dadapy/GeneralBenders/')
# sys.path.append('C:/Users/TEMP/Desktop/GeneralBenders/') #for LRLAB5
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_aprox
from functions.dsda_functions import get_external_information,external_ref,external_ref_neighborhood,solve_subproblem,generate_initialization,initialize_model,solve_with_gdpopt,solve_with_minlp
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,solve_subproblem_aprox,generate_initialization,initialize_model,solve_with_dsda,solve_with_dsda_aprox,sequential_iterative_1,sequential_iterative_2,neighborhood_k_eq_l_natural,neighborhood_k_eq_m_natural,neighborhood_k_eq_l_natural_modified,solve_with_dsda_aprox_tau_only
import logging
# from Scheduling_control_variable_tau_model_reduced import scheduling_and_control,problem_logic_scheduling
# from Scheduling_control_variable_tau_model import scheduling_and_control as scheduling_and_control_GDP 
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N as scheduling_and_control_GDP_complete
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx as scheduling_and_control_GDP_complete_approx
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx_only_tau as scheduling_and_control_GDP_complete_approx_tau_only
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_approx_sequential
from Scheduling_control_variable_tau_model import scheduling_and_control_gdp_N_solvegdp_simpler
from Scheduling_control_variable_tau_model import problem_logic_scheduling, problem_logic_scheduling_tau_only,problem_logic_scheduling_dummy
from Scheduling_control_variable_tau_model import scheduling_only_gdp_N_solvegdp_simpler,scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
import matplotlib.pyplot as plt






if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    nlp_solver='conopt4'
    minlp_solver='dicopt'
    mip_solver='cplex'
    gdp_solver='LBB'
    rel_tol=0
    # # SHORT SCHEDULING
    # ext_vars=[4, 4, 6, 6, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2] #Best solution known from sequential iterative, short scheduling obj=-1148
    # ext_vars=[3, 5, 5, 6, 2, 5, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3] #Solution fron infeasible initialization, obj=-1085
    ext_vars=[4, 4, 5, 5, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2] #Sequential iterative, Also change solve_subproblem_aprox to fix all scheduling desitions
    # ext_vars=[1, 1, 1, 1, 1, 1, 3, 3, 2, 4, 4, 3, 4, 4, 4, 5] #Scheduling only. Remember to activate scheduling only in solution of subproblem
    sub_options={'add_options':['GAMS_MODEL.optfile = 1;','Option Threads =0;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 2 \n','maxcycles 2 \n','infeasder 0','$offecho \n']}
    # sub_options={'add_options':['GAMS_MODEL.optfile = 0;','Option Threads =0;','Option SOLVER = OCTERACT;']}
    # BRANCHING PRIORITIES (tHIS IS DOING NOTHING HERE BECAUSE I HAVE N_I_J FIXED)
    feas_cuts=[ext_vars]
    current_objective=-10.3698 #-11.480940951614194
    best_sol=current_objective
    model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    kwargs={}
    m=model_fun(**kwargs)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)

    size_neigh_out=2*6
    size_neigh_in=2*10
    max_iter_out=10000000
    current_central=ext_vars
    upper_evaluated={}
    



    start=time.time()
    for out in range(max_iter_out):
        print('------Outer iteration= ',out+1)
        upper_evaluated[out+1]=[current_central]
        for out_count in range(size_neigh_out):
            print('   ------------------- Outer Neighbor ',out_count+1,'--------------------------------------------')
            print('   DICOPT:')
            model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
            logic_fun=problem_logic_scheduling
            kwargs={}
            m=model_fun(**kwargs)
            m = external_ref_neighborhood(m=m,x=current_central,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=True,feasibility_cuts=feas_cuts,dynamic_vars=True,neigh_size=2,interactions=10000)
            # m=solve_subproblem(m=m_fixed,subproblem_solver=minlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=True,rel_tol=rel_tol)
            m =solve_with_minlp(m,transformation='hull',minlp=minlp_solver,minlp_options=sub_options,timelimit=3600000,gams_output=False,tee=False,rel_tol=rel_tol) 
            
            Sol_found=[]
            for I in m.I_reactions:
                for J in m.J_reactors:
                    Sol_found.append(math.ceil(pe.value(m.varTime[I,J])/m.delta)-m.minTau[I,J]+1)
            for I_J in m.I_J:
                Sol_found.append(1+round(pe.value(m.Nref[I_J])))
            direction=[]
            for i in range(len(Sol_found)):
                direction.append(Sol_found[i]-ext_vars[i])

                
            if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
                m.dicopt_status='Infeasible'
            else:
                m.dicopt_status='Optimal'
            current_obj_dicopt=m.obj
            if m.dicopt_status == 'Optimal':  

                print('   Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                            
            else:
                print('   Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))        


            print('   SEARCH DIRECTION: ', direction)
            # SOLVE APPROXIMATE SUBPROBLEM TO COMPARE OBJECTIVE FUNCTION
            #TODO: HERE I HAVE TO DO THIS BECAUSE I HAVE MINLP SUBPROBLEMS AND THERE ARE BINARY VARIABLES IN SUBPROBLEMS SUCH AS X_i,j,t, BUT IF SUBPROBLEMS ARE NLP , then only test by dicopt is required
            model_fun =scheduling_and_control_GDP_complete_approx
            logic_fun=problem_logic_scheduling_dummy
            m=model_fun(**kwargs)

            m_fixed = external_ref(m=m,x=Sol_found,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
            m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=False,rel_tol=rel_tol,best_sol=best_sol)

            


            print(' \n   MINLP subproblem approximate evaluation:')
            if m.dsda_status == 'Optimal':  

                print('   Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                            
            else:
                if m.pruned_Status=='Pruned_SchedulingInfeasible':
                    print('   Pruned:', Sol_found, '   |   Lower bound problem infeasible   |   Global Time:', round(time.time()- start, 2))                    
                elif m.pruned_Status=='Pruned_NoImprovementExpected':
                    print('   Pruned:', Sol_found, '   |   No improvement expected   |   Global Time:', round(time.time()- start, 2))  
                else:
                    print('   Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))     
            


            # UPDATE
            feas_cuts.append(Sol_found) 
            upper_evaluated[out+1].append(Sol_found) 
            # VERIFY BREAK CONDITION       
            if m.best_sol<=best_sol:
                best_sol=m.best_sol
                current_central=Sol_found
                print('   New trial point found: ', Sol_found)
                break
            elif out_count+1==size_neigh_out:
                coin=0  
                for current_in in upper_evaluated[out+1]:
                    coin=coin+1
                    out_of_previous=False
                    print('      -------evaluating the inner neighborhood of :',current_in)
                    print('      ------- This is neighbor numer ',coin,' of the second layer')
                    for in_count in range(size_neigh_in):
                        print('      ------------------- Inner Neighbor ',in_count+1,'--------------------------------------------')
                        print('      DICOPT:')
                        model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
                        logic_fun=problem_logic_scheduling
                        kwargs={}
                        m=model_fun(**kwargs)
                        m = external_ref_neighborhood(m=m,x=current_in,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=True,feasibility_cuts=feas_cuts,dynamic_vars=False,neigh_size=2,interactions=10000)
                        # m=solve_subproblem(m=m_fixed,subproblem_solver=minlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=True,rel_tol=rel_tol)
                        m =solve_with_minlp(m,transformation='hull',minlp=minlp_solver,minlp_options=sub_options,timelimit=3600000,gams_output=False,tee=False,rel_tol=rel_tol) 
                        
                        Sol_found=[]
                        for I in m.I_reactions:
                            for J in m.J_reactors:
                                Sol_found.append(math.ceil(pe.value(m.varTime[I,J])/m.delta)-m.minTau[I,J]+1)
                        for I_J in m.I_J:
                            Sol_found.append(1+round(pe.value(m.Nref[I_J])))
                        direction=[]
                        for i in range(len(Sol_found)):
                            direction.append(Sol_found[i]-ext_vars[i])

                            
                        if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
                            m.dicopt_status='Infeasible'
                        else:
                            m.dicopt_status='Optimal'
                        current_obj_dicopt=m.obj
                        if m.dicopt_status == 'Optimal':  

                            print('      Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                                        
                        else:
                            print('      Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))        


                        print('      SEARCH DIRECTION: ', direction)
                        # SOLVE APPROXIMATE SUBPROBLEM TO COMPARE OBJECTIVE FUNCTION
                        #TODO: HERE I HAVE TO DO THIS BECAUSE I HAVE MINLP SUBPROBLEMS AND THERE ARE BINARY VARIABLES IN SUBPROBLEMS SUCH AS X_i,j,t, BUT IF SUBPROBLEMS ARE NLP , then only test by dicopt is required
                        model_fun =scheduling_and_control_GDP_complete_approx
                        logic_fun=problem_logic_scheduling_dummy
                        m=model_fun(**kwargs)

                        m_fixed = external_ref(m=m,x=Sol_found,extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
                        m = solve_subproblem_aprox(m=m_fixed,subproblem_solver=nlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=False,rel_tol=rel_tol,best_sol=best_sol)

                        


                        print(' \n      MINLP subproblem approximate evaluation:')
                        if m.dsda_status == 'Optimal':  

                            print('      Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                                        
                        else:
                            if m.pruned_Status=='Pruned_SchedulingInfeasible':
                                print('      Pruned:', Sol_found, '   |   Lower bound problem infeasible   |   Global Time:', round(time.time()- start, 2))                    
                            elif m.pruned_Status=='Pruned_NoImprovementExpected':
                                print('      Pruned:', Sol_found, '   |   No improvement expected   |   Global Time:', round(time.time()- start, 2))  
                            else:
                                print('      Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))     
                        


                        # UPDATE
                        feas_cuts.append(Sol_found) 
                        # VERIFY BREAK CONDITION       
                        if m.best_sol<=best_sol:
                            best_sol=m.best_sol
                            current_central=Sol_found
                            print('      New trial point found: ', Sol_found)
                            out_of_previous=True
                            break
                        elif in_count+1==size_neigh_in:
                            print('***The objective function is not improving. Optimal solution found')
                            print('***Best objective function found: ',best_sol)
                            print('***Best ext vars:',current_central)
                            print('***CPU time:', round(time.time()- start, 2))
                    if out_of_previous==True:
                        break 





