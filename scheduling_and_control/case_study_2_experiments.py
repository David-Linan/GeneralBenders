from __future__ import division
from pickle import TRUE

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min_ref_2
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_minlp
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_all,neighborhood_k_eq_l_natural,neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
from case_study_2_model import case_2_scheduling_control_gdp_var_proc_time,problem_logic_scheduling
import os

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)




    obj_Selected='profit_max'
    neighdef='2'
    # initialization=[1, 1, 2, 1, 3, 1, 2, 2, 15, 11, 21, 71, 1, 42, 3, 10]# OPTIMAL SOLUTION TO COST MIN WITH FIXED PROCESSING TIMES AT THEIR NOMINAL value
    # initialization=[1, 1, 1, 1, 1, 1, 1, 1, 15, 11, 21, 71, 1, 42, 3, 10]# JUST A FEASIBLE SOLUTION, with processing times initialized at lower bound and batching variables at OPTIMAL SOLUTION TO COST MIN WITH FIXED PROCESSING TIMES
    # initialization=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # initialization= [1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 2, 3, 1, 2]

    initialization=[1, 1, 1, 1, 1, 1, 1, 1]

    # # CPLEX SOLUTION   
    mip_solver='CPLEX'
    minlp_solver='DICOPT'
    sub_options={'add_options':['GAMS_MODEL.optfile = 0;','GAMS_MODEL.threads=0;','option mip='+mip_solver+';\n']}

    LO_PROC_TIME={('T1','U1'):0.1,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):0.1,('T3','U3'):0.1,('T4','U2'):0.1,('T4','U3'):0.1,('T5','U4'):0.1}
    UP_PROC_TIME={('T1','U1'):2,('T2','U2'):2,('T2','U3'):2,('T3','U2'):2,('T3','U3'):2,('T4','U2'):2,('T4','U3'):2,('T5','U4'):2}

    m=case_2_scheduling_control_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=24,last_time_hours=12,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
    start=time.time()
    m=solve_with_minlp(m,transformation='bigm',minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    end=time.time()
    # print('num_disc:',param,'obj:',pe.value(m.obj))
    Sol_found=[]
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))

        textbuffer = io.StringIO()
        for v in m.component_objects(pe.Var, descend_into=True):
            v.pprint(textbuffer)
            textbuffer.write('\n')
        textbuffer.write('\n Objective: \n') 
        textbuffer.write(str(pe.value(m.obj)))  
        file_name='Case_2_results_var_proc_time.txt'  
        with open(os.path.join('C:/Users/dlinanro/Desktop/GeneralBenders/scheduling_and_control',file_name), 'w') as outputfile:
            outputfile.write(textbuffer.getvalue())
    print('Objective DICOPT=',pe.value(m.obj),'best DICOPT=',Sol_found,'cputime DICOPT=',str(end-start))

