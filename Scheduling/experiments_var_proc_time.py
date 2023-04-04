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
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
from scheduling_formulation_variable_proc_time import scheduling_gdp_var_proc_time,problem_logic_scheduling


if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)




    obj_Selected='profit_max'
    # initialization=[1, 1, 2, 1, 3, 1, 2, 2, 15, 11, 21, 71, 1, 42, 3, 10]# OPTIMAL SOLUTION TO COST MIN WITH FIXED PROCESSING TIMES AT THEIR NOMINAL value
    initialization=[1, 1, 1, 1, 1, 1, 1, 1, 15, 11, 21, 71, 1, 42, 3, 10]# JUST A FEASIBLE SOLUTION, with processing times initialized at lower bound and batching variables at OPTIMAL SOLUTION TO COST MIN WITH FIXED PROCESSING TIMES
    



    # # CPLEX SOLUTION   
    # mip_solver='CPLEX'
    # sub_options={'add_options':['GAMS_MODEL.optfile = 0;','GAMS_MODEL.threads=1;','option mip='+mip_solver+';\n']}
    # m=scheduling_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected)
    # m=solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=3600000,gams_output=True,tee=True,rel_tol=0)

    # Sol_found=[]
    # for I in m.I:
    #     for J in m.J:
    #         if m.I_i_j_prod[I,J]==1:
    #             Sol_found.append(math.ceil(pe.value(m.tau_p[I,J])/m.delta)-m.minTau[I,J]+1)
    # for I_J in m.I_J:
    #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))

    # print('Ext vars found=',Sol_found)

    # DBD solution
    m=scheduling_gdp_var_proc_time(x_initial=[1, 1, 2, 1, 3, 1, 2, 2, 15, 11, 21, 71, 1, 42, 3, 10],obj_type=obj_Selected)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    mip_solver='CPLEX'
    sub_options={'add_options':['GAMS_MODEL.optfile = 0;','GAMS_MODEL.threads=0;','option mip='+mip_solver+';\n']}
    infinity_val=1e+2
    maxiter=10000
    neigh=neighborhood_k_eq_2(len(initialization))
    logic_fun=problem_logic_scheduling
    model_fun=scheduling_gdp_var_proc_time
    kwargs={'obj_type':obj_Selected}
    [important_info,important_info_preprocessing,D,x_actual]=run_function_dbd(initialization,infinity_val,mip_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
    print('obj= ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    #DSDA SOLUTION
    # start=time.time()
    # D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,initialization,ext_ref,logic_fun,k = '2',provide_starting_initialization= True,feasible_model='dsda',subproblem_solver = mip_solver,subproblem_solver_options=sub_options,iter_timelimit= 1000,timelimit = 3600,gams_output = False,tee= False,global_tee = True,rel_tol = 1e-3)
    # end=time.time()
    # print('Objective='+str(pe.value(D_SDAsol.obj))+', best='+str(routeDSDA[-1]))
    # print('cputime= '+str(end-start))   





    