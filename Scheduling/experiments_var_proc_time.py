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
from scheduling_formulation_variable_proc_time import scheduling_gdp_var_proc_time,problem_logic_scheduling
import os

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)




    obj_Selected='profit_max'
    neighdef='2'
    print(neighdef)
    # initialization=[1, 1, 2, 1, 3, 1, 2, 2, 15, 11, 21, 71, 1, 42, 3, 10]# OPTIMAL SOLUTION TO COST MIN WITH FIXED PROCESSING TIMES AT THEIR NOMINAL value
    # initialization=[1, 1, 1, 1, 1, 1, 1, 1, 15, 11, 21, 71, 1, 42, 3, 10]# JUST A FEASIBLE SOLUTION, with processing times initialized at lower bound and batching variables at OPTIMAL SOLUTION TO COST MIN WITH FIXED PROCESSING TIMES
    # initialization=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # initialization= [1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 2, 3, 1, 2]

    initialization=[1, 1, 1, 1, 1, 1, 1, 1]

    # # CPLEX SOLUTION   
    mip_solver='CPLEX'
    sub_options={'add_options':['GAMS_MODEL.optfile = 0;','GAMS_MODEL.threads=0;','option mip='+mip_solver+';\n']}

    LO_PROC_TIME={('T1','U1'):0.1,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):0.1,('T3','U3'):0.1,('T4','U2'):0.1,('T4','U3'):0.1,('T5','U4'):0.1}
    UP_PROC_TIME={('T1','U1'):2,('T2','U2'):2,('T2','U3'):2,('T3','U2'):2,('T3','U3'):2,('T4','U2'):2,('T4','U3'):2,('T5','U4'):2}


    for param in range(2,500):
        print('\n------------ num discrete points: ',param,' -------------------------------')
        m=scheduling_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=param,last_time_hours=5,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
        ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
        # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
        [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=False)
        # start=time.time()
        # m=solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0)
        # end=time.time()
        # # print('num_disc:',param,'obj:',pe.value(m.obj))
        # Sol_found=[]
        # for I in m.I:
        #     for J in m.J:
        #         if m.I_i_j_prod[I,J]==1:
        #             for K in m.ordered_set[I,J]:
        #                 if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
        #                     Sol_found.append(K-m.minTau[I,J]+1)
        # # for I_J in m.I_J:
        # #     Sol_found.append(1+round(pe.value(m.Nref[I_J])))

        # print('Objective CPLEX=',pe.value(m.obj),'best CPLEX=',Sol_found,'cputime CPLEX=',str(end-start))

        # textbuffer = io.StringIO()
        # for v in m.component_objects(pe.Var, descend_into=True):
        #     v.pprint(textbuffer)
        #     textbuffer.write('\n')
        # textbuffer.write('\n Objective: \n') 
        # textbuffer.write(str(pe.value(m.obj)))  
        # file_name='Results_var_proc_time.txt'  
        # with open(os.path.join('C:/Users/dlinanro/Desktop/GeneralBenders/Scheduling',file_name), 'w') as outputfile:
        #     outputfile.write(textbuffer.getvalue())
    
        # for I_J in m.I_J:
        #     I=I_J[0]
        #     J=I_J[1]
        #     for K in m.ordered_set[I,J]:
        #         if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
        #             variable_bound_found=K*m.delta
        #     print(I_J)
        #     if sum(pe.value(m.X[I,J,T]) for T in m.T)>=1:
        #         print('maximum variable time [h]: ',max(pe.value(m.varTime[I,J,T]) for T in m.T if round(pe.value(m.X[I,J,T]))==1),'<= current bound (based on discretization) [h]: ',variable_bound_found,'<= initial bound (user) [h]: ',UP_PROC_TIME[(I,J)])








        par_points=param
        # initialization=Sol_found

        mip_solver='CPLEX'
        sub_options={'add_options':['GAMS_MODEL.optfile = 0;','GAMS_MODEL.threads=0;','option mip='+mip_solver+';\n']}
        infinity_val=1e+2
        maxiter=10000
        # neigh=neighborhood_k_eq_all(len(initialization))
        # print(neigh)
        # neigh=neighborhood_k_eq_l_natural(len(initialization))
        logic_fun=problem_logic_scheduling
        model_fun=scheduling_gdp_var_proc_time
        kwargs={'obj_type':obj_Selected,'last_disc_point':par_points,'last_time_hours':5,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME}
        # DBD solution
        # [important_info,important_info_preprocessing,D,x_actual]=run_function_dbd(initialization,infinity_val,mip_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True)
        # print('obj= ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
        #DSDA SOLUTION


        # initialization=[upper_bounds[k] for k in upper_bounds.keys()]


        start=time.time()
        D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,initialization,ext_ref,logic_fun,k = neighdef,provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = mip_solver,subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = False,rel_tol = 0,scaling=False,scale_factor=1,stop_neigh_verif_when_improv=True)
        end=time.time()
        print('Objective D-SDA='+str(pe.value(D_SDAsol.obj))+', best D-SDA='+str(routeDSDA[-1]),'cputime D-SDA= '+str(end-start))  

        scale_init=max([max(abs(initialization[i]-upper_bounds[i+1]) for i in range(len(initialization))   ),max(abs(initialization[i]-lower_bounds[i+1]) for i in range(len(initialization))) ] )
        if param>=3:
            start=time.time()

            kter=-1
            result_x=initialization
            while True:
                kter=kter+1

                if kter==0:
                    initial=initialization
                    route_ini=[]
                    obj_route_ini=[]
                else:
                    initial=result_x
                    route_ini=routeDSDA
                    obj_route_ini=obj_route
                D_SDAsol,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,initial,ext_ref,logic_fun,k = neighdef,provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = mip_solver,subproblem_solver_options=sub_options,iter_timelimit= 86400,timelimit = 86400,gams_output = False,tee= False,global_tee = False,rel_tol = 0,scaling=True,scale_factor=scale_init,stop_neigh_verif_when_improv=False,route_initial=route_ini,obj_route_initial=obj_route_ini)
                result_x=routeDSDA[-1]
                if scale_init==1:
                    break
                # scale_init=scale_init-1
                scale_init=math.ceil(scale_init/2)
    
            end=time.time()
            print('Objective D-SDSA='+str(pe.value(D_SDAsol.obj))+', best D-SDSA='+str(routeDSDA[-1]),'cputime D-SDSA= '+str(end-start))  

            m=D_SDAsol
            for I_J in m.I_J:
                I=I_J[0]
                J=I_J[1]
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        variable_bound_found=K*m.delta
                print(I_J)
                if sum(pe.value(m.X[I,J,T]) for T in m.T)>=1:
                    print('maximum variable time [h]: ',max(pe.value(m.varTime[I,J,T]) for T in m.T if round(pe.value(m.X[I,J,T]))==1),'<= current bound (based on discretization) [h]: ',variable_bound_found,'<= initial bound (user) [h]: ',UP_PROC_TIME[(I,J)])




    