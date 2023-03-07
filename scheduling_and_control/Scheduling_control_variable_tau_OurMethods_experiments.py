from __future__ import division
import sys
# sys.path.insert(0, '/home/dadapy/GeneralBenders/')
# sys.path.append('C:/Users/TEMP/Desktop/GeneralBenders/') #for LRLAB5
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_aprox
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_gdpopt,solve_with_minlp
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

    #Solver declaration
    nlp_solver='conopt4'
    mip_solver='cplex'
    minlp_solver='dicopt'





    ## TEST JUMP BORRAR
    tau_init=[1,1,1,1,1,1] # Initialization of ext vars in the domain of ext vars. This will also be the lower bound of processing times
    sub_options_infeas_init={'add_options':['GAMS_MODEL.optfile = 1;','option mip=convert;\n','$onecho > convert.opt \n','JuMP model_david.jl \n','$offecho \n']}
    kwargs={'x_initial':tau_init,'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    model_fun=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
    m=model_fun(**kwargs)
    m_scheduling = solve_with_minlp(m,transformation='hull',minlp='convert',minlp_options=sub_options_infeas_init,timelimit=3600000,gams_output=True,tee=True,rel_tol=0.05)

    for I_J in m_scheduling.I_J:
        tau_init.append(1+round(pe.value(m_scheduling.Nref[I_J])))

    print('Infeasible initialization of ext-vars: ',tau_init)
 

    ##INFEASIBLE INITIALIZATION
    # tau_init=[1,1,1,1,1,1] # Initialization of ext vars in the domain of ext vars. This will also be the lower bound of processing times
    # sub_options_infeas_init={'add_options':['GAMS_MODEL.optfile = 0;','GAMS_MODEL.threads=1;','option mip='+mip_solver+';\n']}
    # kwargs={'x_initial':tau_init,'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    # model_fun=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
    # m=model_fun(**kwargs)
    # m_scheduling = solve_with_minlp(m,transformation='hull',minlp=mip_solver,minlp_options=sub_options_infeas_init,timelimit=3600000,gams_output=True,tee=True,rel_tol=0.05)

    # for I_J in m_scheduling.I_J:
    #     tau_init.append(1+round(pe.value(m_scheduling.Nref[I_J])))

    # print('Infeasible initialization of ext-vars: ',tau_init)

    # textbuffer = io.StringIO()
    # for v in m_scheduling.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m_scheduling.obj)))    
    # with open('Results_variable_tau_scheduling_only_increased_horizon.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())      



    ## FEASIBLE INITIALIZATION
    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 20000 \n','$offecho \n']}
    # model_fun =scheduling_and_control_gdp_N_approx_sequential
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,sol_tau_ext=sequential_iterative_2(logic_fun,[1,1,1,1,1,1],model_fun,kwargs,ext_ref,rate_tau=1,provide_starting_initialization = False,subproblem_solver=nlp_solver,iter_timelimit = 1000000,subproblem_solver_options=sub_options,gams_output = False,tee = False,global_tee = True,rel_tol = 0.05)

    # Init_found=sol_tau_ext

    # for I_J in m.I_J:
    #     Init_found.append(1+round(pe.value(m.Nref[I_J])))

    # print('Feasible initialization of ext-vars: ',Init_found)

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_sequential_Strategy_improved_increased_horizon.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    feas_init=[4, 4, 5, 5, 3, 3, 4, 2, 5, 5, 6, 7, 2, 7, 6, 8] # for {'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    feas_init=[4, 4, 5, 5, 3, 3, 4, 3, 3, 5, 5, 5, 4, 6, 5, 7] # for {'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    gap=0.05

    # ## EXPERIMENT 1: GAP 0.05, FROM FEASIBLE INITIALIZATION, ENHANCED DSDA, RIGUROUS SOLUTION OF SUBPROBLEMS
    # # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 20000 \n','$offecho \n']}
    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option mip=cplex; \n','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','mipoptfile 1 \n','$offecho \n','$onecho > cplex.opt \n','epgap '+str(gap)+'\n','$offecho \n']}
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda_aprox(model_fun,kwargs,feas_init,ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = minlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = gap)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dsda_rigurous_k_2_increased_horizon.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    # ## EXPERIMENT 2: GAP 0.05, FROM FEASIBLE INITIALIZATION, ENHANCED DSDA, APROX SOLUTION OF SUBPROBLEMS
    # sub_options={'add_options':['GAMS_MODEL.optfile = 0;'+'\n','GAMS_MODEL.threads = 0;']}
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # m,routeDSDA,obj_route=solve_with_dsda_aprox(model_fun,kwargs,feas_init,ext_ref,logic_fun,k = '2',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = nlp_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= False,global_tee = True,rel_tol = gap)
    # print('Objective value: ',str(pe.value(m.obj)))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dsda_aprox_k_2_increased_horizon28.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())
    # ## EXPERIMENT 3: GAP 0.05, FROM FEASIBLE INITIALIZATION, ENHANCED DBD, RIGUROUS SOLUTION OF SUBPROBLEMS
    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option mip=cplex; \n','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','mipoptfile 1 \n','$offecho \n','$onecho > cplex.opt \n','epgap '+str(gap)+'\n','$offecho \n']}
    # initialization=feas_init
    # infinity_val=1e+4 #TODO: DBD FROM FEASIBLE WORKED VERY WELL WITH 1E+4. I HAVE TO USE DIFFFERENT INFINITY VALUES DEPENDING ON STAGE 1 2 OR 3. I have scaled objective in phase 2
    # maxiter=10000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=gap)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dbd_rigurous_k_2_increased_horizon.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())



    # ## EXPERIMENT 4: GAP 0.05, FROM FEASIBLE INITIALIZATION, ENHANCED DBD, APROX SOLUTION OF SUBPROBLEMS
    # sub_options={'add_options':['GAMS_MODEL.optfile = 0;'+'\n','GAMS_MODEL.threads = 0;']}
    # initialization=feas_init
    # infinity_val=1e+5 #TODO: DBD FROM FEASIBLE WORKED VERY WELL WITH 1E+4. I HAVE TO USE DIFFFERENT INFINITY VALUES DEPENDING ON STAGE 1 2 OR 3. I have scaled objective in phase 2
    # maxiter=10000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_GDP_complete_approx
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=gap)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dbd_aprox_k_2_increased_horizon28.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    infeas_init=[1, 1, 1, 1, 1, 1, 5, 3, 5, 9, 3, 9, 2, 8, 7, 9] # for {'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    infeas_init=[1, 1, 1, 1, 1, 1, 5, 3, 5, 10, 3, 7, 5, 8, 7, 9] #for {'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    # ## EXPERIMENT 5: GAP 0.05, FROM INFEASIBLE INITIALIZATION, ENHANCED DBD, RIGUROUS SOLUTION OF SUBPROBLEMS, ONLY FEASIBILITY IN INITIALIZATION
    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option mip=cplex; \n','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','mipoptfile 1 \n','$offecho \n','$onecho > cplex.opt \n','epgap '+str(gap)+'\n','$offecho \n']}
    # initialization=infeas_init
    # infinity_val=1e+4 #TODO: DBD FROM FEASIBLE WORKED VERY WELL WITH 1E+4. I HAVE TO USE DIFFFERENT INFINITY VALUES DEPENDING ON STAGE 1 2 OR 3. I have scaled objective in phase 2
    # maxiter=10000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_GDP_complete_approx
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,minlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=gap)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dbd_rigurous_k_2_increased_horizon_from_infeasible.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    # ## EXPERIMENT 6: GAP 0.05, FROM INFEASIBLE INITIALIZATION, ENHANCED DBD, APROX SOLUTION OF SUBPROBLEMS, ONLY FEASIBILITY IN INITIALIZATION
    # sub_options={'add_options':['GAMS_MODEL.optfile = 0;']}
    # initialization=infeas_init
    # infinity_val=1e+4 #TODO: DBD FROM FEASIBLE WORKED VERY WELL WITH 1E+4. I HAVE TO USE DIFFFERENT INFINITY VALUES DEPENDING ON STAGE 1 2 OR 3. I have scaled objective in phase 2
    # maxiter=10000
    # neigh=neighborhood_k_eq_2(len(initialization))
    # model_fun =scheduling_and_control_GDP_complete_approx
    # model_fun_scheduling=scheduling_only_gdp_N_solvegdp_simpler_lower_bound_tau
    # logic_fun=problem_logic_scheduling_dummy
    # kwargs={'last_time_hours':30,'demand_p1_kmol':4,'demand_p2_kmol':3}
    # m=model_fun(**kwargs)
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd_aprox(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,model_fun_scheduling,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,rel_tol=gap)
    
    # print('Objective value: ',str(pe.value(m.obj)))
    # print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_variable_tau_enhanced_dbd_aprox_k_2_increased_horizon_from_infeasible.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())




    ## EXPERIMENT 7: GAP 0.05, FROM INFEASIBLE INITIALIZATION, ENHANCED DBD, RIGUROUS SOLUTION OF SUBPROBLEMS, FEASIBILITY AND OPTIMALITY IN INITIALIZATION


    ## EXPERIMENT 8: GAP 0.05, FROM INFEASIBLE INITIALIZATION, ENHANCED DBD, APROX SOLUTION OF SUBPROBLEMS, FEASIBILITY AND OPTIMALITY IN INITIALIZATION


#######-------plots------------------------
    # for I in m.I_reactions:
    #     for J in m.J_reactors:
    #         case=(I,J)
    #         t=[]
    #         c1=[]
    #         c2=[]
    #         c3=[]
    #         Tr=[]
    #         Tj=[]
    #         Fhot=[]
    #         Fcold=[]
    #         for N in m.N[case]:
    #             t.append(N)
    #             Tr.append(m.TRvar[case][N].value)
    #             Tj.append(m.TJvar[case][N].value)
    #             Fhot.append(m.Fhot[case][N].value)
    #             Fcold.append(m.Fcold[case][N].value)
    #             c1.append( m.Cvar[case][N,list(m.Q_balance[I])[0]].value)
    #             c2.append( m.Cvar[case][N,list(m.Q_balance[I])[1]].value)
    #             c3.append( m.Cvar[case][N,list(m.Q_balance[I])[2]].value)
                
                
    #         plt.plot(t, c1,label=list(m.Q_balance[I])[0],color='red')
    #         plt.plot(t, c2,label=list(m.Q_balance[I])[1],color='green')
    #         plt.plot(t, c3,label=list(m.Q_balance[I])[2],color='blue')
    #         plt.xlabel('Time [-]')
    #         plt.ylabel('$Concentration [kmol/m^{3}]$')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()
            
    #         plt.plot(t,Tr,label='T_reactor',color='red')
    #         plt.plot(t,Tj,label='T_jacket',color='blue')
    #         plt.xlabel('Time [-]')
    #         plt.ylabel('Temperature [K]')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()
            
    #         plt.plot(t, Fhot,label='F_hot',color='red')
    #         plt.plot(t,Fcold,label='F_cold',color='blue')
    #         plt.xlabel('Time [-]')
    #         plt.ylabel('Flow rate $[m^{3}/h]$')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()    
            
    # # plot of states
    # for k in m.K:
    #     t_pro=[]
    #     state=[]
    #     for t in m.T:
    #         t_pro.append(m.t_p[t])
    #         state.append(pe.value(m.S[k,t]))

    #     plt.plot(t_pro, state,color='red')
    #     plt.xlabel('Time [h]')
    #     plt.ylabel('State level $[/m^{3}]$')
    #     plt.title('state:'+k)
    #     plt.show()

    #--------------------------------- Gantt plot--------------------------------------------
    fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 62)
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Time [h]')
    gnt.set_ylabel('Units')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45, 55])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['Pack', 'Sep', 'R_small', 'R_large','Mix'])
    
    
    # Setting graph attribute
    gnt.grid(False)
    
    # Declaring bars in schedule
    height=9
    already_used=[]
    for j in m.J:
        if j=='Mix':
            lower_y_position=50
        elif j=='R_large':
            lower_y_position=40    
        elif j=='R_small':
            lower_y_position=30    
        elif j=='Sep':
            lower_y_position=20
        elif j=='Pack':
            lower_y_position=10
        for i in m.I:
            if i=='Mix':
                bar_color='tab:red'
            elif i=='R1':
                bar_color='tab:green'    
            elif i=='R2':
                bar_color='tab:blue'    
            elif i=='R3':
                bar_color='tab:orange' 
            elif i=='Sep':
                bar_color='tab:olive'
            elif i=='Pack1':
                bar_color='tab:purple'                
            elif i=='Pack2':
                bar_color='teal'
            for t in m.T:
                try:
                    if i in m.I_reactions and j in m.J_reactors:
                        if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif pe.value(m.X[i,j,t])==1:
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                                
                    else:
                        if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif pe.value(m.X[i,j,t])==1:
                            gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        
                except:
                    pass 
    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    plt.show()
    # plt.savefig("gantt_minlp.png")
    # plt.savefig("gantt_minlp.svg")   

