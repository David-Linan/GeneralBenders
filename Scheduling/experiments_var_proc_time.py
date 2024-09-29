from __future__ import division
from pickle import TRUE

import sys
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd,run_function_dbd_scheduling_cost_min_ref_2
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_minlp,solve_with_gdpopt
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_all,neighborhood_k_eq_l_natural,neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda,external_ref_neighborhood_general
import logging
from scheduling_formulation_variable_proc_time import scheduling_gdp_var_proc_time,problem_logic_scheduling
import os
import matplotlib.pyplot as plt

def generate_plot(m):

    #--------------------------------- Gantt plot--------------------------------------------
    fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 52) #TODO: change depending case study
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value*60*60)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Time [s]')
    gnt.set_ylabel('Units')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45]) #TODO: change depending case study
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['U4', 'U3', 'U2', 'U1']) #TODO: change depending case study
    
    
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
                try:

                    if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                        gnt.broken_barh([(m.t_p[t]*60*60, m.varTime[i,j,t].value*60*60)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                        gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]*60*60+m.varTime[i,j,t].value*60*60)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]*60*60+m.varTime[i,j,t].value*60*60)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                        already_used.append(i)
                    elif round(pe.value(m.X[i,j,t]))==1:
                        gnt.broken_barh([(m.t_p[t]*60*60, m.varTime[i,j,t].value*60*60)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                        gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]*60*60+m.varTime[i,j,t].value*60*60)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]*60*60+m.varTime[i,j,t].value*60*60)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                                              

                except:
                    pass 
    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    # plt.legend()
    # plt.show()
    plt.savefig("Scheduling/CSCHE2024/figure_"+str(param)+".svg")   
    plt.clf()
    plt.cla()
    plt.close()

    



if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)


    obj_Selected='profit_max'
    neighdef='2' #for DSDA and DSDSA only
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
    UP_PROC_TIME={('T1','U1'):20,('T2','U2'):20,('T2','U3'):20,('T3','U2'):20,('T3','U3'):20,('T4','U2'):20,('T4','U3'):20,('T5','U4'):20}
    last_time_hours_=20


    # EXPERIMENTS
    Naive_cplex_experiment=True
    D_SDA_and_DSDSA_experiments=False
    CG_DSDA_experiment=True

    #GENERATE PLOT
    generate_CG_DSDA_plot=True

    first=2
    last=20#500

    for param in range(first,last):
        print('\n------------ num discrete points: ',param,' -------------------------------')
        m=scheduling_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=param,last_time_hours=last_time_hours_,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
        ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
        # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
        [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)

        if Naive_cplex_experiment:
            start=time.time()
            m=solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
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

            print('Objective CPLEX=',pe.value(m.obj),'best CPLEX=',Sol_found,'cputime CPLEX=',str(end-start))

            textbuffer = io.StringIO()
            for v in m.component_objects(pe.Var, descend_into=True):
                v.pprint(textbuffer)
                textbuffer.write('\n')
            textbuffer.write('\n Objective: \n') 
            textbuffer.write(str(pe.value(m.obj)))  
            file_name='Results_var_proc_time.txt'  
            with open(os.path.join('C:/Users/dlinanro/Desktop/GeneralBenders/Scheduling',file_name), 'w') as outputfile:
                outputfile.write(textbuffer.getvalue())
        
            for I_J in m.I_J:
                I=I_J[0]
                J=I_J[1]
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        variable_bound_found=K*m.delta
                # print(I_J)
                # if sum(pe.value(m.X[I,J,T]) for T in m.T)>=1:
                #     print('maximum variable time [h]: ',max(pe.value(m.varTime[I,J,T]) for T in m.T if round(pe.value(m.X[I,J,T]))==1),'<= current bound (based on discretization) [h]: ',variable_bound_found,'<= initial bound (user) [h]: ',UP_PROC_TIME[(I,J)])

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
        kwargs={'obj_type':obj_Selected,'last_disc_point':par_points,'last_time_hours':last_time_hours_,'lower_t_h':LO_PROC_TIME,'upper_t_h':UP_PROC_TIME}


        if D_SDA_and_DSDSA_experiments:
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



        if CG_DSDA_experiment:
            current_central=initialization #initialization of external variables
            original_m=model_fun(**kwargs) # declaration of original GDP model
            max_iter_out=1000000 # Maximum number of iterations
            upper_evaluated={} #evaluated points
            old_obj=1e+10
            teed=True

            # CG_DSDA algorithm
            start=time.time()
            for out in range(max_iter_out):
                if teed:
                    print('------Outer iteration= ',out+1)
                upper_evaluated[out+1]=[current_central] #Update evaluated points with current external variable being evaluated
                
                # perform the reformulation
                for I in original_m.I:
                    for J in original_m.J:
                        if  original_m.I_i_j_prod[I,J]==1:
                            for ind in original_m.ordered_set[I,J]:
                                disjunct=original_m.YR_disjunct[I,J][ind]
                                disjunct.activate()
                original_m = external_ref_neighborhood_general(m=original_m,x=current_central,extra_logic_function=logic_fun,dict_extvar=reformulation_dict)
                # TODO: had to do this outside reformulation function. Generalize!!
                for I in original_m.I:
                    for J in original_m.J:
                        if  original_m.I_i_j_prod[I,J]==1:
                            for ind in original_m.ordered_set[I,J]:
                                disjunct=original_m.YR_disjunct[I,J][ind]
                                if original_m.YR[I,J][ind].is_fixed() and original_m.YR[I,J][ind].value==False:
                                    # for constr in disjunct.component_objects(pe.Constraint, descend_into=True):
                                    # #print(constr.name) #not name, but local_name
                                    # #print(constr.local_name) #not name, but local_name
                                    # # if constr.local_name=='DEF_VAR_TIME':
                                    #     constr.deactivate()
                                    
                                    disjunct.deactivate()
                                    # disjunct.pprint()
                                    # print(disjunct.name,' deactivated') 

                # solve the problem
                m=original_m.clone() #initialize mip problem
                # m=solve_with_gdpopt(m,mip=mip_solver,mip_options=sub_options,timelimit=86400,rel_tol=0,strategy='LOA',tee=True)
                m =solve_with_minlp(m,transformation='bigm',minlp=mip_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=False,rel_tol=0) 
                
                # Extract solution
                Sol_found=[]

                # TODO: Generalize this part
                for I in m.I:
                    for J in m.J:
                        if  m.I_i_j_prod[I,J]==1:
                            for ind in m.ordered_set[I,J]:
                                if(round(pe.value(m.YR_disjunct[I,J][ind].binary_indicator_var))==1):
                                    Sol_found.append(ind)
                                    break
                if teed:
                    print(Sol_found)

                # for v in m.component_data_objects(ctype=pe.Var):
                #     if v.parent_component().name!='X' and v.is_binary()==True:
                #         print(v.parent_component().name,'=',pe.value(v))
                #         # v.pprint()
                #         # for vv in v.items():
                #         #     vv.pprint()
            

                # Generate search direction
                direction=[]
                for i in range(len(Sol_found)):
                    direction.append(Sol_found[i]-current_central[i])              

                if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger': 
                    m.mip_status='Infeasible'
                    #TODO: IN CASE OF INFEASIBILITIES, I should declare infinity objective, however, it is not possible because new neighborhood always have the previous feasible solution in it.
                else:
                    m.mip_status='Optimal'
                if teed:
                    if m.mip_status == 'Optimal':  

                        print('   Evaluated:', Sol_found, '   |   Objective:', round(pe.value(m.obj), 5), '   |   Global Time:', round(time.time()- start, 2))
                                    
                    else:
                        print('   Evaluated infeasible:', Sol_found, '   |   Objective: -    |   Global Time:', round(time.time()- start, 2))        


                    print('   SEARCH DIRECTION: ', direction)                
                # incumbent remains unchanged or objective stops improving?. First one is necessary to finish in the first iteration if it returns central incumbent solution
                if round(sum(abs(j) for j in direction))==0 or old_obj<=round(pe.value(m.obj)):    
                    end=time.time()
                    print('Objective CG-DSDA='+str(pe.value(m.obj))+', best CG-SDSA='+str(Sol_found),'cputime CG-DSDA= '+str(end-start))  
                    if generate_CG_DSDA_plot:
                        generate_plot(m)
                    break
                else:
                    # update objective and variables
                    old_obj=round(pe.value(m.obj))
                    current_central=Sol_found
                    # delete most recently solved mip
                    del m
                    #update









    