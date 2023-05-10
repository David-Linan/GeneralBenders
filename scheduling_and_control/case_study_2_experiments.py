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
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)




    obj_Selected='profit_max'
    neighdef='2'

    initialization=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    # initialization=[1, 1, 1, 1, 1, 1, 1, 1]

    # # CPLEX SOLUTION   
    mip_solver='CPLEX'
    minlp_solver='DICOPT'
    nlp_solver='conopt4'
    transform='bigm'
    last_disc=12
    last_time_h=12
    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=2;','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}

    sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.threads=2;','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver,'\n','$offecho \n','option mip='+mip_solver+';\n']}
    LO_PROC_TIME={('T1','U1'):0.5,('T2','U2'):0.1,('T2','U3'):0.1,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}
    UP_PROC_TIME={('T1','U1'):0.5,('T2','U2'):2,('T2','U3'):2,('T3','U2'):1,('T3','U3'):2.5,('T4','U2'):1,('T4','U3'):5,('T5','U4'):1.5}

    m=case_2_scheduling_control_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=last_disc,last_time_hours=last_time_h,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I for J in m.J if m.I_i_j_prod[I,J]==1}
    ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    start=time.time()
    m=solve_with_minlp(m,transformation=transform,minlp=minlp_solver,minlp_options=sub_options,timelimit=86400,gams_output=False,tee=True,rel_tol=0)
    end=time.time()
    #SAVE INITIALIZATION
    save=generate_initialization(m=m,model_name='case_study_2_feas_sol') 
    # UPDATE MODEL WITH INITIALZIATION
    m=case_2_scheduling_control_gdp_var_proc_time(x_initial=initialization,obj_type=obj_Selected,last_disc_point=last_disc,last_time_hours=last_time_h,lower_t_h=LO_PROC_TIME,upper_t_h=UP_PROC_TIME)
    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    transformation_string = 'gdp.'+transform
    pe.TransformationFactory(transformation_string).apply_to(m)
    m=initialize_model(m,from_feasible=True,feasible_model='case_study_2_feas_sol')   

    Sol_found=[]
    for I in m.I:
        for J in m.J:
            if m.I_i_j_prod[I,J]==1:
                for K in m.ordered_set[I,J]:
                    if round(pe.value(m.YR_disjunct[I,J][K].indicator_var))==1:
                        Sol_found.append(K-m.minTau[I,J]+1)
    for I_J in m.I_J:
        Sol_found.append(1+round(pe.value(m.Nref[I_J])))

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


    TPC1=pe.value(m.TCP1)
    TPC2=pe.value(m.TCP2)
    TPC3=pe.value(m.TCP3)
    TMC=pe.value(m.TMC)
    SALES=pe.value(m.SALES)
    OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)
    print('TPC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('TPC: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('TPC: Variable cost for unit-tasks that do consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJ:',str(OBJVAL))



 
#######-------plots------------------------
    for I in m.I_dynamics:
        for J in m.J_dynamics:
            for T in m.T:
                if pe.value(m.X[I,J,T])==1: 
                    case=(I,J,T)
                    t=[]
                    CA=[]
                    CB=[]
                    CC=[]
                    Tr=[]
                    Tj=[]
                    Fhot=[]
                    Fcold=[]
                    u_input=[]
                    for N in m.N[case]:
                        t.append(N*m.varTime[case].value)
                        Tr.append(m.TRvar[case][N].value)
                        Tj.append(m.TJvar[case][N].value)
                        Fhot.append(m.Fhot[case][N].value)
                        Fcold.append(m.Fcold[case][N].value)
                        CA.append( m.CA[case][N].value)
                        CB.append( m.CB[case][N].value)
                        CC.append( m.CC[case][N].value)
                        u_input.append(m.u_input[case][N].value)
                        
                        
                    plt.plot(t, CA,label='CA',color='red')
                    plt.plot(t, CB,label='CB',color='green')
                    plt.plot(t, CC,label='CC',color='blue')
                    plt.xlabel('Time [h]')
                    plt.ylabel('$Concentration [kmol/m^{3}]$')
                    title=case[0]+' in '+case[1]+' Concentration'
                    plt.title(case[0]+' in '+case[1]+' at '+m.t_p[T]+' h')
                    plt.legend()
                    plt.show()
                    # plt.savefig("figures/"+title+".svg") 
                    plt.clf()
                    plt.cla()
                    plt.close()

                    plt.plot(t,Tr,label='T_reactor',color='red')
                    plt.plot(t,Tj,label='T_jacket',color='blue')
                    plt.xlabel('Time [h]')
                    plt.ylabel('Temperature [K]')
                    title=case[0]+' in '+case[1]+' Temperature'
                    plt.title(case[0]+' in '+case[1]+' at '+m.t_p[T]+' h')
                    plt.legend()
                    plt.show()
                    # plt.savefig("figures/"+title+".svg") 
                    plt.clf()
                    plt.cla()
                    plt.close()
                    
                    plt.plot(t, Fhot,label='F_hot',color='red')
                    plt.plot(t,Fcold,label='F_cold',color='blue')
                    plt.xlabel('Time [h]')
                    plt.ylabel('Flow rate $[m^{3}/h]$')
                    title=case[0]+' in '+case[1]+' Flow rate'
                    plt.title(case[0]+' in '+case[1]+' at '+m.t_p[T]+' h')
                    plt.legend()
                    plt.show()    
                    # plt.savefig("figures/"+title+".svg") 
                    plt.clf()
                    plt.cla()
                    plt.close()

                    plt.plot(t, u_input,color='red')
                    plt.xlabel('Time [h]')
                    plt.ylabel('Flow rate of B $[m^{3}/h]$')
                    title=case[0]+' in '+case[1]+' Flow rate'
                    plt.title(case[0]+' in '+case[1]+' at '+m.t_p[T]+' h')
                    plt.show()    
                    # plt.savefig("figures/"+title+".svg") 
                    plt.clf()
                    plt.cla()
                    plt.close()
    # plot of states
    for k in m.K:
        t_pro=[]
        state=[]
        for t in m.T:
            t_pro.append(m.t_p[t])
            state.append(pe.value(m.S[k,t]))

        plt.plot(t_pro, state,color='red')
        plt.xlabel('Time [h]')
        plt.ylabel('State level $[m^{3}]$')
        title='state '+k
        plt.title(title)
        plt.show()
        # plt.savefig("figures/"+title+".svg") 
        plt.clf()
        plt.cla()
        plt.close()

    #--------------------------------- Gantt plot--------------------------------------------
    fig, gnt = plt.subplots(figsize=(11, 5), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 52) #TODO: change depending case study
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Time [h]')
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
                    if i in m.I_dynamics and j in m.J_dynamics:
                        if pe.value(m.X[i,j,t])==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                            already_used.append(i)
                        elif pe.value(m.X[i,j,t])==1:
                            gnt.broken_barh([(m.t_p[t], m.varTime[i,j,t].value)], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                            gnt.annotate("{:.2f}".format(m.B[i,j,t].value),xy=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+m.varTime[i,j,t].value)/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                                                
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
    # plt.savefig("figures/gantt_minlp.svg")   
    plt.clf()
    plt.cla()
    plt.close()



