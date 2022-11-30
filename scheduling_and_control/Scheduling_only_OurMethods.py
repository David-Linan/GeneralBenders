from __future__ import division
import sys
# sys.path.insert(0, '/home/dadapy/GeneralBenders/')
# sys.path.append('C:/Users/TEMP/Desktop/GeneralBenders/') #for LRLAB5
sys.path.append('C:/Users/dlinanro/Desktop/GeneralBenders/') #for LRSRV1
from functions.d_bd_functions import run_function_dbd
from functions.dsda_functions import get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_gdpopt,solve_with_minlp
import pyomo.environ as pe
from pyomo.gdp import Disjunct, Disjunction
import math
from pyomo.opt.base.solvers import SolverFactory
import io
import time
from functions.dsda_functions import neighborhood_k_eq_2,get_external_information,external_ref,solve_subproblem,generate_initialization,initialize_model,solve_with_dsda
import logging
# from Scheduling_control_variable_tau_model_reduced import scheduling_and_control,problem_logic_scheduling
from Scheduling_only import scheduling as scheduling_GDP 
from Scheduling_only import problem_logic_scheduling
from Reactors_dynamics import  reactor_dynamics
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    #Solver declaration
    nlp_solver='conopt4'
    mip_solver='cplex'
    gdp_solver='LOA'

    # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option optcr=0;\n','option optca=0;\n','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 2000 \n','$offecho \n']}
    sub_options={}

    # #Solve with LD-SDA
    model_fun =scheduling_GDP
    logic_fun=problem_logic_scheduling
    kwargs={}
    m=model_fun(**kwargs)
    ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    m,routeDSDA,obj_route=solve_with_dsda(model_fun,kwargs,[1,1,1,1,1,1],ext_ref,logic_fun,k = 'Infinity',provide_starting_initialization= False,feasible_model='dsda',subproblem_solver = mip_solver,subproblem_solver_options=sub_options,iter_timelimit= 100000,timelimit = 360000,gams_output = False,tee= True,global_tee = True,rel_tol = 0)
    print('Objective value: ',str(pe.value(m.obj)))

    textbuffer = io.StringIO()
    for v in m.component_objects(pe.Var, descend_into=True):
        v.pprint(textbuffer)
        textbuffer.write('\n')
    textbuffer.write('\n Objective: \n') 
    textbuffer.write(str(pe.value(m.obj)))    
    with open('Results_schedling_only_dsda.txt', 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())

    #Solve with pyomo.GDP
    # kwargs={}
    # model_fun=scheduling_GDP 
    # m=model_fun(**kwargs)
    # m = solve_with_gdpopt(m, mip=mip_solver,minlp=minlp_solver,nlp=nlp_solver,minlp_options=sub_options, timelimit=1000,strategy=gdp_solver, mip_output=False, nlp_output=False,rel_tol=0,tee=True)

    #Solve with MINLP
    # kwargs={}
    # model_fun=scheduling_GDP
    # m=model_fun(**kwargs)
    # m = solve_with_minlp(m, transformation='bigm', minlp='sbb', minlp_options=sub_options,gams_output=False,tee=True,rel_tol=0)
    #--- Dynamic model plots
    # ---Results to txt


    #--- Gantt plot
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
    
    
    #Verify reactor dynamics
    m=reactor_dynamics()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    gams_path = os.path.join(dir_path, "gamsfiles/")
    if not(os.path.exists(gams_path)):
        print('Directory for automatically generated files ' + gams_path + ' does not exist. We will create it')
        os.makedirs(gams_path)
    opt1 = SolverFactory('gams')
    sub_options=['option nlp=conopt4;\n','GAMS_MODEL.optfile=1; \n','$onecho > conopt4.opt \n','$offecho \n']
    results = opt1.solve(m, solver='dicopt', tee=True,add_options=sub_options,keepfiles=True,tmpdir=gams_path,symbolic_solver_labels=True)    
    

    for I in m.I_reactions:
        for J in m.J_reactors:
            case=(I,J)
            t=[]
            c1=[]
            c2=[]
            c3=[]
            Tr=[]
            Tj=[]
            Fhot=[]
            Fcold=[]
            for N in m.N[case]:
                t.append(N)
                Tr.append(m.TRvar[case][N].value)
                Tj.append(m.TJvar[case][N].value)
                Fhot.append(m.Fhot[case][N].value)
                Fcold.append(m.Fcold[case][N].value)
                c1.append( m.Cvar[case][N,list(m.Q_balance[I])[0]].value)
                c2.append( m.Cvar[case][N,list(m.Q_balance[I])[1]].value)
                c3.append( m.Cvar[case][N,list(m.Q_balance[I])[2]].value)
                
                
            plt.plot(t, c1,label=list(m.Q_balance[I])[0],color='red')
            plt.plot(t, c2,label=list(m.Q_balance[I])[1],color='green')
            plt.plot(t, c3,label=list(m.Q_balance[I])[2],color='blue')
            plt.xlabel('Time []')
            plt.ylabel('$Concentration [kmol/m^{3}]$')
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            plt.show()
            
            plt.plot(t,Tr,label='T_reactor',color='red')
            plt.plot(t,Tj,label='T_jacket',color='blue')
            plt.xlabel('Time []')
            plt.ylabel('Temperature [K]')
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            plt.show()
            
            plt.plot(t, Fhot,label='F_hot',color='red')
            plt.plot(t,Fcold,label='F_cold',color='blue')
            plt.xlabel('Time []')
            plt.ylabel('$Flow rate [m^{3}/h]$')
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            plt.show()    
            
    