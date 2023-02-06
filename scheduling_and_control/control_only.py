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
from Reactors_dynamics import  reactor_dynamics, reactor_dynamics_verif_luis
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    #Solver declaration
    nlp_solver='conopt4'
    mip_solver='cplex'
    gdp_solver='LOA'


    #Minimum processing times
    m=reactor_dynamics()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    gams_path = os.path.join(dir_path, "gamsfiles_dynamics/")
    if not(os.path.exists(gams_path)):
        print('Directory for automatically generated files ' + gams_path + ' does not exist. We will create it')
        os.makedirs(gams_path)
    opt1 = SolverFactory('gams')
    sub_options=['option nlp='+nlp_solver+';\n','GAMS_MODEL.optfile=1; \n','$onecho > '+nlp_solver+'.opt \n','$offecho \n']
    results = opt1.solve(m, solver=nlp_solver, tee=True,add_options=sub_options,keepfiles=True,tmpdir=gams_path,symbolic_solver_labels=True)    

    textbuffer = io.StringIO()
    for v in m.component_objects(pe.Var, descend_into=True):
        v.pprint(textbuffer)
        textbuffer.write('\n')
    textbuffer.write('\n Objective: \n') 
    textbuffer.write(str(pe.value(m.obj)))    
    with open('Results_dynamic_min_times.txt', 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())  

    #Minimum processing times
    m=reactor_dynamics_verif_luis()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    gams_path = os.path.join(dir_path, "gamsfiles_dynamics/")
    if not(os.path.exists(gams_path)):
        print('Directory for automatically generated files ' + gams_path + ' does not exist. We will create it')
        os.makedirs(gams_path)
    opt1 = SolverFactory('gams')
    sub_options=['option nlp='+nlp_solver+';\n','GAMS_MODEL.optfile=1; \n','$onecho > '+nlp_solver+'.opt \n','$offecho \n']
    results = opt1.solve(m, solver=nlp_solver, tee=True,add_options=sub_options,keepfiles=True,tmpdir=gams_path,symbolic_solver_labels=True)    

    textbuffer = io.StringIO()
    for v in m.component_objects(pe.Var, descend_into=True):
        v.pprint(textbuffer)
        textbuffer.write('\n')
    textbuffer.write('\n Objective: \n') 
    textbuffer.write(str(pe.value(m.obj)))    
    with open('Results_dynamic_min_times_verif_luis.txt', 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())   

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
                t.append(N*pe.value(m.varTime[I,J]))
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
            plt.xlabel('Time [h]')
            plt.ylabel('$Concentration [kmol/m^{3}]$')
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            plt.show()
            
            plt.plot(t,Tr,label='T_reactor',color='red')
            plt.plot(t,Tj,label='T_jacket',color='blue')
            plt.xlabel('Time [h]')
            plt.ylabel('Temperature [K]')
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            plt.show()
            
            plt.plot(t, Fhot,label='F_hot',color='red')
            plt.plot(t,Fcold,label='F_cold',color='blue')
            plt.xlabel('Time [h]')
            plt.ylabel('$Flow rate [m^{3}/h]$')
            plt.title(case[0]+' in '+case[1])
            plt.legend()
            plt.show()    


#-----------------------------------------------------------------

    # #Verification of dynamics
    # m=reactor_dynamics_verif()
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    # gams_path = os.path.join(dir_path, "gamsfiles_dynamics/")
    # if not(os.path.exists(gams_path)):
    #     print('Directory for automatically generated files ' + gams_path + ' does not exist. We will create it')
    #     os.makedirs(gams_path)
    # opt1 = SolverFactory('gams')
    # sub_options=['option nlp='+nlp_solver+';\n','GAMS_MODEL.optfile=1; \n','$onecho > '+nlp_solver+'.opt \n','$offecho \n']
    # results = opt1.solve(m, solver=nlp_solver, tee=True,add_options=sub_options,keepfiles=True,tmpdir=gams_path,symbolic_solver_labels=True)    

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open('Results_dynamic_verification.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())   

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
    #             t.append(N*pe.value(m.varTime[I,J]))
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
    #         plt.xlabel('Time [h]')
    #         plt.ylabel('$Concentration [kmol/m^{3}]$')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()
            
    #         plt.plot(t,Tr,label='T_reactor',color='red')
    #         plt.plot(t,Tj,label='T_jacket',color='blue')
    #         plt.xlabel('Time [h]')
    #         plt.ylabel('Temperature [K]')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()
            
    #         plt.plot(t, Fhot,label='F_hot',color='red')
    #         plt.plot(t,Fcold,label='F_cold',color='blue')
    #         plt.xlabel('Time [h]')
    #         plt.ylabel('$Flow rate [m^{3}/h]$')
    #         plt.title(case[0]+' in '+case[1])
    #         plt.legend()
    #         plt.show()    
