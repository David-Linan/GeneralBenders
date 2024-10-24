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
    minlp_solver='dicopt'
    nlp_solver='conopt4'
    mip_solver='cplex'
    gdp_solver='GLOA'
    if minlp_solver=='dicopt':
        # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','maxcycles 20000 \n','$offecho \n']}
        # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 20000 \n','$offecho \n']}  #USED FOR THE DIFFEREN RUNS!!!!!!! because stop 1 decreases time, withouth affecting solution quaility of subproblems.

        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 1 \n','maxcycles 20000 \n','relaxed 0 \n','$offecho \n']}


        # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','nlpsolver '+nlp_solver,'\n','$offecho \n']}
        # sub_options={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','nlpsolver '+nlp_solver+'\n','stop 2 \n','maxcycles 20000 \n','infeasder 1','$offecho \n']}
    elif minlp_solver=='alphaecp':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='antigone':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n','$onecho > antigone.opt \n','nlp_solver '+nlp_solver+'\n','$offecho \n']}
    elif minlp_solver=='baron':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n','$onecho > baron.opt \n','ExtNLPsolver '+nlp_solver+'\n','NLPSol 6 \n','$offecho \n']}
    elif minlp_solver=='knitro':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='lindo':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='sbb':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','GAMS_MODEL.nodlim = 1000000;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='scip':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='shot':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n','$onecho > shot.opt \n','Subsolver.GAMS.NLP.Solver ='+nlp_solver+'\n','$offecho \n']}
    elif minlp_solver=='xpress':
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']}
    elif minlp_solver=='OCTERACT':
        sub_options={'add_options':['GAMS_MODEL.optfile = 0;','Option Threads =0;','Option SOLVER = OCTERACT;']}
    else:
        sub_options={'add_options':['GAMS_MODEL.optfile = 1;','option nlp='+nlp_solver+';\n','option mip='+mip_solver+';\n']} 
    

    ###### test that subproblems are feasible and tehre is an issue with gdp # and minlp solvers
    # start=time.time()
    # model_fun =scheduling_and_control_gdp_N_solvegdp_simpler
    # logic_fun=problem_logic_scheduling
    # kwargs={'x_initial':[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]}
    # m=model_fun(**kwargs)
    # end=time.time()
    # print('model generation time=',str(end-start))
    # ext_ref={m.YR[I,J]:m.ordered_set[I,J] for I in m.I_reactions for J in m.J_reactors}
    # ext_ref.update({m.YR2[I_J]:m.ordered_set2[I_J] for I_J in m.I_J})
    # start=time.time()
    # [reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds]=get_external_information(m,ext_ref,tee=True)
    # end=time.time()
    # print('get info from model time=',str(end-start))
    # start=time.time()
    # m_fixed = external_ref(m=m,x=[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2],extra_logic_function=logic_fun,dict_extvar=reformulation_dict,mip_ref=False,tee=False)
    # end=time.time()
    # print('ext_Ref_required time=',str(end-start))
    # start=time.time()
    # m = solve_subproblem(m=m_fixed,subproblem_solver=minlp_solver,subproblem_solver_options=sub_options,timelimit=100000000,gams_output=False,tee=True,rel_tol=0)
    # end=time.time()
    # print('solve subproblem time=',str(end-start))
    # solved=generate_initialization(m=m,model_name='sequential_iterative14')


    #############Solve with pyomo.GDP COMPLETE GDP
    # kwargs={'x_initial':[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]}
    # model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    # m=model_fun(**kwargs)
    # solvers=minlp_solver+'_'+nlp_solver+'_'+mip_solver+'_'+gdp_solver
    # name='Results_variable_tau_gdp_complete_'+solvers+'.txt'
    # m = solve_with_gdpopt(m, mip=mip_solver,minlp=minlp_solver,nlp=nlp_solver,minlp_options=sub_options, timelimit=50000,strategy=gdp_solver, mip_output=True, nlp_output=True,minlp_output=True,rel_tol=0,tee=True)

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open(name, 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

    # Solve with MINLP
    kwargs={'x_initial':[4,4,5,5,3,3,3,2,2,3,3,2,2,2,3,2]}
    # kwargs={'x_initial':[4, 4, 5, 5, 3, 3, 4, 3, 3, 5, 5, 5, 4, 6, 5, 7],'last_time_hours':28,'demand_p1_kmol':2,'demand_p2_kmol':2}
    model_fun=scheduling_and_control_gdp_N_solvegdp_simpler
    m=model_fun(**kwargs) #do not provide init
    m=initialize_model(m,from_feasible=True,feasible_model='sequential_iterative14')#provide init
    solvers=minlp_solver+'_'+nlp_solver+'_'+mip_solver
    name='Results_variable_tau_minlp_complete_bigm_'+solvers+'_scheduling14.txt'
    m = solve_with_minlp(m,transformation='bigm',minlp=minlp_solver,minlp_options=sub_options,timelimit=50000,gams_output=False,tee=True,rel_tol=0)

    # textbuffer = io.StringIO()
    # for v in m.component_objects(pe.Var, descend_into=True):
    #     v.pprint(textbuffer)
    #     textbuffer.write('\n')
    # textbuffer.write('\n Objective: \n') 
    # textbuffer.write(str(pe.value(m.obj)))    
    # with open(name, 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())   
####--------Objective function summary---------------------------------
    TPC1=sum(sum(sum(  m.fixed_cost[I,J]*pe.value(m.X[I,J,T]) for J in m.J)for I in m.I)for T in m.T)
    TPC2=sum(sum(sum( m.variable_cost[I,J]*pe.value(m.B[I,J,T]) for J in m.J_noDynamics) for I in m.I_noDynamics) for T in m.T)
    TPC3=sum(sum(sum(pe.value(m.X[I,J,T])*(m.hot_cost*pe.value(m.Integral_hot[I,J][m.N[I,J].last()])   +  m.cold_cost*pe.value(m.Integral_cold[I,J][m.N[I,J].last()])  ) for T in m.T) for I in m.I_reactions)for J in m.J_reactors)
    TMC=sum( m.raw_cost[K]*(m.S0[K]-pe.value(m.S[K,m.lastT])) for K in m.K_inputs)
    SALES=sum( m.revenue[K]*pe.value(m.S[K,m.lastT])  for K in m.K_products)
    OBJVAL=(TPC1+TPC2+TPC3+TMC-SALES)/100
    print('FC: Fixed costs for all unit-tasks: ',str(TPC1))   
    print('VC1: Variable cost for unit-tasks that do not consider dynamics: ', str(TPC2))
    print('VC2: Variable cost for unit-tasks that consider dynamics: ',str(TPC3))
    print('TMC: Total material cost: ',str(TMC))
    print('SALES: Revenue form selling products: ',str(SALES))
    print('OBJ:',str(OBJVAL))
    print('----')
    # print('TCP1 gams:',str(pe.value(m.TCP1)))
    # print('TCP2 gams:',str(pe.value(m.TCP2)))
    # print('TCP3 gams:',str(pe.value(m.TCP3)))
    # print('TMC gams:',str(pe.value(m.TMC)))
    # print('SALES gams:',str(pe.value(m.SALES)))



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

