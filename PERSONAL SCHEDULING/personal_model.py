from __future__ import division
import pyomo.environ as pe
import pyomo.dae as dae
import math
import os
import io
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory
from pyomo.gdp import Disjunct, Disjunction
import itertools
import random
random.seed(10)


def personal_model():

    # ------------pyomo model------------------------------------------------
    #------------------------------------------------------------------------

    m = pe.ConcreteModel(name='personal_model')

    # ------------scalars    ------------------------------------------------   
    m.delta=pe.Param(initialize=1,doc='lenght of time periods of discretized time grid for scheduling [days]')
    m.lastT=pe.Param(initialize=7,doc='last discrete time value in the scheduling time grid')
    
    # -----------sets--------------------------------------------------------
    #Main sets
    m.T=pe.RangeSet(0,m.lastT,1,doc='Discrete time set')
    m.J=pe.Set(initialize=['M1','M2','D1','D2','N1','N2'],doc='Set of Units')
    m.I=pe.Set(initialize=['Diana','David','Carlos','Maria','Lukas','Felipe','Santiago'], doc='Set of tasks')

    #Subsets
    m.I_relief=pe.Set(initialize=['Diana','David','Carlos','Maria'],within=m.I)  
    m.I_staff=pe.Set(initialize=['Lukas','Felipe','Santiago'],within=m.I)  
    m.J_morning=pe.Set(initialize=['M1','M2'],within=m.J)
    m.J_day=pe.Set(initialize=['D1','D2'],within=m.J)
    m.J_night=pe.Set(initialize=['N1','N2'],within=m.J)
    m.J_no_duplicates_1=pe.Set(initialize=['M1','D1','N1'],within=m.J)
    m.J_no_duplicates_2=pe.Set(initialize=['M2','D2','N2'],within=m.J)
    # -----------variables--------------------------------------------------------
    m.X=pe.Var(m.I,m.J,m.T,initialize=0,within=pe.Binary,doc='1 if unit j processes task i starting at time t')  
    m.rate_relief=pe.Var(initialize=0,within=pe.NonNegativeReals,bounds=(0,1),doc='this is maximized. All relief staff should get a percentage of assigned shifts above this range') 

    # -----------constraints--------------------------------------------------------
    m.t_p=pe.Param(m.T,initialize=[j for j in m.T],doc='physical time [units of time]')
    m.tau=pe.Param(m.I,m.J,initialize=1,doc="Processing time with respect to the time grid: how many grid spaces do I need for the task ?")
    m.tau_p=pe.Param(m.I,m.J,initialize=m.tau,mutable=True,default=0,doc="Physical processing time for tasks [units of time]")

    def _E1_UNIT(m,J,T):
        return sum(sum(m.X[I,J,TP] for TP in m.T if TP<=T and TP>=T-m.tau[I,J]+1) for I in m.I ) <=  1
    m.E1_UNIT=pe.Constraint(m.J,m.T,rule=_E1_UNIT,doc='UNIT UTILIZATION')

    def _morning(m,I,T):
        return sum(m.X[I,J,T] for J in m.J_morning)<=1
    m.morning=pe.Constraint(m.I,m.T,rule=_morning,doc='a person cannot be duplicated')

    def _day(m,I,T):
        return sum(m.X[I,J,T] for J in m.J_day)<=1
    m.day=pe.Constraint(m.I,m.T,rule=_day,doc='a person cannot be duplicated')

    def _night(m,I,T):
        return sum(m.X[I,J,T] for J in m.J_night)<=1
    m.night=pe.Constraint(m.I,m.T,rule=_night,doc='a person cannot be duplicated')
    # ---------Objective------------------------------------------------------------
    def _obj(m):
        return sum(sum(sum(m.X[I,J,T] for T in m.T) for I in m.I) for J in m.J)+m.rate_relief
    m.obj = pe.Objective(rule=_obj, sense=pe.maximize)           

    return m



if __name__ == "__main__":


    kwargs={}
    m_fun=personal_model
    m=m_fun(**kwargs)


    # STAFF WITH FIXED SCHEDULE. SIMPLY FIX THE VARIABLES RELATED TO THE SCHEDULE OF FIXED STALL

    for T in m.T:
        for I in m.I_staff:    
            for J in m.J:
                if (sum(round(pe.value(m.X[IP,J,T])) for IP in m.I)>=1 \
                    or sum(round(pe.value(m.X[I,JP,T])) for JP in m.J_morning)>=1\
                    or sum(round(pe.value(m.X[I,JP,T])) for JP in m.J_day)>=1\
                    or sum(round(pe.value(m.X[I,JP,T])) for JP in m.J_night)>=1\
                    or sum(sum(sum(round(pe.value(m.X[IP,JP,TP])) for IP in m.I) for JP in m.J) for TP in m.T)>=0.5*m.T.__len__()*m.J.__len__()):

                    m.X[I,J,T].fix(0)
                else:
                    m.X[I,J,T].fix(round(random.uniform(0, 1)))

    #--------------------------------- Gantt plot--------------------------------------------
    num_units=6
    fig, gnt = plt.subplots(figsize=(11, num_units), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 72)
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Day')
    gnt.set_ylabel('Shift')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45, 55, 65])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['Night 2', 'Night 1', 'Day 2', 'Day 1','Morning 2','Morning 1'])

    # Setting graph attribute
    gnt.grid(False)
    
    # Declaring bars in schedule
    height=9
    already_used=[]
    for j in m.J:
        if j=='M1':
            lower_y_position=60
        elif j=='M2':
            lower_y_position=50    
        elif j=='D1':
            lower_y_position=40    
        elif j=='D2':
            lower_y_position=30
        elif j=='N1':
            lower_y_position=20
        elif j=='N2':
            lower_y_position=10    
        for i in m.I:
            if i=='Diana':
                bar_color='tab:red'
            elif i=='David':
                bar_color='tab:green'    
            elif i=='Carlos':
                bar_color='tab:blue'    
            elif i=='Maria':
                bar_color='tab:orange' 
            elif i=='Lukas':
                bar_color='tab:olive'
            elif i=='Felipe':
                bar_color='tab:purple'                
            elif i=='Santiago':
                bar_color='teal'
            for t in m.T:
                try:
                    if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                        gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                        gnt.annotate(i,xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                        already_used.append(i)
                    elif round(pe.value(m.X[i,j,t]))==1:
                        gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                        gnt.annotate(i,xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        
                except:
                    pass 
    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    plt.show()



    # AVAILABILITY

    # NOTE: users declare the availability for m.J_no_duplicates_1 
    def _avail(m,I_refief,J,T):
        if J in  m.J_no_duplicates_1:
            if sum(round(pe.value(m.X[IP,J,T])) for IP in m.I_staff)>=1: # If staff is assigned, do not declare availability
                return 0
            else:
                return round(random.uniform(0, 1))
        else:
            return 0
    m.avail=pe.Param(m.I_relief,m.J,m.T,mutable=True,within=pe.Reals,initialize=_avail,default=0,doc='If staff member I is available for shift J at time T')


    # NOTE: users declare the availability for m.J_no_duplicates_1, then, I use this info to replicate availability in m.J_no_duplicates_2
    for I in m.I_relief:
        for T in m.T:
                m.avail[I,'M2',T]=pe.value(m.avail[I,'M1',T])
                m.avail[I,'D2',T]=pe.value(m.avail[I,'D1',T])
                m.avail[I,'N2',T]=pe.value(m.avail[I,'N1',T])

    m.avail.pprint()


    # CONSTRAINTS FOR RELIEF STAFF
    def _avail_const(m,I_refief,J,T):
        return m.X[I_refief,J,T] <= pe.value(m.avail[I_refief,J,T])
    m.avail_const=pe.Constraint(m.I_relief,m.J,m.T,rule=_avail_const)

    def _rate_const(m,I_refief):
        return sum(sum(m.X[I_refief,J,T] for J in m.J) for T in m.T)>= m.rate_relief*(      sum(sum(pe.value(m.avail[I_refief,J,T]) for J in m.J_no_duplicates_1) for T in m.T)      )
    m.rate_const=pe.Constraint(m.I_relief,rule=_rate_const)

    mip_solver='CPLEX'
    solver = pe.SolverFactory('gams', solver=mip_solver)
    res = solver.solve(m, tee=False)

    print('All relief staff members are getting at least ',pe.value(m.rate_relief)*100,'% of the shifts they declared as available. Higher percentages are infeasible')
    #--------------------------------- Gantt plot--------------------------------------------
    num_units=6
    fig, gnt = plt.subplots(figsize=(11, num_units), sharex=True, sharey=False)
    # Setting Y-axis limits
    gnt.set_ylim(8, 72)
    
    # Setting X-axis limits
    gnt.set_xlim(0, m.lastT.value*m.delta.value)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Day')
    gnt.set_ylabel('Shift')
    
    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35, 45, 55, 65])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['Night 2', 'Night 1', 'Day 2', 'Day 1','Morning 2','Morning 1'])

    # Setting graph attribute
    gnt.grid(False)
    
    # Declaring bars in schedule
    height=9
    already_used=[]
    for j in m.J:
        if j=='M1':
            lower_y_position=60
        elif j=='M2':
            lower_y_position=50    
        elif j=='D1':
            lower_y_position=40    
        elif j=='D2':
            lower_y_position=30
        elif j=='N1':
            lower_y_position=20
        elif j=='N2':
            lower_y_position=10    
        for i in m.I:
            if i=='Diana':
                bar_color='tab:red'
            elif i=='David':
                bar_color='tab:green'    
            elif i=='Carlos':
                bar_color='tab:blue'    
            elif i=='Maria':
                bar_color='tab:orange' 
            elif i=='Lukas':
                bar_color='tab:olive'
            elif i=='Felipe':
                bar_color='tab:purple'                
            elif i=='Santiago':
                bar_color='teal'
            for t in m.T:
                try:
                    if round(pe.value(m.X[i,j,t]))==1 and all(i!=already_used[kkk] for kkk in range(len(already_used))):
                        gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black",label=i)
                        gnt.annotate(i,xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')
                        already_used.append(i)
                    elif round(pe.value(m.X[i,j,t]))==1:
                        gnt.broken_barh([(m.t_p[t], pe.value(m.tau_p[i,j]))], (lower_y_position, height),facecolors =bar_color,edgecolor="black")
                        gnt.annotate(i,xy=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),xytext=((2*m.t_p[t]+pe.value(m.tau_p[i,j]))/2,(2*lower_y_position+height)/2),fontsize = 15,horizontalalignment='center')                        
                except:
                    pass 
    gnt.tick_params(axis='both', which='major', labelsize=15)
    gnt.tick_params(axis='both', which='minor', labelsize=15) 
    gnt.yaxis.label.set_size(15)
    gnt.xaxis.label.set_size(15)
    plt.legend()
    plt.show()
