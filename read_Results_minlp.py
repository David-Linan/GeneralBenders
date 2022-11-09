from __future__ import division
import pickle
import pandas as pd


#----------------------sbb--------------------------------------------------------
a_file = open("data_distillation_sbb.pkl", "rb")
results_sbb = pickle.load(a_file)
#print(results_sbb)

objectives_sbb={}

for init in results_sbb:
    for method in results_sbb[init]:
        objectives_sbb[init]=results_sbb[init][method][0]
#print(objectives_sbb.values())

times_sbb={}

for init in results_sbb:
    for method in results_sbb[init]:
        times_sbb[init]=results_sbb[init][method][1]
m_time=list(times_sbb.values())
print('average time SBB',sum(m_time)/len(m_time),'s')

status_sbb={}
for init in results_sbb:
    for method in results_sbb[init]:
        status_sbb[init]=results_sbb[init][method][2]

#sbb
pd_objectives_sbb=pd.DataFrame(data=objectives_sbb,index=['Obj_sbb']).T
pd_times_sbb=pd.DataFrame(data=times_sbb,index=['times_sbb']).T
pd_status_sbb=pd.DataFrame(data=status_sbb,index=['times_sbb']).T

sbb_summary=pd.concat([pd_objectives_sbb,pd_times_sbb,pd_status_sbb],axis=1)

#----------------------dicopt--------------------------------------------------------
a_file = open("data_distillation_dicopt.pkl", "rb")
results_dicopt = pickle.load(a_file)
#print(results_dicopt)

objectives_dicopt={}

for init in results_dicopt:
    for method in results_dicopt[init]:
        objectives_dicopt[init]=results_dicopt[init][method][0]
#print(objectives_dicopt.values())

times_dicopt={}

for init in results_dicopt:
    for method in results_dicopt[init]:
        times_dicopt[init]=results_dicopt[init][method][1]
m_time=list(times_dicopt.values())
print('average time dicopt',sum(m_time)/len(m_time),'s')

status_dicopt={}
for init in results_dicopt:
    for method in results_dicopt[init]:
        status_dicopt[init]=results_dicopt[init][method][2]

#dicopt
pd_objectives_dicopt=pd.DataFrame(data=objectives_dicopt,index=['Obj_dicopt']).T
pd_times_dicopt=pd.DataFrame(data=times_dicopt,index=['times_dicopt']).T
pd_status_dicopt=pd.DataFrame(data=status_dicopt,index=['times_dicopt']).T

dicopt_summary=pd.concat([pd_objectives_dicopt,pd_times_dicopt,pd_status_dicopt],axis=1)

#----------------------alphaecp--------------------------------------------------------
a_file = open("data_distillation_alphaecp.pkl", "rb")
results_alphaecp = pickle.load(a_file)
#print(results_alphaecp)

objectives_alphaecp={}

for init in results_alphaecp:
    for method in results_alphaecp[init]:
        objectives_alphaecp[init]=results_alphaecp[init][method][0]
#print(objectives_alphaecp.values())

times_alphaecp={}

for init in results_alphaecp:
    for method in results_alphaecp[init]:
        times_alphaecp[init]=results_alphaecp[init][method][1]
m_time=list(times_alphaecp.values())
print('average time alphaecp',sum(m_time)/len(m_time),'s')

status_alphaecp={}
for init in results_alphaecp:
    for method in results_alphaecp[init]:
        status_alphaecp[init]=results_alphaecp[init][method][2]

#alphaecp
pd_objectives_alphaecp=pd.DataFrame(data=objectives_alphaecp,index=['Obj_alphaecp']).T
pd_times_alphaecp=pd.DataFrame(data=times_alphaecp,index=['times_alphaecp']).T
pd_status_alphaecp=pd.DataFrame(data=status_alphaecp,index=['times_alphaecp']).T

alphaecp_summary=pd.concat([pd_objectives_alphaecp,pd_times_alphaecp,pd_status_alphaecp],axis=1)

#----------------------baron--------------------------------------------------------
a_file = open("data_distillation_baron.pkl", "rb")
results_baron = pickle.load(a_file)
#print(results_baron)

objectives_baron={}

for init in results_baron:
    for method in results_baron[init]:
        objectives_baron[init]=results_baron[init][method][0]
#print(objectives_baron.values())

times_baron={}

for init in results_baron:
    for method in results_baron[init]:
        times_baron[init]=results_baron[init][method][1]
m_time=list(times_baron.values())
print('average time baron',sum(m_time)/len(m_time),'s')

status_baron={}
for init in results_baron:
    for method in results_baron[init]:
        status_baron[init]=results_baron[init][method][2]

#baron
pd_objectives_baron=pd.DataFrame(data=objectives_baron,index=['Obj_baron']).T
pd_times_baron=pd.DataFrame(data=times_baron,index=['times_baron']).T
pd_status_baron=pd.DataFrame(data=status_baron,index=['times_baron']).T

baron_summary=pd.concat([pd_objectives_baron,pd_times_baron,pd_status_baron],axis=1)

#----------------------lindoglobal--------------------------------------------------------
a_file = open("data_distillation_lindoglobal.pkl", "rb")
results_lindoglobal = pickle.load(a_file)
#print(results_lindoglobal)

objectives_lindoglobal={}

for init in results_lindoglobal:
    for method in results_lindoglobal[init]:
        objectives_lindoglobal[init]=results_lindoglobal[init][method][0]
#print(objectives_lindoglobal.values())

times_lindoglobal={}

for init in results_lindoglobal:
    for method in results_lindoglobal[init]:
        times_lindoglobal[init]=results_lindoglobal[init][method][1]
m_time=list(times_lindoglobal.values())
print('average time lindoglobal',sum(m_time)/len(m_time),'s')

status_lindoglobal={}
for init in results_lindoglobal:
    for method in results_lindoglobal[init]:
        status_lindoglobal[init]=results_lindoglobal[init][method][2]

#lindoglobal
pd_objectives_lindoglobal=pd.DataFrame(data=objectives_lindoglobal,index=['Obj_lindoglobal']).T
pd_times_lindoglobal=pd.DataFrame(data=times_lindoglobal,index=['times_lindoglobal']).T
pd_status_lindoglobal=pd.DataFrame(data=status_lindoglobal,index=['times_lindoglobal']).T

lindoglobal_summary=pd.concat([pd_objectives_lindoglobal,pd_times_lindoglobal,pd_status_lindoglobal],axis=1)

##Write excel
with pd.ExcelWriter('output_minlp.xlsx') as writer:  
    sbb_summary.to_excel(writer, sheet_name='sbb')
    dicopt_summary.to_excel(writer, sheet_name='dicopt')
    alphaecp_summary.to_excel(writer, sheet_name='alphaecp')
    baron_summary.to_excel(writer, sheet_name='baron')
    lindoglobal_summary.to_excel(writer, sheet_name='lindoglobal')







