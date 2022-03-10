from __future__ import division
import pickle
import pandas as pd


#when using the closest discrete point that has not been evaluated yet
a_file = open("data_distillation_CUTS_WELL_ORDERED.pkl", "rb")
#when simply re-starting the initialization point by following discrete points in order
#a_file = open("data_distillation_CUTS.pkl", "rb")

#When using random strategy.
#a_file = open("data_distillation_CUTS_RANDOM.pkl", "rb")
results = pickle.load(a_file)
print(results)


#Refers to solving te problem for the first time to see if the solution if feasible. If it is, moduels 1 and 2 are not required
a_file = open("data_distillation_first_iter.pkl", "rb")
results_first_iter = pickle.load(a_file)
#print(results_first_iter)

objectives_method1={}
objectives_method2={}
objectives_method3={}
objectives_method4={}


for init in results:
    for method in results[init]:
        if method=='m1_s3':
            objectives_method1[init]=results[init][method][0]
        elif method=='m2_s3':
            objectives_method2[init]=results[init][method][0]
        elif method=='m3_s3':
            objectives_method3[init]=results[init][method][0]
        elif method=='m4_s3':                
            objectives_method4[init]=results[init][method][0]
#print(objectives_method1.values())
#print(objectives_method2.values())
#print(objectives_method3.values())
#print(objectives_method4.values())



#TODO AVERAGE TIME FOR POINTS NOT DECLARED AS INFEASIBLE IN THE FIRST ITERATION 

times_method1={}
times_method2={}
times_method3={}
times_method4={}


for init in results:

    times_method1[init]=results_first_iter[init]['cuts_first_iter']
    times_method2[init]=results_first_iter[init]['cuts_first_iter']
    times_method3[init]=results_first_iter[init]['cuts_first_iter']
    times_method4[init]=results_first_iter[init]['cuts_first_iter']

    for method in results[init]:
        if any([method=='m1_s3',method=='m1_s2',method=='m1_s1']):
            times_method1[init]=times_method1[init]+results[init][method][1]
        elif any([method=='m2_s3',method=='m2_s2',method=='m2_s1']):
            times_method2[init]=times_method2[init]+results[init][method][1]
        elif any([method=='m3_s3',method=='m3_s2',method=='m3_s1']):
            times_method3[init]=times_method3[init]+results[init][method][1]
        elif any([method=='m4_s3',method=='m4_s2',method=='m4_s1']):                
            times_method4[init]=times_method4[init]+results[init][method][1]

m1_time=list(times_method1.values())
print('average time m1',sum(m1_time)/len(m1_time),'s')
m2_time=list(times_method2.values())
print('average time m2',sum(m2_time)/len(m2_time),'s')
m3_time=list(times_method3.values())
print('average time m3',sum(m3_time)/len(m3_time),'s')
m4_time=list(times_method4.values())
print('average time m4',sum(m4_time)/len(m4_time),'s')

#gloa
a_file = open("data_distillation_gloa.pkl", "rb")
results_gloa = pickle.load(a_file)
#print(results_gloa)

objectives_gloa={}

for init in results_gloa:
    for method in results_gloa[init]:
        objectives_gloa[init]=results_gloa[init][method][0]
#print(objectives_loa.values())

times_gloa={}

for init in results_gloa:
    for method in results_gloa[init]:
        times_gloa[init]=results_gloa[init][method][1]
m_time=list(times_gloa.values())
print('average time GLOA',sum(m_time)/len(m_time),'s')

status_gloa={}

for init in results_gloa:
    for method in results_gloa[init]:
        status_gloa[init]=results_gloa[init][method][2]
#print(status_gloa)


#loa
a_file = open("data_distillation_loa.pkl", "rb")
results_loa = pickle.load(a_file)
#print(results_loa)

objectives_loa={}

for init in results_loa:
    for method in results_loa[init]:
        objectives_loa[init]=results_loa[init][method][0]
#print(objectives_loa.values())

times_loa={}

for init in results_loa:
    for method in results_loa[init]:
        times_loa[init]=results_loa[init][method][1]
m_time=list(times_loa.values())
print('average time LOA',sum(m_time)/len(m_time),'s')

status_loa={}

for init in results_loa:
    for method in results_loa[init]:
        status_loa[init]=results_loa[init][method][2]
#print(status_loa)



#lbb
a_file = open("data_distillation_lbb.pkl", "rb")
results_lbb = pickle.load(a_file)
#print(results_lbb)

objectives_lbb={}

for init in results_lbb:
    for method in results_lbb[init]:
        objectives_lbb[init]=results_lbb[init][method][0]
#print(objectives_lbb.values())

times_lbb={}

for init in results_lbb:
    for method in results_lbb[init]:
        times_lbb[init]=results_lbb[init][method][1]
m_time=list(times_lbb.values())
print('average time lbb',sum(m_time)/len(m_time),'s')

status_lbb={}

for init in results_lbb:
    for method in results_lbb[init]:
        status_lbb[init]=results_lbb[init][method][2]
#print(status_lbb)


#SBB
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


#D_SDA
a_file = open("data_distillation_D_SDA.pkl", "rb")
results_D_SDA = pickle.load(a_file)
#print(results_D_SDA)

objectives_D_SDA={}

for init in results_D_SDA:
    for method in results_D_SDA[init]:
        objectives_D_SDA[init]=results_D_SDA[init][method][0]
#print(objectives_D_SDA.values())

times_D_SDA={}

for init in results_D_SDA:
    for method in results_D_SDA[init]:
        times_D_SDA[init]=results_D_SDA[init][method][1]
m_time=list(times_D_SDA.values())
print('average time D_SDA',sum(m_time)/len(m_time),'s')


status_D_SDA={}

for init in results_D_SDA:
    for method in results_D_SDA[init]:
        status_D_SDA[init]=results_D_SDA[init][method][2]
#print(status_D_SDA)






## TO PANDAS AND EXCEL


#CUTS
pd_objectives_method1=pd.DataFrame(data=objectives_method1,index=['Obj_m1']).T
pd_objectives_method2=pd.DataFrame(data=objectives_method2,index=['Obj_m2']).T
pd_objectives_method3=pd.DataFrame(data=objectives_method3,index=['Obj_m3']).T
pd_objectives_method4=pd.DataFrame(data=objectives_method1,index=['Obj_m4']).T

pd_times_method1=pd.DataFrame(data=times_method1,index=['times_m1']).T
pd_times_method2=pd.DataFrame(data=times_method2,index=['times_m2']).T
pd_times_method3=pd.DataFrame(data=times_method3,index=['times_m3']).T
pd_times_method4=pd.DataFrame(data=times_method1,index=['times_m4']).T

m1_summary=pd.concat([pd_objectives_method1,pd_times_method1],axis=1)
m2_summary=pd.concat([pd_objectives_method2,pd_times_method2],axis=1)
m3_summary=pd.concat([pd_objectives_method3,pd_times_method3],axis=1)
m4_summary=pd.concat([pd_objectives_method4,pd_times_method4],axis=1)

cuts_summary=pd.concat([m1_summary,m2_summary,m3_summary,m4_summary],axis=1)
#print(cuts_summary)

#GLOA
pd_objectives_gloa=pd.DataFrame(data=objectives_gloa,index=['Obj_gloa']).T
pd_times_gloa=pd.DataFrame(data=times_gloa,index=['times_gloa']).T
pd_status_gloa=pd.DataFrame(data=status_gloa,index=['times_gloa']).T

gloa_summary=pd.concat([pd_objectives_gloa,pd_times_gloa,pd_status_gloa],axis=1)


#LOA
pd_objectives_loa=pd.DataFrame(data=objectives_loa,index=['Obj_loa']).T
pd_times_loa=pd.DataFrame(data=times_loa,index=['times_loa']).T
pd_status_loa=pd.DataFrame(data=status_loa,index=['times_loa']).T

loa_summary=pd.concat([pd_objectives_loa,pd_times_loa,pd_status_loa],axis=1)




#lbb
pd_objectives_lbb=pd.DataFrame(data=objectives_lbb,index=['Obj_lbb']).T
pd_times_lbb=pd.DataFrame(data=times_lbb,index=['times_lbb']).T
pd_status_lbb=pd.DataFrame(data=status_lbb,index=['times_lbb']).T

lbb_summary=pd.concat([pd_objectives_lbb,pd_times_lbb,pd_status_lbb],axis=1)

#sbb
pd_objectives_sbb=pd.DataFrame(data=objectives_sbb,index=['Obj_sbb']).T
pd_times_sbb=pd.DataFrame(data=times_sbb,index=['times_sbb']).T
pd_status_sbb=pd.DataFrame(data=status_sbb,index=['times_sbb']).T

sbb_summary=pd.concat([pd_objectives_sbb,pd_times_sbb,pd_status_sbb],axis=1)

#D_SDA
pd_objectives_D_SDA=pd.DataFrame(data=objectives_D_SDA,index=['Obj_D_SDA']).T
pd_times_D_SDA=pd.DataFrame(data=times_D_SDA,index=['times_D_SDA']).T
pd_status_D_SDA=pd.DataFrame(data=status_D_SDA,index=['times_D_SDA']).T

D_SDA_summary=pd.concat([pd_objectives_D_SDA,pd_times_D_SDA,pd_status_D_SDA],axis=1)



##Write excel
with pd.ExcelWriter('output_updated.xlsx') as writer:  
    cuts_summary.to_excel(writer, sheet_name='cuts')
    loa_summary.to_excel(writer, sheet_name='loa')
    gloa_summary.to_excel(writer, sheet_name='gloa')
    lbb_summary.to_excel(writer, sheet_name='lbb')
    sbb_summary.to_excel(writer, sheet_name='sbb')
    D_SDA_summary.to_excel(writer, sheet_name='D_SDA')







######do not use this
# for init in list(results):
#     for method in list(results[init]):
#         new_method=method.replace('sbb','loa')
#         print(new_method)
#         results[init].update({new_method:results[init][method]})
#         del results[init][method]
        
# a_file = open("data_distillation_loa.pkl", "wb")
# pickle.dump(results, a_file)
# a_file.close()


