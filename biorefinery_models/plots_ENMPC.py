import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
#Do not show warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)

test_name='ENMPC_standard_final_fixed_obj'
#'Traditional_final_fixed_obj_const_ph_const_yeast'
# ENMPC_standard_final_fixed_obj

print('------------',test_name,'----------------------')
with open('C:/Users/dlinanro/Desktop/GeneralBenders/saved_'+test_name, 'rb') as loaded_data:
    data = pickle.load(loaded_data)

time_list=data[0]
Hold_up_dict=data[1]
pH_dict=data[2]
yeast_dict=data[3]
C5_dict=data[4]
fiber_dict=data[5]
Concentration_dict=data[6]
objective_dict=data[7]
objective_evaluation=data[8]
Concentration_disturbance_dict_C5=data[9]
Concentration_disturbance_dict_F=data[10]

time_dict={1:time_list,2:time_list,3:time_list}


# EXPORT DATA DO EXCEL
data_excel={}
data_excel['Time']=time_list
for r in [1,2,3]:
    data_excel['Hold_up_h_'+str(r)]=Hold_up_dict[r]
    data_excel['pH_'+str(r)]=pH_dict[r]
    data_excel['yeast_kg_'+str(r)]=yeast_dict[r]
    data_excel['C5_flow_kgs_'+str(r)]=C5_dict[r]
    data_excel['fiber_flow_kgs_'+str(r)]=fiber_dict[r]
    data_excel['objective_evaluation_'+str(r)]=objective_evaluation[r]
    for j in ['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']:
        data_excel['C_'+j+'_gkg_'+str(r)]=Concentration_dict[(j,r)]
    for j in ['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']:
        data_excel['_disturb_C5_'+j+'_gkg_'+str(r)]=Concentration_disturbance_dict_C5[(j,r)]      
    for j in ['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']:  
        data_excel['_disturb_F_'+j+'_gkg_'+str(r)]=Concentration_disturbance_dict_F[(j,r)]

df = pd.DataFrame(data=data_excel)
df.to_excel('data_'+test_name+'.xlsx')

print('\n OBJECTIVE FUNCTION VALUES',objective_dict)
total_obj=0
for r in [1,2,3]:
    current_obj=sum(i for i in objective_dict[r] )
    total_obj=total_obj+current_obj
    print('Total objective function for reactor ',str(r),':',current_obj)
print('Total objective: ',total_obj)
# for r in [1,2,3]:
#     print('reactor ',r)
#     pos=-1
#     for is_objective in objective_evaluation[r]:
#         pos=pos+1
#         if is_objective==1:
#             print(50*yeast_dict[r][pos]-5*Concentration_dict[('Eth',r)][pos]*Hold_up_dict[r][pos])
total_C5_used=0
total_F_used=0
for r in [1,2,3]:
    integral_C5=np.trapz(C5_dict[r],[i*60*60 for i in time_list])
    print('\n Total C5 used by reactor [kg] ',str(r),':',integral_C5)
    total_C5_used=total_C5_used+integral_C5

    integral_F=np.trapz(fiber_dict[r],[i*60*60 for i in time_list])
    print('Total F used by reactor [kg] ',str(r),':',integral_F)
    total_F_used=total_F_used+integral_F
print('\n Total C5 used by all reactors [kg]: ',total_C5_used)
print('Total F used by all reactors  [kg]:',total_F_used)


fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    colors=['b','g','m','r','k','y','c']
    contador=-1
    for j in ['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']:
        if j=='G' or j=='X' or j=='Eth' or j=='Cell':
            contador=contador+1
            plt.plot(time_dict[r],Concentration_dict[(j,r)],colors[contador],label=j+'')

        plt.xlabel('time [h]')
        plt.ylabel('Concentration [g/kg]')
        plt.legend()
plt.show()



fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    contador=-1
    for j in ['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']:
        if j=='CS' or j=='XS' or j=='E':
            contador=contador+1
            plt.plot(time_dict[r],Concentration_dict[(j,r)],colors[contador],label=j+' '+test_name)

        plt.xlabel('time [h]')
        plt.ylabel('Concentration [g/kg]')
        plt.legend()
plt.show()




fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],pH_dict[r],label=test_name)

    plt.xlabel('time [h]')
    plt.ylabel('pH')
    plt.legend()

plt.show()




fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],C5_dict[r],label=test_name)

    plt.xlabel('time [h]')
    plt.ylabel('C5 flow [kg/s]')
    plt.legend()
plt.show()

colors=['b','g','m']
for r in [1,2,3]:
    contador=contador+1
    plt.plot(time_dict[r],C5_dict[r],colors[r-1],label='Reactor '+str(r)+' '+test_name)
    plt.xlabel('time [h]')
    plt.ylabel('C5 flow [kg/s]')
    plt.legend()

plt.show()




fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],fiber_dict[r],label=test_name)
    plt.xlabel('time [h]')
    plt.ylabel('Liquified fibers flow [kg/s]')
    plt.legend()

plt.show()

colors=['b','g','m']
for r in [1,2,3]:
    contador=contador+1
    plt.plot(time_dict[r],fiber_dict[r],colors[r-1],label='Reactor '+str(r)+' '+test_name)
    plt.xlabel('time [h]')
    plt.ylabel('Liquified fibers flow [kg/s]')
    plt.legend()

plt.show()

fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],Hold_up_dict[r],label=test_name)
    plt.xlabel('time [h]')
    plt.ylabel('Hold-up [kg]')
    plt.legend()


plt.show()

fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],yeast_dict[r],label=test_name)

    plt.xlabel('time [h]')
    plt.ylabel('yeast [kg]')
    plt.legend()
plt.show()
