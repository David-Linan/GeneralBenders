import pickle
import matplotlib.pyplot as plt

with open('C:/Users/dlinanro/Desktop/GeneralBenders/saved_ENMPC_test', 'rb') as loaded_data:
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

time_dict={1:time_list,2:time_list,3:time_list}

print('\n OBJECTIVE FUNCTION VALUES',objective_dict)
for r in [1,2,3]:
    print('reactor ',r)
    pos=-1
    for is_objective in objective_evaluation[r]:
        pos=pos+1
        if is_objective==1:
            print(50*yeast_dict[r][pos]-5*Concentration_dict[('Eth',r)][pos]*Hold_up_dict[r][pos])



fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    colors=['b','g','m','r','k','y','c']
    contador=-1
    for j in ['CS', 'XS', 'LS','C','G', 'X', 'F', 'E','AC','Cell','Eth','CO2','ACT','HMF','Base']:
        if j=='G' or j=='X' or j=='Eth' or j=='Cell':
            contador=contador+1
            plt.plot(time_dict[r],Concentration_dict[(j,r)],colors[contador],label=j+' (ENMPC)')

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
            plt.plot(time_dict[r],Concentration_dict[(j,r)],colors[contador],label=j+' (ENMPC)')

        plt.xlabel('time [h]')
        plt.ylabel('Concentration [g/kg]')
        plt.legend()
plt.show()


fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],pH_dict[r],label='ENMPC')

    plt.xlabel('time [h]')
    plt.ylabel('pH')
    plt.legend()

plt.show()




fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],C5_dict[r],label='ENMPC')

    plt.xlabel('time [h]')
    plt.ylabel('C5 flow [kg/s]')
    plt.legend()
plt.show()

colors=['b','g','m']
for r in [1,2,3]:
    contador=contador+1
    plt.plot(time_dict[r],C5_dict[r],colors[r-1],label='Reactor '+str(r)+'(ENMPC)')
    plt.xlabel('time [h]')
    plt.ylabel('C5 flow [kg/s]')
    plt.legend()

plt.show()




fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],fiber_dict[r],label='ENMPC')
    plt.xlabel('time [h]')
    plt.ylabel('Liquified fibers flow [kg/s]')
    plt.legend()

plt.show()

colors=['b','g','m']
for r in [1,2,3]:
    contador=contador+1
    plt.plot(time_dict[r],fiber_dict[r],colors[r-1],label='Reactor '+str(r)+'(ENMPC)')
    plt.xlabel('time [h]')
    plt.ylabel('Liquified fibers flow [kg/s]')
    plt.legend()

plt.show()

fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],Hold_up_dict[r],label='ENMPC')
    plt.xlabel('time [h]')
    plt.ylabel('Hold-up [kg]')
    plt.legend()


plt.show()

fig = plt.figure()
for r in [1,2,3]:
    plt.subplot(3,1,r)
    plt.plot(time_dict[r],yeast_dict[r],label='ENMPC')

    plt.xlabel('time [h]')
    plt.ylabel('yeast [kg]')
    plt.legend()
plt.show()
