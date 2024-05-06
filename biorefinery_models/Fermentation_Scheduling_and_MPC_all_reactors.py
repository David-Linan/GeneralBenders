

import pyomo.environ as pe
import copy



def is_multiple(y,x):
    return y != 0 and y % x == 0

if __name__ == '__main__':
    #Do not show warnings




    # NUMBER OF CYCLES TO BE SIMULATED IN THE SOM
    Number_cycles_som=2

    step=190*60*60/50     #NOTE: We assume this is the sampling time of the system Sampling_time


    include_fed_batch_op_time=False # If fed batch operation time will be included

    # Available reactors
    reactors_list=[1,2,3]


    # MORE TIMES REQUIRED TO DEFINE OPERATION
    Start_new_batch_time_wrt_0=70*60*60  #NOTE: In standar operation, this is the same as fed_batch time, but we will allow fed batch time to be variable
    Fed_batch_time_wrt_0=Start_new_batch_time_wrt_0
    Reaction_end_time_wrt_0=Start_new_batch_time_wrt_0*2
    Inoc_phase_time_wrt_0=10*60*60  # SECONDS
    Cycle_time_wrt_0=Start_new_batch_time_wrt_0*3+step
    # print(step)
    # print(Start_new_batch_time_wrt_0)
    # print(Inoc_phase_time_wrt_0)
    # print(Cycle_time_wrt_0)    
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Start_new_batch_time_wrt_0 and Start_new_batch_time_wrt_0<=cont_time+step:
             Start_new_batch_time_wrt_0=cont_time+step
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Fed_batch_time_wrt_0 and Fed_batch_time_wrt_0<=cont_time+step:
             Fed_batch_time_wrt_0=cont_time
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Inoc_phase_time_wrt_0 and Inoc_phase_time_wrt_0<=cont_time+step:
             Inoc_phase_time_wrt_0=cont_time
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Reaction_end_time_wrt_0 and Reaction_end_time_wrt_0<=cont_time+step:
             Reaction_end_time_wrt_0=cont_time+step
             break
    for cont_time in pe.RangeSet(0,Cycle_time_wrt_0+step,step):
        if cont_time<=Cycle_time_wrt_0 and Cycle_time_wrt_0<=cont_time+step:
             Cycle_time_wrt_0=cont_time+step
             break
    # print(Start_new_batch_time_wrt_0)
    # print(Inoc_phase_time_wrt_0)
    # print(Cycle_time_wrt_0)

    # print(Start_new_batch_time_wrt_0-step)
    # print(Inoc_phase_time_wrt_0-step)
    # print(Cycle_time_wrt_0-step)

    #Start time of SOM operation
    start_time=0 # NOTE: Must always be 0
    #Final time of SOM operation
    end_time=Number_cycles_som*(Cycle_time_wrt_0-step)

    # Dictionary with reactors start-time
    r_start_times={}

    for reactors in reactors_list:
        r_start_times[reactors]=(reactors-1)*Start_new_batch_time_wrt_0

    # print(r_start_times)

    # Dictionary with reactors operation modes

    r_operation_mode={} # reactor: list with operationg mode for every point in time
    for reactors in reactors_list:
        r_operation_mode[reactors]=[]
    # 0: batch reactor turned off, 1: start the execution of a batch reactor, 2: Batch reactor in Inoculum phase, 3: Batch reactor under fed-batch execution, 4: batch reactor without feeds! (if needed) 
    print('Time','  |  Operation mode r1', '  |  Operation mode r2','  |  Operation mode r3')
    time_index=-1
    current_Start_t=copy.deepcopy(r_start_times)
    for cont_time in pe.RangeSet(start_time,end_time,step):
        time_index=time_index+1
        for reactors in reactors_list:
            if cont_time==r_start_times[reactors] or cont_time==Cycle_time_wrt_0+current_Start_t[reactors]:
                r_operation_mode[reactors].append(1)
                current_Start_t[reactors]=cont_time
            elif cont_time>current_Start_t[reactors] and cont_time<=Inoc_phase_time_wrt_0+current_Start_t[reactors]:
                r_operation_mode[reactors].append(2)
            elif cont_time>current_Start_t[reactors]+Inoc_phase_time_wrt_0 and cont_time<= Fed_batch_time_wrt_0+current_Start_t[reactors]:
                r_operation_mode[reactors].append(3)
            elif cont_time>current_Start_t[reactors]+Fed_batch_time_wrt_0 and cont_time<=Reaction_end_time_wrt_0+current_Start_t[reactors]:
                if include_fed_batch_op_time:
                    r_operation_mode[reactors].append(4)
                else:
                    r_operation_mode[reactors].append(3)

            else:
                r_operation_mode[reactors].append(0)


        print(cont_time,'  |  ',r_operation_mode[1][time_index],'  |  ',r_operation_mode[2][time_index],'  |  ',r_operation_mode[3][time_index])

