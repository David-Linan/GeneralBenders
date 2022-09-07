from __future__ import division
import pickle
import pandas as pd
import matplotlib.pyplot as plt
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

#### ---Code to obtain total number of runs

# f_name="test_probabilities_secondary_info_more_rigurous4_25_to_26_naive"
# a_file = open(f_name+".pkl", "rb")
# results = pickle.load(a_file)


# for_next=0
# actual=0
# for key in results:
#     if list(key)[0]-for_next==1:
#         print(actual)    
#     elif key==list(results)[-1]:
#         print(key)
#     for_next=list(key)[0]
#     actual=key

### ---CODE TO OBTAIN PROBABILITY CONSIDERING A MINIMUM VALUE

min_point=10 #KEEP FIXED
max_point=20

files=['8_to_10','11_to_13','14_to_16','17_to_19','20_to_22']

iterat=0
for i in files:
    iterat=iterat+1
    if iterat==1:
        f_name="test_probabilities_more_rigurous4_"+i+"_naive"
        a_file = open(f_name+".pkl", "rb")
        results = pickle.load(a_file)[0]
    else:
        f_name="test_probabilities_more_rigurous4_"+i+"_naive"
        a_file = open(f_name+".pkl", "rb")
        results = merge_two_dicts(results,pickle.load(a_file)[0])
iteration=51
updated_probability={}
for points in range(min_point,max_point+1):
    updated_probability[points]=float(results[(points,iteration)])

num_points=iteration*10
print(updated_probability)


name='Updated_probability_'+str(num_points)+'_points_'+'from_'+str(min_point)+'to'+str(max_point)
pd_data_proba=pd.DataFrame.from_dict(data=updated_probability,orient='index',columns=['probability'])
with pd.ExcelWriter(name+'.xlsx') as writer:
    pd_data_proba.to_excel(writer, sheet_name='Sheet1',startcol=1)


# #### PAT 2: PROBA INFO
# f_name="test_probabilities_more_rigurous4_25_to_26_naive"
# a_file = open(f_name+".pkl", "rb")
# results = pickle.load(a_file)
# print(results)

# #probability
# updated_Results={}
# for i in range(min( [list(results[0].keys())[j][0] for j in range(len(list(results[0].keys()))) ]   )  ,max(     [list(results[0].keys())[j][0] for j in range(len(list(results[0].keys()))) ]   )  +1):
#     partial={}
#     for j in results[0].keys():
#         if j[0]==i:
#             partial[j]=results[0][j]
#     max_index=max([list(partial.keys())[j][1] for j in range(len(list(partial.keys())))])
#     updated_Results[i]=results[0][(i,max_index)]
# print(updated_Results)

# plt.plot(list(updated_Results.keys()),list(updated_Results.values()))
# plt.show()

# #average cpu time
# updated_Results_cpu_time={}
# for i in range(min( [list(results[1].keys())[j][0] for j in range(len(list(results[1].keys()))) ]   )  ,max(     [list(results[1].keys())[j][0] for j in range(len(list(results[1].keys()))) ]   )  +1):
#     partial={}
#     for j in results[1].keys():
#         if j[0]==i:
#             partial[j]=results[1][j]
#     max_index=max([list(partial.keys())[j][1] for j in range(len(list(partial.keys())))])
#     updated_Results_cpu_time[i]=results[1][(i,max_index)]
# print(updated_Results_cpu_time)

# plt.plot(list(updated_Results_cpu_time.keys()),list(updated_Results_cpu_time.values()))
# plt.show()

# pd_data_proba=pd.DataFrame.from_dict(data=updated_Results,orient='index',columns=['probability'])
# pd_data_time=pd.DataFrame.from_dict(data=updated_Results_cpu_time,orient='index',columns=['time'])
# with pd.ExcelWriter(f_name+'.xlsx') as writer:
#     pd_data_proba.to_excel(writer, sheet_name='Sheet1',startcol=1)
#     pd_data_time.to_excel(writer, sheet_name='Sheet1',startcol=3)




# # a_file = open("global_solvers_reactors.pkl", "rb")
# # results = pickle.load(a_file)
# # print(results)