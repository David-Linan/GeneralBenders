from __future__ import division
import pickle
import pandas as pd
import matplotlib.pyplot as plt
f_name="test_probabilities_more_rigurous2_23_to_25"
a_file = open(f_name+".pkl", "rb")
results = pickle.load(a_file)
#print(results[0])

#probability
updated_Results={}
for i in range(min( [list(results[0].keys())[j][0] for j in range(len(list(results[0].keys()))) ]   )  ,max(     [list(results[0].keys())[j][0] for j in range(len(list(results[0].keys()))) ]   )  +1):
    partial={}
    for j in results[0].keys():
        if j[0]==i:
            partial[j]=results[0][j]
    max_index=max([list(partial.keys())[j][1] for j in range(len(list(partial.keys())))])
    updated_Results[i]=results[0][(i,max_index)]
print(updated_Results)

plt.plot(list(updated_Results.keys()),list(updated_Results.values()))
plt.show()

#average cpu time
updated_Results_cpu_time={}
for i in range(min( [list(results[1].keys())[j][0] for j in range(len(list(results[1].keys()))) ]   )  ,max(     [list(results[1].keys())[j][0] for j in range(len(list(results[1].keys()))) ]   )  +1):
    partial={}
    for j in results[1].keys():
        if j[0]==i:
            partial[j]=results[1][j]
    max_index=max([list(partial.keys())[j][1] for j in range(len(list(partial.keys())))])
    updated_Results_cpu_time[i]=results[1][(i,max_index)]
print(updated_Results_cpu_time)

plt.plot(list(updated_Results_cpu_time.keys()),list(updated_Results_cpu_time.values()))
plt.show()

pd_data_proba=pd.DataFrame.from_dict(data=updated_Results,orient='index',columns=['probability'])
pd_data_time=pd.DataFrame.from_dict(data=updated_Results_cpu_time,orient='index',columns=['time'])
with pd.ExcelWriter(f_name+'.xlsx') as writer:
    pd_data_proba.to_excel(writer, sheet_name='Sheet1',startcol=1)
    pd_data_time.to_excel(writer, sheet_name='Sheet1',startcol=3)






# pd_data=pd.DataFrame(data=results).T
# print(pd_data)
# with pd.ExcelWriter('output_cstr_dbd.xlsx') as writer:  
#     pd_data.to_excel(writer, sheet_name='cuts')



# a_file = open("global_solvers_reactors.pkl", "rb")
# results = pickle.load(a_file)
# print(results)