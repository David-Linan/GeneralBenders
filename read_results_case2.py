from __future__ import division
import pickle
import pandas as pd

a_file = open("data_cstr_5_reactors_multistart.pkl", "rb")
results = pickle.load(a_file)
#print(results)


pd_data=pd.DataFrame(data=results).T
print(pd_data)
with pd.ExcelWriter('output_cstr_dbd.xlsx') as writer:  
    pd_data.to_excel(writer, sheet_name='cuts')



a_file = open("global_solvers_reactors.pkl", "rb")
results = pickle.load(a_file)
print(results)