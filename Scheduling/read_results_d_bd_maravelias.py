from __future__ import division
import pickle
import pandas as pd
import matplotlib.pyplot as plt
f_name="data_maravelis_d_bd_rho_variable"
a_file = open(f_name+".pkl", "rb")
important_info,important_info_preprocessing,D,x_actual = pickle.load(a_file)
print(important_info)
print(D)
print(x_actual)
