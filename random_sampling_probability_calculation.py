from cuts_functions import initialization_sampling
from collections import Counter

#random_points_number=2
lower_bounds={1:1,2:1}
upper_bounds={1:5,2:5}
runs=10



actual_runs={}
probability={}
x={}

for points in range(2,3):
    data=[]
    data2=[]
    sorted_Data2=[]
    actual_runs[points]=0
    for runs in range(1,runs+1):
        #print(runs)
        try:
            res=initialization_sampling(points,lower_bounds,upper_bounds)
            res2=[tuple(res[k]) for  k in range(len(res))]
            data=data+[res2]
            #print(data)
            actual_runs[points]=actual_runs[points]+1
        except:
            pass
    data2=[tuple(data[k]) for k in range(len(data))]
    sorted_Data2=[tuple(sorted(t)) for t in data2]
    #print('sorted',sorted_Data2)
    x[points]=dict(Counter(sorted_Data2))
    probability[points]={key:x[points][key]/actual_runs[points] for key in x[points].keys()}
    #print(x[points])
    print("number of points sampled (without repetition):  ",len(x[points]))
    #print(actual_runs[points])
    print("probability of each point:  ",probability[points])

    
