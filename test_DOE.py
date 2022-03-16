import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS

xlimits = np.array([[1, 5], [1, 5]])
sampling = LHS(xlimits=xlimits,criterion='m')

num = 7
x = sampling(num)

print(x)


newx=[] #discrete values
for i in x.tolist():
    newx.append([round(j) for j in i])
print(np.array(newx))

plt.plot(x[:, 0], x[:, 1], "o")
plt.plot(np.array(newx)[:, 0],np.array(newx)[:, 1], ".")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((1,5))
plt.ylim((1,5))
plt.show()
