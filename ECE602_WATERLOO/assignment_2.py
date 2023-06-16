import numpy as np
import matplotlib.pyplot as plt

#PROBLEM STATEMENT
Q=np.array([[0.51,-0.49],[-0.49,0.51]])
p=np.array([[1],[1]])
r=0
x_0=np.array([[100],[100]])

#FUNCTIONS: objective value, gradient value, and l_2 norm of the gradient
def _f_val(x):
    return (1/2)*np.transpose(x)@Q@x+np.transpose(p)@x+r

def _grad_val(x):
    return  (1/2)*(Q+np.transpose(Q))@x+p

def _grad_norm_val(x):
    return np.linalg.norm((1/2)*(Q+np.transpose(Q))@x+p,2) 


# Mu for step size
eig_Q,_=np.linalg.eig(Q)
mu=max(eig_Q)



# THE MM ALGORITHM
x_values={}
f_values={}
gradient_values={}
gradient_norm_values={}

max_iterations=1000
iterations=range(0,max_iterations)

for i in iterations:
    # UPDATE THE VALUE OF X
    if i==0:
        x_k=x_0
        x_next=x_k
    else:
        x_k=x_next
        x_next=x_k-(1/mu)*_grad_val(x_k)

    # STORE RELEVAND INFORMATION
    f=_f_val(x_k)
    grad=_grad_val(x_k)
    grad_norm=_grad_norm_val(x_k)
    x_values[i]=x_k
    f_values[i]=f
    gradient_values[i]=grad
    gradient_norm_values[i]=grad_norm

    # EVALUATE THE STOPPING CONDITION
    if _grad_norm_val(x_k)<=1e-3:
        break


#Number of iterations
num_iter=list(f_values.keys())[-1]+1
print('The number of iterations is: ',num_iter)


#GENERATE PLOTS
k_vect=[i for i in f_values.keys()]
f_vect=[f_values[i].tolist()[0][0] for i in f_values.keys()]
f_grad_norm_vect=[gradient_norm_values[i].tolist() for i in f_values.keys()]


plt.plot(k_vect,f_vect)
plt.xlabel('Iteration k')
plt.ylabel(r'$f(x^{k})$')
plt.show()
plt.plot(k_vect,f_grad_norm_vect)
plt.xlabel('Iteration k')
plt.ylabel(r'$\|\|\nabla f(x^{k})\|\|_{2}$')
plt.show()

