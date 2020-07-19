

#Question #4 Part D


import math
import scipy as sp
import numpy as np
import matplotlib as mpl

#Declaring variables and functions
Q = np.array([[5,2], [2,1]])
b = np.array([3,1])
f = lambda x: 0.5*np.dot(x,np.dot(Q,x)) - np.dot(x,b) #f(x)
eigenvalues = np.linalg.eigvals(Q) #eigenvalues to calculate threshold
a = 1/max(eigenvalues) # must be less than 2 / max lambda 
g = lambda x: Q.dot(x) - b #g(x) or our derivative
x0 = np.array([0,0]) # our initial starting point

#Gradient Descent Method

diff = 100 #set diff as a large number, will set as the distance between updated point and initial starting point
e = 1e-9 #threshold value
ctr = 0 #iteration counter
fxn_arr = [0] #used for plot, saving values for f(x)
counter_arr = [0] #used for plot, saving for number of iterations

while diff >= e:
    x_new = x0 - (a*g(x0)) #set update point
    diff = abs(f(x_new) - f(x0)) #find the distance
    x0 = x_new #update our starting point to the k+1th term
    ctr+=1 #used to plot
    fxn_arr.append(f(x0)) #used to plot
    counter_arr.append(ctr) #used to plot
print(x0) # Point is (0.999, -0.999), so basically (1, -1)

#PLOT: Gradient Descent
from matplotlib import pyplot as plt
plt.plot(counter_arr,fxn_arr)
plt.xlabel("Iterations")
plt.ylabel("f(x_k) Value (minimizer value)")



#CG Method
cg_arr = [0] #used for plot, saving values for f(x)
idx_arr = [0] #used for plot, saving for iterations
idx = 0 #iteration counter


alpha = -100 #setting our alpha value to arbitrary number
beta = -100 # setting our beta value to arbitrary number
if sum([x**2 for x in g(x0)]) == 0:
    x0 = x0 #trivial step, just a "check" to see if the gradient is 0
else:
    d = -g(x0) #setting our d to -g(x0) as seen from the book

while sum([x**2 for x in g(x0)]) != 0: #check ig g(x0) != 0 
    alpha = -(np.dot( g(x0),d))/(np.dot(d,Q.dot(d)))
    x_k = x0 + alpha*d
    if sum([x**2 for x in g(x0)]) == 0: #check ig g(x0) == 0 
        break #if the gradient is 0, break
    beta = (np.dot(g(x_k),Q.dot(d)))/( np.dot(d, Q.dot(d))  ) #calculating beta, as seen in textbook
    d_k = -g(x_k) + beta*d #setting our next interval d value
    d = d_k #update d value
    x0 = x_k #update starting pt
    idx +=1 #used for plot
    cg_arr.append(f(x0)) #used for plot
    idx_arr.append(idx) #used for plot

#PLOT: CG Method
from matplotlib import pyplot as plt
plt.plot(idx_arr,cg_arr)
plt.xlabel("Iterations")
plt.ylabel("f(x_k) Value (minimizer value)")




    

