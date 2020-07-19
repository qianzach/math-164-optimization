
import math
import scipy as sp
import numpy as np
import matplotlib as mpl

x0 = np.array([-2,2])
def f(x): #Rosenbrock's function
    return 100*(x[1]-(x[0]**2))**2 + ((1-x[0])**2) #Rosenbrock's

def g(x): #Gradient of Rosenbrock's Function
    return np.array([ -400*x[0]*(x[1]-x[0]**2) -2*(1-x[0]) , 200*(x[1] - x[0]**2) ])
H = np.array([[1,0],[0,1]]) #estimated matrix is the identity matrix  

#creating secant line search for alpha

def secant_ls(g,x,d): #requires gradient, initial value [-2,2], and d the search direction
    epsilon = 10**(-8) #threshold to avoid too many iterations
    alpha_curr = 0 #starting with stepsize = 0
    alpha = 0.001
    dphi_0 = g(x).dot(d)
    dphi_curr = dphi_0
    max_itr = 100 #set max iterations
    
    itr = 0
    while (abs(dphi_curr) > epsilon*abs(dphi_0) ):
        alpha_0 = alpha_curr
        alpha_curr = alpha
        dphi_old = dphi_curr
        dphi_curr = g(x + alpha_curr*d).dot(d)
        if (dphi_curr - dphi_old) == 0:
            break
        alpha = ((dphi_curr*alpha_0) - (dphi_old*alpha_curr)) / (dphi_curr - dphi_old)
        itr +=1
        if itr >= max_itr and (abs(dphi_curr) > epsilon*abs(dphi_0)):
            break
    return alpha  

#Rank One
r1_arr = [abs(f(x0) - f(np.array([1,1])))]
i_arr = [0]
max_itr = 100 #set a threshold for iterations
i = 0
epsilon = 10**-8 #threshold for abs difference
d_k = -H.dot(g(x0))
while (i < max_itr) or abs(g(x0).dot(g(x0))) > epsilon:
    if sum([x**2 for x in g(x0)]) == 0:
        break
    elif i % 3 == 0:
        d_k = -g(x0)
    a_k = secant_ls(g,x0,d_k)
    
    x_k = x0 + a_k*d_k #compute the next x term
    delta_x = a_k*d_k #change in x
    delta_g = g(x_k) - g(x0) #change in gradient of k+1th term and kth term
    
    if sum([x**2 for x in g(x0)]) == 0:
        print("x0 = ", x0)
        break    
    a = np.array([(delta_x - H.dot(delta_g))]) #reshaping so we can multiply matrices
    a = np.reshape(a,(2,1)) #making it the "row" matrix
    b = a.T #making it the "col" matrix
    H_k = H + np.matmul(a,b)/(delta_g.dot((delta_x - H.dot(delta_g)))) #matrix multiplication    #H_k = H + np.array([[0,0],[0,update]])
    x0 = x_k
    H = H_k
    i +=1
    r1_arr.append(abs(f(x0) - f(np.array([1,1]))))
    i_arr.append(i)
 
 #DFP
dfp_arr = [abs(f(x0) - f(np.array([1,1])))]
i_arr = [0]
max_itr = 100 #set a threshold for iterations
i = 0
epsilon = 10**-8 #threshold for abs difference
d_k = -H.dot(g(x0))
while (i < max_itr) or abs(g(x0).dot(g(x0))) > epsilon:
    if sum([x**2 for x in g(x0)]) == 0:
        break
    elif i % 3 == 0:
        d_k = -g(x0)
    a_k = secant_ls(g,x0,d_k)
    
    x_k = x0 + a_k*d_k #compute the next x term
    delta_x = a_k*d_k #change in x
    delta_g = g(x_k) - g(x0) #change in gradient of k+1th term and kth term
    
    if sum([x**2 for x in g(x0)]) == 0:
        print("x0 = ", x0)
        break    
        
    q =np.array([(delta_x)])
    w = np.array([(delta_x)]).T
    matrix_1 = (np.matmul(w,q) / delta_x.dot(delta_g)) #first 2x2 matrix
    
    e = np.array([delta_x]).T
    r = np.array([delta_g])
    t = np.array([delta_g]).T
    y = np.array([delta_x])
    matrix_2 = np.matmul(np.matmul(H,t),np.dot(r,H)) / (np.matmul(delta_g,H.dot(delta_g)))
    H_k = H + (matrix_1) - matrix_2
    
    x0 = x_k
    H = H_k
    i +=1
    dfp_arr.append(abs(f(x0) - f(np.array([1,1]))))
    i_arr.append(i)

#BFGS
bfgs_arr = [abs(f(x0) - f(np.array([1,1])))]
i_arr = [0]
max_itr = 100 #set a threshold for iterations
i = 0
epsilon = 10**-8 #threshold for abs difference
d_k = -H.dot(g(x0))
while (i < max_itr) or abs(g(x0).dot(g(x0))) > epsilon:
    if sum([x**2 for x in g(x0)]) == 0:
        break
    elif i % 3 == 0:
        d_k = -g(x0)
    a_k = secant_ls(g,x0,d_k)
    
    x_k = x0 + a_k*d_k #compute the next x term
    delta_x = a_k*d_k #change in x
    delta_g = g(x_k) - g(x0) #change in gradient of k+1th term and kth term
    
    if sum([x**2 for x in g(x0)]) == 0:
        print("x0 = ", x0)
        break    
        
    first_const = 1 + ( (delta_g.dot(H.dot(delta_g)))  / (delta_g.dot(delta_x)) ) #1 + dot product constant
    q =np.array([(delta_x)])
    w = np.array([(delta_x)]).T
    matrix_1 = (np.matmul(w,q) / delta_x.dot(delta_g)) #first 2x2 matrix
    
    e = np.array([delta_x]).T
    r = np.array([delta_g])
    t = np.array([delta_g]).T
    y = np.array([delta_x])
    matrix_2 = (np.matmul(np.matmul(e,r),H) + np.matmul(H,np.matmul(t,y)))/(delta_g.dot(delta_x))
    H_k = H + first_const*(matrix_1) - matrix_2
    
    x0 = x_k
    H = H_k
    i +=1
    bfgs_arr.append(abs(f(x0) - f(np.array([1,1]))))
    i_arr.append(i)


