
#1

#defining variables for problem 1
Q = np.array([[1,0],[0,2]])
f = lambda x: (0.5)*np.dot(x,np.dot(Q,x)) - np.dot(x, np.array([1,-1])) + 7
g = lambda x: np.dot(Q,x) - np.array([1,-1])
x0 = np.array([0,0]) #given starting point value
H = np.array([[1,0],[0,1]]) #estimated matrix is the identity matrix  

#Rank One Algorithm
while sum([x**2 for x in g(x0)]) != 0:
    print("x0 = ", x0)
    d_k = -H.dot(g(x0)) 
    a_k = -(g(x0).dot(d_k))/(d_k.dot(np.dot(Q,d_k))) #objective funciton is qudratic, so we can refer to this formula for stepsize

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
    itr +=1
    r1_arr.append(abs(f(x0) - f(np.array([1,-0.5])))) #used for plot
    itr_arr.append(itr) #used for plot

#Plot
from matplotlib import pyplot as plt
plt.plot(itr_arr,r1_arr)

plt.xlabel("Iterations")
plt.ylabel("Absolute difference from f(x_k) and minimizer")

#BFGS 
idx_arr = [0]
bfgs_arr = [abs(f(x0) - f(np.array([1,-0.5])))]
idx = 0
while sum([x**2 for x in g(x0)]) != 0:
    print("x0 = ", x0)
    d_k = -H.dot(g(x0)) 
    a_k = -(g(x0).dot(d_k))/(d_k.dot(np.dot(Q,d_k))) #objective funciton is qudratic, so we can refer to this formula for stepsize

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
    idx += 1
    bfgs_arr.append(abs(f(x0) - f(np.array([1,-0.5])))) #used for plot
    idx_arr.append(idx) #used for plot
#Plot
from matplotlib import pyplot as plt
plt.plot(idx_arr,bfgs_arr)

plt.xlabel("Iterations")
plt.ylabel("Absolute difference from f(x_k) and minimizer")


#2
#Function and Variable Declaration and Definition
x0 = np.array([-2,2])
#f = lambda x1,x2: 100*(x2-(x1**2))**2 + ((1-x1)**2) #Rosenbrock's
def f(x): #Rosenbrock's function
    return 100*(x[1]-(x[0]**2))**2 + ((1-x[0])**2) #Rosenbrock's

def g(x): #Gradient of Rosenbrock's Function
    return np.array([ -400*x[0]*(x[1]-x[0]**2) -2*(1-x[0]) , 200*(x[1] - x[0]**2) ])
#g = lambda x1,x2: np.array([ -400*x1*(x2-x1**2) -2*(1-x1) , 200*(x2 - x1**2) ])
d = -g(x0)

#Line Search Algorithm for step-size alpha_k
def secant_ls(g,x,d): #requires gradient, initial value [-2,2], and d the search direction
    epsilon = 10**(-8) #threshold to avoid too many iterations
    alpha_curr = 0 #starting with stepsize = 0
    alpha = 0.001
    #dphi_0 = g(x).dot(d)
    dphi_0 = g(x).dot(d)
    #print(dphi_zero, " dphi_zero")
    dphi_curr = dphi_0
    max_itr = 100 #set max iterations
    
    itr = 0
    while (abs(dphi_curr) > epsilon*abs(dphi_0) ):
        alpha_0 = alpha_curr
        alpha_curr = alpha
        dphi_old = dphi_curr
        dphi_curr = g(x + alpha_curr*d).dot(d)
        if (dphi_curr - dphi_old) == 0: #check in case bad division
            break
        alpha = ((dphi_curr*alpha_0) - (dphi_old*alpha_curr)) / (dphi_curr - dphi_old)
        itr +=1
        if itr >= max_itr and (abs(dphi_curr) > epsilon*abs(dphi_0)):
            #print("Max iterations achieved with line search")
            break
    return alpha  

#Hestenes-Stiefel Formula
max_itr = 100 #set a threshold for iterations
i = 0
epsilon = 10**-5 #threshold for abs difference
d = -g(x0) #initial direction
while (i < max_itr) or abs(g(x0).dot(g(x0))) > epsilon:
    if sum([x**2 for x in g(x0)]) == 0:
        break
    elif i % 6 == 0:
        d = -g(x0)
    a_k = secant_ls(g,x0,d)
    x_k = x0 + a_k*d
    if sum([x**2 for x in g(x0)]) == 0:
        break
    beta = (np.dot(g(x_k), g(x_k) - g(x0)))/ (d.dot( g(x_k) - g(x0) ) )
    i +=1 #iterate up
    if i % 6 == 0:
        d_k = -g(x_k) + beta*d
        d = d_k
    if abs(f(x0) - f(x_k)) >= 0.00001 :
        hs_arr.append(abs(f(x_k) - f([1,1])))
        ctr_arr.append(i)
    x0 = x_k

#Plot
from matplotlib import pyplot as plt
plt.plot(ctr_arr,hs_arr)

plt.xlabel("Iterations")
plt.ylabel("Absolute difference from f(x_k) and minimizer")

#Polak-Ribiere Formula
pr_arr = [abs(f(x0 - f([1,1])))]
i_arr = [0]

max_itr = 100 #set a threshold for iterations
i = 0
epsilon = 10**-8 #threshold for abs difference
d = -g(x0) #initial direction
while (i < max_itr) or abs(g(x0).dot(g(x0))) > epsilon:
   # print("x0: ", x0)
    if sum([x**2 for x in g(x0)]) == 0:
        break
    elif i % 6 == 0:
        d = -g(x0)
    a_k = secant_ls(g,x0,d)
    #print("a = ", a_k)
    x_k = x0 + a_k*d
    #print("x_k: ", x_k)
    if sum([x**2 for x in g(x0)]) == 0:
        break
    beta = (np.dot(g(x_k), g(x_k) - g(x0)))/ (g(x0).dot(g(x0))) 
    i +=1 #iterate up
    if i % 6 == 0:
        d_k = -g(x_k) + beta*d
        d = d_k
    x0 = x_k
    pr_arr.append(abs(f(x0 - f([1,1])))) #for plot
    i_arr.append(i) #for plot
#Plot
from matplotlib import pyplot as plt
plt.plot(i_arr,pr_arr)

plt.xlabel("Iterations")
plt.ylabel("Absolute difference from f(x_k) and minimizer")

#Fletcher-Reeves Formula
fr_arr = [abs(f(x0 - f([1,1])))]
it_arr = [0]

max_itr = 100 #set a threshold for iterations
i = 0
epsilon = 10**-8 #threshold for abs difference
d = -g(x0) #initial direction
while (i < max_itr) or abs(g(x0).dot(g(x0))) > epsilon:
    if sum([x**2 for x in g(x0)]) == 0:
        break
    elif i % 6 == 0:
        d = -g(x0)
    a_k = secant_ls(g,x0,d) 
    x_k = x0 + a_k*d
    if sum([x**2 for x in g(x0)]) == 0:
        break
    beta = (np.dot(g(x_k), g(x_k)))/ (g(x0).dot(g(x0))) 
    i +=1 #iterate up
    if i % 6 == 0:
        d_k = -g(x_k) + beta*d
        d = d_k
    x0 = x_k
    fr_arr.append(abs(f(x0 - f([1,1])))) #for plot
    it_arr.append(i) #for plot

#Plot
from matplotlib import pyplot as plt
plt.plot(it_arr,fr_arr)

plt.xlabel("Iterations")
plt.ylabel("Absolute difference from f(x_k) and minimizer")


