# Script for Homework 2 Algorithms

#importing libraries
import math
import matplotlib

#6a:

#NOTE: My plot was done on matlab, and the script is as commented:
#x = linspace(0,10,200); 
#y = 8*exp(1-x) + 7*log(x);
#plot(x,y)
#xlim([1 2])
#ylim([7.5 8])

#6b and 6c:

#define f(x)
def f(x):
    # function f(x) definition
    return (8*math.exp(1-x) + 7*(math.log(x)))

#variable declarations
x_min = -10000 #set to a trivial number for our minimizer prior to the Golden
a0 = 1 #lower bound
b0 = 2 #upper bound
u = 0.23 #uncertainty threshold
p = ((3 - math.sqrt(5))/ 2) #according to textbook we set it to this value because we want p < 1/2

#Golden Section Method
#we want an uncertainty value within 0.23
while (b0 - a0) >= u:
    print("a0= , ", a0, " f(a0)=", f(a0), " b0= ", b0, " f(b0)=", f(b0))
    a1 = a0 + p*(b0-a0) #create intermediate points a1 and b1
    b1 = a0 + (1-p)*(b0-a0)
    if f(a1) < f(b1): #want to compare new boundary points
        b0 = b1
    else:
        a0 = a1
x_min = a0


#Derivative function; did NOT use sympy for calculus library
def f_prime(x):
	return -8*math.exp(1-x) + (7 / x)

#variable declarations
x_min = -10000 #set to a trivial number for our minimizer prior to the Golden
a0 = 1 #lower bound
b0 = 2 #upper bound
u = 0.23

#Bisection Method  
x0 = (a0 + b0)/2 #set midpoint as the initial uncertainty interval
if f_prime(x0) == 0:
    x_min = x0 #that is, if x0 is already the minimizer
it = 0
while f_prime(x0) != 0 or (b-a >= u):
    print("Iteration: ",it)
    print("x0=" ,x0," f(x0)=", f(x0) , " f'(x0)= ", f_prime(x0))
    print("a0=" ,a0," f(a0)=", f(a0) , " f'(a0)= ", f_prime(a0))
    print("b0=" ,b0," f(b0)=", f(b0) , " f'(b0)= ", f_prime(b0))
    if f_prime(x0) > 0: # check if f'(x0) is greater than 0
        b0 = x0 # set new intervals [a0, x0] x0 is the new upper bound
        x0 = (a0 + b0)/2 # redefine our x0 as new midpoint
    elif f_prime(x0) < 0: #check if f'(x0) is less than 0
        a0 = x0 # set new intervals [x0,b0] x0 is the new lower bound
        x0 = (a0 + b0)/2 # redefine our x0 as new midpoint
    else: #break case in case x0 = 0; a little trivial considering what my while loop is, haha.
        x_min = x0 #set x_min as x0
        break
    it+=1
x_min = x0 


#2nd Derivative Function
def d2f(x):
    return 8*math.exp(1-x) - (7 / (x*x))

#variable declarations
a0 = 1 #lower bound
b0 = 2 #upper bound
u = 0.23 #uncertainty/epsilon value, we can substitute this with 10e-5 if need be

#Newton's Method
#note, we let a0 be our first "guess"
ctr = 0
while abs(b0 - a0) >= u:
    print("Iteration: ", ctr )
    print("a0=" ,a0," f(a0)=", f(a0) , " f'(a0)= ", f_prime(a0), " f''(a0)=", d2f(a0))
    print("b0=" ,b0," f(b0)=", f(b0) , " f'(b0)= ", f_prime(b0), " f''(b0)=", d2f(b0))

    a0 = b0
    b0 = a0 - (f_prime(a0)/d2f(a0))
    ctr +=1
    
print(a0) #the earliest guess, although b0 would be the next best iteration


#variable declarations
x_min = -10000 #set to a trivial number for our minimizer prior to the Golden
a = 1 #lower bound and our k-1th term
b = 2 #upper bound and kth term
u = 0.23 #uncertainty value

# Secant Method
x_min = a #set to arbitrary values
it = 0
while abs(a-b) >= u:
    print("Iteration: ", it)
    print("x=" ,x_min," f(x)=", f(x_min) , " f'(x)= ", f_prime(x_min))
    print("a=" ,a," f(a)=", f(a) , " f'(a)= ", f_prime(a))
    print("b=" ,b," f(b)=", f(b) , " f'(b)= ", f_prime(b))
    x_min = ((f_prime(b)*a) -(f_prime(a)*b))/(f_prime(b) - f_prime(a))
    a,b = b, x_min    # shift variables (prepare for next loop)
    it+=1
print("x=" ,x_min," f(x)=", f(x_min) , " f'(x)= ", f_prime(x_min))
print("a=" ,a," f(a)=", f(a) , " f'(a)= ", f_prime(a))
print("b=" ,b," f(b)=", f(b) , " f'(b)= ", f_prime(b))

