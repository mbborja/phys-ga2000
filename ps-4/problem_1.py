import numpy as np
import matplotlib.pyplot as plt

######################################################################
# http://www-personal.umich.edu/~mejn/computational-physics/gaussxw.py
# Functions to calculate integration points and weights for Gaussian
# quadrature
#
# x,w = gaussxw(N) returns integration points x and integration
#           weights w such that sum_i w[i]*f(x[i]) is the Nth-order
#           Gaussian approximation to the integral int_{-1}^1 f(x) dx
# x,w = gaussxwab(N,a,b) returns integration points and weights
#           mapped to the interval [a,b], so that sum_i w[i]*f(x[i])
#           is the Nth-order Gaussian approximation to the integral
#           int_a^b f(x) dx
#
# This code finds the zeros of the nth Legendre polynomial using
# Newton's method, starting from the approximation given in Abramowitz
# and Stegun 22.16.6.  The Legendre polynomial itself is evaluated
# using the recurrence relation given in Abramowitz and Stegun
# 22.7.10.  The function has been checked against other sources for
# values of N up to 1000.  It is compatible with version 2 and version
# 3 of Python.
#
# Written by Mark Newman <mejn@umich.edu>, June 4, 2011
# You may use, share, or modify this file freely
#
######################################################################

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

#Part A
def Cv(volume,rho,temp,debye_temp, N):
    """_summary_

    Args:
        volume (_type_): _description_
        rho (_type_): _description_
        temp (_type_): _description_
        debye_temp (_type_): _description_
    """
    k = 1.380648e-23 
    a = 0
    b = debye_temp/temp
    integral = integration(a,b,N)
    return(9 * volume * rho * k * (temp/debye_temp)**3 * integral)

def integrand(x): 
    return ((x**4) * (np.exp(x))/((np.exp(x)-1)**2))    

def integration(a,b, N):
    xp,wp = gaussxwab(N,a,b)
    s = sum(integrand(xp)*wp)
    return(s)

Temp = 100 
rho = 6.022e28
Debye_theta = 428
N = 50
Volume = 0.001
print("Part 1 CV: ", Cv(Volume, rho, Temp, Debye_theta, N))

#Part B
Temp_Range = np.linspace(5, 500, 496)
F = [Cv(Volume, rho, ti, Debye_theta, N) for ti in Temp_Range]
plt.plot(Temp_Range,F)
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity (J/K)")
plt.title("Heat Capacity vs Temperature")
plt.savefig("problem_1_partB.png")
plt.show()

#Part C

N_range = range(1,71,1)
F = [Cv(Volume, rho, Temp, Debye_theta, ni) for ni in N_range]
plt.plot(N_range,F)
plt.xlabel("Value of N")
plt.ylabel("Heat Capacity (J/K)")
plt.title("Test for Convergence of Gaussian Quadrature Integral")
plt.savefig("problem_1_partC.png")
plt.show()
# for i in range(10,71,10):
#     print("CV: ", Cv(Volume, rho, Temp, Debye_theta, i))
    
# x = np.linspace(-10,1000,1)
# I = [Cv(0.01, 6.022e28, 100, 428) for xi in x]
# print(I[0])
# plt.plot(x,I)
# plt.xlabel('x')
# plt.ylabel(r'f(x)')
# plt.show()