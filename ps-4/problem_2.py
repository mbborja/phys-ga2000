import numpy as np
import matplotlib.pyplot as plt

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

def V(x):
    return x ** 4

def Period(a,m,N):
    int_start = 0
    int_end = a
    integral = integration(int_start, int_end, a, N)
    return(np.sqrt(8*m)*integral)

def integrand(x,a):
    return(1/np.sqrt(V(a) - V(x)))

def integration(start, end, a, N):
    xp,wp = gaussxwab(N,start,end)
    s = sum(integrand(xp, a)*wp)
    return(s)

m = 1
a_Range = np.linspace(0,2,100)
F = [Period(ai, m, 20) for ai in a_Range]

plt.plot(a_Range, F)
plt.xlabel("Amplitude (length)")
plt.ylabel("Period ($\sqrt{kg} /m$)")
plt.title("Period of Oscillation vs Aplitude")
plt.savefig("problem_2.png")
plt.show()