from scipy.special import roots_hermite
import numpy as np
import matplotlib.pyplot as plt

# Part A

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

# Part A

def H(n,x):
    if n == 0:
        return 1
    if n == 1:
        return 2*x
    else:
        return(2*x*H(n-1,x) - (2*(n-1)*H(n-2,x)))

def Wavefn(n,x):
    return( (1/(np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi)))) * np.exp(-x**2 / 2) * H(n,x))

x_Range = np.linspace(-4,4,1000)
n_Range = range(0,4)

for n in n_Range:
    F_n = [Wavefn(n,xi) for xi in x_Range]
    plt.plot(x_Range, F_n, label = "N = {n_val}".format(n_val = n))

plt.xlabel("x (m)")
plt.ylabel("$\Psi_n (x)$ $(1/m^{1/2})$")
plt.title("Quantum Harmonic Oscillator Wave Function")
plt.legend()
plt.savefig("problem_3_partA.png")
plt.show()

# Part B

x_Range = np.linspace(-10,10,100)
n = 30

plt.plot(x_Range, Wavefn(n,x_Range), label = "N = {n_val}".format(n_val = n))
plt.xlabel("x (m)")
plt.ylabel("$\Psi_n (x)$ $(1/m^{1/2})$")
plt.title("Quantum Harmonic Oscillator Wave Function")
plt.legend()
plt.savefig("problem_3_partB.png")
plt.show()

# Part C
def integrand(u,n):
    return ((np.tan(u)**2 / np.cos(u)**2) * Wavefn(n,np.tan(u))**2)

def integration(start, end, N, n):
    xp,wp = gaussxwab(N,start, end)
    s = sum(integrand(xp,n)*wp)
    return(s)

def uncert(N,n):
    start = -np.pi/2
    end = np.pi/2
    integral = integration(start, end, N, n)
    return integral

print("Evaluation using gaussian quadrature",np.sqrt(uncert(100,5)))

# xp,wp = gaussxwab(100,-np.pi/2,np.pi/2)
# n = 5
# y = Wavefn(n,np.tan(xp))**2 * np.tan(xp)**2 / np.cos(xp)**2 
# s = sum(y*wp)
# print("Weird",np.sqrt(s))

# # Part D
# (d) Perform the calculation using Gauss-Hermite quadrature
# (scipy can give you the right roots and weights to use). Can you make an exact evaluation
# (meaning zero approximation error) of the integral?

from scipy.special import roots_hermite


def integrand_hermite(x,n):
    return (np.exp(x**2) * (x**2) * (Wavefn(n,x)**2))

def integration_hermite(N, n):
    x,w = roots_hermite(N)
    s = sum(integrand_hermite(x,n)*w)
    return(s)

def uncert(N,n):
    integral = integration_hermite(N, n)
    return integral

print("Evaluation using Gauss-Hermite: ",np.sqrt(uncert(300,5)))