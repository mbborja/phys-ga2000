import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the functions for O(N^3) and O(N^2)
def o_n3(N, a):
    return a * N**3

def o_n2(N, b):
    return b * N**2

def o_n4(N, c):
    return c * N**4

def o_n1(N, d):
    return d * N**1

# Initialize lists to store runtimes and corresponding N values
n_values = []
explicit_runtimes = []
numpy_dot_runtimes = []

for N in range (10,110,10):
    explicit = """import numpy as np
N = {dimension}
C = np.zeros([N,N], float)

A = np.random.randint(0,100, size =(N,N))
B = np.random.randint(0,100, size =(N,N))

for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i,j] += A[i,k]*B[k,j]
""".format(dimension = N)

    numpy_dot = """import numpy as np
N = {dimension}
C = np.zeros([N,N], float)

A = np.random.randint(0,100, size =(N,N))
B = np.random.randint(0,100, size =(N,N))

D = np.dot(A,B)
""".format(dimension = N)

    time1 = timeit.timeit(stmt=explicit, number = 10)
    time2 = timeit.timeit(stmt=numpy_dot, number = 10)
    n_values.append(N)
    explicit_runtimes.append(time1)
    numpy_dot_runtimes.append(time2)


# Fit the data to the O(N^3) and O(N^2) functions
params_n4, _ = curve_fit(o_n4, n_values, explicit_runtimes)
params_n3, _ = curve_fit(o_n3, n_values, explicit_runtimes)
params_n2, _ = curve_fit(o_n2, n_values, explicit_runtimes)
params_n1, _ = curve_fit(o_n1, n_values, numpy_dot_runtimes)
params_n2_dot, _ = curve_fit(o_n2, n_values, numpy_dot_runtimes)

# Extract the coefficients (a and b)
a_n3 = params_n3[0]
b_n2 = params_n2[0]
c_n1 = params_n1[0]
c_n2_dot = params_n2_dot[0]
d_n4 = params_n4[0]


N_fit = np.linspace(min(n_values), max(n_values), 100)
fit_n3 = o_n3(N_fit, a_n3)
fit_n2 = o_n2(N_fit, b_n2)
fit_n1 = o_n1(N_fit, c_n1)
fit_n2_dot = o_n2(N_fit, c_n2_dot)
fit_n4 = o_n4(N_fit, d_n4)

# Plot the runtimes against explicit
plt.figure(figsize=(10, 6))
plt.plot(n_values, explicit_runtimes, label="Explicit")
plt.plot(n_values, numpy_dot_runtimes, label="Numpy Dot")
plt.plot(N_fit, fit_n3, label="O(N^3) Fit against explicit")
plt.plot(N_fit, fit_n2, label="O(N^2) Fit against explicit")
plt.plot(N_fit, fit_n4, label="O(N^4) fit against explicit")

# plt.plot(N_fit, fit_n1, label="O(n) fit against dot()")
# plt.plot(N_fit, fit_n2_dot, label = "O(n^2) fit against dot()")

plt.xlabel("N")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison for Different N Values")
plt.legend()
plt.grid(True)
plt.savefig("Tests.png")

plt.show()



# Plot the runtimes against dot()
plt.figure(figsize=(10, 6))
#plt.plot(n_values, explicit_runtimes, label="Explicit")
plt.plot(n_values, numpy_dot_runtimes, label="Numpy Dot")
# plt.plot(N_fit, fit_n3, label="O(N^3) Fit against explicit")
# plt.plot(N_fit, fit_n2, label="O(N^2) Fit against explicit")
# plt.plot(N_fit, fit_n4, label="O(N^4) fit against explicit")
plt.plot(N_fit, fit_n1, label="O(n) fit against dot()")
plt.plot(N_fit, fit_n2_dot, label = "O(n^2) fit against dot()")
plt.xlabel("N")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison for Different N Values")
plt.legend()
plt.grid(True)
plt.savefig("Tests_dot.png")

plt.show()
