import numpy as np

delta = 0.01

def f(x):
    f = x*(x-1)
    return f

def derivative(point, delta):
    value = (f(point+delta) - f(point))/delta
    return value

for i in range(2,15,2):
    print(f"Delta = {10**-i}, Derivative output= {derivative(1,10**(-i))}")
