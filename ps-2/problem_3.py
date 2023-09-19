import numpy as np
import matplotlib.pyplot as plt

x_start, x_end = -2, 2
y_start, y_end = -2, 2
N = 1000

def mandelbrot(x_start, x_end, y_start, y_end, N):
    """This generates an image of the mandelbrot set within a set of x and y bounds

    Args:
        x_start: x start bound
        x_end: x end bound
        y_start: y start bound
        y_end: y end bound
        N (_type_): Number of divisions between x and y in which to evaluate + 1

    Returns:
        z: mandelbrot set
    """
    delta = (x_end - x_start)/N

    real, imaginary = np.mgrid[x_start:x_end:delta, y_start:y_end:delta]
    c = (real + 1j*imaginary).reshape(imaginary.shape[0], -1).T

    z = np.zeros_like(c)

    for i in range(100):
        z = z*z + c 
    return z
    
plt.figure(figsize=(10,10))
plt.title("Mandelbrot Set")
plt.xlabel("X value")
plt.ylabel("Y value")
plt.imshow(np.absolute(mandelbrot(x_start, x_end, y_start, y_end, N)), extent=(x_start,x_end,y_start,y_end))
plt.show()