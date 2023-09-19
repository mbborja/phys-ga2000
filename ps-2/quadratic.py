import numpy as np

def quadratic_part_a(a,b,c):
    """This uses the part a quadratic formula to find x

    Args:
        a (_type_): quadratic term coefficient
        b (_type_): linear term coefficient
        c (_type_): coefficient

    Returns:
        x_plus: evaluation of x when +- is +
        x_minus: evaluation of x when +- is -
    """
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    
    x_plus = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    x_minus = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return x_plus, x_minus

def quadratic_part_b(a,b,c):
    """This uses the part b quadratic formula to find x

    Args:
        a (_type_): quadratic term coefficient
        b (_type_): linear term coefficient
        c (_type_): coefficient

    Returns:
        x_plus: evaluation of x when -+ is -
        x_minus: evaluation of x when -+ is +
    """
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    
    x_plus = ((2*c)/(-b - np.sqrt(b**2 - 4*a*c)))
    x_minus = ((2*c)/(-b + np.sqrt(b**2 - 4*a*c)))
    return x_plus, x_minus

def quadratic(a,b,c):
    """This uses the approriate formula from parts a and b such that subtraction of numbers with similar magnitudes is limited

     Args:
        a (_type_): quadratic term coefficient
        b (_type_): linear term coefficient
        c (_type_): coefficient

    Returns:
        x_plus: evaluation of x positive
        x_minus: evaluation of x negative
    """
    # We want to try to use expressions for x that don't involve subtraction of numbers of similar magnitudes
    
    # Thus, we use the expression for x_plus in part a) and the expression for x_minus from part b) since they
    # use expressions that involve addition

    # The problem term is b so we must first check its sign before deciding 
    # Which of the formula to use
    
    if b > 0:
        x_plus = 2*c/(-b - np.sqrt(b**2 - 4*a*c))
        x_minus = (-b - np.sqrt(b**2-4*a*c))/2/a 
    else:
        x_plus = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        x_minus = x_minus = ((2*c)/(-b + np.sqrt(b**2 - 4*a*c)))
    
        
    return x_plus , x_minus
    