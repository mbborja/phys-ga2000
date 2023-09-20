import numpy as np

# Store Original Value
original_value = 100.98763

# Cast original value to np.float32
value = np.float32(original_value)

# The method binary_repr returns the binary representation of the original 
# value according to the 1EEE 754 Standards
binary_rep = np.binary_repr(value.view(np.int32), width = 32)
print("Binary representation: " + binary_rep)
# Try to recreate the original value using the binary rep stored value
sign = int(binary_rep[0], 2) # First binary digit represents the sign
exponent= int(binary_rep[1:9], 2) # Next 8 binary digits the exponent in 2^{exponent}
mantissa = int(binary_rep[9:], 2) # Next 23 binary digits will decide the mantissa

# Mantissa is represented like this:
# 1.____ where ____ is described by the following  
# 1/2 1/4 1/8 1/16 1/32 ...
# Example: 1.5 = 10000000000000000000000  

# Formula is sign*2^{exponent}*{mantissa}

# Convert the sign, exponent, and mantissa to decimal
if sign == 0:
    sign = 1
elif sign == 1:
    sign = -1
exponent = exponent - 127 # We subtract 127 and exponents from -126 to 127 can be represented
mantissa = 1 + mantissa / (2**23)

# Construct the float32 value
float_value = sign * 2**exponent * mantissa

print("Original Value: " + str(original_value) + "\n" + "Float Value: " + str(float_value))

difference = original_value - float_value
print("Difference: " + str(difference))


