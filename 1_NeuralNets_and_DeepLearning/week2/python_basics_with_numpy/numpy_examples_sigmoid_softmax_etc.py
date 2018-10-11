import math
import numpy as np
import time

# Define sigmoid function using math library exponentiation
def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x));
    return s;

print(basic_sigmoid(3));
    
print("")

# Generates error when using the math library
#  due to lack of vectorization
# x = [1, 2, 3]
# print(basic_sigmoid(x));

# Define sigmoid function using numpy
def sigmoid(x):
    s = 1 / (1 + np.exp(-x));
    return s;

np_x = np.array([1, 2, 3]);
print(sigmoid(np_x));

# Compute gradient of sigmoid function
def sigmoid_derivative(x):
    s  = sigmoid(x);
    ds = s * (1 - s);
    return ds;
    
print("Sigmoid derivative(x) = " + str(sigmoid_derivative(np_x)));

# Reshaping arrays
# Argument:
# image -- a numpy array of shape (length, height, depth)
# Returns:
# a vector of shape (length*height*depth, 1)
    
def image2vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1);

image = np.array([
       [[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))

# Normalizing rows
#  Implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
#  Divide each row vector of the matrix x by its norm. Each row turns into a vector of unit length
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True); 
    return x / x_norm;

x = np.array([[0, 3, 4], [1, 6, 4]]);
print("normalizeRows(x) = " + str(normalizeRows(x)))
    
# Broadcasting and the softmax function
#  Calculates the softmax for each row of the input x.
#  Argument: x -- A numpy matrix of shape (n,m)
#  Returns: s -- A numpy matrix equal to the softmax of x, of shape (n,m)
def softmax(x):
    x_exp = np.exp(x);
    x_sum = np.sum(x_exp, axis = 1, keepdims = True);
    return x_exp / x_sum;
    
x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0 ,0]]);
print("softmax(x) = " + str(softmax(x)))
    
# Vectorization
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.time()
dot = np.dot(x1,x2)
toc = time.time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.time()
outer = np.outer(x1,x2)
toc = time.time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.time()
mul = np.multiply(x1,x2)
toc = time.time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.time()
dot = np.dot(W,x1)
toc = time.time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    
# Implement the L1 and L2 loss functions
#  Arguments: yhat -- vector of size m (predicted labels)
#             y    -- vector of size m (true labels)
#  Returns:   loss -- the value of the loss function 
          
def L1(yhat, y):
    return np.sum(abs(y - yhat));
    
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

def L2(yhat, y):
    return np.sum((y - yhat) ** 2)    
    # Also works:
    # return np.dot(y-yhat, y-yhat); 
    
print("L2 = " + str(L2(yhat,y)))
    
    
    
    
    
    


