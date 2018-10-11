import numpy as np

x = np.random.randn(32,32,3);

print(x.shape);

y = x.reshape(32*32*3, 1);

print(y.shape);

print("");

a = np.random.randn(2, 3);
b = np.random.randn(2, 1);

c = a + b;

print(c.shape)

#d = np.random.randn(4, 3);
#e = np.random.randn(3, 2);

#f = d * e;

#print(f.shape)

print("");

a = np.random.randn(3, 4);
b = np.random.randn(4, 1);

res1 = np.zeros((3,4));

for i in range(3):
    for j in range(4):
        res1[i][j] = a[i][j] + b[j]
        
print(res1)

res2 = a + b.T;

print("")

print(res2)

print("")

a = np.random.randn(3, 3);
b = np.random.randn(3, 1);
c = a * b;

print(c.shape)

