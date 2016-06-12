import numpy as np
import matplotlib.pyplot as plt

"""
Analytic Centering.

find 

    x \in {x : Ax \preceq b}
    
which minimizes

    -\sum_i \log(b_i - a_i^T x)

Here we have the following polygon:

    y >= 0
    x >= 0
    x + y <= 5
    y <= 3

"""

print('==> Plotting the progression of analytic centering')
fig = plt.figure(figsize=(8,5))

def f(x, A, b):
    return -np.sum(np.log(b - np.dot(A, x)))

def df(x, A, b):
    scale = 1./(b - np.dot(A, x))
    return np.multiply(A, scale).sum(axis=0).reshape(-1,1)

A = np.array([
    [-1, 0],
    [0, -1],
    [1, 1],
    [0, 1]
])

b = np.array([0, 0, 5, 3]).reshape(-1,1)

N = 1000
x = np.linspace(0,5, N)
y = np.linspace(0,3, N)

X, Y = np.meshgrid(x, y)
Z = np.zeros((x.shape[0], y.shape[0]))

print('==> Inneficiently calculating Z values of surface')
# this is super inefficient but 3d plots
# in matplotlib are infuriating
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        x_vec = np.array([X[i,j], Y[i,j]]).reshape(-1,1)
        diff = b - np.dot(A, x_vec)
        if (diff > 1e-6).all():
            Z[i,j] = -np.sum(np.log(diff))

print('--  done.')
print('==> Plotting contours.')

# plot contours of objective function
contours = 15
plt.contour(X, Y, Z, contours)

######################################################
##                    Optimization                  ##
######################################################

print('==> Finding the analytic center using gradient descent.')

x_ = np.array([3,0.2]).reshape(-1,1)
xs = [x_.copy()]
fs = [f(x_, A, b)]
iters = 100
lr = 1e-1

for i in range(iters):
    x_ -= lr*df(x_, A, b)
    xs.append(x_.copy())
    fs.append(f(x_, A, b))

xs = np.array(xs).reshape(-1,2)

plt.plot(xs[:,0], xs[:,1], color='red', label='Gradient Descent Path', zorder=1)
plt.scatter(x_[0], x_[1], color='black', marker='*', zorder=2)

plt.axis('off')
plt.tight_layout()
plt.savefig('../fig/analytic_centering.pdf')

print('--  done.')
