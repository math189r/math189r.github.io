import numpy as np

np.random.seed(0)

X1 = np.random.randn(40,2)*0.5
X2 = np.random.randn(100,2)*0.5 + np.array([1,1])

y1 = np.zeros((X1.shape[0],))
y2 = np.ones((X2.shape[0],))

X = np.zeros((X1.shape[0]+X2.shape[0],3))
X[:X1.shape[0],:2] = X1
X[X1.shape[0]:,:2] = X2

X[:X1.shape[0],2] = y1
X[X1.shape[0]:,2] = y2

np.savetxt(
    'classification.csv', X, fmt='%f,%f,%d',
    header='hours_studied,grade_in_class,pass_exam',
)
