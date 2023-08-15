import numpy as np
import matplotlib.pyplot as plt
import math
import grad_descent
import stochastic_grad_descent
import grad_descent_ridge
import cross_validation

plt.rcParams['text.usetex'] = True
# Read input and extract X and Y.
# ===============================
train_data = np.loadtxt("./data/A2Q2Data_train.csv", delimiter=",")
X = train_data[:, 0:100].T
Y = train_data[:, 100:101]

## Read test data
test_data = np.loadtxt("./data/A2Q2Data_test.csv", delimiter=",")
X_test = test_data[:, 0:100].T
Y_test = test_data[:, 100:101]


# Q2-i Least Square solution w_ml to the regression problem.
w_ml = np.linalg.inv(X@X.T)@X@Y
print("Q2-i: First five values of w_ml")
print(w_ml[0:5])
print("Q2-i: Last five values of w_ml")
print(w_ml[-5:])


# Q2-ii 
g1 = grad_descent.grad_descent(X, Y, w_ml)
g1.do_grad_descent(1000)
g1.plot_error("q2_ii")
g1.plot_diff_wt_wml("q2_ii")
g1.print_report()
print("Error in test data for w in gradient descent is:")
print(np.linalg.norm(X_test.T@g1.w - Y_test) ** 2)


# Q2-iii

## Do stochastic gradient descent with a limit of 1000 iterations.

g2 = stochastic_grad_descent.stochastic_grad_descent(X, Y, w_ml)
g2.do_grad_descent(1000)
g2.plot_error("q2_iii_1000")
g2.plot_diff_wt_wml("q2_iii_1000")
g2.print_report()

## Do stochastic gradient descent with a limit of 10000 iterations.

g2 = stochastic_grad_descent.stochastic_grad_descent(X, Y, w_ml)
g2.do_grad_descent(10000)
g2.plot_error("q2_iii_10000")
g2.plot_diff_wt_wml("q2_iii_10000")
g2.print_report()


# Q2-iv 

C = cross_validation.cross_validate("intermediate_data", w_ml)
C.do_cross_validate(lower=0, upper=5, steps=100, are_log_vals=False)
C.plot_error_lambda("q2_iv")


print("min lambda:")
print(C.min_lambda)
print("Error in test data for w_R is:")
print(np.linalg.norm(X_test.T@C.w - Y_test) ** 2)
print("Error in test data for w_ML is:")
print(np.linalg.norm(X_test.T@w_ml - Y_test) ** 2)