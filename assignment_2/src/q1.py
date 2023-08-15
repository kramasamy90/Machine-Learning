import numpy as np
import matplotlib.pyplot as plt
import math
import em_bernoulli
import em_gauss
import k_means

e = math.exp(1)

# Set parameters, read data
K = 4
X = np.loadtxt("./data/A2Q1.csv", delimiter=",")
X = X.T

# Q1-i
n_iters = 100
em = em_bernoulli.em_bernoulli(X, K)
likelihood = np.zeros(n_iters).reshape(n_iters, 1)
for i in range(100):
    print(i)
    em.initialize()
    em.do_em(n_iters)
    likelihood = likelihood + em.likelihood

likelihood = likelihood / 100
fig, ax = plt.subplots()
ax.plot(likelihood, 'b')
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Log Likelihood")
plt.plot(likelihood)
fig.savefig("plots/q1_i.pdf")
fig.savefig("plots/q1_i.png")


# Q1-ii
n_iters = 100
em = em_gauss.em_gauss(X, K)
likelihood = np.zeros(n_iters).reshape(n_iters, 1)
for i in range(10):
    em.initialize()
    em.do_em(n_iters)
    likelihood = likelihood + em.likelihood

likelihood = likelihood / 10
fig, ax = plt.subplots()
ax.plot(likelihood, 'b')
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Log Likelihood")
plt.plot(likelihood)
fig.savefig("q1_ii.pdf")
fig.savefig("q1_ii.png")


# Q1-iii

d = k_means.k_means(X)
d.lloyd(4)
fig, ax = plt.subplots()
ax.plot(d.errs)
ax.set_xlabel("Number of reassignments")
ax.set_ylabel("Value of objective function")
fig.savefig("plots/q1_iii.png")
fig.savefig("plots/q1_iii.pdf")
