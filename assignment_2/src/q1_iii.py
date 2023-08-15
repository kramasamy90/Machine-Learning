# Import libraries and other python files.
import numpy as np
import k_means
import matplotlib.pyplot as plt

#### General variables
file_type = ".png"

K = 4
X = np.loadtxt("./data/A2Q1.csv", delimiter=",")
X = X.T

# Q1-iii

d = k_means.k_means(X)
d.lloyd(4)
fig, ax = plt.subplots()
ax.plot(d.errs)
ax.set_xlabel("Number of reassignments")
ax.set_ylabel("Value of objective function")
fig.savefig("plots/q1_iii.png")
fig.savefig("plots/q1_iii.pdf")