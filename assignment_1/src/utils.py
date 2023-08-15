import numpy as np


def custom_eigh(X):
	E = np.linalg.eigh(X)
	eigval = E[0][::-1]

	eigvec = []
	n = len(E[1])
	for i in range(n):
		eigvec.append(E[1][i][::-1])
	eigvec = np.array(eigvec)

	return (eigval, eigvec)


if (__name__ == '__main__'):
	x = np.linspace(1, 9, 9).reshape(3, 3)
	custom_eigh(x)