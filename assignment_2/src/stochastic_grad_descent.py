import numpy as np
import matplotlib.pyplot as plt
import grad_descent

plt.rcParams['text.usetex'] = True

class stochastic_grad_descent(grad_descent.grad_descent):
	def get_gradient(self):
		selection = np.random.choice(self.X.shape[1], 100, replace=False)
		selection.sort()
		X = self.X[:, selection]
		Y = self.Y[selection]
		w = self.w
		grad = X.T@w - Y
		grad = 2 * X @ grad
		return (grad / np.linalg.norm(grad))