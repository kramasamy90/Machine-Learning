import numpy as np
import matplotlib.pyplot as plt
import math
import grad_descent

e = math.exp(1)
plt.rcParams['text.usetex'] = True

class grad_descent_ridge(grad_descent.grad_descent):
	def __init__(self, X, Y, w_ml, _lambda):
		super().__init__(X, Y, w_ml)
		self._lambda = _lambda

	def get_gradient(self):
		X = self.X
		Y = self.Y
		w = self.w
		grad = X.T@w - Y
		grad = 2 * X @ grad
		grad = grad + self._lambda * w
		return (grad / np.linalg.norm(grad))

	def error(self):
		w = self.w
		X = self.X
		Y = self.Y
		return ((np.linalg.norm(X.T@w - Y)) ** 2 + (self._lambda * np.linalg.norm(w) ** 2))
	
	def load_date(self, path):
		self.X_train = np.load(path)
		self.Y_train = np.load(path)
		self.X_validate = np.load(path)
		self.Y_validate = np.load(path)


	def do_several_grad_descent(self, lower, upper, log_lambda=False):
		lambdas = np.linspace(lower, upper, 50)	
		if(log_lambda): lambdas = e ** lambdas
		for _lambda in lambdas:
			g = self.grad_descent_ridge(X, Y, w, _lambda)
			g.do_grad_descent(1000)
