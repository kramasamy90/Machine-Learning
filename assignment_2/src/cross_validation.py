import numpy as np
import matplotlib.pyplot as plt
import math
import grad_descent_ridge

e = math.exp(1)
plt.rcParams['text.usetex'] = True

class cross_validate:
	def __init__(self, folder, w_ml):
		self.w_ml = w_ml
		self.X_train = np.load(folder + "/X_train.npy")
		self.Y_train = np.load(folder + "/Y_train.npy")
		self.X_validate = np.load(folder + "/X_validate.npy")
		self.Y_validate = np.load(folder + "/Y_validate.npy")
	
	def do_cross_validate(self, lower, upper, steps, are_log_vals=False):
		self.lower_lambda = lower
		self.upper_lambda = upper
		self.are_log_vals = are_log_vals
		self.errors_in_validation = []
		self.lambdas = np.linspace(lower, upper, steps)	
		if(are_log_vals): self.lambdas = e ** self.lambdas
		self.min_error = np.inf

		for _lambda in self.lambdas:
			# print("lambda: " + str(_lambda))
			g = grad_descent_ridge.grad_descent_ridge(self.X_train, self.Y_train, self.w_ml, _lambda)
			g.do_grad_descent(1000)
			error = np.linalg.norm(self.X_validate.T@(g.w) - self.Y_validate)
			if(error < self.min_error):
				self.min_error = error
				self.min_lambda = _lambda
				self.w = g.w
			self.errors_in_validation.append(error)

		self.errors_in_validation = np.array(
			self.errors_in_validation)

	def plot_error_lambda(self, corename):
		x = self.lambdas
		if(self.are_log_vals): x = np.log(x)
		y = self.errors_in_validation

		plt.style.use("classic")
		fig, ax = plt.subplots()
		ax.plot(x, y)
		ax.scatter(x, y)
		if(self.are_log_vals): ax.set_xlabel("log lambda")
		else: ax.set_xlabel("lambda")
		ax.set_ylabel("Error in Validation set")

		if(self.are_log_vals): scale = "log"
		else: scale  = "lin"
		filename = "./plots/" + corename + "_" + "validation_" + str(self.lower_lambda) + "_" + str(self.upper_lambda) + "_" + str(scale) + ".png"

		fig.savefig(filename)
	
	def test(self):
		print(self.X_train.shape)
		print(self.Y_train.shape)
		print(self.X_validate.shape)
		print(self.Y_validate.shape)