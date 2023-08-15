import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

class grad_descent:
	def __init__(self, X, Y, w_ml):
		self.X = X
		self.Y = Y
		self.w_ml = w_ml
		self.error_t = []
		self.d = X.shape[0]
		self.n = X.shape[1]
		self.w = np.zeros(self.d).reshape(self.d, 1)
		self.diff_wt_wml = []
	
	def get_gradient(self):
		X = self.X
		Y = self.Y
		w = self.w
		grad = X.T@w - Y
		grad = 2 * X @ grad
		return (grad / np.linalg.norm(grad))
	
	def error(self):
		w = self.w
		X = self.X
		Y = self.Y
		return (np.linalg.norm(X.T@w - Y)) ** 2
	
	def do_grad_descent(self, n_iter):
		# err = self.error()	
		self.error_t.append(self.error())
		self.diff_wt_wml.append(np.linalg.norm(self.w - self.w_ml))
		eta = 0.01
		for i in range(n_iter):
			grad = self.get_gradient()
			self.w = self.w - eta * grad
			self.error_t.append(self.error())
			self.diff_wt_wml.append(np.linalg.norm(self.w - self.w_ml))
			if(abs(self.error_t[i+1] - self.error_t[i]) < 0.001): break

	def plot_error(self, q_no):
		plt.style.use("classic")
		x = np.arange(0, len(self.error_t), 1)
		y = np.array(self.error_t)
		fig, ax = plt.subplots()
		ax.plot(x, y)
		ax.set_xlabel("Iteration number", fontsize=16)
		ax.set_ylabel(r'$Error_{(t+1)} - Error_t$', fontsize=16)
		filepath = "plots/" + q_no + "_error_diff.png"
		fig.savefig(filepath)
	
	def plot_diff_wt_wml(self, q_no):
		plt.style.use("classic")
		x = np.arange(0, len(self.error_t), 1)
		y = np.array(self.diff_wt_wml)
		fig, ax = plt.subplots()
		ax.plot(x, y)
		ax.set_xlabel("Iteration number", fontsize=16)
		ax.set_ylabel(r'$\mid w - w_{ML}\mid _{2}$', fontsize=16)
		filepath = "plots/" + q_no + "_diff_wt_wml.png"
		fig.savefig(filepath)

	
	def print_report(self):
		print("Number of rounds of gradient descent = ", len(self.error_t))