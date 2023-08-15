import numpy as np
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import em_bernoulli

# plt.rcParams['text.usetex'] = True
e = math.exp(1)


class em_gauss(em_bernoulli.em_bernoulli):

	def get_probability(self, k, i):
		return multivariate_normal.pdf(self.X[:, i], self.Mu[k, :], self.Sigma[k], allow_singular=True)
	
	def maximization(self):
		self.Pi = self.Lambda @ np.ones(self.N).reshape(self.N, 1)
		self.Pi = self.Pi / self.Pi.sum()
		self.Mu = (self.Lambda @ self.X.T) / (self.Lambda @ (np.ones(self.N*self.D).reshape(self.N, self.D)))
		self.Sigma = []
		for k in range(self.K):
			D_k = np.zeros(self.N * self.N).reshape(self.N, self.N)
			for i in range(self.N):
				D_k[i][i] = self.Lambda[k][i]
			V_1 = np.ones(self.N * self.D).reshape(self.N, self.D)
			self.Sigma.append(self.X @ D_k@ self.X.T / (self.Lambda @ V_1)[k, 0])

	def expectation(self):
		for i in range(self.N):
			for k in range(self.K):
				# self.Lambda[k, i] = np.random.multivariate_normal(self.Mu[k, :], self.Sigma[k])
				self.Lambda[k, i] = self.Pi[k] * self.get_probability(k , i)
		
		self.Lambda = self.Lambda / (np.ones(self.K).reshape(1, self.K) @ self.Lambda)

	def get_likelihood(self):
		ans = 0
		for i in range(self.N):
			prob = 0
			for k in range(self.K):
				prob = prob + self.Pi[k] * self.get_probability(k, i)
			ans = ans + np.log(prob)
		return ans
	
	def do_em(self, n_iters):
		self.likelihood = []
		for i in range(n_iters):
			print(i)
			self.expectation()
			self.maximization()
			self.likelihood.append(self.get_likelihood())
		self.likelihood = np.array(self.likelihood)
