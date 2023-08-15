import numpy as np
import math
import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True
e = math.exp(1)

class em_bernoulli:
	def __init__(self, X, K):
		self.X = X
		self.K = K
		self.D = X.shape[0]
		self.N = X.shape[1]

	def initialize(self):
		self.Lambda = np.zeros(self.K * self.N).reshape(self.K, self.N)
		for i in range(self.N):
			k = np.random.choice(self.K, 1)
			self.Lambda[k, i] =  1
		self.maximization()
	
	def get_probability(self, k, i):
		prob = 1
		for j in range(self.D):
			if(self.X[j, i] == 1):
				prob = prob * self.P[k, j]
			else:
				prob = prob * (1 - self.P[k, j])
		return prob


	def maximization(self):
		self.Pi = self.Lambda @ np.ones(self.N).reshape(self.N, 1)
		self.Pi = self.Pi / self.Pi.sum()
		self.P = (self.Lambda @ self.X.T) / (self.Lambda @ (np.ones(self.N*self.D).reshape(self.N, self.D)))
		self.P[self.P == 0] = 1e-16
		self.P[self.P == 1] = 1 - 1e-16
	
	def expectation(self):
		self.Lambda = np.log(self.P) @ self.X 
		self.Lambda = self.Lambda + (np.log(1 - self.P)) @ (1 - self.X) 
		self.Lambda = self.Lambda + np.log(self.Pi)
		self.Lambda = e ** self.Lambda
		for i in range(self.N):
			self.Lambda[:, i] = self.Lambda[:, i] / self.Lambda[:, i].sum()
		self.Lambda[self.Lambda == 0] = 1e-16
	
	def get_likelihood(self):
		ans = 0
		for i in range(self.N):
			prob_i = 0
			for k in range(self.K):
				prob_i = prob_i + self.Pi[k] * self.get_probability(k , i)
			ans = ans + np.log(prob_i)
		return ans
		
	
	def do_em(self, n_iter):
		self.likelihood = []
		for i in range(n_iter):
			# print(i)
			self.expectation()
			self.maximization()
			self.likelihood.append(self.get_likelihood())
		self.likelihood = np.array(self.likelihood)