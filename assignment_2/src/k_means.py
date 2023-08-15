import numpy as np
import random
import matplotlib.pyplot as plt

class k_means:
	# Constructor and general functions.

	def __init__(self, X):
		self.X = X


	def lloyd(self, k):
		d, n = self.X.shape

		self.k = k
		self.z = np.full(1000, -1)
		self.k_means = np.zeros(d * k).reshape(d, k)
		self.errs = []
		self.iteration = 0
		self.is_static = False

		# Random initialization.
		self.rand_init(k)
		while(not self.is_static):
			self.calc_k_means()
			self.errs.append(self.calc_err())
			self.reassignment()

	
	def rand_init(self, k):
		d, n = self.X.shape
		cluster_ids = np.linspace(0, k-1, k)
		for i in range(n):
			self.z[i] = int(random.choice(cluster_ids))
	

	def calc_k_means(self):
		k = self.k
		d, n = self.X.shape

		freq = np.zeros(k)
		k_means = np.zeros(d * k).reshape(d, k)

		for i in range(n):
			k_means[:, self.z[i]] += self.X[:, i]
			freq[self.z[i]] += 1

		for i in range(k):
			if(freq[i] == 0): continue
			k_means[:, i] = k_means[:, i] / freq[i]

		self.k_means = k_means


	def calc_err(self):
		X = self.X
		d, n = X.shape

		err = 0
		for i in range(n):
			err += np.linalg.norm(X[:, i] - self.k_means[:, self.z[i]])
		
		return err
	
	def arg_min(self, i):
		x = self.X[:, i]
		k_means = self.k_means
		diff = np.linalg.norm(x - k_means[:, self.z[i]])
		k = self.k
		ans = self.z[i]
		for j in range(k):
			new_diff = np.linalg.norm(x - k_means[:, j])
			if(new_diff < diff): ans = j
		return ans
	
	def reassignment(self):
		d, n = self.X.shape
		self.is_static = True
		for i in range(n):
			new_i = self.arg_min(i)
			self.is_static = self.is_static and (self.z[i] == new_i)
			self.z[i] = self.arg_min(i)