import numpy as np
import random
import utils
import plot

class Data:
	'''
	Data contains the following variables:
	*	self.X : Input data. Each column 
		corresponds to one data points.	
	*	self.X_centered: Obtained from 
		centering self.X
	*	self.pc: Eigenvectors obtained after
		running PCA. 
	*	self.K: Stores the kernel.
	'''

	# Constructor and general functions.

	def __init__(self, input_file):
		self.read_X(input_file)


	def read_X(self, input_file):
		'''
		** 	Read and store data in self.X as numpy 
			arrays.
		* 	INPUT: CSV file. Each row should correspond
			to a data point
		* 	OUTPUT: self.X. Contains the data as numpy
			matrix.
		 	-	Each column of this matrix corresponds 
				to a single data point.
		'''
		X_T = []
		with open (input_file) as f:
			for line in f:
				word_list = line.split(",")
				X_T.append([float(word_list[0]), float(word_list[1])])
		
		self.X = (np.array(X_T)).T

	
	def dim(self):
		print(self.X.shape)

	
	def head(self):
		'''
		*	Outputs the top 5x5 elements. 
		*	Kind of like the bash head.	
		'''
		n = min([5, self.X.shape[0]])
		m = min([5, self.X.shape[1]])
		print(self.X[0:n, 0:m])

	
	def center(self):
		'''
		*	Centers the data self.X.
		'''
		self.X_center = self.X
		n_rows = self.X_center.shape[0]
		for i in range(n_rows):
			self.X_center[i, :] = self.X_center[i, :] - self.X_center[i, :].mean()
		return self.X_center

	
	# Functions for PCA. (Q1)


	def gauss_kernel(self, sigma):
		'''
		* 	Outputs a Gaussian kernel for a given dataset
			and a given sigma.
		'''
		d, n = self.X.shape
		E = np.zeros(n * n).reshape(n, n)
		for i in range(n):
			for j in range(n):
				x = self.X[:, i]
				y = self.X[:, j]
				E[i, j] = -1 * (x - y).T @ (x - y) / (2 * sigma **2)
		E = E / (2 * (sigma ** 2))
		E = (np.exp(1)) ** E
		self.K = E
		return (E)


	def polynomial_kernel(self, degree):
		'''
		*	Outputs a polynomial kernel.		
		'''
		self.K = (1 + self.X.T @ self.X) ** degree
		return (self.K)
	

	def center_kernel(self):
		n = self.X.shape[1]
		_1n = np.ones(n * n).reshape(n, n)
		K = self.K
		self.K_centered = K - (_1n @ K) - (K @ _1n) + (_1n @ K @ _1n)
		return self.K_centered
		

	def pca(self, with_centering = True):
		'''
		*	Does PC analysis and stores the eigen vectors
		in the order of increasing eigen values within 
		self.pc
		*	By default does PCA with centered data.
		*	Returns a matrix where i-th column is i-th data 
			point represented in coordinates of the 
			principal components.
		'''
		if(with_centering): X = self.center()
		else: X = self.X
		E = utils.custom_eigh((X @ X.T))
		self.pc = E[1]
		self.X_pc = (self.X.T @ self.pc).T
		return self.X_pc

	
	def kernel_pca(self, kernel_type, param):
		'''
		*	Does kernel	PCA.
		*	kernel_type = {"gauss", "poly"}.
		*	- For gauss, param is sigma.
			- For poly, param is d.
		*	Returns a matrix where i-th column is i-th data 
			point represented in coordinates of the 
			principal components.
		'''
		d, n = self.X.shape
		if (kernel_type == "gauss"):
			K = self.gauss_kernel(param)
		elif (kernel_type == "poly"):
			K = self.polynomial_kernel(param)
		K = self.center_kernel()
		E = utils.custom_eigh(K)
		A = E[1][:, 0:min(n, d)] / (E[0][0:min(n, d)] ** (0.5))
		i = 0
		self.X_kernel_pc = (K @ A).T
		return self.X_kernel_pc


	def get_vars(self):
		'''
		*	Returns an array vars, vars[i] denotes the
			variance of the data along the i-th largest 
			principal component.
		'''
		A = (self.X).T @ self.pc
		n_cols = A.shape[1]
		vars = np.zeros(n_cols)
		for i in range(n_cols):
			vars[i] = np.linalg.norm(A[:, i])
		return vars
	
	# Functions for clustering. (Q2)

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


	def get_H(self, kernel_type, param):
		if(kernel_type=="poly"): K = self.polynomial_kernel(param)	
		elif(kernel_type=="gauss"): K = self.gauss_kernel(param)
		K = self.center_kernel()
		E = utils.custom_eigh(K)
		self.H = E[1][:, 0:self.k]
		n = self.H.shape[0]
		for i in range(n):
			self.H[i, :] = self.H[i, :] / np.linalg.norm(self.H[i, :])


	def spectral_clustering(self, k, kernel_type, param):
		self.k = k
		old_X = self.X
		self.get_H(kernel_type, param)
		self.X = self.H.T
		self.lloyd(k)
		self.X = old_X
	

	def novel_clustering(self, k, kernel_type, param):
		if(kernel_type=="poly"): K = self.polynomial_kernel(param)	
		elif(kernel_type=="gauss"): K = self.gauss_kernel(param)
		n = self.X.shape[1]
		K = self.center_kernel()
		E = utils.custom_eigh(K)
		H = E[1][:, 0:k]
		self.z = np.zeros(n)
		for i in range(n):
			self.z[i] = np.argmax(H[i, :])

		

if(__name__ == '__main__'):
	d = Data("Dataset.csv")
