import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import data

def plot_data(d, FileName, Title):
	fig, ax = plt.subplots()
	ax.scatter(d.X[0, :], d.X[1, :], color="black")
	ax.set_xlabel("dim-1")
	ax.set_ylabel("dim-2")
	fig.suptitle(Title)
	fig.savefig(FileName)


def plot_data_polar(d, FileName, Title, col="black"):
	fig, ax = plt.subplots()
	X = d.X[0, :]
	Y = d.X[1, :]
	R = (X**2 + Y**2)**(0.5)
	Theta = np.arctan(Y/X)
	ax.scatter(R, Theta, c=col)
	ax.set_xlabel("Distance origin")
	ax.set_ylabel("Angle on x-axis")
	fig.suptitle(Title)
	fig.savefig(FileName)


def plot_pc(d, FileName, Title):
	fig, ax = plt.subplots()
	ax.scatter(d.X_pc[0, :], d.X_pc[1, :], color="black")
	ax.set_xlabel("PC-1")
	ax.set_ylabel("PC-2")
	fig.suptitle(Title)
	fig.savefig(FileName)


def plot_kernel_pc(d, FileName, Title):
	fig, ax = plt.subplots()
	ax.scatter(d.X_kernel_pc[0, :], d.X_kernel_pc[1, :], color="black")
	ax.set_xlabel("PC-1")
	ax.set_ylabel("PC-2")
	fig.suptitle(Title)
	fig.savefig(FileName)

def plot_cluster(d, FileName, Title):
	fig, ax = plt.subplots()
	ax.scatter(d.X[0, :], d.X[1, :], c=d.z)
	ax.set_xlabel("dim-1")
	ax.set_ylabel("dim-2")
	fig.suptitle(Title)
	fig.savefig(FileName)

def plot_errf(d, FileName, Title):
	fig, ax = plt.subplots()
	x = []
	for i in range(len(d.errs)):
		x.append(i + 1)
	ax.plot(x, d.errs, c="black")
	ax.scatter(x, d.errs, c="black")
	ax.set_xlabel("Iterations")
	ax.set_ylabel("Error")
	fig.suptitle(Title)
	fig.savefig(FileName)


def voronoi(d, FileName, Title):
	points = [[-30, 0], [20, 0], [0, -30], [0, 30]]
	for i in range(d.k):
		x = d.k_means[0, i]
		y = d.k_means[1, i]
		points.append([x, y])

	points = np.array(points)

	vor = Voronoi(points)

	fig, ax = plt.subplots()
	fig = voronoi_plot_2d(vor)	
	plt.scatter(d.X[0, :], d.X[1, :], c=d.z)
	ax = plt.gca()
	ax.set_xlabel("dim-1")
	ax.set_ylabel("dim-2")
	fig.suptitle(Title)
	fig.savefig(FileName)

if(__name__ == '__main__'):
	d = data.Data("Dataset.csv")