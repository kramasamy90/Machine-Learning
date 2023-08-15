# Import libraries and other python files.
import numpy as np
import data
import plot
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

#### General variables
file_type = ".png"


# Read  and explore data:
#************************

#### Read data
d = data.Data("Dataset.csv")

#### Plot the original data
FileName =  "plots/" + "data" + file_type
plot.plot_data(d,FileName, "Plot of data on cartesian coordinates")

#### Plot the data in polar coordinates.
FileName =  "plots/" + "data_polar" + file_type
plot.plot_data_polar(d, FileName, "Plot of data on polar coordinates")

# Q1
#***

## Q1-i
##=====
'''
Q1 - i
*	Write a piece of code to run the PCA algorithm on this data-set. 
*	How much of the variance in the data-set is explained by each of
	the principal components?
'''

#### Do PCA with centering.
print("PCA with centering")
print("==================")
d.pca() # By default this does PCA by centering the data.
pc = d.pc
for i in range(len(pc[0])):
	print("PC -", i+1, " : ", pc[:, i])
print("\n")

#### Plot data on PC coordinates
FileName =  "plots/" + "PCA" + file_type
plot.plot_pc(d, FileName, "Plot on PC coordinates(w/ centering)")

#### Find variance along each PCs.
print("Variance along each of the principal components")
print("-----------------------------------------------")
vars = d.get_vars()
for i in range(len(vars)):
    print("Variance along PC-", i+1, " is ", (vars/vars.sum() * 100)[i], "%.")
print("\n")

## Q1-ii
##======
'''
Q1 - ii
*	Study the effect of running PCA without centering the data-set.
*	What are your observations?	
*	Does Centering help?
'''

#### PCA without centering
print("PCA without centering")
print("=====================")
d.pca(with_centering=False)
pc = d.pc
for i in range(len(pc[0])):
	print("PC -", i+1, " : ", pc[:, i])
print("\n")

#### Plot data on PC coordinates
FileName =  "plots/" + "PCA_wo_center" + file_type
plot.plot_pc(d, FileName, "Plot on PC coordinates (w/o centering)")

#### Find variance along each PCs.
print("Variance along each of the principal components")
print("-----------------------------------------------")
vars = d.get_vars()
for i in range(len(vars)):
    print("Variance along PC-", i+1, " is ", (vars/vars.sum() * 100)[i], "%.")
print("\n")

#### Print mean of data
print("Print mean of the data points")
print("=============================")
avg = []
for i in range(len(d.X)):
	avg.append(d.X[i].mean())
print("Center of the data: ", avg) # The mean of the data is nearly zero.


## Q1-iii
##=======
'''
Q1 - iii
*	Write a piece of code to implement the Kernel 
	PCA algorithm on this dataset.
	Use the following kernels:
		A: Polynomial, d= {2, 3}.
		B: Gaussian, sigma = {0.1, 0.2, ..., 1}
*	Plot the projection of each point in the dataset 
	onto the top-2 components for each kernel.
	- Use one plot for each kernel and in the case of (B).
	- Use a different plot for each value of sigma
'''

### A. Kernel-PCA with polynomial kernel.
###-----------------------------------

for i in range(2, 4):
	d.kernel_pca(kernel_type="poly", param=i)
	FileName =  "plots/" + "data_kernel_PCA_poly_" + str(i) + file_type
	Title = "Data on PC coordinates (Kernel PCA with polynomial kernel d = " + str(i) + ")" 
	plot.plot_kernel_pc(d, FileName, Title)

### B. Kernel-PCA with Gaussian kernel.
###------------------------------------

for i in range(1, 11):
	d.kernel_pca(kernel_type="gauss", param=(i/10))
	FileName =  "plots/" + "data_kernel_PCA_gauss_" + str(i) + file_type
	Title = "Data on PC coordinates (Kernel PCA with gaussian kernel s = " + str(i/10) + ")" 
	plot.plot_kernel_pc(d, FileName, Title)


## Q1-iv
##======
'''
Q1 - iv
*	Which Kernel do you think is best suited for 
	this dataset and why?
>>	No code for this question. See report.
'''


# Q2
#***

## Q2-i
##=====
'''
Q2-i
*	Write a piece of code to run the algorithm 
	studied in class for the K-means
	problem with k = 4. 
*	Try 5 different random initialization and 
	plot the error function w.r.t iterations in 
	each case. 
*	In each case, plot the clusters obtained in
	different colors.
'''

for i in range(5):
	d.lloyd(4)

	FolderName = "plots/" 
	FileName = FolderName + "cluster_" + str(i+1) + file_type
	Title = "K means clustering (K =4). [" + str(i +1) + "]"
	plot.plot_cluster(d, FileName, Title)

	FileName = FolderName + "Error_Function_" + str(i+1) + file_type
	Title = "K means clustering (K =4). [" + str(i +1) + "]"
	plot.plot_errf(d, FileName, Title)


## Q2-ii
##=====
'''
Q2-ii
*	Fix a random initialization. For K = {2,3,4,5},
	obtain cluster centers according to K-means 
	algorithm using the fixed initialization. 
*	For each value of K, plot the Voronoi regions 
	associated to each cluster center. (You can assume 
	the minimum and maximum value in the data-set to be 
	the range for each component of R 2 ).
'''

### Plot Voronoi regions for K = 4.
###--------------------------------

d = data.Data("Dataset.csv")

for i in range(2, 6):
	d.lloyd(i)
	FileName = "plots/voronoi_k_" + str(i) + file_type
	Title = "Voronoi K = " + str(i)
	plot.voronoi(d, FileName, Title)


## Q2-iii
##=======
''''
Q2-iii
*	Run the spectral clustering algorithm (spectral relaxation of 
	K-means using Kernel-PCA) k = 4. 
*	Choose an appropriate kernel for this data-set and plot the 
	clusters obtained in different colors. Explain your choice of 
	kernel based on the output you obtain.
'''

for i in range(2, 4):
	d.spectral_clustering(4, "poly", i)	
	file_type = ".png"
	FileName = "plots/spectral_cluster_poly_" + str(i) + file_type
	FileNamePolar = "plots/spectral_cluster_poly_polar_" + str(i) + file_type
	Title = "Spectral clustering Polynomial Kernel d = " + str(i)
	plot.plot_cluster(d, FileName, Title)
	plot.plot_data_polar(d, FileNamePolar, Title, d.z)

for i in range(1, 11):
	d.spectral_clustering(4, "gauss", i/10)	
	file_type = ".png"
	FileName = "plots/spectral_cluster_gauss_" + str(i) + file_type
	FileNamePolar = "plots/spectral_cluster_gauss_polar_" + str(i) + file_type
	Title = "Spectral clustering Gaussian Kernel s = " + str(i)
	plot.plot_cluster(d, FileName, Title)
	plot.plot_data_polar(d, FileNamePolar, Title, d.z)
	

## Q2-iv
##=======
'''
Q2-iv
*	Instead of using spectral clustering use the method
	in the assignment.
*	How does this mapping perform on this dataset.
*	Explain your insights.
'''

for i in range(2, 4):
	d.spectral_clustering(4, "poly", i)	
	file_type = ".png"
	FileName = "plots/novel_cluster_poly_" + str(i) + file_type
	FileNamePolar = "plots/novel_cluster_poly_polar_" + str(i) + file_type
	Title = "Novel clustering Polynomial Kernel d = " + str(i)
	plot.plot_cluster(d, FileName, Title)
	plot.plot_data_polar(d, FileNamePolar, Title, d.z)

for i in range(1, 11):
	d.spectral_clustering(4, "gauss", i/10)	
	file_type = ".png"
	FileNamePolar = "plots/novel_cluster_gauss_polar_" + str(i) + file_type
	FileName = "plots/novel_cluster_gauss_" + str(i) + file_type
	Title = "Novel clustering Gaussian Kernel s = " + str(i)
	plot.plot_cluster(d, FileName, Title)
	plot.plot_data_polar(d, FileNamePolar, Title, d.z)
