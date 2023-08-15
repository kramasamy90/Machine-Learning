import numpy as np

train_data = np.loadtxt("./data/A2Q2Data_train.csv", delimiter=",")
X = train_data[:, 0:100].T
Y = train_data[:, 100:101]

indices = [i for i in range(10000)]
np.random.shuffle(indices)
train_indices = indices[0:8000]
validate_indices = indices[8000:10000]
X_train = X[:, train_indices]
Y_train = Y[train_indices]
X_validate = X[:, validate_indices]
Y_validate = Y[validate_indices]

np.save("X_train", X_train)
np.save("Y_train", Y_train)
np.save("X_validate", X_validate)
np.save("Y_validate", Y_validate)

