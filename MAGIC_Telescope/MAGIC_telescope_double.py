import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('magic04.csv')
X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values

# gamma = True, background = False
for i in range(len(y)):
	if y[i] == "g":
		y[i] = 1
	else:
		y[i] = 0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
file = open("MAGIC_Telescope_accuracy_double.txt","a")
file.write("# of Hidden Nodes" + "          " + "Accuracy\n")
for j in range(7,11):
	classifier = Sequential()
	classifier.add(Dense(units = j, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 10))
	classifier.add(Dense(units = j, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)
	
	cm_predicted_values = []
	for i in range(len(y_pred)):
		if (y_pred[i][0] == 1):
			cm_predicted_values.append(1)
		else:
			cm_predicted_values.append(0)
	cm_y_test = []
	for i in range(len(y_test)):
		if (y_test[i] == 1):
			cm_y_test.append(1)
		else:
			cm_y_test.append(0)
	cm = confusion_matrix(cm_y_test, cm_predicted_values)
	accuracy = (cm[0][0] + cm[1][1]) / len(cm_predicted_values)
	file.write(str(j) + "                 " + "          " + str(accuracy) + "\n")

file.close()










