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
		y[i] = True
	else:
		y[i] = False

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from keras.models import load_model
classifier.save('MAGIC_Telescope.h5')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0] + cm[1][1]) / len(y_pred)
file = open("MAGIC_Telescope.txt","w")
file.write("Accuracy: " + str(accuracy))
file.close()










