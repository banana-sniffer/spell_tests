import numpy as np 
import pandas as pd 

data = pd.read_csv('credit_card_client_defaults.csv')
x = data.iloc[1:,1:24].values
y = data.iloc[1:, 24].values

for i in range(len(x)):
	x[i] = list(map(int,x[i]))
y = list(map(int,y))

for i in range(len(x[:,2])):
	if x[:,1][i] == 2:
		x[:,1][i] = 0
	if x[:,2][i] > 4 or x[:,2][i] == 0:
		x[:,2][i] = 4
	if x[:,3][i] > 3 or x[:,3][i] == 0:
		x[:,3][i] = 3

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
cT = ColumnTransformer(transformers=[('cT',OneHotEncoder(categories='auto',drop='first'),[2,3])],remainder='passthrough')
x = cT.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



























