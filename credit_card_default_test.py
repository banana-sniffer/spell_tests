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

from keras.models import load_model
classifier = load_model('taiwan_default.h5')

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
