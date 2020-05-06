# Loading the saved model
from keras.models import load_model
classifier = load_model('cats_vs_dogs.h5')

# Getting all images in the new test folder
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("dataset/single_prediction") if isfile(join("dataset/single_prediction", f))]

# Testing all the images 
import numpy as np
from keras.preprocessing import image
results = []
for img in onlyfiles:
	if (img.startswith('.')):
		continue
	test_image = image.load_img("dataset/single_prediction/" + img,target_size=(64,64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = classifier.predict(test_image)
	if (result[0][0] == 0):
		results.extend([img+":cat"])
	else:
		results.extend([img+":dog"])

for i in range(len(results)):
	print(results[i])

# mapping between cats and dogs
# training_set.class_indices # {cats:0, dogs:1}