from keras.models import load_model

classifier = load_model('cats_vs_dogs.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg",target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
# mapping between cats and dogs
training_set.class_indices # {cats:0, dogs:1}