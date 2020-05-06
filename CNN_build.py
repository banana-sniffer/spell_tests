# Building the CNN

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Getting the dataset from spell
import sys
uploaded_training_set = sys.argv[1]
uploaded_test_set = sys.argv[2]

# Initializing the CNN
classifier = Sequential()

# Step - 1 Convultion
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

# Step - 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step - 5 Adding another convulational layer can increase our accuracy!
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step - 3 Flattening
classifier.add(Flatten())

# Step - 4 Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        uploaded_training_set,
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        uploaded_test_set,
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=8,
        validation_data=test_set,
        validation_steps=2000)

# Making new predictions
# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg",target_size=(64,64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image,axis=0)
# result = classifier.predict(test_image)
# # mapping between cats and dogs
# training_set.class_indices # {cats:0, dogs:1}

# Saving the model that we trained
from keras.models import load_model
classifier.save('cats_vs_dogs.h5')







