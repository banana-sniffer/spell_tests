from tensorflow.python.client import device_lib
device_info = device_lib.list_local_devices()
file = open("Spell_V100.txt","w")
file.write(str(device_info))
file.close()


# from keras import backend as K 
# K.tensorflow_backed._get_available_gpus()

