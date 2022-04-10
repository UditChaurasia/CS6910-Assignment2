from IPython.display import Image
!pip install -q keras
from google.colab import drive
import keras
drive.mount('/content/drive')
!unzip "/content/drive/MyDrive/nature_12K.zip"
import os
import numpy as np
import random
import shutil
train_path = "/content/inaturalist_12K/train/"
train_data = "/content/inaturalist_12K/train_data/"
dataV = "/content/inaturalist_12K/dataV/"
test_data = "/content/inaturalist_12K/val/"
classes = sorted(os.listdir(train_path))
os.mkdir(dataV)
os.mkdir(train_data)
for i in range(0,len(classes) - 1):
  os.mkdir(dataV + classes[i + 1])
  os.mkdir(train_data + classes[i + 1])

  image_data = np.asarray(os.listdir(train_path + classes[i + 1]))
  print(len(image_data))
  total_number_of_files = len(os.listdir(train_path + classes[i + 1]))
  print(total_number_of_files)
  number_array = []
  for num in range(0,total_number_of_files):
    number_array.append(num)
  
  random.shuffle(number_array)
  for temp in range(0, len(number_array)):
    if (temp < 0.9*total_number_of_files):
      shutil.move(train_path + classes[i + 1] + "/" + image_data[number_array[temp]] , train_data + classes[i + 1] + "/" + image_data[number_array[temp]])
    else:
      shutil.move(train_path + classes[i + 1] + "/" + image_data[number_array[temp]] , dataV + classes[i + 1] + "/" + image_data[number_array[temp]])



from keras.models import Model,Sequential
import tensorflow
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Activation,Dropout,Flatten,BatchNormalization

def create_cnn_model(number_of_filters, filter_size, size_of_dense_layer, activation_function, batch_normalization, drop_out):
  model = Sequential()

  # ******************   Layer 1   **************************

  model.add(Conv2D(number_of_filters, (filter_size,filter_size), padding = 'same', input_shape=(224,224,3)))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 2   **************************
  
  model.add(Conv2D(number_of_filters*(2), (filter_size - 1,filter_size - 1), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 3   **************************
  
  model.add(Conv2D(number_of_filters*(4), (filter_size - 2,filter_size - 2), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 4   **************************
  
  model.add(Conv2D(number_of_filters*(8), (filter_size - 3,filter_size - 3), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 5   **************************
  
  model.add(Conv2D(number_of_filters*(16), (filter_size - 4,filter_size - 4), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************  Flattens the Convolution layer  ***************

  model.add(Flatten())


  # ******************  adding the dense layer  ********************

  model.add(Dense(size_of_dense_layer))
  model.add(Activation(activation_function))


  # ******************  adding Batch Normalization (if required)  *****************

  if (batch_normalization == 1):
    model.add(BatchNormalization())

  
  # ******************  adding Dropout  *************************

  model.add(Dropout(drop_out))


  # ******************  adding output layer  **********************

  model.add(Dense(10))
  model.add(Activation('softmax'))


  # *********************  returns final model  ******************************

  return model


from keras.preprocessing.image import ImageDataGenerator

def train_val_test_data(augment):
  data_val = ImageDataGenerator(rescale = 1./255)
  data_test = ImageDataGenerator(rescale = 1./255)

  if (augment):
    data_train = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, horizontal_flip = True)
  else:
    data_train = ImageDataGenerator(rescale = 1./255)

  training_data = data_train.flow_from_directory(train_data, target_size=(224,224), seed=10, batch_size=64,shuffle=True,class_mode="categorical",color_mode="rgb")
  data_for_validation = data_val.flow_from_directory(dataV,target_size=(224,224),seed=10,batch_size=64,shuffle=True,class_mode="categorical",color_mode="rgb")
  data_for_test = data_test.flow_from_directory(test_data,target_size=(224,224),seed=10,batch_size=64,shuffle=True,class_mode="categorical",color_mode="rgb")
  return training_data, data_for_validation, data_for_test  



training_data, data_for_validation, data_for_test = train_val_test_data(False)


final_model = create_cnn_model(16, 7, 512, "relu", False, 0.6)
final_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = 0.0002),loss='categorical_crossentropy',metrics=['accuracy'])
final_model.fit(training_data, validation_data = data_for_validation, epochs = 10)


final_model.save('best_model.h5')
final_model = tensorflow.keras.models.load_model('best_model.h5')

!pip install wandb
import wandb
wandb.login


wandb.init(project='PartB_BackProp',entity='uditch')


import matplotlib.pyplot as plt
import cv2

# ********************** Plots Image ***********************
image, label = data_for_test.next()
test_img = image[random.randint(0, len(image) - 1)]
plt.imshow(test_img);



# ************** Guided Relu ****************************
@tensorflow.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tensorflow.cast(dy > 0,"float32") * tensorflow.cast(x > 0, "float32")*dy
    return tensorflow.nn.relu(x), grad



imageMat = []
test_img = np.expand_dims(test_img, axis = 0)
id = []
dimension = [[None]*3]*10
grad = []
for vr in range(10):

  for tmp in range(1,4):
    if tmp == 1:
      d1 = np.random.randint(final_model.layers[9].output.shape[tmp])
    elif tmp == 2:
      d2 = np.random.randint(final_model.layers[9].output.shape[tmp])
    elif tmp == 3:
      d3 = np.random.randint(final_model.layers[9].output.shape[tmp])

  guided_single_model = Model(inputs = final_model.input, outputs = final_model.layers[9].output[:,d1,d2,d3])
    
  for i in range(1, len(guided_single_model.layers)):
    temp_layer = guided_single_model.layers[i]
    if hasattr(temp_layer, 'activation') and temp_layer.activation == tensorflow.keras.activations.relu:
      temp_layer.activation = guidedRelu


    
    
  with tensorflow.GradientTape() as tape:
    inputs = tensorflow.cast(test_img, tensorflow.float32)
    tape.watch(inputs)
    outputs = guided_single_model(inputs)
        
  grad.append(tape.gradient(outputs,inputs))
  dimension[vr][0] = d1
  dimension[vr][1] = d2
  dimension[vr][2] = d3



for gd in range(0,10):
  gradTemp = grad[gd][0].numpy()
  gradTemp -= np.min(gradTemp)
  var_check = gradTemp.max() - gradTemp.min()
  if (var_check):
    gradTemp /= var_check

  imageMat.append(np.clip(gradTemp, 0, 1))
  id.append("Neuron Index "+"("+str(dimension[gd][0]) + "," + str(dimension[gd][1]) + "," + str(dimension[gd][2])+")")
  

# ********************* Plotting the Image ************************
  
  
  plt.figure(figsize = (3,3))
  #vectorize = plt.subplot(2,5,gd + 1)
  plt.imshow(imageMat[gd])



plt.figure(figsize = (30,40))


for tmp1 in range(0,10):

  vectorize = plt.subplot(2,5,tmp1 + 1)
  plt.imshow(imageMat[tmp1])


wandb.log({"Test Image": [wandb.Image(test_img,caption="test_random_image")]})
wandb.log({"Guided Backprop Image": [wandb.Image(img, caption=lbl) for img,lbl in zip(imageMat,id)]})





