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

def create_cnn_model(number_of_filters, filter_size, size_of_dense_layer, activation_function, batch_normalization, drop_out, layer_mult):
  model = Sequential()

  # ******************   Layer 1   **************************

  model.add(Conv2D(number_of_filters[0], (filter_size[0],filter_size[0]), padding = 'same', input_shape=(224,224,3)))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 2   **************************
  
  model.add(Conv2D(number_of_filters[1], (filter_size[1],filter_size[1]), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 3   **************************
  
  model.add(Conv2D(number_of_filters[2], (filter_size[2],filter_size[2]), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 4   **************************
  
  model.add(Conv2D(number_of_filters[3], (filter_size[3],filter_size[3]), padding = 'same'))
  if (batch_normalization):
    model.add(BatchNormalization())

  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))


  # ******************   Layer 5   **************************
  
  model.add(Conv2D(number_of_filters[4], (filter_size[4],filter_size[4]), padding = 'same'))
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


sweep_config={
    'method': 'random',
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters':{
        'augment':{
            'values':[True,False]
        },
        'number_of_filters':{
            'values':[[160,80,40,20,10],[16,32,64,128,256]]
        },
        'filter_size':{
            'values':[[7,6,5,4,3],[6,5,4,3,2]]
        },
        'size_of_dense_layer':{
            'values':[128,256,512]
        },
        'activation_function':{
            'values':["relu"]
        },
        'batch_normalization':{
            'values':[True,False]
        },
        'drop_out':{
           'values':[0.2,0.3,0.4]
        },
        'learning_rate':{
            'values':[0.0001,0.0002,0.0004]
        },
        'epochs':{
            'values':[5,10]
        },
        'layer_mult':{
            'values':[0]
        }
    }
}



!pip install wandb
import wandb
wandb.login


sweep_id = wandb.sweep(sweep_config,project="A20304", entity="uditchaurasia")


from wandb.keras import WandbCallback
def train():
    config_defaults={
      'augment':False,
      'number_of_filters':64,
      'filter_size':7,
      'size_of_dense_layer':128,
      'activation_function':"relu",
      'batch_normalization':True,
      'drop_out':0.2,
      'learning_rate':0.0008,
      'epochs':5,
      'layer_mult':1
       }
    
    wandb.init(config=config_defaults)
    config=wandb.config
    training_data, data_for_validation, data_for_test = train_val_test_data(config.augment)
    final_model = create_cnn_model(config.number_of_filters, config.filter_size, config.size_of_dense_layer, config.activation_function, config.batch_normalization, config.drop_out, config.layer_mult)
    final_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = config.learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    final_model.fit(training_data, validation_data = data_for_validation, epochs = config.epochs, callbacks=[WandbCallback()])
    final_model.summary()
    final_model.fit(training_data, validation_data = data_for_test)


wandb.agent(sweep_id,train)
