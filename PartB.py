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
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2,ResNet50,Xception,VGG16,VGG19
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Activation,Dropout,Flatten,MaxPool2D
from keras.models import Model,Sequential
import random
from keras.preprocessing.image import ImageDataGenerator
import math


def create_cnn_model(model, layers_to_freeze):
  if (model=="InceptionV3"):
    pretrained= InceptionV3(include_top = True)
  elif (model=="InceptionResNetV2"):
    pretrained= InceptionResNetV2(include_top = True)
  elif (model=="ResNet50"):
    pretrained= ResNet50(include_top = True)
  elif (model=="Xception"):
    pretrained= Xception(include_top = True)
  elif (model=="VGG16"):
    pretrained= VGG16(include_top = True)
  elif (model=="VGG19"):
    pretrained= VGG19(include_top = True)

  last_layer = Dense(10, activation='softmax', name='predictions')(pretrained.layers[-2].output)

  number_of_layers = len(pretrained.layers)
  print(number_of_layers)
  freeze_layers = (layers_to_freeze*number_of_layers)

  for i in range(0, math.ceil(freeze_layers)):
    layer = pretrained.layers[i]
    layer.trainable = False
  
  final_model = Model(inputs=pretrained.input, outputs = last_layer)

  return final_model


from keras.preprocessing.image import ImageDataGenerator

def train_val_test_data():
  data_val = ImageDataGenerator(rescale = 1./255)
  data_test = ImageDataGenerator(rescale = 1./255)
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
        'epochs':{
            'values':[3,4,5,6]
        },
        'layers_to_freeze':{
            'values':[0.65,0.7,0.75,0.8,0.85,0.9]
        },
        'model':{
           'values':["InceptionV3", "InceptionResNetV2","ResNet50","Xception"]
        }
  
    }
}





!pip install wandb
import wandb
wandb.login



sweep_id = wandb.sweep(sweep_config,project="PartB", entity="uditch")






from wandb.keras import WandbCallback
def train():
    config_defaults={
      'epochs':4,
      'model':"Xception",
      'layers_to_freeze':0.75
       }
    wandb.init(config=config_defaults)
    config=wandb.config
    wandb.run.name = "model_{}_layersTofreeze_{}_epochs_{}".format(config.model,\
                                                                       config.layers_to_freeze,\
                                                                       config.epochs)
    training_data, data_for_validation, data_for_test = train_val_test_data()
    trained_model = create_cnn_model(config.model, config.layers_to_freeze)
    trained_model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    trained_model.fit(training_data, validation_data = data_for_validation, epochs = config.epochs, callbacks=[WandbCallback()])
    
    
    
    
    
    
    
wandb.agent(sweep_id,train)
    
    
