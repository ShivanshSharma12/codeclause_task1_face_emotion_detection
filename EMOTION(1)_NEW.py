#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet, preprocess_input 
from keras.losses import categorical_crossentropy


# In[2]:


# Working with pre trained model 

base_model = MobileNet( input_shape=(224,224,3), include_top= False )

for layer in base_model.layers:
  layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(units=7 , activation='softmax' )(x)

# creating our model.
model = Model(base_model.input, x)
     


# In[3]:


model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )
     


# In[4]:


train_datagen = ImageDataGenerator(
     zoom_range = 0.2, 
     shear_range = 0.2, 
     horizontal_flip=True, 
     rescale = 1./255
)

train_data = train_datagen.flow_from_directory(directory= r"C:\Users\SHIVANSH SHARMA\Desktop\codeclause\EMOTION\train", 
                                               target_size=(224,224), 
                                               batch_size=32,
                                  )


train_data.class_indices


# In[5]:


# to visualize the images in the traing data denerator 

t_img , label = train_data.next()

#-----------------------------------------------------------------------------
# function when called will prot the images 
def plotImages(img_arr, label):
  """
  input  :- images array 
  output :- plots the images 
  """
  count = 0
  for im, l in zip(img_arr,label) :
    plt.imshow(im)
    plt.title(im.shape)
    plt.axis = False
    plt.show()
    
    count += 1
    if count == 10:
      break

#-----------------------------------------------------------------------------
# function call to plot the images 
plotImages(t_img, label)


# In[6]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')

# model check point
mc = ModelCheckpoint(filepath="best_model.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')

# puting call back in a list 
call_back = [es, mc]
     
    


# In[7]:


hist = model.fit_generator(train_data, 
                           steps_per_epoch= 10, 
                           epochs= 30, 
                           
                          
                           callbacks=[es,mc])


# In[8]:


# Loading the best fit model 
from keras.models import load_model
model = load_model(r"C:\Users\SHIVANSH SHARMA\Downloads\Emotion-detection-main\Emotion-detection-main/best_model.h5")


# In[9]:


h =  hist.history
h.keys()


# In[10]:


from keras.preprocessing.image import load_img, img_to_array


# In[11]:


# path for the image to see if it predics correct class

path = r"C:\Users\SHIVANSH SHARMA\Desktop\codeclause\EMOTION\train\Angry\download (9).jpg"
img = load_img(path, target_size=(224,224) )

i = img_to_array(img)/255
input_arr = np.array([i])
input_arr.shape


# In[12]:


op = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]  # Replace with your actual class labels

# Predict the class
pred = np.argmax(model.predict(input_arr))

# Print the predicted class label
pred_label = op[pred]
print(f"The image is of {pred_label}")



# In[13]:


# to display the image  
plt.imshow(input_arr[0])
plt.title("input image")
plt.show()


# In[16]:


path = r"C:\Users\SHIVANSH SHARMA\Desktop\codeclause\EMOTION\train\Neutral\images (9).jpg"
img = load_img(path, target_size=(224,224) )

i = img_to_array(img)/255
input_arr = np.array([i])
input_arr.shape

op = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]  

# Predict the class
pred = np.argmax(model.predict(input_arr))

# Print the predicted class label
pred_label = op[pred]
print(f"The image is of {pred_label}")
# to display the image  
plt.imshow(input_arr[0])
plt.title("input image")
plt.show()


# In[26]:


path = r"C:\Users\SHIVANSH SHARMA\Desktop\codeclause\EMOTION\train\Angry\images (17).jpg"
img = load_img(path, target_size=(224,224) )

i = img_to_array(img)/255
input_arr = np.array([i])
input_arr.shape

op = ["Angry", "Happy", "Sad", "Surprise", "Neutral", "Disguist"]  

# Predict the class
pred = np.argmax(model.predict(input_arr))

# Print the predicted class label
pred_label = op[pred]
print(f"The image is of {pred_label}")
# to display the image  
plt.imshow(input_arr[0])
plt.title("input image")
plt.show()


# In[ ]:





# In[ ]:




