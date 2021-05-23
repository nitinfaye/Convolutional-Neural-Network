#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import zipfile


# In[2]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


tf.__version__


# In[4]:


#zip_ref = zipfile.ZipFile("/content/dataset.zip",'r')
#zip_ref.extractall("/tmp")
#zip_ref.close()


# ## Data Preprocessing

# ### Preprocessing the Training set

# In[5]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('datasets/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary',)


# ### Preprocessing the Test set

# In[6]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('datasets/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# ## Building the CNN

# ### Initialising the CNN

# In[7]:


cnn = tf.keras.models.Sequential()


# ### Convolution

# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# ###Pooling

# In[9]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Adding a second convolutional layer

# In[10]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ###Flattening

# In[11]:


cnn.add(tf.keras.layers.Flatten())


# ###Full Connection

# In[12]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ###Output Layer

# In[13]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ##Training the CNN

# ### Compiling the CNN

# In[14]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[15]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# ##Making a single prediction

# In[16]:


from keras.preprocessing import image
test_image = image.load_img('datasets/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[17]:


print(prediction)


# In[31]:


from PIL import Image
im = Image.open(r'datasets/single_prediction/cat_or_dog_1.jpg')
im.show()


# In[32]:


test_image = image.load_img('datasets/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[33]:


print(prediction)


# In[34]:


im = Image.open(r'datasets/single_prediction/cat_or_dog_2.jpg')
im.show()


# In[ ]:




