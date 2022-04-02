#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libaries
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#Data Preprocessing


# In[2]:


#Training Image Preprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory('D:/Software/Project/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[3]:


#Test Image Preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('D:/Software/Project/test_set',
           target_size=(64, 64),
           batch_size=32,
           class_mode='categorical')
#Class Mode have two Category Then Used 'Binary' If More Than Two Then used 'Categorical'


# In[4]:


#Building Model
cnn = tf.keras.models.Sequential()


# In[5]:


#Building Convolution
cnn.add(tf.keras.layers.Conv2D(filters=64 , kernel_size=3 , activation='relu' , input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[6]:


#<------:Activation Function:------>
          #relu function
          #sigmoid function
          #softmax function
          #softplus function
          #softsign function
          #tanh function
          #selu function 
          #elu function
          #exponential function


# In[7]:


cnn.add(tf.keras.layers.Conv2D(filters=64 , kernel_size=3 , activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 , strides=2))
#pool_size = 2 means Matrix Size is 2
#strides->jump


# In[8]:


#randomly sets input units to 0 with a frequency of rate at each step during training time,which helps prevent overfitting
cnn.add(tf.keras.layers.Dropout(0.5))


# In[9]:


#flattens the multi-dimensional input tensors into a single dimension
cnn.add(tf.keras.layers.Flatten())


# In[10]:


#Adding Hidden Layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[11]:


#Output Layer
cnn.add(tf.keras.layers.Dense(units=5 , activation='softmax'))
#Units = 5, Because Their Five Type Of Flower
#Output Like- > [[0. 0. 0. 1. 0.]]


# In[12]:


#The purpose of loss functions is to compute the quantity that a model should seek to minimize during training

cnn.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# In[13]:


#<------:optimizers------>
          #SGD
          #RMSprop
          #Adam
          #Adadelta
          #Adagrad
          #Adamax
          #Nadam
          #Ftrl


# In[14]:


#Training The Model
cnn.fit(x = training_set , validation_data = test_set , epochs = 1)


# In[15]:


training_set.class_indices


# In[16]:


#Preprocessing the New Images For Testing


# In[17]:


from keras.preprocessing import image
test_image = image.load_img('D:/Software/Project/Prediction/rose.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)


# In[18]:


if result[0][0]==1:
    ans = "Daisy"
elif result[0][1]==1:
    ans = "Dandelion"
elif result[0][2]==1:
    ans = "Rose"
elif result[0][3]==1:
    ans = "Sunflower"
elif result[0][4]==1:
    ans = "Tultip"


# In[19]:


import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice")

if ans == "Daisy":
    speaker.Speak(ans)
    speaker.Speak("About Daisy")
    speaker.Speak("Bellis perennis, the daisy, is a European species of the family Asteraceae, often considered the archetypal species of that name. To distinguish this species from other plants known as daisies, it is sometimes qualified as common daisy, lawn daisy or English daisy.")
elif ans == "Dandelion":
    speaker.Speak(ans)
    speaker.Speak("About Dandelion")
    speaker.Speak("Taraxacum is a large genus of flowering plants in the family Asteraceae, which consists of species commonly known as dandelions. The scientific and hobby study of the genus is known as taraxacology.")
elif ans == "Rose":
    speaker.Speak(ans)
    speaker.Speak("About Rose")
    speaker.Speak("A rose is a woody perennial flowering plant of the genus Rosa, in the family Rosaceae, or the flower it bears. There are over three hundred species and tens of thousands of cultivars. They form a group of plants that can be erect shrubs, climbing, or trailing, with stems that are often armed with sharp prickles.")
elif ans == "Sunflower":
    speaker.Speak(ans)
    speaker.Speak("About Sunflower")
    speaker.Speak("Helianthus is a genus comprising about 70 species of annual and perennial flowering plants in the daisy family Asteraceae commonly known as sunflowers. Except for three South American species, the species of Helianthus are native to North America and Central America.")

elif ans == "Tultip":
    speaker.Speak(ans)
    speaker.Speak("About Tulip")
    speaker.Speak("Tulips are a genus of spring-blooming perennial herbaceous bulbiferous geophytes. The flowers are usually large, showy and brightly colored, generally red, pink, yellow, or white. They often have a different colored blotch at the base of the tepals, internally.")


# In[21]:


#Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, 
#and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object)
#is converted back into an object hierarchy.
import pickle


# In[22]:


with open('D:/Software/Project/cnn_pickle','wb') as f:
    pickle.dump(cnn,f)


# In[ ]:


#Here We Save The Model using Keras And Predict


# In[26]:


filename = "D:/Software/Project/cnn_fl.h5"
cnn.save(filename)


# In[38]:


from tensorflow.keras.models import load_model
from keras.preprocessing import image

test_image = image.load_img('D:/Software/Project/Prediction/Sunflowers.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

load_cnn = load_model(filename)
cnn_pred = load_cnn.predict(test_image)


# In[44]:


import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Speak(cnn_pred)
print(cnn_pred)


# In[ ]:





# In[ ]:





# In[ ]:




