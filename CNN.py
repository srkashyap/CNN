#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten


# In[4]:


from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model


# In[5]:


import cv2
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)


# In[6]:


import os
import matplotlib.pyplot as plt
import numpy as np
from time import time


# In[7]:


def loadimages(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith("jpeg")]


# In[8]:


def flist(files):
    l1=[]
    for i in files:
        a = cv2.imread(i,cv2.IMREAD_UNCHANGED)
        a = cv2.resize(a,(224,224))
        l1.append(a)
    return l1


# In[9]:

'        Train Data       '

start = time()
path = "C:\\Users\\CSSKA\\Desktop\\PTC\\CNN\\3"
X = []
Y = []
for i in os.listdir(path):
    if i == 'good' :
        path_new = os.path.join(path,i)
        files_list = sorted(loadimages(path_new))
        [X.append(j/255) for j in flist(files_list)]
        [Y.append(1) for j in flist(files_list)]
        
    elif i == 'bad':
        path_new = os.path.join(path,i)
        files_list = sorted(loadimages(path_new))
        [X.append(j/255) for j in flist(files_list)]
        [Y.append(0) for j in flist(files_list)]
end = time()
print("approx",end-start,"seconds")


# In[48]:


np.shape(X)


# In[12]:


X = np.asarray(X)
Y = np.asarray(keras.utils.to_categorical(Y, 2))
print(X.shape, Y.shape)


# In[13]:


np.shape(X_test), np.shape(X)


# In[14]:


'        Test Data          '
start = time()
path = "C:\\Users\\CSSKA\\Desktop\\PTC\\CNN\\4"
X_test = []
Y_test = []
for i in os.listdir(path):
    if i == 'good' :
        path_new = os.path.join(path,i)
        files_list = sorted(loadimages(path_new))
        [X_test.append(j/255) for j in flist(files_list)]
        [Y_test.append(1) for j in flist(files_list)]
        
    elif i == 'bad':
        path_new = os.path.join(path,i)
        files_list = sorted(loadimages(path_new))
        [X_test.append(j/255) for j in flist(files_list)]
        [Y_test.append(0) for j in flist(files_list)]
end = time()
print("approx",end-start,"seconds")
X_test = np.array(X_test)
Y_test = np.array(keras.utils.to_categorical(Y_test, 2))



# In[28]:


np.shape(X)[1:]


# In[29]:


# model = Sequential()
# # densely-connected layer with 64 units to the model:
# model.add(Conv2D(64,(3,3), activation = 'relu', 
#                  input_shape = np.shape(X)[1:]))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2, activation='softmax'))


# In[30]:


# model = Sequential()
# # densely-connected layer with 64 units to the model:
# model.add(Conv2D(64,(3,3), activation = 'relu', 
#                  input_shape = np.shape(X)[1:]))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# # model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# # model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# # model.add(Dropout(0.2))
# # model.add(Conv2D(32, (3, 3), activation='relu'))
# # model.add(Conv2D(32, (3, 3), activation='relu'))
# # model.add(Conv2D(32, (3, 3), activation='relu'))
# # model.add(Conv2D(32, (3, 3), activation='relu'))
# # model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))


# In[31]:


# model.summary()


# In[32]:


# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['acc'])


# In[33]:


# res = model.fit(X, Y,
#           batch_size=32,
#           epochs=5,
#           verbose=1,
#           validation_data=(X_test, Y_test))


# In[34]:


# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# In[15]:


img_height, img_width = 224, 224
channels_number = 3


# In[39]:


model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling=None, classes=2)
x = model.output
#x = Dropout(0.22)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.30)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
# x = Dropout(0.1)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)


# In[40]:


np.shape(X),np.shape(Y),np.shape(X_test),np.shape(Y_test)


# In[41]:


model.summary()


# In[42]:


from keras.optimizers import Adam
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer= opt, loss='categorical_crossentropy',metrics=['acc']) 
#You can also use recall and precision as your performance metrics for better evaluation. It will be easy to find out overfitting
#Model ready h. Data oad karke train kar lena. Keep learnign rate = 0.0001


# In[43]:


tfmodel = model.fit(np.array(X), Y, epochs=5, batch_size=32, validation_data=(X_test, Y_test))


# In[33]:


plt.plot(tfmodel.history['acc'])
plt.plot(tfmodel.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(tfmodel.history['loss'])
plt.plot(tfmodel.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[54]:


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[66]:




