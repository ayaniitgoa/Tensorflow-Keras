#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:


tf.__version__


# In[4]:


mnist = tf.keras.datasets.mnist


# In[19]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=3)


# In[20]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss,val_acc)


# In[16]:


plt.imshow(x_train[2], cmap = plt.cm.binary)
plt.show()
print(x_train[2])


# In[21]:


model.save('num_prediction.model')


# In[22]:


new_model = tf.keras.models.load_model('num_prediction.model')


# In[23]:


predictions = new_model.predict([x_test]) 


# In[24]:


import numpy as np


# In[30]:


print(np.argmax(predictions[7]))


# In[29]:


plt.imshow(x_test[7])
plt.show()


# In[ ]:




