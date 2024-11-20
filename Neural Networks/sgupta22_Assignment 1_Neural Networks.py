#!/usr/bin/env python
# coding: utf-8

# ###Assignment 1_Neural Networks
# ###Shefali Gupta
# ###Advanced Machine Learning

# In[3]:


# pip install tensorflow


# In[4]:


##importing imdb dataset
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000)


# In[5]:


y_train[0]


# In[6]:


max([max(sequence) for sequence in x_train])


# In[7]:


##Preparing the data


# In[8]:


# step 1: Load the mappings of the dictionary from the word to integer index
word_index = imdb.get_word_index()

# step 2: reverse word index to integer mapping 
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

# step 3: Decode the review, mapping integer to words
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in x_train[0]])


# In[9]:


##Encoding the integer sequences via multi-hot encoding - Multi-hot encoding is applied to transform the lists into binary vectors, where each review is represented as a 10,000-dimensional vector consisting of 0s and 1s.
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

train_data = vectorize_sequences(x_train)
test_data = vectorize_sequences(x_test)


# In[10]:


train_data[0]


# In[11]:


train_labels = np.asarray(y_train).astype("float32")
test_labels = np.asarray(y_test).astype("float32")


# #Constructing a Model with a Single Hidden Layer: 32 Hidden Units and Tanh Activation

# ##Observations:
# - In the neural network designed below, we have a single layer comprising 32 hidden units using the tanh activation function. 
# - The output layer utilizes Sigmoid activation units.

# In[12]:


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(32, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])


# ##Compiling the Model using MSE instead of Binary_Crossentropy

# In[13]:


model.compile(optimizer="adam", 
              loss="mean_squared_error",
              metrics=["accuracy"])


# ##Validating

# During the model development process, a portion of the data is reserved for validation purposes. This validation set plays a crucial role in tuning the model's hyperparameters to find their optimal values. Typically, we iteratively adjust the hyperparameters on the training set and evaluate the model's performance on the validation set. This iterative process continues until we achieve the highest possible accuracy, which serves as our performance metric in this case.

# In[14]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# ##Training the Model
# 
# #The model is set to be trained for 20 epochs with a batch size of 512.
# 
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))

# In[16]:


history_dict = history.history
history_dict.keys()


# ##Plotting the train & Validation Loss

# In[17]:


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ##Plotting the training and Validation Accuracy

# In[18]:


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ##Observations - 
# - Training Accuracy: The training accuracy steadily increases and nearly reaches 99.73% after 20 epochs, indicating effective learning and a good fit to the training data.
# - Validation Accuracy: The validation accuracy initially rises but later starts to decline, eventually stabilizing at around 86.68%. This suggests that the model may begin to overfit the data as the number of epochs increases.

# In[19]:


results = model.evaluate(test_data, test_labels)


# #Adding Dropout layer & Regularizers

# In[26]:


from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras import regularizers


model = keras.Sequential()
model.add(Dense(32,activation='tanh', activity_regularizer=regularizers.L2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", #using optimizer ADAM 
              loss="mean_squared_error",
              metrics=["accuracy"])

          
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

          

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# In[21]:


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[22]:


results = model.evaluate(test_data, test_labels)


# ##Observation - 
# - In the current configuration with one dropout layer set to 0.5 and Regularizers, the the training accuracy improved further to 99.77%, but the validation accuracy remained lower at 83.13%.

# #Exploring a Scenario with Three Hidden Layers: Optimized with Adam, Utilizing Tanh Activation, and MSE Loss

# In[23]:


# Importing required libraries
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras import regularizers

#Defining the neural network model using 3 layered approach with a single dropout layer

model = keras.Sequential()
model.add(Dense(32,activation='tanh')) 
model.add(Dropout(0.5))
#kernel_regularizer=regularizers.L1(0.01), activity_regularizer=regularizers.L2(0.01))
model.add(Dense(32,activation='tanh',kernel_regularizer=regularizers.L1(0.01), activity_regularizer=regularizers.L2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer="adam", #using the optimizer ADAM
              loss="mean_squared_error",
              metrics=["accuracy"])

# Data Splitting        
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

          
# Training the neural network
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# In[24]:


# Plotting Training and Validation Accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluating the results
model.evaluate(test_data, test_labels)


# #Model with 64 Hidden Units with Regularizers

# In[25]:


#In an effort to increase accuracy, I used the results to decide to use a higher count i.e 64 units along with mse loss, and more layers

model = keras.Sequential()
model.add(Dense(64,activation='tanh')) 
model.add(Dropout(0.5))
#kernel_regularizer=regularizers.L1(0.01), activity_regularizer=regularizers.L2(0.01))
model.add(Dense(64,activation='tanh',kernel_regularizer=regularizers.L1(0.01), activity_regularizer=regularizers.L2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer="adam", #using the optimizer ADAM
              loss="mean_squared_error",
              metrics=["accuracy"])

# Data Splitting        
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

          
# Training the neural network
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Evaluating the results
model.evaluate(test_data, test_labels)


# ###Observation - Comparison of models with 32 & 64 hidden unit counts in an effort to maximize accuracy
# 
# ###Results after 20 epochs:
# 
# - Model with 32 units:
# Training Accuracy: 99.16%
# Validation Accuracy: 87.69%
# - Model with 64 units:
# Training Accuracy: 99.45%
# Validation Accuracy: 87.54%
# 
# Results - Both models performed well, but the second model with 64 units achieved slightly higher training accuracy without a significant boost in validation accuracy. 

# ## Overview of the 3 layered Neural Network for IMDB data:
# 
# To begin, we started by gathering the necessary libraries to construct our neural network. After some research, it became evident that TensorFlow is a reliable choice, known for its strong support and implementation within the realm of deep learning libraries, alongside alternatives like PyTorch.
# 
# ####List of Imports are:
# 
# - from tensorflow import keras
# - from tensorflow.keras import layers
# - from keras.layers import Dense
# - from keras.layers import Dropout
# 
# ####Each of these imports plays a crucial role in our implementation. 
# 
# - Keras serves as the high-level API for TensorFlow 2, offering a user-friendly and highly productive interface for tackling machine learning challenges, with a focus on contemporary deep learning. At its core, Keras revolves around the concepts of layers and models. 
# - The simplest model type is the Sequential model, which creates a linear stack of layers. The 'Dense' layer signifies the number of hidden units within the neural network, while 'Dropout' pertains to randomly removing connections, adding a layer of robustness.
# 
# ####Designing of the neural network layers. We initiate our model as follows:
# 
# model = keras.Sequential()
# 
# The Sequential model serves as the foundational structure in Keras, enabling us to stack layers in a sequential fashion. We can effortlessly add layers using the 'add' function. In the snippet below, '32' represents the number of hidden units, and we apply the 'tanh' activation function:
# 
# model.add(Dense(32, activation='tanh'))
# 
# The 3 layers of the neural network:
# 
# - Input Layer: This is where we provide the vector representation of IMDB data.
# - Hidden Layers: These layers house the dense units, and we can stack as many as required based on our needs.
# - Output Layer: Ideally, the output layer comprises a single dense unit.
# 
# For this assignment, I attempted to implement a three-layered approach as specified. Here is the code: 
# 
# model = keras.Sequential([
#     layers.Dense(32, activation="tanh"),
#     layers.Dense(32, activation="tanh"),
#     layers.Dense(32, activation="tanh"),
#     layers.Dense(1, activation="sigmoid")
# ])
# 
# 
# This model is constructed as a sequential neural network, which follows a linear sequence of layers through which data flows. The code specifies the architecture of this neural network as follows:
# 
# - The model comprises four distinct layers, each with its unique characteristics.
# - The initial layer, known as the first hidden layer, consists of 32 neurons or processing units. These neurons employ the hyperbolic tangent (tanh) activation function, enabling them to capture intricate patterns within the data.
# - Subsequently, two more hidden layers, identical to the first in terms of neuron count and activation function, are added to the architecture. This configuration creates a deep neural network with three hidden layers in total.
# - The final layer, referred to as the output layer, plays a crucial role in binary classification tasks. It contains a single neuron and employs the sigmoid activation function. The sigmoid function transforms the network's output values into a range between 0 and 1, rendering it suitable for binary classification problems where the objective is to predict probabilities.
# 
# ####Regarding model compilation, we employed the following configuration:
# 
# model.compile(optimizer="adam", 
#               loss="mean_squared_error",
#               metrics=["accuracy"])
# 
# 
# This code snippet utilizes the 'adam' optimizer with mean squared error (MSE) as the loss function.
# - Optimizer: "adam" is used, known for its efficiency in gradient-based optimization.
# - Loss Function: "mean_squared_error" is employed, ideal for regression tasks to minimize prediction errors.
# - Metrics: The model's performance is evaluated using "accuracy," which measures classification correctness.
# 
# This configuration readies the model for training
# 
# ####Lastly, we split the data into training and validation sets, as demonstrated below:
# 
# x_val = train_data[:10000]
# partial_x_train = train_data[10000:]
# y_val = train_labels[:10000]
# partial_y_train = train_labels[10000:]
# 
# ####Training the data is accomplished using the following code snippet:
# 
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))
# 
# This code indicates that the neural network is trained over 20 epochs, with a batch size of 512, and concurrently validated against the validation data. It is worth mentioning that I experimented with L1 and L2 regularizers, but they did not have a significant impact on the overall validation accuracy.

# #Summary of Neural Network with 3 layers.
# - Utilized the tanh activation function instead of relu.
# - Employed the Adam optimizer.
# - L1 & L2 regularizers are used.
# - Included a dropout layer with a 50% dropout rate during training.
# - Achieved a final accuracy of 99.16%.
# - Obtained a validation accuracy of 87.96%.

# #Model Comparison and Observations

# | Approach  | Training Accuracy | Validation Accuracy | Observations |
# |-----------|-------------------|---------------------|--------------|
# | Single layer, Activation - tanh, Optimizer - Adam, Loss - MSE (Mean Squared Error) | 99.73% | 86.68% | - A reasonable balance between training and validation. The single-layer model with tanh activation and Adam optimizer shows effective learning but potential overfitting. |
# | Single layer, Activation - tanh, Optimizer - Adam, Loss - MSE with dropout and regularizers | 99.77% | 83.13% | - The addition of dropout and regularization improves training accuracy but doesn't significantly boost validation accuracy. In summary, I observed a minor change in training accuracy, but when I attempted to improve the model by adding dropout layers and regularizers, it led to a decline in validation accuracy. This suggests that the original single-layer model may perform better without these additional modifications. |
# | Three layers, Activation - tanh, Optimizer - Adam, Loss - MSE with dropouts and regularizers | 99.16% | 87.69% | - The three-layer model is effective in maintaining both training and validation accuracy. |
# 
# 
# 

# #Conclusions for the Models with 32 Hidden Units
# - The single-layer model with tanh activation and Adam optimizer achieved a training accuracy of 99.73% and a validation accuracy of 86.68%. While it showed effective learning, there were indications of potential overfitting.
# - When adding dropout layers and regularization to the single-layer model with the same configuration, the training accuracy improved to 99.77%, but the validation accuracy dropped to 83.13%. This suggests that the additional modifications did not significantly enhance validation performance and might have caused overfitting.
# - In contrast, the three-layer model with tanh activation, Adam optimizer, and the same loss function (MSE) achieved a training accuracy of 99.16% and a higher validation accuracy of 87.69%. This approach demonstrated effectiveness in maintaining both training and validation accuracy.
# - Overall, the three-layer model appeared to be a promising configuration, achieving good generalization to validation data while maintaining high training accuracy.
