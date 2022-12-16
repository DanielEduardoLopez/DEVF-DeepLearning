# DEVF-Deep Learning
Repository of the challenges &amp; projects from the *Deep Learning* module of the **Master in Data Science & Artificial Intelligence** at [DEV.F](https://www.devf.la/master/data/mx).

# Final Project: **Rabbit/Duck Classifier with Convolutional Neural Networks**

### **1. Goal**

The purpose of the present project is to design and train a classification model from scratch by using Convolutional Neural Networks (CNN) to classify whether a given image corresponds to a rabbit or a duck.

### **2. Classification Model Architecture**

A model with the following architecture is proposed, based on the guidelines from Collet (2016):
1. A first Convolutional layer with 32 filters, a kernel size of (3, 3) and a ReLU activation function.
2. A first Max Pooling layer matrix size of (2, 2).
3. A second Convolutional layer with 32 filters, a kernel size of (3, 3) and a ReLU activation function.
4. A second Max Pooling layer matrix size of (2, 2).
5. A third Convolutional layer with 64 filters, a kernel size of (3, 3) and a ReLU activation function.
5. A third Max Pooling layer matrix size of (2, 2).
6. A Flattening layer for converting the pooled matrices into linear vectors.
7. A full connection layer with 128 nodes and a ReLU activation function, and a dropout of 20%.
8. An output layer of 1 node and a sigmoid activation function.

### **3. Data Augmentation**
An original training dataset of 200 pictures and 200 pictures of ducks was used. The pictures were retrieved from the internet. On the other hand, testing sets of 40 pictures were used by each category.

The dataset can be found
<a href="https://drive.google.com/drive/folders/10PTdJztG3Wb7-Ch21jNQ3ZayyLvWMWLp?usp=sharing"> here</a>.

### 4. **Model Fitting**

The model was fitted with the training dataset and then validated with the testing set.

### **5. Model Testing**

This is the fun part. Besides the issues with the model, a couple of different images resembling rabbits and ducks were pass down to the model to see if it was capable to accurately classify the object.

### **6. Conclusions**

A rabbit/duck classifier was built using convolutional neural networks in Keras and Tensorflow. And, despite the bad results obtained with the testing accuracy and loss metrics, the model was able to correctly classify some pictures of rabbits and ducks. 

The result is positive taking into account the extremely small dataset used for training the model with only 200 pictures by category.

Notwithstanding the above, it is noteworthy that the difference between training and testing accuracy was of about 0.20, which strongly suggests that overfitting might have ocurred within the model. 

Thus, in order to provide generalizability and accurate predictions, it is advisable to train the model with a larger dataset, such as one with 2000-5000 images; as well as using a larger testing dataset such as one with 1000 images to get a more accurate evaluation of the performance of the model.
