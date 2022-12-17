# DEVF-Deep Learning
Repository of the challenges &amp; projects from the *Deep Learning* module of the **Master in Data Science & Artificial Intelligence** at [DEV.F](https://www.devf.la/master/data/mx).

_____

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

The code of the model in Python is as follows:
```python
# Libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialization of the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))

# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Convolution
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))

# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Convolution
classifier.add(Conv2D(filters = 64,kernel_size = (3, 3), activation = "relu"))

# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = "sigmoid"))
```

Please refer to the <a href="https://github.com/DanielEduardoLopez/DEVF-DeepLearning/blob/main/FinalProject_RabbitClassifier_CNN.ipynb"> Jupyter notebook</a> for all the details. 

### **3. Data Augmentation**
An original training dataset of 200 pictures and 200 pictures of ducks was used. The pictures were retrieved from the internet. On the other hand, testing sets of 40 pictures were used by each category.

The dataset can be found
<a href="https://drive.google.com/drive/folders/10PTdJztG3Wb7-Ch21jNQ3ZayyLvWMWLp?usp=sharing"> here</a>.

### 4. **Model Fitting**

The model was fitted with the training dataset and then validated with the testing set.

In this sense, the plot of the **training accuracy by epoch** is shown below:

<p align="center">
	<img src="Classifier_Images/Fig1_TrainingAccuracy.png?raw=true" width=65% height=65%>
</p>

The training accuracy curve raised from an initial value of 0.55 to a final value of about 0.80. So, the profile of the curve falls into the expectable and desirable for this case.

Moreover, the plot of the **training loss by epoch** is shown below:

<p align="center">
	<img src="Classifier_Images/Fig2_TrainingLoss.png?raw=true" width=65% height=65%>
</p>

Likewise, the training loss decreased from an initial value of about 0.68 to a final value of 0.40.

On the other hand, the plot of the **testing accuracy by epoch** is shown below:

<p align="center">
	<img src="Classifier_Images/Fig3_TestingAccuracy.png?raw=true" width=65% height=65%>
</p>

Unlike the training accuracy plot, the testing accuracy beahvior by epoch is erratic and follow no distintive trend. This might suggest two issues: The testing set is too small and the fitted model is incapable of provide generalizable results, which render the model as a **bad** model.

Furthermore, the plot of the **testing loss by epoch** is shown below:

<p align="center">
	<img src="Classifier_Images/Fig4_TestingLoss.png?raw=true" width=65% height=65%>
</p>

As expectable from the testing accuracy plot, the testing loss by epoch is erratic and even ends with a higher loss in comparison with the begining loss, which of course means that the fitted model is useless.

### **5. Model Testing**

This is the fun part. Besides the issues with the model, a couple of different images resembling rabbits and ducks were pass down to the model to see if it was capable to accurately classify the objects.

<p align="center">
	<img src="Classifier_Images/Fig5_TestImage1.png?raw=true" width=35% height=35%>
</p>

```bash
1/1 [==============================] - 0s 26ms/step
array([[1.]], dtype=float32)
```

So, according to the model, the rabbit figure is a **rabbit**.

<p align="center">
	<img src="Classifier_Images/Fig6_TestImage2.png?raw=true" width=35% height=35%>
</p>

```bash
1/1 [==============================] - 0s 23ms/step
array([[1.]], dtype=float32)
```

Then, according to the model, the stuffed toy is a **rabbit**.

<p align="center">
	<img src="Classifier_Images/Fig7_TestImage3.png?raw=true" width=35% height=35%>
</p>

```bash
1/1 [==============================] - 0s 23ms/step
array([[0.]], dtype=float32)
```

As expectable, the duck is a **duck**.

<p align="center">
	<img src="Classifier_Images/Fig8_TestImage4.png?raw=true" width=35% height=35%>
</p>

```bash
1/1 [==============================] - 0s 26ms/step
array([[0.]], dtype=float32)
```

Therefore, according to the model, the duck drawing is a **duck**.

<p align="center">
	<img src="Classifier_Images/Fig9_TestImage5.png?raw=true" width=35% height=35%>
</p>

```bash
1/1 [==============================] - 0s 23ms/step
array([[1.]], dtype=float32)
```

Thus, according to the model, the famous rabbit-duck illusion (which actually inspired this whole project) ir more a **rabbit** than a duck.

### **6. Conclusions**

A rabbit/duck classifier was built using convolutional neural networks in Keras and Tensorflow. And, despite the bad results obtained with the testing accuracy and loss metrics, the model was able to correctly classify some pictures of rabbits and ducks. 

The result is positive taking into account the extremely small dataset used for training the model with only 200 pictures by category.

Notwithstanding the above, it is noteworthy that the difference between training and testing accuracy was of about 0.20, which strongly suggests that overfitting might have ocurred within the model. 

Thus, in order to provide generalizability and accurate predictions, it is advisable to train the model with a larger dataset, such as one with 2000-5000 images; as well as using a larger testing dataset such as one with 1000 images to get a more accurate evaluation of the performance of the model.

### **7. References**
* **Collet, F. (2016).** *Building powerful image classification models using very little data*. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* **Sarkar, T. (2019).** *Keras utility methods for streamlining training of convolutional neural net*. https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb
