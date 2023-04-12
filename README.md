# CatDetectorMark1
Neural Net Project that detects Cat from not Cat
Project Title: Cat vs. Not Cat Image Classification
Table of Contents
Introduction
Data Preparation
Neural Network Model
Model Evaluation
Comparison with Traditional Machine Learning Techniques
Conclusion
Introduction
This project aims to build an image classification model to distinguish between images of cats and images not containing cats. We use a neural network model built with TensorFlow and compare its performance to traditional machine learning techniques.

Data Preparation
The dataset is organized in the data directory, with images of cats and images without cats stored separately. We use the ImageDataGenerator class from TensorFlow to load the dataset, perform data augmentation, and normalize the images. The dataset is split into training (80%) and validation (20%) sets. The rationale for using separate validation and test sets is to evaluate the model's performance on unseen data, ensuring that the model generalizes well to new data.

Neural Network Model
We use a convolutional neural network (CNN) with the following architecture:

Rescaling layer: Scales the input images to a range of [0, 1].
Conv2D layer with 16 filters of size 3x3: This layer helps extract features from the input images.
MaxPooling2D layer
Conv2D layer with 32 filters of size 3x3
MaxPooling2D layer
Conv2D layer with 64 filters of size 3x3
MaxPooling2D layer
Flatten layer
Dense layer with 128 neurons
Dense layer with 1 neuron and sigmoid activation
The model is compiled using the Adam optimizer and binary cross-entropy loss. We train the model for 20 epochs.

Model Evaluation
To evaluate the model's performance, we calculate the confusion matrix for the predictions on the validation dataset. We repeat the evaluation process multiple times and average the results to obtain a more reliable estimate for the performance measures scores.

Comparison with Traditional Machine Learning Techniques
In addition to the neural network model, we implement and compare the performance of two traditional machine learning techniques. The comparison table and visualization will be provided in this section.

Conclusion
This project demonstrates the process of building and evaluating an image classification model using a convolutional neural network and comparing its performance to traditional machine learning techniques. The results show that the neural network model is effective in classifying cat and not cat images, with room for further improvement and optimization.
