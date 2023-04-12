# %%
%pip install seaborn
%pip install numpy
%pip install pandas

# %%
import tensorflow as tf

# Import necessary libraries
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from io import BytesIO

# %%
import torch
print(torch.cuda.is_available())
tf.config.list_physical_devices('GPU')
!pip list


# %%
import tensorflow as tf

# Get a list of physical GPUs available on the computer
gpus = tf.config.experimental.list_physical_devices('GPU')

# Print the list of available GPUs to the console
print("List of GPUs Available: ", gpus)

# Set memory growth for each GPU to avoid running out of memory (OOM) errors
# This loop sets memory growth for each GPU to be dynamic, meaning that the memory
# usage will increase as needed, but will be released when it's no longer needed
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# List the available GPUs again
tf.config.list_physical_devices('GPU')


# %%
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# Get a list of available physical GPUs
physical_devices = tf.config.list_physical_devices('GPU')

# Set the first available GPU as the default device
if len(physical_devices) > 0:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_devices = tf.config.list_logical_devices('GPU')
    print(len(physical_devices), "Physical GPUs,", len(logical_devices), "Logical GPUs")
else:
    print("No GPUs available")
print(tf.config.list_physical_devices('GPU'))

# %% [markdown]
# # Model Development

# %% [markdown]
# `Sequential` is a class that represents a linear stack of layers in a machine learning model.
#     It is commonly used for building CNNs, where the layers are arranged in a sequential order to create a hierarchical feature representation of the input data.
# 
# `Conv2D` is a class that represents a 2D convolutional layer in a CNN.
#     It applies a set of learned filters to the input image, allowing the model to detect specific patterns and features in the image.
# 
# `MaxPooling2D` is a class that represents a 2D max pooling layer in a CNN.
#     It downsamples the output of a convolutional layer by taking the maximum value of a set of adjacent pixels, reducing the spatial dimensions of the feature maps.
# 
# `Dense` is a class that represents a fully connected layer in a neural network.
#     It takes the output of the preceding layer and applies a set of learned weights to generate a set of output values.
# 
# `Flatten` is a class that flattens the output of a preceding layer into a 1D array, which can be passed to a fully connected layer.
# 
# `Dropout` is a regularization technique that randomly drops out some of the output units of a layer during training, reducing the risk of overfitting.
# 
# These classes will be used to define the architecture of a CNN for the task of image classification.
# 

# %%

data_dir = 'data'
# Create the neural network model
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Create an ImageDataGenerator for data augmentation and normalization
data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of the data for validation
)

# Load the dataset and split it into training and validation sets
image_size = (180, 180)
batch_size = 32
train_dataset = data_generator.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    seed=123
)
validation_dataset = data_generator.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    seed=123
)

# Train the neural network with the training and validation datasets
epochs = 20
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset
)



# %%

# Save the trained model for future use
model.save('cat_not_cat_classifierv6.h5')



# %%
# Plot training and validation accuracy and loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the training and validation accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='best')

# Plot the training and validation loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(loc='best')

# %%
# Create a file upload widget
uploader = widgets.FileUpload(accept='image/*', multiple=False)
display(uploader)

# Function to classify the uploaded image
def classify_uploaded_image(change):
    if uploader.data:
        # Load the image from the uploaded data
        img = Image.open(BytesIO(uploader.data[0]))
        
        # Save the image temporarily
        img.save('uploaded_image.jpg', 'JPEG')

        # Load the image and get a prediction using the trained model
        img_array = load_image('uploaded_image.jpg')
        prediction = model.predict(img_array)

        # Display the classification result
        if prediction < 0.5:
            print('The image is classified as NOT CAT')
        else:
            print('The image is classified as CAT')

# Attach the function to the file upload widget
uploader.observe(classify_uploaded_image, names='data')

# %% [markdown]
# Table 0: Model V1 Results attempt 1 w/o CUDA
# |\ Epoch | Loss   | Accuracy | Val Loss | Val Accuracy | Time   | Impact                                                                                | Neat Fact                                                                                                                       |
# |-------|--------|----------|----------|--------------|--------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
# | 1     | 0.6174 | 0.6968   | 0.6111   | 0.6972       | 421s   | The model has relatively low accuracy and the validation accuracy is similar.          | The first epoch typically sets the initial accuracy baseline for the model.                                                   |
# | 2     | 0.6065 | 0.6942   | 0.5805   | 0.6972       | 465s   | The loss has decreased slightly, but the accuracy and validation accuracy are similar. | The model may not be learning much new information in this epoch.                                                               |
# | 3     | 0.5849 | 0.6969   | 0.5756   | 0.7028       | 618s   | The loss has decreased further, and the validation accuracy has increased slightly.    | The model is starting to learn more complex patterns in the data.                                                              |
# | 4     | 0.5796 | 0.6995   | 0.5793   | 0.7016       | 547s   | The loss has decreased further, but the accuracy and validation accuracy are similar.  | The model may have reached a plateau in its learning.                                                                            |
# | 5     | 0.5748 | 0.7061   | 0.5671   | 0.7090       | 537s   | The loss has decreased slightly, and the validation accuracy has increased slightly.  | The model is continuing to learn more complex patterns in the data.                                                              |
# | 6     | 0.5720 | 0.7045   | 0.5631   | 0.7016       | 503s   | The loss has decreased slightly, but the accuracy and validation accuracy are similar. | The model may not be learning much new information in this epoch.                                                               |
# | 7     | 0.5653 | 0.7080   | 0.5524   | 0.7054       | 482s   | The loss has decreased further, and the validation accuracy has increased slightly.    | The model is starting to learn more complex patterns in the data.                                                              |
# | 8     | 0.5582 | 0.7103   | 0.5391   | 0.7200       | 470s   | The loss has decreased further, and the validation accuracy has increased significantly. | The model is making significant improvements in accuracy.                                                                      |
# | 9     | 0.5552 | 0.7130   | 0.5385   | 0.7153       | 469s   | The loss has decreased slightly, and the validation accuracy has increased slightly.  | The model is continuing to learn more complex patterns in the data.                                                              |
# | 10    | 0.5514 | 0.7190   | 0.5219   | 0.7306       | 464s   | The loss has decreased significantly, and the validation accuracy has increased significantly. | The model is making significant improvements in accuracy.                                                                  |
# | 11    | 0.5396 | 0.7270   | 0.5189   | 0.7357       | 471s   | The
# 

# %% [markdown]
# | Epoch | Loss   | Accuracy | Val Loss | Val Accuracy | Time  | Impact                                                                               | Neat Fact                                                                                           |
# |-------|--------|----------|----------|--------------|-------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
# | 11    | 0.5396 | 0.7270   | 0.5189   | 0.7357       | 471s  | The model has shown significant improvement in accuracy compared to previous epochs. | The model may be learning more complex patterns in the data, leading to improved accuracy.         |
# | 12    | 0.5294 | 0.7324   | 0.5054   | 0.7439       | 452s  | The loss has decreased significantly, and the validation accuracy has increased significantly. | The model is making significant improvements in accuracy.                                         |
# | 13    | 0.5251 | 0.7346   | 0.4936   | 0.7556       | 461s  | The loss has decreased significantly, and the validation accuracy has increased significantly. | The model is making significant improvements in accuracy.                                         |
# | 14    | 0.5209 | 0.7382   | 0.4962   | 0.7497       | 428s  | The loss has decreased, but the validation accuracy has decreased slightly.               | The model may be overfitting to the training data.                                               |
# | 15    | 0.5172 | 0.7411   | 0.4852   | 0.7635       | 442s  | The loss has decreased significantly, and the validation accuracy has increased significantly. | The model is making significant improvements in accuracy.                                         |
# | 16    | 0.5163 | 0.7403   | 0.4905   | 0.7584       | 436s  | The loss has decreased, but the validation accuracy has decreased slightly.               | The model may be overfitting to the training data.                                               |
# | 17    | 0.5153 | 0.7420   | 0.4895   | 0.7584       | 418s  | The loss has decreased slightly, but the accuracy and validation accuracy are similar.    | The model may not be learning much new information in this epoch.                                |
# | 18    | 0.5141 | 0.7396   | 0.4886   | 0.7604       | 417s  | The loss has decreased slightly, and the validation accuracy has increased slightly.      | The model is continuing to learn more complex patterns in the data.                              |
# | 19    | 0.5134 | 0.7428   | 0.4876   | 0.7575       | 443s  | The loss has decreased slightly, and the validation accuracy has decreased slightly.      | The model may not be learning much new information in this epoch.                                |
# | 20    | 0.5095 | 0.7433   | 0.4949   | 0.7594       | 424s  | The loss has decreased slightly, and the validation accuracy has decreased slightly.      | The model may not be learning much new information in this epoch.                                |
# 

# %% [markdown]
# # Understanding the Results
# *Table 1: Epoch Metrics*
# | Concept | Definition | Impact | Training Time | Neat Fact |
# | --- | --- | --- | --- | --- |
# | Epoch | One complete pass through the entire training dataset | Increase may improve accuracy but may also overfit | Longer if more epochs are used | Training time can be reduced with early stopping |
# | Loss | A metric that measures the difference between the predicted and actual values | Lower loss indicates better accuracy | Longer if loss is high | Different loss functions are used for different problems |
# | Accuracy | The percentage of correct predictions out of total predictions | Higher accuracy indicates better performance | Longer if accuracy is low | Accuracy is not always the best metric, as it can be biased towards the majority class. |
# | Val Loss | Loss value on the validation set | Lower val loss indicates better accuracy on new data | Longer if val loss is high | Can be used to detect overfitting |
# | Val Accuracy | Accuracy value on the validation set | Higher val accuracy indicates better generalization to new data | Longer if val accuracy is low | Helps to monitor how well the model is generalizing |
# | Time | The time taken to complete one epoch or the entire training process | Longer time may allow for more complex models and better accuracy | Longer with more data or more complex models | Training time can be reduced with distributed training |
# 

# %% [markdown]
# Distributed learning is a method of training machine learning models using multiple computers working in parallel.

# %% [markdown]
# Table 3: Calculation Metrics and Impact
# 
# | Metric       | Formula                                    | Dataset      | Impact                                                  | Neat Fact                                                                                                          |
# |--------------|--------------------------------------------|--------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
# | Loss         | $L = -\frac{1}{n} \sum_{i=1}^n y_i\log(y'_i)$ | Training Set | Measures the difference between predicted and true labels | A lower loss indicates a better fit between the model and the training data.                                      |
# | Accuracy     | $\frac{\text{number of correct predictions}}{\text{total number of predictions}}$ | Training Set | Measures the percentage of correctly classified images | A higher accuracy indicates a better fit between the model and the training data.                                  |
# | Val Loss     | $L = -\frac{1}{n} \sum_{i=1}^n y_i\log(y'_i)$ | Validation Set | Measures the difference between predicted and true labels for the validation set | A lower validation loss indicates that the model is not overfitting to the training set.                            |
# | Val Accuracy | $\frac{\text{number of correct predictions}}{\text{total number of predictions}}$ | Validation Set | Measures the percentage of correctly classified images for the validation set | A higher validation accuracy indicates that the model is not overfitting to the training set.                      |
# | Time         | N/A                                        | N/A          | The time it takes for one epoch to complete              | Training time can be impacted by the size of the dataset, the complexity of the model, and the hardware used.     |
# 

# %% [markdown]
# # Building intuition and understanding around the neural network. 

# %% [markdown]
# This code will create two plots. The first plot displays the training and validation accuracy over time (epochs). The second plot shows the training and validation loss over time (epochs). These plots help visualize how the model's performance changes over the course of training.
# 
# A well-trained model should show an increase in accuracy and a decrease in loss over time for both training and validation datasets. If the validation loss starts to increase while the training loss continues to decrease, this may indicate overfitting. In such a case, consider using techniques like regularization, early stopping, or increasing the size of the dataset to improve the model's performance.

# %% [markdown]
# *Table 6: Techniques to Combat Over Fitting*
# | Technique                 | Purpose                                                                                                       | Implementation                                                                                                                          |
# |---------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
# | Regularization            | Prevent overfitting by adding a penalty term to the loss function, discouraging the model from learning complex features that don't generalize well to the validation dataset. | - L1 regularization: Adds the sum of absolute weights to the loss function. (`tf.keras.regularizers.l1(l1=0.01)`).<br>- L2 regularization: Adds the sum of squared weights to the loss function (`tf.keras.regularizers.l2(l2=0.01)`).<br>- Apply regularization to specific layers in the model, for example:<br>  `tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))` |
# | Early Stopping            | Stop training the model when the validation loss starts to increase, preventing overfitting.                                             | - Use `tf.keras.callbacks.EarlyStopping` with a chosen `monitor` (e.g., 'val_loss') and `patience` (number of epochs with no improvement).<br>- Add the callback to the `fit` method: `callbacks=[early_stopping_callback]`                                      |
# | Increasing Dataset Size   | Improve the model's ability to generalize by providing more diverse examples, reducing overfitting.                                      | - Collect more data: Obtain additional labeled examples.<br>- Data augmentation: Apply random transformations to the existing dataset, such as rotation, scaling, or flipping, to create new examples. Use `tf.keras.preprocessing.image.ImageDataGenerator` with augmentation options.       |
# 

# %%
# Train the neural network with the training and validation datasets
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15
)

# Plot training and validation accuracy and loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the training and validation accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='best')

# Plot the training and validation loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(loc='best')

plt.show()


# %% [markdown]
# # Confusion Matrix

# %% [markdown]
# This code will compute and display the confusion matrix as a heatmap, with labels for each category (Cat and Not Cat). The matrix will show the number of true positives, true negatives, false positives, and false negatives.

# %%
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get the true labels and predicted labels from the validation dataset
y_true = []
y_pred = []

for images, labels in validation_dataset:
    predictions = model.predict(images)
    predictions = np.where(predictions < 0.5, 0, 1).flatten()
    
    y_true.extend(labels)  # Removed .numpy()
    y_pred.extend(predictions)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cat', 'Cat'], yticklabels=['Not Cat', 'Cat'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# %%
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
saved_model = load_model('cat_not_cat_classifierv6.h5')

# Get the true labels and predicted labels from the validation dataset
y_true = []
y_pred = []

for images, labels in validation_dataset:
    predictions = saved_model.predict(images)
    predictions = np.where(predictions < 0.5, 0, 1).flatten()
    
    y_true.extend(labels)
    y_pred.extend(predictions)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cat', 'Cat'], yticklabels=['Not Cat', 'Cat'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# %% [markdown]
# # Hyper Parameters 
# Hyperparameters are the parameters that are set before the training process begins, and they define the structure and behavior of the neural network. These parameters are not updated during the training process like weights and biases. Examples of hyperparameters include the learning rate, batch size, number of layers, and number of hidden units in each layer.
# 
# The accuracy of the neural network can vary significantly as you adjust different hyperparameters. For instance, if the learning rate is too high, the model may overshoot the optimal solution and never converge. If it's too low, the model may take too long to converge. Similarly, batch size can impact the training time and generalization of the model.
# 
# To visualize the impact of hyperparameters on the neural network's accuracy, you can perform a grid search over a range of hyperparameter values and plot the results. The code below will do this for learning rate and batch size:

# %%


learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64]

# Store the validation accuracy for each combination of hyperparameters
validation_accuracies = np.zeros((len(learning_rates), len(batch_sizes)))

for i, lr in enumerate(learning_rates):
    for j, bs in enumerate(batch_sizes):
        # Create and compile the model with the current hyperparameters
        model = create_model()  # Make sure the 'create_model' function exists in your code
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model with the current batch size
        history = model.fit(train_dataset.batch(bs), epochs=10, validation_data=validation_dataset.batch(bs), verbose=0)
        
        # Store the maximum validation accuracy achieved during training
        validation_accuracies[i, j] = max(history.history['val_accuracy'])

# Plot the validation accuracies as a heatmap
sns.heatmap(validation_accuracies, annot=True, cmap='Blues', xticklabels=batch_sizes, yticklabels=learning_rates, fmt='.2f')
plt.xlabel('Batch Size')
plt.ylabel('Learning Rate')
plt.title('Validation Accuracy for Different Hyperparameters')
plt.show()


# %% [markdown]
# # Feature Maps
# This code snippet creates a file upload widget and triggers the visualize_feature_maps function when an image is uploaded. The function will visualize the feature maps of the selected layer for the uploaded image. You can change the selected_layer variable to visualize feature maps of different layers.
# 
# Thecode will display the feature maps generated by the selected layer of the neural network for the given input image. You can change the selected_layer variable to visualize the feature maps of different layers.
# 
# Visualizations of feature maps help us understand how the neural network is processing images by showing how different filters respond to the input image. In the earlier layers of the network, the feature maps typically represent low-level features, such as edges or textures. As you move deeper into the network, the feature maps tend to capture more complex, abstract features, which help the network in making the final classification decision. These visualizations can provide insights into the working of the neural network and help diagnose any potential issues in the model.

# %% [markdown]
# 

# %%


# Create a file upload widget
uploader = widgets.FileUpload(accept='image/*', multiple=False)
display(uploader)

# Function to visualize feature maps for the uploaded image
def visualize_feature_maps(change):
    if uploader.data:
        # Load the image from the uploaded data
        img = Image.open(BytesIO(uploader.data[0]))
        
        # Save the image temporarily
        img.save('uploaded_image.jpg', 'JPEG')

        # Load the image and get feature maps
        img_array = load_image('uploaded_image.jpg')
        
        # Choose the layer you want to visualize
        selected_layer = 1

        # Create a new model with the same input as your original model and the output of the selected layer
        layer_output = model.layers[selected_layer].output
        feature_map_model = Model(inputs=model.input, outputs=layer_output)

        # Get the feature maps for the input image
        feature_maps = feature_map_model.predict(img_array)

        # Visualize the feature maps
        n_maps = feature_maps.shape[-1]
        rows = int(np.sqrt(n_maps))
        cols = n_maps // rows

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.ravel()

        for i in range(rows * cols):
            axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# Attach the function to the file upload widget
uploader.observe(visualize_feature_maps, names='data')


# %% [markdown]
# 

# %%


# %% [markdown]
# ## Machine Learning vs Neural Networks in terms of Image Classification

# %% [markdown]
# *Similarities:*
# 
# 1. Both traditional ML algorithms and neural networks are used to learn patterns in data and make predictions based on those patterns.
# 2. Both involve a training process, where the model learns from a training dataset and is evaluated on a separate validation or test dataset.
# 3. Both types of models have hyperparameters that can be tuned to improve performance.
# 
# *Differences:*
# 
# 1. Traditional ML algorithms (e.g., SVM, Random Forest, K-Nearest Neighbors) are based on specific mathematical principles and are often simpler than neural networks. Neural networks, particularly deep learning models, can have many layers and a large number of parameters, making them more complex and potentially more expressive.
# 2. Neural networks are particularly well-suited for tasks involving high-dimensional data, such as images, audio, or text, where traditional ML algorithms may struggle. Convolutional Neural Networks (CNNs) are especially effective for image classification tasks.
# 3. Traditional ML algorithms usually require manual feature engineering, while neural networks can learn relevant features automatically through their hierarchical structure.
# Neural networks may require more computational resources and longer training times compared to traditional ML algorithms.

# %% [markdown]
# Table 1: Similarities between Traditional ML Algorithms and Neural Networks
# 
# | Aspect                      | Traditional ML Algorithms        | Neural Networks               |
# |-----------------------------|----------------------------------|-------------------------------|
# | Learning from data          | Yes                              | Yes                           |
# | Training process            | Train on a training dataset      | Train on a training dataset   |
# | Hyperparameter tuning       | Yes                              | Yes                           |
# 

# %% [markdown]
# Table 2: Differences between Traditional ML Algorithms and Neural Networks
# | Aspect                      | Traditional ML Algorithms        | Neural Networks               |
# |-----------------------------|----------------------------------|-------------------------------|
# | Model complexity            | Simpler, based on specific math  | Complex, many layers          |
# | Handling high-dimensional data | May struggle                  | Well-suited (e.g., CNNs for images) |
# | Feature engineering         | Manual                           | Automatic                     |
# | Computational resources     | Typically less demanding         | May require more resources    |
# 

# %% [markdown]
# Table 3: Steps to adapt TensorFlow code to work with a Traditional ML Algorithm
# | Step                        | Description                                                  |
# |-----------------------------|--------------------------------------------------------------|
# | 1. Feature extraction       | Extract relevant features from images                        |
# | 2. Data preparation         | Preprocess features and labels, split into train and test    |
# | 3. Model training           | Train chosen ML algorithm on preprocessed data               |
# | 4. Model evaluation         | Evaluate performance using metrics (e.g., accuracy, F1)      |
# | 5. Hyperparameter tuning    | Perform grid search or random search to find optimal params  |
# 

# %% [markdown]
# ##### EXTRA'S ######

# %% [markdown]
# There are several other ways to visualize and build intuition around your neural network model and project. Some of these methods include:
# 
# 1. Visualizing intermediate activations: Visualizing the intermediate activations of the model can help you understand how the different layers are transforming the input image. This can be done by creating a new model with the same input as your original model and the output of the intermediate layers.
# 
# 2. Visualizing filters/weights: Visualizing the filters or weights in the convolutional layers can help you understand the kind of features the network is learning to recognize. Filters in the earlier layers usually capture low-level features like edges and textures, while filters in the deeper layers capture more complex and abstract features.
# 
# 3. Visualizing class activation maps (CAMs): CAMs provide a way to visualize which regions of the input image contribute the most to the model's prediction for a specific class. This can help you understand the spatial information the model is using to make its decision.
# 
# 4. t-SNE or UMAP embeddings: Using dimensionality reduction techniques like t-SNE or UMAP can help you visualize the high-dimensional feature space of your model in a 2D or 3D plot. This can provide insights into how the model is clustering similar images together and separating different classes.
# 
# 5. Training and validation curves: Plotting the training and validation accuracy and loss curves over time can help you understand how well your model is learning from the data and if it is overfitting or underfitting.
# 
# 6. Precision-Recall curves and ROC curves: These curves help you assess the performance of your model across various decision thresholds, and can provide insights into the trade-offs between true positive rate, false positive rate, and overall accuracy.
# 
# 7. Model architecture visualization: Visualizing the architecture of your neural network can help you understand the structure and complexity of the model, and can provide insights into how the different layers interact with each other.
# 
# 8. Saliency maps: Saliency maps highlight the most important regions in the input image for making a specific prediction. This can help you understand which parts of the image the model is focusing on when making a decision.


