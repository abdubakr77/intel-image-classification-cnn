# Intel Image Classification using (Transfer Learning)

## Overview
This project focuses on building a robust multi-class image classification model using deep learning techniques. 

Instead of building a Convolutional Neural Network (CNN) from scratch, this project leverages **Transfer Learning** using the pre-trained **ResNet50** architecture. This approach significantly improves feature extraction, accelerates training, and boosts overall accuracy when classifying natural scenes.

---

## Dataset
The model is trained and evaluated on the **Intel Image Classification dataset**, which contains approximately 24,335 images divided into training, testing, and prediction sets. 

The dataset classifies images into the following **6 categories**:
* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

---

## Objectives
* Apply advanced deep learning techniques to a real-world multi-class image classification task.
* Utilize the **ResNet50** architecture and Transfer Learning to maximize feature extraction efficiency.
* Optimize model performance using callbacks to achieve high accuracy and minimal loss while avoiding overfitting.

---

## Model Architecture
Built with **Keras/TensorFlow**, the model pipeline includes:
* **Base Model:** Pre-trained `ResNet50` (using ImageNet weights) for powerful spatial feature extraction.
* **Pooling:** `GlobalAveragePooling` to reduce dimensionality.
* **Regularization:** `Dropout` layers to prevent the model from overfitting on the training data.
* **Classification Head:** Fully connected (`Dense`) layers culminating in a `Softmax` activation for the 6-class output.
* **Optimization Control:** `EarlyStopping` callback monitoring validation loss to automatically halt training at peak performance.

---

## Technologies Used
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib

---

## Results
*Training history and evaluation metrics (Accuracy and Loss graphs) will be added here upon the completion of the training process.*

---
