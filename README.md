# MLOPS_G116

# Project: NeuroClassify - MLOps Pipeline for Brain Tumor Detection

## 1. Dataset Selection
**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
* **Source:** Kaggle (Masoud Nickparvar)
* **Type:** Medical Imaging (2D MRI Slices)

## 2. Model Selection
**Model:** ResNet-18
* **Architecture:** Convolutional Neural Network (CNN) with residual connections.
* **Rationale:** We selected ResNet-18 because it provides a strong baseline for medical image classification while being computationally efficient. This allows for faster iteration cycles in our MLOps pipeline (training, testing, deployment) without requiring massive GPU resources.

## 3. Project Description

### a. Overall Goal of the Project
The goal is to engineer a robust, end-to-end MLOps pipeline that automates the classification of brain MRI scans into four diagnostic categories. The project focuses on the *operational* aspects of machine learning—such as automated data validation, model versioning, continuous integration/training (CI/CD), and model monitoring—rather than just achieving the highest possible accuracy.

### b. Data Description
We will be using the **Brain Tumor MRI Dataset** aggregated from multiple open-source medical repositories.
* **Total Samples:** Approximately **7,023 images**.
* **Modality:** 2D MRI scans (JPG format).
* **Classes:** 4 categories (Glioma, Meningioma, Pituitary, No Tumor).
* **Size:** ~150 MB.
* **Structure:** The data is organized into train/test folders by class label, simplifying the initial data loading pipeline.

### c. Expected Models
* **Baseline:** We will start with a **ResNet-18** backbone pre-trained on ImageNet, modifying the final fully connected layer to output 4 classes.
* **Future/Experimental:** If the MLOps pipeline is stable, we may experiment with **EfficientNet-B0** for better parameter efficiency or explore **MONAI**-specific implementations (e.g., DenseNet121) tailored for healthcare imaging.
