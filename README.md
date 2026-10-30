# 🩺 PCOS Detection from Ultrasound Images using Transfer Learning (InceptionV3)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

An automated medical image classification project that explores the detection of Polycystic Ovary Syndrome (PCOS) from ovarian ultrasound images using **InceptionV3-based transfer learning**.
This repository contains a complete reproduction of the paper **"Transfer-Based Deep Learning Technique for PCOS Detection using Ultrasound Images"** (presented at NMITCON 2023). The project leverages a fine-tuned **InceptionV3** model, pre-trained on ImageNet, to achieve state-of-the-art performance in classifying Polycystic Ovary Syndrome (PCOS) from ultrasound scans.
This project was developed as part of the Deep Learning Capstone Project (Academic Year 2025–26).
A major focus of this work is investigating dataset leakage in medical image classification and developing an improved evaluation pipeline to obtain more reliable estimates of model generalization

## 📋 Table of Contents

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Implementation Approaches](#-implementation-approaches)
- [Data Leakage Investigation](#️-data-leakage-investigation)
- [Evaluation Strategy](#-evaluation-strategy)
- [Results and Observations](#-results-and-observations)
- [Technologies Used](#️-technologies-used)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Limitations](#️-limitations)
- [Future Improvements](#-future-improvements)
- [Research Reference](#-research-reference)
- [Acknowledgements](#-acknowledgements)

---

---

## 🎯 Project Overview

Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder affecting women of reproductive age. Ultrasound imaging is one of the clinical methods used to assess ovarian morphology during PCOS evaluation.

This project explores the application of deep learning and transfer learning to classify ovarian ultrasound images into two categories:

* **PCOS**: Ultrasound images associated with PCOS.

* **Non-PCOS**: Ultrasound images not associated with PCOS.

The implementation uses **InceptionV3**, a convolutional neural network pretrained on ImageNet, as the feature extraction backbone.

The project follows two implementation stages:

1. Reproduction of the original research methodology.

2. An improved evaluation pipeline developed after identifying potential data leakage in the original dataset split.

The second implementation emphasizes proper dataset separation, external validation and more realistic model evaluation.

**Important**: This is an academic research prototype and is not intended to replace clinical diagnosis or medical assessment.

---

## 🎯 Problem Statement
Deep learning has considerable potential in medical image analysis. However, the reliability of medical AI models depends not only on model architecture but also on dataset quality and evaluation methodology.

An improperly separated dataset can introduce data leakage, resulting in artificially inflated performance metrics.

This project investigates both the classification task and the importance of reliable evaluation in healthcare AI applications.


---
---

## 🎯  Project Objectives
* Develop a deep learning pipeline for PCOS classification using ovarian ultrasound images.

* Reproduce the methodology of an existing research paper using InceptionV3 transfer learning.

* Perform image preprocessing, normalization and data augmentation.

* Investigate potential data leakage between the original training and test datasets.

* Develop a revised validation strategy to reduce evaluation bias.

* Introduce external validation to examine model generalization.

* Evaluate classification performance using standard machine learning metrics.

* Understand the limitations of deep learning-based medical image classification.

---
---

## ✨ Key Features

- **Transfer Learning**: Uses pretrained InceptionV3 for image feature extraction.
- **Image Preprocessing**: Includes grayscale conversion, resizing and normalization.
- **Data Augmentation**: Applies image transformations to improve training diversity.
- **Binary Classification**: Distinguishes between PCOS and non-PCOS ultrasound images.
- **Two Implementation Pipelines**:Preserves the original reproduction and improved evaluation approaches.
- **Data Leakage Investigation**: Introduces external validation data to assess generalization.
- **Performance Evaluation**:Includes accuracy, precision, recall, F1-score and confusion matrix analysis.


## 📁 Dataset

The project uses the publicly available Kaggle dataset:

**PCOS Detection Using Ultrasound Images**

* Dataset creator: Anagha Choudhari

* Platform: Kaggle

* Task: Binary medical image classification

**Dataset link:**

[https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

### Original Dataset Statistics
The original dataset contains the following reported image counts:
| Dataset Split          | PCOS(Infected) |Non-PCOS( Not Infected) | Total   |
|----------------|----------|--------------|---------|
| **Training**   | 781      | 1,143        | 1,924   |
| **Test**       | 787      | 1,145        | 1,932   |
| **Total**      | 1,568    | 2,288        | 3,856   |

>These statistics describe the original dataset organization and should not be interpreted as independent evaluation sample counts.
>
### Dataset considerations:

During dataset investigation, overlapping or highly similar images were identified between the original training and test folders.
This observation raised concerns about the independence of the original evaluation data and motivated the development of the improved validation pipeline.


---

## 🔬 Methodology
The project follows an image classification workflow consisting of preprocessing, augmentation, feature extraction, model training and evaluation.
### 1. Image Preprocessing
The ultrasound images undergo the following preprocessing operations:
- Image loading and dataset verification
- Identification and removal of corrupted images.
- Conversion of images into grayscale using OpenCV.
- Resizing images to 224 × 224 pixels.
- Conversion of grayscale images into three-channel representations for compatibility with InceptionV3.
- Pixel normalization before model training.

### 2. Data Augmentation
Image augmentation is used to introduce additional variations into the training data.
The original implementation explores:
| Augmentation | Description |
|--------------|-------------|
| Horizontal Flip| Horizontally flips the ultrasound image|
|Rotation | Applies small angular rotations |
|Contrast Enhancement | Adjusts image contrast |
| Sharpening| Enhances image details|
| Width/Height Shift| Applies positional transformations|
| Zoom| Changes image magnification|

Both offline and online augmentation strategies are explored in the original implementation.

Augmentation is intended to improve training diversity and reduce overfitting. It does not replace proper dataset separation.

#### 3. Feature Extraction
The project uses InceptionV3 pretrained on ImageNet.

The pretrained convolutional layers extract hierarchical image features, which are subsequently used for binary classification.
#### 4. Classification
The extracted features are passed through a custom classification head to predict the probability of an ultrasound image belonging to the PCOS category.




### 🧠 Model Architecture
Base architecture: InceptionV3 (ImageNet pretrained)

The classification architecture consists of:

InceptionV3 convolutional backbone.

Global Average Pooling layer.

Fully connected Dense layer with 256 neurons and ReLU activation.

Dropout layer with a dropout rate of 0.5.

Output Dense layer with one neuron and sigmoid activation.

The output represents the predicted probability for binary classification.

**Model Parameters:**

| Parameter Type | Count |
|----------------|-------|
| Total params | 22,327,585 |
| Trainable params (Phase 1) | 524,801 |
| Frozen params (Phase 1) | 21,802,784 |

### 4. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Loss Function | Binary Crossentropy |
| Epochs | 10 |

---

## 📈 Results

The model was evaluated on a held-out test set of 1,922 images.

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | 99.48% |
| Precision | 88.2% |
| Recall | 87.2% |
| F1-Score | 86.1% |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PCOS-Detection-Deep-Learning.git
cd PCOS-Detection-Deep-Learning
```
### Step 2: Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/PCOS-Detection-Deep-Learning.git
cd PCOS-Detection-Deep-Learning
```
## Step 3: Run the Notebook

You can run this project in one of two ways:

**Option A: Google Colab (Recommended)**

-  Upload DL_CAPSTONE.ipynb to Google Colab

-  Set runtime type to GPU

-  Run all cells sequentially

**Option B: Local Jupyter Notebook**
```bash
jupyter notebook DL_CAPSTONE.ipynb
```
---

## 💻 Usage

**Quick Start**

The notebook is fully self-contained and will automatically:

- Install required packages

- Download the dataset using kagglehub

- Clean and preprocess all images

- Apply augmentations

- Build and train the InceptionV3 model

- Display results and visualizations

**Expected Output**

When you run the notebook, you'll see:

-  Dataset download progress and structure verification

-   Image cleaning and preprocessing logs

- Sample images with their labels

-  Model architecture summary

-   Training progress with accuracy/loss curves

-  Final evaluation metrics and confusion matrix

---

## 📁 Project Structure


```
PCOS-Detection-Deep-Learning/

├── DL_CAPSTONE.ipynb
├── requirements.txt
├── README.md

├── data/
│   ├── train/
│   │   ├── infected/
│   │   └── notinfected/
│   ├── test/
│   ├── processed_train/
│   ├── processed_test/
│   └── augmented_train/

└── outputs/
    ├── sample_images.png
    ├── model_weights.h5
    └── training_history.png
```

---
## 🔧 Future Improvements

- Implement cross-validation for more robust evaluation

- Add Grad-CAM visualizations for model interpretability

- Experiment with other architectures (ResNet50, EfficientNet)

- Deploy model as a web application using Streamlit or FastAPI

---
## 📜 Citation


@inproceedings{choudhari2023transfer,
  title={Transfer-Based Deep Learning Technique for PCOS Detection using Ultrasound Images},
  author={Choudhari, Anagha and others},
  booktitle={2023 International Conference on Network, Multimedia and Information Technology (NMITCON)},
  year={2023},
  organization={IEEE}
}

---
## 📄 License

This project is licensed under the MIT License.

---
## 🙏 Acknowledgements

-  Anagha Choudhari and team for the original research paper and dataset

-  Kaggle for hosting the dataset

- TensorFlow/Keras team for the deep learning framework

-  Google Colab for providing free GPU resources

