# 🩺 PCOS Detection from Ultrasound Images using Transfer Learning (InceptionV3)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

This repository contains a complete reproduction of the paper **"Transfer-Based Deep Learning Technique for PCOS Detection using Ultrasound Images"** (presented at NMITCON 2023). The project leverages a fine-tuned **InceptionV3** model, pre-trained on ImageNet, to achieve state-of-the-art performance in classifying Polycystic Ovary Syndrome (PCOS) from ultrasound scans.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [1. Preprocessing Pipeline](#1-preprocessing-pipeline)
  - [2. Augmentation Strategy](#2-augmentation-strategy)
  - [3. Model Architecture](#3-model-architecture)
  - [4. Training Configuration](#4-training-configuration)
- [Results](#-results)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

---

## 🎯 Project Overview

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age. Early detection is crucial for effective management. This project uses deep transfer learning to automatically detect PCOS from ultrasound images, providing a non-invasive, efficient, and accurate diagnostic tool.

The model achieves **99.48% accuracy** on the test set, demonstrating the effectiveness of transfer learning in medical image analysis.

---

## 🎯 Problem

PCOS diagnosis using ultrasound is:
- Time-consuming  
- Depends on expert interpretation  
- Prone to errors  

This project builds an **automated detection system**.

---

---

## ✨ Key Features

- **Complete Pipeline**: End-to-end implementation from data loading to model evaluation
- **Robust Preprocessing**: Automatic detection and removal of corrupt images, grayscale conversion, and standardized resizing
- **Advanced Two-Stage Augmentation**:
  - Offline: +3 augmented images per input
  - Online: Real-time augmentation during training
- **State-of-the-Art Architecture**: InceptionV3 pre-trained on ImageNet
- **Two-Phase Training Strategy**:
  - Phase 1: Base model frozen, custom head trained
  - Phase 2: Top layers unfrozen for end-to-end fine-tuning
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix

---


## 📁 Dataset

The project uses the `pcos-detection-using-ultrasound-images` dataset from Kaggle, created by Anagha Choudhari.

### Dataset Statistics

| Split          | Infected | Not Infected | Total   |
|----------------|----------|--------------|---------|
| **Training**   | 781      | 1,143        | 1,924   |
| **Test**       | 787      | 1,145        | 1,932   |
| **Total**      | 1,568    | 2,288        | 3,856   |

> **Note**: The dataset includes a data cleaning step that automatically identifies and removes corrupt images from the dataset.

---

## 🔬 Methodology

### 1. Preprocessing Pipeline

- Convert images to grayscale using OpenCV (`COLOR_BGR2GRAY`)
- Resize all images to 224×224 pixels
- Stack grayscale to 3 channels for InceptionV3 compatibility

### 2. Augmentation Strategy

#### Offline Augmentation (+3 images per input)

| Augmentation | Description |
|--------------|-------------|
| Transform 1 | Horizontal flip + 10° rotation |
| Transform 2 | Contrast enhancement (1.5x boost) -10° rotation |
| Transform 3 | Sharpening + shift transformation |

#### Online Augmentation (during training)

| Augmentation | Range |
|--------------|-------|
| Random rotation | ±20° |
| Width/height shift | ±10% |
| Zoom | ±15% |
| Shear | ±10% |
| Horizontal flip | Yes |
 

### 3. Model Architecture

- InceptionV3 (pretrained on ImageNet)
- GlobalAveragePooling2D
- Dense(256, activation='relu')
- Dropout(0.5)
- Dense(1, activation='sigmoid')

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

