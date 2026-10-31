# 🩺 PCOS Detection from Ultrasound Images using Transfer Learning (InceptionV3)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

An automated medical image classification project that explores the detection of Polycystic Ovary Syndrome (PCOS) from ovarian ultrasound images using **InceptionV3-based transfer learning**.
This repository contains a complete reproduction of the paper **"Transfer-Based Deep Learning Technique for PCOS Detection using Ultrasound Images"** (presented at NMITCON 2023). The project leverages a fine-tuned **InceptionV3** model, pre-trained on ImageNet, to achieve state-of-the-art performance in classifying Polycystic Ovary Syndrome (PCOS) from ultrasound scans.
This project was developed as part of the Deep Learning Capstone Project (Academic Year 2025–26).
A major focus of this work is investigating dataset leakage in medical image classification and developing an improved evaluation pipeline to obtain more reliable estimates of model generalization
**Disclaimer:** This is an academic research prototype. It is not a
> medical device and must not be used to diagnose PCOS or guide clinical
> decisions.
## 📋 Table of Contents

-   [Project Overview](#project-overview)
-   [Problem Statement](#problem-statement)
-   [Project Objectives](#project-objectives)
-   [Key Features](#key-features)
-   [Dataset](#dataset)
-   [Methodology](#methodology)
-   [Model Architecture](#model-architecture)
-   [Implementation Approaches](#implementation-approaches)
-   [Data Leakage Investigation](#data-leakage-investigation)
-   [Evaluation Strategy](#evaluation-strategy)
-   [Results and Observations](#results-and-observations)
-   [Technologies Used](#technologies-used)
-   [Installation and Setup](#installation-and-setup)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)
-   [Research Reference](#research-reference)
-   [Acknowledgements](#acknowledgements)

---

---

## 🎯 Project Overview

Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder that can
affect women of reproductive age. Ovarian ultrasound is one of the
imaging methods used by clinicians when assessing ovarian morphology as
part of PCOS evaluation.

This project explores binary image classification of ovarian ultrasound
scans:

-   **PCOS:** images labelled as infected/PCOS in the dataset.
-   **Non-PCOS:** images labelled as notinfected/normal in the dataset.

The model uses **InceptionV3**, pretrained on ImageNet, as its
convolutional feature-extraction backbone. A custom classification head
predicts the probability of an image belonging to the PCOS class.

The work was developed in two stages: first reproducing the research
workflow, and then revising the data-splitting and evaluation strategy
after investigating possible overlap between the original dataset
folders.


---

## 🎯 Problem Statement
Deep learning models can achieve high scores on medical image
classification tasks, but those scores are meaningful only when the
evaluation data are independent of the training data.

If duplicate or highly similar images appear across training and test
folders, a model may recognize previously seen image content instead of
learning features that generalize to unseen scans. This can lead to
overly optimistic evaluation results.

This project investigates PCOS classification while examining how
dataset overlap and evaluation design can affect the reported
performance of a medical image model.

---
---

## 🎯  Project Objectives
-   Develop a binary image-classification workflow for PCOS detection
    from ovarian ultrasound images.
-   Reproduce an InceptionV3 transfer-learning approach described in an
    existing research paper.
-   Apply image loading, cleaning, resizing, grayscale/RGB conversion
    and normalization.
-   Explore offline and online image augmentation.
-   Investigate possible duplicate or highly similar images in the
    original dataset.
-   Revise the evaluation strategy by creating a validation split from
    the training folder and using a separate external image collection.
-   Evaluate model behaviour using accuracy, precision, recall,
    F1-score, confusion matrices and ROC analysis where available.
-   Examine the limitations of deep learning evaluation on small
    medical-image datasets.

---
---

## ✨ Key Features


-   **InceptionV3 transfer learning:** uses ImageNet-pretrained
    convolutional features.
-   **Two implementation stages:** preserves the research reproduction
    and subsequent evaluation experiments.
-   **Image preprocessing:** includes corruption checks, grayscale or
    RGB conversion, resizing to 224 × 224 and normalization.
-   **Data augmentation:** explores geometric and image-enhancement
    transformations.
-   **Leakage investigation:** examines overlap concerns in the original
    dataset and checks exact duplicates within the training folder.
-   **Revised data split:** creates a stratified training/validation
    split using the original training folder only.
-   **External evaluation:** evaluates predictions on a separate
    collection of 54 images loaded by the revised notebook.
-   **Model interpretation:** includes Grad-CAM heatmap generation in
    the later experimental section.
-   **Performance analysis:** includes classification reports, confusion
    matrices, training curves and ROC-curve code.



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

The first notebook identified and removed **10 unreadable/corrupted
images from the original test folder**. The resulting test folder
contained 1,922 images: 781 PCOS and 1,141 non-PCOS. No corrupted files
were removed from the training folder in that run.
These counts describe the original folder organization. They do not
establish that the two folders contain independent images.

### 2. External image collection

The revised notebook loads an external archive named
`dataverse_files.zip` and organizes images into two classes based on
filenames:

-   14 images assigned to PCOS
-   40 images assigned to non-PCOS
-   **54 images loaded in total**

The notebook contains a comment referring to 454 images, but the saved
execution output shows that **54 images** were actually organized and
evaluated. The source citation and original dataset documentation for
this external archive should be added to the repository before treating
it as a fully documented independent benchmark.

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




### Original reproduction

The original model uses:

1.  InceptionV3 pretrained on ImageNet, with the original classification
    top removed.
2.  Global Average Pooling.
3.  Dense layer with 256 units and ReLU activation.
4.  Dropout with a rate of 0.5.
5.  One sigmoid output unit for binary classification.

  Parameter                           Original configuration
  --------------------------------- ------------------------
  Input shape                                  224 × 224 × 3
  Total parameters                                22,327,585
  Trainable parameters in Phase 1                    524,801
  Frozen parameters in Phase 1                    21,802,784
  Optimizer                                             Adam
  Learning rate                                        0.001
  Batch size                                              32
  Maximum epochs per phase                                10
  Loss                                  Binary cross-entropy

During fine-tuning, the last 50 layers of the InceptionV3 backbone are
unfrozen and the learning rate is reduced by a factor of 10.

### Revised RGB experiment

The first revised configuration uses:

-   InceptionV3 pretrained on ImageNet.
-   Global Average Pooling.
-   Batch Normalization.
-   Dropout (0.3).
-   Dense layer with 128 units and ReLU activation.
-   Batch Normalization.
-   Dropout (0.3).
-   Sigmoid output layer.

The model initially freezes the InceptionV3 backbone. It then unfreezes
the last 50 layers for fine-tuning.

  Parameter                          Revised RGB configuration
  -------------------------------- ---------------------------
  Input shape                                    224 × 224 × 3
  Total parameters                                  22,073,889
  Initially trainable parameters                       266,753
  Initial learning rate                                 0.0001
  Fine-tuning learning rate                            0.00001
  Batch size                                                32
  Stage 1 maximum epochs                                    20
  Stage 2 maximum epochs                                    30
  Loss                                    Binary cross-entropy
  Class imbalance                       Balanced class weights

Early stopping and learning-rate reduction callbacks are used during
training.

### Later grayscale experiment

A later section in the revised notebook defines another InceptionV3
configuration with the last 50 backbone layers trainable, Global Average
Pooling, a 128-unit Dense layer, Dropout (0.5) and a sigmoid output.
This experiment uses a learning rate of 0.0001, batch size 32 and 10
epochs.

This is a separate experimental configuration; its results should not be
merged with those of the revised RGB experiment.

## Implementation Approaches

### Approach 1: Original research reproduction

Notebook: `DL_CAPSTONE.ipynb` (confirm the exact filename used in the
repository).

This implementation follows the original research workflow:

1.  Load the original Kaggle training and test folders.
2.  Remove unreadable images.
3.  Convert images to grayscale and resize them to 224 × 224.
4.  Generate offline augmented copies and define online augmentation.
5.  Train an InceptionV3 model with a frozen backbone.
6.  Fine-tune the last 50 layers.
7.  Evaluate the model on the original test folder.
8.  Generate training curves, a confusion matrix and a classification
    report.

The original test generator is also passed as `validation_data` during
training and is used for checkpoint selection. Consequently, it is not
an untouched final test set.

### Approach 2: Revised RGB evaluation experiment

Notebook: the leakage-corrected notebook.

This configuration:

1.  Loads only the original Kaggle training folder.
2.  Checks for exact duplicate files within that folder.
3.  Creates a stratified 85:15 training/validation split (1,635 training
    images and 289 validation images).
4.  Uses RGB preprocessing and InceptionV3-specific preprocessing.
5.  Applies conservative online augmentation and balanced class weights.
6.  Trains the classification head and then fine-tunes the last 50
    backbone layers.
7.  Evaluates on the 54 images loaded from the external archive.

### Approach 3: Later grayscale experiment

A later section in the same revised notebook contains another
experiment. It creates an 80:20 split from the 1,924-image Kaggle
training folder (1,539 training images and 385 validation images), uses
grayscale preprocessing and trains a separate InceptionV3 configuration.

This section also evaluates on the 54-image external collection and
tests several classification thresholds. It is retained here as a later
experiment, not as an additional independent dataset or a separate
validated model.

## Data Leakage Investigation

The original Kaggle dataset was investigated after the initial model
produced nearly perfect performance.

The project report records overlapping or highly similar images between
the original training and test folders. This raises a concern that the
original test results may be optimistic because the evaluation images
may not be independent of the training data.

The original notebook also uses the original test generator as
validation data during training. This means the same folder contributes
to model selection and final evaluation.

The revised notebook changes the evaluation design by:

-   Excluding the original Kaggle test folder from the revised training
    and evaluation workflow.
-   Creating a new validation split from the original training folder.
-   Introducing a separate external image collection for evaluation.
-   Checking exact file hashes within the original training folder; the
    saved output reports three duplicate files within that folder.

**Important qualification:** the current duplicate-checking code checks
exact hashes within the training folder. It does not document a complete
cross-folder exact-duplicate audit or a perceptual-similarity audit. The
three duplicates are reported but are not removed before the
train/validation split. Therefore, the revised workflow reduces reliance
on the suspect original test folder, but the notebook does not establish
that every possible source of leakage has been eliminated.

## Evaluation Strategy

The project uses different evaluation strategies in the two stages.

  ------------------------------------------------------------------------
  Stage             Training data     Validation/model   Final evaluation
                                      selection          
  ----------------- ----------------- ------------------ -----------------
  Original          Original Kaggle   Original Kaggle    Same original
  reproduction      training folder   test folder        Kaggle test
                                                         folder

  Revised RGB       Kaggle training   Stratified 15%     External
  experiment        folder only       split from         collection (54
                                      training folder    images)

  Later grayscale   Kaggle training   Stratified 20%     External
  experiment        folder only       split from         collection (54
                                      training folder    images)
  ------------------------------------------------------------------------

The revised evaluation code includes:

-   Accuracy
-   Precision
-   Recall/sensitivity
-   Specificity
-   F1-score
-   Confusion matrix
-   ROC curve and ROC-AUC calculation
-   Classification report

The revised RGB experiment selects a classification threshold using
validation-set predictions before applying it to the external
collection. The later grayscale experiment explores thresholds of 0.5,
0.6, 0.7 and 0.8.

## Results and Observations

### 1. Original reproduction

The original notebook reports the following results on the cleaned
original test folder:

  Metric          Recorded result
  ------------- -----------------
  Test images               1,922
  Accuracy                100.00%
  Precision               100.00%
  Recall                  100.00%
  Specificity             100.00%
  F1-score                100.00%

The confusion matrix contains 781 correctly classified PCOS images and
1,141 correctly classified non-PCOS images, with no recorded false
positives or false negatives.

These results must be interpreted cautiously. The original train/test
folders were reported to contain overlapping or highly similar images,
and the original test folder was also used for validation and checkpoint
selection. The 100% score should therefore **not** be presented as
evidence of clinical performance or reliable generalization.

### 2. Revised RGB experiment

The saved classification report for the 54-image external collection
records:

  Class                    Precision   Recall   F1-score   Support
  ---------------------- ----------- -------- ---------- ---------
  Healthy / non-PCOS            1.00     0.10       0.18        40
  PCOS                          0.28     1.00       0.44        14
  **Overall accuracy**                          **0.33**    **54**

The result shows that the model identified the PCOS examples in this run
but incorrectly classified many non-PCOS images as PCOS. The small
external sample and class imbalance also limit the strength of
conclusions.

### 3. Later grayscale experiment

The later grayscale experiment reports the following result at a
classification threshold of 0.8:

  Class                    Precision   Recall   F1-score   Support
  ---------------------- ----------- -------- ---------- ---------
  Healthy / non-PCOS            0.71     0.12       0.21        40
  PCOS                          0.26     0.86       0.39        14
  **Overall accuracy**                          **0.31**    **54**

The notebook also contains results at other thresholds, with external
accuracy between approximately 0.28 and 0.31 in the saved outputs. These
are threshold-dependent results from a later experimental configuration
and should not be combined with the revised RGB experiment.

### Main observation

The original test-folder result was 100%, while the external evaluation
results were substantially lower. This difference highlights why an
apparently strong test score is not sufficient when dataset independence
is uncertain.

The external results also show weak recognition of the non-PCOS class.
Further work is needed before the model can be considered useful beyond
an academic experiment.

## Technologies Used

  Technology           Purpose
  -------------------- ------------------------------------------------------
  Python               Main programming language
  TensorFlow / Keras   Deep learning framework and model training
  InceptionV3          Image feature extraction through transfer learning
  OpenCV               Image loading, colour conversion and resizing
  Pillow (PIL)         Image verification and augmentation
  NumPy                Numerical and image-array operations
  scikit-learn         Data splitting, class weights and evaluation metrics
  Matplotlib           Training curves and visualizations
  Seaborn              Confusion-matrix visualization
  KaggleHub            Downloading the Kaggle dataset
  Google Colab         Notebook execution environment

## Installation and Setup

### Prerequisites

-   Python 3.x
-   pip
-   Jupyter Notebook or Google Colab
-   Internet access for downloading the Kaggle dataset and pretrained
    InceptionV3 weights
-   Optional GPU acceleration for faster training

### 1. Clone the repository

Replace the placeholders with the actual GitHub repository details.

``` bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

### 2. Install dependencies

``` bash
pip install kagglehub tensorflow opencv-python pillow numpy scikit-learn matplotlib seaborn
```

The notebooks also contain package-installation cells. Run the relevant
installation cell if using Google Colab.

### 3. Prepare the external image archive

The revised notebook expects an archive named:

``` text
/content/dataverse_files.zip
```

Upload the external archive to the Colab session before running the
external-data preparation and evaluation cells. The archive's source,
labels and usage permissions should be documented in the repository.

## Usage

### Option A: Google Colab

1.  Open the required notebook in Google Colab.
2.  Select a GPU runtime if available.
3.  Run the notebook cells in order.
4.  Allow `kagglehub` to download the public Kaggle dataset.
5.  For the revised notebook, upload `dataverse_files.zip` before
    running the external-data cells.
6.  Review the preprocessing outputs, training curves, classification
    report and confusion matrix.

### Option B: Local Jupyter Notebook

Install the dependencies, clone the repository and launch Jupyter:

``` bash
pip install jupyter
jupyter notebook
```

Open the notebook you want to reproduce and execute its cells in
sequence. Update any Colab-specific paths, such as
`/content/dataverse_files.zip`, when running locally.

### Notebook order

-   Run the original reproduction notebook to examine the research
    replication and original evaluation.
-   Run the revised notebook to examine the revised split, external
    evaluation and additional experiments.

## Project Structure

The repository should contain the README and the two notebooks. The
exact notebook filenames should match the files committed to GitHub.

``` text
PCOS-Detection-Deep-Learning/
│
├── DL_CAPSTONE.ipynb
├── PCOS_Detection_using_Ultrasound_Images_(Data_leakage_solved)(Deep_Learning).ipynb
├── README.md
│
└── outputs/                  # Optional: generated results, if committed
    ├── training_history.png
    ├── confusion_matrix.png
    └── saved_model/          # Optional: trained model files
```

The notebooks generate model and visualization files during execution.
Add those files to the repository only if they are actually present and
intended to be shared. Dataset images and large model files do not need
to be committed to GitHub.

## Limitations

-   **Small external evaluation set:** only 54 images are shown in the
    saved external evaluation outputs.
-   **Class imbalance:** the external collection contains 40 non-PCOS
    images and 14 PCOS images.
-   **External data provenance:** the notebook uses
    `dataverse_files.zip`, but its source and dataset documentation are
    not identified in the supplied report.
-   **Filename-based labels:** external images are assigned to the
    non-PCOS class when the filename contains `normal`; all other image
    filenames are assigned to PCOS. This rule requires manual label
    verification.
-   **Incomplete duplicate audit:** the revised notebook checks exact
    duplicates within the training folder but does not document a
    complete cross-folder or perceptual-similarity audit.
-   **Duplicates remain in the training folder:** three exact duplicates
    were reported, but the saved workflow does not remove them before
    splitting.
-   **Small validation and external samples:** results may vary with the
    split, sample composition and classification threshold.
-   **Different experimental configurations:** the revised notebook
    contains both an RGB progressive-fine-tuning experiment and a later
    grayscale experiment. Their results should be reported separately.
-   **Preprocessing consistency:** the original notebook generates
    offline augmented images but its training generator uses the
    processed, non-augmented directory.
-   **No clinical validation:** the model has not been evaluated in a
    clinical setting and cannot replace professional assessment.

## Future Improvements

-   Obtain a larger, well-documented external dataset with verified
    labels.
-   Record the external dataset's source, licence, collection details
    and class definitions.
-   Perform exact-hash and perceptual-similarity checks across all
    relevant dataset splits.
-   Remove duplicates before creating training and validation splits.
-   Use patient-level or study-level splitting if patient identifiers
    are available.
-   Keep the final test set completely separate from training, threshold
    selection and model checkpoint selection.
-   Consolidate the revised notebook into one clearly defined final
    pipeline.
-   Compare InceptionV3 with architectures such as ResNet50 and
    EfficientNet under the same evaluation protocol.
-   Add more robust evaluation, including cross-validation and
    confidence intervals.
-   Improve model interpretability and assess Grad-CAM outputs
    carefully.
-   Explore a deployment interface only after stronger independent
    validation.

## Research Reference

The original implementation is based on the following conference paper:

**Choudhari, A. et al. (2023).** *Transfer-Based Deep Learning Technique
for PCOS Detection using Ultrasound Images.*

Presented at the 2023 International Conference on Network, Multimedia
and Information Technology (NMITCON).

``` bibtex
@inproceedings{choudhari2023transfer,
  title={Transfer-Based Deep Learning Technique for PCOS Detection using Ultrasound Images},
  author={Choudhari, Anagha and others},
  booktitle={2023 International Conference on Network, Multimedia and Information Technology (NMITCON)},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgements

-   Anagha Choudhari and the research team for the original research
    paper and Kaggle dataset.
-   Kaggle for hosting the publicly available ultrasound image dataset.
-   The TensorFlow and Keras teams for the deep learning framework.
-   Google Colab for providing the notebook execution environment.
