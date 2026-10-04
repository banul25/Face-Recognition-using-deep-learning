# Facial Emotion Recognition using Deep Learning

An end-to-end deep learning framework designed to automatically recognize and classify human facial emotions from images and live video streams in real time.

This repository compares multiple computer vision architectures — **Custom CNN**, **MobileNetV2**, and **EfficientNetB0** — to identify a suitable model for real-time facial emotion recognition.

## 📌 Project Overview

Facial expressions are a critical form of non-verbal communication. Automated emotion recognition systems can be applied to areas such as:

* Healthcare
* Human-Computer Interaction (HCI)
* E-learning
* Customer experience analysis
* Computer vision applications
* Real-time video analysis

The system classifies facial expressions into **7 distinct emotion categories**:

| Emotion         | Description                 |
| --------------- | --------------------------- |
| 😠 **Angry**    | Angry facial expression     |
| 🤢 **Disgust**  | Disgusted facial expression |
| 😨 **Fear**     | Fearful facial expression   |
| 😄 **Happy**    | Happy facial expression     |
| 😐 **Neutral**  | Neutral facial expression   |
| 😢 **Sad**      | Sad facial expression       |
| 😲 **Surprise** | Surprised facial expression |

## 🎯 Objectives

The main objectives of this project are:

1. **Custom CNN Baseline**
   Build and train a custom Convolutional Neural Network from scratch.

2. **Transfer Learning with MobileNetV2**
   Implement MobileNetV2 as a lightweight architecture suitable for fast inference and resource-constrained devices.

3. **Transfer Learning with EfficientNetB0**
   Use EfficientNetB0 to improve feature extraction and classification performance.

4. **Performance Benchmarking**
   Compare the models using:

   * Accuracy
   * Precision
   * Recall
   * F1-Score
   * Confusion Matrix

5. **Real-Time Integration**
   Integrate the best-performing model with OpenCV for real-time emotion detection from video streams.

## 📊 Evaluation & Model Comparison

### Results Summary

| Model              | Architecture                       |   Accuracy |   Precision |     Recall |   F1-Score | Primary Strength                   |
| ------------------ | ---------------------------------- | ---------: | ----------: | ---------: | ---------: | ---------------------------------- |
| **Custom CNN**     | Built from scratch                 |   Baseline |    Baseline |   Baseline |   Baseline | Baseline comparison                |
| **MobileNetV2**    | Lightweight transfer learning      |       High | **Highest** |   Moderate |   Moderate | Fast inference and edge deployment |
| **EfficientNetB0** | Compound scaling transfer learning | **60.30%** |  **74.41%** | **45.81%** | **0.5670** | Strong overall performance         |

### Key Findings

* **EfficientNetB0** achieved an accuracy of **60.30%**, recall of **45.81%**, and F1-score of **0.5670**.
* EfficientNetB0 provided the strongest overall performance among the evaluated models based on the reported metrics.
* The **F1-score** is particularly useful for this dataset because of class imbalance between emotions.
* **MobileNetV2** demonstrated strong precision while maintaining a lightweight architecture, making it suitable for applications where inference speed and computational efficiency are important.
* Minority classes such as **Disgust** can be more difficult to classify because they contain fewer training examples than classes such as **Happy**.

## 🧠 Model Architectures

### 1. Custom CNN

A Convolutional Neural Network was developed from scratch to establish a baseline.

The model learns visual features such as:

* Edges
* Shapes
* Facial features
* Expression patterns

The Custom CNN provides a reference point for evaluating the transfer-learning models.

### 2. MobileNetV2

MobileNetV2 is a lightweight convolutional architecture designed for efficient computer vision tasks.

It is useful for:

* Mobile devices
* Edge devices
* Real-time applications
* Low-resource environments

Transfer learning allows the model to use previously learned visual features and adapt them to the seven emotion classes.

### 3. EfficientNetB0

EfficientNetB0 uses a compound scaling strategy to balance:

* Network depth
* Network width
* Input resolution

Transfer learning with EfficientNetB0 allows the model to extract useful facial features while reducing the amount of training required from scratch.

## 🛠️ System Pipeline

The overall system follows this workflow:

```text
Input Image / Live Video
          │
          ▼
     Face Detection
          │
          ▼
   Face Region Cropping
          │
          ▼
     Image Resizing
          │
          ▼
    Normalization
          │
          ▼
   EfficientNetB0
          │
          ▼
 Emotion Probabilities
          │
          ▼
 Predicted Emotion
          │
          ▼
  Display on Video Feed
```

## 🎥 Real-Time Inference

The real-time pipeline consists of the following steps:

### 1. Face Detection

OpenCV detects faces from the live camera feed.

### 2. Preprocessing

The detected face region is:

* Cropped from the frame
* Resized to the required input dimensions
* Normalized before being passed to the model

### 3. Emotion Classification

The trained **EfficientNetB0** model generates probability scores for all seven emotion classes.

### 4. Result Overlay

The predicted emotion and confidence score are displayed directly on the video stream.

Example:

```text
┌─────────────────────────────┐
│                             │
│       [ Face Region ]       │
│                             │
│   Emotion: Happy            │
│   Confidence: 87.4%         │
│                             │
└─────────────────────────────┘
```

## 📁 Repository Structure

```text
Face-Recognition-using-deep-learning/
│
├── README.md
│
└── deepFER_Face_recognition_using_deep_learning.ipynb
    # Data processing, model training, evaluation,
    # and experimentation
```

## 🚀 Getting Started

### Prerequisites

Make sure Python is installed on your system.

Install the required dependencies:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn
```

You can also install Jupyter Notebook if it is not already installed:

```bash
pip install notebook
```

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/banul25/Face-Recognition-using-deep-learning.git
```

### 2. Navigate to the Project Directory

```bash
cd Face-Recognition-using-deep-learning
```

### 3. Open the Jupyter Notebook

```bash
jupyter notebook deepFER_Face_recognition_using_deep_learning.ipynb
```

### 4. Run the Notebook

Execute the notebook cells sequentially to:

1. Load the dataset
2. Preprocess the images
3. Prepare training and testing data
4. Train the Custom CNN
5. Train MobileNetV2 using transfer learning
6. Train EfficientNetB0 using transfer learning
7. Evaluate the models
8. Generate performance metrics
9. Analyze the confusion matrices
10. Export the trained model for inference

## 📈 Evaluation Metrics

The models are evaluated using several metrics.

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many predictions for a particular emotion were actually correct.

### Recall

Measures how many actual samples of an emotion were correctly identified.

### F1-Score

The F1-score combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

### Confusion Matrix

A confusion matrix helps identify which emotions are being correctly classified and which emotions are being confused with one another.

## ⚠️ Class Imbalance

The dataset contains different numbers of samples for each emotion.

For example, **Disgust** contains significantly fewer samples than classes such as **Happy**.

This imbalance can affect model performance, particularly recall for minority classes.

Therefore, accuracy alone is not sufficient to evaluate the model. Precision, recall, F1-score, and confusion matrices are also considered.

## 🔮 Future Improvements

### Class Imbalance Mitigation

Improve minority-class performance using:

* Class weighting
* Focal loss
* Data augmentation
* Oversampling

### Web Deployment

Build an interactive application using:

* Streamlit
* FastAPI
* Flask

This would allow users to upload images or use a webcam directly through a web interface.

### Multi-Face Recognition

Extend the real-time system to detect and classify emotions for multiple faces simultaneously.

### Model Optimization

Further optimize the model for real-time deployment using techniques such as:

* Model quantization
* TensorFlow Lite
* ONNX
* Hardware acceleration

### Improved Dataset

Training with a larger and more balanced facial-expression dataset could improve generalization and minority-class performance.

## 📌 Conclusion

This project demonstrates an end-to-end approach to facial emotion recognition using deep learning.

Three architectures are compared:

* **Custom CNN** for establishing a baseline
* **MobileNetV2** for lightweight and efficient inference
* **EfficientNetB0** for improved feature extraction and overall classification performance

Based on the reported evaluation results, **EfficientNetB0 achieved 60.30% accuracy and a 0.5670 F1-score**, while MobileNetV2 provided a lightweight alternative suitable for resource-constrained environments.

The project also extends the trained model toward **real-time emotion recognition using OpenCV**, providing a foundation for future deployment in interactive computer-vision applications.
