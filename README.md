# ✋ Hand Gesture Number Classifier

### CNN-Based Hand Gesture Number Recognition (0–9)

<p>
  A Deep Learning project that recognizes numbers from hand gestures using a custom Convolutional Neural Network (CNN).
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-CNN-red?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Classification-purple" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Classes-10-brightgreen" alt="Classes">
  <img src="https://img.shields.io/badge/Accuracy-98%25-success" alt="Accuracy">
</p>

---

**Recognizing numbers through hand gestures using Convolutional Neural Networks.**

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Highlights](#-project-highlights)
- [Dataset](#-dataset)
- [Data Preprocessing & Augmentation](#-data-preprocessing--augmentation)
- [CNN Architecture](#-cnn-architecture)
- [Model Training](#-model-training)
- [Evaluation & Results](#-evaluation--results)
- [Technologies & Libraries](#-technologies--libraries)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Overview

**Hand Gesture Number Classifier** is a deep learning project designed to recognize numerical hand gestures using image classification techniques.

The system uses a custom-built Convolutional Neural Network (CNN) developed with TensorFlow and Keras to classify hand gesture images into 10 different categories, representing numbers from **0 to 9**.

The model learns visual patterns, hand shapes, and finger configurations from labeled images to identify the corresponding numerical gesture.

The main objective of this project is to explore practical applications of deep learning and computer vision through building, training, evaluating, and testing a CNN-based image classification model.

## ✨ Project Highlights

- 🔟 **10-Class Classification:** Recognizes hand gestures representing numbers from 0 to 9.
- 🧠 **Custom CNN Architecture:** Built with 5 convolutional layers.
- 🖼️ **RGB Image Processing:** Uses 100 × 100 × 3 input images.
- 🔄 **Data Augmentation:** Uses multiple augmentation techniques to improve generalization.
- ⚙️ **SGD Optimization:** Trains the model using Stochastic Gradient Descent.
- 📉 **Learning Rate Scheduling:** Adjusts the learning rate during training.
- 🛑 **Early Stopping:** Helps prevent unnecessary training and overfitting.
- 📊 **Model Evaluation:** Evaluated using classification metrics and a confusion matrix.
- 🎯 **98% Test Accuracy:** Achieved on the reported evaluation set of 50 images.

## 📂 Dataset

The project uses a dataset named **`sign_language_project`**, obtained from Kaggle.

### Dataset Information

| Property | Description |
|---|---|
| Dataset Name | sign_language_project |
| Source | Kaggle |
| Total Images | 2,062 |
| Number of Classes | 10 |
| Class Labels | 0–9 |
| Images per Class | Approximately 206 |
| Image Format | JPG |
| Original Image Size | 100 × 100 |
| Input Image Shape | 100 × 100 × 3 |
| Color Mode | RGB |

### Dataset Classes

The dataset contains 10 classes:

| Class | Meaning |
|---|---|
| 0 | Number 0 |
| 1 | Number 1 |
| 2 | Number 2 |
| 3 | Number 3 |
| 4 | Number 4 |
| 5 | Number 5 |
| 6 | Number 6 |
| 7 | Number 7 |
| 8 | Number 8 |
| 9 | Number 9 |

### Dataset Splitting

The dataset was provided with separate training, validation, and testing directories.

The intended distribution is:

| Subset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

*Note: Exact split counts and per-class distributions should be verified against the original dataset.*

## 🖼️ Data Preprocessing & Augmentation

To improve model robustness and reduce overfitting, several image augmentation techniques were applied during training.

### Applied Techniques

| Technique | Purpose |
|---|---|
| Gaussian Noise | Adds random noise to images to improve robustness. |
| Random Translation | Randomly shifts image contents horizontally and vertically. |
| Random Zoom | Applies random zoom transformations. |
| Random Contrast | Changes image contrast to simulate different lighting conditions. |

### Input Specifications

All images are processed using the following input dimensions:

```python
input_shape = (100, 100, 3)
```

- Height: 100 pixels
- Width: 100 pixels
- Channels: 3 (RGB)

These transformations help the model learn more general visual representations instead of memorizing specific training images.

## 🧠 CNN Architecture

The classification model is built using a custom Convolutional Neural Network.

### Architecture Overview

The network consists of:

| Layer Type | Configuration |
|---|---|
| Conv2D | 5 convolutional layers |
| MaxPooling2D | 3 pooling layers |
| Dense | 512 neurons |
| Activation Functions | Leaky ReLU and ELU |
| Batch Normalization | Applied |
| Dropout | Applied |
| Output Layer | Dense + Softmax |
| Number of Output Classes | 10 |

### Architecture Pipeline

```text
       Input Image
      100 × 100 × 3
             |
             ▼
    Convolutional Layers
       (5 × Conv2D)
             |
             ▼
     Max Pooling Layers
       (3 × Pooling)
             |
             ▼
    Feature Extraction
             |
             ▼
      Batch Normalization
             |
             ▼
         Dropout
             |
             ▼
      Fully Connected
        Dense (512)
             |
             ▼
       Softmax Output
             |
             ▼
       10 Predictions
        (Classes 0–9)
```

*This diagram represents the high-level architecture. The exact layer ordering and filter configurations are defined in the training notebook.*

### Activation Functions

Two activation functions are used in the network:

- **Leaky ReLU:** Allows a small gradient for negative input values.
- **ELU:** Provides smooth negative outputs and helps improve learning behavior.

### Output Layer

The final layer uses the Softmax activation function to produce a probability distribution across the 10 classes.

The predicted class is selected using the maximum probability:

```python
predicted_class = np.argmax(predictions, axis=1)
```

## ⚙️ Model Training

The model was trained using TensorFlow and Keras.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | SGD (Stochastic Gradient Descent) |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 32 |
| Training Epochs | Approximately 14 |
| Learning Rate | Dynamic |
| Learning Rate Strategy | LearningRateScheduler |
| Early Stopping | Enabled |
| Output Activation | Softmax |

### Optimization Strategies

**1. Stochastic Gradient Descent (SGD)**

SGD was selected as the optimization algorithm to update model weights during training.

**2. Learning Rate Scheduling**

A learning rate scheduler was used to adjust the learning rate across different training epochs.

**3. Early Stopping**

Early stopping was implemented to monitor training progress and prevent unnecessary training when model performance stopped improving.

These strategies help control the training process and improve model generalization.

## 📊 Evaluation & Results

The model was evaluated using a classification report containing:

- Accuracy
- Precision
- Recall
- F1-Score

### Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | **98%** |
| Macro Precision | 98% |
| Macro Recall | 98% |
| Macro F1-Score | 98% |
| Weighted Precision | 98% |
| Weighted Recall | 98% |
| Weighted F1-Score | 98% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 1.00 | 1.00 | 5 |
| 1 | 1.00 | 1.00 | 1.00 | 5 |
| 2 | 1.00 | 1.00 | 1.00 | 5 |
| 3 | 1.00 | 1.00 | 1.00 | 5 |
| 4 | 0.83 | 1.00 | 0.91 | 5 |
| 5 | 1.00 | 1.00 | 1.00 | 5 |
| 6 | 1.00 | 1.00 | 1.00 | 5 |
| 7 | 1.00 | 0.80 | 0.89 | 5 |
| 8 | 1.00 | 1.00 | 1.00 | 5 |
| 9 | 1.00 | 1.00 | 1.00 | 5 |

### Evaluation Summary

The model achieved an overall accuracy of **98%** on the reported evaluation set.

Most classes achieved perfect precision, recall, and F1-scores.

The main classification errors occurred in:

- Class 4: Precision of 0.83.
- Class 7: Recall of 0.80.

### Important Evaluation Note

The classification report contains 50 evaluated images, with 5 samples per class.

Although the intended dataset split is 80% training, 10% validation, and 10% testing, the reported evaluation sample count should not be assumed to represent the entire test directory.

Further evaluation on a larger and more diverse test set is recommended to obtain a more reliable estimate of real-world performance.

## 🛠️ Technologies & Libraries

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Scikit-learn | Model evaluation and metrics |
| Matplotlib | Data visualization |
| Pillow (PIL) | Image loading and processing |
| MLxtend | Machine learning visualization utilities |

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Erfanjenab/Hand-Gesture-Number-Classifier.git
```

### 2. Navigate to the Project Directory

```bash
cd Hand-Gesture-Number-Classifier
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib pillow mlxtend
```

## 💻 Usage

The project provides a Jupyter Notebook for model development, training, and evaluation.

### Run the Notebook

```bash
jupyter notebook
```

Open:

```text
sign_language.ipynb
```

Run the notebook cells to explore the dataset, preprocess images, train the CNN, and evaluate its performance.

### Image Prediction

The trained model can be used to classify new hand gesture images.

The prediction pipeline follows these general steps:

1. Load a new hand gesture image.
2. Apply the same preprocessing used during training.
3. Resize the image to 100 × 100 pixels.
4. Convert the image to the expected RGB format.
5. Pass the image to the trained CNN.
6. Obtain class probabilities using Softmax.
7. Select the predicted class using `argmax`.

The input preprocessing must remain consistent with the training pipeline.

## 📁 Project Structure

```text
Hand-Gesture-Number-Classifier/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── sign_language.ipynb
│
├── README.md
│
└── .gitignore
```

*This structure represents the main project files and directories. Additional files may be added as the project evolves.*

## 🔮 Future Improvements

Although the current model achieves promising results, several improvements are planned:

- [ ] Experiment with Transfer Learning.
- [ ] Explore pretrained architectures such as MobileNet and EfficientNet.
- [ ] Improve model generalization using additional training data.
- [ ] Perform more comprehensive evaluation on the complete test set.
- [ ] Optimize model inference performance.
- [ ] Develop a web application for image-based predictions.
- [ ] Integrate the trained model into a web-based interface.
- [ ] Explore real-time hand gesture recognition using a webcam.

## 👨‍💻 Author

**Erfan Jenab**

Computer Engineering Student | Machine Learning & Deep Learning Enthusiast

GitHub: [@Erfanjenab](https://github.com/Erfanjenab)

---

<div align="center">

### ⭐ If you find this project interesting, consider giving it a star!

**Built with Python, TensorFlow, and a passion for Deep Learning.**

</div>
