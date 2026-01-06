# Facial-Expression-Identifier
An AI-based facial expression recognition system that classifies human emotions using **DeepFace embeddings** and a **Deep Neural Network (DNN)**. The system supports **real-time emotion detection** through webcam input.

# Facial Expression Identifier 

An AI-based facial expression recognition system that classifies human emotions using **DeepFace embeddings** and a **Deep Neural Network (DNN)**. The system supports **real-time emotion detection** through webcam input.

## Features
- Facial expression classification using **DeepFace (VGG-Face)** embeddings
- Real-time webcam-based emotion detection
- Face detection using **DNN*
- Model training and evaluation with standard metrics

## Tech Stack
- Python
- OpenCV
- DeepFace
- TensorFlow / Keras
- DNN
- Scikit-learn
- NumPy

## Dataset
- Custom facial image dataset collected from **self and peers**
- Folder-based structure (one folder per emotion)


Dataset is not included for privacy reasons.

##  How It Works
1. Detects face using **MTCNN**
2. Extracts facial embeddings using **DeepFace (VGG-Face)**
3. Trains a **DNN classifier** on embeddings
4. Predicts emotions in real time via webcam

##  Installation
```bash
pip install -r requirements.txt

