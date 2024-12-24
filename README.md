# CS554_Project
# Face Verification for Passport Ownership
>This repository contains the implementation of an automated system for verifying passport ownership by analyzing and matching a claimant's face with the photograph on their passport personalization page. The system employs the Viola-Jones algorithm for face detection and uses Eigenfaces with Principal Component Analysis (PCA) for feature extraction and dimensionality reduction.


> **Keywords**: [Eigenface](https://en.wikipedia.org/wiki/Eigenface), [Identity verification](https://en.wikipedia.org/wiki/Identity_verification_service), [Viola–Jones](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)

> ---

## Authors

Kousar Kousar, Aqsa Shabbir, and Utku Oktay  

---

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Dataset](#data-set)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Robust Face Detection**: Viola-Jones algorithm with Haar-like features and cascaded classifiers.
- **Efficient Feature Extraction**: Eigenfaces with PCA for dimensionality reduction and discriminative representation.
- **Customizable Verification**: Supports SVMs, Decision Trees, and Euclidean Distance classifiers.
- **Diverse Dataset Training**: Ensures adaptability across real-world scenarios using varied datasets.
---
## Getting Started

- Clone the repository:
   ```bash
   cd CS554_Project
---

## Installation

- You need to install the following dependencies in Python3 for this project:
   ```bash
   pip3 install numpy scipy matplotlib torch pandas scikit-learn scikit-image tqdm morphops

---

## Usage Instructions
### Running Viola-Jones Detector
- Use the `VJ_Final.py` script to execute the Viola-Jones Detection part. 
   ```bash
   python VJ_Final.py

 
### Running Eigenfaces
- Use the `eigenfaces.py` script to execute the face verification. 
   ```bash
   python eigenfaces.py 


---
## Dataset Description  

### Viola-Jones Face Detection  
- **Dataset**: WIDER FACE.  
- **Details**: 10,000 examples, equally split between 5,000 positive (faces) and 5,000 negative (non-faces) samples.  
- **Split**:  
  - Training set: 8,000 images (80%).  
  - Testing set: 2,000 images (20%).  

### Eigenfaces  
- **Dataset**: MUCT Dataset (for training) and AT&T Faces (for testing).  
- **Split**:  
  - Training: 30 individuals (~300 images).  
  - Validation: 4 individuals (~40 images).  
  - Testing: 6 individuals (~60 images).  

### Active Appearance Models (AAM)  
- **Dataset**: MUCT Dataset.  
- **Split**:  
  - Training: 30 individuals.  
  - Testing: 10 individuals (~400 total images).

---

## Acknowledgments

This project is our final project for the Bilkent CS 554: Computer Vision Course. We extend our gratitude to Professor Hamdi Dibeklioglu for his invaluable guidance and support throughout all stages of the project.
