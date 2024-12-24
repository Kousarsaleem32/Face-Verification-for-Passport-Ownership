# CS554_Project
# Face Verification for Passport Ownership
>This repository contains the implementation of an automated system for verifying passport ownership by analyzing and matching a claimant's face with the photograph on their passport personalization page. The system employs the Viola-Jones algorithm for face detection and uses Eigenfaces with Principal Component Analysis (PCA) for feature extraction and dimensionality reduction.
>This project is our final project for the Bilkent CS 554: Computer Vision Course. We extend our gratitude to Professor Hamdi Dibeklioglu for his invaluable guidance and support throughout all stages of the project.

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

## Acknowledgments

We would like to thank Göktuğ Gürbüztürk, Efe Erkan, Deniz Aydemir, İrem Aydın, Kerem Ayöz, and Erman Ayday for their contributions to this project.
