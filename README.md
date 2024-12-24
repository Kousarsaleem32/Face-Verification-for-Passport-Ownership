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
- [Citations](#citations)
- [License](#license)
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
   cd Beacon-Reconstruction-Attack
---

## Installation

- You need to install the following dependencies in Python3 for this project:
   ```bash
   pip3 install numpy scipy matplotlib torch pandas 

---

## Usage Instructions
### Running Baseline Simulations
- Use the `baseline.py` script to execute the baseline beacon reconstruction attack. Parameters allow you to configure beacon size and SNP count.
   ```bash
   python baseline.py --beacon_Size 50 --snp_count 30

### Key Arguments

- `--beacon_Size`: Number of individuals targeted for reconstruction.
- `--snp_count`: Size of the SNP subset in the beacon.
 
### Running Optimization Simulations
- Use the `simulate.py` script to execute the beacon reconstruction attack. Parameters allow you to configure beacon size, SNP count, correlation and frequency epoch.
   ```bash
   python simulate.py --beacon_Size 50 --snp_count 30 --corr_epoch 1001 --freq_epoch 501

### Key Arguments

- `--beacon_Size`: Number of individuals targeted for reconstruction.
- `--snp_count`: Size of the SNP subset in the beacon.
- `--corr_epoch`: Stage 1 of optimization for correlation loss.
- `--freq_epoch`: Stage 2 of optimization for frequency loss.

---
## License

[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)


© 2024 Beacon Defender Framework.

**For commercial use, please contact.**


---

## Acknowledgments

We would like to thank Göktuğ Gürbüztürk, Efe Erkan, Deniz Aydemir, İrem Aydın, Kerem Ayöz, and Erman Ayday for their contributions to this project.
