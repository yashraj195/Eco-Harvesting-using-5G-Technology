# 🌾 Eco-Harvesting using 5G Technology 

> AI-powered Eco Harvesting using **ResNet-50 + Transformer Encoder**, deployed with **Gradio** on **Hugging Face Spaces**.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-orange?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow?style=for-the-badge&logo=huggingface)

</p>

---

# 🌱 Overview

Plant diseases significantly reduce crop yield and often remain unnoticed until substantial damage has occurred. Traditional manual inspection is time-consuming, labour-intensive, and prone to human error.

This project presents an **AI-powered disease detection system** capable of identifying multiple paddy leaf diseases from uploaded images using a **Spatio-Temporal Deep Learning (STDD)** architecture.

The application provides instant predictions through an intuitive web interface, making it suitable for precision agriculture and early disease diagnosis.

---

# 🚀 Live Demo

## 🤗 Hugging Face Deployment

**Try the application here:**

### 👉 https://huggingface.co/spaces/yashraj195/capstone01

---

# ✨ Features

- 🌾 Detects **10 Paddy Leaf Classes**
- 🧠 ResNet-50 Spatial Feature Extraction
- 🔄 Transformer Encoder for Temporal Learning
- 📊 Confidence score for every prediction
- 📱 Simple Gradio Web Interface
- ⚡ Fast CPU inference
- 🎨 Modern responsive UI
- ☁️ Deployed on Hugging Face Spaces

---

# 🦠 Diseases Detected

| Disease |
|----------|
| Healthy / Normal |
| Blast |
| Brown Spot |
| Dead Heart |
| Downy Mildew |
| Tungro |
| Hispa |
| Bacterial Leaf Blight |
| Bacterial Leaf Streak |
| Bacterial Panicle Blight |

---

# 🏗 Project Architecture

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
ResNet-50 CNN
(Spatial Feature Extraction)
      │
      ▼
Transformer Encoder
(Temporal Feature Learning)
      │
      ▼
Fully Connected Layer
      │
      ▼
Softmax Classification
      │
      ▼
Disease Prediction
+
Confidence Score
```

---

# 📂 Project Structure

```
Paddy-Disease-Detection/
│
├── app.py                  # Gradio Application
├── model.py                # Model Architecture
├── requirements.txt
├── paddy_stdd_model_full.pt
│
├── assets/
│   ├── screenshots/
│   ├── architecture.png
│   └── workflow.png
│
├── README.md
└── LICENSE
```

---

# 🧠 Model Architecture

The proposed model follows a **Spatio-Temporal Disease Detection (STDD)** framework.

### Spatial Feature Extraction

- ResNet-50
- ImageNet Pretrained Weights
- Global Feature Embeddings

↓

### Temporal Learning

- Multi-head Transformer Encoder
- Sequence Modelling
- Context-aware Feature Learning

↓

### Classification

- Fully Connected Layer
- Softmax Output
- 10 Disease Classes

---

# ⚙️ Tech Stack

| Category | Technology |
|------------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | TorchVision |
| Image Processing | Pillow |
| Frontend | Gradio |
| Deployment | Hugging Face Spaces |
| Data Processing | NumPy, Pandas |

---

# 📸 Application Screenshots

## Home Page

<img width="1897" height="852" alt="image" src="https://github.com/user-attachments/assets/7f9261e0-79e4-4d88-a8e6-a1a3eeeedd19" />

---

## Prediction Result

<img width="1900" height="912" alt="image" src="https://github.com/user-attachments/assets/1d0a88c6-cb4b-4e5e-b6fd-c23179feb281" />

---

## Confidence Scores

<img width="751" height="537" alt="image" src="https://github.com/user-attachments/assets/74cd43a7-a59c-4a5a-8d63-080b2e88afaa" />

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Paddy-Disease-Detection.git
```

Move into the project

```bash
cd Paddy-Disease-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

# 📦 Requirements

- Python 3.10+
- PyTorch
- TorchVision
- Pillow
- NumPy
- Pandas
- Gradio

Install everything using:

```bash
pip install -r requirements.txt
```

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 95.4% |
| Precision | 94.1% |
| Recall | 93.6% |
| Classes | 10 |
| Inference Time | < 3 seconds |

---

# 🌍 Future Improvements

- Mobile Application
- Drone Image Support
- Cloud-based API
- Disease Treatment Recommendations
- Multi-language Support

---

# 👨‍💻 Authors

**Capstone Project**

- Yash Raj

---

# 📜 License

This project is developed for educational and research purposes.

---
