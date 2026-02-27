# AI Chat Harassment Detection System

A deep learning–based NLP system that detects harassment in chat messages using a fine-tuned BERT model.

This project is designed for real-world deployment in chat platforms, social media moderation, and online community safety.

---

# Project Overview

Online harassment is a major global problem. Manual moderation is slow, expensive, and inconsistent.

This system uses a fine-tuned BERT transformer model to automatically detect harassment in real time.

The model classifies text into:

* SAFE
* HARASSMENT

---

# Model Details

Base Model: bert-base-uncased
Architecture: Transformer (BERT)
Framework: PyTorch + HuggingFace

Training dataset size:

* 143,613 training samples
* 15,958 validation samples

---

# Performance

Validation Results:

Accuracy: 96.73%
F1 Score: 0.837
Precision: 0.849
Recall: 0.826

---

# Example Predictions

Input:

You are an idiot

Output:

HARASSMENT (Confidence: 99.98%)

Input:

I hope you have a great day

Output:

SAFE (Confidence: 99.99%)

---

# Project Structure

```
ai-chat-harassment-detection/

models/
saved trained model

notebooks/
training notebooks

data/
dataset files

predict.py
prediction script

requirements.txt
dependencies
```

---

# Installation

Clone repository:

git clone https://github.com/kumarc101203/ai-chat-harassment-detection.git

Go into folder:

cd ai-chat-harassment-detection

Install dependencies:

pip install -r requirements.txt

---

# Usage

Run prediction script:

python predict.py

Enter text and the model will classify it.

---

# Model Download

Full trained model available at:

https://huggingface.co/Silentspy03/bert-harassment-detector

---

# Applications

Chat moderation
Social media monitoring
Online gaming moderation
Community platforms
Customer support filtering

---

# Future Improvements

Multilingual support
Context-aware detection
Real-time deployment API
Web interface

---

# Author

Kumar

AI / Machine Learning Developer

---

# License

MIT License
