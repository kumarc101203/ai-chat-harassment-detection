# AI Chat Harassment Detection — Project Plan

---

# 1. Problem Statement

Online platforms such as chat applications, gaming communities, and social media face a serious problem of harassment and abusive language.

Manual moderation is:

• Slow
• Expensive
• Not scalable

This project aims to build an AI system that can automatically detect harassment in chat messages in real time.

The model will classify text into:

• Harassment
• Non-Harassment

This can help improve user safety and automate moderation.

---

# 2. Project Objectives

Primary Objective:

Build a real-time harassment detection system using a Transformer-based deep learning model.

Secondary Objectives:

• Train a high-accuracy NLP model
• Deploy the model for real-time inference
• Create a dashboard for monitoring
• Build a complete production-ready pipeline

---

# 3. Dataset Information

Dataset Used:

Jigsaw Toxic Comment Classification Dataset

Source:

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Total Samples:

159,571 comments

Original Labels:

• toxic
• severe_toxic
• obscene
• threat
• insult
• identity_hate

---

# 4. Label Engineering

The original dataset contained 6 toxicity labels.

These were combined into a single binary label.

Label Creation Logic:

harassment = 1
If ANY toxicity label is 1

harassment = 0
If ALL toxicity labels are 0

This simplifies the problem into binary classification.

Final Dataset Format:

Columns:

• comment_text
• harassment

---

# 5. Data Pipeline

Raw Data Location:

data/raw/

Processed Data Location:

data/processed/

Steps:

Step 1 — Load Raw Data

Step 2 — Create harassment label

Step 3 — Select required columns

Step 4 — Train Validation Split

Train Size:

90%

Validation Size:

10%

Files Generated:

train_processed.csv
val_processed.csv

---

# 6. Model Architecture

Model Used:

BERT Base Uncased

Source:

https://huggingface.co/bert-base-uncased

Architecture Type:

Transformer Encoder

Number of Parameters:

110 Million

---

# 7. Model Input

Input:

Chat Text

Processing Steps:

Text

↓

Tokenizer

↓

Token IDs

↓

BERT Model

---

# 8. Model Output Layer

Output Layer:

Fully Connected Linear Layer

Output Size:

2 neurons

Classes:

Class 0 — Non Harassment
Class 1 — Harassment

Activation Function:

Softmax

Loss Function:

CrossEntropyLoss

---

# 9. Training Configuration

Hardware Used:

GPU:

NVIDIA RTX 4060 Laptop GPU

VRAM:

8GB

Framework:

PyTorch

Library:

HuggingFace Transformers

---

# 10. Training Hyperparameters

Learning Rate:

2e-5

Batch Size:

08

Epochs:

1 (initial testing)

Max Token Length:

128

Optimizer:

AdamW

---

# 11. Evaluation Metrics

Model performance will be measured using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

These metrics help evaluate real-world performance.

---

# 12. Inference Pipeline

Inference Flow:

User Input Text

↓

Tokenizer

↓

Trained BERT Model

↓

Prediction

↓

Harassment / Non-Harassment Output

---

# 13. Deployment Plan

Backend:

FastAPI

Frontend:

Web Dashboard

Deployment Target:

Local Deployment

Future Deployment:

Cloud Deployment

AWS / Azure / GCP

---

# 14. Folder Structure

Project Structure:

backend/

frontend/

models/

data/

raw/

processed/

notebooks/

deployment/

configs/

src/

---

# 15. Expected Performance

Target Accuracy:

Greater than 90%

Target F1 Score:

Greater than 0.80

---

# 16. Future Improvements

Future Enhancements:

Multi-label classification

Real-time chat integration

Browser extension

Mobile App Integration

Continuous Training Pipeline

---

# 17. Real-World Applications

This system can be used in:

Social Media Platforms

Gaming Platforms

Chat Applications

Online Communities

Education Platforms

---

# 18. Project Outcome

Final Deliverables:

Trained AI Model

Inference Pipeline

Deployment Ready System

Interactive Dashboard

Production-Ready Codebase

---

# 19. Conclusion

This project builds a real-world AI system capable of detecting harassment automatically using modern Transformer-based deep learning.

It demonstrates:

Natural Language Processing

Deep Learning

Model Training

Deployment

Production Pipeline Design

---

END OF PLAN
