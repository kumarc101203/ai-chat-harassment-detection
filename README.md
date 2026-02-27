# AI Chat Harassment Detection using BERT

## Overview

This project builds a deep learning model to detect harassment in chat messages using BERT (Bidirectional Encoder Representations from Transformers).

The model is trained on the Jigsaw Toxic Comment dataset and fine-tuned for binary harassment classification.

---

## Features

* BERT-based text classification
* GPU training using NVIDIA RTX 4060
* HuggingFace Transformers
* PyTorch backend
* Real-world harassment detection pipeline

---

## Dataset

Jigsaw Toxic Comment Classification Dataset

Total samples: 159,571

Class distribution:

* Non-harassment: 89.8%
* Harassment: 10.2%

---

## Model

Base Model:

bert-base-uncased

Fine-tuned for binary classification.

---

## Training

Batch size: 8
Epochs: 1
Learning rate: 2e-5

GPU: NVIDIA RTX 4060 Laptop GPU

---

## Project Structure

```
ai-chat-harassment-detection/
│
├── datasets/
├── notebooks/
├── README.md
├── README_PLAN.md
├── requirements.txt
```

---

## Status

Training in progress.

---

## Future Improvements

* Increase epochs
* Improve accuracy
* Deploy as API
* Build web interface

---

## Author

Kumar
