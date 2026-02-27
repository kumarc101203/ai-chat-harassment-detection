# scripts/predict.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# HuggingFace model repo
MODEL_NAME = "Silentspy03/bert-harassment-detector"


print("Loading model from HuggingFace...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()

print("Model loaded successfully!\n")


# Prediction function
def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        confidence, prediction = torch.max(probs, dim=1)

    label = "HARASSMENT" if prediction.item() == 1 else "SAFE"

    return label, confidence.item()



# Interactive loop
while True:

    text = input("Enter text (or type exit): ")

    if text.lower() == "exit":
        break

    label, confidence = predict(text)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}\n")