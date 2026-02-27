import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "models/roberta-harassment-v3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


# =========================
# USE GPU IF AVAILABLE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()

print("\nModel loaded successfully")
print("Using device:", device)


# =========================
# PREDICTION FUNCTION
# =========================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    label = "HARASSMENT" if pred.item() == 1 else "SAFE"

    return label, confidence.item()


# =========================
# TEST LOOP
# =========================

while True:

    text = input("\nEnter text (type exit to quit): ")

    if text.lower() == "exit":
        break

    label, confidence = predict(text)

    print("Prediction:", label)
    print("Confidence:", round(confidence, 4))