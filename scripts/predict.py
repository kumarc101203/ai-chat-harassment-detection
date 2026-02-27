import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/bert-harassment"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded")


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        harassment_prob = probs[0][1].item()

    label = "HARASSMENT" if harassment_prob > 0.5 else "SAFE"

    return label, harassment_prob


# test loop
while True:

    text = input("\nEnter text: ")

    label, confidence = predict(text)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")