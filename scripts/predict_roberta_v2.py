import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# MODEL PATH
# ==============================

MODEL_PATH = "models/roberta-harassment-v2"


# ==============================
# LOAD TOKENIZER
# ==============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# ==============================
# LOAD MODEL
# ==============================

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


# ==============================
# DEVICE SETUP
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()


print("✅ RoBERTa v2 model loaded successfully")


# ==============================
# PREDICTION FUNCTION
# ==============================

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

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        confidence, predicted_class = torch.max(probs, dim=1)

    label = "HARASSMENT" if predicted_class.item() == 1 else "SAFE"

    return label, confidence.item()


# ==============================
# INTERACTIVE LOOP
# ==============================

while True:

    text = input("\nEnter text (type exit to quit): ")

    if text.lower() == "exit":

        print("Exiting...")

        break


    label, confidence = predict(text)


    print("Prediction:", label)

    print("Confidence:", round(confidence, 4))