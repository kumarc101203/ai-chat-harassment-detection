import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# =====================
# LOAD DATA
# =====================

train_df = pd.read_csv("data/processed/train_processed.csv")
val_df   = pd.read_csv("data/processed/val_processed.csv")

train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)


# =====================
# CLASS WEIGHTS
# =====================

labels = train_df["harassment"]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float)


# =====================
# TOKENIZER
# =====================

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(example):

    return tokenizer(
        example["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)


train_dataset = train_dataset.rename_column("harassment", "labels")
val_dataset   = val_dataset.rename_column("harassment", "labels")

train_dataset.set_format("torch", columns=["input_ids","attention_mask","labels"])
val_dataset.set_format("torch", columns=["input_ids","attention_mask","labels"])


# =====================
# MODEL
# =====================

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)


# =====================
# CUSTOM TRAINER WITH WEIGHTS
# =====================

class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):   

        labels = inputs.get("labels")

        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =====================
# METRICS
# =====================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    acc = accuracy_score(labels, predictions)

    return {

        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# =====================
# TRAINING ARGS
# =====================

training_args = TrainingArguments(

    output_dir="../models/roberta-harassment-v3",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=1,

    eval_strategy="epoch",
    save_strategy="epoch",

    fp16=True,

    report_to="none"
)


# =====================
# TRAINER
# =====================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)


# =====================
# TRAIN
# =====================

trainer.train()


# =====================
# SAVE
# =====================

trainer.save_model("../models/roberta-harassment-v3")

tokenizer.save_pretrained("../models/roberta-harassment-v3")

print("Training complete and model saved.")