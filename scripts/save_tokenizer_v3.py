from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

tokenizer.save_pretrained("../models/roberta-harassment-v3")

print("Tokenizer saved successfully")