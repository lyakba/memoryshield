import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

dataset = load_dataset("npow/memshield-bench")

# Convert labels:
# clean = 0
# poisoned = 1

label_map = {
    "clean": 0,
    "poisoned": 1
}

def convert_labels(example):
    example["labels"] = label_map[example["label"]]
    return example

dataset = dataset.map(convert_labels)
dataset = dataset.remove_columns(["label"])


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------
# 3. Tokenize the text
# --------------------------------------------------

def tokenize(example):
    return tokenizer(
        example["content"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, batched=True)


# --------------------------------------------------
# 4. Load DistilBERT
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# --------------------------------------------------
# 5. Training settings
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./distilbert_results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

print("\nStarting training...\n")

trainer.train()
trainer.save_model("./memoryshield_model")
tokenizer.save_pretrained("./memoryshield_model")


# --------------------------------------------------
# 7. Evaluate on test set
# --------------------------------------------------

print("\nEvaluating on test set...\n")

results = trainer.evaluate(tokenized_dataset["test"])

print(results)