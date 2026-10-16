from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# --------------------------------------------------
# LOAD STAGE 1 DATA
# --------------------------------------------------

print("Loading Stage 1 dataset...")

dataset = load_from_disk("continual_stage1")

print(f"Examples: {len(dataset)}")


# --------------------------------------------------
# PREPARE LABELS
# --------------------------------------------------

def convert_label(example):
    example["labels"] = 1 if example["label"] == "poisoned" else 0
    return example


dataset = dataset.map(convert_label)


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["content"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


dataset = dataset.map(
    tokenize,
    batched=True
)


# --------------------------------------------------
# REMOVE UNNECESSARY COLUMNS
# --------------------------------------------------

dataset = dataset.remove_columns([
    "id",
    "content",
    "label",
    "paired_id",
    "attack_type",
    "difficulty",
    "domain",
    "source",
    "owasp",
    "mitre"
])


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "clean",
        1: "poisoned"
    },
    label2id={
        "clean": 0,
        "poisoned": 1
    }
)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./continual_stage1_model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\n========================================")
print("TRAINING CONTINUAL LEARNING STAGE 1")
print("========================================")

trainer.train()


# --------------------------------------------------
# SAVE
# --------------------------------------------------

trainer.save_model("continual_stage1_model")
tokenizer.save_pretrained("continual_stage1_model")

print("\nStage 1 model saved to:")
print("continual_stage1_model/")