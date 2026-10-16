from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# --------------------------------------------------
# LOAD STAGE 2 DATA
# --------------------------------------------------

print("Loading Stage 2 dataset...")

dataset = load_from_disk("continual_stage2")

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

MODEL_PATH = "./continual_stage1_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


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
# LOAD STAGE 1 MODEL
# --------------------------------------------------

print("\nLoading Stage 1 model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./continual_stage2_model",
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
    train_dataset=dataset
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\n========================================")
print("TRAINING CONTINUAL LEARNING STAGE 2")
print("========================================")

trainer.train()


# --------------------------------------------------
# SAVE
# --------------------------------------------------

trainer.save_model("continual_stage2_model")
tokenizer.save_pretrained("continual_stage2_model")

print("\nStage 2 model saved to:")
print("continual_stage2_model/")
