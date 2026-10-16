from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading continual learning datasets...")

stage1 = load_from_disk("continual_stage1")
stage2 = load_from_disk("continual_stage2")

print(f"Stage 1 examples: {len(stage1)}")
print(f"Stage 2 examples: {len(stage2)}")


# --------------------------------------------------
# REPLAY SAMPLE
# --------------------------------------------------

# Use 60 Stage 1 examples as replay memory.
# A fixed seed makes the experiment reproducible.

replay_size = min(60, len(stage1))

stage1_replay = stage1.shuffle(seed=42).select(
    range(replay_size)
)

print(f"Replay examples: {len(stage1_replay)}")


# --------------------------------------------------
# COMBINE DATA
# --------------------------------------------------

dataset = concatenate_datasets([
    stage2,
    stage1_replay
])

print(f"Combined training examples: {len(dataset)}")


# --------------------------------------------------
# LABELS
# --------------------------------------------------

def convert_label(example):

    example["labels"] = (
        1 if example["label"] == "poisoned"
        else 0
    )

    return example


dataset = dataset.map(convert_label)


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

MODEL_PATH = "./continual_stage1_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


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
# REMOVE UNUSED COLUMNS
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
    output_dir="./continual_replay_model",
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
print(" CONTINUAL LEARNING WITH REPLAY")
print("========================================")

trainer.train()


# --------------------------------------------------
# SAVE
# --------------------------------------------------

trainer.save_model(
    "continual_replay_model"
)

tokenizer.save_pretrained(
    "continual_replay_model"
)

print("\nReplay model saved to:")
print("continual_replay_model/")