from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "./continual_stage1_model"

print("Loading Stage 1 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Stage 1 model loaded.\n")


# --------------------------------------------------
# LOAD OFFICIAL TEST SET
# --------------------------------------------------

print("Loading MemShield-Bench test set...")

dataset = load_dataset("npow/memshield-bench")
test_data = dataset["test"]

print(f"Test examples: {len(test_data)}\n")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_true = []
y_pred = []

attack_types = []
predictions_by_type = []


print("Evaluating...\n")

for i, example in enumerate(test_data):

    text = example["content"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    true_label = 1 if example["label"] == "poisoned" else 0

    y_true.append(true_label)
    y_pred.append(prediction)

    attack_types.append(example["attack_type"])

    predictions_by_type.append({
        "attack_type": example["attack_type"],
        "true": true_label,
        "predicted": prediction
    })

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(test_data)}")


# --------------------------------------------------
# OVERALL METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n========================================")
print("   CONTINUAL LEARNING - STAGE 1")
print("========================================")

print("\n--- Overall Performance ---")

print(f"Accuracy:        {accuracy:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Poisoned Recall: {recall:.4f}")
print(f"Poisoned F1:     {f1:.4f}")


print("\n--- Confusion Matrix ---")

print("                 Predicted")
print("                 Clean  Poisoned")

print(
    f"Actual Clean     {cm[0][0]:5d}  {cm[0][1]:8d}"
)

print(
    f"Actual Poisoned  {cm[1][0]:5d}  {cm[1][1]:8d}"
)


# --------------------------------------------------
# OLD ATTACK FAMILIES
# --------------------------------------------------

OLD_ATTACKS = {
    "T5_backdoor_trigger",
    "T1_instruction_override",
    "T9_security_degradation"
}

old_results = [
    r for r in predictions_by_type
    if r["attack_type"] in OLD_ATTACKS
]

old_true = [r["true"] for r in old_results]
old_pred = [r["predicted"] for r in old_results]

old_recall = recall_score(
    old_true,
    old_pred,
    zero_division=0
)

old_f1 = f1_score(
    old_true,
    old_pred,
    zero_division=0
)

print("\n--- OLD ATTACK FAMILIES ---")

print(f"Examples: {len(old_results)}")
print(f"Recall:   {old_recall:.4f}")
print(f"F1:       {old_f1:.4f}")


# --------------------------------------------------
# PER-ATTACK PERFORMANCE
# --------------------------------------------------

print("\n--- PER-ATTACK PERFORMANCE ---")

for attack in sorted(set(attack_types)):

    attack_results = [
        r for r in predictions_by_type
        if r["attack_type"] == attack
    ]

    true = [r["true"] for r in attack_results]
    pred = [r["predicted"] for r in attack_results]

    # Clean category does not have poisoned recall
    if attack == "none":
        attack_accuracy = accuracy_score(true, pred)

        print(
            f"{attack:30s} "
            f"n={len(true):3d} "
            f"accuracy={attack_accuracy:.4f}"
        )

    else:
        attack_recall = recall_score(
            true,
            pred,
            zero_division=0
        )

        print(
            f"{attack:30s} "
            f"n={len(true):3d} "
            f"recall={attack_recall:.4f}"
        )


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n--- Classification Report ---")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["clean", "poisoned"],
        zero_division=0
    )
)