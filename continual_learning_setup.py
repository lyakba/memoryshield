from datasets import load_dataset, Dataset
from collections import Counter

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading MemShield-Bench...")

dataset = load_dataset("npow/memshield-bench")
train_data = dataset["train"]

print(f"Total training examples: {len(train_data)}")


# --------------------------------------------------
# DEFINE ATTACK GROUPS
# --------------------------------------------------

STAGE_1_TYPES = [
    "T5_backdoor_trigger",
    "T1_instruction_override",
    "T9_security_degradation"
]

STAGE_2_TYPES = [
    "T4_factual_manipulation",
    "T6_experience_poisoning",
    "T7_recommendation_poisoning",
    "T8_configuration_poisoning",
    "T10_c2_persistence"
]


# --------------------------------------------------
# BALANCE STAGE 1
# --------------------------------------------------

# Number of examples per attack type.
# T5 has many more examples, so we limit it.

ATTACK_SAMPLES = 15

stage1_rows = []

# Add clean examples
clean_examples = [
    row for row in train_data
    if row["attack_type"] == "none"
]

stage1_rows.extend(clean_examples[:45])


# Add Stage 1 attack examples
for attack_type in STAGE_1_TYPES:

    examples = [
        row for row in train_data
        if row["attack_type"] == attack_type
    ]

    selected = examples[:ATTACK_SAMPLES]

    stage1_rows.extend(selected)


# --------------------------------------------------
# CREATE STAGE 2
# --------------------------------------------------

stage2_rows = []

for attack_type in STAGE_2_TYPES:

    examples = [
        row for row in train_data
        if row["attack_type"] == attack_type
    ]

    stage2_rows.extend(examples)


# --------------------------------------------------
# CONVERT TO DATASETS
# --------------------------------------------------

stage1_dataset = Dataset.from_list(stage1_rows)
stage2_dataset = Dataset.from_list(stage2_rows)


# --------------------------------------------------
# PRINT DISTRIBUTIONS
# --------------------------------------------------

print("\n========================================")
print("STAGE 1 DATASET")
print("========================================")

print(f"Total examples: {len(stage1_dataset)}")

stage1_counts = Counter(stage1_dataset["attack_type"])

for attack_type, count in stage1_counts.items():
    print(f"{attack_type:30} {count}")


print("\n========================================")
print("STAGE 2 DATASET")
print("========================================")

print(f"Total examples: {len(stage2_dataset)}")

stage2_counts = Counter(stage2_dataset["attack_type"])

for attack_type, count in stage2_counts.items():
    print(f"{attack_type:30} {count}")


# --------------------------------------------------
# SAVE
# --------------------------------------------------

stage1_dataset.save_to_disk("continual_stage1")
stage2_dataset.save_to_disk("continual_stage2")

print("\nDatasets saved:")
print("  continual_stage1/")
print("  continual_stage2/")