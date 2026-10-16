from datasets import load_dataset
from collections import Counter

dataset = load_dataset("npow/memshield-bench")
print(dataset)

print("\n--- ONE VALIDATION EXAMPLE ---")
print(dataset["validation"][0])


# 1. Label distribution

print("\n--- LABEL DISTRIBUTION ---")

train_labels = dataset["train"]["label"]

print(Counter(train_labels))


# 2. Attack type distribution

print("\n--- ATTACK TYPE DISTRIBUTION ---")

attack_types = dataset["train"]["attack_type"]

print(Counter(attack_types))


# 3. Difficulty distribution

print("\n--- DIFFICULTY DISTRIBUTION ---")

difficulties = dataset["train"]["difficulty"]

print(Counter(difficulties))

# 4. Domain distribution

print("\n--- DOMAIN DISTRIBUTION ---")

domains = dataset["train"]["domain"]

print(Counter(domains))