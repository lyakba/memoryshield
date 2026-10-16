from datasets import load_dataset
from collections import defaultdict

dataset = load_dataset("npow/memshield-bench")

examples = defaultdict(list)

for item in dataset["train"]:
    attack = item["attack_type"]

    if attack != "none" and len(examples[attack]) < 2:
        examples[attack].append(item)


for attack, items in examples.items():
    print("\n" + "=" * 60)
    print(attack)
    print("=" * 60)

    for item in items:
        print("\nCONTENT:")
        print(item["content"])

        print("\nDIFFICULTY:", item["difficulty"])
        print("DOMAIN:", item["domain"])
        print("LABEL:", item["label"])


import matplotlib.pyplot as plt
from collections import Counter

# Count clean vs poisoned samples
train_labels = Counter(dataset["train"]["label"])
validation_labels = Counter(dataset["validation"]["label"])
test_labels = Counter(dataset["test"]["label"])

labels = ["clean", "poisoned"]

x = range(len(labels))
width = 0.25

plt.figure(figsize=(8, 6))

plt.bar(
    [i - width for i in x],
    [train_labels[label] for label in labels],
    width,
    label="Train"
)

plt.bar(
    x,
    [validation_labels[label] for label in labels],
    width,
    label="Validation"
)

plt.bar(
    [i + width for i in x],
    [test_labels[label] for label in labels],
    width,
    label="Test"
)

plt.xticks(list(x), labels)

plt.xlabel("Label")
plt.ylabel("Number of Examples")
plt.title("Clean vs Poisoned Samples Across Dataset Splits")

plt.legend()
plt.tight_layout()

plt.savefig("clean_vs_poisoned.png", dpi=300)

plt.show()

from collections import defaultdict

# Store up to 2 examples for each attack type
examples = defaultdict(list)

for item in dataset["train"]:
    attack = item["attack_type"]

    # Skip clean examples
    if attack == "none":
        continue

    # Keep only 2 examples per attack
    if len(examples[attack]) < 2:
        examples[attack].append(item)


# Print the examples
for attack, items in examples.items():

    print("\n" + "=" * 80)
    print(attack)
    print("=" * 80)

    for i, item in enumerate(items, 1):

        print(f"\n--- Example {i} ---")

        print("\nCONTENT:")
        print(item["content"])

        print("\nDifficulty:", item["difficulty"])
        print("Domain:", item["domain"])
        print("Label:", item["label"])

    print()

    