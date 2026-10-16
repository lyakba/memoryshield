from datasets import load_dataset
from collections import Counter

dataset = load_dataset("npow/memshield-bench")

splits = ["train", "validation", "test"]

print("\n--- ATTACK TYPES BY SPLIT ---")

for split in splits:
    attacks = Counter(dataset[split]["attack_type"])

    print(f"\n{split.upper()} ({len(dataset[split])} examples)")

    for attack, count in attacks.items():
        print(f"{attack}: {count}")