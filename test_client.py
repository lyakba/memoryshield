import requests
import time
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

URL = "http://127.0.0.1:8000/memory"

print("Loading MemShield-Bench...")
dataset = load_dataset("npow/memshield-bench")
test_data = dataset["test"]

samples = test_data

print(f"Loaded {len(samples)} test memories.\n")

results = []

for i, example in enumerate(samples):

    memory = example["content"]

    start = time.perf_counter()

    response = requests.post(
        URL,
        json={"memory": memory}
    )

    end = time.perf_counter()

    network_time_ms = (end - start) * 1000

    data = response.json()

    results.append({
        "network_time_ms": network_time_ms,
        "server_time_ms": data["total_processing_ms"],
        "inference_ms": data["inference_ms"],
        "prediction": data["prediction"],
        "true_label": example["label"]
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(samples)}")


# --------------------------------------------------
# SECURITY METRICS
# --------------------------------------------------

y_true = [
    1 if r["true_label"] == "poisoned" else 0
    for r in results
]

y_pred = [
    1 if r["prediction"] == "poisoned" else 0
    for r in results
]


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
# LATENCY
# --------------------------------------------------

network_times = [
    r["network_time_ms"]
    for r in results
]

server_times = [
    r["server_time_ms"]
    for r in results
]

inference_times = [
    r["inference_ms"]
    for r in results
]


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n========================================")
print("      MEMORYSHIELD SECURITY TEST")
print("========================================")

print(f"\nRequests: {len(results)}")

print("\n--- Security Performance ---")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Poisoned Recall:   {recall:.4f}")
print(f"Poisoned F1:       {f1:.4f}")

print("\n--- Confusion Matrix ---")

print("                 Predicted")
print("                 Clean  Poisoned")

print(
    f"Actual Clean     {cm[0][0]:5d}  {cm[0][1]:8d}"
)

print(
    f"Actual Poisoned  {cm[1][0]:5d}  {cm[1][1]:8d}"
)

print("\n--- Errors ---")

false_positives = cm[0][1]
false_negatives = cm[1][0]

print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")


print("\n--- Latency ---")

print(f"HTTP Average:      {sum(network_times)/len(network_times):.2f} ms")
print(f"HTTP Median:       {sorted(network_times)[len(network_times)//2]:.2f} ms")

print(f"Server Average:    {sum(server_times)/len(server_times):.2f} ms")
print(f"Server Median:     {sorted(server_times)[len(server_times)//2]:.2f} ms")

print(f"Inference Average: {sum(inference_times)/len(inference_times):.2f} ms")
print(f"Inference Median:  {sorted(inference_times)[len(inference_times)//2]:.2f} ms")

print(
    f"\nThroughput: {len(results) / (sum(network_times) / 1000):.2f} requests/sec"
)


print("\n--- Classification Report ---")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["clean", "poisoned"],
        zero_division=0
    )
)