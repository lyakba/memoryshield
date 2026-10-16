from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

# Load dataset
dataset = load_dataset("npow/memshield-bench")

# Training data
X_train = dataset["train"]["content"]
y_train = dataset["train"]["label"]

# Test data
X_test = dataset["test"]["content"]
y_test = dataset["test"]["label"]

# Keep attack types from the test set
attack_types = dataset["test"]["attack_type"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# Make predictions
predictions = model.predict(X_test_tfidf)

# Normal evaluation
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# Analyze poisoned examples by attack type
# --------------------------------------------------

stats = defaultdict(lambda: {"total": 0, "correct": 0, "missed": 0})

for attack, true_label, prediction in zip(
    attack_types,
    y_test,
    predictions
):

    # We only care about poisoned examples
    if true_label == "poisoned":

        stats[attack]["total"] += 1

        if prediction == "poisoned":
            stats[attack]["correct"] += 1
        else:
            stats[attack]["missed"] += 1


print("\nPoisoned Detection by Attack Type:")
print("-" * 70)

print(
    f"{'Attack Type':35}"
    f"{'Total':>8}"
    f"{'Correct':>10}"
    f"{'Missed':>9}"
)

print("-" * 70)

for attack, values in stats.items():

    print(
        f"{attack:35}"
        f"{values['total']:>8}"
        f"{values['correct']:>10}"
        f"{values['missed']:>9}"
    )