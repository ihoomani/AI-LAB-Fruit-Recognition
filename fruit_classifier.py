import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 42
SAMPLES = 300
RUNS = 30

# Generate a small synthetic fruit dataset.
# Features: weight (g), color score, roundness score
rng = np.random.default_rng(SEED)

apples = np.column_stack([
    rng.normal(160, 20, SAMPLES // 3),
    rng.normal(0.75, 0.10, SAMPLES // 3),
    rng.normal(0.80, 0.08, SAMPLES // 3)
])

bananas = np.column_stack([
    rng.normal(120, 15, SAMPLES // 3),
    rng.normal(0.65, 0.10, SAMPLES // 3),
    rng.normal(0.25, 0.08, SAMPLES // 3)
])

oranges = np.column_stack([
    rng.normal(180, 20, SAMPLES // 3),
    rng.normal(0.90, 0.08, SAMPLES // 3),
    rng.normal(0.85, 0.07, SAMPLES // 3)
])

X = np.vstack([apples, bananas, oranges])
y = np.array(
    ["apple"] * (SAMPLES // 3)
    + ["banana"] * (SAMPLES // 3)
    + ["orange"] * (SAMPLES // 3)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

def predict_random(X_test, seed):
    rng = np.random.default_rng(seed)
    classes = np.array(["apple", "banana", "orange"])
    return rng.choice(classes, size=len(X_test))

def predict_reflex(X_test):
    predictions = []
    for weight, color, roundness in X_test:
        if roundness < 0.45:
            predictions.append("banana")
        elif color > 0.82:
            predictions.append("orange")
        else:
            predictions.append("apple")
    return np.array(predictions)

# Train a machine-learning model
model = DecisionTreeClassifier(max_depth=4, random_state=SEED)
model.fit(X_train, y_train)

agents = {
    "random": lambda seed: predict_random(X_test, seed),
    "reflex": lambda seed: predict_reflex(X_test),
    "model": lambda seed: model.predict(X_test)
}

# Evaluate agents repeatedly
rows = []

for agent_name, predict in agents.items():
    for run in range(RUNS):
        y_pred = predict(run)
        accuracy = accuracy_score(y_test, y_pred)

        rows.append({
            "agent": agent_name,
            "run": run,
            "accuracy": accuracy
        })

df = pd.DataFrame(rows)

summary = df.groupby("agent")["accuracy"].agg(
    ["mean", "std"]
).round(3)

print("Sample data:")
print(pd.DataFrame(X, columns=[
    "weight_g", "color_score", "roundness_score"
]).head())

print("\nEvaluation results:")
print(summary)

# Plot mean accuracy
summary["mean"].sort_values(ascending=False).plot(
    kind="bar", ylim=(0, 1), ylabel="Accuracy",
    title="Fruit Classification Agent Comparison"
)

plt.tight_layout()
plt.show()
