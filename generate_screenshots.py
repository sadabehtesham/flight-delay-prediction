from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

DATASET_PATH = BASE_DIR / "flights.csv"
SAMPLE_DATASET_PATH = BASE_DIR / "sample_flights.csv"

if DATASET_PATH.exists():
    flights = pd.read_csv(DATASET_PATH, low_memory=False)
else:
    flights = pd.read_csv(SAMPLE_DATASET_PATH, low_memory=False)

sample = flights.iloc[:5000].copy()

# Joint plot
sns.jointplot(data=sample, x="SCHEDULED_ARRIVAL", y="ARRIVAL_TIME", kind="reg")
plt.savefig(SCREENSHOT_DIR / "joint_plot.png", dpi=150)
plt.close("all")

# Heatmap
corr = sample.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / "heatmap.png", dpi=150)
plt.close("all")

# Sample training metrics for terminal-style screenshot
prepared = sample.drop(columns=[
    "YEAR", "FLIGHT_NUMBER", "AIRLINE", "DISTANCE", "TAIL_NUMBER", "TAXI_OUT",
    "SCHEDULED_TIME", "DEPARTURE_TIME", "WHEELS_OFF", "ELAPSED_TIME", "AIR_TIME",
    "WHEELS_ON", "DAY_OF_WEEK", "TAXI_IN", "CANCELLATION_REASON"
], errors="ignore")
prepared = prepared.fillna(prepared.mean(numeric_only=True))
prepared["result"] = [1 if value > 15 else 0 for value in prepared["ARRIVAL_DELAY"]]
prepared = prepared.drop(columns=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ARRIVAL_TIME", "ARRIVAL_DELAY"], errors="ignore")

X = prepared.drop(columns=["result"]).to_numpy()
y = prepared["result"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / 'confusion_matrix.png', dpi=150)
plt.close('all')

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")
ax.text(0.02, 0.95, "Terminal Output", fontsize=15, fontweight="bold", va="top")
ax.text(0.02, 0.78, f"Accuracy: {accuracy:.6f}", fontsize=12, va="top", family="monospace")
ax.text(0.02, 0.62, f"ROC-AUC Score: {roc_auc:.6f}", fontsize=12, va="top", family="monospace")
ax.text(0.02, 0.40, "Model: DecisionTreeClassifier", fontsize=12, va="top", family="monospace")
fig.tight_layout()
fig.savefig(SCREENSHOT_DIR / "terminal_output.png", dpi=150)
plt.close(fig)

print("Saved screenshots to", SCREENSHOT_DIR)
print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
