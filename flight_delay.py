from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "flights.csv"
SAMPLE_DATASET_PATH = BASE_DIR / "sample_flights.csv"

# Load Dataset
if DATASET_PATH.exists():
    flights = pd.read_csv(DATASET_PATH, low_memory=False)
else:
    flights = pd.read_csv(SAMPLE_DATASET_PATH, low_memory=False)
    print("Using sample_flights.csv because the full flights.csv dataset is not included in this repository.")

# Taking first 100000 rows
flights_needed_data = flights[0:100000]

# Data Information
print(flights_needed_data.info())

# Visualization
sns.jointplot(
    data=flights_needed_data,
    x="SCHEDULED_ARRIVAL",
    y="ARRIVAL_TIME"
)
plt.show()

# Correlation Heatmap
corr = flights_needed_data.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm")
plt.show()

# Drop unnecessary columns
flights_needed_data = flights_needed_data.drop(
    [
        'YEAR',
        'FLIGHT_NUMBER',
        'AIRLINE',
        'DISTANCE',
        'TAIL_NUMBER',
        'TAXI_OUT',
        'SCHEDULED_TIME',
        'DEPARTURE_TIME',
        'WHEELS_OFF',
        'ELAPSED_TIME',
        'AIR_TIME',
        'WHEELS_ON',
        'DAY_OF_WEEK',
        'TAXI_IN',
        'CANCELLATION_REASON'
    ],
    axis=1
)

# Fill Missing Values
flights_needed_data = flights_needed_data.fillna(
    flights_needed_data.mean(numeric_only=True)
)

# Create Target Variable
result = []

for row in flights_needed_data['ARRIVAL_DELAY']:
    if row > 15:
        result.append(1)
    else:
        result.append(0)

flights_needed_data['result'] = result

# Remove columns causing leakage
flights_needed_data = flights_needed_data.drop(
    [
        'ORIGIN_AIRPORT',
        'DESTINATION_AIRPORT',
        'ARRIVAL_TIME',
        'ARRIVAL_DELAY'
    ],
    axis=1
)

# Prepare Data
data = flights_needed_data.values

X = data[:, :-1]
y = data[:, -1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ROC-AUC Score
pred_prob = clf.predict_proba(X_test)

auc_score = roc_auc_score(y_test, pred_prob[:, 1])

print("ROC-AUC Score:", auc_score)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))