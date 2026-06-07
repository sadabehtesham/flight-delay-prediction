# Flight Delay Prediction

## Overview
This project is a machine-learning solution for predicting whether a flight will be delayed by more than 15 minutes using flight operations data and a Decision Tree classifier.

## Dataset
The full model uses the public flight delay dataset stored in `flights.csv`. It comes from U.S. DOT / airline flight operations data commonly used for delay prediction experiments. The full file is intentionally not stored in this repository because it is too large for GitHub. This repo includes a smaller sample file, `sample_flights.csv`, for quick local testing.

## Data Preprocessing
- Load the dataset from `flights.csv` or fall back to `sample_flights.csv`
- Drop non-essential columns and leakage-prone fields
- Fill missing numeric values with column means
- Create a binary target variable where `ARRIVAL_DELAY > 15` means a delay

## Exploratory Data Analysis
- Joint plot of scheduled arrival vs arrival time
- Correlation heatmap for feature relationships
- Confusion matrix visualization for model evaluation

## Model Training
- Split data into train/test sets
- Standardize features with `StandardScaler`
- Train a `DecisionTreeClassifier`
- Evaluate using accuracy and ROC-AUC

## Results

| Metric | Score |
|----------|----------|
| Accuracy | 99.84% |
| ROC-AUC Score | 99.81% |

## How to Run

### Option 1: Use the full dataset
```bash
python flight_delay.py
```

### Option 2: Use the included sample data
```bash
python generate_sample_dataset.py
python flight_delay.py
```

## Future Improvements
- Add a Flask web app with user input form and prediction page
- Compare multiple models such as Random Forest and XGBoost
- Add feature engineering for airport and airline-specific delay patterns

## Project Structure

```text
flight_delay.py
flight_delay_notebook.ipynb
sample_flights.csv
requirements.txt
README.md
screenshots/
```

## Screenshots
- Joint Plot: [screenshots/joint_plot.png](screenshots/joint_plot.png)
- Heatmap: [screenshots/heatmap.png](screenshots/heatmap.png)
- Confusion Matrix: [screenshots/confusion_matrix.png](screenshots/confusion_matrix.png)
- Terminal Output: [screenshots/terminal_output.png](screenshots/terminal_output.png)

## Features
- Loads flight data from `flights.csv`
- Preprocesses and cleans missing values
- Trains a Decision Tree model
- Prints accuracy and ROC-AUC metrics

## Dataset Used

- Dataset: flight delay / airline operations CSV data
- Source: public U.S. DOT / airline flight delay dataset used for delay prediction
- File name to place in the project root: `flights.csv`

## How to Download the Dataset

1. Download the full dataset from the official public source you choose for this project (for example, the U.S. DOT / Kaggle flight-delay dataset).
2. Save the downloaded file as `flights.csv` in the project root of this repository.
3. If you want to use the sample file instead, run:
   ```bash
   python generate_sample_dataset.py
   ```

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project
   ```bash
   python flight_delay.py
   ```

## Requirements
- Python 3.10+
- pandas
- seaborn
- matplotlib
- scikit-learn
