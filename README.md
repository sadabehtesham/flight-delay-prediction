# Flight Delay Prediction

This project trains a Decision Tree classifier to predict whether a flight will be delayed by more than 15 minutes using the U.S. flight delay dataset.

The dataset used for the full model is the public flight delay dataset stored in `flights.csv`. It comes from the U.S. DOT / airline flight operations data commonly used for delay prediction experiments. The full file is intentionally not stored in this repository because it is too large for GitHub. This repo includes a smaller sample file, `sample_flights.csv`, for quick local testing.

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

## How to Run the Project

### Option 1: Use the full dataset
```bash
python flight_delay.py
```

### Option 2: Use the included sample data
```bash
python generate_sample_dataset.py
python flight_delay.py
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
