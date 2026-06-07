# Flight Delay Prediction

This project trains a Decision Tree classifier to predict whether a flight will be delayed by more than 15 minutes using the U.S. flight delay dataset.

The full `flights.csv` dataset is intentionally not stored in this repository because it is too large for GitHub. The repository includes a small sample file, `sample_flights.csv`, for quick local testing.

## Features
- Loads flight data from `flights.csv`
- Preprocesses and cleans missing values
- Trains a Decision Tree model
- Prints accuracy and ROC-AUC metrics

## Dataset Download

1. Download the full flight-delay dataset from the official source (for example, the U.S. DOT / Kaggle flight-delay dataset).
2. Save the downloaded CSV as `flights.csv` in the project root.
3. Re-run the project with:
   ```bash
   python flight_delay.py
   ```

If you want a quick smoke test without the full dataset, use the included sample file:

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
