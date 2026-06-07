# Flight Delay Prediction

This project trains a Decision Tree classifier to predict whether a flight will be delayed by more than 15 minutes using the U.S. flight delay dataset.

## Features
- Loads flight data from `flights.csv`
- Preprocesses and cleans missing values
- Trains a Decision Tree model
- Prints accuracy and ROC-AUC metrics

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
