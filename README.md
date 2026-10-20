# NYC-Taxi-Fare-Prediction
Machine learning project comparing Linear Regression and Random Forest for NYC taxi fare prediction.
# NYC Taxi Fare Prediction

## Project Overview

This project uses machine learning to predict New York City yellow taxi fares from historical trip data.

## Dataset

Source: NYC Taxi & Limousine Commission

Dataset: January 2025 Yellow Taxi Trip Records

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Visual Studio Code
- Microsoft Excel
- GitHub

## Machine Learning Models

Two regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor

## Model Performance

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE | $2.37 | $1.71 |
| RMSE | $4.14 | $3.24 |
| R² | 0.935 | 0.960 |

Random Forest achieved the lowest mean absolute error on the test sample.

## Project Outputs

- `nyc_taxi_project.py` — Machine learning pipeline
- `taxi_predictions.csv` — Actual and predicted fares
- `model_metrics.csv` — Model evaluation results
- `feature_importance.csv` — Random Forest feature importance
- `NYC_Taxi_Model_Report.xlsx` — Excel analysis report

## Conclusion

Random Forest outperformed Linear Regression on the selected test sample. Future improvements could include time-based validation, additional feature engineering, and more detailed error analysis.
