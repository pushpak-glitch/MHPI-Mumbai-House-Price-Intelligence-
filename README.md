# MHPI-Mumbai-House-Price-Intelligence-
End-to-end data science project for predicting Mumbai housing prices using Python, data pre-processing, and Random Forest regression.
# MHPI(Mumbai-House-Price-Intelligence)
Advanced Random Forest machine learning model predicting Mumbai property prices (in crores) using comprehensive urban housing dataset with 25,000+ records.Predicts Price_Crore from 10+ features including location (latitude/longitude), Distance_From_Mall_In_Km, distance_From_metro_In_km, house size, bedrooms, bathrooms, parking

# Mumbai-House-Price-Intelligence

This project predicts house prices in Mumbai using Machine Learning models.  
Different regression algorithms are trained and evaluated using **RMSE**, and the best model is used for prediction.

## Models Used
- Random Forest Regressor

## Project Files
- `stratified_shuffle_split.py` → Splits dataset into train and test sets
- `RMSE_cal.py` → Trains models and compares RMSE
- `final_prediction.py` → Trains final model and generates predictions

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## How to Run

Install dependencies:

```
pip install pandas
pip install numpy
pip install scikit-learn
pip install joblib
pip install os
```

## Run scripts:

python stratified_shuffle_split.py
</br>
python RMSE_cal.py
</br>
python final_prediction.py


## Output predictions will be saved in:


output.csv


## Author
Pushpak Kumar Sahu
