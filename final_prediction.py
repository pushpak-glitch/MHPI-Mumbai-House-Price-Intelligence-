import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

if not os.path.exists(MODEL_FILE):
 
    df = pd.read_csv("/Users/pushpak/mumbai urban house prediction/final_train_data.csv")   

    y = df["Price_Crore"]
    x = df.drop(["Price_Crore"], axis=1)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_prepared = pipeline.fit_transform(x)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  
    model.fit(X_prepared, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print(" Model trained and saved ")

else:
    
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("/Users/pushpak/mumbai urban house prediction/final_test_without_label.csv")
    input_prepared = pipeline.transform(input_data)

    predictions = model.predict(input_prepared)
    input_data["prediction"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Prediction done. output.csv saved")