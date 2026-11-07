"""Central paths and settings so every script runs from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "mumbai_house_raw_data.csv"
PROCESSED = ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED / "train.csv"
TEST_FILE = PROCESSED / "test.csv"
TEST_NO_LABEL_FILE = PROCESSED / "test_without_label.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "price_model.joblib"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "images"

TARGET = "Price_Crore"
FEATURES = ["Longitude", "Latitude", "House_Age", "Total_Rooms", "Total_Bedrooms",
            "Bathrooms", "Parking", "Dist_Mall_KM", "Dist_Metro_KM"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
