"""Score new listings: python src/predict.py [input.csv] [output.csv]"""
import sys

import joblib
import pandas as pd

from config import FEATURES, MODEL_FILE, OUTPUTS, TEST_NO_LABEL_FILE


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else TEST_NO_LABEL_FILE
    dst = sys.argv[2] if len(sys.argv) > 2 else OUTPUTS / "predictions.csv"
    if not MODEL_FILE.exists():
        sys.exit("No trained model found. Run `python src/train_evaluate.py` first.")
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(src)
    df["predicted_price_crore"] = model.predict(df[FEATURES]).round(2)
    df.to_csv(dst, index=False)
    print(f"Predicted {len(df):,} prices -> {dst}")


if __name__ == "__main__":
    main()
