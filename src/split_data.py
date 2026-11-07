"""Stratified 80/20 split on price quintiles (so train and test cover every price band)."""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from config import PROCESSED, RANDOM_STATE, RAW_DATA, TARGET, TEST_FILE, TEST_NO_LABEL_FILE, TEST_SIZE, TRAIN_FILE


def main() -> None:
    df = pd.read_csv(RAW_DATA)
    bins = pd.qcut(df[TARGET], q=5, labels=False)
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(df, bins))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    test.drop(columns=TARGET).to_csv(TEST_NO_LABEL_FILE, index=False)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows -> {PROCESSED}")


if __name__ == "__main__":
    main()
