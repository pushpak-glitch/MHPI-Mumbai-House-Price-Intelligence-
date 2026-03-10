
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

df =pd.read_csv("/Users/pushpak/mumbai urban house prediction/mumbai house raw data.csv ")


# creating bins for better suffle
df["price_bin"] = pd.cut(
    df["Price_Crore"],
    bins=[0, 2, 4, 6, 8, np.inf],
    labels=[1, 2, 3, 4, 5]
)


split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in split.split(df, df["price_bin"]):
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]

final_train_data = train_data.drop("price_bin",axis=1)
final_test_data = test_data.drop("price_bin",axis=1)
final_test_data2 = final_test_data.drop("Price_Crore",axis=1)


final_test_data.to_csv("final_test_data.csv",index=False)
final_train_data.to_csv("final_train_data.csv",index =False)
Test_data_without_label = final_test_data2.to_csv("final_test_without_label.csv",index = False)





