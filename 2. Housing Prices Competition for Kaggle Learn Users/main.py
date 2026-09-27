import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")

    train_data = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_folder, "test.csv"))

    y = train_data["SalePrice"]

    features = [c for c in train_data.columns if c not in ["Id", "SalePrice"]]
    combined = pd.concat([train_data[features], test_data[features]], keys=["train", "test"])

    numeric = combined.select_dtypes(include=["int64", "float64"]).columns
    combined[numeric] = combined[numeric].fillna(combined[numeric].median())

    categorical = combined.select_dtypes(include=["object"]).columns
    combined[categorical] = combined[categorical].fillna("None")

    dummies = pd.get_dummies(combined)

    x_train, x_val, y_train, y_val = train_test_split(dummies.loc["train"], y, test_size=0.2, random_state=1)
    
    model = RandomForestRegressor(100, random_state=1)
    model.fit(x_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(x_val)))
    print(f"Validation RMSE: {rmse:.2f}")

    predictions = model.predict(dummies.loc["test"])
    output = pd.DataFrame({"Id": test_data.Id, "SalePrice": predictions})
    output_path = os.path.join(os.path.dirname(__file__), "submission.csv")
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()