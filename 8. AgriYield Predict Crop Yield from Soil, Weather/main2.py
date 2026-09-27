import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    train = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test = pd.read_csv(os.path.join(data_folder, "test.csv"))

    x = train.drop(["field_id", "yield"], axis=1)
    x_test = test.drop(["field_id"], axis=1)
    y = (train["yield"] - np.mean(train["yield"])) / np.std(train["yield"])

    scale = StandardScaler()

    x = scale.fit_transform(x)
    x_test = scale.transform(x_test)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

    model = SVR(kernel="rbf")
    model.fit(x_train, y_train)
    np.random.seed(1)

    val_preds = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()