import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")

    train_data = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_folder, "test.csv"))

    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]

    combined = pd.concat([train_data[features], test_data[features]], keys=["train", "test"])
    dummies = pd.get_dummies(combined)

    x_train, x_val, y_train, y_val = train_test_split(dummies.loc["train"], y, test_size=0.15, random_state=1, stratify=y)

    model = RandomForestClassifier(100, max_depth=5, random_state=1)
    model.fit(x_train, y_train)
    print(f"Validation accuracy: {model.score(x_val, y_val):.4f}")

    predictions = model.predict(dummies.loc["test"])
    output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
    output_path = os.path.join(os.path.dirname(__file__), "submission.csv")
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()