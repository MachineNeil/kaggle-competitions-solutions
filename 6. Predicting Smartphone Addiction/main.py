import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def analyze_data(train, test):
    print(f"\n>>> Training shape: {train.shape}; testing shape: {test.shape}.\n")
    print(train["addicted_label"].value_counts(normalize=True))
    print("\n>>> Missing training data (%):")
    print((train.isna().sum() / len(train) * 100).round(2))
    print("\n>>> Missing testing data (%):")
    print((test.isna().sum() / len(test) * 100).round(2))
    print("\n>>> Dtypes:")
    print(train.dtypes)
    print("\n>>> Categorical values:")
    for col in train.select_dtypes(include="object").columns:
        print(col, train[col].unique())
    print("\n")

def map_features(dataframe):
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    dataframe["stress_level"] = dataframe["stress_level"].map(stress_map)
    return dataframe

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    train = map_features(pd.read_csv(os.path.join(data_folder, "train.csv")))
    test = map_features(pd.read_csv(os.path.join(data_folder, "test.csv")))
    analyze_data(train, test)

    features = [c for c in train.columns if c not in ["id", "addicted_label"]]
    combined = pd.concat([train[features], test[features]], keys=["train", "test"])
    numeric = combined.select_dtypes(include=["float64"]).columns
    combined[numeric] = combined[numeric].fillna(train[numeric].median())
    dummies = pd.get_dummies(combined, dummy_na=True, columns=["gender", "academic_work_impact"])

    y = train["addicted_label"]
    x_train, x_val, y_train, y_val = train_test_split(dummies.loc["train"], y, test_size=0.15, random_state=1, stratify=y)

    model = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=1)
    model.fit(x_train, y_train)

    val_preds = model.predict_proba(x_val)[:, 1]
    print("AUC baseline:", roc_auc_score(y_val, val_preds))

    predictions = model.predict_proba(dummies.loc["test"])[:, 1]
    output = pd.DataFrame({"id": test["id"], "addicted_label": predictions})
    output_path = os.path.join(os.path.dirname(__file__), "submissions.csv")
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()