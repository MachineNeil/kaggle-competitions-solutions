import os
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score

def analyze_data(label, train, test):
    print("\n----------------------------------------------------------------\n")
    print(f">>> TRAIN shape:\t{train.shape}\n>>> TEST shape:\t{test.shape}")

    if train[label].dtype in ["object", "category"]:
        label_counts = train['yield'].value_counts(normalize=True)
        number_of_labels = len(label_counts)
        if number_of_labels > 20:
            print(f"\n>>> {number_of_labels} unique labels (e.g. {", ".join(label_counts.unique()[:5])})")
        else:
            print(f"\n>>> Normalized label value proportions:\n{label_counts}")
    
    missing_in_test = set(train.columns) - set(test.columns)
    if missing_in_test:
        print(f"\n>>> Columns in TRAIN missing from TEST:\t{', '.join(missing_in_test)}.\n")
    frame = pd.DataFrame({
        "Dtypes": train.dtypes,
        "N/A TRAIN (%)": (train.isna().sum() / len(train) * 100).round(2),
        "N/A TEST (%)": (test.isna().sum() / len(test) * 100).round(2)
    }).sort_values("N/A TRAIN (%)", ascending=False)
    print(frame)

    columns = train.select_dtypes(include=["object", "category"]).columns
    if not columns.empty:
        print("\n>>> Categorical values:")
        for c in columns:
            uniques = train[c].unique()
            number_of_uniques = len(uniques)
            if number_of_uniques > 20:
                print(f"{c}: {number_of_uniques} unique values (e.g.: {", ".join(uniques[:5])})")
            else:
                print(c, uniques)
    
    print(f"\n>>> TRAIN sample:\n{train.sample()}")
    
    print("\n----------------------------------------------------------------")

def tuned_random_forest(x, y):
    parameters = {
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_leaf": randint(1, 12),
        "max_features": uniform(0.2, 0.8),
        "max_samples": uniform(0.5, 0.5)
    }

    base_model = RandomForestRegressor(
        n_estimators=300,
        random_state=1,
        n_jobs=1
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=parameters,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=1,
        n_jobs=-1
    )

    search.fit(x, y)
    print(f"\n>>> Best parameters:\n{"\n".join(f"{p[0]}: {p[1]}" for p in search.best_params_.items())}")
    
    return search.best_estimator_

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    train = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test = pd.read_csv(os.path.join(data_folder, "test.csv"))

    label_column = "yield"
    analyze_data(label_column, train, test)

    features = [c for c in train.columns if c not in ["field_id", label_column]]
    x = train[features]
    y = train[label_column]

    model = tuned_random_forest(x, y)

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_scores = -cross_val_score(
        estimator=model,
        X=x,
        y=y,
        cv=cv,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )

    print(f"\n>>> RMSE cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
 
    model.fit(x, y)
    predictions = model.predict(test[features])

    output = pd.DataFrame({"field_id": test["field_id"], "yield": predictions})
    output_path = os.path.join(os.path.dirname(__file__), "submissions.csv")
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()