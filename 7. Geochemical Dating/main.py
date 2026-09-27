import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV

def analyze_data(train, test):
    print("\n----------------------------------------------------------------\n")
    print(f">>> TRAIN shape:\t{train.shape}\n>>> TEST shape:\t{test.shape}")

    print(f"\nNormalized label value proportions:\n{train['Label'].value_counts(normalize=True)}")

    missing_in_test = set(train.columns) - set(test.columns)
    if missing_in_test:
        print(f"\nColumns in TRAIN missing from TEST:\t{', '.join(missing_in_test)}.\n")
    frame = pd.DataFrame({
        "Dtypes": train.dtypes,
        "N/A TRAIN (%)": (train.isna().sum() / len(train) * 100).round(2),
        "N/A TEST (%)": (test.isna().sum() / len(test) * 100).round(2)
    }).sort_values("N/A TRAIN (%)", ascending=False)
    print(frame)

    columns = train.select_dtypes(include=["object", "category"]).columns
    if not columns.empty:
        print("\nCategorical values:")
        for c in columns:
            uniques = train[c].unique()
            if len(uniques) > 20:
                print(f"{c}: {len(uniques)} unique values (e.g.: {uniques[:5]}).")
            else:
                print(c, uniques)
    
    print("\n----------------------------------------------------------------\n")

def tune_random_forest(x, y, cv):
    param_dist = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5, None],
    }
    base_model = RandomForestClassifier(class_weight="balanced", random_state=1, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=1,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x, y)
    print(f"\nBest parameters: {search.best_params_}")
    print(search.best_estimator_)

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    train = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test = pd.read_csv(os.path.join(data_folder, "test.csv"))
    # analyze_data(train, test)
    
    features = [c for c in train.columns if c not in ["Label", "Id"]]
    x = train[features]
    y = train["Label"]
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    tune_random_forest(x, y, cv)

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        max_depth=20 ,
        class_weight="balanced", 
        random_state=1, 
        n_jobs=-1
    )
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    
    print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
 
    model.fit(x, y)
    predictions = model.predict(test[features])
 
    output = pd.DataFrame({"Label": predictions, "Id": test["Id"]})
    output_path = os.path.join(os.path.dirname(__file__), "submissions.csv")
    output.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    main()