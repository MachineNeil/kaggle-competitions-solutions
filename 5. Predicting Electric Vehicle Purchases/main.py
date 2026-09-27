import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

data_folder = os.path.join(os.path.dirname(__file__), "data")
train = pd.read_csv(os.path.join(data_folder, "train.csv"))
test = pd.read_csv(os.path.join(data_folder, "test.csv"))

y = train["Will_Buy_EV"]
features = [c for c in train.columns if c not in ["id", "Will_Buy_EV"]]

combined = pd.concat([train[features], test[features]], keys=["train", "test"])
dummies = pd.get_dummies(combined)

x_train, x_val, y_train, y_val = train_test_split(dummies.loc["train"], y, test_size=0.15, random_state=1, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=1)
model.fit(x_train, y_train)
val_proba = model.predict_proba(x_val)[:, 1]
print(f"Validation ROC AUC: {roc_auc_score(y_val, val_proba):.4f}")

predictions = model.predict_proba(dummies.loc["test"])[:, 1]
output = pd.DataFrame({"id": test["id"], "Will_Buy_EV": predictions})
output_path = os.path.join(os.path.dirname(__file__), "submissions.csv")
output.to_csv(output_path, index=False)