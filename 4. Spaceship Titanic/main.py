import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_folder = os.path.join(os.path.dirname(__file__), "data")
train = pd.read_csv(os.path.join(data_folder, "train.csv"))
test = pd.read_csv(os.path.join(data_folder, "test.csv"))

features = [c for c in train.columns if c not in ["PassengerId", "Name", "Transported"]]
combined = pd.concat([train[features], test[features]], keys=["train", "test"])

combined[["Deck", "Num", "Side"]] = combined["Cabin"].str.split("/", expand=True)
combined = combined.drop(columns=["Cabin", "Num"])

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
combined[spend_cols] = combined[spend_cols].fillna(0)

combined["Group"] = train["PassengerId"].str.split("_").str[0]
combined["GroupSize"] = combined.groupby("Group")["Group"].transform("count")

numeric = combined.select_dtypes(include=["int64", "float64"]).columns
combined[numeric] = combined[numeric].fillna(combined[numeric].median())

categorical = combined.select_dtypes(include=["object"]).columns
combined[categorical] = combined[categorical].fillna("None")

dummies = pd.get_dummies(combined)

y = train["Transported"]

x_train, x_val, y_train, y_val = train_test_split(dummies.loc["train"], y, test_size=0.15, random_state=1, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=1)
model.fit(x_train, y_train)
print(f"Validation accuracy: {model.score(x_val, y_val):.4f}")

predictions = model.predict(dummies.loc["test"])
output = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": predictions})
output_path = os.path.join(os.path.dirname(__file__), "submissions.csv")
output.to_csv(output_path, index=False)