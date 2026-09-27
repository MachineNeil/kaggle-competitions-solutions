import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

data_folder = os.path.join(os.path.dirname(__file__), "data")

train = pd.read_csv(os.path.join(data_folder, "train.csv"))
test = pd.read_csv(os.path.join(data_folder, "test.csv"))

y = train.iloc[:, 0].values
x = train.iloc[:, 1:].values.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.15, random_state=1, stratify=y
)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64
)

predictions = np.argmax(model.predict(test.values.reshape(-1, 28, 28, 1)), axis=1)
output = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions })
output_path = os.path.join(os.path.dirname(__file__), "submission.csv")
output.to_csv(output_path, index=False)