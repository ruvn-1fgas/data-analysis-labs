import keras
from keras.models import Sequential
from keras import layers

import pandas as pd

# PREPROCESSING

INPUT_PATH = "input/digit-recognizer/"
INPUT_SIZE = 784
OUTPUT_SIZE = 10

train_data = pd.read_csv(f"{INPUT_PATH}train.csv")
test_data = pd.read_csv(f"{INPUT_PATH}test.csv")

y_train = train_data.iloc[:, 0].values
print(f"y_train - {y_train}")

x_train = train_data.iloc[:, 1:].values
print(f"x_train - {x_train}")

x_test = test_data.values

# data normalizing (1 and 0 for each pixel)
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# MODEL CREATION AND TRAINING

model = Sequential(
    layers=[
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(OUTPUT_SIZE, activation="softmax"),
    ]
)

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.3, verbose=2)

# SUBMISSION

predictions = model.predict(x_test)

submissions = pd.DataFrame(
    {
        "ImageId": list(range(1, len(predictions) + 1)),
        "Label": predictions.argmax(axis=1),
    }
)

submissions.to_csv("submission_lab_2.csv", index=False, header=True)
