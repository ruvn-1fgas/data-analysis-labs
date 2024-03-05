from PIL import Image
import pandas as pd


INPUT_PATH = "input/digit-recognizer/"
OUTPUT_PATH = "images/"
INPUT_SIZE = 784

train_data = pd.read_csv(f"{INPUT_PATH}train.csv")
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

for i in range(len(x_train)):
    img = Image.new("L", (28, 28))
    pixels = img.load()
    for j in range(28):
        for k in range(28):
            pixels[j, k] = int(x_train[i][j * 28 + k])
    img.save(f"{OUTPUT_PATH}{y_train[i]}_{i}.png")
    print(f"Saved {OUTPUT_PATH}{y_train[i]}_{i}.png")

print("Done")
