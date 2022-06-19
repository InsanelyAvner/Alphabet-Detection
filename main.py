import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from PIL import Image
import PIL.ImageOps

X = np.load("image.npz")['arr_0']
y = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3500, test_size=500)

# Scale data
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# Create classifier
clf = LogisticRegression(solver = "saga", multi_class="multinomial").fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# PIL format image
im_pil = Image.open(r"alphabat.jpg")
im_bw = im_pil.convert("L")

im_bw_resize = im_bw.resize((22, 30), Image.ANTIALIAS)
px_filter = 20
min_px = np.percentile(im_bw_resize, px_filter)

# Scale image
im_bw_resize_scaled = np.clip(im_bw_resize - min_px, 0, 255)
max_px = np.max(im_bw_resize)

# Convert data into an array
im_bw_resize_scaled = np.asarray(im_bw_resize_scaled) / max_px

test_sample = np.array(im_bw_resize_scaled).reshape(1, 660)
test_pred = clf.predict(test_sample)

print(f"The predicted class is: {test_pred[1]}")