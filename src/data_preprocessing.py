import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(128, 128)):
    X = []
    y = []
    classes = {"real": 0, "fake": 1}

    # Iterate through each class (real/fake)
    for label, value in classes.items():
        folder_path = os.path.join(data_dir, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(value)
    return np.array(X), np.array(y)

def preprocess_data(data_dir, img_size=(128, 128), test_size=0.2):
    # Separate the data into train and test directories
    X_train, y_train = load_data(os.path.join(data_dir, 'train'), img_size)
    X_test, y_test = load_data(os.path.join(data_dir, 'test'), img_size)

    return (X_train, y_train), (X_test, y_test)
