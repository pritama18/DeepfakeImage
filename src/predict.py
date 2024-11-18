import cv2
import numpy as np
import tensorflow as tf

def make_prediction(model_path, img_path, img_size=(128, 128)):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read the image.")
        return
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)
    label = "Fake" if prediction[0][0] > 0.5 else "Real"
    
    print(f"Prediction for {img_path}: {label}")
