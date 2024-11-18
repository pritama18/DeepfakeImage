from src.model_training import train_model
from src.predict import make_prediction

def deepfake_image_detection(
    data_dir=None, 
    model_path=None, 
    epochs=10, 
    batch_size=32, 
    predict=None
):
    """
    Function to handle deepfake image detection training or prediction.

    Args:
        data_dir (str): Path to the dataset directory (required for training).
        model_path (str): Path to save or load the trained model.
        epochs (int): Number of epochs for training (default: 10).
        batch_size (int): Batch size for training (default: 32).
        predict (str): Path to an image for prediction (optional).
    """
    if predict:
        if not model_path:
            raise ValueError("Model path must be provided for prediction.")
        make_prediction(model_path, predict)
    else:
        if not data_dir or not model_path:
            raise ValueError("Both data_dir and model_path must be provided for training.")
        train_model(data_dir, model_path, epochs, batch_size)

# Example usage
if __name__ == "__main__":
    # Replace these with appropriate paths and parameters for testing
    deepfake_image_detection(
        data_dir="./data", 
        model_path="./model/model.h5", 
        epochs=1, 
        batch_size=16, 
        predict=None  # Replace with a path to an image for prediction, if needed
    )
