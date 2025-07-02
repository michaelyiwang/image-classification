# Import required libraries
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model, Model
from keras.preprocessing import image
from pathlib import Path

def preprocessed_image(image_path: str, mean: float, std: float) -> np.ndarray:
    '''
    Preprocesses an image for prediction.
    Args:
        image_path (str): Path to the image file.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    '''
    resized_img = image.load_img(image_path, target_size=(32, 32))
    resized_img_array = image.img_to_array(resized_img)
    normalized_img = (resized_img_array - mean) / (std + 1e-7)
    normalized_img = normalized_img.reshape((1, 32, 32, 3))

    return normalized_img

def predict_image(model: Model, image: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
    '''    
    Predicts the class of a given image using the trained model.
    Args:
        model (keras.Model): Trained Keras model.
        image (np.ndarray): Preprocessed image.
        class_names (list): List of class names.
    Returns:
        str: Predicted class name.
        float: Confidence of the prediction.
    '''
    prediction = model.predict(image)
    predicted_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_idx]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    (X_train, _), (_, _) = cifar10.load_data()
    mean = np.mean(X_train)
    std = np.std(X_train)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    model = load_model("my_model.keras")
    image_folder = Path("images")
    for image_path in image_folder.iterdir():
        normalized_img = preprocessed_image(image_path, mean=mean, std=std)
        predicted_class, confidence = predict_image(model=model, image=normalized_img, class_names=class_names)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f} %")
        original_image = image.load_img(image_path)
        plt.figure(figsize=(4, 4))
        plt.imshow(original_image)
        plt.xticks([])
        plt.yticks([])
        plt.show()