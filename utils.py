import cv2
import numpy as np
import tensorflow as tf
from sympy import sympify, SympifyError

# Load the trained model
try:
    model = tf.keras.models.load_model("model/mnist_model.h5")
except (IOError, ImportError):
    model = None

# Define the character map
char_map = {i: str(i) for i in range(10)}
char_map[10] = '+'
char_map[11] = '-'
char_map[12] = '*'

def preprocess_image(image):
    """
    Preprocesses the image for character segmentation.
    """
    img = np.array(image)
    # Ensure the image is in BGR format for OpenCV
    if len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh, img

def segment_characters(preprocessed_image):
    """
    Segments the characters from the preprocessed image.
    """
    contours, _ = cv2.findContours(
        preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    segmented_characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 10:
            char_image = preprocessed_image[y : y + h, x : x + w]
            segmented_characters.append(char_image)
    return segmented_characters

def predict_character(char_image):
    """
    Predicts the character from a segmented image.
    """
    if model is None:
        return '?'

    # Heuristic for operator detection
    height, width = char_image.shape
    aspect_ratio = width / height
    non_zero_pixels = cv2.countNonZero(char_image)
    total_pixels = width * height
    density = non_zero_pixels / total_pixels

    if aspect_ratio > 1.5 and density < 0.3:
        return '-'
    if aspect_ratio < 0.5 and density < 0.3:
        return '+'
    if density < 0.15:
        return '*'

    # Preprocess for the model
    resized = cv2.resize(char_image, (28, 28))
    normalized = resized.astype('float32') / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28, 1))

    # Predict the character
    prediction = model.predict(reshaped)
    char_index = np.argmax(prediction)

    return char_map.get(char_index, '?')

def solve_equation(equation_str):
    """
    Solves the given mathematical equation string.
    """
    try:
        # Replace 'x' with '*' for multiplication if needed
        equation_str = equation_str.replace('x', '*')
        # Use sympify to convert the string to a sympy expression
        expr = sympify(equation_str)
        # Solve the expression
        solution = expr.evalf()
        return solution
    except (SympifyError, TypeError, ZeroDivisionError) as e:
        return f"Error: {e}"
