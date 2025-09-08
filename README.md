# MathsSolver

This is a web application that can solve handwritten math equations from an uploaded image. The application is built using Streamlit, Python, OpenCV, TensorFlow, and SymPy.

## Features

-   Upload an image of a handwritten math equation.
-   The application will preprocess the image and segment the characters.
-   A Convolutional Neural Network (CNN) model is used to recognize the digits.
-   The recognized characters are reconstructed into a mathematical expression.
-   The equation is solved using SymPy, and the solution is displayed.

## How to run the application

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment and install the dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    Before running the application, you need to train the CNN model. Run the following command:
    ```bash
    python train_model.py
    ```
    This will train the model and save it in the `model/` directory.

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

5.  **Open your web browser and navigate to the URL provided by Streamlit.**

## Project Structure

-   `app.py`: The main Streamlit application file.
-   `utils.py`: Contains utility functions for image preprocessing, character segmentation, prediction, and equation solving.
-   `model/`: This directory contains the trained CNN model.
-   `pages/`: Contains additional pages for the Streamlit application.
-   `requirements.txt`: A list of the Python dependencies for the project.

## Limitations

-   The current model is trained on the MNIST dataset, which only contains digits (0-9). Therefore, it can only recognize digits. Operators like `+`, `-`, `*`, `/` are detected using simple heuristics, which may not be accurate for all cases.
-   For a more robust solution, the CNN model should be trained on a comprehensive dataset of handwritten math symbols.