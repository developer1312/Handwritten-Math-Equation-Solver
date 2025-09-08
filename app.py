import streamlit as st
from PIL import Image
import utils
import numpy as np

st.set_page_config(page_title="Handwritten Math Equation Solver", page_icon="🧮", layout="wide")

st.title("Handwritten Math Equation Solver")

st.sidebar.title("About")
st.sidebar.info(
    "This is a web application that can solve handwritten math equations."
)

uploaded_file = st.file_uploader("Upload an image of a handwritten math equation", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("")
    st.write("Solving the equation...")

    # Preprocess and segment the image
    preprocessed_image, _ = utils.preprocess_image(image)
    segmented_characters = utils.segment_characters(preprocessed_image)

    # Recognize characters and reconstruct the equation
    equation = ""
    for char_image in segmented_characters:
        char = utils.predict_character(char_image)
        equation += char

    st.write(f"Recognized Equation: {equation}")

    # Solve the equation
    solution = utils.solve_equation(equation)
    st.write(f"Solution: {solution}")
