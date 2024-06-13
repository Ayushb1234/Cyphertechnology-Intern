import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('Project_model.pkl')

# Define the species mapping
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

st.title("Iris Species Classification")

# Input fields for the iris measurements
st.header("Enter Iris Measurements:")
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Button to classify
if st.button('Classify'):
    # Make prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display the results
    st.subheader("Prediction:")
    st.write(f"The predicted species is: {species_mapping[prediction[0]]}")

    st.subheader("Prediction Probability:")
    for species, proba in zip(species_mapping.values(), prediction_proba[0]):
        st.write(f"{species}: {proba:.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸", layout="centered")
    st.run()
