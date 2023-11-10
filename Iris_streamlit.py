import streamlit as st
import pickle
import numpy as np

# Load the trained K-NN model
model = pickle.load(open('FrancoBraconi_trained_IRIS_classification_model.sav', 'rb'))

# Set app title and add an image
st.title('Iris Flower Species Prediction 🌼')
st.image('irir_flowers_classification.png', use_column_width=True)

# Input fields for sepal and petal measurements
st.header('Adjust Flower Measurements (in cm)')
sepal_length = st.slider('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.slider('Petal Length', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.slider('Petal Width', min_value=0.0, max_value=10.0, value=0.2)

# Make a prediction when the user clicks the 'Predict' button
if st.button('Predict'):
    input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display the predicted species and its probability
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction[0]]  #Explicitly convert to integer

    st.subheader('Prediction:')
    st.write(f'Predicted Species: {predicted_species}')
    st.write('Prediction Probabilities:')
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f'{species_names[i]}: {prob:.2f}')
