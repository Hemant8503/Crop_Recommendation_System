import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open('model_crop.pkl', 'rb'))
st.title("Crop Recommendation System")
st.write("Enter the soil and environmental parameters to get a crop recommendation:")

#user inputs
N = st.number_input("Nitrogen content in soil: ", min_value=0.0, max_value=100.0, step=0.1)
P = st.number_input("Phosphorous content in soil: ", min_value=0.0, max_value=100.0, step=0.1)
K = st.number_input("Potassium content in soil: ", min_value=0.0, max_value=100.0, step=0.1)
temperature = st.number_input("Temperature (°C): ", min_value=-10, max_value=50, step=1)
humidity = st.number_input("Humidity (%): ", min_value=0, max_value=100, step=1)
pH = st.number_input("Soil pH: ", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm): ", min_value=0, max_value=1000, step=10)


# Define function to make predictions
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    # Create a numpy array with the input values
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    
    # Use the model to make a prediction
    prediction = model.predict(input_values)
    
    # Return the predicted crop label
    return prediction[0]



# Add a button to trigger the recommendation
if st.button("Recommend"):

    # 69	55	38	22.708838	82.639414	5.700806	271.324860
    pred = predict_crop(N, P, K, temperature, humidity, pH, rainfall)
    if pred == 1:
        st.success("Rice is the best crop to be cultivated right there")
    elif pred == 2:
        st.success("Maize is the best crop to be cultivated right there")
    elif pred == 3:
        st.success("Jute is the best crop to be cultivated right there")
    elif pred == 4:
        st.success("Cotton is the best crop to be cultivated right there")
    elif pred == 5:
        st.success("Coconut is the best crop to be cultivated right there")
    elif pred == 6:
        st.success("Papaya is the best crop to be cultivated right there")
    elif pred == 7:
        st.success("Orange is the best crop to be cultivated right there")
    elif pred == 8:
        st.success("Apple is the best crop to be cultivated right there")
    elif pred == 9:
        st.success("Muskmelon is the best crop to be cultivated right there")
    elif pred == 10:
        st.success("Watermelon is the best crop to be cultivated right there")
    elif pred == 11:
        st.success("Grapes is the best crop to be cultivated right there")
    elif pred == 12:
        st.success("Mango is the best crop to be cultivated right there")
    elif pred == 13:
        st.success("Banana is the best crop to be cultivated right there")
    elif pred == 14:
        st.success("Pomegranate is the best crop to be cultivated right there")
    elif pred == 15:
        st.success("Lentil is the best crop to be cultivated right there")
    elif pred == 16:
        st.success("Blackgram is the best crop to be cultivated right there")
    elif pred == 17:
        st.success("Mothbeans is the best crop to be cultivated right there")
    elif pred == 19:
        st.success("Pigeonpeas is the best crop to be cultivated right there")
    elif pred == 20:
        st.success("Kidneybeans is the best crop to be cultivated right there")
    elif pred == 21:
        st.success("Chickpea is the best crop to be cultivated right there")
    elif pred == 22:
        st.success("Coffee is the best crop to be cultivated right there")
    else:
        st.warning("Sorry, we could not determine the best crop to be cultivated with the provided data.")

        

