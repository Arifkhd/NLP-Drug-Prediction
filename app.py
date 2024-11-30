import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load('NL.pkl')

# Streamlit UI
st.title('Drug Review Condition Classifier')

st.markdown('### Enter the drug review below to classify the condition:')

# User input
review_text = st.text_area("Patient Review")

if st.button("Classify"):
    if review_text:
        # Make prediction
        prediction = model.predict([review_text])[0]  # Directly get the predicted condition
        
        st.markdown(f'### Predicted Condition: {prediction}')
    else:
        st.error("Please enter a review!")
