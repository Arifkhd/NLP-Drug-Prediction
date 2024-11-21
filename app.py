import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load('NLP.pkl')

# Streamlit UI
st.title('Drug Review Condition Classifier')

st.markdown('### Enter the drug review below to classify the condition:')

# User input
review_text = st.text_area("Patient Review")

if st.button("Classify"):
    if review_text:
        # Make prediction
        prediction = model.predict([review_text])
        predicted_condition = model.classes_[prediction]  # Convert back to original label
        
        st.markdown(f'### Predicted Condition: {predicted_condition}')
    else:
        st.error("Please enter a review!")
