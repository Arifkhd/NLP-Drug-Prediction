import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load('NL.pkl')

# Define a simple drug recommendation dictionary for different conditions
drug_recommendations = {
    "Depression": ["Fluoxetine", "Sertraline", "Citalopram", "Duloxetine", "Venlafaxine"],
    "Hypertension": ["Lisinopril", "Amlodipine", "Hydrochlorothiazide"],
    "Diabetes": ["Metformin", "Insulin", "Sulfonylureas", "DPP-4 inhibitors", "SGLT2 inhibitors"],
    "High Blood Pressure": ["Losartan", "Propranolol", "Atenolol"],
    "Anxiety": ["Lorazepam", "Alprazolam", "Diazepam"]
}

# Streamlit UI
st.title('Drug Review Condition Classifier')

st.markdown('### Enter the drug review below to classify the condition:')

# User input
review_text = st.text_area("Patient Review")

# Function to validate user input
def validate_input(text):
    text = text.strip()
    if len(text) < 5:
        return "Input is too short. Please enter a meaningful review."
    elif not any(char.isalpha() for char in text):
        return "Input must contain alphabetic characters. Please try again."
    return "valid"

if st.button("Classify"):
    if review_text:
        # Validate input
        validation_result = validate_input(review_text)
        
        if validation_result == "valid":
            # Make prediction
            prediction = model.predict([review_text])[0]  # Directly get the predicted condition
            
            st.markdown(f'### Predicted Condition: {prediction}')
            
            # Get drug recommendation based on predicted condition
            recommended_drugs = drug_recommendations.get(prediction, ["No recommendations available for this condition."])
            
            st.markdown(f'### Recommended Drugs: {", ".join(recommended_drugs)}')
        else:
            st.error(validation_result)  # Show validation error
    else:
        st.error("Please enter a review!")
