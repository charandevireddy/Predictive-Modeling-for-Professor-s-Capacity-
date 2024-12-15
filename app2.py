import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load CSV data
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Verify column headers
    required_columns = ['Professor', 'Student Name', 'Mid Marks', 'Sem Marks', 'Project Marks', 'Feedback']
    if not all(col in df.columns for col in required_columns):
        st.error("CSV is missing required columns: 'Professor', 'Student Name', 'Mid Marks', 'Sem Marks', 'Project Marks', 'Feedback'")
    else:
        # Aggregate data per professor
        professor_stats = df.groupby('Professor').agg({
            'Mid Marks': 'mean',
            'Sem Marks': 'mean',
            'Project Marks': 'mean',
            'Feedback': 'mean'
        }).reset_index()
        
        # Calculate success rate as the target variable
        professor_stats['Success Rate'] = (
            professor_stats[['Mid Marks', 'Sem Marks', 'Project Marks', 'Feedback']].mean(axis=1)
        )

        # Display data
        st.write("Professor Statistics Data")
        st.write(professor_stats)

        # Features and target
        X = professor_stats[['Mid Marks', 'Sem Marks', 'Project Marks', 'Feedback']]
        y = professor_stats['Success Rate']

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select and train the model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.write("Model Evaluation")
        st.write(f"Root Mean Squared Error: {rmse}")

        # Predict success rate for each professor
        professor_stats['Predicted Success Rate'] = model.predict(X)

        st.write("Professor Success Rate Predictions")
        st.write(professor_stats[['Professor', 'Predicted Success Rate']])
