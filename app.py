from gettext import translation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib  

# Load the trained model and scaler
iso_forest = joblib.load('iso_forest_model.pkl')  
scaler = joblib.load('scaler.pkl') 

def classify_fraudulent_transactions(new_data: pd.DataFrame, model, scaler: StandardScaler) -> pd.DataFrame:
    #check if data is correct
    if 'Amount' not in new_data.columns or 'Time' not in new_data.columns:
        raise ValueError("The dataset must include 'Amount' and 'Time' columns.")
    
    #clean the data
    object_columns=new_data.loc[:,new_data.dtypes=="object"].columns
    for i in object_columns:
        new_data[i] = pd.to_numeric(new_data[i], errors='coerce')
    new_data.dropna(axis=0,inplace=True)


    #scale the data as trained
    new_data[['Amount', 'Time']] = scaler.transform(new_data[['Amount', 'Time']])
    #make predictions
    predictions = model.predict(new_data)
    #filtter fraud transactions
    new_data['Fraudulent'] = [1 if x == -1 else 0 for x in predictions]
    #return the prediction
    return new_data

st.title('Credit Card Fraud Detection')

uploaded_file = st.file_uploader("Upload an Excel file with credit card transactions", type="xlsx")

if uploaded_file is not None:
    new_data = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data.head())
    
    try:
        results = classify_fraudulent_transactions(new_data, iso_forest, scaler)
        st.write("Results with Fraudulent Column:")
        st.write(results)
        
        # Visualizations
        st.subheader("Distribution of Transaction Amounts")
        fig, ax = plt.subplots()
        sns.histplot(results[results['Fraudulent'] == 0]['Amount'], bins=50, kde=True, label='Non-Fraudulent', ax=ax)
        sns.histplot(results[results['Fraudulent'] == 1]['Amount'], bins=50, kde=True, label='Fraudulent', ax=ax)
        ax.set_title('Distribution of Transaction Amounts')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Distribution of Time")
        fig, ax = plt.subplots()
        sns.histplot(results[results['Fraudulent'] == 0]['Time'], bins=50, kde=True, label='Non-Fraudulent', ax=ax)
        sns.histplot(results[results['Fraudulent'] == 1]['Time'], bins=50, kde=True, label='Fraudulent', ax=ax)
        ax.set_title('Distribution of Time')
        ax.legend()
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error: {e}")

