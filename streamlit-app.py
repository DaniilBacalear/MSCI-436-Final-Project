from module import evaluate_csv
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def get_prediction(data_set):
    dataframe = pd.read_csv(data_set)
    print('loading model')
    prediction = evaluate_csv(dataframe)
    return prediction


def display_prediction(prediction):
    st.dataframe(prediction[["amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest",
                             "newbalanceDest", "isFraud", "predictedIsFraud"]])


def make_pi_chart(prediction):
    fraud_percentage = (prediction['predictedIsFraud'].sum() / len(prediction['predictedIsFraud'])) * 100
    valid_percentage = 100 - fraud_percentage
    labels = 'Fraudulent', 'Valid'
    colors = ['#E52C04', '#1EEC06']
    sizes = [fraud_percentage, valid_percentage]
    explode = (0.1, 0.0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            colors=colors, shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)


uploaded_file = st.file_uploader('upload csv', type=['csv'], accept_multiple_files=False, key="fileUploader")

if uploaded_file is not None:
    prediction = get_prediction(uploaded_file)
    display_prediction(prediction)
    make_pi_chart(prediction)

