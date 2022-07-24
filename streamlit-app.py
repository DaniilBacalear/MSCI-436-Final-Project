from module import evaluate_csv
import pandas as pd
# import pyautogui
import streamlit as st
import matplotlib.pyplot as plt


def add_spaces(n):
    for _ in range(n):
        st.write("")


def get_prediction(data_set):
    df = pd.read_csv(data_set)
    prediction = evaluate_csv(df)
    return prediction


def display_prediction(df):
    st.dataframe(df[["amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest",
                             "newbalanceDest", "isFraud", "predictedIsFraud"]])


def display_pi_chart(df):
    print("making pi chart")
    fraud_percentage = (df['predictedIsFraud'].sum() / len(df['predictedIsFraud'])) * 100
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


def display_metrics(df: pd.DataFrame):
    total_transactions, fraud_count, total_fraud_amount = st.columns(3)
    total_transactions.metric(label="Total Transactions Analyzed", value=len(df))
    fraud_count.metric(label="Fraudulent Transactions Detected", value=len(df[df.predictedIsFraud == 1]))
    total_fraud_amount.metric(label="Total Fraud Amount ($)", value=sum(df[df.predictedIsFraud == 1]["amount"]))


uploaded_file = st.file_uploader('upload csv', type=['csv'], accept_multiple_files=False, key="fileUploader")

if uploaded_file is not None:
    prediction = get_prediction(uploaded_file)
    add_spaces(1)
    st.title("Predictions")
    display_prediction(prediction)
    add_spaces(2)
    st.title("Metrics")
    display_metrics(prediction)
    add_spaces(3)
    st.title("Fraudulent vs Valid Transactions")
    display_pi_chart(prediction)
    # if st.button(label="Clear Results"):
    #     pyautogui.hotkey("ctrl", "F5")


