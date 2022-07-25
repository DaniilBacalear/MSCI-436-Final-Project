from module import evaluate_csv
from sklearn import metrics
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


def add_spaces(n):
    for _ in range(n):
        st.write("")


def get_prediction(data_set) -> pd.DataFrame:
    df = pd.read_csv(data_set)
    prediction = evaluate_csv(df)
    return prediction


def display_prediction(df):
    highlight_fraud = lambda row: ['background: red' if row.predictedIsFraud == 1 else '' for col in row]
    if "isFraud" in df.columns:
        st.dataframe(df[["amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest",
                         "newbalanceDest", "isFraud", "predictedIsFraud"]].style.apply(highlight_fraud, axis=1))
    else:
        st.dataframe(df[["amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest",
                         "newbalanceDest", "predictedIsFraud"]].style.apply(highlight_fraud, axis=1))


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


def display_confusion_matrix(df: pd.DataFrame):
    confusion_matrix = metrics.confusion_matrix(df["isFraud"], df["predictedIsFraud"])
    confusion_matrix_output = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                             display_labels=[False, True])
    confusion_matrix_output.plot()
    st.pyplot()


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
    if "isFraud" in prediction.columns:
        add_spaces(3)
        st.title("Confusion Matrix")
        display_confusion_matrix(prediction)
