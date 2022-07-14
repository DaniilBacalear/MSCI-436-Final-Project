import streamlit as st
from module import evaluate_csv
import pandas as pd
uploaded_file = st.file_uploader('upload csv', type=['csv'],accept_multiple_files=False,key="fileUploader")

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    print('loading model')
    df = evaluate_csv(dataframe)
    df = df.drop([''])
    st.dataframe(evaluate_csv(dataframe))