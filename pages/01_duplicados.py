import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Análisis de Duplicados")


#st.write("Subir archivo")
file_upload = st.file_uploader("Subir archivo para analisis",type=['csv'])

if file_upload:
    df = pd.read_csv(file_upload)
    st.write(df.head())

    colorg = df.columns
    featorg = st.selectbox('Seleccionar columna original',options=colorg)
    featdup = st.selectbox('Seleccionar columna duplicado',options=colorg)

    st.write(featorg)
    st.write(featdup)

     
    
