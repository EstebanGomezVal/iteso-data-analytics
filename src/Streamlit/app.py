import streamlit as st
import pandas as pd

st.title("Titanic")
st.header("Data set")
st.markdown("*Este es una base de datos del Titanic*")

df = pd.read_csv("Titanic-Dataset.csv")
st.dataframe(df)
