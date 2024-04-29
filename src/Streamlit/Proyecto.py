import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("Visualización del Titanic")
st.header("Data set")
st.markdown("*Este es una base de datos acerca del Titanic*")