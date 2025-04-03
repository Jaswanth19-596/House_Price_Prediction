import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor



with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('df.pkl', 'rb') as file:
    df = pickle.load(file)


st.title("Welcome to House Price Prediction")


# st.dataframe(df.head())

st.header('Enter the Details of the House')


bedrooms = st.selectbox("Enter Number of Bedrooms : ", sorted(df['Num_Bedrooms'].unique()))
lot_size = st.slider("Enter Lot Size", min_value = 0, max_value = 10, step = 1)
square_foot = st.slider("Enter Square Foot Area", min_value = 0, max_value = 10000, step = 100)


button = st.button("Submit")

if button:
    df = pd.DataFrame(columns = ['Square_Footage', 'Num_Bedrooms', 'Lot_Size'])

    df.loc[0] = [square_foot, bedrooms, lot_size]

    result = round(pipeline.predict(df)[0])

    min_value = result - 9800
    max_value = result + 9800
    st.text(f"The house could be between {min_value} and {max_value} Rupees")



