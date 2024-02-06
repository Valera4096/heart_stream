import joblib

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold
#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



ml_pipeline = joblib.load('ml_pipeline.pkl')



age = st.number_input('Введите ваш возвраст')


Sex1 = st.selectbox(
    'Ваш пол?',
    ('Мужской', 'Женский'))

ChestPainType = st.selectbox(
    'Тип боли в груди?',
    ('TA: типичная стенокардия', 'ATA: атипичная стенокардия', 'NAP: неангинальная боль', 'ASY: бессимптомная'))


RestingBP = st.number_input('Артериальное давление в состоянии покоя [мм рт. ст.]')

Cholesterol = st.number_input('холестерин сыворотки')

FastingBS = st.selectbox(
    'уровень сахара в крови натощак Больше 120 мг/дл ?',
    ('Да' , 'Нет'))

RestingECG = st.selectbox(
    'ЭКГ покоя__: результаты электрокардиограммы покоя',
    ('Нормальный: нормальный' ,'ST: наличие аномалий',' ГЛЖ: вероятная или определенная гипертрофия левого желудочка по критериям Эстеса'))

MaxHR = st.number_input('Достигнутая максимальная частота пульса')


ExerciseAngina = st.selectbox(
    'стенокардия, вызванная физической нагрузкой',
    ('Да' , 'Нет'))


Oldpeak = st.slider('Числовое значение, измеренное в депрессии?', 0.0, 7.0, 0.5)

ST_Slope = st.selectbox(
    'наклон пикового сегмента ST при нагрузке',
    ('Наклон вверх', 'Плоский', 'Наклон вниз'))

if st.button('Дать предсказание'):
    
    Sex = 'M' if Sex1 == 'Мужской' else 'F'
    if ChestPainType == 'TA: типичная стенокардия':
        ChestPainType = 'TA'
        
    elif ChestPainType == 'ATA: атипичная стенокардия':
        ChestPainType = 'ATA'
        
    elif ChestPainType == 'NAP: неангинальная боль':
        ChestPainType = 'NAP'
        
    elif ChestPainType == 'ASY: бессимптомная':
        ChestPainType = 'ASY'
        
    FastingBS = 1 if FastingBS== 'Да' else 0
        
    if RestingECG == 'Нормальный: нормальный':
        RestingECG = 'Normal'
    elif RestingECG == 'ST: наличие аномалий':
        RestingECG = 'ST'
    elif RestingECG == ' ГЛЖ: вероятная или определенная гипертрофия левого желудочка по критериям Эстеса':
        RestingECG = 'LVH'
        
        
    ExerciseAngina = 'Y' if FastingBS == 'Да' else 'N'
    
    if ST_Slope == 'Наклон вверх':
        ST_Slope = 'Up'
    elif ST_Slope == 'Плоский':
        ST_Slope = 'Flat'
    elif ST_Slope == 'Наклон вниз':
        ST_Slope = 'Down'
    
    data = pd.DataFrame({
    'Age': [age],
    'Sex': [Sex],
    'ChestPainType': [ChestPainType],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'RestingECG': [RestingECG],
    'MaxHR': [MaxHR],
    'ExerciseAngina': [ExerciseAngina],
    'Oldpeak': [Oldpeak],
    'ST_Slope': [ST_Slope]})
    
    res = ml_pipeline.predict(data)


   
    st.write('Здоров' if res == 0 else ' Не здоров')
    
    if res == 0:
        st.image('maxresdefault.jpg')
    else:
        st.image('povezlo-povezlo.jpg')