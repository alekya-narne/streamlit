import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boy Child Preference in Indian Mothers
Masters Project
""")
st.write('---')

loadedmodel = pickle.load(open('/home/student/Desktop/my_churn_app/combinemodel.sav', 'rb'))

#Visualisation
chart_select = st.sidebar.selectbox(
    label ="Type of chart",
    options=['Scatterplots','Lineplots','Histogram','Boxplot']
)

if chart_select == 'Scatterplots':
    st.sidebar.subheader('Scatterplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.scatter(data_frame=df_boston,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Histogram':
    st.sidebar.subheader('Histogram Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        plot = px.histogram(data_frame=df_boston,x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplots':
    st.sidebar.subheader('Lineplots Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.line(df_boston,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Boxplot':
    st.sidebar.subheader('Boxplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.box(df_boston,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    v159 = st.sidebar.slider('Frequency of watching TV',0, 3, 1)
    hv208 = st.sidebar.slider('Has Television', 0, 1, 1)
    v157 = st.sidebar.slider('Frequency of reading Newspaper/Magazine', 0, 3, 1)
    hv210 = st.sidebar.slider('Has bicycle', 0, 1, 1)
    hv106_01 = st.sidebar.slider('Highest Education Level obtained', 0, 4, 1)
    hv211 = st.sidebar.slider('Has Motorcycle/Scooter', 0, 1, 1)
    thevtotal = st.sidebar.slider('Beating wife justified', 0, 6, 1)
    v158 = st.sidebar.slider('Frequency of Listening to Radio', 0, 3, 1)
    v731 = st.sidebar.slider('Employement in the last 12 months', 0, 1, 1)
    sh37n = st.sidebar.slider('Has access to Internet', 0, 1, 1)
    hv243a = st.sidebar.slider('Has Mobile/Telephone', 0, 1, 1)
    d102 = st.sidebar.slider('Husbands number of Control Issues', 0, 7, 1)
    hv207 = st.sidebar.slider('Has radio', 0, 1, 1)
    five = st.sidebar.slider('Experienced instances of Domestic Violence',0, 28, 1)
    sh37o = st.sidebar.slider('Has a Computer',0, 1, 1)
    three = st.sidebar.slider('Experienced any Emotional Violence',0, 10, 1)
    hv212 = st.sidebar.slider('Has Car/Truck',0, 1, 1)
    sh37z = st.sidebar.slider('Has a Tractor',0, 1, 1)
    hv243c = st.sidebar.slider('Has Animal-Drawn Cart',0, 1, 1)
    hv221 = st.sidebar.slider('Has Telephone (land-line)',0, 1, 1)
    ten = st.sidebar.slider('Suffered Injuries from Abuse from Husband',0, 6, 1)
    data = {
            'v157': v157,
            'v158': v158,
            'v159': v159,
            'hv207': hv207,
            'hv208': hv208,
            'hv221': hv221,
            'hv243a': hv243a,
            'sh37n': sh37n,
            'sh37o': sh37o,
            'hv210': hv210,
            'hv211': hv211,
            'hv212': hv212,
            'hv243c': hv243c,
            'sh37z': sh37z,
            'hv106_01': hv106_01,
            'v731': v731,
            'd102': d102,
            'thevtotal': thevtotal,
            '103s': three,
            '105s': five,
            '110s': ten
            #'v159': v159,
            #'hv208': hv208,
            #'v157': v157,
            #'hv210': hv210,
            #'hv106_01': hv106_01,
            #'hv211': hv211,
            #'thevtotal': thevtotal,
            #'v158': v158
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Apply Model to Make Prediction
prediction = loadedmodel.predict(df)

st.header('Classification of boy preference')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(loadedmodel)
shap_values = explainer.shap_values(df)
if st.button('Show SHAP Graphs'):
    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, df)
    st.pyplot(bbox_inches='tight')
    #st.write('---')
    #plt.title('Feature importance based on SHAP values')
    #shap.summary_plot(shap_values, df, plot_type="bar")
    #st.pyplot(bbox_inches='tight')



