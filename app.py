import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pickle
from xgboost import XGBClassifier
from feature_engine.creation import CyclicalFeatures
import time
import shap

st.set_page_config(page_title="Ad Campaign Dashboard", layout="wide", initial_sidebar_state='expanded')
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

df_dashboard = pd.read_csv('df_dashboard.csv', parse_dates=['Timestamp'])
df_test = pd.read_csv('df_predict.csv', parse_dates=['Timestamp'])
xgb = pickle.load(open('xgb.pkl', 'rb'))

def dashboard_page(df_dashboard):
    st.header('Ad Campaign Dashboard')

    col1_, col2_, col3_, col4_ = st.columns(4, gap='medium')
    with col1_:
        dates = list(df_dashboard['Timestamp'].sort_values(ascending=True).dt.strftime('%Y %B').unique())
        option_date = st.selectbox('Select date', ['All'] + dates)

    if option_date == 'All':
        clicked = df_dashboard['Clicked on Ad']
        filtered = df_dashboard
    else:
        clicked = df_dashboard[df_dashboard['Timestamp'].dt.strftime('%Y %B') == option_date]['Clicked on Ad']
        filtered = df_dashboard[df_dashboard['Timestamp'].dt.strftime('%Y %B') == option_date]

    with st.expander('Dataset', expanded=False):
        if option_date == 'All':
            st.dataframe(df_dashboard)
        else:
            st.dataframe(df_dashboard[df_dashboard['Timestamp'].dt.strftime('%Y %B') == option_date])

    col1, col2, col3, col4 = st.columns(4, gap='medium')
    with col1:
        st.metric(label='Clicks', value=clicked.sum())

    with col2:
        st.metric(label='Impression', value=len(clicked))

    with col3:
        st.metric(label='Click-through Rate', value=str(round(clicked.sum()/len(clicked),2)*100)+' %')

    with col4:
        st.metric(label='Cost per Click', value='$ 0.97')
    
    col5, col6, col7 = st.columns(3, gap='medium')
    with col5:
        fig = px.line(filtered.groupby(pd.Grouper(key='Timestamp', freq='D'))['Clicked on Ad'].sum().reset_index(),
            x='Timestamp',
            y='Clicked on Ad',
            line_shape='spline',
            title='Clicks Over Time')
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        monthly_ad = filtered[['Ad Category', 'Clicked on Ad']].groupby('Ad Category').sum().reset_index()
        fig = px.pie(monthly_ad,
            names='Ad Category',
            values='Clicked on Ad',
            hole=0.5,
            title='Ad Preference',
            color_discrete_sequence=px.colors.sequential.RdBu
            )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=30)

        )
        st.plotly_chart(fig, use_container_width=True)

    with col7:
        map_data = px.data.gapminder().query("year == 2007")[['country', 'continent']]
        fig = px.choropleth(
            map_data,
            locations="country",
            locationmode="country names",
            color="continent",
            hover_name="country",            
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            title="Clicks by Region",
            geo=dict(
                showcoastlines=True,
                coastlinecolor="Black",
                showland=False,
                showframe=False
            ),
            margin=dict(l=20, r=20, t=30, b=30),
        )

        st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    cola, colb, colc = st.columns((1,3,1))
    with colb:
        st.image('logo.png')
    selected = option_menu(
        None,
        ['Dashboard', 'Predict'],
        icons=['bi-graph-up-arrow', 'gear', 'sliders'],
        menu_icon='cast',
        default_index=0
    )

def preprocess(df):
    df['Hour'] = df['Timestamp'].dt.hour
    cyclic = CyclicalFeatures(variables=None, drop_original=True)
    conv = cyclic.fit_transform(df[['Hour']])
    df = pd.concat([df, conv], axis=1).drop('Hour', axis=1)

    df['Weekend'] = df['Timestamp'].apply(lambda x: x.weekday() >= 5)
    df['Weekend'] = df['Weekend'].astype(int)

    df = df.drop('Timestamp', axis=1)
    df = pd.get_dummies(df, dtype='int')

    return df  

def predict(df):
    df_preprocessed = preprocess(df)
    preds = xgb.predict(df_preprocessed[xgb.get_booster().feature_names])
    df_preds = pd.concat([pd.DataFrame(preds, columns=['Clicked on Ad?']), df], axis=1)

    return df_preds


def predict_page():

    with st.expander('Model details', expanded=True):
        col8, col9, col10 = st.columns(3)
        with col8:
            st.metric(label='Model name', value='XGBoost')
        with col9:
            st.metric(label='Accuracy', value='94 %')
        with col10:
            st.metric(label='Date created', value='01/01/2023')

        x = pd.date_range(start='2022-12-31', end='2023-09-01', freq='M')
        y = [0.97, 0.97, 0.96, 0.96, 0.93, 0.94, 0.93, 0.94, 0.94]

        model_acc_plot = px.line(x=x, y=y, title='Model accuracy over time', markers=True, labels={'x': 'Date', 'y': 'Accuracy'})
        model_acc_plot.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(model_acc_plot, use_container_width=True)

    upload = st.button('Upload sample input dataset', type='primary')
    if upload == True:
        with st.expander('Dataset input'):
            st.write(df_test.replace({1:'Yes', 0:'No'}))

        df_result = predict(df_test)

        with st.spinner('Loading'):
            time.sleep(1)

        with st.expander('Predicted output'):
            st.write(df_result.replace({1:'Yes', 0:'No'}))

        with st.expander('Feature importance'):
            col13, col14 = st.columns(2)
            with col14:
                explainer = shap.Explainer(xgb)
                shap_values = explainer.shap_values(preprocess(df_test).iloc[df_result[df_result['Clicked on Ad?'] == 1].index])
                shap.summary_plot(shap_values, features=preprocess(df_test).iloc[df_result[df_result['Clicked on Ad?'] == 1].index],
                                        alpha=0.5,
                                        max_display=5,
                                        show=False
                                        )
                fig1 = plt.gcf()
                fig1.set_size_inches(10.5, 5.8)
                for ax1 in fig1.get_axes():
                    for item1 in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
                        item1.set_fontsize(16)
                fig1.suptitle("Impact of top 5 features towards positive class", fontsize=16)
                fig1.tight_layout()
                st.pyplot(fig1, use_container_width=True)

            with col13:
                plt.clf()
                shap.summary_plot(shap_values, features=preprocess(df_test).iloc[df_result[df_result['Clicked on Ad?'] == 1].index],
                                        alpha=0.6,
                                        plot_type='bar',
                                        show=False
                                        )
                fig2 = plt.gcf()
                fig2.set_size_inches(10.5, 6)
                for ax2 in fig2.get_axes():
                    for item2 in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
                        item2.set_fontsize(16)
                fig2.suptitle("Average impact of all features towards positive class", fontsize=16)
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)




if selected == 'Dashboard':
    dashboard_page(df_dashboard)
elif selected == 'Predict':
    predict_page()