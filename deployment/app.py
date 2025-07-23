import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title='Skin Type Recognition',
    layout='wide',
    initial_sidebar_state='expanded'
)

page = st.sidebar.radio('Pilih Halaman', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()

else:
    prediction.run()