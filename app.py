import streamlit as st
import pandas as pd
import numpy as np
from os import path

st.title('Analyse und Vorhesage von Fahrradunfälle')

data = pd.read_csv('data/Berlin-incidents.csv', on_bad_lines='skip') #path folder of the data file

st.write(data) #displays the table of data
# Create a text element and let the reader know the data is loading.

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
#data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

#st.subheader('Raw data')
#st.write(data)
st.subheader('Map of all Nearly accidents')
st.map(data)
