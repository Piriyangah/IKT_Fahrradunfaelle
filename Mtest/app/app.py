import streamlit as st
import pandas as pd
import numpy as np
from os import path

st.set_page_config(
    page_title="Machine learning",
    page_icon="🚲",
)

st.title("Prädiktive Analyse von Fahrradunfällen basierend auf Beinahe-Unfällen und realen Unfällen aus dem Jahr 2022")
st.sidebar.success("Überblick über die Projektaufgabe")


#data = pd.read_csv('data/Berlin-incidents.csv', on_bad_lines='skip') #path folder of the data file

#st.write(data) #displays the table of data
# Create a text element and let the reader know the data is loading.

#data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
#data = load_data(10000)
# Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading data...done!')

#st.subheader('Raw data')
#st.write(data)
#st.subheader('Map of all Nearly accidents')
#st.map(data)



