import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
import folium
import geopandas as gpd
import json



st.title('First Responder Action Prioritization')

# DATE_COLUMN = 'tweet_text', 'class_label','disaster_type','disaster_year','country'
DATA_URL = ('../Data/clean_data.csv')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    # return data[['tweet_text', 'class_label','disaster_type','disaster_year','country']]
    return data[['tweet_text', 'actionable', 'country']]

# data_load_state = st.sidebar.text('Loading data...')
data = load_data(10000)
# data_load_state.text("Data loaded!")

# if st.checkbox('Show raw data', key='show_data'):
#     st.subheader('Raw data')
#     st.write(data)

countries_to_highlight = []
st.sidebar.header("Choose the places you want to go in prio:")

for index, event in data.head(20).iterrows():
    tweet_text = event['tweet_text']
    actionable = event['actionable']

    if actionable:  # Check if the event is actionable
        st.sidebar.markdown('---')
        st.sidebar.write(tweet_text)
        take_key = f"take_{index}"
        take = st.sidebar.checkbox('I take it', key=take_key)
        if take:
            countries_to_highlight = event['country']
            if countries_to_highlight.find(', '):
                countries_to_highlight = countries_to_highlight.split(', ')
            # st.markdown('---')
            # st.write(tweet_text)
            # st.markdown('---')
        # st.sidebar.markdown('---')



# Load the GeoJSON file -- country boundaries
with open('../Data/countries_boundaries.geojson', 'r') as f:
    world_geojson = json.load(f)



def style_function(feature):
    if feature['properties']['NAME'] in countries_to_highlight:
        return {
            'fillColor': '#ffaf00',
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0.5
        }
    return {
        'fillColor': '#fafafa',
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.1
    }

m = folium.Map(location=[20, 0], zoom_start=2.3)

folium.GeoJson(
    world_geojson,
    style_function=style_function
).add_to(m)

folium_static(m)
