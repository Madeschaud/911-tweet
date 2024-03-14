import streamlit as st
import pandas as pd
import requests
from tweet_911.params import URL_API
import time

st.set_page_config(page_title="911")

DATA_URL = ('../Data/presentation.csv')


st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap')

        .css-b3z5c9 { width: 100%; }
        .css-1y4p8pa { max-width:70%}
        footer {visibility: hidden;}
        .css-10trblm {
            color:#1DA1F2;
            font-family: "Noto Sans", sans-serif;
        }
        # h1 {
        #     color:#FFF;
        #     font-size:44px !important;
        #     font-weight:700 !important;
        #     font-family: "Noto Sans", sans-serif;
        # }
        .css-q3bdcp { width: 100%; }
        button  {
            background-color:#1DA1F2 !important;
            color:#FFFFFF !important;
            border-color: #1DA1F2 !important;
            border-radius:20px  !important;
            padding:10px 20px 10px 20px !important;
            margin-top:50% !important;
        }
        .stButton{
            display:flex;
            justify-content:center;
        }
        .css-1y4p8pa{
            padding-bottom:0px;
        }

        .css-1d3srhg p{
            border:1px solid rgba(49, 51, 63, 0.2);;
            border-radius:20px;
            text-align:center;
            padding:18px;
            color:white;
        }
         .e16nr0p34 p a{
             color:white;
        }

    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.head(3)
    return data[['tweet_text', 'class_label', 'disaster_year', 'place', 'disaster_type', 'disaster', 'actionable']]

data = load_data(200)

if 'current_index' not in st.session_state:
    # st.session_state.current_index = 0
    # st.session_state.id_stage = {'id': 0, 'state':0} # 0: Initial, 1: Disaster, 2: Actionable
    st.session_state.stage = 0 # 0: Initial, 1: Disaster, 2: Actionable
    # st.session_state.display_tweet = data.iloc[0]['tweet_text']
    st.session_state.display_tweet = data['tweet_text']
    st.session_state.params = { 'tweet': st.session_state.display_tweet }

# def next_tweet():
#     if st.session_state.current_index < len(data) - 1:
#         st.session_state.current_index += 1
#     else:
#         st.session_state.current_index = 0
#     # st.session_state.display_tweet = data.iloc[st.session_state.current_index]['tweet_text']
#     st.session_state.stage = 0
#     st.session_state.params = { 'tweet': st.session_state.display_tweet }

def mark_disaster():
    for index in range(0, len(st.session_state.display_tweet)):
        response = requests.get(f'{URL_API}/predict_disaster', { 'tweet': st.session_state.display_tweet.iloc[index] })
        prediction = response.json()
        pred = prediction['tweet_accurate']
        print(pred)
        if round(pred, 2) > 0.3:
            col1.write(st.session_state.display_tweet.iloc[index])

        # else:
        #     next_tweet()

def mark_actionable():

    response = requests.get(f'{URL_API}/predict_disaster', st.session_state.params)
    prediction = response.json()
    pred = prediction['tweet_accurate']
    if round(pred, 2) > 0.3:
        st.session_state.stage = 2
    # else:
    #     next_tweet()

def display_tweet():
    with col1:
        # for index in range(0, len(st.session_state.display_tweet)):

        if st.session_state.stage == 1:
            print('here')
            col1.write(st.session_state.display_tweet)
            st.button('Is this tweet actionable?', on_click=mark_actionable)
col1, col2 = st.columns([1, 1], gap="small")

with st.sidebar:
    if st.session_state.stage == 0:
        for index in range(0, len(st.session_state.display_tweet)):
            st.write(st.session_state.display_tweet.iloc[index])

with col1:
    st.markdown("<h1 style='text-align: center;'>Disaster<br>Tweet</h1>", unsafe_allow_html=True)
    st.markdown('---')


with col2:
    st.markdown("<h1 style='text-align: center;'>Actionable<br>Tweet</h1>", unsafe_allow_html=True)
    st.markdown('---')
#     if st.session_state.stage == 2:
#         st.write(st.session_state.display_tweet)
#         st.button('I want a new Tweet', on_click=next_tweet)
# # with footer:
st.button('Find if the tweets are disasters', on_click=mark_disaster, key='button_state')
