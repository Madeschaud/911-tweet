import streamlit as st
import pandas as pd
import requests

DATA_URL = 'tweet_911/app/data_streamlit/presentation.csv'

URL_API = st.secrets['URL_API']

st.set_page_config(page_title="911")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap')

        # .css-b3z5c9 { width: 100%; }
        # .css-1y4p8pa { max-width:70%}
        footer {visibility: hidden;}
        .css-10trblm {
            color:#1DA1F2;
            font-family: "Noto Sans", sans-serif;
        }
        .css-q3bdcp { width: 100%; }
        button  {
            background-color:#1DA1F2 !important;
            color:#FFFFFF !important;
            border-color: #1DA1F2 !important;
            border-radius:20px  !important;
            padding:10px 20px 10px 20px !important;
            # margin-top:30% !important;
        }
        .stButton{
            display:flex;
            justify-content:center;
        }
        .css-1y4p8pa{
            padding-bottom:0px;
        }
        # .css-1p1nwyz .css-5rimss p
        # .css-90qg9c .css-5rimss p
        # .css-90qg9c{
        #     border:1px solid rgba(49, 51, 63, 0.2)  !important;
        #     border-radius:20px;
        #     text-align:center;
        #     padding:18px;
        #     color:white;
        # }
        #  .css-1p1nwyz .stMarkdown .css-5rimss p a{
        #      color:white;
        # }
        .e16nr0p34 p{
            border:1px solid rgba(49, 51, 63, 0.2);
            border-radius:20px;
            text-align:center;
            padding:18px;
            color:white;
        }
        .stMarkdown .e16nr0p34 p a{
            color:white;
        }

        .css-6qob1r{
            background-color:#1DA1F2
        }
        .css-uf99v8, .css-1avcm0n{
            background-color:white;
        }


    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.head(5)
    return data[['tweet_text', 'class_label', 'disaster_year', 'place', 'disaster_type', 'disaster', 'actionable']]

data = load_data(200)

if 'display_tweet' not in st.session_state:
    st.session_state.stage = 0 # 0: Initial, 1: Disaster, 2: Actionable
    st.session_state.display_tweet = data['tweet_text']
    st.session_state.new_tweets = []

def mark_disaster():
    print(URL_API)
    for index in range(0, len(st.session_state.display_tweet)):
        response = requests.get(f'{URL_API}/predict_disaster', { 'tweet': st.session_state.display_tweet.iloc[index] })
        prediction = response.json()
        pred = prediction['tweet_accurate']
        print(pred)
        if round(pred, 2) > 0.3:
            col1.write(st.session_state.display_tweet.iloc[index])
            st.session_state.new_tweets.append(st.session_state.display_tweet.iloc[index])
    st.session_state.stage = 1
    # st.rerun()


def mark_actionable():
    print(st.session_state.new_tweets)
    for index in range(0, len(st.session_state.new_tweets)):
        response = requests.get(f'{URL_API}/predict_actionable', { 'tweet': st.session_state.new_tweets[index] })
        prediction = response.json()
        pred = prediction['tweet_accurate']
        print(pred)
        if round(pred, 2) > 0.3:
            col2.write(st.session_state.new_tweets[index])
    st.session_state.stage = 2


col1, col2 = st.columns([1, 1], gap="small")

with st.sidebar:
    for index in range(0, len(st.session_state.display_tweet)):
        st.sidebar.write(st.session_state.display_tweet.iloc[index])

with col1:
    st.markdown("<h1 style='text-align: center;'>Disaster<br>Tweet</h1>", unsafe_allow_html=True)
    st.markdown('---')
    if st.session_state.stage == 0:
        st.button('Find if the tweets are disasters', on_click=mark_disaster, key='button_state')
        print(st.session_state.stage)


with col2:
    st.markdown("<h1 style='text-align: center;'>Actionable<br>Tweet</h1>", unsafe_allow_html=True)
    st.markdown('---')
    if st.session_state.stage == 1:
        st.button('Find if the tweets are actionable', on_click=mark_actionable, key='button_state')
        print(st.session_state.stage)
