import streamlit as st
import pandas as pd
st.set_page_config(page_title="911")

DATA_URL = ('../Data/presentation.csv')


st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap')

        .css-b3z5c9 { width: 100%; }
        .css-1y4p8pa { max-width:70%}
        # .css-11orhzd {margin-top:100px}
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        .css-10trblm {
            color:#1DA1F2;
        font-family: "Noto Sans", sans-serif;

        }
        .css-q3bdcp { width: 100%; }
        button  {
            color:#1DA1F2; border-color: #1DA1F2;
        }
        button::hover  {
            color:#1DA1F2; border-color: #1DA1F2;
        }

    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data[['tweet_text', 'class_label', 'disaster_year', 'place', 'disaster_type', 'disaster', 'actionable']]

data = load_data(200)

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    st.session_state.stage = 0  # 0: Initial, 1: Disaster, 2: Actionable
    st.session_state.display_tweet = data.iloc[0]['tweet_text']

def next_tweet():
    if st.session_state.current_index < len(data) - 1:
        st.session_state.current_index += 1
    else:
        st.session_state.current_index = 0
    st.session_state.display_tweet = data.iloc[st.session_state.current_index]['tweet_text']
    st.session_state.stage = 0

def mark_disaster():
    st.session_state.stage = 1

def mark_actionable():
    st.session_state.stage = 2



col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    st.markdown("<h1 style='text-align: center;'>All the<br>tweets</h1>", unsafe_allow_html=True)
    st.markdown('---')
    if st.session_state.stage == 0:
        st.write(st.session_state.display_tweet)
        st.button('Find if this tweet is a disaster', on_click=mark_disaster, key='button_state')

with col2:
    st.markdown("<h1 style='text-align: center;'>Disaster<br>Tweet</h1>", unsafe_allow_html=True)
    st.markdown('---')
    if st.session_state.stage == 1:
        st.write(st.session_state.display_tweet)
        st.button('Is this tweet actionable?', on_click=mark_actionable)

with col3:
    st.markdown("<h1 style='text-align: center;'>Actionable<br>Tweet</h1>", unsafe_allow_html=True)
    st.markdown('---')
    if st.session_state.stage == 2:
        st.write(st.session_state.display_tweet)


st.markdown('---')

_, col_middle, _ = st.columns([1,1,1])
with col_middle:
    st.button('I want a new Tweet', on_click=next_tweet)
