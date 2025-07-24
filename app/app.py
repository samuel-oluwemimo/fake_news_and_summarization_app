import streamlit as st
import time

from streamlit import spinner

from url_extract import extract_article_details


st.warning("U.S Politics News Article Processing App")
text = st.chat_input('Input your news article url here...')

with st.spinner():
    time.sleep(1)
    with st.chat_message("ai"):
        st.write("👋 Hey there! I’m NewsBot — your personal news companion. I’m here to help you make sense of the news! 📰✨ Just drop a link to any article, and I’ll: 🔍 Analyze it to check if it’s real or fake 🧠 Summarize it so you can get the gist without reading the whole thing Let’s dive into the headlines — one link at a time! 🚀")

if text:
    with st.spinner("Processing...", show_time=True):
        # details = extract_article_details(text)
        time.sleep(5)
    # st.write("Title:", details['title'])
    # st.write("Content:", details['content'])
    with st.chat_message("user"):
        st.write("fuck thou")
        # st.write("Headline:", details['title'])
        #
        # if result == "real":
        #     st.badge(f"{result} | {score:.2f}", color="green")
        # else:
        #     st.badge(f"{result} | {score:.2f}", color="red")
        #
        # st.write(summary)