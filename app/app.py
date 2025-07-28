import streamlit as st
import time

from streamlit import spinner

from url_extract import extract_article_details
from services.classfication_service import classify_service
from services.summarization_service import summarize_service


st.warning("U.S Politics News Article Processing App")
text = st.chat_input('Input your news article url here...')

with st.spinner():
    time.sleep(1)
    with st.chat_message("ai"):
        st.write("👋 Hey there! I’m NewsBot — your personal news companion. I’m here to help you make sense of the news! 📰✨ Just drop a link to any article, and I’ll: 🔍 Analyze it to check if it’s real or fake 🧠 Summarize it so you can get the gist without reading the whole thing Let’s dive into the headlines — one link at a time! 🚀")

if text:
    with st.spinner("Processing...", show_time=True):
        details = extract_article_details(text)
        classify_result = classify_service(details['content'])
        sum_result = summarize_service(details['content'])
        cat_label = classify_result['category'][0]
        cat_score = classify_result['category'][1]
        summary = sum_result['category']
    with st.chat_message("ai"):
        st.write("Headline:", details['title'])
        if cat_label == "real":
            st.badge(f"{cat_label} | {cat_score:.2f}", color="green")
        else:
            st.badge(f"{cat_label} | {cat_score:.2f}", color="red")
        st.write(summary)