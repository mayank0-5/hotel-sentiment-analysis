import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hotel Sentiment Analyzer", layout="wide")
st.title("Hotel Review Sentiment Analyzer")
st.write("Analyze sentiment of hotel reviews instantly!")

st.markdown("---")
st.subheader("Try It Yourself!")
user_review = st.text_area("Enter a hotel review here:")

if st.button("Analyze Sentiment"):
    if user_review:
        analysis = TextBlob(user_review)
        score = analysis.sentiment.polarity
        if score > 0.1:
            st.success("Positive Review! Score: " + str(round(score,2)))
        elif score < -0.1:
            st.error("Negative Review! Score: " + str(round(score,2)))
        else:
            st.warning("Neutral Review! Score: " + str(round(score,2)))

st.markdown("---")
df = pd.read_csv("data/sentiment_results.csv")
st.subheader("Dataset Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Positive", len(df[df["TextBlob_Sentiment"] == "Positive"]))
col3.metric("Negative", len(df[df["TextBlob_Sentiment"] == "Negative"]))
st.dataframe(df[["Review", "Rating", "TextBlob_Sentiment"]].head(20))