import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

tfidf = pickle.load(open("tfidf.pickle", "rb"))
model = pickle.load(open("svc.pickle", "rb"))


def check_tweet(tweet):
    ps = PorterStemmer()
    tweet = tweet.lower()
    tweet = re.sub("[^a-zA-Z]", " ", tweet)
    tweet = tweet.split()
    text = [ps.stem(word) for word in tweet if word not in stopwords.words("english")]
    text = " ".join(text)
    text = [text]
    text = tfidf.transform(text)
    prediction = model.predict(text)

    interpretations = {
        0: "Age",
        1: "Ethnicity",
        2: "Gender",
        3: "Not Cyberbullying",
        4: "Other Cyberbullying",
        5: "Religion",
    }

    if prediction == 0:
        return 'This is Age related cyberbullying tweet'
    elif prediction == 1:
        return 'This is Ethnicity related cyberbullying tweet'
    elif prediction == 2:
        return 'This is Gender related cyberbullying tweet'
    elif prediction == 3:
        return 'This is not a cyberbullying tweet'
    elif prediction == 4:
        return 'This is some other cyberbullying tweet'
    elif prediction == 5:
        return 'This is Religion related cyberbullying tweet'
    else:
        return "Sorry, I am not able to help."

image=Image.open('images/stop_cyberbully_logo.jpg')
st.image(image,width=700)

st.title("CYBERBULLY TWEET DETECTION")

st.write("This app classifies tweet between 6 categories:")

st.write(
    """
        - Age
        - Ethnicity
        - Gender
        - Religion
        - Other Cyberbullying
        - Not Cyberbullying
    """
)

input_tweet = st.text_area("Enter tweet below:", placeholder="Type here", height=150)


if st.button("Search"):
    if input_tweet:
        st.header("Entered tweet:")
        st.write(input_tweet)
        st.write(
            """
                ***
        """
        )
        st.header("Prediction:")
        st.write(check_tweet(input_tweet))
    else:
        st.write("No tweet entered!")

st.write(
    """
***
"""
)
