import streamlit as st
import joblib
from googletrans import Translator
from gtts import gTTS
import os
import smtplib

# Load the trained model, label encoder, and TF-IDF vectorizer
svm_model = joblib.load('model/dream_interpretation_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Initialize the translator
translator = Translator()

# Streamlit UI Setup
st.set_page_config(page_title="Dream Interpretation System", page_icon="🌙")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('static/earth-planet.gif'); /* Ensure this path is correct */
        background-size: cover; /* Ensures the background covers the entire element */
        background-position: center; /* Centers the background image */
        background-repeat: no-repeat; /* Prevents the background from repeating */
    }
    .title {
        text-align: center; 
        white-space: nowrap; 
        overflow: visible; 
        width: auto; 
        display: inline-block; 
    }
    .subheader {
        text-align: right; 
        white-space: nowrap; 
        overflow: visible; 
        width: auto; 
        display: inline-block; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.subheader("☪︎السلام علیکم ☪︎")
st.subheader("☝🏼لا إله إلا الله")
st.markdown('<div class="title">Dream Interpretation System 🌙</div>', unsafe_allow_html=True)
st.subheader(" ")
st.markdown('<div class="title">Model Trained By Ibn Sirin\'s Dictionary of Dreams</div>', unsafe_allow_html=True)

# Developer Information
st.sidebar.header("Developer Information")
st.sidebar.text("Developer: Mr. Soul Hacker")
st.sidebar.text("Contact Details: +91XXX XXXXX")
st.sidebar.text("Instagram: @loser___xxxx")
st.sidebar.text("Facebook: @loser0fXXXX]")
st.sidebar.text("GitHub: @Alfaj01")
st.sidebar.text("Email: AlfajXXX@gmail.com")
st.sidebar.markdown("## Feedback")
feedback = st.sidebar.text_area("Leave your feedback here:")
if st.sidebar.button("Submit Feedback"):
    if feedback:
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login('your_email@gmail.com', 'your_email_password')  # Update with your email and password
                message = f"Feedback received:\n\n{feedback}"
                server.sendmail('your_email@gmail.com', 'your_email@gmail.com', message)
                st.sidebar.success("Feedback submitted successfully!")
        except Exception as e:
            st.sidebar.error("Failed to send feedback.")

# User Input Options: Text Only (Removed Voice Input due to PyAudio issues)
st.markdown('<div class="subheader">Enter your dream description</div>', unsafe_allow_html=True)
user_input = st.text_area("Describe your dream here...")

# Language Detection and Translation to English
if user_input:
    detected_lang = translator.detect(user_input).lang
    st.write(f"Detected Language: {detected_lang}")

    translated_input = translator.translate(user_input, src=detected_lang, dest='en').text
    st.write(f"Translated Text: {translated_input}")

    input_vector = tfidf.transform([translated_input]).toarray()
    prediction = svm_model.predict(input_vector)
    predicted_label = label_encoder.inverse_transform(prediction)

    st.write(f"**Dream Interpretation (in English):** *{predicted_label[0]}*")
    
    translated_interpretation = translator.translate(predicted_label[0], src='en', dest=detected_lang).text
    st.write(f"**Dream Interpretation (in Original Language - {detected_lang}):** *{translated_interpretation}*")

    if st.button("Read Aloud"):
        tts = gTTS(translated_interpretation, lang=detected_lang)
        tts.save("interpretation.mp3")
        os.system("start interpretation.mp3")  # Use "start" for Windows, "open" for macOS, or "xdg-open" for Linux
        st.audio("interpretation.mp3", format='audio/mp3')
