import streamlit as st
import joblib
from googletrans import Translator
from gtts import gTTS
import os
import speech_recognition as sr
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
    }
    .title, .subheader, .sidebar-header {
        text-align: center;
        white-space: nowrap; /* Prevent line break */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="title">Dream Interpretation System 🌙</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Model Trained By Ibn Sirin\'s Dictionary of Dreams Book and use some other resources also</div>', unsafe_allow_html=True)

# Developer Information
st.sidebar.markdown('<div class="sidebar-header">Developer Information</div>', unsafe_allow_html=True)
st.sidebar.text("Developer: Mr. Soul Hacker")
st.sidebar.text("Contact Details: +91XXX XXXXX")
st.sidebar.text("Instagram: @loser___xxxx")
st.sidebar.text("Facebook: @loser0fXXXX")
st.sidebar.text("GitHub: @Alfaj01")
st.sidebar.text("Email: AlfajXXX@gmail.com")

# Feedback Functionality
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

# User Input Options: Text or Voice
st.subheader("Enter your dream description:")
input_type = st.radio("Choose input type:", ("Text", "Voice"), index=1)

user_input = ""

if input_type == "Text":
    user_input = st.text_area("Describe your dream here...", placeholder="Type your dream description here...")
else:
    if st.button("Record"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio_data = recognizer.listen(source)
            st.write("Recognizing...")
            try:
                user_input = recognizer.recognize_google(audio_data)
                st.write(f"Recognized Text: {user_input}")
            except sr.UnknownValueError:
                st.write("Sorry, I could not understand your speech. Please try again.")
            except sr.RequestError:
                st.write("Could not request results from the speech service; check your network connection.")

# Submit Button
if st.button("Submit Dream Description"):
    if user_input:
        # Language Detection and Translation to English
        detected_lang = translator.detect(user_input).lang

        translated_input = translator.translate(user_input, src=detected_lang, dest='en').text

        input_vector = tfidf.transform([translated_input]).toarray()
        prediction = svm_model.predict(input_vector)
        predicted_label = label_encoder.inverse_transform(prediction)

        st.write(f"**Dream Interpretation (in English):** *{predicted_label[0]}*")
        
        translated_interpretation = translator.translate(predicted_label[0], src='en', dest=detected_lang).text
        st.write(f"**Dream Interpretation (in Original Language - {detected_lang}):** *{translated_interpretation}*")

        # Store the interpretation for read aloud
        st.session_state.translated_interpretation = translated_interpretation
    else:
        st.warning("Please enter or record your dream description before submitting.")

# Read Aloud Functionality
if 'translated_interpretation' in st.session_state:
    if st.button("Read Aloud"):
        tts = gTTS(st.session_state.translated_interpretation, lang=detected_lang)
        audio_file_path = "interpretation.mp3"
        tts.save(audio_file_path)
        st.audio(audio_file_path, format='audio/mp3')
        os.remove(audio_file_path)  # Clean up the audio file after playing
