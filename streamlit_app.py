import streamlit as st
import joblib
from googletrans import Translator
from gtts import gTTS
import os
import smtplib
import speech_recognition as sr
from datetime import datetime
import uuid

# Load the trained model, label encoder, and TF-IDF vectorizer
svm_model = joblib.load('model/dream_interpretation_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Initialize the translator
translator = Translator()

# Function to save dream and interpretation to a session file
def save_dream_to_file(dream, interpretation):
    # Append the dream and interpretation to the session file
    with open(session_file_path, 'a') as f:  # Use 'a' mode to append
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"DREAM DESCRIPTION:\n{dream}\n")
        f.write(f"DREAM INTERPRETATION:\n{interpretation}\n\n")

# Streamlit UI Setup
st.set_page_config(page_title="Dream Interpretation System", page_icon="🌙")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('static/earth-planet.gif');
        background-size: cover;
    }
    .title, .sidebar-header {
        text-align: center;
        white-space: nowrap;
    }
    .subheader {
        text-align: left;  /* Right align text */
        white-space: nowrap;
    }
    .responsive-text {
        font-size: calc(1em + 1vw);  /* Responsive font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Title
st.markdown('<div class="title responsive-text">Dream Interpretation System 🌙</div>', unsafe_allow_html=True)
st.markdown('<div class="title responsive-text">Supported All Language as a Input</div>', unsafe_allow_html=True)
# #ASCII Art as a raw string literal
# ascii_art = r"""
# <pre style="text-align: center;">
#           ⠀        
#         </pre>    
# """

# # Display ASCII art using Markdown with the <pre> tag
# st.markdown(ascii_art, unsafe_allow_html=True)
st.markdown('<div class="title responsive-text">🌀 Dream 🌀⠀</div>', unsafe_allow_html=True)
st.markdown('<div class="title responsive-text">Model Trained By Ibn Sirin\'s Dictionary of Dreams and others resource also.</div>', unsafe_allow_html=True)

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
st.markdown('<div class="subheader responsive-text">Enter your dream description:</div>', unsafe_allow_html=True)
input_type = st.radio("Choose input type:", ("Text", "Voice"))

# Hidden text input for storing voice input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Create a unique session identifier
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generate a unique session ID

# Define session file path
session_file_path = f".hidden_folder/dream_interpretations_{st.session_state.session_id}.txt"
os.makedirs('.hidden_folder', exist_ok=True)  # Ensure the hidden folder exists

if input_type == "Text":
    st.session_state.user_input = st.text_area("Describe your dream here...", placeholder="Type your dream description here...")
else:
    st.warning("Voice input is not supported on Streamlit Cloud. Please use the text input option.")
    st.session_state.user_input = st.text_area("Describe your dream here...", placeholder="Type your dream description here...")

    # if st.button("Record"):
    #     recognizer = sr.Recognizer()
    #     with sr.Microphone() as source:
    #         st.write("Listening...")
    #         audio_data = recognizer.listen(source)
    #         st.write("Recognizing...")
    #         try:
    #             recognized_text = recognizer.recognize_google(audio_data)
    #             st.session_state.user_input = recognized_text  # Store recognized text in session state
    #             st.write(f"Recognized Text: {recognized_text}")
    #         except sr.UnknownValueError:
    #             st.write("Sorry, I could not understand your speech. Please try again.")
    #         except sr.RequestError:
    #             st.write("Could not request results from the speech service; check your network connection.")

# Submit Button
if st.button("Submit Dream Description"):
    user_input = st.session_state.user_input
    if user_input:
        detected_lang = translator.detect(user_input).lang
        translated_input = translator.translate(user_input, src=detected_lang, dest='en').text

        input_vector = tfidf.transform([translated_input]).toarray()
        prediction = svm_model.predict(input_vector)
        predicted_label = label_encoder.inverse_transform(prediction)

        st.write(f"**Dream Interpretation (in English):** *{predicted_label[0]}*")

        translated_interpretation = translator.translate(predicted_label[0], src='en', dest=detected_lang).text
        st.write(f"**Dream Interpretation (in Original Language - {detected_lang}):** *{translated_interpretation}*")

        st.session_state.translated_interpretation = translated_interpretation
        st.session_state.detected_lang = detected_lang

        # Save dream description and interpretation to the session file
        save_dream_to_file(user_input, translated_interpretation)

    else:
        st.warning("Please enter or record your dream description before submitting.")

# Read Aloud Functionality
if 'translated_interpretation' in st.session_state and 'detected_lang' in st.session_state:
    if st.button("Read Aloud"):
        tts = gTTS(st.session_state.translated_interpretation, lang=st.session_state.detected_lang)
        audio_file_path = "interpretation.mp3"
        tts.save(audio_file_path)
        st.audio(audio_file_path, format='audio/mp3')
        os.remove(audio_file_path)  # Clean up the audio file after playing
