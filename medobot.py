import streamlit as st
import pytesseract
from PIL import Image
import google.generativeai as genai
import cv2
import numpy as np
import speech_recognition as sr  # Importing speech recognition

# Set page configuration with title, icon, and layout
st.set_page_config(page_title="Medicine Chatbot", page_icon="💊", layout="wide")

# Configure the Gemini API key
GEMINI_API_KEY = "AIzaSyD7BFUmBU4TOe117vlD5KoG-6s8VBGc3g8"
genai.configure(api_key=GEMINI_API_KEY)

# OpenFDA API key (placeholder)
OPENFDA_API_KEY = "N5hhfDgmkdMg9hin1og1IcwvRTjXomqmplLm6bwb"

# Function to extract text from an image using Tesseract OCR
def extract_medicine_name(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to preprocess the image (grayscale and thresholding)
def preprocess_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(thresh_image)

# Function to get information from Gemini GPT using the medicine name
def get_medicine_info_from_gemini(medicine_name):
    chat_model = genai.GenerativeModel('gemini-pro')
    chat = chat_model.start_chat(history=[])
    query = f"What can you tell me about {medicine_name}?"
    response = chat.send_message(query)
    return response.text

# Function to fetch nearby hospitals using Gemini
def fetch_nearby_hospitals(location):
    chat_model = genai.GenerativeModel('gemini-pro')
    chat = chat_model.start_chat(history=[])
    query = f"Please suggest the best hospitals near {location}."
    response = chat.send_message(query)
    return response.text

# Function to recognize speech and search for medicine info
def recognize_speech_and_search():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        if text:
            st.write(f"You asked: **{text}**")  # Display the recognized text
            response = get_medicine_info_from_gemini(text)
            st.write(f"👨‍⚕️ MedoBot: {response}")  # Add assistant-like response
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand the audio. Please try again.")
    except sr.RequestError as e:
        st.error(f"Error with the Google Speech Recognition service: {e}")

# Sidebar title
st.sidebar.title("💊 Medicine Chatbot")

# Sidebar chatbot features
st.sidebar.markdown("""
### Chatbot Features:
- Get information about various medicines
- Ask for dosage, side effects, and usage
- Enter your location to find nearby hospitals
- Optimized for desktop, tablet, and mobile experiences.
""")
st.sidebar.markdown("---")  # Line break added here

# Location input in the sidebar
location = st.sidebar.text_input("Enter your location (city or address)", placeholder="Type your location...")

# Main page title and description for the medicine chatbot
st.markdown("""
<h1 style='display:inline;'>
    <span style='color:red;'>MEDO</span><span style='color:white;'>BOT</span>
</h1>
""", unsafe_allow_html=True)

# Input area for the user's medicine query
medicine_name = st.text_input("Enter medicine name or how do you feel", placeholder="Type medicine name...")

# Progress bar to simulate the response processing
progress_bar = st.progress(0)

# Trigger the response if there's input and the button is clicked
if st.button("💬 Get Medicine Information"):
    if medicine_name:
        progress_bar.progress(50)  # Simulate loading
        try:
            query = f"What can you tell me about {medicine_name}?"
            response = get_medicine_info_from_gemini(medicine_name)
            progress_bar.progress(100)  # Complete loading
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a medicine name!")

# Separator
st.markdown("---")

# Section for uploading an image to extract the medicine name
st.header("Upload Medicine Image")

# Medicine image upload
uploaded_image = st.file_uploader("Upload Medicine Image", type=["png", "jpg", "jpeg"])

# If the user uploads an image
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    image = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(image)
    extracted_medicine_name = extract_medicine_name(preprocessed_image)

    if st.button("🔍 Search Medicine Info from Image"):
        if extracted_medicine_name:
            progress_bar.progress(50)  # Simulate loading
            try:
                response = get_medicine_info_from_gemini(extracted_medicine_name)
                progress_bar.progress(100)  # Complete loading
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Could not extract a valid medicine name from the image.")

# Add button to find nearby hospitals
if st.sidebar.button("🏥 Find Nearby Hospitals"):
    if location:
        try:
            hospital_info = fetch_nearby_hospitals(location)
            st.sidebar.write("Nearby Hospitals:")
            st.sidebar.write(hospital_info)
        except Exception as e:
            st.error(f"An error occurred while fetching hospitals: {e}")
    else:
        st.warning("Please enter a location!")

# Button to use voice input for medicine name
if st.button("🗣️Talk to me"):
    recognize_speech_and_search()
