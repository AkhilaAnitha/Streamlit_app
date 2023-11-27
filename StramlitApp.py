import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the MNIST model for image classification
mnist_model = tf.keras.models.load_model(r"C:\Users\akhil\Desktop\Deep Learn\STREAMLIT\MODEL.SAVE\mnist_model.h5")

# Load the CNN model for tumor prediction
tumor_model = tf.keras.models.load_model(r"C:\Users\akhil\Desktop\Deep Learn\STREAMLIT\MODEL.SAVE\tumor_detection_model.h5")

# Load the tokenizer for spam detection
tokenizer = Tokenizer()
# Include code to fit the tokenizer to your text data, for instance:
# tokenizer.fit_on_texts(your_text_data)
# Uncomment and replace 'your_text_data' with the actual text data

# Load the trained RNN model for spam detection
rnn_model = tf.keras.models.load_model(r"C:\Users\akhil\Desktop\Deep Learn\STREAMLIT\MODEL.SAVE\spam_model.h5")

# Load the trained LSTM model for spam detection
lstm_model = tf.keras.models.load_model(r"C:\Users\akhil\Desktop\Deep Learn\STREAMLIT\MODEL.SAVE\sentimental_model.h5")

st.title('MultiModelysis App')

choice = st.sidebar.selectbox('Select Analysis Category', ('Image Categorization', 'Text Examination'))

if choice == 'Image Categorization':
    st.subheader('Image Categorization')

    # Choose Image Task in the sidebar
    imageChoice = st.sidebar.selectbox('Select Image Assignment', ('Handwritten Digits Recognition', 'Tumor Prediction'))

    if imageChoice == 'Handwritten Digits Recognition':
        st.subheader('Handwritten Digits Recognition')
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            img = img.reshape(1, 28, 28, 1)

            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            prediction = mnist_model.predict(img)
            st.write(f"Prediction: {np.argmax(prediction)}")

    elif imageChioce == 'Tumor Prediction':
        st.subheader('Tumor Prediction')
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img = cv2.resize(img_array, (128, 128))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            prediction = tumor_model.predict(img)
            if prediction > 0.5:
                st.write("Prediction: Tumor Detected")
            else:
                st.write("Prediction: No Tumor Detected")

elif choice == 'Text Examination':
    st.subheader('Text Examination')

    # Choose Text Task in the sidebar
    textChoice = st.sidebar.selectbox('Select Text Analysis Task', ('Spam Detection-RNN', 'Sentiment Analysis'))

    # Move text_input declaration outside of the if block
    text_input = st.text_area("Enter text here")

    if textChoice == 'Spam Detection-RNN':
        st.subheader('Spam Detection-RNN')

        if st.button('Detect Spam'):
            # Tokenize and pad the input text
            sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = pad_sequences(sequence, maxlen=50)

            # Make predictions using rnn_model
            prediction = rnn_model.predict(padded_sequence)

            # Display prediction result
            st.write("Your Predicted Result is: Spam" if prediction > 0.5 else "Your Predicted Result is: Not Spam")

    elif textChoice == 'Sentiment Analysis':
        st.subheader('Sentiment Analysis')

        if st.button('Assess Sentiment'):
            # Tokenize and pad the input text
            sequence = tokenizer.texts_to_sequences([text_input])
            maxlen_lstm = 500  # Use the correct maxlen for your LSTM model
            padded_sequence = pad_sequences(sequence, maxlen=maxlen_lstm)

            # Make predictions using lstm_model
            prediction = lstm_model.predict(padded_sequence)

            # Display prediction result
            st.write("Your Sentiment Analysis Result is: Positive" if prediction > 0.5 else "Your Sentiment Analysis Result is: Negative")