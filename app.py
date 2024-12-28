import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the LSTM model with relative path
def load_model_relative():
    model_path = os.path.join(os.path.dirname(__file__), 'Dense_Spam_Detection.h5')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None
    return load_model(model_path)

# Define the function to preprocess the user's input message
def preprocess_message(message):
    # Convert the message to lowercase
    message = message.lower()
    # Tokenize the message
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([message])
    # Convert the message to a sequence of integers
    sequence = tokenizer.texts_to_sequences([message])
    # Pad the sequence with zeros so that it has the same length as the sequences used to train the model
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=50)
    return padded_sequence

# Main function to interact with the user
def main():
    # Load the model
    model = load_model_relative()
    if model is None:
        return

    print("Spam Detector")
    print("----------------")
    
    # Ask the user to input a message
    message = input("Enter a message: ")

    # Preprocess the message and make a prediction
    if message:
        processed_message = preprocess_message(message)
        prediction = model.predict(processed_message)

        # Display the prediction
        if prediction > 0.5:
            print(f"This message is spam with a probability of {prediction[0][0] * 100:.2f}%.")
        else:
            print(f"This message is ham with a probability of {(1 - prediction[0][0]) * 100:.2f}%.")
    
if __name__ == '__main__':
    main()
