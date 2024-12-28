# SMS-Spam-Detection-System-Using-NLP

This project focuses on creating a spam detection system for SMS messages using deep learning techniques in TensorFlow2. Three different architectures—Dense Network, LSTM, and Bi-LSTM—are employed to build the spam detection model. The models' accuracies are compared and evaluated to determine the best one.

Dataset Link: SMS Spam Collection The dataset from the UCI Machine Learning Repository contains 5,574 SMS messages, with 4,827 labeled as ham (non-spam) and 747 as spam. The dataset is divided into 4,000 messages for training and 1,574 messages for testing.

Steps Involved:

Load and Explore the Data: Load the dataset into a Pandas DataFrame and analyze the distribution of ham and spam messages.

Prepare Train-Test Data: Tokenize the messages and one-hot encode their labels. Split the dataset into training and testing sets in an 80:20 ratio.

Train the Spam Detection Model: Train the three models—Dense Network, LSTM, and Bi-LSTM—using the training dataset. Evaluate the models using the validation dataset.

Compare and Select the Final Model: Compare the models' accuracies and select the best-performing one.

Use the Final Trained Classifier to Classify New Messages: Use the final model to classify new messages as ham or spam.

Usage:

Clone the repository: git clone https://github.com/username/sms-spam-detection.git

Install the required packages:

bash
cd sms-spam-detection
pip install -r requirements.txt
Download the dataset from the UCI Machine Learning Repository and place it in the data directory:

bash
mkdir data
wget https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection -O data/spam.csv
Run the .pynb script to train the models and select the best one.

Run the Streamlit app: streamlit run app.py

Accuracy of the Models:

Dense Network: 98.5%

SVM: 97.6%

Bi-LSTM: 98.8%

LSTM: 98.6%

Streamlit App: A Streamlit app has been developed to demonstrate the final model's functionality. The app accepts a message as input and predicts whether it is ham or spam. Access the app here.

Conclusion: In this project, a spam detection system for SMS messages was built using deep learning techniques in TensorFlow2. Three different architectures were used, and the Bi-LSTM model achieved the highest accuracy of 98.8%. The final model has been deployed as a Streamlit app to showcase its functionality.
