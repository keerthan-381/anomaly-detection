import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the saved Logistic Regression model
logistic_model_filename = r"logistic_regression_model.pkl"
loaded_logistic_model = joblib.load(logistic_model_filename)

# Load the saved KNN model
KNN_model_filename = r"KNN_model.pkl"
loaded_KNN_model = joblib.load(KNN_model_filename)

# Load the saved Decision Tree model
decision_tree_model_filename = r"DecisionTree_model.pkl"
loaded_decision_tree_model = joblib.load(decision_tree_model_filename)

# Define the selected features
selected_features = [
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'count',
    'same_srv_rate',
    'diff_srv_rate',
    'dst_host_srv_count',
    'dst_host_same_srv_rate'
]

# Initialize label encoders for categorical features
label_encoders = {}  # Dictionary to store label encoders for categorical features
categorical_columns = ['protocol_type', 'service', 'flag']

# Create a Streamlit app
st.title("Network Intrusion Detection")

# Choose the model
selected_model = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Decision Tree"])

# Collect user input for the selected features
user_input = {}
for feature_name in selected_features:
    if feature_name in categorical_columns:  # Check if the feature is categorical
        # Load the label encoder for the categorical feature if it exists, or create a new one
        if feature_name in label_encoders:
            label_encoder = label_encoders[feature_name]
        else:
            label_encoder = LabelEncoder()
            label_encoders[feature_name] = label_encoder

        # Fit the label encoder on the entire training data (You need your training data for this)
        # For demonstration purposes, let's assume you have a list of possible values for each categorical feature.
        # In practice, you should use your actual training data.
        possible_values = {
            'protocol_type': ['tcp', 'udp', 'icmp'],
            'service': ['ftp_data', 'http', 'telnet', 'other', 'private', 'http', 'remote_job', 'name', 'netbios_ns', 'eco_i', 'mtp', 'domain_u', 'supdup', 'uucp_path'],  # Add more service values from your training data
            'flag': ['SF', 'S0', 'REJ']  # Add more flag values from your training data
        }
        label_encoder.fit(possible_values[feature_name])

        user_input[feature_name] = st.text_input(f"Enter {feature_name}:", value=possible_values[feature_name][0])
    else:
        user_input[feature_name] = st.number_input(f"Enter {feature_name}:")

# Make predictions using the selected model
if selected_model == "Logistic Regression":
    if st.button("Predict"):
        input_data = []
        for feature_name in selected_features:
            feature_value = user_input[feature_name]
            if feature_name in categorical_columns:
                # Transform the categorical feature using the corresponding label encoder
                label_encoder = label_encoders[feature_name]
                feature_value = label_encoder.transform([feature_value])[0]
            input_data.append(feature_value)

        input_data = np.array(input_data).reshape(1, -1)  # Reshape to match the model's input shape
        prediction = loaded_logistic_model.predict(input_data)

        # Display the prediction
        if prediction[0] == 0:
            st.write("The model predicts the class as: normal")
        elif prediction[0] == 1:
            st.write("The model predicts the class as: anomaly")
        else:
            st.write("Invalid prediction value")

elif selected_model == "KNN":
    if st.button("Predict"):
        input_data = []
        for feature_name in selected_features:
            feature_value = user_input[feature_name]
            if feature_name in categorical_columns:
                # Transform the categorical feature using the corresponding label encoder
                label_encoder = label_encoders[feature_name]
                feature_value = label_encoder.transform([feature_value])[0]
            input_data.append(feature_value)

        input_data = np.array(input_data).reshape(1, -1)  # Reshape to match the model's input shape
        prediction = loaded_KNN_model.predict(input_data)

        # Display the prediction
        if prediction[0] == 0:
            st.write("The model predicts the class as: normal")
        elif prediction[0] == 1:
            st.write("The model predicts the class as: anomaly")
        else:
            st.write("Invalid prediction value")

elif selected_model == "Decision Tree":
    if st.button("Predict"):
        input_data = []
        for feature_name in selected_features:
            feature_value = user_input[feature_name]
            if feature_name in categorical_columns:
                # Transform the categorical feature using the corresponding label encoder
                label_encoder = label_encoders[feature_name]
                feature_value = label_encoder.transform([feature_value])[0]
            input_data.append(feature_value)

        input_data = np.array(input_data).reshape(1, -1)  # Reshape to match the model's input shape
        prediction = loaded_decision_tree_model.predict(input_data)

        # Display the prediction
        if prediction[0] == 0:
            st.write("The model predicts the class as: normal")
        elif prediction[0] == 1:
            st.write("The model predicts the class as: anomaly")
        else:
            st.write("Invalid prediction value")
