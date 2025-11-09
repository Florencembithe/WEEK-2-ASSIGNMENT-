import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- 1. Load Data ---
# Assuming you have downloaded the 'Training.csv' and 'Testing.csv' files from Kaggle
# and uploaded them to your Colab environment or run in a local environment.
# Dataset: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

try:
    df_train = pd.read_csv('Training.csv')
    df_test = pd.read_csv('Testing.csv')
except FileNotFoundError:
    print("Error: Ensure 'Training.csv' and 'Testing.csv' are in the current directory.")
    # Exit or use a placeholder/smaller dataset if this is an issue
    # For this submission, assume files are loaded.

# --- 2. Data Preprocessing and Cleaning ---

# Drop the last empty/unnecessary column if it exists (common issue in this dataset)
if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop('Unnamed: 133', axis=1)

# The data is already in a good format: 1 (symptom present) or 0 (symptom absent)
# The 'prognosis' column is the target label.

# Separate features (X) and target (y)
X_train = df_train.drop('prognosis', axis=1)
y_train = df_train['prognosis']
X_test = df_test.drop('prognosis', axis=1)
y_test = df_test['prognosis']

# --- 3. Model Training (Random Forest Classifier) ---

# Initialize the model
# Using a few basic hyperparameters for a good start
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
print("Training the Random Forest Classifier...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 4. Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation Results ---")
print(f"Accuracy Score: **{accuracy:.4f}**")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# --- 5. Project Demo & Prediction Function ---

# Function to get user input symptoms (simulated)
def predict_disease(model, symptoms_list, all_symptoms):
    """Predicts a disease based on a list of symptoms."""
    
    # Create an empty input array (vector) for the 132 possible symptoms
    input_data = np.zeros(len(all_symptoms))
    
    # Set the columns (symptoms) to 1 if they are present
    for symptom in symptoms_list:
        try:
            # Find the index of the symptom in the full list of columns
            symptom_index = all_symptoms.columns.get_loc(symptom)
            input_data[symptom_index] = 1
        except KeyError:
            print(f"Warning: Symptom '{symptom}' not found in the trained features.")
            
    # Reshape the data for prediction (1 sample, 132 features)
    input_df = pd.DataFrame([input_data], columns=all_symptoms.columns)
    
    # Predict the disease
    predicted_disease = model.predict(input_df)[0]
    
    return predicted_disease

# --- Example of a Hypothetical Patient Input (Demo) ---

# Get the list of all possible symptoms from the training data
all_symptoms = X_train

# Define a set of symptoms for a demo case (e.g., for Fungal infection)
demo_symptoms = [
    'itching', 
    'skin_rash', 
    'nodal_skin_eruptions', 
    'dischromic _patches' 
]

print("\n--- Project Demo: Predicting Disease ---")
print(f"Patient reported symptoms: **{', '.join(demo_symptoms)}**")

# Get the prediction
prediction = predict_disease(model, demo_symptoms, all_symptoms)

print(f"\nPredicted Disease: **{prediction}**")

# --- End of Code ---
