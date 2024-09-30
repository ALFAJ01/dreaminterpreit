import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Step 1: Load the dataset
file_path = "DreamData_Islamick.csv"  
dream_data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Select relevant columns
dream_data = dream_data[['Dream_Description', 'Interpretation']]

# Drop rows with missing values
dream_data.dropna(subset=['Dream_Description', 'Interpretation'], inplace=True)

# Encode the 'Interpretation' labels
label_encoder = LabelEncoder()
dream_data['Label'] = label_encoder.fit_transform(dream_data['Interpretation'])

# Step 3: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=10000000)
X = tfidf.fit_transform(dream_data['Dream_Description']).toarray()

# Step 4: Train-Test Split
y = dream_data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Model Training using SVM with multiple epochs
svm_model = LinearSVC()
epochs = 7  # Number of training epochs

for epoch in range(epochs):
    svm_model.fit(X_train, y_train)
    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Step 6: Model Evaluation
y_pred = svm_model.predict(X_test)
print("Final Model Accuracy:", accuracy_score(y_test, y_pred))
unique_labels = list(set(y_test) | set(y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, labels=unique_labels, target_names=label_encoder.inverse_transform(unique_labels)))

# Step 7: Predicting on New Data
def predict_dream_interpretation(dream_text):
    """Function to predict dream interpretation category."""
    dream_vector = tfidf.transform([dream_text]).toarray()
    prediction = svm_model.predict(dream_vector)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Save model and encoders
os.makedirs('model', exist_ok=True)  # Create directory if it doesn't exist
joblib.dump(svm_model, 'model/dream_interpretation_model.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')

# Example Prediction Loop
while True:
    example_dream = input("Enter Your Dream : ")
    print("Predicted Interpretation:", predict_dream_interpretation(example_dream))
