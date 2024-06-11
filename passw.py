from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load data and train model
try:
    df = pd.read_csv('pw2.csv', error_bad_lines=False)
    print("CSV loaded successfully.")
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")
    exit(1)

# Check if CSV has the correct columns
if 'password' not in df.columns or 'strength' not in df.columns:
    print("CSV does not contain the required columns: 'password' and 'strength'.")
    exit(1)

# Drop rows with missing values in 'password' or 'strength' columns
df.dropna(subset=['password', 'strength'], inplace=True)

X = df['password']
y = df['strength']

# Convert passwords to TF-IDF features
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'password_strength_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    password = [data['password']]
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('password_strength_model.pkl')
    password_vect = vectorizer.transform(password)
    prediction = model.predict(password_vect)
    return jsonify({'strength': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
