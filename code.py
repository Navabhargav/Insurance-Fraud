import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("insurance_claims.csv")

# Drop irrelevant columns
df.drop(['policy_number', 'policy_bind_date', 'policy_state'], axis=1, inplace=True, errors='ignore')

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
categorical_cols = ['insured_sex', 'insured_education_level', 'insured_occupation', 'incident_state']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Text Cleaning & Preprocessing
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    return text

df['cleaned_claim_description'] = df['incident_description'].astype(str).apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(df['cleaned_claim_description']).toarray()

# Feature Selection
X_numeric = df.drop(['fraud_reported', 'incident_description', 'cleaned_claim_description'], axis=1)
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combine numerical and text-based features
X_final = np.hstack((X_numeric_scaled, X_text))

# Target Variable
y = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
