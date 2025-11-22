import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("dataset.csv")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

df["clean_text"] = df["text"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["is_malicious"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
#model = LogisticRegression()

#  Support Vector Machine (better accuracy)
model = SVC(kernel='linear', probability=True)

# Random Forest
# model = RandomForestClassifier(n_estimators=200)

#  XGBoost (best performance)
# model = XGBClassifier(eval_metric='logloss')

print("Training model...")
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(model, "prompt_detector_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")
