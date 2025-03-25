import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load Data
df = pd.read_csv('mail_data.csv')
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Build Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train Model
pipeline.fit(X_train, y_train)

# Save Model
joblib.dump(pipeline, 'model.pkl')
print("Model trained and saved as 'model.pkl'")
