import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text
from src.model import train_model
from src.utils import evaluate_model

# Load dataset
df = pd.read_csv("data/train.csv")

# Clean text data
df['clean_text'] = df['text'].apply(clean_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model("logistic", X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)
