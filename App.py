# Import libraries
import pandas as pd  # for handling data
import nltk
import string

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords (only once)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("data.csv")

# Text cleaning function
def preprocess(text):
    text = text.lower()  # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()  # split into words
    words = [word for word in words if word not in stopwords.words('english')]  # remove common words
    return " ".join(words)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess)

# Convert labels to numbers
data['label'] = data['label'].map({'real': 1, 'fake': 0})

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Convert text into numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Take user input
news = input("\nEnter news text:\n")

# Preprocess input
news_clean = preprocess(news)

# Convert input to vector
news_vec = vectorizer.transform([news_clean])

# Predict
prediction = model.predict(news_vec)

# Output result
if prediction[0] == 1:
    print("This news is REAL")
else:
    print("This news is FAKE")