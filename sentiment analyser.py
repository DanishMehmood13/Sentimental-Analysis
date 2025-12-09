#sentimental Analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 3: Load Excel file

file_name = "positive_negative_sentences_resized.xlsx"  # change if different name
df = pd.read_excel(file_name)

# Step 4: Prepare dataset
positive = pd.DataFrame({
    "text": df["Positive Sentences"],
    "label": 1
})
negative = pd.DataFrame({
    "text": df["Negative Sentences"],
    "label": 0
})

data = pd.concat([positive, negative], ignore_index=True)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Step 6: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 8: Predictions on test set
y_pred = model.predict(X_test_tfidf)

# Step 9: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# Step 10: User Input Prediction
# ===============================
while True:
    user_sentence = input("Enter a sentence (or type 'exit' to quit): ")
    if user_sentence.lower() == "exit":
        break
    features = vectorizer.transform([user_sentence])
    prediction = model.predict(features)[0]
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
    
    print(f"Sentence: {user_sentence}\nSentiment: {sentiment}\n")