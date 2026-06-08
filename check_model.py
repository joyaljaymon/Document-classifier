from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. This is the exact same ML logic from your app.py
training_texts = [
    "Invoice invoice due payment balance total amount bill cash receipt",
    "Contract agreement terms parties nondisclosure legal liability breach",
    "Resume experience skills education employment job history curriculum vitae",
    "Medical health clinical patient report prescription records doctor diagnosis"
]
labels = ["Billing / Invoice", "Legal / Contract", "Employment / Resume", "Healthcare / Medical"]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(training_texts)
model = MultinomialNB()
model.fit(X, labels)

# 2. Test Input (Change this text to test different categories!)
test_text = "Please find attached the bill for last month's consulting services. Total amount due must be paid."

# 3. Predict
processed_input = vectorizer.transform([test_text])
prediction = model.predict(processed_input)[0]

print("\n" + "="*50)
print(f"📄 Testing Document Classifier...")
print(f"📥 Input Text: '{test_text}'")
print(f"🔮 Predicted Category: {prediction}")
print("="*50 + "\n")