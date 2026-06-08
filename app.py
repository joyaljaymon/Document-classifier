import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Setup Page Configurations
st.set_page_config(page_title="Document Classifier", page_icon="📄", layout="centered")
st.title("📄 Intelligent Document Classifier")
st.write("Upload a text snippet or document content to classify its department/type.")

# 2. Mock Training Data (To keep it self-contained and fast)
@st.cache_resource
def train_model():
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
    return vectorizer, model

vectorizer, model = train_model()

# 3. User Interface Input
user_input = st.text_area("Paste your document text here:", height=200, 
                          placeholder="e.g., Please find attached the billing statement for services rendered...")

# 4. Prediction Logic
if st.button("Classify Document", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text before classifying.")
    else:
        # Transform and Predict
        processed_input = vectorizer.transform([user_input])
        prediction = model.predict(processed_input)[0]
        
        # Display Result
        st.success("### Classification Result")
        st.metric(label="Predicted Department / Type", value=prediction)

st.markdown("---")
st.caption("Built with Scikit-Learn & Streamlit")