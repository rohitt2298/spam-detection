
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Detector", page_icon="🛡️")

st.title("🛡️ Spam Email & SMS Detection")
st.write("Enter a message to detect spam.")

@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data = data.iloc[:, :2]
    data.columns = ["label", "message"]
    data = data.dropna()

    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data = data.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"],
        test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model()

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

message = st.text_area("Enter your message")

if st.button("Check Message"):
    if message.strip():
        result = model.predict([message])[0]
        if result == 1:
            st.error("⚠️ SPAM MESSAGE DETECTED!")
        else:
            st.success("✅ THIS IS NOT SPAM!")
    else:
        st.warning("Please enter a message.")
