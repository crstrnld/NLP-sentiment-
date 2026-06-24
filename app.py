import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Sentiment Analysis NLP (Positif/Negatif)")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV (kolom: text, label)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview Dataset:", df.head())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    # Evaluasi
    st.subheader("Evaluasi Model")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    st.pyplot(fig)

    # Prediksi teks baru
    st.subheader("Coba Prediksi Sentimen")
    user_input = st.text_area("Masukkan teks review:")
    if user_input:
        user_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_tfidf)[0]
        st.write("Sentimen Prediksi:", prediction)
