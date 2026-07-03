import re
import joblib
import pandas as pd
import gradio as gr
import tensorflow as tf

from pypdf import PdfReader
from docx import Document
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline


svm_model = joblib.load("models/svm_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")
ada_model = joblib.load("models/adaboost_model.pkl")

tfidf = joblib.load("models/tfidf_vectorizer.pkl")
tokenizer = joblib.load("models/tokenizer.pkl")

fnn_model = tf.keras.models.load_model("models/fnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")


llm_explainer = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

llm_writer = pipeline(
    "text-generation",
    model="distilgpt2"
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_file_text(file_path):
    if file_path is None:
        return ""

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif file_path.endswith(".docx"):
        document = Document(file_path)
        text = ""

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

        return text

    return ""


def text_stats(text):
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]

    word_count = len(words)
    sentence_lengths = [len(s.split()) for s in sentences]
    vocab_richness = len(set(words)) / word_count if word_count > 0 else 0

    stats = f"""
Word Count: {word_count}
Sentence Lengths: {sentence_lengths}
Vocabulary Richness: {round(vocab_richness, 3)}
"""

    return stats


def label_result(prediction):
    return "AI-generated" if prediction == 1 else "Human-written"


def predict_model(model_choice, clean):
    if model_choice == "SVM":
        prediction = svm_model.predict([clean])[0]
        score = svm_model.decision_function([clean])[0]
        confidence = min(abs(score) / 3, 1)

    elif model_choice == "Decision Tree":
        prediction = dt_model.predict([clean])[0]
        confidence = max(dt_model.predict_proba([clean])[0])

    elif model_choice == "AdaBoost":
        prediction = ada_model.predict([clean])[0]
        confidence = max(ada_model.predict_proba([clean])[0])

    elif model_choice == "FNN":
        vector = tfidf.transform([clean]).toarray()
        probability = fnn_model.predict(vector)[0][0]
        prediction = int(probability > 0.5)
        confidence = probability if prediction == 1 else 1 - probability

    elif model_choice == "LSTM":
        seq = tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=200)
        probability = lstm_model.predict(pad)[0][0]
        prediction = int(probability > 0.5)
        confidence = probability if prediction == 1 else 1 - probability

    else:
        seq = tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=200)
        probability = cnn_model.predict(pad)[0][0]
        prediction = int(probability > 0.5)
        confidence = probability if prediction == 1 else 1 - probability

    return prediction, confidence


def llm_prediction_explanation(text, result, confidence):
    short_text = text[:1000]

    prompt = f"""
Explain in 2 short sentences why this text may be classified as {result}.
Confidence score: {round(confidence * 100, 2)}%.
Text: {short_text}
Answer:
"""

    response = llm_explainer(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        return_full_text=False
    )[0]["generated_text"]

    return response.strip()


def llm_writing_analysis(text):
    short_text = text[:1000]

    prompt = f"""
Analyze this writing in 2 short sentences. Mention whether it sounds formal,
repetitive, natural, or human-like.
Text: {short_text}
Answer:
"""

    response = llm_writer(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        return_full_text=False
    )[0]["generated_text"]

    return response.strip()


def analyze_text(file_input, text_input, model_choice):
    file_text = extract_file_text(file_input)

    if file_text.strip() != "":
        text = file_text
    else:
        text = text_input

    if text.strip() == "":
        return (
            "Please upload a file or enter text.",
            "",
            "",
            pd.DataFrame(),
            "",
            "",
            ""
        )

    clean = clean_text(text)

    prediction, confidence = predict_model(model_choice, clean)
    result = label_result(prediction)

    stats = text_stats(text)

    rows = []

    for model in ["SVM", "Decision Tree", "AdaBoost", "FNN", "LSTM", "CNN"]:
        pred, conf = predict_model(model, clean)

        rows.append({
            "Model": model,
            "Prediction": label_result(pred),
            "Confidence": round(conf * 100, 2)
        })

    comparison_df = pd.DataFrame(rows)

    explanation = llm_prediction_explanation(
        text,
        result,
        confidence
    )

    writing_analysis = llm_writing_analysis(text)

    report = f"""
AI vs Human Text Detection Report

Selected Model: {model_choice}
Prediction: {result}
Confidence: {round(confidence * 100, 2)}%

Text Statistics:
{stats}

LLM Explanation:
{explanation}

LLM Writing Analysis:
{writing_analysis}
"""

    with open("analysis_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    return (
        result,
        str(round(confidence * 100, 2)) + "%",
        stats,
        comparison_df,
        explanation,
        writing_analysis,
        "analysis_report.txt"
    )


app = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.File(
            label="Upload PDF or Word Document",
            file_types=[".pdf", ".docx"],
            type="filepath"
        ),
        gr.Textbox(
            label="Or Enter Text",
            lines=6
        ),
        gr.Dropdown(
            choices=["SVM", "Decision Tree", "AdaBoost", "FNN", "LSTM", "CNN"],
            value="Decision Tree",
            label="Choose Detection Model"
        )
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Text Statistics"),
        gr.Dataframe(label="Model Comparison"),
        gr.Textbox(label="LLM Explanation"),
        gr.Textbox(label="LLM Writing Analysis"),
        gr.File(label="Download Report")
    ],
    title="AI vs Human Text Detector",
    description="Upload a PDF or Word document, or paste text directly."
)

app.launch()