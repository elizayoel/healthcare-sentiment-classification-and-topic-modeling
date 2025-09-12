import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import re
import emoji
import contractions
import string
import nltk
import io
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from st_aggrid import AgGrid, GridOptionsBuilder

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(layout="wide")

# ========================================CONSTANTS=========================================
MAX_LEN = 215
lemmatizer = WordNetLemmatizer()

# ==============================TEXT PREPROCESSING FUNCTIONS==============================
def keep_translated_text(text):
    if isinstance(text, str):
        parts = re.split(r'\(Translated by Google\)', text)
        if len(parts) > 1:
            return parts[1].split("(Original)")[0].strip()
    return text

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = keep_translated_text(text)
    text = contractions.fix(text)
    text = emoji.demojize(text, language='en')
    text = re.sub(r'(:[^:\s]+?:)', r' \1 ', text)
    text = text.lower()
    sentences = sent_tokenize(text)
    tokenized = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word not in string.punctuation]
        words = [word for word in words if not re.fullmatch(r'[^\w\s]+', word)]
        tokenized.append(words)
    pos_tagged = [pos_tag(sentence) for sentence in tokenized]
    lemmatized = [
        [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in sentence]
        for sentence in pos_tagged
    ]
    cleaned = [s for s in lemmatized if s]
    return ' '.join([word for sentence in cleaned for word in sentence])

# ==================================LOAD MODEL AND TOKENIZER==================================
@st.cache_resource
def load_sentiment_model():
    try:
        with open("tokenizer1.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        model = load_model("tuned_bilstm.keras")
        return tokenizer, model
    except FileNotFoundError:
        st.error("🚨 Critical Error: Model or tokenizer file not found.")
        st.error("Please ensure 'tokenizer1.pkl' and 'tuned_bilstm.keras' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        st.stop()

tokenizer, model = load_sentiment_model()

# =============================== LIME EXPLANATION FUNCTION =================================
def generate_and_display_lime(text_to_explain, predicted_label):
    """Generates and displays the LIME explanation chart and caption."""
    with st.spinner("Explaining with LIME..."):
        class_names = ['Negative', 'Positive']
        def predict_prob(texts):
            cleaned_batch = [preprocess(t) for t in texts]
            seqs = tokenizer.texts_to_sequences(cleaned_batch)
            pads = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
            preds = model.predict(pads)
            return np.hstack([1 - preds, preds])

        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text_to_explain, predict_prob, num_features=10)
        
        exp_list = exp.as_list()
        exp_df = pd.DataFrame(exp_list, columns=['feature', 'weight'])
        
        fig = px.bar(
            exp_df,
            x='weight',
            y='feature',
            orientation='h',
            labels={'feature': 'Word', 'weight': 'Contribution Weight'},
            color='weight',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            title={
                'text': f"LIME-based Word Contributions to '{predicted_label}' Prediction",
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            margin=dict(t=80, b=40),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        explanation_text = "🔴 Left: Contributes to 'Negative' Prediction &nbsp; | &nbsp; 🟢 Right: Contributes to 'Positive' Prediction"
        st.markdown(
            f"<div style='text-align: center; color: #555; font-size: 14px; margin-bottom: 20px;'>{explanation_text}</div>",
            unsafe_allow_html=True
        )

#======================================= MAIN INTERFACE ========================================
st.title("🚑 Healthcare Sentiment Classification with BiLSTM")
st.markdown(
    """
    <div style="background-color:#91C8E4; color:black; padding:16px; border-radius:10px;">
        <h4>💡 What is BiLSTM?</h4>
        <p>
        A <strong>BiLSTM</strong> (Bidirectional Long Short-Term Memory) is a type of deep learning model that reads text in both directions (forward and backward).<br>
        It improves upon regular LSTM, which can remember important information across long sequences.<br>
        The bidirectional design helps the model learn from both past and future words, allowing for better understanding of each word’s meaning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown("We will classify input reviews as **Positive** or **Negative**, and explain predictions using **LIME**.")

input_mode = st.radio("Choose input mode:", ["Single Text", "CSV Upload"], horizontal=True)

if input_mode == "Single Text":
    text_input = st.text_area("✍️ Enter a review to analyze:", height=150)
    if st.button("Predict Sentiment"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = preprocess(text_input)
            if not cleaned:
                st.warning("⚠️ The input text was empty after preprocessing. Please enter meaningful text.")
            else:
                st.markdown("---")
                st.markdown(
                    f"""
                    <div style="
                        background-color: #91c8e4;
                        color: black;
                        padding: 16px;
                        border-radius: 10px;
                    ">
                        <strong>Cleaned Text:</strong><br>{cleaned}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
                prob = model.predict(padded)[0][0]
                label = "Positive" if prob >= 0.5 else "Negative"

                st.markdown("---")
                st.markdown(f"### 🧾 Prediction: **{label}**")
                st.markdown(f"**Score:** `{prob:.4f}`")

                st.markdown("ℹ️ The probability of the review being **positive**.<br>Scores near **1.0** are positive; scores near **0.0** are negative.", unsafe_allow_html = True)
                
                st.markdown("---")
                st.markdown("#### Confidence Meter")

                confidence = float(prob) if label == "Positive" else float(1 - prob)
                confidence_percent = confidence * 100
                
                if confidence_percent >= 95: level = "Extremely Confident"
                elif confidence_percent >= 80: level = "Very Confident"
                elif confidence_percent >= 60: level = "Fairly Confident"
                else: level = "Less Certain (Ambiguous)"

                st.progress(confidence)
                st.markdown(f"The model is **{confidence_percent:.2f}%** confident that this review is **{label}**.<br>Confidence Level: **{level}**", unsafe_allow_html=True)
                st.markdown("---")

                with st.expander("💡 See how the model made this prediction"):
                    generate_and_display_lime(text_input, label)
                
                with st.container(border=True):
                    st.info("Explore what people are actually talking about using our **topic analysis** page!", icon="🔬")
                    if st.button("**Try Topic Analysis**"):
                        st.switch_page("pages/2_Topic_Analysis.py")
                

else: #==============================================CSV UPLOAD=======================
    
    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
        else:
            review_col = st.selectbox("Select the column containing review text:", df.columns)

            if st.button("Run Sentiment Analysis"):
                if pd.api.types.is_numeric_dtype(df[review_col]):
                    st.warning(f"⚠️ The selected column '{review_col}' appears to be numeric. Please select a text column.")
                elif df[review_col].dropna().empty:
                    st.warning(f"⚠️ The selected column '{review_col}' is empty or contains only null values.")
                else:
                    with st.spinner("Processing..."):
                        df["cleaned"] = df[review_col].astype(str).apply(preprocess)
                        seqs = tokenizer.texts_to_sequences(df["cleaned"])
                        padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
                        probs = model.predict(padded).flatten()
                        df["Sentiment"] = np.where(probs >= 0.5, "Positive", "Negative")
                        df["Score"] = probs.round(4)
                        st.session_state.df = df
                        st.session_state.review_col = review_col
                    st.success("Prediction complete!")
            
            if "df" in st.session_state:
                df = st.session_state.df
                review_col = st.session_state.review_col
                pos_count = (df["Sentiment"] == "Positive").sum()
                neg_count = (df["Sentiment"] == "Negative").sum()
                total = len(df)

                st.markdown("### 📊 Sentiment Summary")
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.write(f"🟢😊 **Positive:** {pos_count} ({(pos_count/total)*100:.2f}%)")
                        st.write(f"🔴☹️ **Negative:** {neg_count} ({(neg_count/total)*100:.2f}%)")
                        st.markdown("---")
                        if neg_count > pos_count:
                            st.warning(
                                f"🔍 **Action Needed:** The majority of reviews ({neg_count}) are negative. "
                                f"Please examine the feedback carefully to identify recurring concerns such as delays, poor communication, or unsatisfactory service. These areas should be prioritised for improvement."
                            )
                        else:
                            st.success(
                                f"✅ **Good Standing:** Most reviews **({pos_count})** reflect a **positive experience**. "
                                f"Focus on recognising common strengths like staff professionalism, facility cleanliness, or treatment quality. These positive aspects should be preserved and reinforced."
                            )
                        
                with col2:
                    chart_df = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Count": [pos_count, neg_count]})
                    fig_pie = px.pie(chart_df, names="Sentiment", values="Count", color="Sentiment",
                                color_discrete_map={"Positive": "#8cd47e", "Negative": "#e15858"},
                                title="Sentiment Distribution")
                    fig_pie.update_layout(title_x=0.3, margin=dict(t=60, b=1, l=10, r=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")
                st.markdown("""
                - **Original Review**: The review text exactly as it appears in your uploaded file.
                - **Cleaned Text**: The processed version of the review after cleaning that is fed to the model.
                - **Sentiment**: The final prediction from the model, categorized as 'Positive' or 'Negative'.
                - **Score**: The probability of the review being positive. Scores near **1.0** are positive; scores near **0.0** are negative.
                """)

                sentiment_filter = st.selectbox("Filter by Sentiment", options=["All", "Positive", "Negative"])
                
                filtered_df = df if sentiment_filter == "All" else df[df["Sentiment"] == sentiment_filter]
                
                st.info("💡 **Curious about a specific review?** Select its checkbox in the table below to generate an explanation.", icon="✅")

                gb = GridOptionsBuilder.from_dataframe(filtered_df[[review_col, 'cleaned', 'Sentiment', 'Score']])
                gb.configure_column(review_col, headerName="Original Review", width=350, wrapText=True, autoHeight=True)
                gb.configure_column("cleaned", headerName="Cleaned Text", width=350, wrapText=True, autoHeight=True)
                gb.configure_column("Sentiment", width=60)
                gb.configure_column("Score", width=60)
                gb.configure_selection('single', use_checkbox=True)
                grid_options = gb.build()

                grid_response = AgGrid(filtered_df, gridOptions=grid_options, theme="streamlit",
                                       fit_columns_on_grid_load=True, allow_unsafe_jscode=True,
                                       enable_enterprise_modules=False, update_mode="SELECTION_CHANGED")

                st.download_button("Download classification results as CSV", df.to_csv(index=False).encode("utf-8"),
                                  "sentiment_predictions.csv", "text/csv")

                selected_rows = grid_response['selected_rows']
                if selected_rows is not None and not selected_rows.empty:
                    selected_row_data = selected_rows.iloc[0]
                    selected_review = selected_row_data[review_col]
                    predicted_label = selected_row_data['Sentiment']
                    with st.expander("📖 View Full Review & LIME Explanation", expanded=True):
                        st.markdown("---")
                        st.markdown(f"**Sentiment:** {selected_row_data['Sentiment']} | **Score:** `{selected_row_data['Score']}`")
                        st.markdown(f"📝 **Review:**\n> {selected_review}")
                        st.markdown("### 👨🏻‍🏫 LIME Explanation")
                        generate_and_display_lime(selected_review, predicted_label)
                st.markdown("---")
                with st.container(border=True):
                    st.info("Explore what people are actually talking about using our **topic analysis** page!.", icon="🔬")
                    if st.button("**Try Topic Analysis**"):
                        st.switch_page("pages/2_Topic_Analysis.py")