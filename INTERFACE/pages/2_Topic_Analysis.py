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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bertopic import BERTopic
import textwrap
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")

MODEL_PATH = "MODEL_REDUCED_OUTLIERS"
TOPIC_INFO_PATH = "INSPECTED_TOPIC.csv"
MIN_WORDS = 5
MAX_LEN = 215
lemmatizer = WordNetLemmatizer()


# ==================================PREPROCESSING==========================================
def keep_translated_text(text):
    if isinstance(text, str):
        parts = re.split(r'\(Translated by Google\)', text)
        if len(parts) > 1:
            return parts[1].split("(Original)")[0].strip()
    return text

def expand_cont(text):
    if isinstance(text, str) and text.strip():
        try:
            return contractions.fix(text)
        except IndexError:
            return text
    return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


#===================================FOR SENTIMENT!!!===============================
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
def preprocess_for_sentiment(text):
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


# ==================================LOAD MODELS AND DATA===================================
@st.cache_resource
def load_topic_models_and_data():
    try:
        topic_model = BERTopic.load(MODEL_PATH)
        df_topics = pd.read_csv(TOPIC_INFO_PATH)
        df_topics.set_index('topic_id', inplace=True)
        return topic_model, df_topics
    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        st.error(f"Please ensure '{MODEL_PATH}/' and '{TOPIC_INFO_PATH}' are in the correct directory.")
        return None, None

@st.cache_data
def load_review_data():
    """Loads the large dataframe with all reviews and their topics."""
    try:
        df_reviews = pd.read_csv("FINAL_REDUCED_DATA.csv")
        return df_reviews
    except FileNotFoundError:
        st.error("🚨 Review data file not found. Please specify the correct filename in the `load_review_data` function.")
        return None

@st.cache_resource
def load_sentiment_model_and_tokenizer():
    try:
        with open("tokenizer1.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        model = load_model("tuned_bilstm.keras")
        return tokenizer, model
    except FileNotFoundError:
        st.error("🚨 Sentiment model or tokenizer file not found.")
        return None, None

#CALL FUNCTION
topic_model, df_topics = load_topic_models_and_data()
df_reviews = load_review_data()
sentiment_tokenizer, sentiment_model = load_sentiment_model_and_tokenizer()
# ==========================================INTERFACE==========================================

# ----------------------------------------GENERAL INFO-----------------------------------------
st.title("🩺 Healthcare Review Topic Explorer with BERTopic")
info_box_html = textwrap.dedent("""
    <div style="background-color:#E6F3FF; color:black; padding:16px; border-radius:10px;">
        <h4>💡 What is BERTopic?</h4>
        <p>
        <strong>BERTopic</strong> is a powerful topic modeling technique that leverages transformer embeddings and clustering to discover topics in text data. It groups semantically similar documents to identify coherent themes without needing a predefined number of topics.<br><br>
        This explorer uses a model trained on healthcare reviews to classify new text into predefined topics like <strong> "Service Quality" </strong>, <strong> "Attitude" </strong>,  <strong> "Scheduling"</strong>,  and many more!
        </p>
    </div>
""")
st.markdown(info_box_html, unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------INPUT OPTION------------------------------------------
input_mode = st.radio("Choose your input mode:", ["Single Text Analysis", "Batch CSV Upload"], horizontal=True)

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.topic_id = None
    st.session_state.topic_info = None
    st.session_state.sentiment_label = None

if "batch_results_df" not in st.session_state:
    st.session_state.batch_results_df = None

# =====================================SINGLE TEXT MODE======================================
if input_mode == "Single Text Analysis":

    def reset_prediction_state():
        st.session_state.prediction_made = False
        st.session_state.topic_id = None
        st.session_state.topic_info = None
        st.session_state.sentiment_label = None
        
    st.subheader("Analyze a Single Review")
    text_input = st.text_area(
        "✍️ Enter review text below:",
        height=150,
        on_change=reset_prediction_state
    )

    if st.button("Find Topic"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        elif len(text_input.strip().split()) < MIN_WORDS:
            st.warning(f"Please enter at least {MIN_WORDS} words for a reliable prediction.")
        elif topic_model is None or df_topics is None or df_reviews is None:
            st.error("Model or data not loaded. Cannot perform prediction.")
        else:
            with st.spinner("Analyzing Topic and Sentiment..."):
                translated_text = keep_translated_text(text_input)
                expanded_text = expand_cont(translated_text)
                clean_text_for_topic = preprocess_text(expanded_text)
                predicted_id, _ = topic_model.transform(clean_text_for_topic)
                topic_id = predicted_id[0]
                
                sentiment_label = "N/A"
                if sentiment_model and sentiment_tokenizer:
                    clean_text_for_sentiment = preprocess_for_sentiment(text_input)
                    if clean_text_for_sentiment:
                        seq = sentiment_tokenizer.texts_to_sequences([clean_text_for_sentiment])
                        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
                        prob = sentiment_model.predict(padded)[0][0]
                        sentiment_label = "Positive 😊" if prob >= 0.5 else "Negative 😠"
                
                st.session_state.topic_id = predicted_id[0]
                st.session_state.sentiment_label = sentiment_label      
                
                if st.session_state.topic_id != -1:
                    st.session_state.topic_info = df_topics.loc[st.session_state.topic_id]
                
                st.session_state.prediction_made = True

    if st.session_state.prediction_made:
        topic_id = st.session_state.topic_id
        topic_info = st.session_state.topic_info
        sentiment_label = st.session_state.sentiment_label
        
        st.markdown("### 🧾 Analysis Result")
        
        if topic_id == -1:
            st.info("Sorry, this review could not be assigned to a specific topic (classified as an Outlier).")
            st.markdown(f"#### Sentiment: **{sentiment_label}**")
        else:
            with st.expander("View Prediction Details", expanded=True):
                st.markdown(f"""
                - **Sentiment:** {sentiment_label}
                - **Main Topic:** {topic_info['name']}
                - **General Category:** {topic_info['general_topic']}
                """) 
            st.info(
                "**How are these topics defined?**\n\n"
                "Topic names are primarily identified using top keywords extracted by the BERTopic model, "
                "strengthened with manual human judgement on topic clusters.\n"
                "For example: *'staff'*, *'rude'* → *'Attitude'*.\n\n"
                "💡 **Want to know what a review is really about?** We will show you the top keywords that form this topic."
            )
            st.markdown("---")
            
            try:
                keyword_str = topic_info['top10keywords']
                score_str = topic_info['ctfidf_score']
                keywords = [kw.strip() for kw in keyword_str.split(',')]
                scores = [float(s.strip()) for s in score_str.split(',')]
                df_chart = pd.DataFrame({'Keyword': keywords, 'Score': scores}).sort_values(by='Score', ascending=True)
                #BAR CHART
                fig = px.bar(
                    df_chart,
                    x='Score',
                    y='Keyword',
                    orientation='h',
                    color='Score',  
                    color_continuous_scale='Sunset' 
                )

                fig.update_layout(
                    title={
                        'text': f"Keyword Importance for '{topic_info['name']}'",
                        'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
                    },
                    yaxis_title="Keyword",
                    xaxis_title="c-TF-IDF Score",
                    margin=dict(t=80, b=40) 
                )
    
                st.plotly_chart(fig, use_container_width=True)
    
            except Exception as e:
                st.warning("Could not generate the keyword importance chart.")
                st.markdown("**Keywords that define this topic:**")
                st.info(f"{topic_info['top10keywords']}")
            
            st.markdown("---")
            with st.expander("Show Example Reviews Similar to this Topic"):
                if df_reviews is not None:
                    topic_column = 'TOPIC_ID'
                    text_column = 'original_text'
                    if topic_column in df_reviews.columns and text_column in df_reviews.columns:
                        similar_reviews = df_reviews[df_reviews[topic_column] == topic_id]
                        if not similar_reviews.empty:
                            num_samples = min(10, len(similar_reviews))
                            for review_text in similar_reviews[text_column].sample(num_samples):
                                st.info(review_text)
                        else:
                            st.warning("No example reviews found for this topic.")
                    else:
                        st.error(f"Error: Review CSV is missing required columns '{topic_column}' or '{text_column}'.")                
                
            with st.container(border=True):
                st.info("Explore what all the topics mean in detail.", icon="📚")
                if st.button("See Topic Information"):
                    st.switch_page("pages/3_Topic_Information.py")
#======================================================BATCH CSV========================================================
elif input_mode == "Batch CSV Upload":
    st.subheader("Analyze a Batch of Reviews from a CSV File")
    
    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_peek = pd.read_csv(uploaded_file, nrows=0)
            review_col = st.selectbox(
                "Select the column containing review text:",
                df_peek.columns,
                index=None,
                placeholder="Choose a column..."
            )

            if review_col:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, dtype={review_col: str})
                
                if st.button("Analyze Topics for this CSV"):
                    if df_upload[review_col].dropna().empty:
                        st.error(f"The selected column '{review_col}' is empty. Please choose another one.")
                    else:
                        with st.spinner(f"Analyzing {len(df_upload)} reviews... This may take a moment."):
                            docs = df_upload[review_col].fillna('').astype(str).tolist()
                            processed_docs_topic = [preprocess_text(expand_cont(keep_translated_text(doc))) for doc in docs]
                            predicted_ids, _ = topic_model.transform(processed_docs_topic)
                            
                            df_upload['topic_id'] = predicted_ids
                            df_upload['topic_name'] = [df_topics.loc[tid]['name'] if tid != -1 else 'Outlier' for tid in predicted_ids]
                            df_upload['general_topic'] = [df_topics.loc[tid]['general_topic'] if tid != -1 else 'Outlier' for tid in predicted_ids]

                            if sentiment_model and sentiment_tokenizer:
                                processed_docs_sentiment = [preprocess_for_sentiment(doc) for doc in docs]
                                seqs = sentiment_tokenizer.texts_to_sequences(processed_docs_sentiment)
                                padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
                                probs = sentiment_model.predict(padded).flatten()
                                df_upload['Sentiment'] = np.where(probs >= 0.5, "Positive", "Negative")
                            
                            st.session_state.batch_results_df = df_upload
                            st.session_state.review_col = review_col
                            st.success("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred while reading or processing the file: {e}")

    if st.session_state.batch_results_df is not None:
        results_df = st.session_state.batch_results_df
        review_col = st.session_state.get('review_col', results_df.columns[0])
        
        st.markdown("---")
        st.markdown("### 📊 Overall Summary")
        
        #TOPIC DISTRIBUTION BAR CHART
        topic_counts = results_df['topic_name'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        fig_topic = px.bar(
            topic_counts, x='Count', y='Topic', orientation='h',
            title='Topic Distribution'
        )
        fig_topic.update_layout(title_x=0.5, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_topic, use_container_width=True)

        if 'Sentiment' in results_df.columns:
            st.markdown("#### Overall Sentiment Distribution")
            with st.container(border=True):
                sentiment_counts = results_df['Sentiment'].value_counts()
                pos_count = sentiment_counts.get('Positive', 0)
                neg_count = sentiment_counts.get('Negative', 0)
                total_count = pos_count + neg_count

                if total_count > 0:
                    pos_percent = (pos_count / total_count) * 100
                    neg_percent = (neg_count / total_count) * 100
                    st.write(f"🟢 **Positive:** {pos_count} reviews ({pos_percent:.1f}%)")
                    st.write(f"🔴 **Negative:** {neg_count} reviews ({neg_percent:.1f}%)")
                else:
                    st.write("No sentiment data to display.")

        st.markdown("---")
        st.markdown("###  Sentiment Breakdown by Topic")

        unique_topics_for_viz = results_df[results_df['topic_id'] != -1]['topic_name'].unique()

        if len(unique_topics_for_viz) > 0:
            topic_to_visualize = st.selectbox(
                "Choose a topic to visualize its sentiment breakdown:",
                unique_topics_for_viz
            )

            if topic_to_visualize:
                topic_specific_df = results_df[results_df['topic_name'] == topic_to_visualize]
                sentiment_counts_topic = topic_specific_df['Sentiment'].value_counts().reset_index()
                sentiment_counts_topic.columns = ['Sentiment', 'Count']

                col1, col2 = st.columns([2, 2])

                with col1:
                    st.markdown(f"**Sentiment for '{topic_to_visualize}'**")
                    
                    fig_sentiment_by_topic = px.pie(
                        sentiment_counts_topic,
                        names='Sentiment',
                        values='Count',
                        color='Sentiment',
                        color_discrete_map={'Positive': '#8cd47e', 'Negative': '#e15858'}
                    )
                    st.plotly_chart(fig_sentiment_by_topic, use_container_width=True)

                with col2:
                    with st.container(border=True):
                        counts_dict = pd.Series(sentiment_counts_topic.Count.values, index=sentiment_counts_topic.Sentiment).to_dict()
                        pos_count = counts_dict.get('Positive', 0)
                        neg_count = counts_dict.get('Negative', 0)
                        total = pos_count + neg_count

                        if total > 0:
                            st.markdown(f"##### Interpretation for '{topic_to_visualize}'")
                            
                            st.write(f"🟢 **Positive:** {pos_count} reviews ({(pos_count/total)*100:.1f}%)")
                            st.write(f"🔴 **Negative:** {neg_count} reviews ({(neg_count/total)*100:.1f}%)")
                            st.markdown("---")

                            if neg_count > pos_count:
                                st.warning(f"Feedback on the **'{topic_to_visualize}'** topic is predominantly **negative**. This indicates a potential area for improvement.")
                            elif pos_count > neg_count:
                                st.success(f"Feedback on the **'{topic_to_visualize}'** topic is predominantly **positive**. This suggests a key strength to maintain or highlight.")
                            else:
                                st.info(f"Feedback on the **'{topic_to_visualize}'** topic is **mixed**. This may point to an inconsistent experience worth investigating.")
        
        st.markdown("---")
        st.markdown("### 📄 Analysis Results Table")
        st.markdown("""
            - **Original Text**: The raw review text from your uploaded file.
            - **Main Topic**: The specific topic assigned to the review (e.g., 'Service Quality', 'Attitude').
            - **Broad Topic**: The general category the main topic belongs to (e.g., 'Staff Interaction', 'Operations').
            - **Sentiment**: The predicted sentiment ('Positive' or 'Negative') for the review.
        """)
        st.info(
            "**How are these topics defined?**\n\n"
            "Topic names are primarily identified using top keywords extracted by the BERTopic model, "
            "strengthened with manual human judgement on topic clusters.\n"
            "For example: *'staff'*, *'rude'* → *'Attitude'*.\n\n"
            "💡 **Want to know what a review is really about?** Select its checkbox in the table below to reveal the key topics and themes detected."
        )
        
        filter1, filter2, filter3 = st.columns(3)

        with filter1:
            selected_main_topics = st.multiselect("Filter by Main Topic", results_df['topic_name'].unique())
        
        with filter2:
            selected_broad_topics = st.multiselect("Filter by Broad Topic", results_df['general_topic'].unique())

        with filter3:
            if 'Sentiment' in results_df.columns:
                selected_sentiments = st.multiselect("Filter by Sentiment", results_df['Sentiment'].unique())
            else:
                selected_sentiments = []

        filtered_df = results_df
        if selected_main_topics:
            filtered_df = filtered_df[filtered_df['topic_name'].isin(selected_main_topics)]
        if selected_broad_topics:
            filtered_df = filtered_df[filtered_df['general_topic'].isin(selected_broad_topics)]
        if selected_sentiments:
            filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]

        num_results = len(filtered_df)
        if selected_main_topics or selected_broad_topics or selected_sentiments:
            if num_results > 0:
                st.success(f"✅ Found {num_results} matching reviews.")
            else:
                st.warning("⚠️ No results match the selected filters. Please try a different combination.")
        
        columns_to_show = [review_col, 'topic_id', 'topic_name', 'general_topic', 'Sentiment']
        if 'Sentiment' not in filtered_df.columns:
            columns_to_show.remove('Sentiment')
        
        df_display = filtered_df[columns_to_show].rename(columns={
            review_col: "Original Text",
            'topic_id': "Topic ID",
            'topic_name': "Main Topic",
            'general_topic': "Broad Topic"
        })
        
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_column("Original Text", wrapText=True, autoHeight=True, width=500)
        gb.configure_column("Topic ID", width=90)
        gb.configure_column("Main Topic", width=150)
        gb.configure_column("Broad Topic", width=150)
        gb.configure_column("Sentiment", width=100)
        gb.configure_selection('single', use_checkbox=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            df_display,
            gridOptions=grid_options,
            theme='streamlit',
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            update_mode="SELECTION_CHANGED"
        )
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Full Analysis as CSV",
            data=csv_data,
            file_name='topic_analysis_results.csv',
            mime='text/csv'
        )
#==============================KEYWORD DISPLAY BASED ON ID================================
        selected_rows = grid_response['selected_rows']

        if selected_rows is not None and not selected_rows.empty:
            selected_row_data = selected_rows.iloc[0]
            topic_id = selected_row_data['Topic ID']
            topic_name = selected_row_data['Main Topic']
            review_text = selected_row_data['Original Text']

            with st.expander("📖 View Full Review & Topic Keywords", expanded=True):
                st.markdown(f"📝 **Review:**\n> {review_text}")
                st.markdown("---")
                
                if topic_id == -1:
                    st.info("This review is an **Outlier** and has no specific topic keywords.")
                else:
                    topic_info = df_topics.loc[topic_id]
                    
                    # KEYWORD BAR CHART (Styled)
                    try:
                        keywords = [kw.strip() for kw in topic_info['top10keywords'].split(',')]
                        scores = [float(s.strip()) for s in topic_info['ctfidf_score'].split(',')]
                        df_chart = pd.DataFrame({'Keyword': keywords, 'Score': scores}).sort_values(by='Score', ascending=True)

                        fig_kw = px.bar(
                            df_chart, 
                            x='Score', 
                            y='Keyword', 
                            orientation='h',
                            color='Score',
                            color_continuous_scale='Sunset'
                        )
                        
                        fig_kw.update_layout(
                            title={
                                'text': f"Keyword Importance for '{topic_name}' (ID: {topic_id})",
                                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
                            },
                            yaxis_title="Keyword", 
                            xaxis_title="c-TF-IDF Score",
                            margin=dict(t=80, b=40)
                        )

                        st.plotly_chart(fig_kw, use_container_width=True)
                    except Exception as e:
                        st.warning("Could not generate the keyword importance chart for this topic.")

#============================================LINK TO TOPIC ANALYSIS======================================
        with st.container(border=True):
            st.info("Explore what all the topics mean in detail.", icon="📚")
            if st.button("See Topic Information"):
                st.switch_page("pages/3_Topic_Information.py")