#do this everytime u open vs code
#open POWERSHELL
# cd "C:\Users\eliza\Documents\Semester V\Sentiment Analysis\CITY\MODEL BUILDING\Interface"
#.\venv311\Scripts\Activate.ps1

import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Sentiment & Topic Modeling",
    page_icon="🧠",
    layout="centered"
)

# TITLE
st.title("🩺 Sentiment & Topic Modeling for Healthcare Customer Reviews")

# INTRO
st.markdown("""
Welcome to the **Healthcare Review Insight Explorer**, a smart tool that analyses real-world reviews from clinics, hospitals, and other healthcare businesses.  
This app combines **deep learning (BiLSTM)** and **topic modeling (BERTopic)** to help you understand what patients are saying and how they feel.

---

### 🗺️ Overview of Available Pages
Use the **buttons below** or the **sidebar** to navigate.
""")

# ---------------------------------------------
st.subheader("1️⃣ Sentiment Classification + LIME Explanation")

st.markdown("""
Classify reviews as **Positive** or **Negative** using a fine-tuned BiLSTM model.  
Visualize what influenced the prediction with a LIME explanation and batch-process entire CSVs.

**Features include:**
- Cleaned review text and prediction score
- Confidence meter with interpretation
- LIME-based word contribution chart
- Pie chart for batch results, CSV export
""")

if st.button("➡️ Go to Sentiment Analysis"):
    st.switch_page("pages/1_Sentiment_Analysis.py")

# ---------------------------------------------
st.subheader("2️⃣ Topic Prediction + Keyword Exploration")

st.markdown("""
Discover common issues and strengths using **BERTopic** on either a single review or a batch of reviews.  
Explore keyword importance and real examples tied to each topic.

**Features include:**
- Keyword importance chart (c-TF-IDF)
- Topic name, general category, and example reviews
- Topic distribution and sentiment breakdown by topic
- Interactive AgGrid table with filters and keyword lookup
""")

if st.button("➡️ Go to Topic Analysis"):
    st.switch_page("pages/2_Topic_Analysis.py")

# ---------------------------------------------
st.subheader("3️⃣ Topic Structure & Cluster Information")

st.markdown("""
Explore the overall structure of topics used in the system.

**Features include:**
- Overview of topic cluster → main topic → general topic hierarchy
- Drill-down views by general topic
- Bar chart of review volume by topic
- Keyword sets for each topic cluster
""")

if st.button("➡️ Go to Topic Information"):
    st.switch_page("pages/3_Topic_Information.py")

# ---------------------------------------------
st.subheader("4️⃣ Nationwide Insights & Business Profiles")

st.markdown("""
Visualise trends across all states and healthcare categories, or drill down into specific businesses.

**Nationwide View:**
- Sentiment distribution and most active states
- Monthly topic trends
- Sentiment by general topic
- Top healthcare review categories

**Business View:**
- Business profile, sample reviews
- Sentiment and topic breakdown
- Sentiment vs rating and monthly trend
- Location map
""")

if st.button("➡️ Go to Nationwide Insights"):
    st.switch_page("pages/4_Nationwide_Insights.py")

# ---------------------------------------------
st.markdown("---")
st.success("Use the sidebar or buttons above to begin exploring.")
