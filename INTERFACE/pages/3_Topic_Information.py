import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Topic Information Explorer", layout="wide")

# ========================================LOAD AND PREPRO DATA======================================
@st.cache_data
def load_data():
    df = pd.read_csv("INSPECTED_TOPIC.csv")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    return df

df = load_data()

num_topic_ids = df["topic_id"].nunique()
num_main_topics = df["name"].nunique()
num_general_topics = df["general_topic"].nunique()

st.title("🧠 Topic Information Explorer")
st.markdown(
    f"""
    <div style="background-color:#91C8E4; color:black; padding:16px; border-radius:10px;">
        <h4>🧠 Topic Modeling Overview</h4>
        <p>
        <strong>BERTopic</strong> was used to uncover themes in healthcare reviews using transformer-based embeddings and clustering.<br><br>
        • The model initially generated <strong>{df['topic_id'].nunique()} unique topic clusters</strong> (also known as <em>Topic Cluster IDs</em>) using unsupervised machine learning.<br>
        • These were manually inspected and assigned <strong>{df['name'].nunique()} unique subtopics</strong> like <em>staff attitude</em>, <em>price transparency</em>, or <em>vaccine process</em>.<br>
        • The subtopics were then grouped into <strong>{df['general_topic'].nunique()} general topics</strong> like <em>Service Quality</em>, <em>Appointments & Scheduling</em>, and <em>Billing & Payment</em>.<br><br>
        Use this explorer to:
        <ul>
            <li>Understand how reviews are clustered and labeled</li>
            <li>Explore which subtopics belong to which broader themes</li>
            <li>Analyse which areas of healthcare receive the most feedback</li>
        </ul>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

#=====================================SUMMARY METRICS===============================
unique_general = df["general_topic"].nunique()
unique_main = df["name"].nunique()
total_reviews = df["count"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("📂 Unique General Topics", unique_general)
col2.metric("🧩 Unique Main Topics", unique_main)
col3.metric("📝 Total Review Count", total_reviews)

#=========================================BAR CHART TOTAL REVIEW PER GENERAL TOPIC======================
st.subheader("📊 Total Review Count per General Topic")
general_topic_summary = df.groupby("general_topic")["count"].sum().reset_index().sort_values(by="count", ascending=False)

fig1 = px.bar(
    general_topic_summary,
    x="general_topic",
    y="count",
    title="Review Volume per General Topic",
    labels={"general_topic": "General Topic", "count": "Total Review Count"},
    color="general_topic",
)
st.plotly_chart(fig1, use_container_width=True)

# ========================================GROUP SUBTOPICS UNDER GENERAL TOPIC==========================
st.subheader("🔍 Explore Main Topics Within General Topics")

# Group by general_topic and main topic, and aggregate count
grouped = df.groupby(["general_topic", "name"])["count"].sum().reset_index()
selected_general = st.selectbox("Select a General Topic to Explore", sorted(df["general_topic"].dropna().unique()))

filtered_group = grouped[grouped["general_topic"] == selected_general].sort_values(by="count", ascending=False)

st.markdown(f"**Main topics under**: `{selected_general}`")
fig2 = px.bar(
    filtered_group,
    x="count",
    y="name",
    orientation="h",
    labels={"name": "Main Topic", "count": "Review Count"},
    title=f"Main Topics in '{selected_general}'",
    color="name"
)
st.plotly_chart(fig2, use_container_width=True)


with st.expander("📋 Show Data Table"):
    st.dataframe(filtered_group, use_container_width=True)

st.subheader("🗝️ Top Keywords from Topic Cluster IDs Under This Subtopic")

# Filter full df for selected general topic
subset = df[df["general_topic"] == selected_general]

# Get unique topic names under this general topic
for topic_name in filtered_group["name"].unique():
    sub_df = subset[subset["name"] == topic_name]
    
    with st.expander(f"🔹 {topic_name} ({len(sub_df)} topic cluster ID(s))"):
        for _, row in sub_df.iterrows():
            st.markdown(f"**Topic Cluster ID {row['topic_id']}** — `{row['top10keywords']}`")
