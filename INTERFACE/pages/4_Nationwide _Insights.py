import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

# ======================================== PAGE CONFIG =========================================
st.set_page_config(page_title="Healthcare Insights", layout="wide")

# ========================================= ALLOWED CATEGORIES =========================================
# filtered categories
ALLOWED_CATEGORIES = {
    'Abortion clinic', 'Acupuncture clinic', 'Acupuncturist', 'Addiction treatment center', 'Adult day care center',
    'Adult foster care service', 'Aerobics instructor', 'Aged care', 'Alcoholism treatment program', 'Allergist',
    'Alternative medicine clinic', 'Alternative medicine practitioner', 'Ambulance service', 'Anesthesiologist',
    'Aromatherapy service', 'Aromatherapy supply store', 'Assisted living facility', 'Audiologist', 'Ayurvedic clinic',
    'Beautician', 'Beauty product supplier', 'Beauty products wholesaler', 'Beauty salon', 'Beauty supply store',
    'Biofeedback therapist', 'Birth center', 'Birth control center', 'Blood bank', 'Blood donation center',
    'Blood testing service', 'Body shaping class', 'Cancer treatment center', 'Cannabis store', 'Cardiologist',
    'Child care agency', 'Child health care centre', 'Child psychiatrist', 'Child psychologist', 'Childbirth class',
    'Children policlinic', "Children's hospital", 'Chinese medicine clinic', 'Chinese medicine store', 'Chiropractor',
    'Community health center', 'Cosmetic dentist', 'Cosmetic surgeon', 'Craniosacral therapy', 'Day care center',
    'Day spa', 'Deaf service', 'Dental clinic', 'Dental hygienist', 'Dental implants periodontist',
    'Dental implants provider', 'Dental laboratory', 'Dental radiology', 'Dental supply store', 'Dentist',
    'Denture care center', 'Dermatologist', 'Diabetes center', 'Diabetes equipment supplier', 'Diabetologist',
    'Diagnostic center', 'Dialysis center', 'Dietitian', 'Disability services & support organisation', 'Doctor',
    'Domestic abuse treatment center', 'Drug store', 'Drug testing service', 'Eating disorder treatment center',
    'Endodontist', 'Endoscopist', 'Eye care center', 'Facial spa', 'Fertility clinic', 'Fitness center', 'Foot care',
    'Foot massage parlor', 'Free clinic', 'Foster care service', 'Gastroenterologist', 'Gastrointestinal surgeon',
    'General hospital', 'General practitioner', 'Geriatrician', 'Gym', 'Gynecologist', 'HIV testing center',
    'Hair care', 'Hair transplantation clinic', 'Health', 'Health and beauty', 'Health and beauty shop',
    'Health consultant', 'Health counselor', 'Health food restaurant', 'Health food store', 'Health insurance agency',
    'Health resort', 'Health spa', 'Hearing aid repair service', 'Hearing aid store', 'Heart hospital', 'Hematologist',
    'Herbal medicine store', 'Herbalist', 'Holistic medicine practitioner', 'Home health care service', 'Homeopath',
    'Homeopathic pharmacy', 'Hospice', 'Hospital', 'Hospital department', 'Hospital equipment and supplies',
    'Hyperbaric medicine physician', 'Hypnotherapy service', 'Immunologist', 'Infectious disease physician',
    'Internal medicine ward', 'Internist', 'Kinesiologist', 'Lasik surgeon', 'Lymph drainage therapist', 'MRI center',
    'Mammography service', 'Massage', 'Massage therapist', 'Massage spa', 'Maternity hospital', 'Medical center',
    'Medical billing service', 'Medical certificate service', 'Medical clinic', 'Medical diagnostic imaging center',
    'Medical equipment supplier', 'Medical examiner', 'Medical group', 'Medical laboratory', 'Medical office',
    'Medical spa', 'Medical supply store', "Men's health physician", 'Mental health clinic', 'Mental health service',
    'Naturopathic practitioner', 'Neonatal physician', 'Nephrologist', 'Neurologist', 'Neurosurgeon',
    'Nurse practitioner', 'Nursing agency', 'Nursing home', 'Nutritionist', 'Obstetrician-gynecologist',
    'Occupational health service', 'Occupational medical physician', 'Occupational safety and health',
    'Occupational therapist', 'Oncologist', 'Ophthalmologist', 'Ophthalmology clinic', 'Optician', 'Optometrist',
    'Oral and maxillofacial surgeon', 'Oral surgeon', 'Organic drug store', 'Oriental medicine clinic',
    'Oriental medicine store', 'Orthodontist', 'Orthopedic shoe store', 'Orthopedic surgeon', 'Orthoptist',
    'Orthotics & prosthetics service', 'Osteopath', 'Otolaryngologist', 'Otolaryngology clinic', 'Pain control clinic',
    'Pain management physician', 'Paternity testing service', 'Pathologist', 'Pediatric cardiologist',
    'Pediatric clinic', 'Pediatric dentist', 'Pediatric ophthalmologist', 'Pediatric orthopedic surgeon',
    'Pediatric surgeon', 'Pediatrician', 'Perinatal center', 'Periodontist', 'Pharmaceutical company', 'Pharmacy',
    'Physiatrist', 'Physical examination center', 'Physical fitness program', 'Physical therapist',
    'Physical therapy clinic', 'Physician assistant', 'Physician referral service', 'Physiotherapy equipment supplier',
    'Plastic surgeon clinic', 'Pregnancy care center', 'Podiatrist', 'Private hospital', 'Proctologist',
    'Prosthodontist', 'Psychiatric hospital', 'Psychiatrist', 'Psychoanalyst', 'Psychologist',
    'Psychoneurological specialized clinic', 'Psychosomatic medical practitioner', 'Psychotherapist',
    'Public health department', 'Public medical center', 'Pulmonologist', 'Radiologist', 'Reflexologist',
    'Registered general nurse', 'Rehabilitation center', 'Reiki therapist', 'Reproductive health clinic',
    'Rheumatologist', 'STD testing service', 'Sexologist', 'Skin care clinic', 'Sleep clinic', 'Specialized clinic',
    'Specialized hospital', 'Speech pathologist', 'Sports injury clinic', 'Sports massage therapist',
    'Sports medicine clinic', 'Sports medicine physician', 'Sports nutrition store', 'Std clinic', 'Surgeon',
    'Surgical center', 'Surgical oncologist', 'Surgical supply store', 'TB clinic', 'Teeth whitening service',
    'Thai massage', 'Thai massage therapist', 'Therapists', 'University hospital', 'Urgent care center', 'Urologist',
    'Vascular surgeon', 'Veterans hospital', 'Walk-in clinic', 'Weight loss service', 'Wellness center',
    'Wellness program', "Women's health clinic", 'X-ray equipment supplier', 'X-ray lab', 'Yoga instructor',
    'Yoga retreat center', 'Pilates'
}

# ========================================= LOAD DATA =========================================
@st.cache_data
def load_data():
    """Loads and caches the main dataset."""
    try:
        df = pd.read_csv("BIZ_VIZ.csv")
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except FileNotFoundError:
        st.error("Error: 'BIZ_VIZ.csv' not found. Please make sure the data file is in the same directory as the script.")
        return pd.DataFrame()

df = load_data()

# ========================================= SESSION STATE INITIALIZATION =========================================
if 'selected_business' not in st.session_state:
    st.session_state.selected_business = None
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None

# ========================================= SIDEBAR FOR NAVIGATION =========================================
st.sidebar.title("Business Explorer")

if st.session_state.selected_business:
    if st.sidebar.button("← Back to Nationwide View"):
        st.session_state.selected_business = None
        st.session_state.selected_category = None
        st.rerun()

st.sidebar.markdown("---")

if not df.empty:
    # --- MODIFIED SECTION: Filter categories for the sidebar ---
    category_cols = [f"category_{i}" for i in range(1, 22)]
    all_categories_raw = pd.unique(df[category_cols].values.ravel())
    
    # Create a filtered list for display that only contains the allowed categories
    display_categories = sorted([c for c in all_categories_raw if c in ALLOWED_CATEGORIES])

    def on_category_change():
        st.session_state.selected_business = None

    st.sidebar.markdown(
        f"""
        <div style="
            font-size: 0.85em;
            font-style: italic;
            color: grey;
            margin-top: -15px;
            margin-bottom: 10px;
            text-align: left;
        ">
            {len(display_categories)} relevant categories available.
        </div>
        """,
        unsafe_allow_html=True
    )
    selected_category = st.sidebar.selectbox(
        "1. Choose a Category",
        display_categories,  
        index=display_categories.index(st.session_state.selected_category) if st.session_state.selected_category in display_categories else None,
        placeholder="Select a category...",
        on_change=on_category_change
    )
    st.session_state.selected_category = selected_category

    
    if st.session_state.selected_category:
        filtered_df = df[df[category_cols].apply(lambda row: st.session_state.selected_category in row.values, axis=1)]
        business_names = sorted(filtered_df["business_name"].unique())
        st.markdown("---")

        st.sidebar.markdown(
            f"""
            <div style="
                font-size: 0.85em;
                font-style: italic;
                color: grey;
                margin-bottom: 10px;
                text-align: left;
            ">
                Found <strong>{len(business_names)}</strong> businesses in <em>{st.session_state.selected_category}</em>.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        selected_business_from_user = st.sidebar.selectbox(
            "2. Choose a Business to View Profile",
            business_names,
            index=None,
            placeholder="Select a business..."
        )
        
        if selected_business_from_user and st.session_state.selected_business != selected_business_from_user:
            st.session_state.selected_business = selected_business_from_user
            st.rerun()

# ========================================= MAIN PAGE LOGIC =========================================
if st.session_state.selected_business is None:
    
    st.markdown("<h1 style='text-align: center;'>📊 Nationwide Healthcare Review Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # METRICS CARDS
    spacer1, col1, col2, col3, col4, spacer2 = st.columns([1, 2, 2, 2, 2, 1])
    with col1:
        st.markdown(f"""
            <div style="border: 1px solid #ccc; border-radius: 15px; padding: 10px 0; background-color: #f9f9f9; text-align: center; box-shadow: 1px 1px 6px rgba(0,0,0,0.05);">
                <div style="font-size: 16px; font-weight: 600; color: #333;">Total Reviews</div>
                <div style="font-size: 24px; font-weight: bold; color: #222;">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="border: 1px solid #ccc; border-radius: 15px; padding: 10px 0; background-color: #f9f9f9; text-align: center; box-shadow: 1px 1px 6px rgba(0,0,0,0.05);">
                <div style="font-size: 16px; font-weight: 600; color: #333;">Total Businesses</div>
                <div style="font-size: 24px; font-weight: bold; color: #222;">{df['gmap_id'].nunique():,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div style="border: 1px solid #ccc; border-radius: 15px; padding: 10px 0; background-color: #f9f9f9; text-align: center; box-shadow: 1px 1px 6px rgba(0,0,0,0.05);">
                <div style="font-size: 16px; font-weight: 600; color: #333;">Total States</div>
                <div style="font-size: 24px; font-weight: bold; color: #222;">{df['state'].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div style="border: 1px solid #ccc; border-radius: 15px; padding: 10px 0; background-color: #f9f9f9; text-align: center; box-shadow: 1px 1px 6px rgba(0,0,0,0.05);">
                <div style="font-size: 16px; font-weight: 600; color: #333;">Total Reviewers</div>
                <div style="font-size: 24px; font-weight: bold; color: #222;">{df['user_id'].nunique():,}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # SENTIMENT & STATE TRENDS
    st.markdown("<h3 style='text-align: center;'>Nationwide Sentiment & State Trends</h3>", unsafe_allow_html=True)
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center;'>Sentiment Distribution</h4>", unsafe_allow_html=True)
            sentiment_counts = df['sentiment_label'].value_counts().reset_index()
            fig_sentiment = px.pie(
                sentiment_counts, 
                names='sentiment_label', 
                values='count', 
                hole=0.4, 
                color='sentiment_label', 
                color_discrete_map={'POSITIVE': '#8cd47e', 'NEGATIVE': '#e15858'}
            )

            st.plotly_chart(fig_sentiment, use_container_width=True)
    with gcol2:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center;'>Most Reviewed States</h4>", unsafe_allow_html=True)
            state_counts = df['state'].value_counts().nlargest(10).reset_index()
            fig_state = px.bar(state_counts, x='state', y='count', color='count', color_continuous_scale='Blues', labels={'state': 'State', 'count': 'Review Count'})
            st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Nationwide Topic Analysis</h3>", unsafe_allow_html=True)

    # =======================================================CHART 1: TRENDING TOPIC====================================
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Trending Topics Over Time</h4>", unsafe_allow_html=True)
        
        # DATA FOR CHART
        df_topics_time = df[df['name'].notna() & (df['name'] != 'Outlier')].copy()
        df_topics_time['Month'] = df_topics_time['time'].dt.to_period('M').astype(str)
        
        # TOP 10 TOPICS TO POPULATE FILTER
        top_10_topics = df_topics_time['name'].value_counts().nlargest(10).index.tolist()
        
        selected_topics = st.multiselect(
            "Select topics to display on the trend chart:",
            options=top_10_topics,
            default=top_10_topics[:5] #show top 5 as default
        )

        if selected_topics:
            topics_over_time = df_topics_time[df_topics_time['name'].isin(selected_topics)]
            topics_over_time_grouped = topics_over_time.groupby(['Month', 'name']).size().reset_index(name='Review Count')
            
            fig_topics_time = px.area(
                topics_over_time_grouped,
                x='Month',
                y='Review Count',
                color='name',
                title='Monthly Review Volume by Topic',
                labels={'name': 'Topic'}
            )
            st.plotly_chart(fig_topics_time, use_container_width=True)
        else:
            st.info("Select one or more topics to see their trends over time.")

    st.markdown("<br>", unsafe_allow_html=True) 

    # ========================SENTIMENT BREAKDOWN BY GEN TOPIC===================================
    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Sentiment Breakdown by General Topic</h4>", unsafe_allow_html=True)
        
        # Prepare data
        sentiment_by_topic = df[df['general_topic'].notna() & (df['general_topic'] != 'Outlier')].groupby(['general_topic', 'sentiment_label']).size().reset_index(name='Count')
        
        fig_sentiment_topic = px.bar(
            sentiment_by_topic,
            x='general_topic',
            y='Count',
            color='sentiment_label',
            title='Sentiment Distribution per General Topic',
            labels={'general_topic': 'General Topic', 'sentiment_label': 'Sentiment'},
            barmode='group',
            color_discrete_map={'POSITIVE': '#8cd47e', 'NEGATIVE': '#e15858'}
        )
        st.plotly_chart(fig_sentiment_topic, use_container_width=True)


    # REVIEWS OVER TIME
    st.subheader(f"Number of Reviews in {df['state'].nunique()} States Over Time")
    review_time = df.groupby(df['time'].dt.to_period("M")).size().reset_index()
    review_time.columns = ['Month', 'Review Count']
    review_time['Month'] = review_time['Month'].astype(str)
    fig_time = px.line(review_time, x='Month', y='Review Count', markers=True, title="Monthly Review Trend")
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    # MOST REVIEWED BUSINESS CATEGORIES
    st.subheader("Most Reviewed Business Categories")
    top_x = st.slider("Select Top N Business Categories", min_value=5, max_value=50, value=10)
    # filter the main category for the chart to only show relevant ones
    main_categories = df[df['category_1'].isin(ALLOWED_CATEGORIES)]['category_1'].value_counts().nlargest(top_x).reset_index()
    fig_business = px.bar(main_categories, x='count', y='category_1', orientation='h', color='count', color_continuous_scale='Viridis', labels={'category_1': 'Business Category', 'count': 'Review Count'})
    fig_business.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_business, use_container_width=True)

else:
    business_df = df[df["business_name"] == st.session_state.selected_business]
    if not business_df.empty:
        business_info = business_df.iloc[0]

        st.title(f"🏥 {business_info['business_name']}")
        st.markdown(f"**Gmap ID:** {business_info['gmap_id']}")
        if pd.notna(business_info.get("url", "")):
            st.markdown(f"🔗 [View on Google Maps]({business_info['url']})")
        st.markdown("---")

        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            st.markdown("### 📍 Address & Info")
            st.write(f"**Address:** {business_info['address']}")
            st.write(f"**State:** {business_info['state']}")
            st.write(f"**Categories:**")
            for i in range(1, 22):
                cat = business_info.get(f"category_{i}", "")
                if pd.notna(cat) and cat:
                    st.write(f"- {cat}")
        with col2:
            st.markdown("### 💵 Business Data")
            st.write(f"**Price Range:** {business_info['price']}")
            st.write(f"**Avg Rating:** {business_info['avg_rating']}")
            st.write(f"**Review Count:** {business_info['num_of_reviews']}")
        with col3:
            st.markdown("### ⏰ Opening Hours")
            try:
                hours = eval(business_info["hours"])
                for day, time in hours:
                    st.write(f"- {day}: {time}")
            except:
                st.write("No data available")

        st.markdown("---")
        description = business_info.get("description")
        if isinstance(description, str) and description.strip() and description.lower() != 'nan':
            with st.expander("📝 Description", expanded=True):
                st.write(description)

        misc_data = business_info.get("MISC")
        if isinstance(misc_data, str) and misc_data.strip() and misc_data.lower() != 'nan':
            with st.expander("📋 MISC Info", expanded=False):
                try:
                    misc = eval(misc_data)
                    if isinstance(misc, dict) and misc:
                        for section, items in misc.items():
                            st.write(f"**{section}:** {', '.join(items)}")
                    else:
                         st.write("No MISC data available.")
                except:
                    st.write("MISC data could not be displayed.")

        st.markdown("### 🧠 Sample Reviews")
        

        columns_for_sample = [
            "translated_text", "sentiment_label", "rating", "time",
            "name", "general_topic"
        ]
        sample = business_df[columns_for_sample].sort_values("time", ascending=False).head(10)
        
        sample["time"] = sample["time"].dt.strftime("%Y-%m-%d")
        sample = sample.reset_index(drop=True)
        sample.index = sample.index + 1

        st.dataframe(sample.rename(columns={
            "translated_text": "Review",
            "rating": "Rating",
            "time": "Date",
            "sentiment_label": "Sentiment",
            "name": "Topic",
            "general_topic": "General Topic"
        }))
# ==================== TOPIC ANALYSIS FOR THIS BUSINESS ====================
        st.markdown("---")
        st.markdown("### 🔬 Topic & Sentiment Analysis for This Business")

        try:
            # filter out outliers for a cleaner chart
            business_topics_df = business_df[business_df['name'].notna() & (business_df['name'] != 'Outlier')]

            if not business_topics_df.empty:
                topic_counts_biz = business_topics_df['name'].value_counts().reset_index()
                fig_topic_biz = px.bar(
                    topic_counts_biz,
                    x='count',
                    y='name',
                    orientation='h',
                    labels={'name': 'Topic', 'count': 'Number of Reviews'},
                    color='count',
                    color_continuous_scale='Agsunset'
                )

                fig_topic_biz.update_layout(
                    title={
                        'text': f"Most Common Topics for {st.session_state.selected_business}",
                        'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
                        'font': {
                            'size':20
                        }
                    },
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_topic_biz, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # sentiment breakdown by topic for the business
                sentiment_by_topic_biz = business_topics_df.groupby(['name', 'sentiment_label']).size().reset_index(name='Count')

                # calcu dynamic width for the chart based on topic = biar bs scroll
                sentiment_by_topic_biz = business_topics_df.groupby(['name', 'sentiment_label']).size().reset_index(name='Count')

                fig_sentiment_topic_biz = px.bar(
                    sentiment_by_topic_biz,
                    x='name',
                    y='Count',
                    color='sentiment_label',
                    labels={'name': 'Topic', 'sentiment_label': 'Sentiment'},
                    barmode='group',
                    color_discrete_map={'POSITIVE': '#8cd47e', 'NEGATIVE': '#e15858'}
                )
                fig_sentiment_topic_biz.update_layout(
                    title={
                        'text': f"Sentiment for Each Topic at {st.session_state.selected_business}",
                        'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
                    }
                )
                st.plotly_chart(fig_sentiment_topic_biz, use_container_width=True)
            else:
                st.info("No specific topics (excluding outliers) were found in the reviews for this business.")
        
        except Exception as e:
            st.warning(f"Could not generate topic analysis charts for this business. Error: {e}")



        monthly = business_df.groupby(business_df["time"].dt.to_period("M")).size().reset_index()
        monthly.columns = ["Month", "Review Count"]
        monthly["Month"] = monthly["Month"].astype(str)
        fig_time_biz = px.line(monthly, x="Month", y="Review Count", markers=True)

        st.markdown("---")
        fig_time_biz.update_layout(
            title={
                'text': "Review Volume Over Time",
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {
                    'size': 20  
                }
            }
        )
        st.plotly_chart(fig_time_biz, use_container_width=True)

        sentiment_rating = business_df.groupby("rating")["sentiment_label"].value_counts().unstack().fillna(0)
        
        fig_bar = px.bar(
            sentiment_rating, 
            barmode="group",
            color_discrete_map={'POSITIVE': '#8cd47e', 'NEGATIVE': '#e15858'}
        )

        st.markdown("---")
        fig_bar.update_layout(
            title={
                'text': "Sentiment vs Rating",
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {
                    'size':20
                }
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 🗺 Location")
        lat, lon = business_info["latitude"], business_info["longitude"]
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12),
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=300,
            )],
        ))
