import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import requests
from io import StringIO

# Set page config
st.set_page_config(page_title="Movie Ratings Dashboard", layout="wide")

# Load data
@st.cache_data
def load_movie_df():
    url = 'https://raw.githubusercontent.com/Agnieszka-Kamieniksba23169/Dashboard_CA2/main/movie_df.csv'
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        df['primary_genre'] = df['genres'].apply(lambda x: x.split('|')[0])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        return df
    else:
        return None

# Load the dataset
df = load_movie_df()
if df is None:
    st.error("❌ Failed to load movie data.")
    st.stop()

# App Title and Description
st.title("🎬 Movie Ratings Dashboard for Young Adults (18–35)")
st.markdown("**Explore rating trends, genre preferences, and user engagement — built for machine learning and recommender systems**")

# Sidebar filters
st.sidebar.header("🔍 Filters")
genre_options = df['primary_genre'].unique().tolist()
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + genre_options)
year_range = st.sidebar.slider("Select Year Range", int(df['year'].min()), int(df['year'].max()), (2000, 2020))
rating_threshold = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0)

# Apply filters
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['primary_genre'] == selected_genre]
filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
filtered_df = filtered_df[filtered_df['rating'] >= rating_threshold]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Analytics", "📈 Time Trends", "🌟 Engagement", "🧪 Advanced Insights"])

# ---------- Tab 1: Visual Analytics ----------
with tab1:
    st.subheader("Top 20 Most Rated Movies")
    top_movies = filtered_df['title'].value_counts().nlargest(20).reset_index()
    top_movies.columns = ['title', 'count']
    fig1 = px.bar(top_movies, x='count', y='title', orientation='h',
                  color='count', color_continuous_scale='Viridis',
                  labels={'count': 'Number of Ratings', 'title': 'Movie Title'})
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("⭐ Ratings Distribution")
    fig2 = px.histogram(filtered_df, x='rating', nbins=20,
                        color_discrete_sequence=['#636EFA'])
    fig2.update_layout(bargap=0.2, xaxis_title='Rating', yaxis_title='Count')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Average Rating by Primary Genre")
    genre_ratings = filtered_df.groupby('primary_genre')['rating'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(genre_ratings, x='primary_genre', y='rating',
                  color='rating', color_continuous_scale='Turbo')
    st.plotly_chart(fig3, use_container_width=True)

# ---------- Tab 2: Time Trends ----------
with tab2:
    st.subheader("📅 Ratings Over Time (Monthly)")
    monthly_counts = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).size().reset_index(name='rating_count')
    monthly_counts['date'] = monthly_counts['date'].dt.to_timestamp()
    fig4 = px.line(monthly_counts, x='date', y='rating_count', markers=True)
    fig4.update_layout(xaxis_title='Date', yaxis_title='Ratings Count')
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("⏰ Ratings by Hour of Day")
    if 'hour' in filtered_df.columns:
        hourly_counts = filtered_df.groupby('hour')['rating'].count().reset_index(name='count')
        fig5 = px.bar(hourly_counts, x='hour', y='count', color='count',
                      color_continuous_scale='Cividis')
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Hour data not available in the dataset.")

# ---------- Tab 3: Engagement ----------
with tab3:
    st.subheader("☁️ Genre Word Cloud")
    genre_list = [genre for sublist in filtered_df['genres'].dropna().str.split('|').tolist() for genre in sublist]
    genre_freq = Counter(genre_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genre_freq)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.subheader("👥 User Engagement Distribution")
    user_activity = filtered_df['userId'].value_counts().reset_index()
    user_activity.columns = ['userId', 'rating_count']
    fig6 = px.histogram(user_activity, x='rating_count', nbins=50,
                        color_discrete_sequence=['#00CC96'])
    fig6.update_layout(xaxis_title='Ratings per User', yaxis_title='Number of Users')
    st.plotly_chart(fig6, use_container_width=True)

# ---------- Tab 4: Advanced Insights ----------
with tab4:
    st.subheader("📉 Correlation Matrix of Numerical Features")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).drop(columns=['userId'], errors='ignore')
    if not numeric_cols.empty:
        corr = numeric_cols.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("No numeric columns available for correlation analysis.")

    st.subheader("🎯 Average Rating vs. Number of Ratings (per Movie)")
    rating_stats = filtered_df.groupby('title').agg({'rating': ['mean', 'count']}).reset_index()
    rating_stats.columns = ['title', 'avg_rating', 'num_ratings']
    fig_scatter = px.scatter(rating_stats, x='num_ratings', y='avg_rating',
                             size='num_ratings', color='avg_rating',
                             labels={'num_ratings': 'Number of Ratings', 'avg_rating': 'Average Rating'},
                             color_continuous_scale='Plasma')
    st.plotly_chart(fig_scatter, use_container_width=True)

# Footer
st.markdown("---")
st.caption("🚀 Built for younger adult data explorers (18–35). Ideal for ML and recommendation engines.")
