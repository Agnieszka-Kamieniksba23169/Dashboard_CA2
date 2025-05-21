
import requests
from io import StringIO
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Movie Ratings Dashboard", layout="wide")

# Title
st.title("🎬 Movie Ratings Dashboard")
st.markdown("**Explore rating trends, genre preferences, and user engagement — powered by data & perfect for Machine Learning**")

# Load data

@st.cache_data
def load_movie_df():
    url = 'https://raw.githubusercontent.com/Agnieszka-Kamieniksba23169/Dashboard_CA2/refs/heads/main/movie_df.csv'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load movies data.")
        return None


# Call the data loader and assign to movie_df
movie_df = load_movie_df()


# Sidebar filters
st.sidebar.header("🔍 Filters")
genre_options = df['primary_genre'].unique().tolist()
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + genre_options)

if selected_genre != "All":
    df = df[df['primary_genre'] == selected_genre]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics", "📈 Time Trends", "🌟 Engagement"])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("Top 20 Most Rated Movies")
    top_movies = df['title'].value_counts().nlargest(20).reset_index()
    top_movies.columns = ['title', 'count']
    fig1 = px.bar(top_movies, x='count', y='title', orientation='h',
                  color='count', color_continuous_scale='Viridis',
                  labels={'count': 'Number of Ratings', 'title': 'Movie Title'})
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Ratings Distribution")
    fig2 = px.histogram(df, x='rating', nbins=9, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Average Rating by Primary Genre")
    genre_ratings = df.groupby('primary_genre')['rating'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(genre_ratings, x='primary_genre', y='rating',
                  color='rating', color_continuous_scale='Turbo')
    st.plotly_chart(fig3, use_container_width=True)

# ---------- Tab 2 ----------
with tab2:
    st.subheader("Ratings Over Time (Monthly)")
    monthly_counts = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='rating_count')
    monthly_counts['date'] = monthly_counts['date'].dt.to_timestamp()
    fig4 = px.line(monthly_counts, x='date', y='rating_count', markers=True)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Ratings by Hour of Day")
    hourly_counts = df.groupby('hour')['rating'].count().reset_index(name='count')
    fig5 = px.bar(hourly_counts, x='hour', y='count', color='count',
                  color_continuous_scale='Cividis')
    st.plotly_chart(fig5, use_container_width=True)

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Genre Word Cloud")
    genre_list = [genre for sublist in df['genres'].str.split('|').tolist() for genre in sublist]
    genre_freq = Counter(genre_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genre_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("User Engagement Distribution")
    user_activity = df['userId'].value_counts().reset_index()
    user_activity.columns = ['userId', 'rating_count']
    fig6 = px.histogram(user_activity, x='rating_count', nbins=50,
                        color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig6, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🚀 Built for younger adult data explorers. Ideal for machine learning and recommender systems in online retail.")
