import streamlit as st
import pandas as pd
import plotly.express as px

# Load datasets
ratings = pd.read_csv("processed_ratings.csv")
top_movies = pd.read_csv("top_movies.csv")
user_freq = pd.read_csv("user_frequency.csv")

# Page Config
st.set_page_config(page_title="Youth Movie Rating Dashboard", layout="wide")

st.title("🎮 Movie Ratings Dashboard for Young Adults (18–35)")
st.markdown("Interactive dashboard to explore user behavior and ML potential from movie ratings.")

# Summary Cards
col1, col2, col3 = st.columns(3)
col1.metric("🎟️ Total Ratings", f"{len(ratings):,}")
col2.metric("👤 Unique Users", f"{ratings['userId'].nunique():,}")
col3.metric("🎞️ Movies Rated", f"{ratings['movieId'].nunique():,}")

st.markdown("---")

# Top Movies by Rating Count
st.subheader("🔥 Top 10 Most Rated Movies")
fig_top_movies = px.bar(top_movies, x='movieId', y='count',
                        hover_data=['mean'], labels={'count': 'Number of Ratings'},
                        color='mean', title="Top Rated Movies by Number of Ratings")
st.plotly_chart(fig_top_movies, use_container_width=True)

# Rating Activity Over Time
st.subheader("🕒 Rating Activity Over Time")
activity = ratings.groupby('hour')['rating'].count().reset_index()
fig_activity = px.line(activity, x='hour', y='rating', markers=True,
                       title="Ratings by Hour of Day", labels={'rating': 'Number of Ratings'})
st.plotly_chart(fig_activity, use_container_width=True)

# User Rating Frequency
st.subheader("👤 User Rating Frequency")
fig_users = px.histogram(user_freq, x='num_ratings', nbins=50, title="User Rating Distribution",
                         labels={'num_ratings': 'Ratings per User'})
st.plotly_chart(fig_users, use_container_width=True)

# Heatmap of Ratings (Machine Learning Suitability)
st.subheader("🧠 Ratings Heatmap (ML Suitability)")

# Sample data for heatmap
sample = ratings.sample(1000, random_state=42)
pivot = sample.pivot_table(index='userId', columns='movieId', values='rating')
fig_heatmap = px.imshow(pivot, color_continuous_scale='Viridis',
                        labels=dict(color="Rating"),
                        title="User-Movie Interaction Matrix (Sampled)")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("👩‍💻 Designed for young adults (18–35) with interactive exploration, minimal clutter, and machine learning relevance.")
