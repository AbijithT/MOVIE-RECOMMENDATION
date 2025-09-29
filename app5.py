import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
from keras.models import load_model

# Set Streamlit page configuration
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# CSS for Background Image & Centered Text with Larger Fonts
page_bg_img = """
<style>
    .stApp {
        background: url("https://miro.medium.com/v2/resize:fit:1400/1*qR08Jxq0IHdvFtBsUhCe3Q.jpeg") no-repeat center center fixed;
        background-size: cover;
    }
    h1 {
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        color: white;
        text-shadow: 3px 3px 5px black;
    }
    h2 {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
    p, label, div[data-testid="stSelectbox"] {
        text-align: center;
        font-size: 20px;
        color: white;
        font-weight: bold;
    }
    .block-container {
        max-width: 900px;
        margin: auto;
        background: rgba(0, 0, 0, 0.7);
        padding: 30px;
        border-radius: 15px;
    }
</style>
"""

# Apply background image and styling
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load dataset and similarity matrix
try:
    movies = pd.read_csv('top10K-TMDB-movies.csv')
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Required files are missing. Please upload them.")
    movies = pd.DataFrame()
    similarity = None

# Load trained DNN model
try:
    model = load_model('model.h5')
except FileNotFoundError:
    st.error("Error: model.h5 file not found.")
    model = None

# Prepare movie titles for selection
movies_list = movies['title'].values if 'title' in movies else []

# UI Header
st.markdown("<h1>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<h2>🔎 Discover movies based on your preferences</h2>", unsafe_allow_html=True)

# Movie selection dropdown (centered)
selectvalues = st.selectbox("Select a movie:", movies_list, key="movie_selector")

# API Key for fetching movie posters
API_KEY = 'c7ec19ffdd3279641fb606d19ceb9bb1'

def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return f"https://image.tmdb.org/t/p/w500/{data.get('poster_path', '')}" if data.get('poster_path') else "https://via.placeholder.com/500x750?text=No+Image"
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie):
    """Recommends movies based on similarity"""
    if similarity is None or movies.empty:
        return [], []
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])
        recommended_movies = [movies.iloc[i[0]].title for i in distances[1:6]]
        recommended_posters = [fetch_poster(movies.iloc[i[0]].id) for i in distances[1:6]]
        return recommended_movies, recommended_posters
    except IndexError:
        st.error("Movie not found in the dataset.")
        return [], []

# Display recommendations when button is clicked
if st.button('🔍 Show Recommendations'):
    movie_names, movie_posters = recommend(selectvalues)
    if movie_names and movie_posters:
        st.markdown("<h2>Recommended Movies</h2>", unsafe_allow_html=True)
        cols = st.columns(len(movie_names))
        for col, name, poster in zip(cols, movie_names, movie_posters):
            with col:
                st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>{name}</p>", unsafe_allow_html=True)
                st.image(poster)
