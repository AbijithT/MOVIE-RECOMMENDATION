import streamlit as st
import pickle
import requests

# Load your movie data and similarity matrix
movies = pickle.load(open('movies_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies_list = movies['title'].values
st.header('Movie Recommender System')
selectvalues = st.selectbox('Select movie from dropdown', movies_list)

API_KEY = 'c7ec19ffdd3279641fb606d19ceb9bb1'

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        data = response.json()
        poster_path = data['poster_path']
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return full_path
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")
        return None

def search_movie_id(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        results = data.get('results')
        if results:
            return results[0]['id']
        else:
            st.warning("Movie not found!")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching for movie ID: {e}")
        return None

def recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movies = []
        recommended_posters = []

        for i in distances[1:6]:
            movie_id = movies.iloc[i[0]].id
            recommended_movies.append(movies.iloc[i[0]].title)
            poster_url = fetch_poster(movie_id)
            if poster_url:
                recommended_posters.append(poster_url)
            else:
                recommended_posters.append("https://via.placeholder.com/500x750?text=No+Image")

        return recommended_movies, recommended_posters
    except IndexError:
        st.error("Movie not found in the dataset.")
        return [], []

if st.button('Show Recommendations'):
    movie_names, movie_posters = recommend(selectvalues)
    if movie_names and movie_posters:
        cols = st.columns(5)
        for col, name, poster in zip(cols, movie_names, movie_posters):
            with col:
                st.text(name)
                st.image(poster)
