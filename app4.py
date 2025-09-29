import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Example of defining a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))  # Adjust input_dim to match your feature size
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Example data (replace with your actual data)
import numpy as np
X_train = np.random.rand(1000, 10)  # 1000 samples, 10 features
y_train = np.random.randint(2, size=1000)  # 1000 binary labels

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model to a file
model.save('model.h5')

# Load dataset and similarity matrix
movies = pd.read_csv('top10K-TMDB-movies.csv')
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load the deep learning model
model = load_model('model.h5')

# Prepare the list of movie titles
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
        return "https://via.placeholder.com/500x750?text=No+Image"

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
            recommended_posters.append(poster_url)

        return recommended_movies, recommended_posters
    except IndexError:
        st.error("Movie not found in the dataset.")
        return [], []

def qpr_model(movie_index):
    # Placeholder for the QPR model computation
    # Replace this with the actual QPR model logic
    qpr_prediction = np.random.random()  # Example placeholder value
    return qpr_prediction

def predict_with_dnn_and_qpr(movie_index):
    # Get QPR prediction
    qpr_prediction = qpr_model(movie_index)
    
    # Combine QPR result with movie index or other features for DNN input
    input_data = np.array([[movie_index, qpr_prediction]])  # Example, adjust based on actual model input requirements
    
    # Get DNN prediction
    dnn_prediction = model.predict(input_data)
    return dnn_prediction

if st.button('Show Recommendations'):
    movie_names, movie_posters = recommend(selectvalues)
    if movie_names and movie_posters:
        cols = st.columns(5)
        for col, name, poster in zip(cols, movie_names, movie_posters):
            with col:
                st.text(name)
                st.image(poster)

# Custom interface for QPR and DNN based model
st.write("## A Recommendation Model Based on Deep Neural Network")
st.write("- Addresses data sparsity and scalability issues in traditional collaborative filtering.")
st.write("- Combines Quadric Polynomial Regression (QPR) Model and Deep Neural Network (DNN) Model")

if st.button('Run QPR and DNN Model'):
    movie_index = movies[movies['title'] == selectvalues].index[0]
    prediction = predict_with_dnn_and_qpr(movie_index)
    st.write(f"Prediction: {prediction}")
