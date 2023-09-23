import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


# setting the title of the app
st.title('Recommender System - Collaborative Filtering')

# description
st.write('This movie recommender system is based on collaborative filtering. It uses the movielens dataset and a trained model to generate top 10 movie recommendations for a user.')

# loading the movielens dataset
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# viewing the dataset
col1, col2 = st.columns(2)
with col1:
    st.write('### Ratings dataset:')
    st.write(ratings)

with col2:
    st.write('### Movies dataset:')
    st.write(movies)


# loading the trained model
model = load_model('collaborative_filtering_model.tf')


# Generating top 10 movie recommendations for a user
st.write(f'## Get top 10 movie recommendations for a user:')

# input a user
user_input = st.text_input('Enter a user id to get movie recommendations:')

# Generating top 10 movie recommendations for the user
if user_input:
    user_id = int(user_input)
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movies = user_ratings.merge(movies, on='movieId')
    user_movies = user_movies.sort_values(by='rating', ascending=False)
    user_movies = user_movies[user_movies['rating'] >= 4.0]
    user_movies = user_movies['title'].tolist()
    user_movies = user_movies[:10]

    # displaying the top 10 movie recommendations
    st.write(f'#### Top 10 movie recommendations for user {user_id}:')
    for i, movie in enumerate(user_movies):
        st.write(f'{i+1}. {movie}')