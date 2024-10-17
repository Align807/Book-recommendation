# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:15:09 2024

@author: align
"""
import streamlit as st
import pickle

# Define the recommendation function
def recommend_books_knn(book_title, model_knn, df, tfidf_matrix, num_recommendations=5):
    idx = df[df['Book-Title'] == book_title].index[0]
    book_vector = tfidf_matrix[idx].reshape(1, -1)
    distances, indices = model_knn.kneighbors(book_vector, n_neighbors=num_recommendations + 1)
    recommended_indices = indices.flatten()[1:]
    return df[['Book-Title', 'Book-Author', 'Image-URL-L']].iloc[recommended_indices]

# Load the pickled environment
with open('recommendation_model.pkl', 'rb') as f:
    recommend_books_knn, df, tfidf_matrix, model_knn = pickle.load(f)

st.title('Book Recommendation System')

# User input
book_title = st.sidebar.text_input('Enter a book title')

#Harry Potter and the Sorcerer's Stone (Book 1)
#The Art of Makeup

# Recommend books

if st.sidebar.button('Recommend'):
    if book_title:
        recommended_books = recommend_books_knn(book_title, model_knn=model_knn, df=df, tfidf_matrix=tfidf_matrix)
        if not recommended_books.empty:
            st.subheader('Recommendations')
            for _, book in recommended_books.iterrows():
                st.image(book['Image-URL-L'], caption=f"{book['Book-Title']} by {book['Book-Author']}", width=150)
                st.markdown("---")
        else:
            st.write('No recommendations found. Please enter a different book title.')
    else:
        st.write('Please enter a book title to get recommendations.')