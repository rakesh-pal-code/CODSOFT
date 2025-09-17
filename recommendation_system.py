"""
Streamlit Recommendation System App
Single-file Streamlit app that includes an HTML/CSS "lead" (header) and
implements two simple recommenders:
  1) Content-based (TF-IDF on genres + description)
  2) Collaborative filtering (cosine similarity on user-rating matrix)

How to run:
  1. Install requirements: pip install streamlit scikit-learn pandas numpy
  2. Run: streamlit run streamlit_recommender_app.py

This file is intentionally self-contained and uses a small in-memory sample dataset.
Replace `movies_df` and `ratings_df` with your real data as needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample data
# -----------------------------
# Small movies dataset (id, title, genres, description)
movies_data = [
    (1, "The Space Between", "Sci-Fi|Adventure", "A small crew journeys through space to find a new home."),
    (2, "Quiet Harbor", "Drama|Romance", "Two strangers find solace and love in a coastal town."),
    (3, "Cyber Run", "Action|Sci-Fi", "A courier in a cyberpunk city must deliver a package while hunted."),
    (4, "Baker Street Blues", "Crime|Mystery", "An amateur detective uncovers secrets behind an old bakery."),
    (5, "Forest Echoes", "Documentary|Nature", "A calming look at the lives of forests across the seasons."),
    (6, "Midnight Sonata", "Drama|Music", "A musician battles self-doubt on the path to her solo performance."),
    (7, "Galactic Ways", "Sci-Fi|Adventure", "A light-hearted space travel romp with unexpected companions."),
    (8, "The Last Note", "Drama|Romance|Music", "Two rival musicians come together for one final composition."),
]

movies_df = pd.DataFrame(movies_data, columns=["movieId", "title", "genres", "description"]) 

# Small user-item ratings (userId, movieId, rating)
ratings_data = [
    (1, 1, 5),
    (1, 3, 4),
    (1, 7, 5),
    (2, 2, 5),
    (2, 6, 4),
    (2, 8, 5),
    (3, 4, 5),
    (3, 1, 2),
    (3, 5, 4),
    (4, 3, 5),
    (4, 7, 4),
    (5, 5, 5),
    (5, 2, 3),
]
ratings_df = pd.DataFrame(ratings_data, columns=["userId", "movieId", "rating"])

# -----------------------------
# Utility: build user-item matrix
# -----------------------------
@st.cache_data
def build_user_item_matrix(ratings, movies):
    users = sorted(ratings['userId'].unique())
    movie_ids = movies['movieId'].tolist()
    matrix = pd.DataFrame(0, index=users, columns=movie_ids, dtype=float)
    for _, row in ratings.iterrows():
        matrix.at[row['userId'], row['movieId']] = row['rating']
    return matrix

user_item_matrix = build_user_item_matrix(ratings_df, movies_df)

# -----------------------------
# Content-based recommender
# -----------------------------
@st.cache_data
def train_content_model(movies):
    # Combine genres + description into a single text
    corpus = (movies['genres'].fillna('') + ' ' + movies['description'].fillna(''))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(corpus)
    return tfidf, tfidf_matrix

content_vectorizer, content_tfidf = train_content_model(movies_df)

@st.cache_data
def content_recommend(movie_id, movies, tfidf_matrix, topn=5):
    # Find index of movie_id
    try:
        idx = movies.index[movies['movieId'] == movie_id][0]
    except Exception:
        return []
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    # ignore itself
    sim_scores[idx] = -1
    top_idx = np.argsort(sim_scores)[::-1][:topn]
    return movies.iloc[top_idx][['movieId', 'title', 'genres']]

# -----------------------------
# Collaborative recommender (user-based using cosine similarity)
# -----------------------------
@st.cache_data
def user_similarity_matrix(user_item):
    sim = cosine_similarity(user_item)
    sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)
    return sim_df

user_sim = user_similarity_matrix(user_item_matrix)

@st.cache_data
def collaborative_recommend(user_id, user_item, user_sim_df, movies, topn=5):
    if user_id not in user_item.index:
        return []
    # weighted sum of other users' ratings
    sims = user_sim_df.loc[user_id]
    # avoid self
    sims = sims.drop(user_id)
    # multiply ratings by similarity
    weighted = user_item.T.dot(sims)  # columns are users, index movieId
    # normalize by sum of similarities for unrated handling
    sim_sums = (user_item != 0).T.dot(sims)
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = weighted / sim_sums
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)
    # remove already-rated movies
    rated = user_item.loc[user_id]
    scores[rated[rated > 0].index] = 0
    top_movies = scores.sort_values(ascending=False).head(topn)
    result = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title', 'genres']]
    # maintain order from top_movies
    result = result.set_index('movieId').loc[top_movies.index].reset_index()
    return result

# -----------------------------
# Streamlit UI + HTML/CSS lead
# -----------------------------
st.set_page_config(page_title='Simple Recommender', layout='wide')

lead_html = r"""
<div class="lead-wrap">
  <h1>Simple Recommender</h1>
  <p class="subtitle">Try content-based and collaborative recommendations — editable, single-file demo.</p>
</div>
<style>
.lead-wrap{
  background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
  color: white;
  padding: 28px;
  border-radius: 12px;
  margin-bottom: 20px;
}
.lead-wrap h1{ margin:0; font-size:28px; }
.lead-wrap .subtitle{ margin:4px 0 0 0; opacity:0.95 }

.card{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
.small{ font-size:13px; color:#555 }
</style>
"""

# Render the lead using components so CSS applies neatly
st.components.v1.html(lead_html, height=120)

col1, col2 = st.columns([1, 2])

with col1:
    st.header('Select')
    st.subheader('Your user or a movie')
    user_list = user_item_matrix.index.tolist()
    selected_user = st.selectbox('Pick a user (for collaborative):', options=user_list)
    st.write('---')
    movie_options = movies_df['title'].tolist()
    selected_movie_title = st.selectbox('Pick a movie (for content-based):', options=['(none)'] + movie_options)
    if selected_movie_title != '(none)':
        selected_movie_id = int(movies_df[movies_df['title'] == selected_movie_title]['movieId'].iloc[0])
    else:
        selected_movie_id = None

    st.write('---')
    st.subheader('Quick actions')
    if st.button('Show sample data'):
        st.write('Movies:')
        st.dataframe(movies_df)
        st.write('Ratings:')
        st.dataframe(ratings_df)

with col2:
    st.header('Recommendations')
    st.subheader('Collaborative (user-based)')
    if selected_user is not None:
        coll = collaborative_recommend(selected_user, user_item_matrix, user_sim, movies_df, topn=5)
        if len(coll) == 0:
            st.info('No collaborative recommendations available for this user (insufficient data).')
        else:
            st.table(coll)

    st.subheader('Content-based (movie similarity)')
    if selected_movie_id is not None:
        cb = content_recommend(selected_movie_id, movies_df, content_tfidf, topn=5)
        if cb.empty:
            st.info('No content-based recommendations (movie not found).')
        else:
            st.table(cb)

# -----------------------------
# Bonus: Let user input preferences (simple hybrid approach)
# -----------------------------
st.write('---')
st.header('Preference-driven suggestions (hybrid)')
with st.expander('Tell me what you like (genres or short words)'):
    pref_text = st.text_input('Enter genres / keywords (e.g. "Sci-Fi space music")')
    n_results = st.slider('How many suggestions?', 1, 10, 5)
    if st.button('Recommend for me based on keywords'):
        if pref_text.strip() == '':
            st.warning('Please type some keywords or genres first.')
        else:
            # Build a query vector and compare
            q = content_vectorizer.transform([pref_text])
            sim = cosine_similarity(q, content_tfidf).flatten()
            top_idx = np.argsort(sim)[::-1][:n_results]
            res = movies_df.iloc[top_idx][['movieId', 'title', 'genres']]
            st.table(res)

st.write('\n---\n')
st.markdown('<div class="small">This is a toy demo. For production: use more data, persist models, support cold-start, and evaluate with metrics (precision@k, MAP).</div>', unsafe_allow_html=True)

# -----------------------------
# End of file
# -----------------------------
def collaborative_recommend(user_id, user_item, user_sim_df, movies, topn=5):
    if user_id not in user_item.index:
        return []

    sims = user_sim_df.loc[user_id].drop(user_id)  # similarity with other users
    sims = sims.reindex(user_item.index, fill_value=0)  # align indexes

    # weighted ratings
    weighted = user_item.T.dot(sims.values)
    sim_sums = (user_item != 0).T.dot(sims.values)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = weighted / sim_sums
    scores = pd.Series(scores, index=user_item.columns)
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)

    # remove movies the user already rated
    rated = user_item.loc[user_id]
    scores[rated[rated > 0].index] = 0

    # top recommendations
    top_movies = scores.sort_values(ascending=False).head(topn)
    result = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title', 'genres']]
    result = result.set_index('movieId').loc[top_movies.index].reset_index()
    return result
