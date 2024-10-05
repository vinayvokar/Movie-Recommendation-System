from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.sentiment import SentimentIntensityAnalyzer

# Load movie data (ensure movies.csv, critic.csv, and rating.csv are in the correct location)
movies = pd.read_csv('data/movies.csv')
critics = pd.read_csv('data/critic_reviews.csv')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(review):
    score = sia.polarity_scores(review)
    return score['compound']  # Returns a value between -1 and 1

# Add sentiment score to critic reviews
critics['sentiment_score'] = critics['review_content'].apply(get_sentiment_score)

# TF-IDF Vectorizer for genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# Cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim):
    idx = movies.index[movies['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    
    # Get average sentiment scores for the recommended movies
    recommended_movies = movies['title'].iloc[movie_indices]
    recommended_sentiment_scores = []
    for movie in recommended_movies:
        # Calculate the average sentiment score from critics for the recommended movies
        avg_sentiment = critics[critics['movie_id'] == movies[movies['title'] == movie].iloc[0]['movie_id']]['sentiment_score'].mean()
        recommended_sentiment_scores.append(avg_sentiment)
    
    return list(zip(recommended_movies, recommended_sentiment_scores))

# Home page view
def index(request):
    return render(request, 'index.html')

# Recommendation view
def recommend(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        recommendations = recommend_movies(title)
        return render(request, 'index.html', {'title': title, 'recommendations': recommendations})
    return render(request, 'index.html')
