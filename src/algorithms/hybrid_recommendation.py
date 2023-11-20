from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def train_collaborative_filtering_model(df):
    """
    Trains a collaborative filtering model using Surprise's SVD algorithm.

    Parameters:
    - df: DataFrame, the cleaned movie recommendation data.

    Returns:
    - model: trained collaborative filtering model.
    """

    # Load data into Surprise format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['User', 'Movie', 'Rating']], reader)

    # Split data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize SVD algorithm
    model = SVD()

    # Train the model on the training set
    model.fit(trainset)

    # Make predictions on the test set
    predictions = model.test(testset)

    # Evaluate model performance (optional)
    accuracy.rmse(predictions)

    return model


def train_content_based_filtering_model(df):
    """
    Trains a content-based filtering model using movie genres.

    Parameters:
    - df: DataFrame, the cleaned movie recommendation data.

    Returns:
    - tfidf_matrix: TF-IDF matrix of movie genres.
    """

    # Create a TF-IDF vectorizer for movie genres
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fill missing genre values with an empty string
    df['genre'] = df['genre'].fillna('')

    # Fit and transform the TF-IDF vectorizer on movie genres
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['genre'])

    return tfidf_matrix


def hybrid_recommendation(user_id, collaborative_model, content_based_matrix, df, n=5):
    """
    Generates hybrid movie recommendations for a specific user or provides recommendations for the best-rated movies.

    Parameters:
    - user_id: str, the user for whom recommendations are needed.
    - collaborative_model: trained collaborative filtering model.
    - content_based_matrix: TF-IDF matrix of movie genres.
    - df: DataFrame, the cleaned movie recommendation data.
    - n: int, the number of recommendations to return.

    Returns:
    - recommendations: list of tuples (Movie, Combined Score).
    """

    # Check if the user ID exists in the dataset
    if user_id in df['User'].unique():
        # Collaborative Filtering: Get top-rated movies for the user
        user_movies = df[df['User'] == user_id]['Movie'].unique()
        collaborative_predictions = [(movie, collaborative_model.predict(user_id, movie).est) for movie in user_movies]

        # Content-Based Filtering: Calculate cosine similarity between user's rated movies and all movies
        user_rated_movies = df[df['User'] == user_id]['Movie'].unique()
        content_based_scores = []
        for movie in df['Movie'].unique():
            if movie not in user_rated_movies:
                movie_index = df[df['Movie'] == movie].index[0]
                similarity_score = linear_kernel(content_based_matrix[movie_index], content_based_matrix).flatten()[0]
                content_based_scores.append((movie, similarity_score))

        # Combine Collaborative and Content-Based Scores
        combined_scores = []
        for movie, collaborative_score in collaborative_predictions:
            content_based_score = next((score for m, score in content_based_scores if m == movie), 0)
            combined_score = 0.7 * collaborative_score + 0.3 * content_based_score  # Adjust weights as needed
            combined_scores.append((movie, combined_score))

        # Sort recommendations by combined score in descending order
        sorted_recommendations = sorted(combined_scores, key=lambda x: x[1], reverse=True)

        # Return the top n recommendations
        recommendations = sorted_recommendations[:n]

    else:
        # If user ID does not exist, recommend the best-rated movies in the dataset
        best_rated_movies = df.groupby('Movie')['Rating'].mean().sort_values(ascending=False).head(n)
        recommendations = list(zip(best_rated_movies.index, best_rated_movies.values))

    return recommendations
