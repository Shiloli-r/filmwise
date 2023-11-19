import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_user_preferences(cleaned_data):
    # Explore user preferences and extract insights

    # Calculate average rating per user
    avg_rating_per_user = cleaned_data.groupby('User')['Rating'].mean()

    # Calculate the distribution of ratings
    rating_distribution = cleaned_data['Rating'].value_counts()

    # Explore favorite genres and most rated movies
    genres = cleaned_data['genre'].value_counts().head(5)
    favorite_genres = cleaned_data.groupby('genre')['Rating'].mean().sort_values(ascending=False)

    # Visualize insights
    visualize_insights(avg_rating_per_user, rating_distribution, favorite_genres)

    # Return extracted insights for potential use in the recommendation algorithm
    return {
        'avg_rating_per_user': avg_rating_per_user,
        'rating_distribution': rating_distribution,
        'genres': genres,
        'favorite_genres': favorite_genres
    }


def visualize_insights(avg_rating_per_user, rating_distribution, favorite_genres):
    # Visualization of insights (can be expanded based on your specific analysis)
    plt.figure(figsize=(12, 6))

    # Plot average rating per user
    plt.subplot(2, 2, 1)
    avg_rating_per_user.plot(kind='bar', color='skyblue')
    plt.title('Average Rating per User')
    plt.xlabel('User')
    plt.ylabel('Average Rating')

    # Plot rating distribution
    plt.subplot(2, 2, 2)
    rating_distribution.sort_index().plot(kind='bar', color='salmon')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Plot favorite genres
    plt.subplot(2, 2, 3)
    favorite_genres.plot(kind='bar', color='lightgreen')
    plt.title('Favorite Genres (Average Rating)')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')

    plt.tight_layout()
    plt.show()

