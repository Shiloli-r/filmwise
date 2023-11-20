# main.py

from src.data import data_analysis, data_preprocessing as preprocess
from src.algorithms import hybrid_recommendation as algorithm


def start_cli(collaborative_model, content_based_matrix, data):
    print("\n------------------------ Filmwise Movie Recommender CLI ------------------------")

    while True:
        print("\nEnter a user Name to get movie recommendations (type 'exit' to quit):")
        user_input = input("User Name: ")

        if user_input.lower() == 'exit':
            print("Exiting Filmwise Movie Recommender CLI. Goodbye!")
            break

        # Get hybrid recommendations for the specified user
        recommendations = algorithm.hybrid_recommendation(user_input, collaborative_model, content_based_matrix, data, n=5)

        # Display recommendations
        print(f"\nTop 5 Movie Recommendations for User {user_input}:\n")
        for i, (movie, rating) in enumerate(recommendations, start=1):
            print(f"{i}. {movie} - Predicted Rating: {rating:.2f}")


def main():
    # Load and clean the data
    cleaned_data = preprocess.clean_data("../data/raw/raw_data.csv")
    print("\n------------- cleaned data ------------------------")
    print(cleaned_data)
    print("\n------------- expanded data ------------------------")
    expanded_data = preprocess.expand_data(cleaned_data, "../data/catalog/movies.csv", "../data/cleaned/expanded.csv")
    print(expanded_data)

    # Perform data analysis to extract insights
    user_preferences = data_analysis.extract_user_preferences(expanded_data)
    print("\n------------------- User Preferences -------------------------------")
    print(user_preferences)
    print("\n--------------------- Watch History --------------------------------")
    watch_history = data_analysis.get_watch_history(expanded_data)
    print(watch_history)

    # Train collaborative filtering model
    collaborative_model = algorithm.train_collaborative_filtering_model(expanded_data)

    # Train content-based filtering model
    content_based_matrix = algorithm.train_content_based_filtering_model(expanded_data)

    # Get hybrid recommendations for the specified user
    start_cli(collaborative_model, content_based_matrix, expanded_data)


if __name__ == "__main__":
    main()
